//! Generate payload-free implementation traces for TLC conformance checking.

use async_trait::async_trait;
use chrono::Utc;
use generalist::provider::Provider;
use generalist::tools::{SearchConversationsTool, SearchMemoriesTool};
use generalist::{
    Agent, CompletionRequest, CompletionResponse, ContentBlock, DeliveryMode, EpisodeOutcome,
    Error, Message, ModelKind, ModelTrace, ModelTraceSnapshot, PromptQueue, Result, SavedState,
    StopReason, Tool, ToolCallOutcome, ToolRegistry, TurnControl, TurnOutcome, WorkspaceScope,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct ScriptedProvider {
    responses: Mutex<VecDeque<Result<CompletionResponse>>>,
}

impl ScriptedProvider {
    fn new(responses: Vec<Result<CompletionResponse>>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }
}

#[async_trait(?Send)]
impl Provider for ScriptedProvider {
    fn id(&self) -> &'static str {
        "model-conformance"
    }

    fn model(&self) -> &str {
        "deterministic"
    }

    async fn complete(&self, _request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        self.responses
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop_front()
            .ok_or_else(|| Error::Other("Conformance provider script was exhausted".to_string()))?
    }
}

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Return a deterministic result"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object", "additionalProperties": false})
    }

    async fn execute(&self, _input: Value) -> Result<String> {
        Ok("echoed".to_string())
    }
}

fn tool_response() -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "echo".to_string(),
            input: json!({}),
            id: "tool-call-1".to_string(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    }
}

fn answer_response() -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::Text {
            text: "done".to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: None,
    }
}

fn refusal_response() -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::Text {
            text: "declined".to_string(),
        }],
        stop_reason: StopReason::Refusal,
        usage: None,
    }
}

#[derive(Debug, Clone, Copy)]
enum AsyncCase {
    ToolAnswer,
    Refusal,
    ProviderFailure,
    CancelBeforeProvider,
    DeniedBatch,
    DeniedPython,
    NestedDenial,
    UndeclaredTool,
    TruncatedBatch,
    CancelTruncatedBatch,
    IterationLimit,
    Steering,
    CancelledBatch,
    QueueEdits,
}

struct DenyNamed(&'static str);

#[async_trait]
impl generalist::ToolPermissionHandler for DenyNamed {
    async fn check_permission(
        &self,
        request: &generalist::ToolExecutionRequest,
    ) -> generalist::PermissionDecision {
        if request.tool_name == self.0 {
            generalist::PermissionDecision::Deny
        } else {
            generalist::PermissionDecision::Allow
        }
    }
}

struct CancelPermission(generalist::CancelHandle);

#[async_trait]
impl generalist::ToolPermissionHandler for CancelPermission {
    async fn check_permission(
        &self,
        _request: &generalist::ToolExecutionRequest,
    ) -> generalist::PermissionDecision {
        self.0.cancel();
        std::future::pending().await
    }
}

fn batch_response(count: usize) -> CompletionResponse {
    let mut response = tool_response();
    response.content = (0..count)
        .map(|index| ContentBlock::ToolUse {
            name: "echo".into(),
            input: json!({}),
            id: format!("call-{index}"),
        })
        .collect();
    response
}

fn python_response() -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "python".into(),
            input: json!({"code": "import tools\nfor _ in range(2):\n    try:\n        tools.echo()\n    except Exception:\n        pass"}),
            id: "python-1".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    }
}

#[derive(Serialize)]
struct Scenario {
    name: String,
    max_rounds: usize,
    #[serde(flatten)]
    trace: ModelTraceSnapshot,
}

async fn async_runtime_trace(case: AsyncCase) -> Result<Scenario> {
    use AsyncCase::*;
    let trace = ModelTrace::for_models(&[ModelKind::AsyncRuntime]);
    let queue = PromptQueue::with_model_trace(trace.clone());
    let (cancel, mut control) = TurnControl::for_turn(queue.clone());
    let mut registry = match case {
        DeniedBatch | NestedDenial => {
            ToolRegistry::with_permission_handler(Box::new(DenyNamed("echo")))
        }
        DeniedPython => ToolRegistry::with_permission_handler(Box::new(DenyNamed("python"))),
        CancelledBatch => {
            ToolRegistry::with_permission_handler(Box::new(CancelPermission(cancel.clone())))
        }
        _ => ToolRegistry::new(),
    };
    registry.register(Arc::new(Echo))?;
    let responses = match case {
        ToolAnswer | UndeclaredTool => vec![Ok(tool_response()), Ok(answer_response())],
        Refusal => vec![Ok(refusal_response())],
        ProviderFailure => vec![Err(Error::Api {
            status: 401,
            message: "deterministic failure".into(),
            retry_after: None,
            error_type: None,
        })],
        CancelBeforeProvider => Vec::new(),
        DeniedBatch | CancelledBatch => vec![Ok(batch_response(2))],
        DeniedPython | NestedDenial => vec![Ok(python_response())],
        TruncatedBatch | CancelTruncatedBatch => {
            let mut response = batch_response(2);
            response.stop_reason = StopReason::MaxTokens;
            vec![Ok(response), Ok(answer_response())]
        }
        IterationLimit => vec![Ok(tool_response()), Ok(tool_response())],
        Steering => vec![Ok(answer_response()), Ok(answer_response())],
        QueueEdits => vec![Ok(answer_response())],
    };
    if matches!(case, QueueEdits) {
        // Duplicate text must not hide a lost deletion or wrong claimed ID.
        let deleted = queue.enqueue("duplicate", DeliveryMode::FollowUp);
        let kept = queue.enqueue("duplicate", DeliveryMode::FollowUp);
        let restored = queue.enqueue("duplicate", DeliveryMode::FollowUp);
        assert!(queue.delete(deleted));
        assert!(queue.move_by(kept, 1));
        assert!(queue.move_by(kept, -1));
        assert_eq!(queue.restore_latest().unwrap().id, restored);
        assert!(queue.reconcile_goal_continuation(true));
        assert!(!queue.reconcile_goal_continuation(true));
        assert!(queue.reconcile_goal_continuation(false));
        drop(queue.claim_follow_up().unwrap()); // rollback, then retry below
    } else {
        queue.enqueue("exercise runtime", DeliveryMode::FollowUp);
    }
    let claim = queue.claim_follow_up().expect("one claimable follow-up");
    let mut agent = Agent::new(
        Box::new(ScriptedProvider::new(responses)),
        registry,
        "conformance",
    );
    agent.code_mode = matches!(case, DeniedPython | NestedDenial | UndeclaredTool);
    agent.max_iterations = 2;
    agent.max_retries = 0;
    agent.set_model_trace(trace.clone());
    agent.begin_queued_turn(&claim.prompts()[0]);
    claim.commit();
    if matches!(case, CancelBeforeProvider) {
        cancel.cancel();
    }
    let mut steered = false;
    let mut checkpoints = 0;
    let actual = agent
        .run_started_turn(
            &mut |event| {
                if matches!(case, Steering)
                    && !steered
                    && matches!(event, generalist::AgentEvent::ApiCallStarted)
                {
                    let id = queue.enqueue("duplicate", DeliveryMode::FollowUp);
                    assert!(queue.toggle_delivery(id, true));
                    queue.enqueue("duplicate", DeliveryMode::Steer);
                    steered = true;
                }
                let truncated = matches!(event, generalist::AgentEvent::Notice(ref text)
                    if text.contains("mid-tool-call"));
                if matches!(case, CancelTruncatedBatch) && truncated {
                    queue.enqueue("retain until next turn", DeliveryMode::Steer);
                    cancel.cancel();
                }
                if let generalist::AgentEvent::HistoryCheckpoint { history, .. } = event {
                    assert!(generalist::history_tool_protocol_is_valid(&history));
                    checkpoints += 1;
                }
            },
            &mut control,
        )
        .await;
    let expected = match case {
        Refusal => TurnOutcome::Refused,
        CancelBeforeProvider | CancelledBatch | CancelTruncatedBatch => TurnOutcome::Interrupted,
        DeniedBatch | DeniedPython | NestedDenial => TurnOutcome::PausedOnDenial,
        IterationLimit => TurnOutcome::MaxIterationsReached,
        _ => TurnOutcome::Completed,
    };
    if matches!(case, ProviderFailure) {
        assert!(matches!(actual, Err(Error::Api { status: 401, .. })));
    } else {
        assert_eq!(actual?, expected, "{case:?}");
        if !matches!(case, CancelBeforeProvider) {
            assert!(checkpoints > 0);
        }
    }
    assert!(generalist::history_tool_protocol_is_valid(agent.history()));
    if matches!(case, CancelTruncatedBatch) {
        assert_eq!(queue.len(), 1);
        queue.normalize_steers();
        assert_eq!(queue.snapshot()[0].delivery, DeliveryMode::FollowUp);
    } else {
        assert!(queue.is_empty());
    }
    if matches!(case, DeniedBatch) {
        assert_eq!(agent.registry.execution_history().len(), 1);
        assert_eq!(agent.history().last().unwrap().content.len(), 2);
    }
    if matches!(case, UndeclaredTool | TruncatedBatch | DeniedPython) {
        assert!(agent.registry.execution_history().is_empty());
    }
    Ok(Scenario {
        name: format!("{case:?}"),
        max_rounds: 2,
        trace: trace.snapshot(),
    })
}

async fn storage_runtime_trace() -> Result<ModelTraceSnapshot> {
    let trace = ModelTrace::for_models(&[ModelKind::MemoryRuntime, ModelKind::ArchiveScopeRuntime]);
    let temp = tempfile::tempdir()
        .map_err(|error| Error::Other(format!("Failed to create trace directory: {error}")))?;
    let project = temp.path().join("project");
    fs::create_dir_all(project.join(".git"))
        .map_err(|error| Error::Other(format!("Failed to create trace project: {error}")))?;
    let scope = WorkspaceScope::discover(&project)?;

    let history = generalist::HistoryStore::open_with_model_trace(
        temp.path().to_path_buf(),
        scope.clone(),
        trace.clone(),
    )?;
    let memory = generalist::EpisodicMemory::open_scoped_with_model_trace(
        temp.path().join("episodes.sqlite3"),
        scope.clone(),
        trace.clone(),
    )?;
    assert!(memory
        .record_settled_turn(
            "capture is paused",
            &[Message::user_text("capture is paused")],
            EpisodeOutcome::Completed,
            "test",
            "model",
            Utc::now(),
        )
        .await?
        .is_none());
    memory.set_capture_enabled(true).await?;

    let mut state = SavedState::new(scope, "test".to_string(), "model".to_string());
    state
        .conversation_history
        .push(Message::user_text("conformance needle"));
    if history.save_if_absent(&state, "saved")?.is_none() {
        return Err(Error::Other(
            "Conformance history did not create a fresh named archive".to_string(),
        ));
    }
    if history.save_if_absent(&state, "saved")?.is_some() {
        return Err(Error::Other(
            "Conformance history unexpectedly replaced a named archive".to_string(),
        ));
    }
    history.save(&state, "discarded")?;
    if !history.forget_current_archive("discarded")? {
        return Err(Error::Other(
            "Conformance history did not forget the selected archive".to_string(),
        ));
    }

    let forgotten = memory
        .record_settled_turn(
            "discarded episode",
            &[Message::user_text("discarded episode")],
            EpisodeOutcome::Completed,
            "test",
            "model",
            Utc::now(),
        )
        .await?
        .ok_or_else(|| {
            Error::Other("Enabled conformance memory did not record an episode".to_string())
        })?;
    let forget_result = memory.forget(&forgotten).await?;
    if !matches!(
        forget_result,
        generalist::ForgetResult::Deleted | generalist::ForgetResult::DeletedCheckpointPending(_)
    ) {
        return Err(Error::Other(
            "Conformance memory did not forget the selected episode".to_string(),
        ));
    }

    let recorded = memory
        .record_settled_turn(
            "conformance needle",
            &[Message::user_text("conformance needle")],
            EpisodeOutcome::Completed,
            "test",
            "model",
            Utc::now(),
        )
        .await?;
    recorded.ok_or_else(|| {
        Error::Other("Enabled conformance memory did not record an episode".to_string())
    })?;

    let mut registry = ToolRegistry::new();
    registry.set_model_trace(trace.clone());
    registry.register(Arc::new(SearchConversationsTool::new(history)))?;
    registry.register(Arc::new(SearchMemoriesTool::new(memory)))?;

    let conversation = registry
        .execute_tool(
            "search_conversations",
            json!({"query": "conformance needle", "scope": "current"}),
            "history-permission-1".to_string(),
        )
        .await;
    if conversation.outcome != ToolCallOutcome::Success {
        return Err(Error::Other(
            "Conformance conversation search failed".to_string(),
        ));
    }

    let memory = registry
        .execute_tool(
            "search_memories",
            json!({"query": "conformance needle", "scope": "current"}),
            "memory-permission-1".to_string(),
        )
        .await;
    if memory.outcome != ToolCallOutcome::Success {
        return Err(Error::Other("Conformance memory search failed".to_string()));
    }

    Ok(trace.snapshot())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| Error::Other("Usage: model_conformance <output.json>".to_string()))?;
    let mut scenarios = Vec::new();
    for case in [
        AsyncCase::ToolAnswer,
        AsyncCase::Refusal,
        AsyncCase::ProviderFailure,
        AsyncCase::CancelBeforeProvider,
        AsyncCase::DeniedBatch,
        AsyncCase::DeniedPython,
        AsyncCase::NestedDenial,
        AsyncCase::UndeclaredTool,
        AsyncCase::TruncatedBatch,
        AsyncCase::CancelTruncatedBatch,
        AsyncCase::IterationLimit,
        AsyncCase::Steering,
        AsyncCase::CancelledBatch,
        AsyncCase::QueueEdits,
    ] {
        scenarios.push(async_runtime_trace(case).await?);
    }
    scenarios.push(Scenario {
        name: "Storage".into(),
        max_rounds: 2,
        trace: storage_runtime_trace().await?,
    });
    let bytes = serde_json::to_vec_pretty(&scenarios)?;
    fs::write(&output, bytes).map_err(|error| {
        Error::Other(format!(
            "Failed to write conformance trace {}: {error}",
            output.display()
        ))
    })?;
    Ok(())
}
