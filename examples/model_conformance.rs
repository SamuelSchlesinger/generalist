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
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct ScriptedProvider {
    responses: Mutex<VecDeque<CompletionResponse>>,
}

impl ScriptedProvider {
    fn new(responses: Vec<CompletionResponse>) -> Self {
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
            .ok_or_else(|| Error::Other("Conformance provider script was exhausted".to_string()))
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

async fn async_runtime_trace() -> Result<ModelTraceSnapshot> {
    let trace = ModelTrace::for_models(&[ModelKind::AsyncRuntime]);
    let queue = PromptQueue::with_model_trace(trace.clone());
    queue.enqueue("exercise the runtime", DeliveryMode::FollowUp);
    let claim = queue
        .claim_follow_up()
        .ok_or_else(|| Error::Other("Conformance follow-up was not claimable".to_string()))?;
    let prompt = claim.prompts()[0].clone();

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Echo))?;
    let provider = ScriptedProvider::new(vec![tool_response(), answer_response()]);
    let mut agent = Agent::new(Box::new(provider), registry, "conformance");
    agent.code_mode = false;
    agent.max_iterations = 2;
    agent.set_model_trace(trace.clone());
    agent.begin_queued_turn(&prompt);
    claim.commit();

    let (_cancel, mut control) = TurnControl::for_turn(queue);
    let outcome = agent.run_started_turn(&mut |_| {}, &mut control).await?;
    if outcome != TurnOutcome::Completed {
        return Err(Error::Other(format!(
            "Conformance turn ended unexpectedly: {outcome:?}"
        )));
    }
    Ok(trace.snapshot())
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
    let async_trace = async_runtime_trace().await?;
    let storage_trace = storage_runtime_trace().await?;
    let snapshot = ModelTraceSnapshot {
        async_runtime: async_trace.async_runtime,
        memory_runtime: storage_trace.memory_runtime,
        archive_scope_runtime: storage_trace.archive_scope_runtime,
    };
    let bytes = serde_json::to_vec_pretty(&snapshot)?;
    fs::write(&output, bytes).map_err(|error| {
        Error::Other(format!(
            "Failed to write conformance trace {}: {error}",
            output.display()
        ))
    })?;
    Ok(())
}
