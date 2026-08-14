use super::*;
use crate::error::{Error, Result};
use crate::goal::UPDATE_GOAL_TOOL_NAME;
use crate::permissions::{
    AlwaysDenyPermissions, PermissionDecision, PolicyPermissions, ToolExecutionRequest,
    ToolPermissionHandler,
};
use crate::provider::Provider;
use crate::runtime::{DeliveryMode, PromptQueue, TurnControl};
use crate::tool::{Tool, ToolCallOutcome, ToolRegistry};
use crate::types::{
    estimate_tokens, CompletionDelta, CompletionLimits, CompletionRequest, CompletionResponse,
    ContentBlock, Message, StopReason,
};
use async_trait::async_trait;
use serde_json::json;
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{oneshot, Notify};
use tokio::task::LocalSet;

/// A provider that plays back scripted responses (or errors).
struct Script {
    steps: Mutex<Vec<Result<CompletionResponse>>>,
    calls: AtomicUsize,
}

impl Script {
    fn new(steps: Vec<Result<CompletionResponse>>) -> Self {
        let mut steps = steps;
        steps.reverse();
        Self {
            steps: Mutex::new(steps),
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait(?Send)]
impl Provider for Script {
    fn id(&self) -> &'static str {
        "script"
    }
    fn model(&self) -> &str {
        "scripted"
    }
    async fn complete(&self, _request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.steps.lock().unwrap().pop().expect("script exhausted")
    }
}

struct Echo;

#[async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> &str {
        "echo"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _input: Value) -> Result<String> {
        Ok("x".repeat(100_000))
    }
}

fn text_response(text: &str) -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        stop_reason: StopReason::EndTurn,
        usage: None,
    }
}

fn tool_response() -> CompletionResponse {
    CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "echo".into(),
            input: json!({}),
            id: "t1".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    }
}

fn agent_with(script: Script) -> Agent {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Echo)).unwrap();
    let mut agent = Agent::new(Box::new(script), registry, "test");
    // Most tests in this helper exercise the legacy/direct execution
    // path specifically. Code-mode tests construct their own agent.
    agent.code_mode = false;
    agent
}

struct GatedFinalProvider {
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
    first_started: Arc<Notify>,
    first_release: Mutex<Option<oneshot::Receiver<()>>>,
}

#[async_trait(?Send)]
impl Provider for GatedFinalProvider {
    fn id(&self) -> &'static str {
        "gated"
    }

    fn model(&self) -> &str {
        "gated"
    }

    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.requests
            .lock()
            .unwrap()
            .push(request.messages.to_vec());
        if call == 0 {
            self.first_started.notify_one();
            let release = self.first_release.lock().unwrap().take();
            if let Some(release) = release {
                let _ = release.await;
            }
            Ok(text_response("first answer"))
        } else {
            Ok(text_response("answer after steer"))
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn steering_queued_during_final_response_gets_another_model_call() {
    LocalSet::new()
        .run_until(async {
            let calls = Arc::new(AtomicUsize::new(0));
            let requests = Arc::new(Mutex::new(Vec::new()));
            let first_started = Arc::new(Notify::new());
            let (release, first_release) = oneshot::channel();
            let provider = GatedFinalProvider {
                calls: Arc::clone(&calls),
                requests: Arc::clone(&requests),
                first_started: Arc::clone(&first_started),
                first_release: Mutex::new(Some(first_release)),
            };
            let queue = PromptQueue::default();
            let queue_for_turn = queue.clone();

            let task = tokio::task::spawn_local(async move {
                let mut agent = Agent::new(Box::new(provider), ToolRegistry::new(), "test");
                agent.begin_turn("initial");
                let (_cancel, mut control) = TurnControl::for_turn(queue_for_turn);
                let mut events = Vec::new();
                let outcome = agent
                    .run_started_turn(&mut |event| events.push(event), &mut control)
                    .await
                    .unwrap();
                (agent, outcome, events)
            });

            first_started.notified().await;
            let steer_id = queue.enqueue("correct that", DeliveryMode::Steer);
            release.send(()).unwrap();
            let (agent, outcome, events) = task.await.unwrap();

            assert_eq!(outcome, TurnOutcome::Completed);
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            assert!(queue.is_empty());
            assert!(events.iter().any(|event| {
                matches!(
                    event,
                    AgentEvent::SteeringCommitted { prompts }
                        if prompts.iter().any(|prompt| prompt.id == steer_id)
                )
            }));
            assert_eq!(agent.history.len(), 4);
            assert_eq!(agent.history[2].role, "user");
            assert!(agent.history[2].text().contains("correct that"));
            let captured = requests.lock().unwrap();
            assert_eq!(captured.len(), 2);
            assert!(captured[1]
                .iter()
                .any(|message| message.text().contains("correct that")));
        })
        .await;
}

struct WaitTool {
    started: Arc<Notify>,
}

#[async_trait]
impl Tool for WaitTool {
    fn name(&self) -> &str {
        "wait"
    }

    fn description(&self) -> &str {
        "wait forever"
    }

    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }

    async fn execute(&self, _input: Value) -> Result<String> {
        self.started.notify_one();
        std::future::pending::<()>().await;
        unreachable!()
    }
}

#[tokio::test(flavor = "current_thread")]
async fn interruption_pairs_the_running_and_unstarted_tool_uses() {
    LocalSet::new()
        .run_until(async {
            let started = Arc::new(Notify::new());
            let response = CompletionResponse {
                content: vec![
                    ContentBlock::ToolUse {
                        name: "wait".into(),
                        input: json!({}),
                        id: "running".into(),
                    },
                    ContentBlock::ToolUse {
                        name: "wait".into(),
                        input: json!({}),
                        id: "not-started".into(),
                    },
                ],
                stop_reason: StopReason::ToolUse,
                usage: None,
            };
            let mut registry = ToolRegistry::new();
            registry
                .register(Arc::new(WaitTool {
                    started: Arc::clone(&started),
                }))
                .unwrap();
            let queue = PromptQueue::default();
            let (cancel, mut control) = TurnControl::for_turn(queue);

            let task = tokio::task::spawn_local(async move {
                let mut agent =
                    Agent::new(Box::new(Script::new(vec![Ok(response)])), registry, "test");
                agent.code_mode = false;
                agent.begin_turn("go");
                let mut events = Vec::new();
                let outcome = agent
                    .run_started_turn(&mut |event| events.push(event), &mut control)
                    .await
                    .unwrap();
                (agent, outcome, events)
            });

            started.notified().await;
            cancel.cancel();
            let (agent, outcome, events) = task.await.unwrap();
            assert_eq!(outcome, TurnOutcome::Interrupted);
            assert!(events.iter().any(|event| {
                matches!(
                    event,
                    AgentEvent::ToolCallFinished {
                        outcome: ToolCallOutcome::Cancelled,
                        ..
                    }
                )
            }));

            let result_message = &agent.history[2];
            let result_ids = result_message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(result_ids, vec!["running", "not-started"]);
            assert!(result_message.content.iter().all(|block| {
                matches!(
                    block,
                    ContentBlock::ToolResult {
                        is_error: Some(true),
                        ..
                    }
                )
            }));
            assert!(history_tool_protocol_is_valid(&agent.history));
            for event in events {
                if let AgentEvent::HistoryCheckpoint { history, .. } = event {
                    assert!(history_tool_protocol_is_valid(&history));
                }
            }
        })
        .await;
}

struct GatedPermission {
    started: Arc<Notify>,
    answer: Mutex<Option<oneshot::Receiver<PermissionDecision>>>,
}

#[async_trait]
impl ToolPermissionHandler for GatedPermission {
    async fn check_permission(&self, _request: &ToolExecutionRequest) -> PermissionDecision {
        self.started.notify_one();
        let answer = self
            .answer
            .lock()
            .unwrap()
            .take()
            .expect("one permission request");
        answer.await.unwrap_or(PermissionDecision::Deny)
    }
}

struct CountingDenyPermission {
    checks: Arc<AtomicUsize>,
}

#[async_trait]
impl ToolPermissionHandler for CountingDenyPermission {
    async fn check_permission(&self, _request: &ToolExecutionRequest) -> PermissionDecision {
        self.checks.fetch_add(1, Ordering::SeqCst);
        PermissionDecision::Deny
    }
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_wins_over_a_ready_permission_before_steering() {
    LocalSet::new()
        .run_until(async {
            let started = Arc::new(Notify::new());
            let (answer, answer_rx) = oneshot::channel();
            let mut registry = ToolRegistry::with_permission_handler(Box::new(GatedPermission {
                started: Arc::clone(&started),
                answer: Mutex::new(Some(answer_rx)),
            }));
            registry.register(Arc::new(Echo)).unwrap();
            let queue = PromptQueue::default();
            let steer = queue.enqueue("do not commit me", DeliveryMode::Steer);
            let (cancel, mut control) = TurnControl::for_turn(queue.clone());
            let response = tool_response();

            let task = tokio::task::spawn_local(async move {
                let mut agent =
                    Agent::new(Box::new(Script::new(vec![Ok(response)])), registry, "test");
                agent.code_mode = false;
                agent.begin_turn("go");
                let mut events = Vec::new();
                let outcome = agent
                    .run_started_turn(&mut |event| events.push(event), &mut control)
                    .await
                    .unwrap();
                (agent, outcome, events)
            });

            started.notified().await;
            answer.send(PermissionDecision::Allow).unwrap();
            cancel.cancel();
            let (agent, outcome, events) = task.await.unwrap();

            assert_eq!(outcome, TurnOutcome::Interrupted);
            assert_eq!(queue.snapshot()[0].id, steer);
            assert!(!events
                .iter()
                .any(|event| matches!(event, AgentEvent::SteeringCommitted { .. })));
            assert!(history_tool_protocol_is_valid(&agent.history));
        })
        .await;
}

#[tokio::test(flavor = "current_thread")]
async fn provider_cancellation_commits_no_partial_assistant_message() {
    LocalSet::new()
        .run_until(async {
            let calls = Arc::new(AtomicUsize::new(0));
            let requests = Arc::new(Mutex::new(Vec::new()));
            let first_started = Arc::new(Notify::new());
            let (_release, first_release) = oneshot::channel();
            let provider = GatedFinalProvider {
                calls,
                requests,
                first_started: Arc::clone(&first_started),
                first_release: Mutex::new(Some(first_release)),
            };
            let queue = PromptQueue::default();
            let steer = queue.enqueue("still queued", DeliveryMode::Steer);
            let queue_for_turn = queue.clone();
            let (cancel, mut control) = TurnControl::for_turn(queue_for_turn);

            let task = tokio::task::spawn_local(async move {
                let mut agent = Agent::new(Box::new(provider), ToolRegistry::new(), "test");
                agent.begin_turn("initial");
                let outcome = agent
                    .run_started_turn(&mut |_| {}, &mut control)
                    .await
                    .unwrap();
                (agent, outcome)
            });

            first_started.notified().await;
            cancel.cancel();
            let (agent, outcome) = task.await.unwrap();

            assert_eq!(outcome, TurnOutcome::Interrupted);
            assert_eq!(agent.history.len(), 1);
            assert_eq!(queue.snapshot()[0].id, steer);
            assert!(history_tool_protocol_is_valid(&agent.history));
        })
        .await;
}

struct GatedStreamingProvider {
    started: Arc<Notify>,
}

#[async_trait(?Send)]
impl Provider for GatedStreamingProvider {
    fn id(&self) -> &'static str {
        "gated-stream"
    }

    fn model(&self) -> &str {
        "gated-stream"
    }

    async fn complete(&self, _request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        unreachable!("streaming path only")
    }

    async fn complete_streaming(
        &self,
        _request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        on_delta(CompletionDelta::Text("visible but uncommitted".to_string()))?;
        self.started.notify_one();
        std::future::pending::<Result<CompletionResponse>>().await
    }
}

#[tokio::test(flavor = "current_thread")]
async fn cancelling_a_partial_stream_marks_the_visible_text_uncommitted() {
    LocalSet::new()
        .run_until(async {
            let started = Arc::new(Notify::new());
            let queue = PromptQueue::default();
            let (cancel, mut control) = TurnControl::for_turn(queue);
            let provider = GatedStreamingProvider {
                started: Arc::clone(&started),
            };

            let task = tokio::task::spawn_local(async move {
                let mut agent = Agent::new(Box::new(provider), ToolRegistry::new(), "test");
                agent.begin_turn("initial");
                let mut events = Vec::new();
                let outcome = agent
                    .run_started_turn(&mut |event| events.push(event), &mut control)
                    .await
                    .unwrap();
                (agent, outcome, events)
            });

            started.notified().await;
            cancel.cancel();
            let (agent, outcome, events) = task.await.unwrap();

            assert_eq!(outcome, TurnOutcome::Interrupted);
            assert_eq!(agent.history.len(), 1);
            assert!(events.iter().any(|event| {
                matches!(
                    event,
                    AgentEvent::AssistantStreamAborted { reason }
                        if reason.contains("interrupted before")
                )
            }));
        })
        .await;
}

#[tokio::test]
async fn custom_provider_oversized_final_response_never_enters_history() {
    let mut agent = Agent::new(
        Box::new(Script::new(vec![Ok(text_response("too large"))])),
        ToolRegistry::new(),
        "test",
    );
    agent.completion_limits = CompletionLimits {
        max_response_bytes: 4,
        ..CompletionLimits::default()
    };
    let mut events = Vec::new();

    let error = agent
        .run_turn("question", &mut |event| events.push(event))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("payload limit of 4 bytes"));
    assert_eq!(agent.history.len(), 1);
    assert!(matches!(
        agent.history[0].content[0],
        ContentBlock::Text { .. }
    ));
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::ApiCallFinished { usage: None })));
    assert!(!events.iter().any(|event| matches!(
        event,
        AgentEvent::AssistantText(_) | AgentEvent::StreamCommitted { .. }
    )));
}

#[tokio::test]
async fn custom_stream_stops_on_host_callback_limit_and_marks_preview_uncommitted() {
    struct LimitAwareStream {
        attempted: Arc<AtomicUsize>,
    }

    #[async_trait(?Send)]
    impl Provider for LimitAwareStream {
        fn id(&self) -> &'static str {
            "limit-aware"
        }

        fn model(&self) -> &str {
            "limit-aware"
        }

        async fn complete(&self, _request: CompletionRequest<'_>) -> Result<CompletionResponse> {
            unreachable!("streaming path only")
        }

        async fn complete_streaming(
            &self,
            _request: CompletionRequest<'_>,
            on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
        ) -> Result<CompletionResponse> {
            for fragment in ["abc", "def", "must-not-run"] {
                self.attempted.fetch_add(1, Ordering::SeqCst);
                on_delta(CompletionDelta::Text(fragment.into()))?;
            }
            unreachable!("the host limit must stop this provider")
        }
    }

    let attempted = Arc::new(AtomicUsize::new(0));
    let mut agent = Agent::new(
        Box::new(LimitAwareStream {
            attempted: Arc::clone(&attempted),
        }),
        ToolRegistry::new(),
        "test",
    );
    agent.completion_limits = CompletionLimits {
        max_response_bytes: 5,
        ..CompletionLimits::default()
    };
    let mut events = Vec::new();

    let error = agent
        .run_turn("question", &mut |event| events.push(event))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("payload limit of 5 bytes"));
    assert_eq!(attempted.load(Ordering::SeqCst), 2);
    assert_eq!(agent.history.len(), 1);
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::AssistantTextDelta(text) if text == "abc")));
    assert!(!events
        .iter()
        .any(|event| matches!(event, AgentEvent::AssistantTextDelta(text) if text == "def")));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::AssistantStreamAborted { reason }
            if reason.contains("failed before it was committed")
    )));
}

#[tokio::test]
async fn ignored_callback_limit_error_still_prevents_custom_provider_commit() {
    struct IgnoresCallbackError;

    #[async_trait(?Send)]
    impl Provider for IgnoresCallbackError {
        fn id(&self) -> &'static str {
            "ignores-callback"
        }

        fn model(&self) -> &str {
            "ignores-callback"
        }

        async fn complete(&self, _request: CompletionRequest<'_>) -> Result<CompletionResponse> {
            unreachable!("streaming path only")
        }

        async fn complete_streaming(
            &self,
            _request: CompletionRequest<'_>,
            on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
        ) -> Result<CompletionResponse> {
            on_delta(CompletionDelta::Text("abc".into()))?;
            let _ignored = on_delta(CompletionDelta::Text("def".into()));
            Ok(text_response("ok"))
        }
    }

    let mut agent = Agent::new(Box::new(IgnoresCallbackError), ToolRegistry::new(), "test");
    agent.completion_limits = CompletionLimits {
        max_response_bytes: 5,
        ..CompletionLimits::default()
    };
    let mut events = Vec::new();

    let error = agent
        .run_turn("question", &mut |event| events.push(event))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("payload limit of 5 bytes"));
    assert_eq!(agent.history.len(), 1);
    assert!(events
        .iter()
        .any(|event| matches!(event, AgentEvent::AssistantStreamAborted { .. })));
    assert!(!events
        .iter()
        .any(|event| matches!(event, AgentEvent::StreamCommitted { .. })));
}

#[tokio::test]
async fn oversized_tool_burst_is_rejected_before_execution_or_commit() {
    let response = CompletionResponse {
        content: (0..2)
            .map(|index| ContentBlock::ToolUse {
                name: "echo".into(),
                input: json!({}),
                id: format!("tool-{index}"),
            })
            .collect(),
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let mut agent = agent_with(Script::new(vec![Ok(response)]));
    agent.completion_limits = CompletionLimits {
        max_tool_uses: 1,
        ..CompletionLimits::default()
    };
    let mut events = Vec::new();

    let error = agent
        .run_turn("question", &mut |event| events.push(event))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("2 tool calls"));
    assert_eq!(agent.history.len(), 1);
    assert!(agent.registry.execution_history().is_empty());
    assert!(!events
        .iter()
        .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. })));
}

#[tokio::test]
async fn iteration_limit_leaves_late_steering_for_controller_normalization() {
    let queue = PromptQueue::default();
    let steer = queue.enqueue("too late", DeliveryMode::Steer);
    let (_cancel, mut control) = TurnControl::for_turn(queue.clone());
    let mut agent = agent_with(Script::new(vec![Ok(tool_response())]));
    agent.max_iterations = 1;
    agent.begin_turn("go");

    let outcome = agent
        .run_started_turn(&mut |_| {}, &mut control)
        .await
        .unwrap();

    assert_eq!(outcome, TurnOutcome::MaxIterationsReached);
    assert_eq!(queue.snapshot()[0].id, steer);
    assert_eq!(queue.snapshot()[0].delivery, DeliveryMode::Steer);
    assert!(history_tool_protocol_is_valid(&agent.history));

    queue.normalize_steers();
    assert_eq!(queue.snapshot()[0].delivery, DeliveryMode::FollowUp);
}

#[tokio::test]
async fn refusal_with_tool_uses_is_repaired_before_checkpointing() {
    let response = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "echo".into(),
            input: json!({}),
            id: "refused-tool".into(),
        }],
        stop_reason: StopReason::Refusal,
        usage: None,
    };
    let mut agent = agent_with(Script::new(vec![Ok(response)]));
    let mut checkpoints = Vec::new();
    let outcome = agent
        .run_turn("go", &mut |event| {
            if let AgentEvent::HistoryCheckpoint { history, .. } = event {
                checkpoints.push(history);
            }
        })
        .await
        .unwrap();

    assert_eq!(outcome, TurnOutcome::Refused);
    assert!(agent.registry.execution_history().is_empty());
    assert!(history_tool_protocol_is_valid(&agent.history));
    assert!(!checkpoints.is_empty());
    assert!(checkpoints
        .iter()
        .all(|history| history_tool_protocol_is_valid(history)));
}

#[tokio::test]
async fn denial_inside_code_mode_pauses_the_outer_turn() {
    let code = r#"
import tools
try:
    tools.mirror(marker="denied")
except Exception:
    pass
print("script continued")
"#;
    let response = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "python".into(),
            input: json!({"code": code}),
            id: "python-call".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let mut registry = ToolRegistry::with_permission_handler(Box::new(PolicyPermissions::new(
        vec!["python".into()],
        false,
    )));
    registry.register(Arc::new(Mirror)).unwrap();
    let mut agent = Agent::new(Box::new(Script::new(vec![Ok(response)])), registry, "test");
    let mut saw_nested_denial = false;

    let outcome = agent
        .run_turn("go", &mut |event| {
            if matches!(
                event,
                AgentEvent::ToolCallFinished {
                    ref name,
                    outcome: ToolCallOutcome::Denied,
                    ..
                } if name == "mirror"
            ) {
                saw_nested_denial = true;
            }
        })
        .await
        .unwrap();

    assert_eq!(outcome, TurnOutcome::PausedOnDenial);
    assert!(saw_nested_denial);
    assert!(history_tool_protocol_is_valid(&agent.history));
    assert!(matches!(
        &agent.history[2].content[0],
        ContentBlock::ToolResult {
            is_error: Some(true),
            ..
        }
    ));
}

#[test]
fn history_validator_rejects_dangling_or_orphan_tool_blocks() {
    let assistant_use = Message::assistant(vec![ContentBlock::ToolUse {
        name: "echo".into(),
        input: json!({}),
        id: "tool".into(),
    }]);
    let matching_result = Message::user(vec![ContentBlock::ToolResult {
        content: "ok".into(),
        tool_use_id: "tool".into(),
        is_error: None,
    }]);
    let orphan_result = Message::user(vec![ContentBlock::ToolResult {
        content: "bad".into(),
        tool_use_id: "other".into(),
        is_error: Some(true),
    }]);

    assert!(!history_tool_protocol_is_valid(std::slice::from_ref(
        &assistant_use
    )));
    assert!(!history_tool_protocol_is_valid(std::slice::from_ref(
        &orphan_result
    )));
    assert!(history_tool_protocol_is_valid(&[
        assistant_use,
        matching_result
    ]));
}

#[test]
fn code_mode_advertises_only_python() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Mirror)).unwrap();
    let agent = Agent::new(Box::new(Script::new(vec![])), registry, "test");

    let defs = agent.model_tool_defs();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "python");
    assert!(defs[0].description.contains("tools.mirror"));
    assert!(defs[0].description.contains("__doc__"));
    assert!(defs[0].description.contains("already bound to `tools`"));
    assert!(defs[0]
        .description
        .contains("never emit `<name>` or `tools.<name>` as a native tool call"));
}

#[test]
fn active_goal_adds_only_the_host_control_tool_to_code_mode() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Mirror)).unwrap();
    let mut agent = Agent::new(Box::new(Script::new(vec![])), registry, "test");
    agent.set_goal(Some("finish the task".into()));

    let defs = agent.model_tool_defs();
    assert_eq!(
        defs.iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["python", UPDATE_GOAL_TOOL_NAME]
    );
    assert_eq!(
        defs[1].input_schema["properties"]["status"]["enum"],
        json!(["complete"])
    );
}

#[tokio::test]
async fn code_mode_rejects_unadvertised_direct_tool_calls() {
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Mirror)).unwrap();
    let direct_call = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            // Some OpenAI-compatible models copy the Python expression
            // out of the prompt and return it as a native function name.
            name: "tools.mirror".into(),
            input: json!({"marker": "must-not-run"}),
            id: "t1".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let mut agent = Agent::new(
        Box::new(Script::new(vec![
            Ok(direct_call),
            Ok(text_response("done")),
        ])),
        registry,
        "test",
    );

    let mut events = Vec::new();
    let outcome = agent
        .run_turn("go", &mut |event| events.push(event))
        .await
        .unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(agent.registry.execution_history().is_empty());
    assert!(!events.iter().any(|event| {
        matches!(
            event,
            AgentEvent::ToolCallStarted { name, .. } if name == "tools.mirror"
        )
    }));
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentEvent::Notice(message)
                if message.contains("Rejected undeclared native tool call `tools.mirror`")
        )
    }));
    match &agent.history[2].content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => {
            assert_eq!(*is_error, Some(true));
            assert!(content.contains("code mode permits only"));
            assert!(content.contains("tools.mirror(...)"));
            assert!(!content.contains("tools.tools.mirror"));
        }
        other => panic!("expected tool result, got {:?}", other),
    }
    assert!(history_tool_protocol_is_valid(&agent.history));
}

#[tokio::test]
async fn malformed_code_mode_input_fails_before_permission_check() {
    let malformed_call = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "python".into(),
            input: json!({"_unparsed_arguments": "not json"}),
            id: "malformed-python".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let checks = Arc::new(AtomicUsize::new(0));
    let registry = ToolRegistry::with_permission_handler(Box::new(CountingDenyPermission {
        checks: Arc::clone(&checks),
    }));
    let mut agent = Agent::new(
        Box::new(Script::new(vec![
            Ok(malformed_call),
            Ok(text_response("recovered")),
        ])),
        registry,
        "test",
    );
    let mut events = Vec::new();

    let outcome = agent
        .run_turn("go", &mut |event| events.push(event))
        .await
        .unwrap();

    assert_eq!(outcome, TurnOutcome::Completed);
    assert_eq!(checks.load(Ordering::SeqCst), 0);
    assert!(agent.registry.execution_history().is_empty());
    assert!(events.iter().any(|event| {
        matches!(
            event,
            AgentEvent::ToolCallFinished {
                name,
                outcome: ToolCallOutcome::Failed,
                ..
            } if name == "python"
        )
    }));
    match &agent.history[2].content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => {
            assert_eq!(*is_error, Some(true));
            assert!(content.contains("string `code` field"));
            assert!(content.contains("Retry the `python` tool call"));
        }
        other => panic!("expected tool result, got {:?}", other),
    }
    assert!(history_tool_protocol_is_valid(&agent.history));
}

#[tokio::test]
async fn tool_results_are_truncated_in_history() {
    let mut agent = agent_with(Script::new(vec![
        Ok(tool_response()),
        Ok(text_response("done")),
    ]));
    agent.max_tool_result_chars = 500;
    let outcome = agent.run_turn("go", &mut |_| {}).await.unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);

    // history: user, assistant(tool_use), user(tool_result), assistant(text)
    assert_eq!(agent.history.len(), 4);
    match &agent.history[2].content[0] {
        ContentBlock::ToolResult { content, .. } => {
            assert!(
                content.chars().count() <= 600,
                "not truncated: {}",
                content.len()
            );
            assert!(content.contains("truncated"));
        }
        other => panic!("expected tool result, got {:?}", other),
    }
}

#[tokio::test]
async fn history_survives_api_errors_after_tool_execution() {
    let mut agent = agent_with(Script::new(vec![
        Ok(tool_response()),
        Err(Error::Api {
            status: 400,
            message: "boom".into(),
            retry_after: None,
            error_type: None,
        }),
    ]));
    let result = agent.run_turn("go", &mut |_| {}).await;
    assert!(result.is_err());
    // The user message, the assistant tool_use, and the tool result must
    // all still be present so the side effects are on record.
    assert_eq!(agent.history.len(), 3);
    assert!(agent.history[1]
        .content
        .iter()
        .any(|b| matches!(b, ContentBlock::ToolUse { .. })));
}

#[tokio::test]
async fn transient_errors_are_retried() {
    let script = Script::new(vec![
        Err(Error::Api {
            status: 529,
            message: "overloaded".into(),
            retry_after: None,
            error_type: None,
        }),
        Ok(text_response("recovered")),
    ]);
    let mut agent = agent_with(script);
    let mut retried = false;
    let outcome = agent
        .run_turn("hi", &mut |e| {
            if matches!(e, AgentEvent::Retrying { .. }) {
                retried = true;
            }
        })
        .await
        .unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(retried);
}

/// Empirical check for the estimate cache: repeated context_tokens()
/// calls on an unchanging history must be far cheaper than one fresh
/// estimate per call. Run with `cargo test -- --nocapture estimate_cache`.

#[test]
fn estimate_cache_is_fast_on_repeated_calls() {
    let mut agent = agent_with(Script::new(vec![]));
    for i in 0..100 {
        agent.history.push(Message::user_text(format!(
            "user message {i} with a fair amount of text to serialize {}",
            "x".repeat(400)
        )));
        agent
            .history
            .push(Message::assistant(vec![ContentBlock::Text {
                text: format!("assistant reply {i} {}", "y".repeat(400)),
            }]));
    }
    let calls = 2_000;
    let start = std::time::Instant::now();
    let mut acc = 0u64;
    for _ in 0..calls {
        acc += agent.context_tokens();
    }
    let elapsed = start.elapsed();
    eprintln!(
        "BENCH context_tokens: {calls} calls in {:?} ({:.1} ns/call), acc={acc}",
        elapsed,
        elapsed.as_nanos() as f64 / calls as f64
    );
    assert!(
        elapsed.as_millis() < 100,
        "cache ineffective: {:?}",
        elapsed
    );
}

#[test]
fn estimated_context_cache_tracks_appends() {
    let mut agent = agent_with(Script::new(vec![]));
    // No provider measurement yet: falls back to the estimate.
    let empty = agent.context_tokens();
    assert_eq!(empty, estimate_tokens(&agent.history));
    // Cached: same value without a provider measurement.
    assert_eq!(agent.context_tokens(), empty);

    // An append invalidates the cache (length changed, same revision).
    agent.begin_turn("hello world, this is a longer message");
    let after = agent.context_tokens();
    assert_eq!(after, estimate_tokens(&agent.history));
    assert!(after > empty);

    // A provider measurement always wins over the estimate.
    agent.last_context_tokens = Some(123);
    assert_eq!(agent.context_tokens(), 123);
}

#[test]
fn estimated_context_cache_tracks_in_place_steering() {
    let mut agent = agent_with(Script::new(vec![]));
    agent.begin_turn("initial");
    let before = agent.context_tokens();

    let queue = PromptQueue::default();
    queue.enqueue("x".repeat(400), DeliveryMode::Steer);
    let (_cancel, control) = TurnControl::for_turn(queue);
    assert!(agent.commit_steering(&control, &mut |_| {}));

    let after = agent.context_tokens();
    assert_eq!(after, estimate_tokens(agent.history()));
    assert!(after > before);
}

#[test]
fn retry_delay_honors_retry_after_as_floor_with_cap() {
    // Pure backoff: 1, 2, 4, 8.
    assert_eq!(retry_delay_secs(0, None), 1);
    assert_eq!(retry_delay_secs(1, None), 2);
    assert_eq!(retry_delay_secs(3, None), 8);
    // Retry-After raises the floor when it exceeds the backoff.
    assert_eq!(retry_delay_secs(0, Some(30)), 30);
    assert_eq!(retry_delay_secs(0, Some(1)), 1);
    // ... but never past the 60s cap, however large the header.
    assert_eq!(retry_delay_secs(0, Some(3600)), 60);
    assert_eq!(retry_delay_secs(2, Some(0)), 4);
}

/// End-to-end code mode: the model "writes" one script that calls a tool
/// several times through the generated `tools` module. All bridged calls
/// happen before the next provider round-trip; only the script's stdout
/// becomes the outer tool result. Requires python3 on PATH.
/// Echoes its input back — small output, unlike `Echo` above.
struct Mirror;

#[async_trait]
impl Tool for Mirror {
    fn name(&self) -> &str {
        "mirror"
    }
    fn description(&self) -> &str {
        "mirror"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, input: Value) -> Result<String> {
        Ok(format!("mirror:{}", input))
    }
}

#[tokio::test]
async fn code_mode_bridges_tool_calls_into_scripts() {
    let code = r#"
# The runner preloads `tools`; an explicit import is optional.
results = [tools.mirror(marker=f"xyzzy-{i}") for i in range(3)]
print("BRIDGED:", "|".join(results))
try:
    tools.mirror_not_a_tool()
except Exception as e:
    print("RAISED OK")
"#;
    let script_call = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "python".into(),
            input: json!({"code": code}),
            id: "t1".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Mirror)).unwrap();
    let mut agent = Agent::new(
        Box::new(Script::new(vec![
            Ok(script_call),
            Ok(text_response("done")),
        ])),
        registry,
        "test",
    );
    let mut bridged_calls = 0;
    let outcome = agent
        .run_turn("go", &mut |e| {
            if let AgentEvent::ToolCallStarted { name, .. } = &e {
                if name == "mirror" {
                    bridged_calls += 1;
                }
            }
        })
        .await
        .unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);

    // The script's stdout is the tool result the model sees...
    let result_text = match &agent.history[2].content[0] {
        ContentBlock::ToolResult { content, .. } => content.clone(),
        other => panic!("expected tool result, got {:?}", other),
    };
    assert_eq!(
        bridged_calls, 3,
        "script did not batch calls: {}",
        result_text
    );
    assert!(
        result_text.contains("BRIDGED:"),
        "missing bridge output: {}",
        result_text
    );
    assert!(result_text.contains("xyzzy-0"));
    assert!(result_text.contains("xyzzy-2"));
    assert!(
        result_text.contains("RAISED OK"),
        "bad tool name must raise: {}",
        result_text
    );
    // ...and the bridged echo result appears nowhere else in history
    // (it reached the model only because the script chose to print it).
    assert_eq!(agent.history.len(), 4);
}

/// A script that succeeds but prints nothing must not return an empty
/// tool result: the model would otherwise get no signal that its bridged
/// tool calls ran. Requires python3 on PATH.
#[tokio::test]
async fn silent_script_reports_bridged_call_count() {
    let code = r#"
import tools
tools.mirror(marker="silent")
# no print: side effect only
"#;
    let script_call = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "python".into(),
            input: json!({"code": code}),
            id: "t1".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(Mirror)).unwrap();
    let mut agent = Agent::new(
        Box::new(Script::new(vec![
            Ok(script_call),
            Ok(text_response("done")),
        ])),
        registry,
        "test",
    );
    let outcome = agent.run_turn("go", &mut |_| {}).await.unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);

    let result_text = match &agent.history[2].content[0] {
        ContentBlock::ToolResult { content, .. } => content.clone(),
        other => panic!("expected tool result, got {:?}", other),
    };
    assert!(
        result_text.contains("1 tool call") && result_text.contains("no output"),
        "silent script should report its bridged calls, got: {:?}",
        result_text
    );
}

#[tokio::test]
async fn truncated_tool_calls_are_failed_not_executed() {
    let truncated = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: "echo".into(),
            input: json!({}),
            id: "t1".into(),
        }],
        stop_reason: StopReason::MaxTokens,
        usage: None,
    };
    let mut agent = agent_with(Script::new(vec![Ok(truncated), Ok(text_response("ok"))]));
    let mut tool_ran = false;
    let outcome = agent
        .run_turn("go", &mut |e| {
            if matches!(e, AgentEvent::ToolCallStarted { .. }) {
                tool_ran = true;
            }
        })
        .await
        .unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(!tool_ran, "truncated tool call must not execute");
    match &agent.history[2].content[0] {
        ContentBlock::ToolResult {
            is_error, content, ..
        } => {
            assert_eq!(*is_error, Some(true));
            assert!(content.contains("Re-issue"));
        }
        other => panic!("expected tool result, got {:?}", other),
    }
}

#[tokio::test]
async fn compaction_summarizes_old_history_and_preserves_recent() {
    // Script: first call answers the compaction request, second the turn.
    let mut agent = agent_with(Script::new(vec![
        Ok(text_response("SUMMARY-OF-EARLIER-WORK")),
        Ok(text_response("done")),
    ]));
    for i in 0..6 {
        agent
            .history
            .push(Message::user_text(format!("question {}", i)));
        agent
            .history
            .push(Message::assistant(vec![ContentBlock::Text {
                text: format!("answer {} {}", i, "x".repeat(400)),
            }]));
    }
    agent.compaction_threshold_tokens = 10; // force compaction
    agent.compaction_keep_recent_tokens = 200;

    let before = agent.history.len();
    let history_revision = agent.history_revision();
    let outcome = agent
        .run_turn("latest question", &mut |_| {})
        .await
        .unwrap();
    assert_eq!(outcome, TurnOutcome::Completed);
    assert_ne!(agent.history_revision(), history_revision);
    assert!(agent.history.len() < before, "history did not shrink");
    match &agent.history[0].content[0] {
        ContentBlock::Text { text } => {
            assert!(
                text.contains("[Context summary"),
                "no summary marker: {}",
                text
            );
            assert!(text.contains("SUMMARY-OF-EARLIER-WORK"));
        }
        other => panic!("expected text, got {:?}", other),
    }
    // The latest user question and final answer survive verbatim.
    let flat: String = agent
        .history
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("|");
    assert!(flat.contains("latest question"));
    assert!(flat.ends_with("done"));
}

struct RecordingScript {
    steps: Mutex<Vec<Result<CompletionResponse>>>,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
}

#[async_trait(?Send)]
impl Provider for RecordingScript {
    fn id(&self) -> &'static str {
        "recording"
    }
    fn model(&self) -> &str {
        "recording"
    }
    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        self.requests
            .lock()
            .unwrap()
            .push(request.messages.to_vec());
        self.steps.lock().unwrap().pop().expect("script exhausted")
    }
}

fn tool_pair(id: &str, result: &str) -> [Message; 2] {
    [
        Message::assistant(vec![ContentBlock::ToolUse {
            name: "echo".into(),
            input: json!({"payload": "p"}),
            id: id.into(),
        }]),
        Message::user(vec![ContentBlock::ToolResult {
            content: result.to_string(),
            tool_use_id: id.into(),
            is_error: None,
        }]),
    ]
}

#[tokio::test]
async fn compaction_request_replays_tool_history_as_plain_text() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = RecordingScript {
        steps: Mutex::new(vec![Ok(text_response("TOOL-HEAVY-SUMMARY"))]),
        requests: Arc::clone(&requests),
    };
    let mut agent = Agent::new(Box::new(provider), ToolRegistry::new(), "test");
    agent.history.push(Message::user_text("start"));
    agent.history.extend(tool_pair("t1", &"y".repeat(2000)));
    agent.history.extend(tool_pair("t2", &"y".repeat(2000)));
    agent
        .history
        .push(Message::assistant(vec![ContentBlock::Text {
            text: "found it".into(),
        }]));
    agent.history.push(Message::user_text("next question"));
    agent
        .history
        .push(Message::assistant(vec![ContentBlock::Text {
            text: "recent answer".into(),
        }]));
    agent.compaction_keep_recent_tokens = 50;

    assert!(agent.compact(&mut |_| {}).await.unwrap());

    // The summarization request must be valid without tool definitions:
    // no tool_use/tool_result blocks, their contents rendered as text.
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let compaction_request = &captured[0];
    assert!(compaction_request.iter().all(|message| {
        message.content.iter().all(|block| {
            !matches!(
                block,
                ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. }
            )
        })
    }));
    let flat: String = compaction_request
        .iter()
        .map(|m| m.text())
        .collect::<Vec<_>>()
        .join("|");
    assert!(flat.contains("[called tool echo"));
    assert!(flat.contains("[tool result]"));
    drop(captured);

    assert!(history_tool_protocol_is_valid(&agent.history));
    assert!(agent.history[0].text().contains("TOOL-HEAVY-SUMMARY"));
}

#[tokio::test]
async fn mid_turn_compaction_cuts_at_tool_result_boundary() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let provider = RecordingScript {
        steps: Mutex::new(vec![Ok(text_response("MID-TURN-SUMMARY"))]),
        requests: Arc::clone(&requests),
    };
    let mut agent = Agent::new(Box::new(provider), ToolRegistry::new(), "test");
    // One long tool-use turn: no plain user boundary after the opener.
    agent.history.push(Message::user_text("start big goal"));
    agent.history.extend(tool_pair("t1", &"y".repeat(2000)));
    agent.history.extend(tool_pair("t2", &"y".repeat(2000)));
    agent.history.extend(tool_pair("t3", "small recent result"));
    agent.compaction_keep_recent_tokens = 100;

    assert!(
        agent.compact(&mut |_| {}).await.unwrap(),
        "mid-turn history must still be compactable"
    );

    assert!(history_tool_protocol_is_valid(&agent.history));
    assert!(agent.history[0].text().contains("MID-TURN-SUMMARY"));
    // The boundary's orphaned tool results were rendered as text.
    let boundary = &agent.history[1];
    assert_eq!(boundary.role, "user");
    assert!(boundary
        .content
        .iter()
        .all(|block| !matches!(block, ContentBlock::ToolResult { .. })));
    assert!(boundary
        .text()
        .contains("[tool result from a compacted call]"));
    // The recent, un-summarized pair survives verbatim as tool blocks.
    assert!(agent.history.iter().any(|message| {
        message.content.iter().any(|block| {
            matches!(
                block,
                ContentBlock::ToolResult { content, .. } if content == "small recent result"
            )
        })
    }));
}

/// Denies exactly one tool name, allowing everything else.
struct DenyNamed(&'static str);

#[async_trait]
impl ToolPermissionHandler for DenyNamed {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        if request.tool_name == self.0 {
            PermissionDecision::Deny
        } else {
            PermissionDecision::Allow
        }
    }
}

struct Blocked;

#[async_trait]
impl Tool for Blocked {
    fn name(&self) -> &str {
        "blocked"
    }
    fn description(&self) -> &str {
        "always denied by the test handler"
    }
    fn input_schema(&self) -> Value {
        json!({"type": "object"})
    }
    async fn execute(&self, _input: Value) -> Result<String> {
        Ok("must never run behind a denial".into())
    }
}

#[tokio::test]
async fn denial_mid_batch_stops_the_remaining_calls() {
    let batch = CompletionResponse {
        content: vec![
            ContentBlock::ToolUse {
                name: "echo".into(),
                input: json!({}),
                id: "t1".into(),
            },
            ContentBlock::ToolUse {
                name: "blocked".into(),
                input: json!({}),
                id: "t2".into(),
            },
            ContentBlock::ToolUse {
                name: "echo".into(),
                input: json!({}),
                id: "t3".into(),
            },
        ],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let mut registry = ToolRegistry::with_permission_handler(Box::new(DenyNamed("blocked")));
    registry.register(Arc::new(Echo)).unwrap();
    registry.register(Arc::new(Blocked)).unwrap();
    let mut agent = Agent::new(Box::new(Script::new(vec![Ok(batch)])), registry, "test");
    agent.code_mode = false;

    let outcome = agent.run_turn("go", &mut |_| {}).await.unwrap();
    assert_eq!(outcome, TurnOutcome::PausedOnDenial);
    assert!(history_tool_protocol_is_valid(&agent.history));

    let results = &agent.history[2];
    assert_eq!(results.content.len(), 3);
    match &results.content[2] {
        ContentBlock::ToolResult {
            content,
            tool_use_id,
            is_error,
        } => {
            assert_eq!(tool_use_id, "t3");
            assert_eq!(*is_error, Some(true));
            assert!(content.contains("was not executed"), "got: {content}");
        }
        other => panic!("expected tool result, got {:?}", other),
    }
    // The tool after the denial never reached execution.
    assert_eq!(agent.registry.execution_history().len(), 2);
}

#[test]
fn duplicate_tool_use_ids_are_rejected_by_the_protocol_validator() {
    let duplicate = vec![
        Message::user_text("go"),
        Message::assistant(vec![
            ContentBlock::ToolUse {
                name: "echo".into(),
                input: json!({}),
                id: "same".into(),
            },
            ContentBlock::ToolUse {
                name: "echo".into(),
                input: json!({}),
                id: "same".into(),
            },
        ]),
        Message::user(vec![ContentBlock::ToolResult {
            content: "x".into(),
            tool_use_id: "same".into(),
            is_error: None,
        }]),
    ];
    assert!(!history_tool_protocol_is_valid(&duplicate));
}

#[tokio::test]
async fn streamed_text_is_not_double_emitted() {
    struct Streamy;

    #[async_trait(?Send)]
    impl Provider for Streamy {
        fn id(&self) -> &'static str {
            "streamy"
        }
        fn model(&self) -> &str {
            "streamy"
        }
        async fn complete(&self, _r: CompletionRequest<'_>) -> Result<CompletionResponse> {
            unreachable!("streaming path only")
        }
        async fn complete_streaming(
            &self,
            _r: CompletionRequest<'_>,
            on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
        ) -> Result<CompletionResponse> {
            on_delta(CompletionDelta::Reasoning("because ".to_string()))?;
            on_delta(CompletionDelta::Reasoning("evidence".to_string()))?;
            on_delta(CompletionDelta::Text("hel".to_string()))?;
            on_delta(CompletionDelta::Text("lo".to_string()))?;
            Ok(CompletionResponse {
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "because evidence".to_string(),
                        signature: String::new(),
                    },
                    ContentBlock::Text {
                        text: "hello".to_string(),
                    },
                ],
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }
    }

    let mut agent = Agent::new(Box::new(Streamy), ToolRegistry::new(), "test");
    let mut deltas = String::new();
    let mut reasoning = String::new();
    let mut full_blocks = 0;
    let mut committed = None;
    agent
        .run_turn("hi", &mut |event| match event {
            AgentEvent::AssistantTextDelta(t) => deltas.push_str(&t),
            AgentEvent::ReasoningDelta(t) => reasoning.push_str(&t),
            AgentEvent::AssistantText(_) => full_blocks += 1,
            AgentEvent::StreamCommitted { text, reasoning } => committed = Some((text, reasoning)),
            _ => {}
        })
        .await
        .unwrap();
    assert_eq!(deltas, "hello");
    assert_eq!(reasoning, "because evidence");
    assert_eq!(full_blocks, 0, "streamed text must not be re-emitted whole");
    assert_eq!(
        committed,
        Some((
            Some("hello".to_string()),
            Some("because evidence".to_string())
        ))
    );
    assert!(matches!(
        agent.history[1].content[0],
        ContentBlock::Thinking { .. }
    ));
}

#[tokio::test]
async fn active_goal_is_injected_without_entering_conversation_history() {
    struct CaptureSystem {
        systems: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait(?Send)]
    impl Provider for CaptureSystem {
        fn id(&self) -> &'static str {
            "capture-system"
        }

        fn model(&self) -> &str {
            "capture-system"
        }

        async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
            self.systems
                .lock()
                .unwrap()
                .push(request.system.unwrap_or_default().to_string());
            Ok(text_response("done"))
        }
    }

    let systems = Arc::new(Mutex::new(Vec::new()));
    let provider = CaptureSystem {
        systems: Arc::clone(&systems),
    };
    let mut agent = Agent::new(Box::new(provider), ToolRegistry::new(), "base instructions");
    agent.set_goal(Some("  ship the async TUI  ".into()));
    assert_eq!(agent.goal(), Some("ship the async TUI"));

    agent.run_turn("first", &mut |_| {}).await.unwrap();
    assert!(agent
        .history
        .iter()
        .all(|message| !message.text().contains("Active session goal")));

    agent.set_goal(None);
    agent.run_turn("second", &mut |_| {}).await.unwrap();
    let systems = systems.lock().unwrap();
    assert!(systems[0].contains("base instructions"));
    assert!(systems[0].contains("## Active session goal"));
    assert!(systems[0].contains("ship the async TUI"));
    assert_eq!(systems[1], "base instructions");
}

#[tokio::test]
async fn update_goal_completes_without_capability_permission() {
    let completion = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: UPDATE_GOAL_TOOL_NAME.into(),
            input: json!({"status": "complete"}),
            id: "goal-complete".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let registry = ToolRegistry::with_permission_handler(Box::new(AlwaysDenyPermissions));
    let mut agent = Agent::new(
        Box::new(Script::new(vec![
            Ok(completion),
            Ok(text_response("verified and done")),
        ])),
        registry,
        "test",
    );
    agent.set_goal(Some("ship the goal loop".into()));
    let mut completed_goal = None;

    let outcome = agent
        .run_turn("begin", &mut |event| {
            if let AgentEvent::GoalCompleted { goal } = event {
                completed_goal = Some(goal);
            }
        })
        .await
        .unwrap();

    assert_eq!(outcome, TurnOutcome::Completed);
    assert_eq!(completed_goal.as_deref(), Some("ship the goal loop"));
    assert_eq!(agent.goal(), None);
    assert!(agent.registry.execution_history().is_empty());
    assert!(history_tool_protocol_is_valid(agent.history()));
    assert!(agent
        .history()
        .last()
        .unwrap()
        .text()
        .contains("verified and done"));
}

#[tokio::test]
async fn invalid_goal_completion_keeps_the_goal_active() {
    let invalid = CompletionResponse {
        content: vec![ContentBlock::ToolUse {
            name: UPDATE_GOAL_TOOL_NAME.into(),
            input: json!({"status": "complete", "because": "turn ending"}),
            id: "invalid-goal-complete".into(),
        }],
        stop_reason: StopReason::ToolUse,
        usage: None,
    };
    let mut agent = Agent::new(
        Box::new(Script::new(vec![
            Ok(invalid),
            Ok(text_response("still working")),
        ])),
        ToolRegistry::new(),
        "test",
    );
    agent.set_goal(Some("finish everything".into()));

    let outcome = agent.run_turn("begin", &mut |_| {}).await.unwrap();

    assert_eq!(outcome, TurnOutcome::Completed);
    assert_eq!(agent.goal(), Some("finish everything"));
    assert!(matches!(
        &agent.history()[2].content[0],
        ContentBlock::ToolResult {
            is_error: Some(true),
            ..
        }
    ));
}

#[tokio::test]
async fn non_retryable_errors_surface_immediately() {
    let mut agent = agent_with(Script::new(vec![Err(Error::Api {
        status: 401,
        message: "bad key".into(),
        retry_after: None,
        error_type: None,
    })]));
    let result = agent.run_turn("hi", &mut |_| {}).await;
    assert!(result.is_err());
    // User message is preserved for the next attempt.
    assert_eq!(agent.history.len(), 1);
}
