//! The agent loop: request → tool calls → results → repeat.
//!
//! [`Agent`] owns the conversation history, the provider, and the tool
//! registry. Progress is surfaced through [`AgentEvent`] callbacks so a UI
//! (or a test harness) can render what is happening without the loop knowing
//! anything about terminals.
//!
//! Design points, learned the hard way:
//!
//! - **History survives errors.** The history is mutated in place as the turn
//!   progresses, so an API failure after tools have run never erases the
//!   record of what actually happened.
//! - **Tool results are truncated before entering history.** A web crawl can
//!   return megabytes; without a cap every subsequent request would carry it.
//! - **Transient API errors are retried with backoff** before surfacing.
//! - **Denials are structured.** A permission denial ends the turn cleanly so
//!   the user can redirect, and is never inferred from result text.

use crate::error::Result;
use crate::provider::Provider;
use crate::tool::{ToolCallOutcome, ToolRegistry};
use crate::types::{
    estimate_tokens, truncate_middle, CompletionRequest, ContentBlock, Message, StopReason, Usage,
};
use serde_json::Value;
use std::time::Duration;

/// Progress notifications emitted during [`Agent::run_turn`].
#[derive(Debug)]
pub enum AgentEvent {
    /// A provider request is starting (show a spinner).
    ApiCallStarted,
    /// The provider request finished (hide the spinner). Fires on both
    /// success and failure so UI state is always balanced.
    ApiCallFinished { usage: Option<Usage> },
    /// Assistant-visible text, complete block. Emitted only when the
    /// provider did not stream (otherwise the same text already arrived as
    /// deltas).
    AssistantText(String),
    /// A streamed fragment of assistant text. Render incrementally; a final
    /// `ApiCallFinished` closes the message.
    AssistantTextDelta(String),
    /// A tool call is about to be checked for permission and executed.
    /// Emitted *before* execution so the user always sees the input first.
    ToolCallStarted { name: String, input: Value },
    /// A tool call finished; `content` is the (already truncated) result.
    ToolCallFinished {
        name: String,
        outcome: ToolCallOutcome,
        content: String,
    },
    /// A transient API error occurred; the agent will retry after a delay.
    Retrying {
        attempt: u32,
        max_retries: u32,
        delay_secs: u64,
        error: String,
    },
    /// Something the user should know (truncation, refusal, ...).
    Notice(String),
}

/// How a turn ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    /// The model produced a final response.
    Completed,
    /// The user denied a tool call; the loop stopped so they can redirect.
    PausedOnDenial,
    /// The iteration cap was reached before the model finished.
    MaxIterationsReached,
    /// The provider refused to answer (safety).
    Refused,
}

pub struct Agent {
    provider: Box<dyn Provider>,
    pub registry: ToolRegistry,
    pub system_prompt: String,
    pub history: Vec<Message>,
    /// Cap on request → tool → request rounds within a single turn.
    pub max_iterations: usize,
    /// `max_tokens` per completion.
    pub max_tokens: u32,
    /// Cap (in characters) on a single tool result as stored in history.
    pub max_tool_result_chars: usize,
    /// Retries for transient API errors, with exponential backoff.
    pub max_retries: u32,
    /// Code mode (Unix only): advertise a `python` tool whose scripts can
    /// call every registered tool via a generated `tools` module, keeping
    /// intermediate tool results out of the model's context.
    pub code_mode: bool,
    /// Compact (summarize) older history when the context reaches this many
    /// tokens. `u64::MAX` disables compaction.
    pub compaction_threshold_tokens: u64,
    /// How much recent history (in estimated tokens) stays verbatim when
    /// compacting.
    pub compaction_keep_recent_tokens: u64,
    /// Context size measured by the provider on the last completion.
    last_context_tokens: Option<u64>,
}

impl Agent {
    pub fn new(
        provider: Box<dyn Provider>,
        registry: ToolRegistry,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            registry,
            system_prompt: system_prompt.into(),
            history: Vec::new(),
            max_iterations: 100,
            max_tokens: 16_000,
            max_tool_result_chars: 40_000,
            max_retries: 3,
            code_mode: cfg!(unix),
            compaction_threshold_tokens: 150_000,
            compaction_keep_recent_tokens: 20_000,
            last_context_tokens: None,
        }
    }

    /// Rough context estimate: provider-measured when available, chars/4
    /// over the serialized history otherwise.
    pub fn context_tokens(&self) -> u64 {
        self.last_context_tokens
            .unwrap_or_else(|| estimate_tokens(&self.history))
    }

    /// Whether a tool call should be routed to the code-mode runner instead
    /// of the registry. A registered tool named `python` always wins, so
    /// library users can override the built-in behavior.
    fn is_code_mode_call(&self, name: &str) -> bool {
        cfg!(unix) && self.code_mode && name == "python" && !self.registry.has_tool(name)
    }

    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    pub fn set_provider(&mut self, provider: Box<dyn Provider>) {
        self.provider = provider;
    }

    /// Run one user turn to completion (or pause/refusal/cap).
    ///
    /// On `Err`, the history still contains everything that happened up to
    /// the failure — including tool calls that already executed — so the
    /// conversation can continue after a retry or a model switch.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> Result<TurnOutcome> {
        self.history.push(Message::user_text(user_input));

        for _ in 0..self.max_iterations {
            if self.context_tokens() > self.compaction_threshold_tokens {
                if let Err(e) = self.compact(on_event).await {
                    on_event(AgentEvent::Notice(format!("Compaction failed: {}", e)));
                }
            }

            let (response, streamed) = self.complete_with_retry(on_event).await?;
            if let Some(usage) = &response.usage {
                self.last_context_tokens = Some(
                    usage.input_tokens
                        + usage.output_tokens
                        + usage.cache_read_input_tokens.unwrap_or(0)
                        + usage.cache_creation_input_tokens.unwrap_or(0),
                );
            }

            self.history
                .push(Message::assistant(response.content.clone()));

            if !streamed {
                for block in &response.content {
                    if let ContentBlock::Text { text } = block {
                        if !text.is_empty() {
                            on_event(AgentEvent::AssistantText(text.clone()));
                        }
                    }
                }
            }

            if response.stop_reason == StopReason::Refusal {
                on_event(AgentEvent::Notice(
                    "The model declined to continue with this request.".to_string(),
                ));
                return Ok(TurnOutcome::Refused);
            }

            let tool_uses: Vec<_> = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { name, input, id } => {
                        Some((name.clone(), input.clone(), id.clone()))
                    }
                    _ => None,
                })
                .collect();

            if tool_uses.is_empty() {
                if response.stop_reason == StopReason::MaxTokens {
                    on_event(AgentEvent::Notice(
                        "Response was cut off at the output-token limit and may be incomplete."
                            .to_string(),
                    ));
                }
                return Ok(TurnOutcome::Completed);
            }

            // A truncated response can contain tool calls whose JSON arguments
            // parsed but are silently incomplete. Executing them risks acting
            // on corrupt input (e.g. a half-written file edit), so fail every
            // call and let the model re-issue them with complete arguments.
            if response.stop_reason == StopReason::MaxTokens {
                on_event(AgentEvent::Notice(
                    "Response hit the output-token limit mid-tool-call; asking the model to re-issue."
                        .to_string(),
                ));
                let results = tool_uses
                    .into_iter()
                    .map(|(_, _, id)| ContentBlock::ToolResult {
                        content: "This tool call was not executed: the response hit the output \
                                  token limit, so its arguments may be truncated. Re-issue the \
                                  tool call with complete arguments."
                            .to_string(),
                        tool_use_id: id,
                        is_error: Some(true),
                    })
                    .collect();
                self.history.push(Message::user(results));
                continue;
            }

            let mut results = Vec::with_capacity(tool_uses.len());
            let mut denied = false;
            for (name, input, id) in tool_uses {
                on_event(AgentEvent::ToolCallStarted {
                    name: name.clone(),
                    input: input.clone(),
                });

                let mut result = if self.is_code_mode_call(&name) {
                    self.execute_code_mode(input, id, on_event).await
                } else {
                    self.registry.execute_tool(&name, input, id).await
                };
                if let ContentBlock::ToolResult { content, .. } = &mut result.block {
                    *content = truncate_middle(content, self.max_tool_result_chars);
                }
                let content = match &result.block {
                    ContentBlock::ToolResult { content, .. } => content.clone(),
                    _ => String::new(),
                };
                on_event(AgentEvent::ToolCallFinished {
                    name,
                    outcome: result.outcome,
                    content,
                });

                if result.outcome == ToolCallOutcome::Denied {
                    denied = true;
                }
                results.push(result.block);
            }

            // Every tool_use gets a result — required by the APIs even when
            // a call was denied.
            self.history.push(Message::user(results));

            if denied {
                on_event(AgentEvent::Notice(
                    "Tool call denied — pausing so you can redirect.".to_string(),
                ));
                return Ok(TurnOutcome::PausedOnDenial);
            }
        }

        on_event(AgentEvent::Notice(format!(
            "Stopped after {} tool-execution rounds without completing.",
            self.max_iterations
        )));
        Ok(TurnOutcome::MaxIterationsReached)
    }

    /// Execute a code-mode `python` call: permission-check it like any other
    /// tool, then run the script with the tool bridge attached.
    #[cfg(unix)]
    async fn execute_code_mode(
        &mut self,
        input: serde_json::Value,
        id: String,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> crate::tool::ToolCallResult {
        use crate::permissions::{PermissionDecision, ToolExecutionRequest};
        use crate::tool::ToolCallResult;

        let make_result = |content: String, outcome: ToolCallOutcome, id: String| ToolCallResult {
            block: ContentBlock::ToolResult {
                content,
                tool_use_id: id,
                is_error: (outcome != ToolCallOutcome::Success).then_some(true),
            },
            outcome,
        };

        let request = ToolExecutionRequest {
            tool_use_id: id.clone(),
            tool_name: "python".to_string(),
            input: input.clone(),
            tool_description: "Execute a Python script (may call other tools via the bridge)"
                .to_string(),
        };
        match self.registry.check_permission(&request).await {
            PermissionDecision::Allow => {}
            PermissionDecision::Deny => {
                return make_result(
                    "The user declined to run this script.".to_string(),
                    ToolCallOutcome::Denied,
                    id,
                )
            }
            PermissionDecision::DenyWithReason(reason) => {
                return make_result(
                    format!("The user declined to run this script: {}", reason),
                    ToolCallOutcome::Denied,
                    id,
                )
            }
        }

        let Some(code) = input.get("code").and_then(|v| v.as_str()) else {
            return make_result(
                "Missing 'code' field".to_string(),
                ToolCallOutcome::Failed,
                id,
            );
        };
        let timeout_secs = input
            .get("timeout_seconds")
            .and_then(|v| v.as_u64())
            .unwrap_or(crate::codemode::DEFAULT_TIMEOUT_SECS)
            .clamp(1, crate::codemode::MAX_TIMEOUT_SECS);

        let script =
            crate::codemode::run_script(code, timeout_secs, &mut self.registry, on_event).await;
        let outcome = if script.failed {
            ToolCallOutcome::Failed
        } else {
            ToolCallOutcome::Success
        };
        make_result(script.content, outcome, id)
    }

    #[cfg(not(unix))]
    async fn execute_code_mode(
        &mut self,
        _input: serde_json::Value,
        id: String,
        _on_event: &mut dyn FnMut(AgentEvent),
    ) -> crate::tool::ToolCallResult {
        crate::tool::ToolCallResult {
            block: ContentBlock::ToolResult {
                content: "Code mode is unavailable on this platform".to_string(),
                tool_use_id: id,
                is_error: Some(true),
            },
            outcome: ToolCallOutcome::Failed,
        }
    }

    /// Summarize older history into a single message, keeping recent turns
    /// verbatim. Returns `Ok(false)` when there is nothing safe to compact.
    pub async fn compact(&mut self, on_event: &mut dyn FnMut(AgentEvent)) -> Result<bool> {
        const COMPACTION_INSTRUCTION: &str =
            "Summarize the conversation above for continuation in a fresh context. Preserve: \
             the user's goals and constraints; key findings and decisions with their \
             rationale; exact file paths, function names, commands, URLs, and error messages \
             that may be needed again; the current state of in-progress work and what \
             remains. Dense plain prose and lists; no preamble.";

        let Some(cut) = self.compaction_cut() else {
            return Ok(false);
        };
        let mut to_summarize: Vec<Message> = self.history[..cut].to_vec();
        to_summarize.push(Message::user_text(COMPACTION_INSTRUCTION));
        let request = CompletionRequest {
            system: Some("You produce faithful, dense summaries of agent conversations."),
            messages: &to_summarize,
            tools: &[],
            max_tokens: 2_000,
        };
        let response = self.provider.complete(request).await?;
        let summary = response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if summary.is_empty() {
            return Err(crate::error::Error::Other(
                "empty compaction summary".to_string(),
            ));
        }
        let replaced = cut;
        self.history.splice(
            0..cut,
            [Message::user_text(format!(
                "[Context summary — {} earlier messages were compacted]\n{}",
                replaced, summary
            ))],
        );
        self.last_context_tokens = None;
        on_event(AgentEvent::Notice(format!(
            "Compacted {} messages into a summary (context ~{}k tokens).",
            replaced,
            estimate_tokens(&self.history) / 1000,
        )));
        Ok(true)
    }

    /// Index of the first message to keep verbatim. Everything before it is
    /// summarized. The boundary is a plain user turn (no tool results), so
    /// tool_use/tool_result pairs are never split.
    fn compaction_cut(&self) -> Option<usize> {
        let mut acc: u64 = 0;
        let mut cut = None;
        for (i, message) in self.history.iter().enumerate().rev() {
            acc += estimate_tokens(std::slice::from_ref(message));
            if acc >= self.compaction_keep_recent_tokens {
                cut = Some(i);
                break;
            }
        }
        let mut cut = cut?;
        while cut > 0 {
            let message = &self.history[cut];
            let plain_user = message.role == "user"
                && !message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolResult { .. }));
            if plain_user {
                break;
            }
            cut -= 1;
        }
        // Need at least two messages ahead of the boundary for a summary to
        // be worth a model call.
        (cut >= 2).then_some(cut)
    }

    async fn complete_with_retry(
        &self,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> Result<(crate::types::CompletionResponse, bool)> {
        #[allow(unused_mut)]
        let mut tools = self.registry.get_tool_defs();
        #[cfg(unix)]
        if self.code_mode && !self.registry.has_tool("python") {
            let code_only = self.registry.code_only_tool_defs();
            tools.push(crate::codemode::python_tool_def(&tools, &code_only));
        }
        let mut attempt: u32 = 0;
        loop {
            on_event(AgentEvent::ApiCallStarted);
            let request = CompletionRequest {
                system: Some(&self.system_prompt),
                messages: &self.history,
                tools: &tools,
                max_tokens: self.max_tokens,
            };
            let mut streamed = false;
            let result = {
                let mut forward = |text: String| {
                    streamed = true;
                    on_event(AgentEvent::AssistantTextDelta(text));
                };
                self.provider
                    .complete_streaming(request, &mut forward)
                    .await
            };
            match result {
                Ok(response) => {
                    on_event(AgentEvent::ApiCallFinished {
                        usage: response.usage.clone(),
                    });
                    return Ok((response, streamed));
                }
                Err(e) if e.is_retryable() && attempt < self.max_retries => {
                    on_event(AgentEvent::ApiCallFinished { usage: None });
                    let delay_secs = 1u64 << attempt; // 1, 2, 4 ...
                    on_event(AgentEvent::Retrying {
                        attempt: attempt + 1,
                        max_retries: self.max_retries,
                        delay_secs,
                        error: e.to_string(),
                    });
                    tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                    attempt += 1;
                }
                Err(e) => {
                    on_event(AgentEvent::ApiCallFinished { usage: None });
                    return Err(e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::provider::Provider;
    use crate::tool::Tool;
    use crate::types::CompletionResponse;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

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
        Agent::new(Box::new(script), registry, "test")
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

    /// End-to-end code mode: the model "writes" a script that calls the echo
    /// tool through the generated `tools` module; the bridged result must
    /// reach the script (not the model context) and the script's stdout must
    /// become the tool result. Requires python3 on PATH.
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

    #[cfg(unix)]
    #[tokio::test]
    async fn code_mode_bridges_tool_calls_into_scripts() {
        let code = r#"
import tools
result = tools.mirror(marker="xyzzy")
print("BRIDGED:", result)
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
            bridged_calls, 1,
            "no bridged call; script output: {}",
            result_text
        );
        assert!(
            result_text.contains("BRIDGED:"),
            "missing bridge output: {}",
            result_text
        );
        assert!(result_text.contains("xyzzy"));
        assert!(
            result_text.contains("RAISED OK"),
            "bad tool name must raise: {}",
            result_text
        );
        // ...and the bridged echo result appears nowhere else in history
        // (it reached the model only because the script chose to print it).
        assert_eq!(agent.history.len(), 4);
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
        let outcome = agent
            .run_turn("latest question", &mut |_| {})
            .await
            .unwrap();
        assert_eq!(outcome, TurnOutcome::Completed);
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
                on_delta: &mut dyn FnMut(String),
            ) -> Result<CompletionResponse> {
                on_delta("hel".to_string());
                on_delta("lo".to_string());
                Ok(text_response("hello"))
            }
        }

        let mut agent = Agent::new(Box::new(Streamy), ToolRegistry::new(), "test");
        let mut deltas = String::new();
        let mut full_blocks = 0;
        agent
            .run_turn("hi", &mut |event| match event {
                AgentEvent::AssistantTextDelta(t) => deltas.push_str(&t),
                AgentEvent::AssistantText(_) => full_blocks += 1,
                _ => {}
            })
            .await
            .unwrap();
        assert_eq!(deltas, "hello");
        assert_eq!(full_blocks, 0, "streamed text must not be re-emitted whole");
    }

    #[tokio::test]
    async fn non_retryable_errors_surface_immediately() {
        let mut agent = agent_with(Script::new(vec![Err(Error::Api {
            status: 401,
            message: "bad key".into(),
        })]));
        let result = agent.run_turn("hi", &mut |_| {}).await;
        assert!(result.is_err());
        // User message is preserved for the next attempt.
        assert_eq!(agent.history.len(), 1);
    }
}
