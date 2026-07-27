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
use crate::runtime::{QueuedPrompt, TurnControl};
use crate::tool::{ToolCallOutcome, ToolCallResult, ToolRegistry};
use crate::types::{
    estimate_tokens, truncate_middle, CompletionDelta, CompletionRequest, ContentBlock, Message,
    StopReason, Usage,
};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashSet;
use std::time::Duration;

fn cancelled_tool_result(tool_use_id: String) -> ContentBlock {
    ContentBlock::ToolResult {
        content: "This tool call was cancelled before completion. Its side effects, if any, \
                  may be incomplete; inspect the relevant state before retrying."
            .to_string(),
        tool_use_id,
        is_error: Some(true),
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct StreamedKinds {
    text: bool,
    reasoning: bool,
}

fn emit_stream_aborted(
    on_event: &mut dyn FnMut(AgentEvent),
    streamed: StreamedKinds,
    reason: String,
) {
    if streamed.text {
        on_event(AgentEvent::AssistantStreamAborted {
            reason: reason.clone(),
        });
    }
    if streamed.reasoning {
        on_event(AgentEvent::ReasoningStreamAborted { reason });
    }
}

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
    /// A provider-supplied reasoning fragment. This is inspectable UI data,
    /// not assistant-visible conversation text.
    ReasoningDelta(String),
    /// A provider attempt emitted visible deltas but failed or was cancelled
    /// before a complete response could enter conversation history.
    AssistantStreamAborted { reason: String },
    /// A provider attempt emitted reasoning but failed or was cancelled before
    /// that reasoning could enter committed history.
    ReasoningStreamAborted { reason: String },
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
    /// Steering prompts were committed to history at a safe boundary.
    SteeringCommitted { prompts: Vec<QueuedPrompt> },
    /// A history-valid boundary suitable for durable autosave. This is never
    /// emitted between an assistant tool use and its user tool result.
    HistoryCheckpoint {
        history: Vec<Message>,
        context_tokens: u64,
    },
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
    /// The controller interrupted the turn after repairing any unfinished
    /// tool-use/result pairs.
    Interrupted,
}

pub struct Agent {
    provider: Box<dyn Provider>,
    pub registry: ToolRegistry,
    /// Base instructions shared by every provider request.
    pub system_prompt: String,
    /// Durable user-authored objective appended to the base instructions.
    goal: Option<String>,
    /// Conversation history. Exposed read-only through [`Agent::history`] so
    /// token-accounting caches cannot be bypassed by in-place mutations.
    history: Vec<Message>,
    /// Cap on request → tool → request rounds within a single turn.
    pub max_iterations: usize,
    /// `max_tokens` per completion.
    pub max_tokens: u32,
    /// Cap (in characters) on a single tool result as stored in history.
    pub max_tool_result_chars: usize,
    /// Retries for transient API errors, with exponential backoff.
    pub max_retries: u32,
    /// Code mode: advertise only a `python` tool. Its scripts can
    /// call every registered tool via a generated `tools` module, keeping
    /// intermediate tool results out of the model's context and allowing one
    /// model round-trip to orchestrate many tool calls.
    pub code_mode: bool,
    /// Compact (summarize) older history when the context reaches this many
    /// tokens. `u64::MAX` disables compaction.
    pub compaction_threshold_tokens: u64,
    /// How much recent history (in estimated tokens) stays verbatim when
    /// compacting.
    pub compaction_keep_recent_tokens: u64,
    /// Context size measured by the provider on the last completion.
    last_context_tokens: Option<u64>,
    /// Changes only when existing message indices are invalidated.
    history_revision: u64,
    /// Memoized `estimate_tokens(&history)`: valid while the history has the
    /// same revision and length. Appends change the length; the one internal
    /// index-preserving mutation explicitly clears this cache.
    estimated_tokens_cache: std::cell::Cell<Option<(u64, usize, u64)>>,
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
            goal: None,
            history: Vec::new(),
            max_iterations: 100,
            max_tokens: 16_000,
            max_tool_result_chars: 40_000,
            max_retries: 3,
            code_mode: true,
            compaction_threshold_tokens: 150_000,
            compaction_keep_recent_tokens: 20_000,
            last_context_tokens: None,
            history_revision: 0,
            estimated_tokens_cache: std::cell::Cell::new(None),
        }
    }

    /// Rough context estimate: provider-measured when available, chars/4
    /// over the serialized history otherwise.
    pub fn context_tokens(&self) -> u64 {
        if let Some(measured) = self.last_context_tokens {
            return measured;
        }
        let key = (self.history_revision, self.history.len());
        if let Some((revision, len, estimate)) = self.estimated_tokens_cache.get() {
            if (revision, len) == key {
                return estimate;
            }
        }
        let estimate = estimate_tokens(&self.history);
        self.estimated_tokens_cache
            .set(Some((key.0, key.1, estimate)));
        estimate
    }

    fn invalidate_estimated_tokens_cache(&self) {
        self.estimated_tokens_cache.set(None);
    }

    /// Whether the built-in code-mode runner owns the model-facing tool
    /// interface. A registered tool named `python` always wins, so library
    /// users can override the built-in behavior.
    fn builtin_code_mode_enabled(&self) -> bool {
        self.code_mode && !self.registry.has_tool("python")
    }

    fn is_code_mode_call(&self, name: &str) -> bool {
        self.builtin_code_mode_enabled() && name == "python"
    }

    /// Tool definitions sent to the provider. In code mode, ordinary tool
    /// schemas are folded into the python tool's description rather than
    /// advertised as independently callable tools.
    fn model_tool_defs(&self) -> Vec<crate::types::ToolDef> {
        if self.builtin_code_mode_enabled() {
            let available = self.registry.get_tool_defs();
            let code_only = self.registry.code_only_tool_defs();
            return vec![crate::codemode::python_tool_def(&available, &code_only)];
        }

        self.registry.get_tool_defs()
    }

    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    pub fn set_provider(&mut self, provider: Box<dyn Provider>) {
        self.provider = provider;
    }

    /// The complete conversation history.
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// The active session objective, if one has been set with the host UI.
    pub fn goal(&self) -> Option<&str> {
        self.goal.as_deref()
    }

    /// Replace or clear the active session objective.
    ///
    /// Goals are instruction context rather than conversation messages, so
    /// changing one does not rewrite history. Provider token accounting is
    /// invalidated because the next request has a different system prompt.
    pub fn set_goal(&mut self, goal: Option<String>) {
        let goal = goal.and_then(|goal| {
            let trimmed = goal.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        });
        if self.goal != goal {
            self.goal = goal;
            self.last_context_tokens = None;
        }
    }

    fn effective_system_prompt(&self) -> Cow<'_, str> {
        let Some(goal) = self.goal() else {
            return Cow::Borrowed(&self.system_prompt);
        };
        Cow::Owned(format!(
            "{}\n\n## Active session goal\n\n{}\n\n\
             Treat this as durable user intent. Keep making concrete progress toward it \
             unless the user changes or clears the goal. Verify completion against the \
             actual workspace state before claiming it is done.",
            self.system_prompt, goal
        ))
    }

    /// Replace persisted conversation history and discard the provider's
    /// measurement of the previous history.
    pub fn replace_history(&mut self, history: Vec<Message>) {
        self.history = history;
        self.last_context_tokens = None;
        self.history_revision = self.history_revision.wrapping_add(1);
        self.invalidate_estimated_tokens_cache();
    }

    /// Clear conversation history and its cached context measurement.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.last_context_tokens = None;
        self.history_revision = self.history_revision.wrapping_add(1);
        self.invalidate_estimated_tokens_cache();
    }

    /// Revision for consumers that retain an index into conversation history.
    ///
    /// Appends keep indices stable. Replacement, clearing, and compaction
    /// increment this value.
    pub fn history_revision(&self) -> u64 {
        self.history_revision
    }

    /// Record the initial user message without crossing an await boundary.
    ///
    /// The TUI uses this to commit a two-phase queue claim only after the
    /// conversation owns the prompt.
    pub fn begin_turn(&mut self, user_input: &str) {
        self.history.push(Message::user_text(user_input));
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
        self.begin_turn(user_input);
        let mut control = TurnControl::detached();
        self.run_started_turn(on_event, &mut control).await
    }

    /// Continue a turn whose initial user message has already been recorded.
    ///
    /// The asynchronous TUI supplies a real [`TurnControl`] so terminal input,
    /// steering, and cooperative cancellation keep progressing while this
    /// future is pending. Library callers normally use [`Agent::run_turn`].
    pub async fn run_started_turn(
        &mut self,
        on_event: &mut dyn FnMut(AgentEvent),
        control: &mut TurnControl,
    ) -> Result<TurnOutcome> {
        for iteration in 0..self.max_iterations {
            if control.is_cancelled() {
                return Ok(TurnOutcome::Interrupted);
            }

            if self.context_tokens() > self.compaction_threshold_tokens {
                let compacted = {
                    let compact = self.compact(on_event);
                    tokio::pin!(compact);
                    tokio::select! {
                        result = &mut compact => Some(result),
                        _ = control.cancelled() => None,
                    }
                };
                match compacted {
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        on_event(AgentEvent::Notice(format!("Compaction failed: {error}")));
                    }
                    None => return Ok(TurnOutcome::Interrupted),
                }
            }

            let Some((response, streamed)) = self.complete_with_retry(on_event, control).await?
            else {
                return Ok(TurnOutcome::Interrupted);
            };
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

            if !streamed.reasoning {
                for block in &response.content {
                    if let ContentBlock::Thinking { thinking, .. } = block {
                        if !thinking.is_empty() {
                            on_event(AgentEvent::ReasoningDelta(thinking.clone()));
                        }
                    }
                }
            }

            if !streamed.text {
                for block in &response.content {
                    if let ContentBlock::Text { text } = block {
                        if !text.is_empty() {
                            on_event(AgentEvent::AssistantText(text.clone()));
                        }
                    }
                }
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

            if response.stop_reason == StopReason::Refusal {
                if !tool_uses.is_empty() {
                    let results = tool_uses
                        .iter()
                        .map(|(_, _, id)| ContentBlock::ToolResult {
                            content: "This tool call was not executed because the model refused \
                                      the request."
                                .to_string(),
                            tool_use_id: id.clone(),
                            is_error: Some(true),
                        })
                        .collect();
                    self.history.push(Message::user(results));
                }
                on_event(AgentEvent::Notice(
                    "The model declined to continue with this request.".to_string(),
                ));
                self.emit_checkpoint(on_event);
                return Ok(TurnOutcome::Refused);
            }

            if tool_uses.is_empty() {
                if control.is_cancelled() {
                    self.emit_checkpoint(on_event);
                    return Ok(TurnOutcome::Interrupted);
                }
                if iteration + 1 < self.max_iterations && self.commit_steering(control, on_event) {
                    continue;
                }
                if response.stop_reason == StopReason::MaxTokens {
                    on_event(AgentEvent::Notice(
                        "Response was cut off at the output-token limit and may be incomplete."
                            .to_string(),
                    ));
                }
                self.emit_checkpoint(on_event);
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
                let steered =
                    iteration + 1 < self.max_iterations && self.commit_steering(control, on_event);
                if !steered {
                    self.emit_checkpoint(on_event);
                }
                continue;
            }

            let mut results = Vec::with_capacity(tool_uses.len());
            let mut denied = false;
            let mut cancelled = false;
            let mut index = 0;
            while index < tool_uses.len() {
                if control.is_cancelled() {
                    for (_, _, id) in &tool_uses[index..] {
                        results.push(cancelled_tool_result(id.clone()));
                    }
                    cancelled = true;
                    break;
                }

                let (name, input, id) = tool_uses[index].clone();
                if self.builtin_code_mode_enabled() && !self.is_code_mode_call(&name) {
                    // A compatible server/model may still emit a function name
                    // that was never advertised (some models copy a
                    // `tools.foo(...)` expression out of the prompt). Treat
                    // that as a provider-protocol violation, never as tool
                    // activity: pair it for history validity, explain the
                    // required boundary, and let the next model round retry.
                    let bridge_name = name.strip_prefix("tools.").unwrap_or(&name);
                    let content = format!(
                        "The provider emitted undeclared native tool call `{name}`. It was not \
                         executed: code mode permits only the model-facing `python` tool. Retry \
                         with a Python script using `import tools; tools.{bridge_name}(...)`."
                    );
                    on_event(AgentEvent::Notice(format!(
                        "Rejected undeclared native tool call `{name}` before execution; asking \
                         the model to retry through code mode."
                    )));
                    results.push(ContentBlock::ToolResult {
                        content,
                        tool_use_id: id,
                        is_error: Some(true),
                    });
                    index += 1;
                    continue;
                }

                on_event(AgentEvent::ToolCallStarted {
                    name: name.clone(),
                    input: input.clone(),
                });

                let cancellation_id = id.clone();
                let maybe_result = {
                    let execution = async {
                        if self.is_code_mode_call(&name) {
                            self.execute_code_mode(input, id, on_event).await
                        } else {
                            self.registry.execute_tool(&name, input, id).await
                        }
                    };
                    tokio::pin!(execution);
                    tokio::select! {
                        result = &mut execution => Some(result),
                        _ = control.cancelled() => None,
                    }
                };

                let Some(mut result) = maybe_result else {
                    let content =
                        "Tool execution was interrupted; its completion is unknown.".to_string();
                    on_event(AgentEvent::ToolCallFinished {
                        name,
                        outcome: ToolCallOutcome::Cancelled,
                        content,
                    });
                    results.push(cancelled_tool_result(cancellation_id));
                    for (_, _, id) in &tool_uses[index + 1..] {
                        results.push(cancelled_tool_result(id.clone()));
                    }
                    cancelled = true;
                    break;
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
                index += 1;
            }

            // Every tool_use gets a result — required by the APIs even when
            // a call was denied or cancellation interrupts the batch.
            self.history.push(Message::user(results));

            // The controller may have processed an interrupt while the final
            // tool or permission future became ready. Do not steer merely
            // because that future won the inner select race.
            cancelled |= control.is_cancelled();
            if cancelled {
                on_event(AgentEvent::Notice(
                    "Turn interrupted; unfinished tool calls were recorded as cancelled."
                        .to_string(),
                ));
                self.emit_checkpoint(on_event);
                return Ok(TurnOutcome::Interrupted);
            }

            if iteration + 1 < self.max_iterations && self.commit_steering(control, on_event) {
                continue;
            }

            if denied {
                on_event(AgentEvent::Notice(
                    "Tool call denied — pausing so you can redirect.".to_string(),
                ));
                self.emit_checkpoint(on_event);
                return Ok(TurnOutcome::PausedOnDenial);
            }
            self.emit_checkpoint(on_event);
        }

        on_event(AgentEvent::Notice(format!(
            "Stopped after {} tool-execution rounds without completing.",
            self.max_iterations
        )));
        Ok(TurnOutcome::MaxIterationsReached)
    }

    /// Commit every queued steer at a history-valid boundary. There is no
    /// await between claiming the stable IDs, updating history, and committing
    /// the claim, so cancellation cannot strand a prompt mid-transition.
    fn commit_steering(
        &mut self,
        control: &TurnControl,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> bool {
        let Some(claim) = control.claim_steering() else {
            return false;
        };
        let prompts = claim.prompts().to_vec();
        let text_blocks = prompts
            .iter()
            .map(|prompt| ContentBlock::Text {
                text: prompt.text.clone(),
            })
            .collect::<Vec<_>>();
        if let Some(last) = self
            .history
            .last_mut()
            .filter(|message| message.role == "user")
        {
            last.content.extend(text_blocks);
        } else {
            self.history.push(Message::user(text_blocks));
        }
        self.invalidate_estimated_tokens_cache();
        let prompts = claim.commit();
        on_event(AgentEvent::SteeringCommitted { prompts });
        self.emit_checkpoint(on_event);
        true
    }

    fn emit_checkpoint(&self, on_event: &mut dyn FnMut(AgentEvent)) {
        debug_assert!(
            history_tool_protocol_is_valid(&self.history),
            "attempted to checkpoint history with an unpaired tool use/result"
        );
        on_event(AgentEvent::HistoryCheckpoint {
            history: self.history.clone(),
            context_tokens: self.context_tokens(),
        });
    }

    /// Execute a code-mode `python` call: permission-check it like any other
    /// tool, then run the script with the tool bridge attached.
    async fn execute_code_mode(
        &mut self,
        input: serde_json::Value,
        id: String,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> crate::tool::ToolCallResult {
        use crate::permissions::{PermissionDecision, ToolExecutionRequest};
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
        let outcome = if script.denied {
            ToolCallOutcome::Denied
        } else if script.failed {
            ToolCallOutcome::Failed
        } else {
            ToolCallOutcome::Success
        };
        make_result(script.content, outcome, id)
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
        self.history_revision = self.history_revision.wrapping_add(1);
        self.last_context_tokens = None;
        self.invalidate_estimated_tokens_cache();
        on_event(AgentEvent::Notice(format!(
            "Compacted {} messages into a summary (context ~{}k tokens).",
            replaced,
            estimate_tokens(&self.history) / 1000,
        )));
        self.emit_checkpoint(on_event);
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
        control: &mut TurnControl,
    ) -> Result<Option<(crate::types::CompletionResponse, StreamedKinds)>> {
        let tools = self.model_tool_defs();
        let system_prompt = self.effective_system_prompt();
        let mut attempt: u32 = 0;
        loop {
            on_event(AgentEvent::ApiCallStarted);
            let request = CompletionRequest {
                system: Some(system_prompt.as_ref()),
                messages: &self.history,
                tools: &tools,
                max_tokens: self.max_tokens,
            };
            let mut streamed = StreamedKinds::default();
            let maybe_result = {
                let mut forward = |delta: CompletionDelta| match delta {
                    CompletionDelta::Text(text) => {
                        streamed.text = true;
                        on_event(AgentEvent::AssistantTextDelta(text));
                    }
                    CompletionDelta::Reasoning(reasoning) => {
                        streamed.reasoning = true;
                        on_event(AgentEvent::ReasoningDelta(reasoning));
                    }
                };
                let completion = self.provider.complete_streaming(request, &mut forward);
                tokio::pin!(completion);
                tokio::select! {
                    result = &mut completion => Some(result),
                    _ = control.cancelled() => None,
                }
            };
            let Some(result) = maybe_result else {
                emit_stream_aborted(
                    on_event,
                    streamed,
                    "interrupted before the response was committed".to_string(),
                );
                on_event(AgentEvent::ApiCallFinished { usage: None });
                return Ok(None);
            };
            match result {
                Ok(response) => {
                    on_event(AgentEvent::ApiCallFinished {
                        usage: response.usage.clone(),
                    });
                    return Ok(Some((response, streamed)));
                }
                Err(e) if e.is_retryable() && attempt < self.max_retries => {
                    emit_stream_aborted(
                        on_event,
                        streamed,
                        format!("stream failed before it was committed: {e}"),
                    );
                    on_event(AgentEvent::ApiCallFinished { usage: None });
                    let delay_secs = retry_delay_secs(attempt, e.retry_after());
                    on_event(AgentEvent::Retrying {
                        attempt: attempt + 1,
                        max_retries: self.max_retries,
                        delay_secs,
                        error: e.to_string(),
                    });
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(delay_secs)) => {}
                        _ = control.cancelled() => return Ok(None),
                    }
                    attempt += 1;
                }
                Err(e) => {
                    emit_stream_aborted(
                        on_event,
                        streamed,
                        format!("stream failed before it was committed: {e}"),
                    );
                    on_event(AgentEvent::ApiCallFinished { usage: None });
                    return Err(e);
                }
            }
        }
    }
}

/// Delay before the next retry: exponential backoff (1, 2, 4, ...) with the
/// server's `Retry-After` as a lower bound.
///
/// The header is a floor, not an override: rate limits often clear sooner
/// than its worst-case value, so we never wait *less* than it asks, but also
/// never longer than our own schedule would. The 60s cap keeps a turn
/// interruptible and stops a hostile or buggy header from parking the loop.
fn retry_delay_secs(attempt: u32, retry_after: Option<u64>) -> u64 {
    (1u64 << attempt.min(20))
        .max(retry_after.unwrap_or(0))
        .min(60)
}

/// Whether every assistant tool use has exactly one result in the immediately
/// following user message, with no orphan results or role inversions.
///
/// This is the executable counterpart of `ToolHistoryIsValid` in
/// `spec/AsyncRuntime.tla`.
pub fn history_tool_protocol_is_valid(history: &[Message]) -> bool {
    let mut expected_results: Option<HashSet<&str>> = None;

    for message in history {
        let tool_uses = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let tool_results = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        if !tool_uses.is_empty() && message.role != "assistant" {
            return false;
        }
        if !tool_results.is_empty() && message.role != "user" {
            return false;
        }

        if let Some(expected) = expected_results.take() {
            let actual = tool_results.iter().copied().collect::<HashSet<_>>();
            if message.role != "user" || actual.len() != tool_results.len() || actual != expected {
                return false;
            }
        } else if !tool_results.is_empty() {
            return false;
        }

        if !tool_uses.is_empty() {
            let expected = tool_uses.iter().copied().collect::<HashSet<_>>();
            if expected.len() != tool_uses.len() {
                return false;
            }
            expected_results = Some(expected);
        }
    }

    expected_results.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::permissions::{
        PermissionDecision, PolicyPermissions, ToolExecutionRequest, ToolPermissionHandler,
    };
    use crate::provider::Provider;
    use crate::runtime::{DeliveryMode, PromptQueue};
    use crate::tool::Tool;
    use crate::types::CompletionResponse;
    use async_trait::async_trait;
    use serde_json::json;
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

    #[tokio::test(flavor = "current_thread")]
    async fn cancellation_wins_over_a_ready_permission_before_steering() {
        LocalSet::new()
            .run_until(async {
                let started = Arc::new(Notify::new());
                let (answer, answer_rx) = oneshot::channel();
                let mut registry =
                    ToolRegistry::with_permission_handler(Box::new(GatedPermission {
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
            on_delta: &mut dyn FnMut(CompletionDelta),
        ) -> Result<CompletionResponse> {
            on_delta(CompletionDelta::Text("visible but uncommitted".to_string()));
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
        assert!(defs[0].description.contains("Input schema"));
        assert!(defs[0]
            .description
            .contains("never emit `<name>` or `tools.<name>` as a native tool call"));
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
import tools
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
                on_delta: &mut dyn FnMut(CompletionDelta),
            ) -> Result<CompletionResponse> {
                on_delta(CompletionDelta::Reasoning("because ".to_string()));
                on_delta(CompletionDelta::Reasoning("evidence".to_string()));
                on_delta(CompletionDelta::Text("hel".to_string()));
                on_delta(CompletionDelta::Text("lo".to_string()));
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
        agent
            .run_turn("hi", &mut |event| match event {
                AgentEvent::AssistantTextDelta(t) => deltas.push_str(&t),
                AgentEvent::ReasoningDelta(t) => reasoning.push_str(&t),
                AgentEvent::AssistantText(_) => full_blocks += 1,
                _ => {}
            })
            .await
            .unwrap();
        assert_eq!(deltas, "hello");
        assert_eq!(reasoning, "because evidence");
        assert_eq!(full_blocks, 0, "streamed text must not be re-emitted whole");
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
    async fn non_retryable_errors_surface_immediately() {
        let mut agent = agent_with(Script::new(vec![Err(Error::Api {
            status: 401,
            message: "bad key".into(),
            retry_after: None,
        })]));
        let result = agent.run_turn("hi", &mut |_| {}).await;
        assert!(result.is_err());
        // User message is preserved for the next attempt.
        assert_eq!(agent.history.len(), 1);
    }
}
