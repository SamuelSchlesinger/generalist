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

use crate::goal::update_goal_tool_def;
use crate::model_trace::{AsyncModelAction, ModelTrace};
use crate::provider::Provider;
use crate::runtime::{PromptSource, QueuedPrompt};
use crate::tool::ToolRegistry;
use crate::types::{estimate_tokens, CompletionLimits, Message};
use std::borrow::Cow;

mod compaction;
mod completion;
mod dispatch;
mod events;
mod protocol;
mod turn;

#[cfg(test)]
mod tests;

pub use events::{AgentEvent, TurnOutcome};
pub use protocol::history_tool_protocol_is_valid;

#[cfg(test)]
pub(crate) use completion::retry_delay_secs;

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
    /// Optional provider output-token request. `None` delegates to the
    /// adapter/provider where its protocol permits.
    pub max_tokens: Option<u32>,
    /// Host-authoritative response limits, independent of provider token
    /// accounting or compliance with `max_tokens`.
    pub completion_limits: CompletionLimits,
    /// Cap (in characters) on a single tool result as stored in history.
    pub max_tool_result_chars: usize,
    /// Retries for transient API errors, with exponential backoff.
    pub max_retries: u32,
    /// Code mode: advertise `python` as the only capability tool. Its scripts
    /// can call every registered tool via a generated `tools` module, keeping
    /// intermediate tool results out of the model's context and allowing one
    /// model round-trip to orchestrate many tool calls. An active goal also
    /// advertises the host-owned `update_goal` control tool.
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
    model_trace: Option<ModelTrace>,
    model_cancel_recorded: std::cell::Cell<bool>,
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
            max_tokens: None,
            completion_limits: CompletionLimits::default(),
            max_tool_result_chars: 40_000,
            max_retries: 3,
            code_mode: true,
            compaction_threshold_tokens: 150_000,
            compaction_keep_recent_tokens: 20_000,
            last_context_tokens: None,
            history_revision: 0,
            estimated_tokens_cache: std::cell::Cell::new(None),
            model_trace: None,
            model_cancel_recorded: std::cell::Cell::new(false),
        }
    }

    #[doc(hidden)]
    pub fn set_model_trace(&mut self, model_trace: ModelTrace) {
        self.registry.set_model_trace(model_trace.clone());
        self.model_trace = Some(model_trace);
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

    fn trace_async(&self, action: AsyncModelAction) {
        if let Some(trace) = &self.model_trace {
            trace.record_async(action);
        }
    }

    fn traced_outcome(&self, outcome: TurnOutcome) -> TurnOutcome {
        if outcome == TurnOutcome::Interrupted {
            self.trace_cancel_request();
            self.trace_async(AsyncModelAction::FinishCancellation);
        } else {
            self.trace_async(AsyncModelAction::SettleTurn);
        }
        outcome
    }

    fn trace_cancel_request(&self) {
        if !self.model_cancel_recorded.replace(true) {
            self.trace_async(AsyncModelAction::RequestCancel);
        }
    }

    /// Whether the built-in code-mode runner owns the model-facing tool
    /// interface. A registered tool named `python` always wins, so library
    /// users can override the built-in behavior.
    fn builtin_code_mode_enabled(&self) -> bool {
        self.code_mode && !self.registry.has_tool("python")
    }

    /// Whether the built-in Python runner currently owns the model-facing
    /// capability boundary.
    ///
    /// This is useful to host UIs that describe the effective tool surface:
    /// a library user can disable code mode or override the reserved runner by
    /// registering a custom tool named `python`.
    pub fn uses_builtin_code_mode(&self) -> bool {
        self.builtin_code_mode_enabled()
    }

    /// Tool definitions sent to the provider. In code mode, ordinary
    /// capability schemas are folded into the python tool's description.
    /// `update_goal` remains a native host control so it cannot be hidden
    /// inside, or permission-gated by, a code-mode script.
    fn model_tool_defs(&self) -> Vec<crate::types::ToolDef> {
        let mut definitions = if self.builtin_code_mode_enabled() {
            let available = self.registry.get_tool_defs();
            let code_only = self.registry.code_only_tool_defs();
            vec![crate::codemode::python_tool_def(&available, &code_only)]
        } else {
            self.registry.get_tool_defs()
        };

        if self.goal.is_some() {
            definitions.push(update_goal_tool_def());
        }
        definitions
    }

    /// Exact tool definitions that the next provider request will advertise.
    ///
    /// In built-in code mode this is the synthetic `python` runner plus the
    /// host-owned `update_goal` control when a goal is active. Registered
    /// bridge capabilities remain inspectable through [`ToolRegistry`].
    pub fn advertised_tool_defs(&self) -> Vec<crate::types::ToolDef> {
        self.model_tool_defs()
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
             unless the user changes or clears the goal. Preserve its full scope rather than \
             substituting a smaller task. An ordinary final response does not finish the goal: \
             the host will prompt again while it remains active. When, and only when, the full \
             objective is achieved and verified against the actual current state, call \
             `update_goal` with status `complete`. Do not call it merely because this turn is \
             ending or because partial progress was made.",
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

    /// Record a claimed queued prompt while preserving host provenance.
    pub fn begin_queued_turn(&mut self, prompt: &QueuedPrompt) {
        if prompt.source == PromptSource::GoalContinuation {
            debug_assert!(crate::goal::is_goal_continuation_prompt(&prompt.text));
            self.history.push(Message::goal_continuation());
        } else {
            self.begin_turn(&prompt.text);
        }
    }
}
