//! Progress events and turn outcomes surfaced by the agent loop.

use crate::runtime::QueuedPrompt;
use crate::tool::ToolCallOutcome;
use crate::types::{ContentBlock, Message, Usage};
use serde_json::Value;

pub(super) fn cancelled_tool_result(tool_use_id: String) -> ContentBlock {
    ContentBlock::ToolResult {
        content: "This tool call was cancelled before completion. Its side effects, if any, \
                  may be incomplete; inspect the relevant state before retrying."
            .to_string(),
        tool_use_id,
        is_error: Some(true),
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct StreamedKinds {
    pub(super) text: bool,
    pub(super) reasoning: bool,
}

pub(super) fn emit_stream_aborted(
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

/// Progress notifications emitted during [`Agent::run_turn`](super::Agent::run_turn).
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
    /// [`AgentEvent::StreamCommitted`] reconciles a successful bounded preview.
    AssistantTextDelta(String),
    /// A provider-supplied reasoning fragment. This is inspectable UI data,
    /// not assistant-visible conversation text.
    ReasoningDelta(String),
    /// The authoritative display contents for kinds that streamed during a
    /// successful provider attempt. Emitted only after the response entered
    /// conversation history, so a UI may replace a bounded live preview
    /// without mistaking an uncommitted stream for durable conversation.
    StreamCommitted {
        text: Option<String>,
        reasoning: Option<String>,
    },
    /// The live display pump omitted fragment bytes to keep its pending
    /// preview bounded. This is display-only: a successful attempt is later
    /// reconciled by [`AgentEvent::StreamCommitted`].
    StreamDisplayTruncated {
        text_bytes: usize,
        reasoning_bytes: usize,
    },
    /// A provider attempt emitted visible deltas but failed or was cancelled
    /// before a complete response could enter conversation history.
    AssistantStreamAborted { reason: String },
    /// A provider attempt emitted reasoning but failed or was cancelled before
    /// that reasoning could enter committed history.
    ReasoningStreamAborted { reason: String },
    /// A tool call is about to execute. Ordinary capability tools are checked
    /// for permission; host control tools are not. Emitted *before* execution
    /// so the user always sees the input first.
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
    /// Display notification that the model marked the active goal complete.
    /// Persistence changes only at the following atomic history checkpoint.
    GoalCompleted { goal: String },
    /// A history-valid boundary suitable for durable autosave. This is never
    /// emitted between an assistant tool use and its user tool result.
    HistoryCheckpoint {
        history: Vec<Message>,
        goal: Option<String>,
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
