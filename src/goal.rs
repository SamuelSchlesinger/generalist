//! Host-owned active-goal protocol.
//!
//! Goal continuation is represented as an ordinary queued follow-up so it
//! inherits the runtime's stable-ID, persistence, and cancellation boundaries.
//! Completion is different from ordinary capability work: the model calls the
//! reserved `update_goal` control tool, which [`crate::Agent`] executes without
//! a permission prompt.

use crate::types::ToolDef;
use serde_json::json;

/// Reserved model-facing control tool used to finish an active goal.
pub const UPDATE_GOAL_TOOL_NAME: &str = "update_goal";

/// The host-authored follow-up inserted while a goal remains active.
pub const GOAL_CONTINUATION_PROMPT: &str = "\
<generalist_internal_context source=\"goal\">
Continue working toward the active session goal. Preserve its full scope and make concrete \
progress rather than merely reporting status. When, and only when, the complete goal has been \
achieved and verified against the actual current state, call update_goal with status \"complete\". \
If required work remains, leave the goal active; the host will continue prompting.
</generalist_internal_context>";

/// Whether text is the exact host-authored goal continuation prompt.
pub fn is_goal_continuation_prompt(text: &str) -> bool {
    text == GOAL_CONTINUATION_PROMPT
}

pub(crate) fn update_goal_tool_def() -> ToolDef {
    ToolDef {
        name: UPDATE_GOAL_TOOL_NAME.to_string(),
        description: "Mark the active session goal complete. Call this only after the full \
                      user objective has been achieved and verified against the actual current \
                      state. Do not call it merely because a turn is ending, progress was made, \
                      or a smaller substitute was completed."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete"],
                    "description": "The verified terminal status of the active goal"
                }
            },
            "required": ["status"],
            "additionalProperties": false
        }),
    }
}
