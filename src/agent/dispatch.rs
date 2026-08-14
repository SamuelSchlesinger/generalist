//! Host-control tool execution: goal updates and code-mode scripts.

use super::{Agent, AgentEvent};
use crate::tool::{ToolCallOutcome, ToolCallResult};
use crate::types::ContentBlock;
use serde_json::Value;

impl Agent {
    pub(super) fn is_code_mode_call(&self, name: &str) -> bool {
        self.builtin_code_mode_enabled() && name == "python"
    }

    /// Apply the permission-free, host-owned goal completion control.
    ///
    /// This runs synchronously at the same safe tool-batch boundary as an
    /// ordinary tool result. The following provider request receives the
    /// paired result and can produce the final user-facing summary without the
    /// now-completed goal being injected again.
    pub(super) fn execute_goal_update(
        &mut self,
        input: Value,
        id: String,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> ToolCallResult {
        let result = |content: String, outcome: ToolCallOutcome| ToolCallResult {
            block: ContentBlock::ToolResult {
                content,
                tool_use_id: id,
                is_error: (outcome != ToolCallOutcome::Success).then_some(true),
            },
            outcome,
        };

        let valid_complete = input.as_object().is_some_and(|object| {
            object.len() == 1
                && object
                    .get("status")
                    .and_then(Value::as_str)
                    .is_some_and(|status| status == "complete")
        });
        if !valid_complete {
            return result(
                "Invalid update_goal input: expected exactly {\"status\":\"complete\"}."
                    .to_string(),
                ToolCallOutcome::Failed,
            );
        }

        let Some(goal) = self.goal.clone() else {
            return result(
                "No active goal exists; nothing was changed.".to_string(),
                ToolCallOutcome::Failed,
            );
        };
        self.set_goal(None);
        on_event(AgentEvent::GoalCompleted { goal });
        result(
            "The active goal is complete. Give the user a concise final result.".to_string(),
            ToolCallOutcome::Success,
        )
    }

    /// Execute a code-mode `python` call: validate its input, permission-check
    /// it like any other tool, then run the script with the tool bridge
    /// attached.
    pub(super) async fn execute_code_mode(
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

        let Some(code) = input.get("code").and_then(Value::as_str).map(str::to_owned) else {
            return make_result(
                "Invalid python tool input: expected an object with a string `code` field. \
                 Retry the `python` tool call with JSON arguments like \
                 {\"code\":\"<Python source>\"}."
                    .to_string(),
                ToolCallOutcome::Failed,
                id,
            );
        };
        let timeout_secs = input
            .get("timeout_seconds")
            .and_then(Value::as_u64)
            .unwrap_or(crate::codemode::DEFAULT_TIMEOUT_SECS)
            .clamp(1, crate::codemode::MAX_TIMEOUT_SECS);

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

        let script =
            crate::codemode::run_script(&code, timeout_secs, &mut self.registry, on_event).await;
        let outcome = if script.denied {
            ToolCallOutcome::Denied
        } else if script.failed {
            ToolCallOutcome::Failed
        } else {
            ToolCallOutcome::Success
        };
        make_result(script.content, outcome, id)
    }
}
