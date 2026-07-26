//! Tool trait and registry.

use crate::error::Result;
use crate::execution::{ExecutionState, ToolExecution};
use crate::permissions::{
    AlwaysAllowPermissions, PermissionDecision, ToolExecutionRequest, ToolPermissionHandler,
};
use crate::types::{ContentBlock, ToolDef};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// A capability the model can invoke during a conversation.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique tool name (what the model calls).
    fn name(&self) -> &str;

    /// Description shown to the model. Say *when* to use the tool, not just
    /// what it does — models trigger tools from this text.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input.
    fn input_schema(&self) -> Value;

    /// Execute with validated-by-the-model input; return output text.
    async fn execute(&self, input: Value) -> Result<String>;

    /// Progressive-disclosure tools are excluded from the direct tool list
    /// but remain callable from code-mode scripts. In built-in code mode all
    /// tools are script-only; this flag additionally keeps the full schema
    /// out of the `python` tool description and leaves it in the generated
    /// `tools` module's docstring instead. MCP tools use this to keep heavy
    /// schemas out of the context.
    fn code_only(&self) -> bool {
        false
    }

    fn to_tool_def(&self) -> ToolDef {
        ToolDef {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

/// How a tool call ended. This is a structured signal — never inferred from
/// the result text — so callers can distinguish a permission denial from an
/// ordinary tool failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallOutcome {
    Success,
    Failed,
    Denied,
    Cancelled,
}

/// The result block to feed back to the model, plus what actually happened.
#[derive(Debug)]
pub struct ToolCallResult {
    pub block: ContentBlock,
    pub outcome: ToolCallOutcome,
}

impl ToolCallResult {
    fn new(tool_use_id: String, content: String, outcome: ToolCallOutcome) -> Self {
        let is_error = match outcome {
            ToolCallOutcome::Success => None,
            _ => Some(true),
        };
        Self {
            block: ContentBlock::ToolResult {
                content,
                tool_use_id,
                is_error,
            },
            outcome,
        }
    }
}

/// Holds the available tools, runs them behind a permission handler, and
/// records execution history.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    order: Vec<String>,
    executions: Vec<ToolExecution>,
    permission_handler: Box<dyn ToolPermissionHandler>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::with_permission_handler(Box::new(AlwaysAllowPermissions))
    }

    pub fn with_permission_handler(handler: Box<dyn ToolPermissionHandler>) -> Self {
        Self {
            tools: HashMap::new(),
            order: Vec::new(),
            executions: Vec::new(),
            permission_handler: handler,
        }
    }

    pub fn set_permission_handler(&mut self, handler: Box<dyn ToolPermissionHandler>) {
        self.permission_handler = handler;
    }

    /// Run a request through this registry's permission handler without
    /// executing anything. Used for agent-level tools (e.g. code mode) that
    /// are not in the registry but must obey the same policy.
    pub async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        self.permission_handler.check_permission(request).await
    }

    /// Register a tool. Errors if the name is already taken.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(crate::error::Error::Other(format!(
                "Tool '{}' already registered",
                name
            )));
        }
        self.order.push(name.clone());
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Direct tool definitions in registration order, excluding code-only
    /// tools. When built-in code mode is active, these definitions are folded
    /// into the sole model-facing `python` tool instead of being independently
    /// callable.
    ///
    /// The order is deterministic on purpose: providers that cache the prompt
    /// prefix would otherwise miss the cache whenever a HashMap iteration
    /// order changed.
    pub fn get_tool_defs(&self) -> Vec<ToolDef> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .filter(|tool| !tool.code_only())
            .map(|tool| tool.to_tool_def())
            .collect()
    }

    /// Every tool, including code-only ones — the set exposed to code-mode
    /// scripts through the generated `tools` module.
    pub fn get_bridge_tool_defs(&self) -> Vec<ToolDef> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| tool.to_tool_def())
            .collect()
    }

    /// Definitions of progressive-disclosure tools, used to list their names
    /// without folding their full schemas into the `python` description.
    pub fn code_only_tool_defs(&self) -> Vec<ToolDef> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .filter(|tool| tool.code_only())
            .map(|tool| tool.to_tool_def())
            .collect()
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.order.clone()
    }

    /// Execute a tool call. Infallible by design: every failure mode becomes
    /// a `ToolResult` block the model can react to, with the real outcome
    /// reported alongside.
    pub async fn execute_tool(
        &mut self,
        tool_name: &str,
        input: Value,
        tool_use_id: String,
    ) -> ToolCallResult {
        let Some(tool) = self.tools.get(tool_name).cloned() else {
            return ToolCallResult::new(
                tool_use_id,
                format!("Tool '{}' not found", tool_name),
                ToolCallOutcome::Failed,
            );
        };

        let mut execution =
            ToolExecution::new(tool_use_id.clone(), tool_name.to_string(), input.clone());

        let request = ToolExecutionRequest {
            tool_use_id: tool_use_id.clone(),
            tool_name: tool_name.to_string(),
            input: input.clone(),
            tool_description: tool.description().to_string(),
        };

        match self.permission_handler.check_permission(&request).await {
            PermissionDecision::Allow => {
                execution.state = ExecutionState::Executing;
                let result = tool.execute(input).await;
                let call_result = match result {
                    Ok(output) => {
                        execution.complete(Ok(output.clone()));
                        ToolCallResult::new(tool_use_id, output, ToolCallOutcome::Success)
                    }
                    Err(e) => {
                        let message = e.to_string();
                        execution.complete(Err(message.clone()));
                        ToolCallResult::new(
                            tool_use_id,
                            format!("Tool execution failed: {}", message),
                            ToolCallOutcome::Failed,
                        )
                    }
                };
                self.executions.push(execution);
                call_result
            }
            PermissionDecision::Deny => {
                execution.deny("Permission denied");
                self.executions.push(execution);
                ToolCallResult::new(
                    tool_use_id,
                    "The user declined to run this tool.".to_string(),
                    ToolCallOutcome::Denied,
                )
            }
            PermissionDecision::DenyWithReason(reason) => {
                execution.deny(&reason);
                self.executions.push(execution);
                ToolCallResult::new(
                    tool_use_id,
                    format!("The user declined to run this tool: {}", reason),
                    ToolCallOutcome::Denied,
                )
            }
        }
    }

    pub fn execution_history(&self) -> &[ToolExecution] {
        &self.executions
    }

    pub fn clear_history(&mut self) {
        self.executions.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::permissions::AlwaysDenyPermissions;
    use serde_json::json;

    struct Echo;

    #[async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echo the input"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(&self, input: Value) -> Result<String> {
            Ok(input.to_string())
        }
    }

    struct Fails;

    #[async_trait]
    impl Tool for Fails {
        fn name(&self) -> &str {
            "fails"
        }
        fn description(&self) -> &str {
            "Always fails"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(&self, _input: Value) -> Result<String> {
            Err(crate::error::Error::Tool(
                "Permission denied (os error 13)".into(),
            ))
        }
    }

    #[tokio::test]
    async fn success_denial_and_failure_have_distinct_outcomes() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(Echo)).unwrap();
        registry.register(Arc::new(Fails)).unwrap();

        let ok = registry
            .execute_tool("echo", json!({"a": 1}), "id1".into())
            .await;
        assert_eq!(ok.outcome, ToolCallOutcome::Success);

        // A tool error whose text contains "denied" must NOT read as a denial.
        let failed = registry
            .execute_tool("fails", json!({}), "id2".into())
            .await;
        assert_eq!(failed.outcome, ToolCallOutcome::Failed);

        let missing = registry.execute_tool("nope", json!({}), "id3".into()).await;
        assert_eq!(missing.outcome, ToolCallOutcome::Failed);

        let mut denying = ToolRegistry::with_permission_handler(Box::new(AlwaysDenyPermissions));
        denying.register(Arc::new(Echo)).unwrap();
        let denied = denying.execute_tool("echo", json!({}), "id4".into()).await;
        assert_eq!(denied.outcome, ToolCallOutcome::Denied);
    }

    #[test]
    fn tool_defs_preserve_registration_order() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(Fails)).unwrap();
        registry.register(Arc::new(Echo)).unwrap();
        let names: Vec<String> = registry
            .get_tool_defs()
            .iter()
            .map(|d| d.name.clone())
            .collect();
        assert_eq!(names, vec!["fails", "echo"]);
    }
}
