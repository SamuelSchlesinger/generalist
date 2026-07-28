//! Tool trait and registry.

use crate::error::Result;
use crate::execution::{ExecutionState, ToolExecution};
use crate::model_trace::{ArchiveModelAction, AsyncModelAction, MemoryModelAction, ModelTrace};
use crate::permissions::{
    AlwaysAllowPermissions, PermissionDecision, ToolExecutionRequest, ToolPermissionHandler,
};
use crate::scope::ScopeFilter;
use crate::types::{ContentBlock, ToolDef};
use crate::Error;
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

    /// Execute after the registry has authorized this exact tool call.
    ///
    /// Ordinary tools inherit the adapter to [`Self::execute`]. Sensitive
    /// tools can reject direct execution and override this method so their
    /// backend receives an unforgeable authorization capability.
    async fn execute_authorized(
        &self,
        input: Value,
        authorization: &ToolAuthorization,
    ) -> Result<String> {
        authorization.require_exact(self.name(), &input)?;
        self.execute(input).await
    }

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

/// Cross-scope disclosure operation authorized by one exact tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisclosureCapability {
    SearchMemories,
    ReadMemory,
    SearchConversations,
    ReadConversation,
}

impl DisclosureCapability {
    fn tool_name(self) -> &'static str {
        match self {
            Self::SearchMemories => "search_memories",
            Self::ReadMemory => "read_memory",
            Self::SearchConversations => "search_conversations",
            Self::ReadConversation => "read_conversation",
        }
    }

    fn archive_kind(self) -> &'static str {
        match self {
            Self::SearchMemories | Self::ReadMemory => "memory",
            Self::SearchConversations | Self::ReadConversation => "history",
        }
    }
}

/// Proof that the registry's permission handler allowed one exact request.
///
/// Construction is private to [`ToolRegistry`]. A sensitive tool may derive a
/// narrower [`DisclosureGrant`] only when its name and complete JSON input
/// still match the authorized request.
#[derive(Debug)]
pub struct ToolAuthorization {
    request: ToolExecutionRequest,
}

impl ToolAuthorization {
    fn new(request: ToolExecutionRequest) -> Self {
        Self { request }
    }

    fn require_exact(&self, tool_name: &str, input: &Value) -> Result<()> {
        if self.request.tool_name != tool_name || self.request.input != *input {
            return Err(Error::Other(
                "Tool execution no longer matches the authorized request".to_string(),
            ));
        }
        Ok(())
    }

    pub fn disclosure_grant(
        &self,
        capability: DisclosureCapability,
        input: &Value,
    ) -> Result<DisclosureGrant> {
        self.require_exact(capability.tool_name(), input)?;
        let scope = input
            .get("scope")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Other("Authorized archive request has no scope".to_string()))
            .and_then(ScopeFilter::parse)?;
        Ok(DisclosureGrant {
            capability,
            scope,
            input: input.clone(),
        })
    }
}

/// Narrow capability required by cross-scope storage APIs.
///
/// It carries the exact authorized input as well as the parsed categorical
/// scope, preventing a tool from substituting a different query, ID, or
/// expected scope after permission was granted.
#[derive(Debug)]
pub struct DisclosureGrant {
    capability: DisclosureCapability,
    scope: ScopeFilter,
    input: Value,
}

impl DisclosureGrant {
    pub fn ensure_search(
        &self,
        capability: DisclosureCapability,
        query: &str,
        scope: ScopeFilter,
    ) -> Result<()> {
        self.ensure_capability(capability, scope)?;
        if self.input.get("query").and_then(Value::as_str) != Some(query) {
            return Err(Error::Other(
                "Archive search query differs from the authorized request".to_string(),
            ));
        }
        Ok(())
    }

    pub fn ensure_read(
        &self,
        capability: DisclosureCapability,
        id: &str,
        scope: ScopeFilter,
        expected_scope: &str,
    ) -> Result<()> {
        self.ensure_capability(capability, scope)?;
        if self.input.get("id").and_then(Value::as_str) != Some(id)
            || self.input.get("expected_scope").and_then(Value::as_str) != Some(expected_scope)
        {
            return Err(Error::Other(
                "Archive read target differs from the authorized request".to_string(),
            ));
        }
        Ok(())
    }

    fn ensure_capability(
        &self,
        capability: DisclosureCapability,
        scope: ScopeFilter,
    ) -> Result<()> {
        if self.capability != capability || self.scope != scope {
            return Err(Error::Other(
                "Archive disclosure capability does not match this operation".to_string(),
            ));
        }
        Ok(())
    }
}

/// Holds the available tools, runs them behind a permission handler, and
/// records execution history.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    order: Vec<String>,
    executions: Vec<ToolExecution>,
    permission_handler: Box<dyn ToolPermissionHandler>,
    model_trace: Option<ModelTrace>,
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
            model_trace: None,
        }
    }

    pub fn set_permission_handler(&mut self, handler: Box<dyn ToolPermissionHandler>) {
        self.permission_handler = handler;
    }

    #[doc(hidden)]
    pub fn set_model_trace(&mut self, trace: ModelTrace) {
        self.model_trace = Some(trace);
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
        if name == crate::goal::UPDATE_GOAL_TOOL_NAME {
            return Err(crate::error::Error::Other(format!(
                "Tool '{}' is reserved for the host goal controller",
                name
            )));
        }
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
    /// into the sole model-facing `python` capability tool instead of being
    /// independently callable. Host controls such as `update_goal` are owned
    /// by [`crate::Agent`] and never enter this registry.
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
        let archive_request = archive_request(&request);
        let model_request_id = self.model_trace.as_ref().map(ModelTrace::next_request_id);
        if let Some(trace) = &self.model_trace {
            trace.record_async(AsyncModelAction::AskPermission {
                request_id: model_request_id
                    .clone()
                    .expect("a model trace allocated no permission ID"),
            });
            if let Some((kind, filter)) = archive_request {
                trace.record_archive(ArchiveModelAction::RequestSearch {
                    kind: kind.to_string(),
                    filter,
                });
                if kind == "memory" {
                    trace.record_memory(MemoryModelAction::RequestSearch { filter });
                }
            }
        }

        match self.permission_handler.check_permission(&request).await {
            PermissionDecision::Allow => {
                if let Some(trace) = &self.model_trace {
                    trace.record_async(AsyncModelAction::AllowPermission {
                        request_id: model_request_id
                            .clone()
                            .expect("a model trace allocated no permission ID"),
                    });
                }
                execution.state = ExecutionState::Executing;
                let authorization = ToolAuthorization::new(request);
                let result = tool.execute_authorized(input, &authorization).await;
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
                self.record_denial(model_request_id.as_deref(), archive_request);
                execution.deny("Permission denied");
                self.executions.push(execution);
                ToolCallResult::new(
                    tool_use_id,
                    "The user declined to run this tool.".to_string(),
                    ToolCallOutcome::Denied,
                )
            }
            PermissionDecision::DenyWithReason(reason) => {
                self.record_denial(model_request_id.as_deref(), archive_request);
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

    fn record_denial(
        &self,
        request_id: Option<&str>,
        archive_request: Option<(&'static str, ScopeFilter)>,
    ) {
        let Some(trace) = &self.model_trace else {
            return;
        };
        trace.record_async(AsyncModelAction::DenyPermission {
            request_id: request_id
                .expect("a model trace allocated no permission ID")
                .to_string(),
        });
        if let Some((kind, _)) = archive_request {
            trace.record_archive(ArchiveModelAction::DenySearch);
            if kind == "memory" {
                trace.record_memory(MemoryModelAction::DenySearch);
            }
        }
    }
}

fn archive_request(request: &ToolExecutionRequest) -> Option<(&'static str, ScopeFilter)> {
    let capability = match request.tool_name.as_str() {
        "search_memories" => DisclosureCapability::SearchMemories,
        "read_memory" => DisclosureCapability::ReadMemory,
        "search_conversations" => DisclosureCapability::SearchConversations,
        "read_conversation" => DisclosureCapability::ReadConversation,
        _ => return None,
    };
    let filter = request
        .input
        .get("scope")
        .and_then(Value::as_str)
        .and_then(|scope| ScopeFilter::parse(scope).ok())?;
    Some((capability.archive_kind(), filter))
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
    fn disclosure_grants_are_bound_to_the_exact_authorized_call() {
        let input = json!({
            "query": "needle",
            "scope": "other_projects",
        });
        let authorization = ToolAuthorization::new(ToolExecutionRequest {
            tool_use_id: "request-1".to_string(),
            tool_name: "search_memories".to_string(),
            input: input.clone(),
            tool_description: "search".to_string(),
        });
        let grant = authorization
            .disclosure_grant(DisclosureCapability::SearchMemories, &input)
            .unwrap();

        grant
            .ensure_search(
                DisclosureCapability::SearchMemories,
                "needle",
                ScopeFilter::OtherProjects,
            )
            .unwrap();
        assert!(grant
            .ensure_search(
                DisclosureCapability::SearchMemories,
                "substituted",
                ScopeFilter::OtherProjects,
            )
            .is_err());
        assert!(grant
            .ensure_search(
                DisclosureCapability::SearchMemories,
                "needle",
                ScopeFilter::All,
            )
            .is_err());
        assert!(authorization
            .disclosure_grant(
                DisclosureCapability::SearchMemories,
                &json!({"query": "needle", "scope": "all"}),
            )
            .is_err());

        let read_input = json!({
            "id": "episode-1",
            "scope": "all",
            "expected_scope": "/project",
            "offset": 0,
        });
        let read_authorization = ToolAuthorization::new(ToolExecutionRequest {
            tool_use_id: "request-2".to_string(),
            tool_name: "read_memory".to_string(),
            input: read_input.clone(),
            tool_description: "read".to_string(),
        });
        let read_grant = read_authorization
            .disclosure_grant(DisclosureCapability::ReadMemory, &read_input)
            .unwrap();
        read_grant
            .ensure_read(
                DisclosureCapability::ReadMemory,
                "episode-1",
                ScopeFilter::All,
                "/project",
            )
            .unwrap();
        assert!(read_grant
            .ensure_read(
                DisclosureCapability::ReadMemory,
                "episode-2",
                ScopeFilter::All,
                "/project",
            )
            .is_err());
        assert!(read_grant
            .ensure_read(
                DisclosureCapability::ReadMemory,
                "episode-1",
                ScopeFilter::All,
                "/other-project",
            )
            .is_err());
        assert!(read_authorization
            .disclosure_grant(
                DisclosureCapability::ReadMemory,
                &json!({
                    "id": "episode-1",
                    "scope": "all",
                    "expected_scope": "/project",
                    "offset": 1,
                }),
            )
            .is_err());
    }

    #[tokio::test]
    async fn model_permissions_use_fresh_trace_local_request_ids() {
        let trace = ModelTrace::for_models(&[crate::ModelKind::AsyncRuntime]);
        let mut registry = ToolRegistry::new();
        registry.set_model_trace(trace.clone());
        registry.register(Arc::new(Echo)).unwrap();

        for _ in 0..2 {
            let result = registry
                .execute_tool("echo", json!({}), "reused-provider-id".to_string())
                .await;
            assert_eq!(result.outcome, ToolCallOutcome::Success);
        }

        let requests = trace
            .snapshot()
            .async_runtime
            .into_iter()
            .filter_map(|action| match action {
                AsyncModelAction::AskPermission { request_id } => Some(request_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(requests.len(), 2);
        assert_ne!(requests[0], requests[1]);
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

    struct ReservedGoalControl;

    #[async_trait]
    impl Tool for ReservedGoalControl {
        fn name(&self) -> &str {
            crate::goal::UPDATE_GOAL_TOOL_NAME
        }
        fn description(&self) -> &str {
            "must not shadow the host goal controller"
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object"})
        }
        async fn execute(&self, _input: Value) -> Result<String> {
            Ok(String::new())
        }
    }

    #[test]
    fn host_goal_control_name_is_reserved() {
        let mut registry = ToolRegistry::new();
        let error = registry
            .register(Arc::new(ReservedGoalControl))
            .unwrap_err();
        assert!(error.to_string().contains("reserved"));
        assert!(!registry.has_tool(crate::goal::UPDATE_GOAL_TOOL_NAME));
    }
}
