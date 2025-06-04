use crate::error::{Error, Result};
use crate::execution::{ExecutionState, ToolExecution};
use crate::message::ContentBlock;
use crate::permissions::{
    AlwaysAllowPermissions, PermissionDecision, ToolExecutionRequest, ToolPermissionHandler,
};
use crate::request::ToolDef;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Trait defining a tool that Claude can use during conversations
///
/// Implement this trait to create custom tools that extend Claude's capabilities.
/// Tools can perform any computation or side effect and return results back to Claude.
///
/// # Example
///
/// ```rust
/// use claude::{Tool, ToolDef};
/// use serde_json::{json, Value};
/// use async_trait::async_trait;
///
/// struct Calculator;
///
/// #[async_trait]
/// impl Tool for Calculator {
///     fn name(&self) -> &str {
///         "calculator"
///     }
///     
///     fn description(&self) -> &str {
///         "Performs basic arithmetic operations"
///     }
///     
///     fn input_schema(&self) -> Value {
///         json!({
///             "type": "object",
///             "properties": {
///                 "expression": {
///                     "type": "string",
///                     "description": "Mathematical expression to evaluate"
///                 }
///             },
///             "required": ["expression"]
///         })
///     }
///     
///     async fn execute(&self, input: Value) -> Result<String, claude::Error> {
///         let expr = input["expression"]
///             .as_str()
///             .ok_or_else(|| claude::Error::Other("Missing expression".to_string()))?;
///         
///         // Simplified - real implementation would parse and evaluate
///         match expr {
///             "2+2" => Ok("4".to_string()),
///             _ => Ok("Complex calculation not implemented".to_string()),
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get the unique name of this tool
    fn name(&self) -> &str;

    /// Get a human-readable description of what this tool does
    fn description(&self) -> &str;

    /// Get the JSON schema defining the expected input format
    fn input_schema(&self) -> Value;

    /// Execute the tool with the given input parameters
    ///
    /// # Arguments
    ///
    /// * `input` - The input parameters as a JSON Value matching the schema
    ///
    /// # Returns
    ///
    /// Returns a Result containing either the tool's output as a string or an error
    async fn execute(&self, input: Value) -> Result<String>;

    /// Convert this tool to a ToolDef for use with the Claude API
    fn to_tool_def(&self) -> ToolDef {
        ToolDef {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }
}

/// Registry for managing available tools
///
/// The `ToolRegistry` maintains a collection of tools that Claude can use,
/// handles tool execution with permission checking, and tracks execution history.
///
/// # Example
///
/// ```rust
/// use claude::{ToolRegistry, Tool};
/// use std::sync::Arc;
/// # use async_trait::async_trait;
/// # use serde_json::Value;
/// # struct MyTool;
/// # #[async_trait]
/// # impl Tool for MyTool {
/// #     fn name(&self) -> &str { "my_tool" }
/// #     fn description(&self) -> &str { "A custom tool" }
/// #     fn input_schema(&self) -> Value { serde_json::json!({}) }
/// #     async fn execute(&self, input: Value) -> Result<String, claude::Error> { Ok("Done".to_string()) }
/// # }
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut registry = ToolRegistry::new();
///
/// // Register a tool
/// registry.register(Arc::new(MyTool))?;
///
/// // Check available tools
/// assert!(registry.has_tool("my_tool"));
///
/// // Get tool definitions for Claude
/// let tool_defs = registry.get_tool_defs();
/// assert_eq!(tool_defs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    executions: Vec<ToolExecution>,
    permission_handler: Box<dyn ToolPermissionHandler>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            executions: Vec::new(),
            permission_handler: Box::new(AlwaysAllowPermissions),
        }
    }

    /// Create a new tool registry with a custom permission handler
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::{ToolRegistry, LoggingPermissions};
    ///
    /// let registry = ToolRegistry::with_permission_handler(
    ///     Box::new(LoggingPermissions)
    /// );
    /// ```
    pub fn with_permission_handler(handler: Box<dyn ToolPermissionHandler>) -> Self {
        Self {
            tools: HashMap::new(),
            executions: Vec::new(),
            permission_handler: handler,
        }
    }

    /// Set a new permission handler for this registry
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::{ToolRegistry, InteractivePermissions};
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.set_permission_handler(Box::new(
    ///     InteractivePermissions::new(|req| {
    ///         println!("Allow tool '{}'?", req.tool_name);
    ///         true // Allow all for this example
    ///     })
    /// ));
    /// ```
    pub fn set_permission_handler(&mut self, handler: Box<dyn ToolPermissionHandler>) {
        self.permission_handler = handler;
    }

    /// Register a new tool in the registry
    ///
    /// # Errors
    ///
    /// Returns an error if a tool with the same name is already registered
    ///
    /// # Example
    ///
    /// ```rust
    /// # use claude::{ToolRegistry, Tool};
    /// # use std::sync::Arc;
    /// # use async_trait::async_trait;
    /// # use serde_json::Value;
    /// # struct MyTool;
    /// # #[async_trait]
    /// # impl Tool for MyTool {
    /// #     fn name(&self) -> &str { "my_tool" }
    /// #     fn description(&self) -> &str { "A custom tool" }
    /// #     fn input_schema(&self) -> Value { serde_json::json!({}) }
    /// #     async fn execute(&self, input: Value) -> Result<String, claude::Error> { Ok("Done".to_string()) }
    /// # }
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Arc::new(MyTool))?;
    /// # Ok::<(), claude::Error>(())
    /// ```
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(Error::Other(format!("Tool '{}' already registered", name)));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Get tool definitions for all registered tools
    ///
    /// Returns a vector of ToolDef structs that can be sent to the Claude API
    pub fn get_tool_defs(&self) -> Vec<ToolDef> {
        self.tools.values().map(|tool| tool.to_tool_def()).collect()
    }

    /// Check if a tool with the given name is registered
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get a reference to a specific tool by name
    pub fn get_tool(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Get all registered tool names
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Execute a tool with permission checking
    ///
    /// # Arguments
    ///
    /// * `tool_name` - Name of the tool to execute
    /// * `input` - Input parameters for the tool
    /// * `tool_use_id` - Unique identifier for this tool execution
    ///
    /// # Returns
    ///
    /// Returns Ok with a ContentBlock containing the result or error
    pub async fn execute_tool(
        &mut self,
        tool_name: &str,
        input: Value,
        tool_use_id: String,
    ) -> Result<ContentBlock> {
        // Find the tool
        let tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| Error::Other(format!("Tool '{}' not found", tool_name)))?
            .clone();

        // Create execution record
        let mut execution =
            ToolExecution::new(tool_use_id.clone(), tool_name.to_string(), input.clone());

        // Check permissions
        let request = ToolExecutionRequest {
            tool_use_id: tool_use_id.clone(),
            tool_name: tool_name.to_string(),
            input: input.clone(),
            tool_description: tool.description().to_string(),
        };

        let decision = self.permission_handler.check_permission(&request).await;

        match decision {
            PermissionDecision::Allow => {
                execution.state = ExecutionState::Executing;
                self.executions.push(execution.clone());

                // Execute the tool
                match tool.execute(input).await {
                    Ok(output) => {
                        // Update execution record
                        if let Some(exec) = self.executions.iter_mut().find(|e| e.id == tool_use_id)
                        {
                            exec.complete(Ok(output.clone()));
                        }

                        Ok(ContentBlock::ToolResult {
                            content: output,
                            tool_use_id,
                            is_error: None,
                        })
                    }
                    Err(e) => {
                        let error_msg = e.to_string();

                        // Update execution record
                        if let Some(exec) = self.executions.iter_mut().find(|e| e.id == tool_use_id)
                        {
                            exec.complete(Err(error_msg.clone()));
                        }

                        Ok(ContentBlock::ToolResult {
                            content: format!("Tool execution failed: {}", error_msg),
                            tool_use_id,
                            is_error: Some(true),
                        })
                    }
                }
            }
            PermissionDecision::Deny => {
                execution.deny("Permission denied");
                self.executions.push(execution);

                Ok(ContentBlock::ToolResult {
                    content: "Tool execution denied".to_string(),
                    tool_use_id,
                    is_error: Some(true),
                })
            }
            PermissionDecision::DenyWithReason(reason) => {
                execution.deny(&reason);
                self.executions.push(execution);

                Ok(ContentBlock::ToolResult {
                    content: format!("Tool execution denied: {}", reason),
                    tool_use_id,
                    is_error: Some(true),
                })
            }
        }
    }

    /// Get the execution history
    pub fn execution_history(&self) -> &[ToolExecution] {
        &self.executions
    }

    /// Clear the execution history
    pub fn clear_history(&mut self) {
        self.executions.clear();
    }

    /// Get execution statistics
    ///
    /// Returns a summary of tool executions including counts by status
    pub fn execution_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total".to_string(), self.executions.len());

        let mut completed = 0;
        let mut failed = 0;
        let mut denied = 0;
        let mut executing = 0;

        for exec in &self.executions {
            match &exec.state {
                ExecutionState::Completed { .. } => completed += 1,
                ExecutionState::Failed { .. } => failed += 1,
                ExecutionState::Denied { .. } => denied += 1,
                ExecutionState::Executing => executing += 1,
                _ => {}
            }
        }

        stats.insert("completed".to_string(), completed);
        stats.insert("failed".to_string(), failed);
        stats.insert("denied".to_string(), denied);
        stats.insert("executing".to_string(), executing);

        stats
    }
}
