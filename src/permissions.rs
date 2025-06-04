use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashSet;

/// Decision on whether to allow a tool execution
///
/// Returned by [`ToolPermissionHandler::check_permission`] to control
/// whether a tool execution should proceed.
///
/// # Example
///
/// ```rust
/// use claude::PermissionDecision;
///
/// // Always allow calculator
/// fn check_calculator() -> PermissionDecision {
///     PermissionDecision::Allow
/// }
///
/// // Deny dangerous operations
/// fn check_file_delete() -> PermissionDecision {
///     PermissionDecision::DenyWithReason(
///         "File deletion is not permitted".to_string()
///     )
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionDecision {
    /// Allow the tool execution
    Allow,
    /// Deny the tool execution
    Deny,
    /// Deny with a custom message
    DenyWithReason(String),
}

/// Information about a tool execution request for permission checking
///
/// Contains all the information a [`ToolPermissionHandler`] needs to make
/// an informed decision about whether to allow a tool execution.
///
/// # Example
///
/// ```rust
/// use claude::{ToolExecutionRequest, PermissionDecision};
/// use serde_json::json;
///
/// fn check_request(request: &ToolExecutionRequest) -> PermissionDecision {
///     // Check tool name
///     if request.tool_name == "dangerous_tool" {
///         return PermissionDecision::Deny;
///     }
///     
///     // Check input parameters
///     if let Some(path) = request.input.get("path") {
///         if path.as_str() == Some("/etc/passwd") {
///             return PermissionDecision::DenyWithReason(
///                 "Cannot access system files".to_string()
///             );
///         }
///     }
///     
///     PermissionDecision::Allow
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ToolExecutionRequest {
    /// Unique identifier for this tool use
    pub tool_use_id: String,
    /// Name of the tool being called
    pub tool_name: String,
    /// Input parameters for the tool
    pub input: Value,
    /// Description of what the tool does
    pub tool_description: String,
}

/// Trait for handling tool execution permissions
///
/// The `ToolPermissionHandler` trait provides a flexible way to control tool execution.
/// Implement this trait to add security, logging, or interactive approval to your
/// Claude-powered applications.
///
/// # Built-in Implementations
///
/// - [`AlwaysAllowPermissions`]: Allows all tool executions (default)
/// - [`AlwaysDenyPermissions`]: Denies all tool executions
/// - [`LoggingPermissions`]: Logs tool requests before allowing
/// - [`InteractivePermissions`]: Prompts for user approval
/// - [`PolicyPermissions`]: Allows/denies based on tool names
///
/// # Example
///
/// ```rust
/// use claude::{ToolPermissionHandler, ToolExecutionRequest, PermissionDecision};
/// use async_trait::async_trait;
///
/// struct CustomPermissions;
///
/// #[async_trait]
/// impl ToolPermissionHandler for CustomPermissions {
///     async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
///         // Only allow calculator tool
///         if request.tool_name == "calculator" {
///             PermissionDecision::Allow
///         } else {
///             PermissionDecision::DenyWithReason(
///                 format!("Tool '{}' is not allowed", request.tool_name)
///             )
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait ToolPermissionHandler: Send + Sync {
    /// Check if a tool execution should be allowed
    ///
    /// # Arguments
    ///
    /// * `request` - Information about the tool execution request
    ///
    /// # Returns
    ///
    /// A [`PermissionDecision`] indicating whether to allow or deny execution
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision;
}

/// Permission handler that always allows tool execution
///
/// This is the default permission handler used by [`ToolRegistry`].
/// It allows all tool executions without any checks.
///
/// # Example
///
/// ```rust
/// use claude::{ToolRegistry, AlwaysAllowPermissions};
///
/// // Create a registry with default permissions (allows all)
/// let mut registry = ToolRegistry::new();
///
/// // Explicitly set to always allow
/// registry.set_permission_handler(Box::new(AlwaysAllowPermissions));
/// ```
pub struct AlwaysAllowPermissions;

#[async_trait]
impl ToolPermissionHandler for AlwaysAllowPermissions {
    async fn check_permission(&self, _request: &ToolExecutionRequest) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

/// Permission handler that always denies tool execution
///
/// Useful for testing or when you want to completely disable tool execution.
///
/// # Example
///
/// ```rust
/// use claude::{ToolRegistry, AlwaysDenyPermissions};
///
/// // Create a registry that won't execute any tools
/// let mut registry = ToolRegistry::with_permission_handler(
///     Box::new(AlwaysDenyPermissions)
/// );
/// ```
pub struct AlwaysDenyPermissions;

#[async_trait]
impl ToolPermissionHandler for AlwaysDenyPermissions {
    async fn check_permission(&self, _request: &ToolExecutionRequest) -> PermissionDecision {
        PermissionDecision::Deny
    }
}

/// Permission handler that logs tool requests before allowing
///
/// Prints tool execution requests to stderr before allowing them.
/// Useful for debugging and monitoring tool usage.
///
/// # Example
///
/// ```rust
/// use claude::{ToolRegistry, LoggingPermissions};
///
/// // Create a registry that logs all tool executions
/// let mut registry = ToolRegistry::with_permission_handler(
///     Box::new(LoggingPermissions)
/// );
/// ```
pub struct LoggingPermissions;

#[async_trait]
impl ToolPermissionHandler for LoggingPermissions {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        eprintln!("[TOOL PERMISSION] Allowing tool '{}' with input: {}", 
            request.tool_name, 
            serde_json::to_string_pretty(&request.input).unwrap_or_default()
        );
        PermissionDecision::Allow
    }
}

/// Interactive permission handler that prompts the user for approval
///
/// This handler will display tool execution requests to the user and ask
/// for approval before allowing execution. Useful for safety-critical applications.
pub struct InteractivePermissions<F>
where
    F: Fn(&ToolExecutionRequest) -> bool + Send + Sync,
{
    /// Callback function that returns true to allow, false to deny
    prompt_callback: F,
}

impl<F> InteractivePermissions<F>
where
    F: Fn(&ToolExecutionRequest) -> bool + Send + Sync,
{
    /// Create a new interactive permission handler with a custom prompt callback
    ///
    /// The callback should display the tool request to the user and return
    /// true to allow execution, false to deny.
    pub fn new(prompt_callback: F) -> Self {
        Self { prompt_callback }
    }
}

#[async_trait]
impl<F> ToolPermissionHandler for InteractivePermissions<F>
where
    F: Fn(&ToolExecutionRequest) -> bool + Send + Sync,
{
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        if (self.prompt_callback)(request) {
            PermissionDecision::Allow
        } else {
            PermissionDecision::DenyWithReason("User denied permission".to_string())
        }
    }
}

/// Policy-based permission handler that allows or denies based on tool names
///
/// Maintains an allow-list of tool names and can be configured with a default
/// policy for tools not in the list.
///
/// # Example
///
/// ```rust
/// use claude::{ToolRegistry, PolicyPermissions};
///
/// // Only allow specific tools
/// let allowed_tools = vec![
///     "calculator".to_string(),
///     "get_weather".to_string(),
/// ];
///
/// let policy = PolicyPermissions::new(allowed_tools, false);
/// let mut registry = ToolRegistry::with_permission_handler(
///     Box::new(policy)
/// );
/// ```
pub struct PolicyPermissions {
    /// Set of allowed tool names
    allowed_tools: HashSet<String>,
    /// Whether to allow tools not in the allowed set
    default_allow: bool,
}

impl PolicyPermissions {
    /// Create a new policy permission handler
    ///
    /// # Arguments
    ///
    /// * `allowed_tools` - Set of tool names that are allowed
    /// * `default_allow` - Whether to allow tools not in the allowed set
    pub fn new(allowed_tools: Vec<String>, default_allow: bool) -> Self {
        Self {
            allowed_tools: allowed_tools.into_iter().collect(),
            default_allow,
        }
    }
}

#[async_trait]
impl ToolPermissionHandler for PolicyPermissions {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        if self.allowed_tools.contains(&request.tool_name) {
            PermissionDecision::Allow
        } else if self.default_allow {
            PermissionDecision::Allow
        } else {
            PermissionDecision::DenyWithReason(
                format!("Tool '{}' is not in the allowed tools list", request.tool_name)
            )
        }
    }
}