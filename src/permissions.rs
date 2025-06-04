use async_trait::async_trait;
use colored::*;
use dialoguer::{theme::ColorfulTheme, Select};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

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
        eprintln!(
            "[TOOL PERMISSION] Allowing tool '{}' with input: {}",
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
            PermissionDecision::DenyWithReason(format!(
                "Tool '{}' is not in the allowed tools list",
                request.tool_name
            ))
        }
    }
}

/// Format a diff for pretty display
fn format_diff_for_display(diff: &str) -> String {
    let mut formatted = String::new();

    for line in diff.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            // File headers
            formatted.push_str(&format!("{}\n", line.bright_blue()));
        } else if line.starts_with("@@") {
            // Hunk headers
            formatted.push_str(&format!("{}\n", line.cyan()));
        } else if line.starts_with("+") && !line.starts_with("+++") {
            // Added lines
            formatted.push_str(&format!("{}\n", line.green()));
        } else if line.starts_with("-") && !line.starts_with("---") {
            // Removed lines
            formatted.push_str(&format!("{}\n", line.red()));
        } else if line.starts_with(" ") {
            // Context lines
            formatted.push_str(&format!("{}\n", line.dimmed()));
        } else {
            // Other lines
            formatted.push_str(&format!("{}\n", line));
        }
    }

    formatted
}

/// Advanced permission handler with memory for always/never decisions
pub struct MemoryPermissionHandler {
    always_allow: Arc<Mutex<HashSet<String>>>,
    always_deny: Arc<Mutex<HashSet<String>>>,
}

impl MemoryPermissionHandler {
    pub fn new() -> Self {
        Self {
            always_allow: Arc::new(Mutex::new(HashSet::new())),
            always_deny: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Create a new handler that shares state with an existing one
    pub fn with_shared_state(
        always_allow: Arc<Mutex<HashSet<String>>>,
        always_deny: Arc<Mutex<HashSet<String>>>,
    ) -> Self {
        Self {
            always_allow,
            always_deny,
        }
    }

    /// Get the always_allow set for state management
    pub fn always_allow(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_allow)
    }

    /// Get the always_deny set for state management
    pub fn always_deny(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_deny)
    }

    /// Update the always_allow set
    pub fn set_always_allow(&self, tools: HashSet<String>) {
        *self.always_allow.lock().unwrap() = tools;
    }

    /// Update the always_deny set
    pub fn set_always_deny(&self, tools: HashSet<String>) {
        *self.always_deny.lock().unwrap() = tools;
    }
}

#[async_trait]
impl ToolPermissionHandler for MemoryPermissionHandler {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        // Check if we have a remembered decision
        {
            let always_allow = self.always_allow.lock().unwrap();
            if always_allow.contains(&request.tool_name) {
                eprintln!(
                    "{} Automatically allowing '{}' (previously set to always allow)",
                    "✓".green(),
                    request.tool_name.cyan()
                );
                return PermissionDecision::Allow;
            }
        }

        {
            let always_deny = self.always_deny.lock().unwrap();
            if always_deny.contains(&request.tool_name) {
                eprintln!(
                    "{} Automatically denying '{}' (previously set to never allow)",
                    "✗".red(),
                    request.tool_name.cyan()
                );
                return PermissionDecision::DenyWithReason(
                    "Tool was previously set to never allow".to_string(),
                );
            }
        }

        // No remembered decision, prompt the user
        println!("\n{}", "⚠️  Tool Permission Request".yellow().bold());
        println!("{}", "─".repeat(50).dimmed());
        println!("Tool: {}", request.tool_name.cyan().bold());
        println!("Description: {}", request.tool_description.dimmed());

        // Special formatting for patch_file tool
        if request.tool_name == "patch_file" {
            if let Some(path) = request.input.get("path").and_then(|v| v.as_str()) {
                println!("Target file: {}", path.yellow());
            }
            if let Some(diff) = request.input.get("diff").and_then(|v| v.as_str()) {
                println!("\n{}", "Proposed changes:".bold());
                println!("{}", "─".repeat(50).dimmed());
                print!("{}", format_diff_for_display(diff));
                println!("{}", "─".repeat(50).dimmed());
            } else {
                println!(
                    "Input: {}",
                    serde_json::to_string_pretty(&request.input)
                        .unwrap_or_default()
                        .dimmed()
                );
            }
        } else {
            println!(
                "Input: {}",
                serde_json::to_string_pretty(&request.input)
                    .unwrap_or_default()
                    .dimmed()
            );
        }
        println!();

        let choices = vec![
            "Yes (always allow this tool)",
            "Yes (just this once)",
            "No (never allow this tool)",
            "No (just this once)",
        ];

        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Allow this tool to execute?")
            .items(&choices)
            .default(1) // Default to "Yes (just this once)"
            .interact()
            .unwrap();

        match selection {
            0 => {
                // Yes (always)
                let mut always_allow = self.always_allow.lock().unwrap();
                always_allow.insert(request.tool_name.clone());
                println!(
                    "{} Tool '{}' will be automatically allowed in the future",
                    "✓".green(),
                    request.tool_name.cyan()
                );
                PermissionDecision::Allow
            }
            1 => {
                // Yes (once)
                PermissionDecision::Allow
            }
            2 => {
                // No (never)
                let mut always_deny = self.always_deny.lock().unwrap();
                always_deny.insert(request.tool_name.clone());
                println!(
                    "{} Tool '{}' will be automatically denied in the future",
                    "✗".red(),
                    request.tool_name.cyan()
                );
                PermissionDecision::DenyWithReason(
                    "User chose to never allow this tool".to_string(),
                )
            }
            3 => {
                // No (once)
                PermissionDecision::DenyWithReason(
                    "User denied permission for this execution".to_string(),
                )
            }
            _ => unreachable!(),
        }
    }
}
