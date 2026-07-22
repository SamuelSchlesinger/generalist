//! Tool execution permission handling.

use crate::types::truncate_middle;
use async_trait::async_trait;
use colored::*;
use dialoguer::{theme::ColorfulTheme, Select};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Decision on whether a tool call may run.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionDecision {
    Allow,
    Deny,
    DenyWithReason(String),
}

/// Everything a handler needs to decide about one tool call.
#[derive(Debug, Clone)]
pub struct ToolExecutionRequest {
    pub tool_use_id: String,
    pub tool_name: String,
    pub input: Value,
    pub tool_description: String,
}

/// Decides whether tool calls run. Implementations range from "always yes"
/// to interactive prompting with remembered choices.
#[async_trait]
pub trait ToolPermissionHandler: Send + Sync {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision;
}

/// Allows everything. The default for library use; do not pair with tools
/// like `bash` unless the environment is trusted.
pub struct AlwaysAllowPermissions;

#[async_trait]
impl ToolPermissionHandler for AlwaysAllowPermissions {
    async fn check_permission(&self, _request: &ToolExecutionRequest) -> PermissionDecision {
        PermissionDecision::Allow
    }
}

/// Denies everything. Useful for tests and dry runs.
pub struct AlwaysDenyPermissions;

#[async_trait]
impl ToolPermissionHandler for AlwaysDenyPermissions {
    async fn check_permission(&self, _request: &ToolExecutionRequest) -> PermissionDecision {
        PermissionDecision::Deny
    }
}

/// Allows or denies by tool name.
pub struct PolicyPermissions {
    allowed_tools: HashSet<String>,
    default_allow: bool,
}

impl PolicyPermissions {
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
        if self.allowed_tools.contains(&request.tool_name) || self.default_allow {
            PermissionDecision::Allow
        } else {
            PermissionDecision::DenyWithReason(format!(
                "Tool '{}' is not in the allowed tools list",
                request.tool_name
            ))
        }
    }
}

/// Pretty-print a unified diff with colors.
fn format_diff_for_display(diff: &str) -> String {
    let mut formatted = String::new();
    for line in diff.lines() {
        let rendered = if line.starts_with("+++") || line.starts_with("---") {
            line.bright_blue().to_string()
        } else if line.starts_with("@@") {
            line.cyan().to_string()
        } else if line.starts_with('+') {
            line.green().to_string()
        } else if line.starts_with('-') {
            line.red().to_string()
        } else {
            line.dimmed().to_string()
        };
        formatted.push_str(&rendered);
        formatted.push('\n');
    }
    formatted
}

/// Interactive handler with remembered always/never decisions per tool.
///
/// Remembered "always allow" is per tool *name*, which means a remembered
/// `bash` approval covers every future command. To keep that meaningful, the
/// auto-allow path still prints the full input before execution.
pub struct MemoryPermissionHandler {
    always_allow: Arc<Mutex<HashSet<String>>>,
    always_deny: Arc<Mutex<HashSet<String>>>,
}

impl Default for MemoryPermissionHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPermissionHandler {
    pub fn new() -> Self {
        Self {
            always_allow: Arc::new(Mutex::new(HashSet::new())),
            always_deny: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Create a handler sharing decision state with another.
    pub fn with_shared_state(
        always_allow: Arc<Mutex<HashSet<String>>>,
        always_deny: Arc<Mutex<HashSet<String>>>,
    ) -> Self {
        Self {
            always_allow,
            always_deny,
        }
    }

    pub fn always_allow(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_allow)
    }

    pub fn always_deny(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_deny)
    }

    pub fn set_always_allow(&self, tools: HashSet<String>) {
        *self.always_allow.lock().unwrap() = tools;
    }

    pub fn set_always_deny(&self, tools: HashSet<String>) {
        *self.always_deny.lock().unwrap() = tools;
    }

    fn print_request(request: &ToolExecutionRequest) {
        println!("\n{}", "⚠️  Tool Permission Request".yellow().bold());
        println!("{}", "─".repeat(50).dimmed());
        println!("Tool: {}", request.tool_name.cyan().bold());
        println!("Description: {}", request.tool_description.dimmed());

        // Show diffs as diffs; everything else as pretty JSON.
        let diff = (request.tool_name == "patch_file")
            .then(|| request.input.get("diff").and_then(|v| v.as_str()))
            .flatten();
        if let Some(diff) = diff {
            if let Some(path) = request.input.get("path").and_then(|v| v.as_str()) {
                println!("Target file: {}", path.yellow());
            }
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
        println!();
    }
}

#[async_trait]
impl ToolPermissionHandler for MemoryPermissionHandler {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        {
            let always_allow = self.always_allow.lock().unwrap();
            if always_allow.contains(&request.tool_name) {
                // Auto-approved, but always show what is about to run.
                let compact = serde_json::to_string(&request.input).unwrap_or_default();
                eprintln!(
                    "{} Auto-allowing {} {}",
                    "✓".green(),
                    request.tool_name.cyan(),
                    truncate_middle(&compact, 300).dimmed()
                );
                return PermissionDecision::Allow;
            }
        }

        {
            let always_deny = self.always_deny.lock().unwrap();
            if always_deny.contains(&request.tool_name) {
                eprintln!(
                    "{} Auto-denying '{}' (previously set to never allow)",
                    "✗".red(),
                    request.tool_name.cyan()
                );
                return PermissionDecision::DenyWithReason(
                    "Tool was previously set to never allow".to_string(),
                );
            }
        }

        Self::print_request(request);

        let choices = vec![
            "Yes (always allow this tool)",
            "Yes (just this once)",
            "No (never allow this tool)",
            "No (just this once)",
        ];

        // On prompt failure (e.g. Ctrl-C at the prompt) fall through to a
        // one-time denial rather than panicking.
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Allow this tool to execute?")
            .items(&choices)
            .default(1)
            .interact()
            .unwrap_or(3);

        match selection {
            0 => {
                self.always_allow
                    .lock()
                    .unwrap()
                    .insert(request.tool_name.clone());
                println!(
                    "{} Tool '{}' will be automatically allowed in the future",
                    "✓".green(),
                    request.tool_name.cyan()
                );
                PermissionDecision::Allow
            }
            1 => PermissionDecision::Allow,
            2 => {
                self.always_deny
                    .lock()
                    .unwrap()
                    .insert(request.tool_name.clone());
                println!(
                    "{} Tool '{}' will be automatically denied in the future",
                    "✗".red(),
                    request.tool_name.cyan()
                );
                PermissionDecision::DenyWithReason(
                    "User chose to never allow this tool".to_string(),
                )
            }
            _ => PermissionDecision::DenyWithReason(
                "User denied permission for this execution".to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(name: &str) -> ToolExecutionRequest {
        ToolExecutionRequest {
            tool_use_id: "id".into(),
            tool_name: name.into(),
            input: json!({}),
            tool_description: "desc".into(),
        }
    }

    #[tokio::test]
    async fn policy_permissions_respect_allowlist_and_default() {
        let allow_list = PolicyPermissions::new(vec!["calculator".into()], false);
        assert_eq!(
            allow_list.check_permission(&request("calculator")).await,
            PermissionDecision::Allow
        );
        assert!(matches!(
            allow_list.check_permission(&request("bash")).await,
            PermissionDecision::DenyWithReason(_)
        ));

        let default_allow = PolicyPermissions::new(vec![], true);
        assert_eq!(
            default_allow.check_permission(&request("bash")).await,
            PermissionDecision::Allow
        );
    }

    #[tokio::test]
    async fn memory_handler_remembers_decisions_without_prompting() {
        let handler = MemoryPermissionHandler::new();
        handler.set_always_allow(["echo".to_string()].into());
        handler.set_always_deny(["bash".to_string()].into());

        assert_eq!(
            handler.check_permission(&request("echo")).await,
            PermissionDecision::Allow
        );
        assert!(matches!(
            handler.check_permission(&request("bash")).await,
            PermissionDecision::DenyWithReason(_)
        ));
    }
}
