//! Tool execution permission handling.

use crate::types::truncate_middle;
use async_trait::async_trait;
use colored::*;
use dialoguer::{theme::ColorfulTheme, Select};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

/// Decision on whether a tool call may run.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionDecision {
    Allow,
    Deny,
    DenyWithReason(String),
}

/// The four choices presented by an interactive permission prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionChoice {
    AllowAlways,
    AllowOnce,
    DenyAlways,
    DenyOnce,
}

/// A consistent snapshot of remembered per-tool decisions.
///
/// The sets are disjoint. If legacy or externally shared state contains the
/// same tool in both sets, deny wins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RememberedPermissionPolicy {
    pub always_allow: HashSet<String>,
    pub always_deny: HashSet<String>,
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

/// UI adapter used by [`MemoryPermissionHandler`] for decisions that have not
/// already been remembered.
///
/// This is asynchronous so a terminal frontend can broker the request to its
/// single event reactor without blocking model progress or adding a second
/// terminal reader.
#[async_trait]
pub trait PermissionPrompt: Send + Sync {
    async fn choose(&self, request: &ToolExecutionRequest) -> PermissionChoice;

    /// Surface a remembered choice without opening another prompt.
    fn automatic_decision(&self, _request: &ToolExecutionRequest, _allowed: bool) {}
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

struct ConsolePermissionPrompt;

impl ConsolePermissionPrompt {
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
impl PermissionPrompt for ConsolePermissionPrompt {
    async fn choose(&self, request: &ToolExecutionRequest) -> PermissionChoice {
        let request = request.clone();
        tokio::task::spawn_blocking(move || {
            Self::print_request(&request);
            Self::choose_blocking()
        })
        .await
        .unwrap_or(PermissionChoice::DenyOnce)
    }

    fn automatic_decision(&self, request: &ToolExecutionRequest, allowed: bool) {
        let compact = serde_json::to_string(&request.input).unwrap_or_default();
        if allowed {
            eprintln!(
                "{} Auto-allowing {} {}",
                "✓".green(),
                request.tool_name.cyan(),
                truncate_middle(&compact, 300).dimmed()
            );
        } else {
            eprintln!(
                "{} Auto-denying '{}' (previously set to never allow)",
                "✗".red(),
                request.tool_name.cyan()
            );
        }
    }
}

impl ConsolePermissionPrompt {
    fn choose_blocking() -> PermissionChoice {
        let choices = [
            "Yes (always allow this tool)",
            "Yes (just this once)",
            "No (never allow this tool)",
            "No (just this once)",
        ];
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Allow this tool to execute?")
            .items(choices)
            .default(1)
            .interact()
            .unwrap_or(3);
        match selection {
            0 => PermissionChoice::AllowAlways,
            1 => PermissionChoice::AllowOnce,
            2 => PermissionChoice::DenyAlways,
            _ => PermissionChoice::DenyOnce,
        }
    }
}

/// Request sent by an asynchronous permission prompt to the UI reactor.
#[derive(Debug)]
pub struct PermissionRequest {
    pub id: u64,
    pub request: ToolExecutionRequest,
    pub reply: oneshot::Sender<PermissionChoice>,
}

/// Permission-related events consumed by the UI reactor.
#[derive(Debug)]
pub enum PermissionUiEvent {
    Request(PermissionRequest),
    Automatic {
        request: ToolExecutionRequest,
        allowed: bool,
    },
}

/// Prompt implementation that delegates every interactive choice to a single
/// UI event loop and awaits its correlated one-shot reply.
pub struct PermissionBrokerPrompt {
    sender: mpsc::UnboundedSender<PermissionUiEvent>,
    next_id: AtomicU64,
}

impl PermissionBrokerPrompt {
    pub fn new(sender: mpsc::UnboundedSender<PermissionUiEvent>) -> Self {
        Self {
            sender,
            next_id: AtomicU64::new(1),
        }
    }
}

#[async_trait]
impl PermissionPrompt for PermissionBrokerPrompt {
    async fn choose(&self, request: &ToolExecutionRequest) -> PermissionChoice {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (reply, answer) = oneshot::channel();
        let event = PermissionUiEvent::Request(PermissionRequest {
            id,
            request: request.clone(),
            reply,
        });
        if self.sender.send(event).is_err() {
            return PermissionChoice::DenyOnce;
        }
        answer.await.unwrap_or(PermissionChoice::DenyOnce)
    }

    fn automatic_decision(&self, request: &ToolExecutionRequest, allowed: bool) {
        let _ = self.sender.send(PermissionUiEvent::Automatic {
            request: request.clone(),
            allowed,
        });
    }
}

/// Interactive handler with remembered always/never decisions per tool.
///
/// Remembered "always allow" is per tool *name*, which means a remembered
/// `bash` approval covers every future command. To keep that meaningful, the
/// auto-allow path still prints the full input before execution.
#[derive(Clone)]
pub struct MemoryPermissionHandler {
    always_allow: Arc<Mutex<HashSet<String>>>,
    always_deny: Arc<Mutex<HashSet<String>>>,
    prompt: Arc<dyn PermissionPrompt>,
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
            prompt: Arc::new(ConsolePermissionPrompt),
        }
    }

    /// Create an empty remembered policy using a custom interactive frontend.
    pub fn with_prompt(prompt: Arc<dyn PermissionPrompt>) -> Self {
        Self {
            always_allow: Arc::new(Mutex::new(HashSet::new())),
            always_deny: Arc::new(Mutex::new(HashSet::new())),
            prompt,
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
            prompt: Arc::new(ConsolePermissionPrompt),
        }
    }

    /// Create a handler sharing both remembered state and a custom frontend.
    pub fn with_shared_state_and_prompt(
        always_allow: Arc<Mutex<HashSet<String>>>,
        always_deny: Arc<Mutex<HashSet<String>>>,
        prompt: Arc<dyn PermissionPrompt>,
    ) -> Self {
        Self {
            always_allow,
            always_deny,
            prompt,
        }
    }

    pub fn always_allow(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_allow)
    }

    pub fn always_deny(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_deny)
    }

    pub fn set_always_allow(&self, tools: HashSet<String>) {
        let mut always_allow = self.always_allow.lock().unwrap();
        let mut always_deny = self.always_deny.lock().unwrap();
        always_deny.retain(|tool| !tools.contains(tool));
        *always_allow = tools;
    }

    pub fn set_always_deny(&self, tools: HashSet<String>) {
        let mut always_allow = self.always_allow.lock().unwrap();
        let mut always_deny = self.always_deny.lock().unwrap();
        always_allow.retain(|tool| !tools.contains(tool));
        *always_deny = tools;
    }

    /// Replace both remembered sets atomically. A deny wins if the supplied
    /// policy contains a contradictory legacy entry.
    pub fn replace_remembered_policy(
        &self,
        mut always_allow: HashSet<String>,
        always_deny: HashSet<String>,
    ) {
        always_allow.retain(|tool| !always_deny.contains(tool));
        let mut current_allow = self.always_allow.lock().unwrap();
        let mut current_deny = self.always_deny.lock().unwrap();
        *current_allow = always_allow;
        *current_deny = always_deny;
    }

    /// Take a consistent, fail-closed snapshot for display or persistence.
    pub fn remembered_policy(&self) -> RememberedPermissionPolicy {
        let current_allow = self.always_allow.lock().unwrap();
        let current_deny = self.always_deny.lock().unwrap();
        let mut always_allow = current_allow.clone();
        always_allow.retain(|tool| !current_deny.contains(tool));
        RememberedPermissionPolicy {
            always_allow,
            always_deny: current_deny.clone(),
        }
    }

    /// Remove any remembered decision for one exact tool name.
    pub fn reset_remembered_tool(&self, tool: &str) -> bool {
        let mut always_allow = self.always_allow.lock().unwrap();
        let mut always_deny = self.always_deny.lock().unwrap();
        always_allow.remove(tool) | always_deny.remove(tool)
    }

    /// Remove every remembered decision and return the number of distinct
    /// affected tools.
    pub fn clear_remembered_policy(&self) -> usize {
        let mut always_allow = self.always_allow.lock().unwrap();
        let mut always_deny = self.always_deny.lock().unwrap();
        let count = always_allow.union(&always_deny).count();
        always_allow.clear();
        always_deny.clear();
        count
    }

    fn remember(&self, tool: &str, allowed: bool) {
        let mut always_allow = self.always_allow.lock().unwrap();
        let mut always_deny = self.always_deny.lock().unwrap();
        if allowed {
            always_deny.remove(tool);
            always_allow.insert(tool.to_string());
        } else {
            always_allow.remove(tool);
            always_deny.insert(tool.to_string());
        }
    }
}

#[async_trait]
impl ToolPermissionHandler for MemoryPermissionHandler {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        {
            let always_deny = self.always_deny.lock().unwrap();
            if always_deny.contains(&request.tool_name) {
                self.prompt.automatic_decision(request, false);
                return PermissionDecision::DenyWithReason(
                    "Tool was previously set to never allow".to_string(),
                );
            }
        }

        {
            let always_allow = self.always_allow.lock().unwrap();
            if always_allow.contains(&request.tool_name) {
                self.prompt.automatic_decision(request, true);
                return PermissionDecision::Allow;
            }
        }

        match self.prompt.choose(request).await {
            PermissionChoice::AllowAlways => {
                self.remember(&request.tool_name, true);
                PermissionDecision::Allow
            }
            PermissionChoice::AllowOnce => PermissionDecision::Allow,
            PermissionChoice::DenyAlways => {
                self.remember(&request.tool_name, false);
                PermissionDecision::DenyWithReason(
                    "User chose to never allow this tool".to_string(),
                )
            }
            PermissionChoice::DenyOnce => PermissionDecision::DenyWithReason(
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

    #[test]
    fn remembered_policy_replacement_reset_and_clear_are_disjoint() {
        let handler = MemoryPermissionHandler::new();
        handler.replace_remembered_policy(
            ["bash".to_string(), "read_file".to_string()].into(),
            ["bash".to_string(), "patch_file".to_string()].into(),
        );

        let policy = handler.remembered_policy();
        assert_eq!(policy.always_allow, ["read_file".to_string()].into());
        assert_eq!(
            policy.always_deny,
            ["bash".to_string(), "patch_file".to_string()].into()
        );
        assert!(handler.reset_remembered_tool("bash"));
        assert!(!handler.reset_remembered_tool("missing"));
        assert_eq!(handler.clear_remembered_policy(), 2);
        assert_eq!(handler.clear_remembered_policy(), 0);
        assert!(handler.remembered_policy().always_allow.is_empty());
        assert!(handler.remembered_policy().always_deny.is_empty());
    }

    #[tokio::test]
    async fn contradictory_shared_policy_denies_fail_closed() {
        let handler = MemoryPermissionHandler::with_shared_state(
            Arc::new(Mutex::new(["bash".to_string()].into())),
            Arc::new(Mutex::new(["bash".to_string()].into())),
        );

        assert!(matches!(
            handler.check_permission(&request("bash")).await,
            PermissionDecision::DenyWithReason(_)
        ));
        let policy = handler.remembered_policy();
        assert!(policy.always_allow.is_empty());
        assert_eq!(policy.always_deny, ["bash".to_string()].into());
    }

    #[tokio::test]
    async fn broker_correlates_the_ui_reply_with_its_request() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let handler =
            MemoryPermissionHandler::with_prompt(Arc::new(PermissionBrokerPrompt::new(sender)));
        let task =
            tokio::spawn(async move { handler.check_permission(&request("patch_file")).await });

        let PermissionUiEvent::Request(pending) =
            receiver.recv().await.expect("permission request")
        else {
            panic!("expected interactive request");
        };
        assert_eq!(pending.id, 1);
        assert_eq!(pending.request.tool_name, "patch_file");
        pending.reply.send(PermissionChoice::AllowOnce).unwrap();
        assert_eq!(task.await.unwrap(), PermissionDecision::Allow);
    }

    #[tokio::test]
    async fn dropped_broker_reply_denies_instead_of_hanging() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let handler =
            MemoryPermissionHandler::with_prompt(Arc::new(PermissionBrokerPrompt::new(sender)));
        let task = tokio::spawn(async move { handler.check_permission(&request("bash")).await });

        let PermissionUiEvent::Request(pending) =
            receiver.recv().await.expect("permission request")
        else {
            panic!("expected interactive request");
        };
        drop(pending.reply);
        assert!(matches!(
            task.await.unwrap(),
            PermissionDecision::DenyWithReason(_)
        ));
    }

    #[tokio::test]
    async fn broker_request_ids_keep_out_of_order_replies_correlated() {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let prompt = Arc::new(PermissionBrokerPrompt::new(sender));
        let first_prompt = Arc::clone(&prompt);
        let first = tokio::spawn(async move { first_prompt.choose(&request("first")).await });
        let second_prompt = Arc::clone(&prompt);
        let second = tokio::spawn(async move { second_prompt.choose(&request("second")).await });

        let PermissionUiEvent::Request(request_a) = receiver.recv().await.expect("first request")
        else {
            panic!("expected interactive request");
        };
        let PermissionUiEvent::Request(request_b) = receiver.recv().await.expect("second request")
        else {
            panic!("expected interactive request");
        };
        assert_ne!(request_a.id, request_b.id);

        let (first_reply, second_reply) = if request_a.request.tool_name == "first" {
            (request_a.reply, request_b.reply)
        } else {
            (request_b.reply, request_a.reply)
        };
        second_reply.send(PermissionChoice::DenyOnce).unwrap();
        first_reply.send(PermissionChoice::AllowOnce).unwrap();

        assert_eq!(first.await.unwrap(), PermissionChoice::AllowOnce);
        assert_eq!(second.await.unwrap(), PermissionChoice::DenyOnce);
    }
}
