//! Tool execution permission handling.

use crate::types::truncate_middle;
use async_trait::async_trait;
use dialoguer::{console::style, theme::ColorfulTheme, Select};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use tokio::sync::{mpsc, oneshot};

/// Tools whose interactive "always allow" is remembered per exact input
/// rather than per name: one prompt-fatigue click on an arbitrary-code tool
/// must never become a standing grant for every future command. Explicit
/// name-level policy (the `/permission` command or a loaded saved state)
/// remains honored as configured.
const EXACT_INPUT_TOOLS: [&str; 2] = ["bash", "python"];

/// Whether an interactive "always allow" for this tool is scoped to the
/// exact input instead of the tool name.
pub fn remembers_exact_input(tool_name: &str) -> bool {
    EXACT_INPUT_TOOLS.contains(&tool_name)
}

/// Session key for one exact (tool, input) grant. `\u{0}` cannot appear in a
/// tool name, so the key is unambiguous.
fn exact_input_key(tool_name: &str, input: &Value) -> String {
    format!("{tool_name}\u{0}{input}")
}

/// Poison-tolerant lock: the guarded sets are plain collections, so a panic
/// in another thread must not cascade into the permission path.
fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

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
            style(line).blue().bright().to_string()
        } else if line.starts_with("@@") {
            style(line).cyan().to_string()
        } else if line.starts_with('+') {
            style(line).green().to_string()
        } else if line.starts_with('-') {
            style(line).red().to_string()
        } else {
            style(line).dim().to_string()
        };
        formatted.push_str(&rendered);
        formatted.push('\n');
    }
    formatted
}

struct ConsolePermissionPrompt;

impl ConsolePermissionPrompt {
    fn print_request(request: &ToolExecutionRequest) {
        println!("\n{}", style("⚠️  Tool Permission Request").yellow().bold());
        println!("{}", style("─".repeat(50)).dim());
        println!("Tool: {}", style(&request.tool_name).cyan().bold());
        println!("Description: {}", style(&request.tool_description).dim());

        // Show diffs as diffs; everything else as pretty JSON.
        let diff = (request.tool_name == "patch_file")
            .then(|| request.input.get("diff").and_then(|v| v.as_str()))
            .flatten();
        if let Some(diff) = diff {
            if let Some(path) = request.input.get("path").and_then(|v| v.as_str()) {
                println!("Target file: {}", style(path).yellow());
            }
            println!("\n{}", style("Proposed changes:").bold());
            println!("{}", style("─".repeat(50)).dim());
            print!("{}", format_diff_for_display(diff));
            println!("{}", style("─".repeat(50)).dim());
        } else {
            println!(
                "Input: {}",
                style(serde_json::to_string_pretty(&request.input).unwrap_or_default()).dim()
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
            Self::choose_blocking(remembers_exact_input(&request.tool_name))
        })
        .await
        .unwrap_or(PermissionChoice::DenyOnce)
    }

    fn automatic_decision(&self, request: &ToolExecutionRequest, allowed: bool) {
        let compact = serde_json::to_string(&request.input).unwrap_or_default();
        if allowed {
            eprintln!(
                "{} Auto-allowing {} {}",
                style("✓").for_stderr().green(),
                style(&request.tool_name).for_stderr().cyan(),
                style(truncate_middle(&compact, 300)).for_stderr().dim()
            );
        } else {
            eprintln!(
                "{} Auto-denying '{}' (previously set to never allow)",
                style("✗").for_stderr().red(),
                style(&request.tool_name).for_stderr().cyan()
            );
        }
    }
}

impl ConsolePermissionPrompt {
    fn choose_blocking(exact_input: bool) -> PermissionChoice {
        let choices = [
            if exact_input {
                "Yes (always allow this exact input, this session)"
            } else {
                "Yes (always allow this tool)"
            },
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

/// Interactive handler with remembered always/never decisions.
///
/// Remembered "always allow" is per tool *name* for ordinary tools; for
/// arbitrary-code tools (see [`remembers_exact_input`]) an interactive
/// approval is remembered per exact input for this session only. The
/// auto-allow path still prints the full input before execution.
#[derive(Clone)]
pub struct MemoryPermissionHandler {
    always_allow: Arc<Mutex<HashSet<String>>>,
    always_deny: Arc<Mutex<HashSet<String>>>,
    /// Session-scoped exact (tool, input) grants; never persisted.
    always_allow_exact: Arc<Mutex<HashSet<String>>>,
    prompt: Arc<dyn PermissionPrompt>,
}

impl Default for MemoryPermissionHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPermissionHandler {
    pub fn new() -> Self {
        Self::with_shared_state_and_prompt(
            Arc::new(Mutex::new(HashSet::new())),
            Arc::new(Mutex::new(HashSet::new())),
            Arc::new(ConsolePermissionPrompt),
        )
    }

    /// Create an empty remembered policy using a custom interactive frontend.
    pub fn with_prompt(prompt: Arc<dyn PermissionPrompt>) -> Self {
        Self::with_shared_state_and_prompt(
            Arc::new(Mutex::new(HashSet::new())),
            Arc::new(Mutex::new(HashSet::new())),
            prompt,
        )
    }

    /// Create a handler sharing decision state with another.
    pub fn with_shared_state(
        always_allow: Arc<Mutex<HashSet<String>>>,
        always_deny: Arc<Mutex<HashSet<String>>>,
    ) -> Self {
        Self::with_shared_state_and_prompt(
            always_allow,
            always_deny,
            Arc::new(ConsolePermissionPrompt),
        )
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
            always_allow_exact: Arc::new(Mutex::new(HashSet::new())),
            prompt,
        }
    }

    /// Lock both name sets in a fixed order (allow, then deny) so every call
    /// site preserves the same deadlock-free ordering.
    fn lock_both(
        &self,
    ) -> (
        MutexGuard<'_, HashSet<String>>,
        MutexGuard<'_, HashSet<String>>,
    ) {
        (locked(&self.always_allow), locked(&self.always_deny))
    }

    pub fn always_allow(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_allow)
    }

    pub fn always_deny(&self) -> Arc<Mutex<HashSet<String>>> {
        Arc::clone(&self.always_deny)
    }

    pub fn set_always_allow(&self, tools: HashSet<String>) {
        let (mut always_allow, mut always_deny) = self.lock_both();
        always_deny.retain(|tool| !tools.contains(tool));
        *always_allow = tools;
    }

    pub fn set_always_deny(&self, tools: HashSet<String>) {
        let (mut always_allow, mut always_deny) = self.lock_both();
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
        let (mut current_allow, mut current_deny) = self.lock_both();
        let mut exact = locked(&self.always_allow_exact);
        *current_allow = always_allow;
        *current_deny = always_deny;
        // Exact grants are deliberately not persisted. Replacing policy from
        // a saved session must not carry hidden executable approvals over
        // from the conversation that was active before the load.
        exact.clear();
    }

    /// Take a consistent, fail-closed snapshot for persistence.
    /// Session-scoped exact-input grants are deliberately excluded.
    pub fn remembered_policy(&self) -> RememberedPermissionPolicy {
        let (current_allow, current_deny) = self.lock_both();
        let mut always_allow = current_allow.clone();
        always_allow.retain(|tool| !current_deny.contains(tool));
        RememberedPermissionPolicy {
            always_allow,
            always_deny: current_deny.clone(),
        }
    }

    /// Count session-only exact-input grants by tool without disclosing the
    /// potentially sensitive executable inputs themselves.
    pub fn session_exact_allow_counts(&self) -> Vec<(String, usize)> {
        let mut counts = BTreeMap::<String, usize>::new();
        for key in locked(&self.always_allow_exact).iter() {
            if let Some((tool, _)) = key.split_once('\0') {
                *counts.entry(tool.to_string()).or_default() += 1;
            }
        }
        counts.into_iter().collect()
    }

    /// Remove any remembered decision for one exact tool name, including its
    /// session-scoped exact-input grants.
    pub fn reset_remembered_tool(&self, tool: &str) -> bool {
        let (mut always_allow, mut always_deny) = self.lock_both();
        let mut exact = locked(&self.always_allow_exact);
        let prefix = format!("{tool}\u{0}");
        let had_exact = exact.iter().any(|key| key.starts_with(&prefix));
        exact.retain(|key| !key.starts_with(&prefix));
        always_allow.remove(tool) | always_deny.remove(tool) | had_exact
    }

    /// Remove every remembered decision and return the number of distinct
    /// affected tools.
    pub fn clear_remembered_policy(&self) -> usize {
        let (mut always_allow, mut always_deny) = self.lock_both();
        let mut exact = locked(&self.always_allow_exact);
        let mut affected = always_allow
            .union(&always_deny)
            .cloned()
            .collect::<HashSet<_>>();
        affected.extend(
            exact
                .iter()
                .filter_map(|key| key.split_once('\0').map(|(tool, _)| tool.to_string())),
        );
        always_allow.clear();
        always_deny.clear();
        exact.clear();
        affected.len()
    }

    fn remember(&self, request: &ToolExecutionRequest, allowed: bool) {
        let tool = request.tool_name.as_str();
        if allowed && remembers_exact_input(tool) {
            // An interactive approval of an arbitrary-code tool covers this
            // exact input only, and only for this session.
            locked(&self.always_allow_exact).insert(exact_input_key(tool, &request.input));
            return;
        }
        let (mut always_allow, mut always_deny) = self.lock_both();
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
            let always_deny = locked(&self.always_deny);
            if always_deny.contains(&request.tool_name) {
                self.prompt.automatic_decision(request, false);
                return PermissionDecision::DenyWithReason(
                    "Tool was previously set to never allow".to_string(),
                );
            }
        }

        {
            let always_allow = locked(&self.always_allow);
            if always_allow.contains(&request.tool_name) {
                self.prompt.automatic_decision(request, true);
                return PermissionDecision::Allow;
            }
        }

        if remembers_exact_input(&request.tool_name)
            && locked(&self.always_allow_exact)
                .contains(&exact_input_key(&request.tool_name, &request.input))
        {
            self.prompt.automatic_decision(request, true);
            return PermissionDecision::Allow;
        }

        match self.prompt.choose(request).await {
            PermissionChoice::AllowAlways => {
                self.remember(request, true);
                PermissionDecision::Allow
            }
            PermissionChoice::AllowOnce => PermissionDecision::Allow,
            PermissionChoice::DenyAlways => {
                self.remember(request, false);
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

    /// Prompt scripted to return a fixed choice, counting invocations.
    struct ScriptedPrompt {
        choice: PermissionChoice,
        prompts: std::sync::atomic::AtomicU64,
        automatic: std::sync::atomic::AtomicU64,
    }

    impl ScriptedPrompt {
        fn new(choice: PermissionChoice) -> Arc<Self> {
            Arc::new(Self {
                choice,
                prompts: 0.into(),
                automatic: 0.into(),
            })
        }
    }

    #[async_trait]
    impl PermissionPrompt for ScriptedPrompt {
        async fn choose(&self, _request: &ToolExecutionRequest) -> PermissionChoice {
            self.prompts.fetch_add(1, Ordering::SeqCst);
            self.choice
        }
        fn automatic_decision(&self, _request: &ToolExecutionRequest, _allowed: bool) {
            self.automatic.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn request_with_input(name: &str, input: Value) -> ToolExecutionRequest {
        ToolExecutionRequest {
            tool_use_id: "id".into(),
            tool_name: name.into(),
            input,
            tool_description: "desc".into(),
        }
    }

    #[tokio::test]
    async fn interactive_always_allow_persists_and_undenies_ordinary_tools() {
        let prompt = ScriptedPrompt::new(PermissionChoice::AllowAlways);
        let handler = MemoryPermissionHandler::with_prompt(Arc::clone(&prompt) as _);
        handler.set_always_deny(["read_file".to_string()].into());
        // First deny is remembered; resetting clears it so the prompt runs.
        assert!(matches!(
            handler.check_permission(&request("read_file")).await,
            PermissionDecision::DenyWithReason(_)
        ));
        assert!(handler.reset_remembered_tool("read_file"));

        assert_eq!(
            handler.check_permission(&request("read_file")).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 1);
        // Remembered: the second identical call never prompts.
        assert_eq!(
            handler.check_permission(&request("read_file")).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 1);
        assert!(handler
            .remembered_policy()
            .always_allow
            .contains("read_file"));
    }

    #[tokio::test]
    async fn interactive_always_allow_on_exec_tools_covers_only_the_exact_input() {
        let prompt = ScriptedPrompt::new(PermissionChoice::AllowAlways);
        let handler = MemoryPermissionHandler::with_prompt(Arc::clone(&prompt) as _);

        let ls = request_with_input("bash", json!({"command": "ls"}));
        assert_eq!(
            handler.check_permission(&ls).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 1);

        // The identical command is auto-allowed without another prompt...
        assert_eq!(
            handler.check_permission(&ls).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 1);
        assert_eq!(prompt.automatic.load(Ordering::SeqCst), 1);

        // ...but a different command prompts again.
        let rm = request_with_input("bash", json!({"command": "rm -rf /"}));
        assert_eq!(
            handler.check_permission(&rm).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 2);

        // Nothing leaks into the persisted name-level policy.
        assert!(handler.remembered_policy().always_allow.is_empty());
        // Resetting the tool clears the session grants.
        assert!(handler.reset_remembered_tool("bash"));
        assert_eq!(
            handler.check_permission(&ls).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn clearing_exact_input_grants_reports_and_revokes_the_tool() {
        let prompt = ScriptedPrompt::new(PermissionChoice::AllowAlways);
        let handler = MemoryPermissionHandler::with_prompt(Arc::clone(&prompt) as _);
        let command = request_with_input("bash", json!({"command": "printf safe"}));

        assert_eq!(
            handler.check_permission(&command).await,
            PermissionDecision::Allow
        );
        assert_eq!(handler.clear_remembered_policy(), 1);
        assert_eq!(
            handler.check_permission(&command).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn replacing_loaded_policy_clears_session_exact_input_grants() {
        let prompt = ScriptedPrompt::new(PermissionChoice::AllowAlways);
        let handler = MemoryPermissionHandler::with_prompt(Arc::clone(&prompt) as _);
        let command = request_with_input("bash", json!({"command": "printf safe"}));

        assert_eq!(
            handler.check_permission(&command).await,
            PermissionDecision::Allow
        );
        assert_eq!(
            handler.session_exact_allow_counts(),
            vec![("bash".to_string(), 1)]
        );
        handler.replace_remembered_policy(HashSet::new(), HashSet::new());
        assert!(handler.session_exact_allow_counts().is_empty());

        assert_eq!(
            handler.check_permission(&command).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn explicit_name_level_exec_allow_is_still_honored() {
        // A deliberate `/permission`-style grant is configuration, not a
        // prompt-fatigue click, and keeps name-level semantics.
        let prompt = ScriptedPrompt::new(PermissionChoice::DenyOnce);
        let handler = MemoryPermissionHandler::with_prompt(Arc::clone(&prompt) as _);
        handler.set_always_allow(["bash".to_string()].into());
        let ls = request_with_input("bash", json!({"command": "ls"}));
        assert_eq!(
            handler.check_permission(&ls).await,
            PermissionDecision::Allow
        );
        assert_eq!(prompt.prompts.load(Ordering::SeqCst), 0);
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
