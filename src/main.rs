use dialoguer::console::style;
use generalist::mcp::{McpConfig, McpRegistrationOutcome, McpRegistrationReport};
use generalist::provider::{
    anthropic, openrouter, AnthropicProvider, OpenAiProvider, OpenRouterProvider, Provider,
};
use generalist::tools::*;
use generalist::tui::{TerminalUi, UiAction};
use generalist::{
    conversation_transcript, history_tool_protocol_is_valid, is_local_command,
    latest_assistant_reasoning, latest_assistant_text, parse_local_command, truncate_middle, Agent,
    AgentEvent, CopyCommand, DeliveryMode, EpisodeOutcome, EpisodicMemory, Error, GoalCommand,
    HistoryCommand, HistoryStore, LocalCommand, McpCommand, MemoryCommand, MemoryEvent,
    MemoryPermissionHandler, MessageOrigin, PermissionBrokerPrompt, PermissionChoice,
    PermissionRequest, PermissionUiEvent, ProfilePaths, PromptQueue, PromptSource, Result,
    SavedState, ToolRegistry, TurnControl, TurnOutcome, UsageCommand, WorkspaceScope,
};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

mod commands;
mod display_channel;

use commands::*;
use display_channel::*;

const AUTOSAVE_NAME: &str = "autosave";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
const DEFAULT_LOCAL_MODEL: &str = "qwen3.8:latest";
const THIRD_PARTY_LICENSES: &str = include_str!("../THIRD_PARTY_LICENSES.txt");

fn write_atomically(path: &std::path::Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Other(format!("{} has no parent directory", path.display())))?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".generalist-state-")
        .tempfile_in(parent)
        .map_err(|error| Error::Other(format!("Failed to create state file: {error}")))?;
    temporary
        .write_all(contents)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| Error::Other(format!("Failed to flush state file: {error}")))?;
    temporary.persist(path).map_err(|error| {
        Error::Other(format!(
            "Failed to replace {}: {}",
            path.display(),
            error.error
        ))
    })?;
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| Error::Other(format!("Failed to flush {}: {error}", parent.display())))
}

struct ApiKeys {
    anthropic: Option<String>,
    openai: Option<String>,
    openrouter: Option<String>,
    openai_base_url: String,
}

impl ApiKeys {
    fn from_env() -> Self {
        let openai_base_url = env::var("OPENAI_BASE_URL").ok();
        Self {
            anthropic: env::var("ANTHROPIC_API_KEY")
                .or_else(|_| env::var("CLAUDE_API_KEY"))
                .ok(),
            // Local OpenAI-compatible servers generally do not inspect the
            // key, so a configured base URL is sufficient to enable them.
            openai: env::var("OPENAI_API_KEY")
                .ok()
                .or_else(|| openai_base_url.as_ref().map(|_| "unused".to_string())),
            openrouter: env::var("OPENROUTER_API_KEY").ok(),
            openai_base_url: openai_base_url
                .unwrap_or_else(|| generalist::provider::openai::DEFAULT_BASE_URL.to_string()),
        }
    }

    fn available_providers(&self) -> Vec<&'static str> {
        let mut providers = Vec::new();
        if self.anthropic.is_some() {
            providers.push("anthropic");
        }
        if self.openai.is_some() {
            providers.push("openai");
        }
        if self.openrouter.is_some() {
            providers.insert(0, "openrouter");
        }
        providers
    }

    fn provider_label(&self, provider: &str) -> String {
        match provider {
            "anthropic" => "Anthropic".to_string(),
            "openrouter" => "OpenRouter".to_string(),
            "openai" if self.openai_base_url == generalist::provider::openai::DEFAULT_BASE_URL => {
                "OpenAI".to_string()
            }
            "openai" => "OpenAI-compatible".to_string(),
            other => other.to_string(),
        }
    }
}

fn build_provider(keys: &ApiKeys, provider: &str, model: String) -> Result<Box<dyn Provider>> {
    match provider {
        "anthropic" => {
            let key = keys
                .anthropic
                .clone()
                .ok_or_else(|| Error::Other("ANTHROPIC_API_KEY is not set".to_string()))?;
            Ok(Box::new(AnthropicProvider::new(key, model)?))
        }
        "openai" => {
            let key = keys
                .openai
                .clone()
                .ok_or_else(|| Error::Other("OPENAI_API_KEY is not set".to_string()))?;
            Ok(Box::new(OpenAiProvider::new(
                key,
                keys.openai_base_url.clone(),
                model,
            )?))
        }
        "openrouter" => {
            let key = keys
                .openrouter
                .clone()
                .ok_or_else(|| Error::Other("OPENROUTER_API_KEY is not set".to_string()))?;
            Ok(Box::new(OpenRouterProvider::new(key, model)?))
        }
        other => Err(Error::Other(format!("Unknown provider '{other}'"))),
    }
}

fn default_remote_provider_and_model(keys: &ApiKeys) -> Option<(String, String)> {
    keys.openrouter.as_ref().map(|_| {
        (
            "openrouter".to_string(),
            openrouter::DEFAULT_MODEL.to_string(),
        )
    })
}

fn build_registry(
    todo_file_path: &Path,
    permission_handler: &MemoryPermissionHandler,
    history_store: &HistoryStore,
    memory: Option<&EpisodicMemory>,
) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::with_permission_handler(Box::new(permission_handler.clone()));
    registry.register(Arc::new(ReadFileTool))?;
    registry.register(Arc::new(PatchFileTool))?;
    registry.register(Arc::new(ListDirectoryTool))?;
    registry.register(Arc::new(BashTool))?;
    registry.register(Arc::new(WeatherTool))?;
    registry.register(Arc::new(HttpFetchTool))?;
    registry.register(Arc::new(WikipediaTool))?;
    registry.register(Arc::new(Z3SolverTool))?;
    registry.register(Arc::new(TodoTool::new(todo_file_path)))?;
    registry.register(Arc::new(FirecrawlCrawlTool))?;
    registry.register(Arc::new(FirecrawlSearchTool))?;
    registry.register(Arc::new(FirecrawlMapTool))?;
    registry.register(Arc::new(FirecrawlExtractTool))?;
    registry.register(Arc::new(SearchConversationsTool::new(
        history_store.clone(),
    )))?;
    registry.register(Arc::new(ReadConversationTool::new(history_store.clone())))?;
    if let Some(memory) = memory {
        registry.register(Arc::new(SearchMemoriesTool::new(memory.clone())))?;
        registry.register(Arc::new(ReadMemoryTool::new(memory.clone())))?;
    }
    Ok(registry)
}

fn make_saved_state(
    agent: &Agent,
    handler: &MemoryPermissionHandler,
    queue: &PromptQueue,
    scope: &WorkspaceScope,
) -> SavedState {
    let policy = handler.remembered_policy();
    SavedState {
        scope: scope.clone(),
        provider: agent.provider().id().to_string(),
        model: agent.provider().model().to_string(),
        goal: agent.goal().map(str::to_string),
        conversation_history: agent.history().to_vec(),
        always_allow_tools: policy.always_allow,
        always_deny_tools: policy.always_deny,
        queued_prompts: queue.snapshot(),
    }
}

async fn save_named_session(
    name: &str,
    agent: &Agent,
    permission_handler: &MemoryPermissionHandler,
    queue: &PromptQueue,
    history_store: &HistoryStore,
) -> Result<PathBuf> {
    let state = make_saved_state(agent, permission_handler, queue, history_store.scope());
    let name = name.to_string();
    with_history_store(history_store, move |store| store.save(&state, &name)).await?
}

async fn save_new_named_session(
    name: &str,
    agent: &Agent,
    permission_handler: &MemoryPermissionHandler,
    queue: &PromptQueue,
    history_store: &HistoryStore,
) -> Result<Option<PathBuf>> {
    let state = make_saved_state(agent, permission_handler, queue, history_store.scope());
    let name = name.to_string();
    with_history_store(history_store, move |store| {
        store.save_if_absent(&state, &name)
    })
    .await?
}

/// The `/save` flow: resolve a name (prompting when absent), refuse the
/// reserved autosave name, and confirm before replacing an existing save.
async fn run_save_command(
    ctx: &mut ReactorCtx<'_>,
    agent: &Agent,
    name: Option<&str>,
) -> Result<()> {
    let prompted_name = if name.is_none() {
        let default = format!("chat_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
        terminal(ctx.ui.prompt("Save conversation as", &default).await)?
    } else {
        None
    };
    let Some(name) = name
        .map(str::to_string)
        .or(prompted_name)
        .map(|name| name.trim().to_string())
    else {
        return Ok(());
    };
    if name == AUTOSAVE_NAME {
        ctx.ui.error(
            "The live autosave name is reserved; choose another name for a durable checkpoint.",
        );
        return Ok(());
    }
    match save_new_named_session(
        &name,
        agent,
        ctx.permission_handler,
        ctx.queue,
        ctx.history_store,
    )
    .await
    {
        Ok(Some(path)) => {
            ctx.ui
                .info(&format!("Saved session '{name}' to {}", path.display()));
            return Ok(());
        }
        Err(error) => {
            ctx.ui.error(&format!("Failed to save '{name}': {error}"));
            return Ok(());
        }
        Ok(None) => {}
    }

    // The name is taken: confirm replacement of a valid current-scope save.
    let requested = name.clone();
    let existing = with_history_store(ctx.history_store, move |store| {
        store.inspect_current_archive(&requested)
    })
    .await
    .and_then(|result| result);
    match existing {
        Err(error) => ctx
            .ui
            .error(&format!("Saved-session replacement refused: {error}")),
        Ok(None) => ctx.ui.error(&format!(
            "A path for saved session '{name}' already exists but is not a valid current-scope \
             save; refusing to overwrite it."
        )),
        Ok(Some(_)) => {
            let choices = vec!["Cancel".to_string(), format!("Replace '{name}'")];
            let title = format!(
                "Replace saved session '{name}'? The prior checkpoint will be overwritten."
            );
            if terminal(ctx.ui.select(&title, &choices).await)? == Some(1) {
                match save_named_session(
                    &name,
                    agent,
                    ctx.permission_handler,
                    ctx.queue,
                    ctx.history_store,
                )
                .await
                {
                    Ok(path) => ctx.ui.info(&format!(
                        "Replaced saved session '{name}' at {}",
                        path.display()
                    )),
                    Err(error) => ctx.ui.error(&format!(
                        "Failed to replace saved session '{name}': {error}"
                    )),
                }
            } else {
                ctx.ui
                    .info(&format!("Kept existing saved session '{name}'."));
            }
        }
    }
    Ok(())
}

async fn load_named_session(
    ctx: &mut ReactorCtx<'_>,
    name: &str,
    agent: &mut Agent,
    keys: &ApiKeys,
) -> Result<usize> {
    let requested = name.to_string();
    let SavedState {
        scope: _,
        provider,
        model,
        goal,
        conversation_history,
        always_allow_tools,
        always_deny_tools,
        queued_prompts,
    } = with_history_store(ctx.history_store, move |store| store.load(&requested)).await??;
    if !history_tool_protocol_is_valid(&conversation_history) {
        return Err(Error::Other(
            "Saved conversation has an unpaired tool use/result; refusing to load it".to_string(),
        ));
    }
    match build_provider(keys, &provider, model) {
        Ok(provider) => agent.set_provider(provider),
        Err(error) => ctx.ui.error(&format!(
            "Saved API '{provider}' is unavailable ({error}); keeping the current API."
        )),
    }
    ctx.permission_handler
        .replace_remembered_policy(always_allow_tools, always_deny_tools);
    agent.set_goal(goal);
    agent.replace_history(conversation_history);
    ctx.queue.replace(queued_prompts);
    ctx.queue
        .reconcile_goal_continuation(agent.goal().is_some());
    ctx.ui.load_history(agent.history());
    ctx.ui.set_goal(agent.goal());
    ctx.ui.sync_queue(ctx.queue);
    ctx.ui.set_session(
        agent.provider().display_name(),
        agent.provider().model(),
        agent.registry.tool_names().len(),
    );
    ctx.ui.set_context_tokens(agent.context_tokens());
    Ok(agent.history().len())
}

struct DurableBoundary {
    provider: String,
    model: String,
    goal: Option<String>,
    history: Vec<generalist::Message>,
}

impl DurableBoundary {
    fn from_agent(agent: &Agent) -> Self {
        Self {
            provider: agent.provider().id().to_string(),
            model: agent.provider().model().to_string(),
            goal: agent.goal().map(str::to_string),
            history: agent.history().to_vec(),
        }
    }

    fn before_agent(
        provider: &dyn Provider,
        goal: Option<String>,
        history: Vec<generalist::Message>,
    ) -> Self {
        Self {
            provider: provider.id().to_string(),
            model: provider.model().to_string(),
            goal,
            history,
        }
    }

    /// The complete autosave state as of this boundary.
    fn snapshot(
        &self,
        history_store: &HistoryStore,
        permission_handler: &MemoryPermissionHandler,
        queue: &PromptQueue,
    ) -> SavedState {
        let policy = permission_handler.remembered_policy();
        SavedState {
            scope: history_store.scope().clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
            goal: self.goal.clone(),
            conversation_history: self.history.clone(),
            always_allow_tools: policy.always_allow,
            always_deny_tools: policy.always_deny,
            queued_prompts: queue.snapshot(),
        }
    }
}

enum AutosaveMessage {
    Save(Box<SavedState>),
    Flush(std::sync::mpsc::SyncSender<()>),
}

/// Writes autosave snapshots on a dedicated thread so their fsyncs never
/// stall the single-threaded UI reactor.
///
/// Saves are coalesced: when the writer falls behind, only the newest queued
/// snapshot is written — each snapshot is the complete autosave state, so a
/// later one strictly supersedes an earlier one. Write failures surface
/// through [`AutosaveWriter::drain_errors`] at the next reactor tick.
struct AutosaveWriter {
    sender: std::sync::mpsc::Sender<AutosaveMessage>,
    errors: Arc<std::sync::Mutex<Vec<String>>>,
}

impl AutosaveWriter {
    fn spawn(store: HistoryStore) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<AutosaveMessage>();
        let errors: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let worker_errors = Arc::clone(&errors);
        let spawned = std::thread::Builder::new()
            .name("generalist-autosave".to_string())
            .spawn(move || {
                while let Ok(message) = receiver.recv() {
                    let mut latest = None;
                    let mut flushes = Vec::new();
                    let mut pending = Some(message);
                    while let Some(message) = pending.take() {
                        match message {
                            AutosaveMessage::Save(state) => latest = Some(state),
                            AutosaveMessage::Flush(done) => flushes.push(done),
                        }
                        pending = receiver.try_recv().ok();
                    }
                    if let Some(state) = latest {
                        if let Err(error) = store.save(&state, AUTOSAVE_NAME) {
                            worker_errors
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .push(format!("Failed to persist autosave: {error}"));
                        }
                    }
                    for done in flushes {
                        let _ = done.send(());
                    }
                }
            });
        if let Err(error) = spawned {
            errors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(format!(
                    "Autosave writer failed to start; conversation autosave is disabled: {error}"
                ));
        }
        Self { sender, errors }
    }

    /// Queue the newest snapshot; never blocks the reactor.
    fn save(&self, state: SavedState) {
        if self
            .sender
            .send(AutosaveMessage::Save(Box::new(state)))
            .is_err()
        {
            let message = "Autosave writer stopped; conversation autosave is disabled.";
            let mut errors = self
                .errors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !errors.iter().any(|error| error == message) {
                errors.push(message.to_string());
            }
        }
    }

    /// Surface any accumulated writer errors through the UI.
    fn drain_errors(&self, ui: &mut TerminalUi) {
        for error in self.take_errors() {
            ui.error(&error);
        }
    }

    fn take_errors(&self) -> Vec<String> {
        let mut errors = self
            .errors
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        errors.drain(..).collect()
    }

    /// Wait until every snapshot queued before this call reached disk.
    fn flush(&self) -> Result<()> {
        let (done, wait) = std::sync::mpsc::sync_channel(1);
        if self.sender.send(AutosaveMessage::Flush(done)).is_err() {
            let mut errors = self.take_errors();
            errors.push("Autosave writer stopped before shutdown flush.".to_string());
            return Err(Error::Other(errors.join("\n")));
        }
        match wait.recv_timeout(Duration::from_secs(30)) {
            Ok(()) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                return Err(Error::Other(
                    "Timed out waiting for the autosave shutdown flush.".to_string(),
                ));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(Error::Other(
                    "Autosave writer stopped during the shutdown flush.".to_string(),
                ));
            }
        }
        let errors = self.take_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Other(errors.join("\n")))
        }
    }
}

/// Deny a stale interactive permission request and surface automatic
/// decisions. Used by every reactor that has no active turn to route
/// permission requests to.
fn handle_stale_permission_event(ui: &mut TerminalUi, event: PermissionUiEvent) {
    match event {
        PermissionUiEvent::Request(request) => {
            let _ = request.reply.send(PermissionChoice::DenyOnce);
            ui.error("Ignored a stale permission request with no active turn.");
        }
        PermissionUiEvent::Automatic { request, allowed } => {
            note_automatic_permission(ui, &request.tool_name, allowed);
        }
    }
}

/// One shared wording for "a remembered policy decided this without a
/// prompt", so every reactor reports it identically.
fn note_automatic_permission(ui: &mut TerminalUi, tool_name: &str, allowed: bool) {
    ui.status(&format!(
        "{} was {} by remembered policy",
        tool_name,
        if allowed {
            "auto-allowed"
        } else {
            "auto-denied"
        }
    ));
}

/// The shared 50ms reactor tick: surface background autosave errors, advance
/// animations, and redraw when dirty.
fn reactor_tick(ui: &mut TerminalUi, autosave: &AutosaveWriter) -> Result<()> {
    autosave.drain_errors(ui);
    ui.tick();
    ui.draw_if_dirty()
        .map_err(|error| Error::Other(error.to_string()))
}

/// The stable bundle of collaborators threaded through every reactor loop.
///
/// Every drive loop destructures this at entry (tokio::select! needs
/// disjoint field borrows), so the struct's job is keeping the loop
/// signatures uniform and small.
struct ReactorCtx<'a> {
    ui: &'a mut TerminalUi,
    queue: &'a PromptQueue,
    history_store: &'a HistoryStore,
    permission_handler: &'a MemoryPermissionHandler,
    permission_rx: &'a mut mpsc::UnboundedReceiver<PermissionUiEvent>,
    memory_events: &'a mut mpsc::UnboundedReceiver<MemoryEvent>,
    memory: Option<&'a EpisodicMemory>,
    autosave: &'a AutosaveWriter,
}

struct StartupRecoveryPlan {
    goal: Option<String>,
    history: Vec<generalist::Message>,
    always_allow_tools: std::collections::HashSet<String>,
    always_deny_tools: std::collections::HashSet<String>,
    queued_prompts: Vec<generalist::runtime::QueuedPrompt>,
    invalid_queue: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum McpServerState {
    Configured,
    Connecting,
    Connected {
        discovered_tools: usize,
        registered_tools: usize,
    },
    Failed(String),
    Skipped,
}

#[derive(Debug, Clone)]
struct McpRuntime {
    config: McpConfig,
    servers: BTreeMap<String, McpServerState>,
}

impl McpRuntime {
    fn new(config: McpConfig) -> Self {
        let servers = config
            .servers
            .keys()
            .map(|name| (name.clone(), McpServerState::Configured))
            .collect();
        Self { config, servers }
    }

    fn configured_targets(&self) -> BTreeSet<String> {
        self.servers.keys().cloned().collect()
    }

    fn retry_targets(&self, requested: Option<&str>) -> Result<BTreeSet<String>> {
        if let Some(name) = requested {
            let Some(state) = self.servers.get(name) else {
                return Err(Error::Other(format!(
                    "No configured MCP server named '{name}'. Use /mcp status."
                )));
            };
            return match state {
                McpServerState::Failed(_) | McpServerState::Skipped => {
                    Ok([name.to_string()].into_iter().collect())
                }
                McpServerState::Connected { .. } => Err(Error::Other(format!(
                    "MCP server '{name}' is already connected."
                ))),
                McpServerState::Configured | McpServerState::Connecting => Err(Error::Other(
                    format!("MCP server '{name}' has not finished its current connection attempt."),
                )),
            };
        }

        let targets = self
            .servers
            .iter()
            .filter_map(|(name, state)| match state {
                McpServerState::Failed(_) | McpServerState::Skipped => Some(name.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if targets.is_empty() {
            Err(Error::Other(
                "No failed or skipped MCP servers are available to retry.".to_string(),
            ))
        } else {
            Ok(targets)
        }
    }

    fn mark_connecting(&mut self, targets: &BTreeSet<String>) {
        for name in targets {
            if let Some(state) = self.servers.get_mut(name) {
                *state = McpServerState::Connecting;
            }
        }
    }

    fn apply_report(&mut self, report: &McpRegistrationReport) {
        let state = match &report.outcome {
            McpRegistrationOutcome::Connected {
                discovered_tools,
                registered_tools,
            } => McpServerState::Connected {
                discovered_tools: *discovered_tools,
                registered_tools: *registered_tools,
            },
            McpRegistrationOutcome::ConnectionFailed { error } => {
                McpServerState::Failed(format!("connection: {error}"))
            }
            McpRegistrationOutcome::ToolListFailed { error } => {
                McpServerState::Failed(format!("tools/list: {error}"))
            }
            McpRegistrationOutcome::RegistrationFailed {
                discovered_tools,
                error,
            } => McpServerState::Failed(format!(
                "registered 0/{discovered_tools} discovered tool(s): {error}"
            )),
        };
        self.servers.insert(report.server_name.clone(), state);
    }

    fn mark_skipped(&mut self, targets: &BTreeSet<String>) {
        for name in targets {
            if matches!(self.servers.get(name), Some(McpServerState::Connecting)) {
                self.servers.insert(name.clone(), McpServerState::Skipped);
            }
        }
    }

    fn status(&self) -> String {
        if self.servers.is_empty() {
            return "MCP: configuration contains no servers.".to_string();
        }
        let connected = self
            .servers
            .values()
            .filter(|state| matches!(state, McpServerState::Connected { .. }))
            .count();
        let mut lines = vec![format!(
            "MCP servers: {connected}/{} connected",
            self.servers.len()
        )];
        for (name, state) in &self.servers {
            let detail = match state {
                McpServerState::Configured => "configured · not attempted".to_string(),
                McpServerState::Connecting => "connecting".to_string(),
                McpServerState::Connected {
                    discovered_tools,
                    registered_tools,
                } => {
                    format!("connected · {registered_tools}/{discovered_tools} tool(s) registered")
                }
                McpServerState::Failed(error) => {
                    format!("failed · {}", truncate_middle(error, 300))
                }
                McpServerState::Skipped => "skipped · retry available".to_string(),
            };
            lines.push(format!("- {name}: {detail}"));
        }
        lines.push("Use /mcp retry [server] for failed or skipped servers.".to_string());
        lines.join("\n")
    }
}

/// Classify exactly the state that the existing crash-recovery policy admits.
/// A goal is independent of queued work and always survives. Conversation
/// history and remembered permissions are restored only with a non-empty,
/// protocol-valid queue; otherwise startup begins a fresh conversation.
fn plan_startup_recovery(state: Option<SavedState>) -> StartupRecoveryPlan {
    let Some(state) = state else {
        return StartupRecoveryPlan {
            goal: None,
            history: Vec::new(),
            always_allow_tools: Default::default(),
            always_deny_tools: Default::default(),
            queued_prompts: Vec::new(),
            invalid_queue: false,
        };
    };
    let SavedState {
        scope: _,
        provider: _,
        model: _,
        goal,
        conversation_history,
        always_allow_tools,
        always_deny_tools,
        queued_prompts,
    } = state;
    if queued_prompts.is_empty() {
        return StartupRecoveryPlan {
            goal,
            history: Vec::new(),
            always_allow_tools: Default::default(),
            always_deny_tools: Default::default(),
            queued_prompts,
            invalid_queue: false,
        };
    }
    if !history_tool_protocol_is_valid(&conversation_history) {
        return StartupRecoveryPlan {
            goal,
            history: Vec::new(),
            always_allow_tools: Default::default(),
            always_deny_tools: Default::default(),
            queued_prompts: Vec::new(),
            invalid_queue: true,
        };
    }
    StartupRecoveryPlan {
        goal,
        history: conversation_history,
        always_allow_tools,
        always_deny_tools,
        queued_prompts,
        invalid_queue: false,
    }
}

fn recover_startup_runtime(
    history_store: &HistoryStore,
    permission_handler: &MemoryPermissionHandler,
    ui: &mut TerminalUi,
) -> (Option<String>, Vec<generalist::Message>, PromptQueue) {
    let plan = plan_startup_recovery(history_store.load(AUTOSAVE_NAME).ok());
    if plan.invalid_queue {
        ui.error("Autosave has an unpaired tool use/result; queued work was not recovered.");
    }
    let count = plan.queued_prompts.len();
    if count > 0 {
        permission_handler
            .replace_remembered_policy(plan.always_allow_tools, plan.always_deny_tools);
        ui.load_history(&plan.history);
        ui.info(&format!(
            "Recovered {count} queued message(s) with their conversation context."
        ));
    }
    (
        plan.goal,
        plan.history,
        PromptQueue::from_saved(plan.queued_prompts),
    )
}

async fn drive_mcp_discovery(
    ctx: &mut ReactorCtx<'_>,
    registry: &mut ToolRegistry,
    config: &McpConfig,
    runtime: &mut McpRuntime,
    targets: &BTreeSet<String>,
    durable: &DurableBoundary,
) -> Result<bool> {
    let ReactorCtx {
        ui,
        queue,
        history_store,
        permission_handler,
        permission_rx,
        memory_events,
        autosave,
        ..
    } = ctx;
    let ui: &mut TerminalUi = ui;
    runtime.mark_connecting(targets);
    let base_bridge_count = registry.tool_names().len();
    ui.set_bridge_count(base_bridge_count);
    ui.set_busy(true, "Connecting tools · input stays live");
    ui.info(
        "MCP discovery is running. Compose or manage queued work now; Esc skips remaining servers.",
    );
    ui.status("Connecting tools · input stays live");
    terminal(ui.draw())?;

    let (progress_tx, mut progress_rx) =
        mpsc::unbounded_channel::<(McpRegistrationReport, usize)>();
    let (interrupted, exit_requested, completed_reports) = {
        let discovery = generalist::mcp::register_named_servers_with_reports(
            registry,
            config,
            targets,
            move |report, registered_total| {
                let _ = progress_tx.send((report.clone(), registered_total));
            },
        );
        tokio::pin!(discovery);
        let mut ticker = tokio::time::interval(Duration::from_millis(50));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                Some((report, registered_total)) = progress_rx.recv() => {
                    runtime.apply_report(&report);
                    ui.set_bridge_count(base_bridge_count + registered_total);
                    ui.info(&report.display_line());
                    ui.status("Connecting tools · input stays live");
                }
                reports = &mut discovery => break (false, false, Some(reports)),
                Some(permission_event) = permission_rx.recv() => {
                    handle_stale_permission_event(ui, permission_event);
                }
                Some(event) = memory_events.recv() => handle_memory_event(ui, event),
                _ = ticker.tick() => reactor_tick(ui, autosave)?,
                terminal_event = ui.next_event() => {
                    let action = terminal(ui.handle_event(terminal(terminal_event)?, queue))?;
                    let persist_queue = action.requires_queue_persist();
                    match action {
                        UiAction::Submit { text, delivery } => {
                            enqueue_submission(ui, queue, text, delivery, false);
                        }
                        UiAction::Interrupt => break (true, false, None),
                        UiAction::Exit => break (true, true, None),
                        UiAction::None
                        | UiAction::QueueChanged
                        | UiAction::Permission { .. } => {}
                    }
                    if persist_queue {
                        autosave.save(durable.snapshot(history_store, permission_handler, queue));
                    }
                }
            }
        }
    };

    while let Ok((report, registered_total)) = progress_rx.try_recv() {
        runtime.apply_report(&report);
        ui.set_bridge_count(base_bridge_count + registered_total);
        ui.info(&report.display_line());
    }
    if let Some(reports) = completed_reports {
        for report in reports {
            runtime.apply_report(&report);
        }
    }
    if interrupted {
        runtime.mark_skipped(targets);
    }

    // The discovery future no longer owns the registry here, so this is the
    // authoritative count even if cancellation retained only earlier servers.
    ui.set_bridge_count(registry.tool_names().len());
    if interrupted && !exit_requested {
        ui.info(
            "MCP discovery skipped; tools connected before the interrupt remain available. Use /mcp retry to reconnect skipped servers.",
        );
    }
    ui.set_busy(
        false,
        if interrupted {
            "Discovery skipped"
        } else {
            "Ready"
        },
    );
    terminal(ui.draw())?;
    Ok(exit_requested)
}

/// After a driven operation settles, deliver every event still queued in
/// the checkpoint and display channels before touching the agent again.
fn drain_pending_agent_events(
    checkpoint_rx: &mut mpsc::UnboundedReceiver<AgentEvent>,
    event_rx: &AgentDisplayReceiver,
    mut apply: impl FnMut(AgentEvent),
) {
    while let Ok(event) = checkpoint_rx.try_recv() {
        apply(event);
    }
    while let Some(events) = event_rx.try_recv_batch() {
        for event in events {
            apply(event);
        }
    }
}

fn apply_runtime_event(
    ui: &mut TerminalUi,
    queue: &PromptQueue,
    history_store: &HistoryStore,
    permission_handler: &MemoryPermissionHandler,
    autosave: &AutosaveWriter,
    durable: &mut DurableBoundary,
    event: AgentEvent,
) {
    let steering = matches!(&event, AgentEvent::SteeringCommitted { .. });
    match event {
        AgentEvent::HistoryCheckpoint {
            history,
            goal,
            context_tokens,
        } => {
            ui.set_context_tokens(context_tokens);
            durable.history = history;
            durable.goal = goal;
            autosave.save(durable.snapshot(history_store, permission_handler, queue));
        }
        event => ui.handle_agent_event(event),
    }
    if steering {
        ui.sync_queue(queue);
    }
}

struct CliArgs {
    local_model: Option<String>,
    gemini: bool,
    global_scope: bool,
    max_tokens: Option<u32>,
    show_licenses: bool,
}

fn print_usage() {
    println!(
        "Usage: generalist [--global] [--gemini] [--local [model]] [--max-tokens <count>] [--licenses]"
    );
    println!();
    println!("  --global          Use the explicit cross-project history/memory scope");
    println!("                    (default: project scope discovered from the working directory)");
    println!("  --gemini          Use Gemini 3.7 Flash through OpenRouter");
    println!("                    (requires OPENROUTER_API_KEY; --local takes precedence)");
    println!("  --local [model]   Run against a local OpenAI-compatible server");
    println!(
        "                    (default {}, override with OPENAI_BASE_URL).",
        OLLAMA_BASE_URL
    );
    println!(
        "                    Model defaults to {} if omitted.",
        DEFAULT_LOCAL_MODEL
    );
    println!("  --max-tokens N    Request at most N output tokens per ordinary completion");
    println!("                    (default: Anthropic model maximum; provider default elsewhere)");
    println!("  --licenses        Show third-party licenses and exit");
    println!("  -h, --help        Show this help");
}

fn parse_max_tokens(value: &str) -> std::result::Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("invalid --max-tokens value '{value}'"))?;
    if parsed == 0 {
        return Err("--max-tokens must be greater than zero".to_string());
    }
    Ok(parsed)
}

fn parse_args_from(args: impl IntoIterator<Item = String>) -> CliArgs {
    let mut local_model = None;
    let mut gemini = false;
    let mut global_scope = false;
    let mut max_tokens = None;
    let mut show_licenses = false;
    let mut args = args.into_iter().peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--global" => global_scope = true,
            "--gemini" => gemini = true,
            "--licenses" => show_licenses = true,
            "--local" => {
                let model = match args.peek() {
                    Some(next) if !next.starts_with('-') => args.next().unwrap(),
                    _ => DEFAULT_LOCAL_MODEL.to_string(),
                };
                local_model = Some(model);
            }
            value if value.starts_with("--local=") => {
                local_model = Some(value["--local=".len()..].to_string());
            }
            "--max-tokens" => {
                let Some(value) = args.next() else {
                    eprintln!(
                        "{} missing value",
                        style("Invalid argument:").for_stderr().red()
                    );
                    print_usage();
                    std::process::exit(1);
                };
                max_tokens = Some(parse_max_tokens(&value).unwrap_or_else(|error| {
                    eprintln!("{} {error}", style("Invalid argument:").for_stderr().red());
                    print_usage();
                    std::process::exit(1);
                }));
            }
            value if value.starts_with("--max-tokens=") => {
                let value = &value["--max-tokens=".len()..];
                max_tokens = Some(parse_max_tokens(value).unwrap_or_else(|error| {
                    eprintln!("{} {error}", style("Invalid argument:").for_stderr().red());
                    print_usage();
                    std::process::exit(1);
                }));
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!(
                    "{} {}",
                    style("Unknown argument:").for_stderr().red(),
                    other
                );
                print_usage();
                std::process::exit(1);
            }
        }
    }
    CliArgs {
        local_model,
        gemini,
        global_scope,
        max_tokens,
        show_licenses,
    }
}

fn parse_args() -> CliArgs {
    parse_args_from(env::args().skip(1))
}

fn forced_provider_and_model(cli: &CliArgs) -> Option<(String, String)> {
    if let Some(model) = &cli.local_model {
        Some(("openai".to_string(), model.clone()))
    } else if cli.gemini {
        Some((
            "openrouter".to_string(),
            openrouter::GEMINI_3_7_FLASH_MODEL.to_string(),
        ))
    } else {
        None
    }
}

fn terminal<T>(result: io::Result<T>) -> Result<T> {
    result.map_err(|error| Error::Other(format!("Terminal error: {error}")))
}

async fn choose_provider(
    ui: &mut TerminalUi,
    keys: &ApiKeys,
    available: &[&'static str],
) -> Result<Option<String>> {
    if available.len() == 1 {
        return Ok(Some(available[0].to_string()));
    }
    let labels = available
        .iter()
        .map(|provider| keys.provider_label(provider))
        .collect::<Vec<_>>();
    Ok(terminal(ui.select("Select API", &labels).await)?.map(|index| available[index].to_string()))
}

async fn choose_model(ui: &mut TerminalUi, provider: &str) -> Result<Option<String>> {
    if provider == "anthropic" {
        let models = anthropic::SUGGESTED_MODELS
            .iter()
            .map(|model| model.to_string())
            .collect::<Vec<_>>();
        Ok(terminal(ui.select("Select model", &models).await)?.map(|index| models[index].clone()))
    } else if provider == "openrouter" {
        let models = openrouter::SUGGESTED_MODELS
            .iter()
            .map(|model| model.to_string())
            .collect::<Vec<_>>();
        Ok(terminal(ui.select("Select model", &models).await)?.map(|index| models[index].clone()))
    } else {
        let default = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
        terminal(ui.prompt("Model name", &default).await)
    }
}

async fn choose_provider_and_model(
    ui: &mut TerminalUi,
    keys: &ApiKeys,
    available: &[&'static str],
) -> Result<Option<(String, String)>> {
    let Some(provider) = choose_provider(ui, keys, available).await? else {
        return Ok(None);
    };
    let Some(model) = choose_model(ui, &provider).await? else {
        return Ok(None);
    };
    Ok(Some((provider, model)))
}

fn enqueue_submission(
    ui: &mut TerminalUi,
    queue: &PromptQueue,
    text: String,
    requested: DeliveryMode,
    turn_active: bool,
) {
    if text.trim().eq_ignore_ascii_case("/help") {
        ui.open_help();
        return;
    }
    let delivery = if !turn_active || is_local_command(&text) {
        DeliveryMode::FollowUp
    } else {
        requested
    };
    queue.enqueue(text, delivery);
    ui.sync_queue(queue);
    ui.status(&format!(
        "Queued {} message · {} waiting",
        delivery.label(),
        queue.len()
    ));
}

/// Everything identifying one settled turn for episodic capture.
struct EpisodeContext<'a> {
    prompt: &'a str,
    prompt_source: PromptSource,
    history_start: usize,
    history_revision: u64,
    started_at: chrono::DateTime<chrono::Utc>,
}

async fn drive_started_turn(
    ctx: &mut ReactorCtx<'_>,
    agent: &mut Agent,
    episode: EpisodeContext<'_>,
) -> Result<bool> {
    let ReactorCtx {
        ui,
        queue,
        history_store,
        permission_handler,
        permission_rx,
        memory_events,
        memory,
        autosave,
    } = ctx;
    let ui: &mut TerminalUi = ui;
    let mut durable = DurableBoundary::from_agent(agent);
    let episode_provider = agent.provider().id().to_string();
    let episode_model = agent.provider().model().to_string();
    let (cancel_handle, mut control) = TurnControl::for_turn(queue.clone());
    let (event_tx, event_rx) = agent_display_channel();
    let (checkpoint_tx, mut checkpoint_rx) = mpsc::unbounded_channel();
    let mut exit_requested = false;

    ui.set_turn_active(true);
    ui.set_busy(true, "Thinking");
    ui.draw().map_err(|error| Error::Other(error.to_string()))?;

    let outcome = {
        let mut on_event = move |event: AgentEvent| {
            if matches!(&event, AgentEvent::HistoryCheckpoint { .. }) {
                let _ = checkpoint_tx.send(event);
            } else {
                event_tx.send(event);
            }
        };
        let turn = agent.run_started_turn(&mut on_event, &mut control);
        tokio::pin!(turn);
        let mut ticker = tokio::time::interval(Duration::from_millis(50));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut pending_permission: Option<PermissionRequest> = None;

        loop {
            tokio::select! {
                biased;
                Some(event) = checkpoint_rx.recv() => {
                    apply_runtime_event(
                        ui,
                        queue,
                        history_store,
                        permission_handler,
                        autosave,
                        &mut durable,
                        event,
                    );
                }
                result = &mut turn => {
                    if let Some(pending) = pending_permission.take() {
                        ui.close_permission(pending.id);
                    }
                    break result;
                }
                Some(permission_event) = permission_rx.recv() => {
                    match permission_event {
                        PermissionUiEvent::Request(request) => {
                            if let Some(stale) = pending_permission.replace(request) {
                                ui.close_permission(stale.id);
                                let _ = stale.reply.send(PermissionChoice::DenyOnce);
                            }
                            let pending = pending_permission.as_ref().expect("permission stored");
                            ui.open_permission(pending.id, pending.request.clone());
                            ui.draw().map_err(|error| Error::Other(error.to_string()))?;
                        }
                        PermissionUiEvent::Automatic { request, allowed } => {
                            note_automatic_permission(ui, &request.tool_name, allowed);
                        }
                    }
                }
                Some(event) = memory_events.recv() => handle_memory_event(ui, event),
                _ = ticker.tick() => reactor_tick(ui, autosave)?,
                terminal_event = ui.next_event() => {
                    let action = terminal(
                        ui.handle_event(terminal(terminal_event)?, queue)
                    )?;
                    let persist_queue = action.requires_queue_persist();
                    match action {
                        UiAction::None | UiAction::QueueChanged => {}
                        UiAction::Submit { text, delivery } => {
                            enqueue_submission(ui, queue, text, delivery, true);
                        }
                        UiAction::Interrupt => {
                            if let Some(pending) = pending_permission.take() {
                                ui.close_permission(pending.id);
                                let _ = pending.reply.send(PermissionChoice::DenyOnce);
                            }
                            cancel_handle.cancel();
                            ui.status("Interrupting safely…");
                        }
                        UiAction::Exit => {
                            exit_requested = true;
                            if let Some(pending) = pending_permission.take() {
                                ui.close_permission(pending.id);
                                let _ = pending.reply.send(PermissionChoice::DenyOnce);
                            }
                            cancel_handle.cancel();
                            ui.status("Interrupting before exit…");
                        }
                        UiAction::Permission { id, choice } => {
                            if pending_permission.as_ref().is_some_and(|pending| pending.id == id) {
                                let pending = pending_permission.take().expect("matched permission");
                                let _ = pending.reply.send(choice);
                            }
                        }
                    }
                    if persist_queue {
                        autosave.save(durable.snapshot(history_store, permission_handler, queue));
                    }
                }
                // Keep ordinary display events last in this biased reactor.
                // Consecutive fragments are one bounded preview batch, so
                // provider chunking cannot amplify pending queue records.
                events = event_rx.recv_batch() => {
                    for event in events {
                        apply_runtime_event(
                            ui,
                            queue,
                            history_store,
                            permission_handler,
                            autosave,
                            &mut durable,
                            event,
                        );
                    }
                }
            }
        }
    };

    drain_pending_agent_events(&mut checkpoint_rx, &event_rx, |event| {
        apply_runtime_event(
            ui,
            queue,
            history_store,
            permission_handler,
            autosave,
            &mut durable,
            event,
        );
    });
    queue.normalize_steers();
    let continue_goal = should_continue_goal(agent.goal().is_some(), exit_requested, &outcome);
    let goal_queue_changed = queue.reconcile_goal_continuation(continue_goal);
    if continue_goal && goal_queue_changed {
        ui.info("Goal remains active; queued an automatic continuation.");
    }
    ui.sync_queue(queue);
    autosave.save(make_saved_state(
        agent,
        permission_handler,
        queue,
        history_store.scope(),
    ));
    let episode_outcome = match &outcome {
        Ok(outcome) => EpisodeOutcome::from(*outcome),
        Err(_) => EpisodeOutcome::Error,
    };
    if let Some(memory) = *memory {
        if history_tool_protocol_is_valid(agent.history()) {
            let episode_history = if agent.history_revision() == episode.history_revision {
                agent
                    .history()
                    .get(episode.history_start..)
                    .unwrap_or_default()
            } else {
                &[]
            };
            let prompt_origin = match episode.prompt_source {
                PromptSource::User => MessageOrigin::Conversation,
                PromptSource::GoalContinuation => MessageOrigin::GoalContinuation,
            };
            if let Err(error) = memory.enqueue_settled_turn_with_origin(
                episode.prompt,
                prompt_origin,
                episode_history,
                episode_outcome,
                &episode_provider,
                &episode_model,
                episode.started_at,
            ) {
                ui.error(&format!("Failed to queue settled episode: {error}"));
            }
        } else {
            ui.error("Skipped episodic capture because settled history is not protocol-valid.");
        }
    }

    match outcome {
        Ok(TurnOutcome::Completed | TurnOutcome::Refused) => ui.set_busy(false, "Ready"),
        Ok(TurnOutcome::PausedOnDenial) => {
            ui.set_busy(false, "Paused after denial");
            ui.info("Tool denied. Queued work will continue as a new turn.");
        }
        Ok(TurnOutcome::MaxIterationsReached) => {
            ui.set_busy(false, "Iteration limit reached");
            ui.info("Iteration limit reached; queued work will continue separately.");
        }
        Ok(TurnOutcome::Interrupted) => {
            ui.cancel_running_activity();
            ui.set_busy(false, "Interrupted");
            ui.info("Turn interrupted cleanly; unfinished tools were paired with error results.");
        }
        Err(error) => {
            ui.set_busy(false, "Error");
            ui.error(&error.to_string());
            ui.info("Conversation state is preserved; queued work can continue.");
        }
    }
    ui.set_context_tokens(agent.context_tokens());
    ui.draw().map_err(|error| Error::Other(error.to_string()))?;
    Ok(exit_requested)
}

async fn drive_compaction(ctx: &mut ReactorCtx<'_>, agent: &mut Agent) -> Result<bool> {
    let ReactorCtx {
        ui,
        queue,
        history_store,
        permission_handler,
        permission_rx,
        memory_events,
        autosave,
        ..
    } = ctx;
    let ui: &mut TerminalUi = ui;
    let mut durable = DurableBoundary::from_agent(agent);
    let before = agent.context_tokens();
    let (event_tx, event_rx) = agent_display_channel();
    let (checkpoint_tx, mut checkpoint_rx) = mpsc::unbounded_channel();
    let mut exit_requested = false;
    ui.set_busy(true, "Compacting context");

    let compacted = {
        let mut on_event = move |event: AgentEvent| {
            if matches!(&event, AgentEvent::HistoryCheckpoint { .. }) {
                let _ = checkpoint_tx.send(event);
            } else {
                event_tx.send(event);
            }
        };
        let operation = agent.compact(&mut on_event);
        tokio::pin!(operation);
        let mut ticker = tokio::time::interval(Duration::from_millis(50));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                biased;
                Some(event) = checkpoint_rx.recv() => apply_runtime_event(
                    ui,
                    queue,
                    history_store,
                    permission_handler,
                    autosave,
                    &mut durable,
                    event,
                ),
                result = &mut operation => break Some(result),
                Some(permission_event) = permission_rx.recv() => {
                    handle_stale_permission_event(ui, permission_event);
                }
                Some(event) = memory_events.recv() => handle_memory_event(ui, event),
                _ = ticker.tick() => reactor_tick(ui, autosave)?,
                terminal_event = ui.next_event() => {
                    let action =
                        terminal(ui.handle_event(terminal(terminal_event)?, queue))?;
                    let persist_queue = action.requires_queue_persist();
                    match action {
                        UiAction::Submit { text, .. } => {
                            enqueue_submission(
                                ui,
                                queue,
                                text,
                                DeliveryMode::FollowUp,
                                true,
                            );
                        }
                        UiAction::Interrupt => break None,
                        UiAction::Exit => {
                            exit_requested = true;
                            break None;
                        }
                        UiAction::None | UiAction::QueueChanged | UiAction::Permission { .. } => {}
                    }
                    if persist_queue {
                        autosave.save(durable.snapshot(history_store, permission_handler, queue));
                    }
                }
                // See the active-turn reactor above: fragment-amplified
                // display backlog is coalesced before this branch wakes.
                events = event_rx.recv_batch() => {
                    for event in events {
                        apply_runtime_event(
                            ui,
                            queue,
                            history_store,
                            permission_handler,
                            autosave,
                            &mut durable,
                            event,
                        );
                    }
                },
            }
        }
    };

    drain_pending_agent_events(&mut checkpoint_rx, &event_rx, |event| {
        apply_runtime_event(
            ui,
            queue,
            history_store,
            permission_handler,
            autosave,
            &mut durable,
            event,
        );
    });
    autosave.save(make_saved_state(
        agent,
        permission_handler,
        queue,
        history_store.scope(),
    ));
    match compacted {
        Some(Ok(true)) => ui.info(&format!(
            "Context compacted: ~{}k → ~{}k tokens",
            before / 1_000,
            agent.context_tokens() / 1_000
        )),
        Some(Ok(false)) => ui.info("Nothing to compact yet."),
        Some(Err(error)) => ui.error(&format!("Compaction failed: {error}")),
        None => ui.info("Compaction interrupted before changing history."),
    }
    ui.set_busy(false, "Ready");
    ui.set_context_tokens(agent.context_tokens());
    Ok(exit_requested)
}

enum CommandFlow {
    Continue,
    Exit,
}

fn replace_goal(agent: &mut Agent, ui: &mut TerminalUi, queue: &PromptQueue, goal: Option<String>) {
    agent.set_goal(goal);
    queue.reconcile_goal_continuation(agent.goal().is_some());
    ui.set_goal(agent.goal());
    ui.sync_queue(queue);
    ui.set_context_tokens(agent.context_tokens());
    if let Some(goal) = agent.goal() {
        ui.info(&format!("Active goal set: {}", truncate_middle(goal, 400)));
    } else {
        ui.info("Active goal cleared.");
    }
}

fn should_continue_goal(
    goal_active: bool,
    exit_requested: bool,
    outcome: &Result<TurnOutcome>,
) -> bool {
    goal_active
        && !exit_requested
        && matches!(
            outcome,
            Ok(TurnOutcome::Completed | TurnOutcome::MaxIterationsReached)
        )
}

fn handle_memory_event(ui: &mut TerminalUi, event: MemoryEvent) {
    match event {
        MemoryEvent::CaptureFailed(error) => {
            ui.error(&format!("Failed to record settled episode: {error}"));
        }
    }
}

async fn drive_memory_command(
    ctx: &mut ReactorCtx<'_>,
    command: MemoryCommand<'_>,
    exports_directory: &Path,
    agent: &Agent,
) -> Result<bool> {
    let ReactorCtx {
        ui,
        queue,
        history_store,
        permission_handler,
        permission_rx,
        memory_events,
        memory,
        autosave,
    } = ctx;
    let ui: &mut TerminalUi = ui;
    let Some(memory) = *memory else {
        ui.error("Episodic memory is unavailable; see the startup error.");
        return Ok(false);
    };
    ui.set_busy(true, "Memory");
    let operation = run_memory_command(command, memory, exports_directory);
    tokio::pin!(operation);
    let mut ticker = tokio::time::interval(Duration::from_millis(50));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let exit_requested = loop {
        tokio::select! {
            biased;
            result = &mut operation => {
                match result {
                    Ok(lines) => {
                        for line in lines {
                            ui.info(&line);
                        }
                    }
                    Err(error) => ui.error(&format!("Memory command failed: {error}")),
                }
                break false;
            }
            Some(permission_event) = permission_rx.recv() => {
                handle_stale_permission_event(ui, permission_event);
            }
            Some(event) = memory_events.recv() => handle_memory_event(ui, event),
            _ = ticker.tick() => reactor_tick(ui, autosave)?,
            terminal_event = ui.next_event() => {
                let action = terminal(ui.handle_event(terminal(terminal_event)?, queue))?;
                let persist_queue = action.requires_queue_persist();
                match action {
                    UiAction::Submit { text, delivery } => {
                        enqueue_submission(ui, queue, text, delivery, false);
                    }
                    UiAction::Exit => break true,
                    UiAction::Interrupt => {
                        ui.status("The dispatched memory transaction cannot be safely cancelled.");
                    }
                    UiAction::None
                    | UiAction::QueueChanged
                    | UiAction::Permission { .. } => {}
                }
                if persist_queue {
                    autosave.save(make_saved_state(
                        agent,
                        permission_handler,
                        queue,
                        history_store.scope(),
                    ));
                }
            }
        }
    };
    ui.set_busy(false, "Ready");
    Ok(exit_requested)
}

#[allow(clippy::too_many_arguments)]
async fn execute_command(
    ctx: &mut ReactorCtx<'_>,
    text: &str,
    agent: &mut Agent,
    profile_paths: &ProfilePaths,
    mcp_runtime: &mut Option<McpRuntime>,
    keys: &ApiKeys,
    available: &[&'static str],
) -> Result<CommandFlow> {
    let command = parse_local_command(text).unwrap_or(LocalCommand::Unknown(text.trim()));
    match command {
        LocalCommand::Exit => return Ok(CommandFlow::Exit),
        LocalCommand::Help => terminal(ctx.ui.show_help().await)?,
        LocalCommand::Compact => {
            if drive_compaction(ctx, agent).await? {
                return Ok(CommandFlow::Exit);
            }
        }
        LocalCommand::Clear => {
            agent.clear_history();
            ctx.ui.clear_conversation();
            ctx.ui.set_context_tokens(0);
            ctx.ui
                .info("Conversation cleared. The active goal was preserved.");
        }
        LocalCommand::Save(name) => run_save_command(ctx, agent, name).await?,
        LocalCommand::Load(name) => {
            let selected_name = if let Some(name) = name {
                Some(name.to_string())
            } else {
                let saved = with_history_store(ctx.history_store, |store| store.list()).await?;
                if saved.is_empty() {
                    ctx.ui.info("No saved conversations found.");
                    None
                } else {
                    terminal(ctx.ui.select("Load conversation", &saved).await)?
                        .map(|index| saved[index].clone())
                }
            };
            if let Some(name) = selected_name {
                match load_named_session(ctx, &name, agent, keys).await {
                    Ok(count) => ctx.ui.info(&format!(
                        "Loaded saved session '{name}' ({count} messages)."
                    )),
                    Err(error) => ctx.ui.error(&format!("Failed to load '{name}': {error}")),
                }
            }
        }
        LocalCommand::Model => {
            if let Some((provider_name, model)) =
                choose_provider_and_model(ctx.ui, keys, available).await?
            {
                match build_provider(keys, &provider_name, model) {
                    Ok(provider) => {
                        agent.set_provider(provider);
                        ctx.ui.set_session(
                            agent.provider().display_name(),
                            agent.provider().model(),
                            agent.registry.tool_names().len(),
                        );
                        ctx.ui.info("Model switched.");
                    }
                    Err(error) => ctx.ui.error(&format!("Failed to switch model: {error}")),
                }
            }
        }
        LocalCommand::Mcp(McpCommand::Status) => {
            if let Some(runtime) = mcp_runtime {
                ctx.ui.info(&runtime.status());
            } else {
                ctx.ui.info(&format!(
                    "MCP: no configuration was loaded. Add {} and restart.",
                    profile_paths.mcp_config().display()
                ));
            }
        }
        LocalCommand::Mcp(McpCommand::Retry(requested)) => {
            let Some(runtime) = mcp_runtime else {
                ctx.ui
                    .error("MCP retry is unavailable because no configuration was loaded.");
                return Ok(CommandFlow::Continue);
            };
            match runtime.retry_targets(requested) {
                Err(error) => ctx.ui.error(&format!("MCP retry refused: {error}")),
                Ok(targets) => {
                    let config = runtime.config.clone();
                    let durable = DurableBoundary::from_agent(agent);
                    if drive_mcp_discovery(
                        ctx,
                        &mut agent.registry,
                        &config,
                        runtime,
                        &targets,
                        &durable,
                    )
                    .await?
                    {
                        return Ok(CommandFlow::Exit);
                    }
                    ctx.ui.set_session(
                        agent.provider().display_name(),
                        agent.provider().model(),
                        agent.registry.tool_names().len(),
                    );
                }
            }
        }
        LocalCommand::Copy(CopyCommand::Select) => terminal(ctx.ui.enter_copy_mode())?,
        LocalCommand::Copy(copy) => {
            let (payload, empty_message) = match copy {
                CopyCommand::Last => (
                    latest_assistant_text(agent.history()),
                    "No committed assistant response to copy.",
                ),
                CopyCommand::All => (
                    conversation_transcript(agent.history()),
                    "No committed conversation text to copy.",
                ),
                CopyCommand::Reasoning => (
                    latest_assistant_reasoning(agent.history()),
                    "No inspectable committed provider reasoning to copy.",
                ),
                CopyCommand::Select => unreachable!("selection copy handled above"),
            };
            if let Some(payload) = payload {
                match ctx.ui.request_clipboard_copy(&payload) {
                    Ok(bytes) => ctx.ui.info(&format!(
                        "Sent {bytes} bytes to the terminal clipboard via OSC 52. If the terminal blocks it, use /copy select."
                    )),
                    Err(error) => ctx.ui.error(&format!(
                        "Clipboard request failed: {error}. Use /copy select for native selection."
                    )),
                }
            } else {
                ctx.ui.info(empty_message);
            }
        }
        LocalCommand::Permissions(command) => {
            ctx.ui
                .info(&run_permission_command(command, ctx.permission_handler));
        }
        LocalCommand::Tools(command) => {
            ctx.ui.info(&run_tools_command(command, agent));
        }
        LocalCommand::Usage(UsageCommand::Show) => {
            let report = ctx.ui.provider_usage_report();
            ctx.ui.info(&report);
        }
        LocalCommand::Usage(UsageCommand::Reset) => {
            ctx.ui.reset_provider_usage();
            ctx.ui.info("Provider usage counters reset. Conversation, context, and provider state were unchanged.");
        }
        LocalCommand::History(HistoryCommand::Forget(name)) => {
            if name == AUTOSAVE_NAME {
                ctx.ui.error(
                    "The active autosave cannot be forgotten; use /clear to replace its conversation content.",
                );
            } else {
                let requested = name.to_string();
                let existing = with_history_store(ctx.history_store, move |store| {
                    store.inspect_current_archive(&requested)
                })
                .await
                .and_then(|result| result);
                match existing {
                    Ok(None) => ctx.ui.info(&format!(
                        "No current-scope saved session named '{name}'. Use /history list."
                    )),
                    Err(error) => ctx.ui.error(&format!("History deletion refused: {error}")),
                    Ok(Some(_)) => {
                        let choices = vec!["Cancel".to_string(), format!("Delete '{name}'")];
                        let title = format!(
                            "Forget saved session '{name}'? Backups and prior copies remain."
                        );
                        if terminal(ctx.ui.select(&title, &choices).await)? == Some(1) {
                            let requested = name.to_string();
                            let deleted = with_history_store(ctx.history_store, move |store| {
                                store.forget_current_archive(&requested)
                            })
                            .await
                            .and_then(|result| result);
                            match deleted {
                                Ok(true) => ctx.ui.info(&format!(
                                    "Deleted current-scope saved session '{name}'. Prior copies, backups, and filesystem snapshots are not erased."
                                )),
                                Ok(false) => ctx.ui.info(&format!(
                                    "Saved session '{name}' disappeared before deletion."
                                )),
                                Err(error) => {
                                    ctx.ui.error(&format!("History deletion failed: {error}"))
                                }
                            }
                        } else {
                            ctx.ui.info(&format!("Kept saved session '{name}'."));
                        }
                    }
                }
            }
        }
        LocalCommand::History(command) => {
            match run_history_command(command, ctx.history_store).await {
                Ok(output) => ctx.ui.info(&output),
                Err(error) => ctx.ui.error(&format!("History inspection failed: {error}")),
            }
        }
        LocalCommand::Goal(GoalCommand::Edit) => {
            let current = agent.goal().unwrap_or_default();
            if let Some(goal) =
                terminal(ctx.ui.prompt("Active goal (empty clears)", current).await)?
            {
                replace_goal(agent, ctx.ui, ctx.queue, Some(goal));
            }
        }
        LocalCommand::Goal(GoalCommand::Show) => {
            if let Some(goal) = agent.goal() {
                ctx.ui.info(&format!("Active goal: {goal}"));
            } else {
                ctx.ui
                    .info("No active goal. Use /goal <objective> to set one.");
            }
        }
        LocalCommand::Goal(GoalCommand::Clear) => replace_goal(agent, ctx.ui, ctx.queue, None),
        LocalCommand::Goal(GoalCommand::Set(goal)) => {
            replace_goal(agent, ctx.ui, ctx.queue, Some(goal.to_string()))
        }
        LocalCommand::Memory(command) => {
            if drive_memory_command(ctx, command, profile_paths.exports_directory(), agent).await? {
                return Ok(CommandFlow::Exit);
            }
        }
        LocalCommand::Unknown(command) => {
            ctx.ui
                .info(&format!("Unknown local command: {command}. Use /help."));
        }
    }
    Ok(CommandFlow::Continue)
}

/// The base system prompt plus per-run context: archive scope, the skills
/// index, and project notes (AGENTS.md/CLAUDE.md).
fn compose_system_prompt(scope: &WorkspaceScope, profile_paths: &ProfilePaths) -> String {
    let mut system_prompt = include_str!("../SYSTEM_PROMPT.md").to_string();
    system_prompt.push_str(&format!(
        "\n\n## Active archive scope\n\nThis run's history and episodic-memory scope is `{}`. \
         Current-scope storage is isolated from every other project. Global or other-project \
         archives are available only through the permission-gated search/read tools; never \
         assume or retrieve them automatically.",
        scope.display_name()
    ));
    if let Some(index) = generalist::skills::skills_index(profile_paths.skills_directory()) {
        system_prompt.push_str(&index);
    }
    if let Some(project_root) = scope.project_root() {
        for name in ["AGENTS.md", "CLAUDE.md"] {
            let path = project_root.join(name);
            if let Ok(notes) = fs::read_to_string(&path) {
                system_prompt.push_str(&format!(
                    "\n\n## Project notes ({})\n\n{notes}",
                    path.display()
                ));
                break;
            }
        }
    }
    system_prompt
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = parse_args();
    if cli.show_licenses {
        print!("{THIRD_PARTY_LICENSES}");
        return Ok(());
    }
    let profile_paths = ProfilePaths::discover(); // profile-path-allow: one startup resolution

    let env_path = profile_paths.environment_file();
    if env_path.exists() {
        dotenvy::from_path(env_path).ok();
    }

    let mut keys = ApiKeys::from_env();
    if cli.local_model.is_some() {
        if env::var("OPENAI_BASE_URL").is_err() {
            keys.openai_base_url = OLLAMA_BASE_URL.to_string();
        }
        keys.openai.get_or_insert_with(|| "unused".to_string());
    }
    if cli.gemini && cli.local_model.is_none() && keys.openrouter.is_none() {
        eprintln!(
            "{} --gemini requires OPENROUTER_API_KEY (in the environment or {}).",
            style("Configuration error:").for_stderr().red(),
            profile_paths.environment_file().display()
        );
        std::process::exit(1);
    }
    let available = keys.available_providers();
    if available.is_empty() {
        eprintln!("{}", style("No API key found.").for_stderr().red());
        eprintln!(
            "Set at least one of these (in the environment or {}):",
            profile_paths.environment_file().display()
        );
        eprintln!("  ANTHROPIC_API_KEY=...   for Anthropic models");
        eprintln!("  OPENAI_API_KEY=...      for OpenAI or a compatible server");
        eprintln!(
            "  OPENROUTER_API_KEY=...  for OpenRouter (Kimi K3 by default; --gemini selects Gemini 3.7 Flash)"
        );
        eprintln!("  OPENAI_BASE_URL=...     optional, e.g. {OLLAMA_BASE_URL} for Ollama");
        eprintln!("Or run against a local model directly: generalist --local <model>");
        std::process::exit(1);
    }

    let mut ui = terminal(TerminalUi::start("Starting", "selecting model"))?;
    let working_directory = env::current_dir()
        .map_err(|error| Error::Other(format!("Failed to read working directory: {error}")))?;
    let scope = if cli.global_scope {
        WorkspaceScope::global()
    } else {
        WorkspaceScope::discover(&working_directory)?
    };
    let history_store = HistoryStore::open_profile(&profile_paths, scope.clone())?;
    ui.info(&format!("Storage scope: {}", scope.display_name()));

    let (memory_event_tx, mut memory_event_rx) = mpsc::unbounded_channel();
    let memory = match EpisodicMemory::open_scoped_with_events(
        profile_paths.memory_database().to_path_buf(),
        scope.clone(),
        Some(memory_event_tx),
    ) {
        Ok(memory) => Some(memory),
        Err(error) => {
            ui.error(&format!("Episodic memory is unavailable: {error}"));
            None
        }
    };
    let provider_and_model = match forced_provider_and_model(&cli) {
        Some(requested) => Some(requested),
        None => match default_remote_provider_and_model(&keys) {
            Some(default) => Some(default),
            None => choose_provider_and_model(&mut ui, &keys, &available).await?,
        },
    };
    let Some((provider_name, model)) = provider_and_model else {
        return Ok(());
    };
    let provider = build_provider(&keys, &provider_name, model)?;
    ui.set_session(provider.display_name(), provider.model(), 0);

    let (permission_tx, mut permission_rx) = mpsc::unbounded_channel();
    let permission_prompt = Arc::new(PermissionBrokerPrompt::new(permission_tx));
    let permission_handler = MemoryPermissionHandler::with_prompt(permission_prompt);
    let mut registry = build_registry(
        profile_paths.todo_file(),
        &permission_handler,
        &history_store,
        memory.as_ref(),
    )?;

    let (startup_goal, startup_history, queue) =
        recover_startup_runtime(&history_store, &permission_handler, &mut ui);
    queue.normalize_steers();
    queue.reconcile_goal_continuation(startup_goal.is_some());
    ui.sync_queue(&queue);
    ui.set_goal(startup_goal.as_deref());
    ui.set_session(
        provider.display_name(),
        provider.model(),
        registry.tool_names().len(),
    );
    let startup_durable = DurableBoundary::before_agent(
        provider.as_ref(),
        startup_goal.clone(),
        startup_history.clone(),
    );
    let autosave = AutosaveWriter::spawn(history_store.clone());
    let mut mcp_runtime = match McpConfig::load_checked(profile_paths.mcp_config()) {
        Ok(config) => config.map(McpRuntime::new),
        Err(error) => {
            ui.error(&error.to_string());
            None
        }
    };
    let mut ctx = ReactorCtx {
        ui: &mut ui,
        queue: &queue,
        history_store: &history_store,
        permission_handler: &permission_handler,
        permission_rx: &mut permission_rx,
        memory_events: &mut memory_event_rx,
        memory: memory.as_ref(),
        autosave: &autosave,
    };
    if let Some(runtime) = mcp_runtime.as_mut() {
        let config = runtime.config.clone();
        let targets = runtime.configured_targets();
        if drive_mcp_discovery(
            &mut ctx,
            &mut registry,
            &config,
            runtime,
            &targets,
            &startup_durable,
        )
        .await?
        {
            return Ok(());
        }
    }

    let system_prompt = compose_system_prompt(&scope, &profile_paths);

    let mut agent = Agent::new(provider, registry, system_prompt);
    agent.max_tokens = cli.max_tokens;
    agent.set_goal(startup_goal);
    if !startup_history.is_empty() {
        agent.replace_history(startup_history);
    }
    ctx.ui.sync_queue(ctx.queue);
    ctx.ui.set_goal(agent.goal());
    ctx.ui.set_session(
        agent.provider().display_name(),
        agent.provider().model(),
        agent.registry.tool_names().len(),
    );
    ctx.ui.set_context_tokens(agent.context_tokens());

    let mut ticker = tokio::time::interval(Duration::from_millis(50));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut exiting = false;

    while !exiting {
        ctx.queue.normalize_steers();
        ctx.ui.sync_queue(ctx.queue);

        if let Some(claim) = ctx.queue.claim_follow_up() {
            let prompt = claim.prompts()[0].clone();
            if prompt.source == PromptSource::GoalContinuation && agent.goal().is_none() {
                claim.commit();
                ctx.ui.sync_queue(ctx.queue);
                continue;
            } else if is_local_command(&prompt.text) {
                claim.commit();
                exiting = matches!(
                    execute_command(
                        &mut ctx,
                        &prompt.text,
                        &mut agent,
                        &profile_paths,
                        &mut mcp_runtime,
                        &keys,
                        &available,
                    )
                    .await?,
                    CommandFlow::Exit
                );
            } else {
                let started_at = chrono::Utc::now();
                let episode_history_start = agent.history().len();
                let episode_history_revision = agent.history_revision();
                agent.begin_queued_turn(&prompt);
                claim.commit();
                if prompt.source == PromptSource::GoalContinuation {
                    ctx.ui.info("Continuing the active goal automatically.");
                } else {
                    ctx.ui.push_user(prompt.text.trim());
                }
                ctx.autosave.save(make_saved_state(
                    &agent,
                    ctx.permission_handler,
                    ctx.queue,
                    ctx.history_store.scope(),
                ));
                exiting = drive_started_turn(
                    &mut ctx,
                    &mut agent,
                    EpisodeContext {
                        prompt: &prompt.text,
                        prompt_source: prompt.source,
                        history_start: episode_history_start,
                        history_revision: episode_history_revision,
                        started_at,
                    },
                )
                .await?;
            }
            ctx.autosave.save(make_saved_state(
                &agent,
                ctx.permission_handler,
                ctx.queue,
                ctx.history_store.scope(),
            ));
            continue;
        }

        ctx.ui.set_busy(false, "Ready");
        let ReactorCtx {
            ui,
            queue,
            history_store,
            permission_handler,
            permission_rx,
            memory_events,
            autosave,
            ..
        } = &mut ctx;
        let ui: &mut TerminalUi = ui;
        tokio::select! {
            biased;
            Some(permission_event) = permission_rx.recv() => {
                handle_stale_permission_event(ui, permission_event);
            }
            Some(event) = memory_events.recv() => handle_memory_event(ui, event),
            _ = ticker.tick() => reactor_tick(ui, autosave)?,
            terminal_event = ui.next_event() => {
                let action =
                    terminal(ui.handle_event(terminal(terminal_event)?, queue))?;
                let persist_queue = action.requires_queue_persist();
                match action {
                    UiAction::Submit { text, delivery } => {
                        enqueue_submission(ui, queue, text, delivery, false);
                    }
                    UiAction::Exit => exiting = true,
                    UiAction::None
                    | UiAction::QueueChanged
                    | UiAction::Interrupt
                    | UiAction::Permission { .. } => {}
                }
                if persist_queue {
                    autosave.save(make_saved_state(
                        &agent,
                        permission_handler,
                        queue,
                        history_store.scope(),
                    ));
                }
            }
        }
    }

    // Everything queued before exit must reach disk, and a final write error
    // must not disappear merely because the reactor has stopped ticking.
    let mut shutdown_errors = Vec::new();
    if let Err(error) = autosave.flush() {
        shutdown_errors.push(error.to_string());
    }
    if let Some(memory) = &memory {
        if let Err(error) = memory.flush().await {
            shutdown_errors.push(format!(
                "Failed to flush episodic memory before exit: {error}"
            ));
        }
    }
    if !shutdown_errors.is_empty() {
        return Err(Error::Other(shutdown_errors.join("\n")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use generalist::{PermissionCommand, ToolsCommand};

    #[test]
    fn memory_export_uses_supplied_profile_directory() {
        let home = tempfile::tempdir().unwrap();
        let profile = ProfilePaths::new(home.path());

        let path = write_memory_export(profile.exports_directory(), &[]).unwrap();

        assert_eq!(path.parent(), Some(profile.exports_directory()));
        assert_eq!(fs::read_to_string(path).unwrap(), "[]");
    }

    #[test]
    fn gemini_flag_selects_gemini_3_7_flash_through_openrouter() {
        let cli = parse_args_from(
            ["--gemini", "--global", "--max-tokens=4096"]
                .into_iter()
                .map(str::to_string),
        );

        assert!(cli.gemini);
        assert!(cli.global_scope);
        assert_eq!(cli.max_tokens, Some(4096));
        assert_eq!(
            forced_provider_and_model(&cli),
            Some((
                "openrouter".to_string(),
                "google/gemini-3.7-flash".to_string()
            ))
        );
    }

    #[test]
    fn licenses_flag_selects_the_embedded_notice_output() {
        let cli = parse_args_from(["--licenses"].into_iter().map(str::to_string));

        assert!(cli.show_licenses);
        assert!(THIRD_PARTY_LICENSES.starts_with("THIRD-PARTY LICENSES\n"));
        assert!(THIRD_PARTY_LICENSES.contains("CDLA-Permissive-2.0"));
        assert!(THIRD_PARTY_LICENSES.contains("Unicode-3.0"));
    }

    #[test]
    fn local_model_keeps_precedence_over_gemini_shortcut() {
        let cli = parse_args_from(
            ["--gemini", "--local", "custom-local-model"]
                .into_iter()
                .map(str::to_string),
        );

        assert_eq!(
            forced_provider_and_model(&cli),
            Some(("openai".to_string(), "custom-local-model".to_string()))
        );
    }

    #[test]
    fn max_tokens_parser_accepts_explicit_positive_values_only() {
        assert_eq!(parse_max_tokens("32000").unwrap(), 32_000);
        assert!(parse_max_tokens("0")
            .unwrap_err()
            .contains("greater than zero"));
        assert!(parse_max_tokens("unbounded")
            .unwrap_err()
            .contains("invalid"));
    }
    use generalist::mcp::McpServerConfig;
    use generalist::{ContentBlock, Message, Tool};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn million_fragment_burst_has_one_hard_bounded_preview_record() {
        let (sender, receiver) = agent_display_channel();
        sender.send(AgentEvent::ApiCallStarted);
        for index in 0..1_000_000 {
            if index % 2 == 0 {
                sender.send(AgentEvent::AssistantTextDelta("x".to_string()));
            } else {
                sender.send(AgentEvent::ReasoningDelta("r".to_string()));
            }
        }
        sender.send(AgentEvent::ApiCallFinished { usage: None });

        assert_eq!(receiver.buffered_records(), 3);
        assert_eq!(receiver.pending_preview_bytes(), MAX_PENDING_STREAM_BYTES);
        assert!(matches!(
            receiver.try_recv_batch().unwrap().as_slice(),
            [AgentEvent::ApiCallStarted]
        ));

        let preview = receiver.try_recv_batch().unwrap();
        assert!(matches!(
            preview.first(),
            Some(AgentEvent::AssistantTextDelta(_))
        ));
        let retained = preview
            .iter()
            .map(|event| match event {
                AgentEvent::AssistantTextDelta(text) | AgentEvent::ReasoningDelta(text) => {
                    text.len()
                }
                _ => 0,
            })
            .sum::<usize>();
        let omitted = preview
            .iter()
            .find_map(|event| match event {
                AgentEvent::StreamDisplayTruncated {
                    text_bytes,
                    reasoning_bytes,
                } => Some(text_bytes + reasoning_bytes),
                _ => None,
            })
            .expect("the bounded preview must disclose omitted bytes");
        assert_eq!(retained, MAX_PENDING_STREAM_BYTES);
        assert_eq!(omitted, 1_000_000 - MAX_PENDING_STREAM_BYTES);
        assert!(matches!(
            receiver.try_recv_batch().unwrap().as_slice(),
            [AgentEvent::ApiCallFinished { usage: None }]
        ));
        assert!(receiver.try_recv_batch().is_none());
    }

    #[test]
    fn structural_events_split_stream_batches_without_reordering() {
        let (sender, receiver) = agent_display_channel();
        sender.send(AgentEvent::AssistantTextDelta("before".to_string()));
        sender.send(AgentEvent::Notice("boundary".to_string()));
        sender.send(AgentEvent::AssistantTextDelta("after".to_string()));

        assert_eq!(receiver.buffered_records(), 3);
        assert!(matches!(
            receiver.try_recv_batch().unwrap().as_slice(),
            [AgentEvent::AssistantTextDelta(text)] if text == "before"
        ));
        assert!(matches!(
            receiver.try_recv_batch().unwrap().as_slice(),
            [AgentEvent::Notice(message)] if message == "boundary"
        ));
        assert!(matches!(
            receiver.try_recv_batch().unwrap().as_slice(),
            [AgentEvent::AssistantTextDelta(text)] if text == "after"
        ));
    }

    struct CatalogTool {
        name: String,
        description: String,
        code_only: bool,
        executions: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Tool for CatalogTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn input_schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            })
        }

        async fn execute(&self, _input: serde_json::Value) -> Result<String> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Ok("executed".to_string())
        }

        fn code_only(&self) -> bool {
            self.code_only
        }
    }

    fn catalog_agent(executions: Arc<AtomicUsize>) -> Agent {
        let mut registry = ToolRegistry::new();
        registry
            .register(Arc::new(CatalogTool {
                name: "zeta_search".to_string(),
                description: "Search remote records\nwith a stable query.".to_string(),
                code_only: true,
                executions: executions.clone(),
            }))
            .unwrap();
        registry
            .register(Arc::new(CatalogTool {
                name: "alpha_read".to_string(),
                description: "Read one local record.".to_string(),
                code_only: false,
                executions,
            }))
            .unwrap();
        let provider = OpenAiProvider::new(
            "unused".to_string(),
            generalist::provider::openai::DEFAULT_BASE_URL.to_string(),
            "test-model".to_string(),
        )
        .unwrap();
        Agent::new(Box::new(provider), registry, "test")
    }

    #[test]
    fn structured_state_does_not_collide_with_legacy_input_history_file() {
        let home = tempfile::tempdir().unwrap();
        let legacy = home.path().join(".generalist_history"); // profile-path-allow: legacy fixture
        fs::write(&legacy, "old input history\n").unwrap();

        let store = HistoryStore::open(home.path().to_path_buf(), WorkspaceScope::Global).unwrap();
        assert_eq!(fs::read_to_string(&legacy).unwrap(), "old input history\n");

        let mut state = SavedState::new(WorkspaceScope::Global, "openai".into(), "model-v1".into());
        store.save(&state, AUTOSAVE_NAME).unwrap();
        state.model = "model-v2".into();
        let autosave = store.save(&state, AUTOSAVE_NAME).unwrap();
        assert!(autosave.starts_with(store.directory()));
        assert_eq!(store.load(AUTOSAVE_NAME).unwrap().model, "model-v2");
    }

    #[test]
    fn autosave_flush_persists_the_latest_queued_snapshot() {
        let home = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(home.path().to_path_buf(), WorkspaceScope::Global).unwrap();
        let writer = AutosaveWriter::spawn(store.clone());
        let first = SavedState::new(WorkspaceScope::Global, "openai".into(), "first".into());
        let latest = SavedState::new(WorkspaceScope::Global, "openai".into(), "latest".into());

        writer.save(first);
        writer.save(latest);
        writer.flush().unwrap();

        assert_eq!(store.load(AUTOSAVE_NAME).unwrap().model, "latest");
    }

    #[test]
    fn startup_recovery_keeps_a_goal_but_admits_context_only_with_queued_work() {
        let mut goal_only =
            SavedState::new(WorkspaceScope::Global, "openai".into(), "model".into());
        goal_only.goal = Some("finish the work".into());
        goal_only
            .conversation_history
            .push(Message::user_text("settled old context"));
        goal_only.always_allow_tools.insert("bash".into());

        let plan = plan_startup_recovery(Some(goal_only));
        assert_eq!(plan.goal.as_deref(), Some("finish the work"));
        assert!(plan.history.is_empty());
        assert!(plan.always_allow_tools.is_empty());
        assert!(plan.queued_prompts.is_empty());
        assert!(!plan.invalid_queue);

        let mut queued = SavedState::new(WorkspaceScope::Global, "openai".into(), "model".into());
        queued.goal = Some("finish the work".into());
        queued
            .conversation_history
            .push(Message::user_text("committed context"));
        queued.always_allow_tools.insert("bash".into());
        queued.queued_prompts.push(generalist::QueuedPrompt {
            id: 9,
            text: "resume me".into(),
            delivery: DeliveryMode::FollowUp,
            source: PromptSource::User,
        });

        let plan = plan_startup_recovery(Some(queued));
        assert_eq!(plan.history[0].text(), "committed context");
        assert!(plan.always_allow_tools.contains("bash"));
        assert_eq!(plan.queued_prompts[0].id, 9);
        assert!(!plan.invalid_queue);
    }

    #[test]
    fn startup_recovery_rejects_a_queued_invalid_tool_boundary() {
        let mut state = SavedState::new(WorkspaceScope::Global, "openai".into(), "model".into());
        state.goal = Some("preserve this goal".into());
        state
            .conversation_history
            .push(Message::assistant(vec![ContentBlock::ToolUse {
                name: "python".into(),
                input: json!({}),
                id: "dangling".into(),
            }]));
        state.queued_prompts.push(generalist::QueuedPrompt {
            id: 4,
            text: "must not recover".into(),
            delivery: DeliveryMode::FollowUp,
            source: PromptSource::User,
        });

        let plan = plan_startup_recovery(Some(state));
        assert_eq!(plan.goal.as_deref(), Some("preserve this goal"));
        assert!(plan.invalid_queue);
        assert!(plan.history.is_empty());
        assert!(plan.queued_prompts.is_empty());
    }

    #[test]
    fn mcp_runtime_retries_only_failed_or_skipped_servers() {
        let config = McpConfig {
            servers: ["alpha", "beta", "gamma"]
                .into_iter()
                .map(|name| {
                    (
                        name.to_string(),
                        McpServerConfig::Http {
                            url: format!("https://{name}.example/mcp"),
                        },
                    )
                })
                .collect(),
        };
        let mut runtime = McpRuntime::new(config);
        let targets = runtime.configured_targets();
        runtime.mark_connecting(&targets);
        runtime.apply_report(&McpRegistrationReport {
            server_name: "alpha".into(),
            outcome: McpRegistrationOutcome::Connected {
                discovered_tools: 2,
                registered_tools: 2,
            },
        });
        runtime.apply_report(&McpRegistrationReport {
            server_name: "beta".into(),
            outcome: McpRegistrationOutcome::ConnectionFailed {
                error: "offline".into(),
            },
        });
        runtime.mark_skipped(&targets);

        assert_eq!(
            runtime.retry_targets(None).unwrap(),
            ["beta".to_string(), "gamma".to_string()]
                .into_iter()
                .collect()
        );
        assert_eq!(
            runtime.retry_targets(Some("beta")).unwrap(),
            ["beta".to_string()].into_iter().collect()
        );
        assert!(runtime
            .retry_targets(Some("alpha"))
            .unwrap_err()
            .to_string()
            .contains("already connected"));
        assert!(runtime
            .retry_targets(Some("missing"))
            .unwrap_err()
            .to_string()
            .contains("No configured MCP server"));

        let status = runtime.status();
        assert!(status.contains("MCP servers: 1/3 connected"));
        assert!(status.find("- alpha:").unwrap() < status.find("- beta:").unwrap());
        assert!(status.contains("alpha: connected · 2/2 tool(s) registered"));
        assert!(status.contains("beta: failed"));
        assert!(status.contains("gamma: skipped · retry available"));
    }

    #[test]
    fn permission_commands_render_sorted_policy_and_revoke_it() {
        let handler = MemoryPermissionHandler::new();
        handler.replace_remembered_policy(
            ["z_tool".to_string(), "a_tool".to_string()].into(),
            ["m_tool".to_string()].into(),
        );

        assert_eq!(
            run_permission_command(PermissionCommand::List, &handler),
            "Remembered tool permissions:\nAlways allow:\n  a_tool\n  z_tool\nAlways deny:\n  m_tool"
        );
        assert!(
            run_permission_command(PermissionCommand::Reset("a_tool"), &handler)
                .contains("next permissioned use will ask")
        );
        assert!(!handler.remembered_policy().always_allow.contains("a_tool"));
        assert_eq!(
            run_permission_command(PermissionCommand::Clear, &handler),
            "Cleared 2 remembered tool permission(s); their next permissioned use will ask."
        );
        assert_eq!(
            run_permission_command(PermissionCommand::List, &handler),
            "No remembered tool permissions; the next permissioned use of each tool will ask."
        );
    }

    #[test]
    fn tool_catalog_is_stable_searchable_and_never_executes_capabilities() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut agent = catalog_agent(executions.clone());

        let list = run_tools_command(ToolsCommand::List, &agent);
        assert!(list.contains("Model-facing tools: python"));
        assert!(list.contains("2 total (1 schema-on-demand)"));
        assert!(list.find("tools.alpha_read").unwrap() < list.find("tools.zeta_search").unwrap());
        assert!(list.contains("tools.zeta_search [schema on demand]"));
        assert!(!list.contains("records\nwith"));

        let search = run_tools_command(ToolsCommand::Search("REMOTE records"), &agent);
        assert!(search.contains("1 of 2"));
        assert!(search.contains("tools.zeta_search"));
        assert!(!search.contains("tools.alpha_read"));

        let detail = run_tools_command(ToolsCommand::Show("ALPHA_READ"), &agent);
        assert!(detail.contains("Tool tools.alpha_read"));
        assert!(detail.contains("compact signature and description are preloaded"));
        assert!(detail.contains("\"required\": ["));
        assert!(detail.contains("\"query\""));

        assert_eq!(
            run_tools_command(ToolsCommand::Show("update_goal"), &agent),
            "Tool 'update_goal' is advertised only while an active goal exists."
        );
        agent.set_goal(Some("finish catalog test".to_string()));
        let control = run_tools_command(ToolsCommand::Show("update_goal"), &agent);
        assert!(control.contains("model-facing host control"));
        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert!(agent.history().is_empty());
    }

    #[test]
    fn tool_catalog_caps_lists_and_oversized_details() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut agent = catalog_agent(executions.clone());
        for index in 0..TOOL_LIST_LIMIT {
            agent
                .registry
                .register(Arc::new(CatalogTool {
                    name: format!("bulk_{index:03}"),
                    description: "bulk catalog entry".to_string(),
                    code_only: false,
                    executions: executions.clone(),
                }))
                .unwrap();
        }
        agent
            .registry
            .register(Arc::new(CatalogTool {
                name: "oversized".to_string(),
                description: "x".repeat(TOOL_DETAIL_CHARS + 500),
                code_only: false,
                executions: executions.clone(),
            }))
            .unwrap();

        let list = run_tools_command(ToolsCommand::List, &agent);
        assert_eq!(
            list.lines()
                .filter(|line| line.starts_with("  tools."))
                .count(),
            TOOL_LIST_LIMIT
        );
        assert!(list.contains("more omitted; narrow the list"));

        let detail = run_tools_command(ToolsCommand::Show("oversized"), &agent);
        assert!(detail.contains("characters omitted; this display is bounded"));
        assert!(detail.chars().count() < TOOL_DETAIL_CHARS + 200);
        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert!(agent.history().is_empty());
    }

    #[tokio::test]
    async fn history_catalog_is_sorted_searchable_sanitized_and_non_mutating() {
        let home = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(home.path().to_path_buf(), WorkspaceScope::Global).unwrap();
        let mut alpha = SavedState::new(WorkspaceScope::Global, "openai".into(), "model-a".into());
        alpha
            .conversation_history
            .push(Message::user_text("ordinary alpha session"));
        store.save(&alpha, "alpha").unwrap();

        let mut zeta = SavedState::new(WorkspaceScope::Global, "openai".into(), "model-z".into());
        zeta.goal = Some("prospective archive goal".into());
        zeta.conversation_history.push(Message::user_text(
            "saved-session-search-needle from the user",
        ));
        zeta.conversation_history.push(Message::assistant(vec![
            ContentBlock::Thinking {
                thinking: "private reasoning".into(),
                signature: "private signature".into(),
            },
            ContentBlock::ToolUse {
                name: "bash".into(),
                input: json!({"secret": "private input"}),
                id: "private-id".into(),
            },
            ContentBlock::Text {
                text: "sanitized assistant text".into(),
            },
        ]));
        zeta.conversation_history
            .push(Message::user(vec![ContentBlock::ToolResult {
                content: "private output".into(),
                tool_use_id: "private-id".into(),
                is_error: Some(false),
            }]));
        let path = store.save(&zeta, "zeta").unwrap();
        let before = fs::read(&path).unwrap();

        let list = run_history_command(HistoryCommand::List, &store)
            .await
            .unwrap();
        assert!(list.find("  alpha").unwrap() < list.find("  zeta").unwrap());
        let search = run_history_command(HistoryCommand::Search("SEARCH-NEEDLE"), &store)
            .await
            .unwrap();
        assert!(search.contains("1 current-scope saved session(s)"));
        assert!(search.contains("zeta"));
        assert!(!search.contains("alpha"));
        let show = run_history_command(HistoryCommand::Show("zeta"), &store)
            .await
            .unwrap();
        for included in [
            "Saved session zeta",
            "prospective archive goal",
            "saved-session-search-needle",
            "sanitized assistant text",
            "tool: bash (input omitted)",
            "tool result: success (content omitted)",
        ] {
            assert!(show.contains(included), "missing {included}");
        }
        for excluded in [
            "private reasoning",
            "private signature",
            "private input",
            "private output",
            "private-id",
        ] {
            assert!(!show.contains(excluded), "displayed {excluded}");
        }
        assert!(run_history_command(HistoryCommand::Show("missing"), &store)
            .await
            .unwrap()
            .contains("No current-scope saved session"));
        assert!(run_history_command(HistoryCommand::Forget("zeta"), &store)
            .await
            .unwrap_err()
            .to_string()
            .contains("interactive host confirmation"));
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[test]
    fn persistence_rejects_an_invalid_tool_protocol_boundary() {
        let home = tempfile::tempdir().unwrap();
        let store = HistoryStore::open(home.path().to_path_buf(), WorkspaceScope::Global).unwrap();
        let mut state = SavedState::new(WorkspaceScope::Global, "openai".into(), "model".into());
        state
            .conversation_history
            .push(Message::assistant(vec![ContentBlock::ToolUse {
                name: "python".into(),
                input: json!({}),
                id: "dangling".into(),
            }]));

        let error = store.save(&state, "invalid").unwrap_err().to_string();
        assert!(error.contains("unpaired tool use/result"));
    }

    #[test]
    fn goal_slash_commands_parse_without_losing_objective_text() {
        assert_eq!(
            parse_local_command("/goal"),
            Some(LocalCommand::Goal(GoalCommand::Edit))
        );
        assert_eq!(
            parse_local_command("/GOAL edit"),
            Some(LocalCommand::Goal(GoalCommand::Edit))
        );
        assert_eq!(
            parse_local_command("/goal show"),
            Some(LocalCommand::Goal(GoalCommand::Show))
        );
        assert_eq!(
            parse_local_command("/goal clear"),
            Some(LocalCommand::Goal(GoalCommand::Clear))
        );
        assert_eq!(
            parse_local_command("  /goal   ship the async TUI  "),
            Some(LocalCommand::Goal(GoalCommand::Set("ship the async TUI")))
        );
        assert_eq!(parse_local_command("/exit"), Some(LocalCommand::Exit));
        assert_eq!(parse_local_command("ordinary prompt"), None);
    }

    #[test]
    fn active_goal_continues_only_after_normal_settlement() {
        assert!(should_continue_goal(
            true,
            false,
            &Ok(TurnOutcome::Completed)
        ));
        assert!(should_continue_goal(
            true,
            false,
            &Ok(TurnOutcome::MaxIterationsReached)
        ));
        assert!(!should_continue_goal(
            false,
            false,
            &Ok(TurnOutcome::Completed)
        ));
        assert!(!should_continue_goal(
            true,
            true,
            &Ok(TurnOutcome::Completed)
        ));
        assert!(!should_continue_goal(
            true,
            false,
            &Ok(TurnOutcome::Interrupted)
        ));
        assert!(!should_continue_goal(
            true,
            false,
            &Ok(TurnOutcome::PausedOnDenial)
        ));
        assert!(!should_continue_goal(
            true,
            false,
            &Ok(TurnOutcome::Refused)
        ));
        assert!(!should_continue_goal(
            true,
            false,
            &Err(Error::Other("provider failed".into()))
        ));
    }

    #[test]
    fn openrouter_kimi_is_the_default_remote_when_configured() {
        let keys = ApiKeys {
            anthropic: Some("anthropic-key".into()),
            openai: Some("openai-key".into()),
            openrouter: Some("openrouter-key".into()),
            openai_base_url: generalist::provider::openai::DEFAULT_BASE_URL.into(),
        };

        assert_eq!(
            keys.available_providers(),
            vec!["openrouter", "anthropic", "openai"]
        );
        assert_eq!(
            default_remote_provider_and_model(&keys),
            Some(("openrouter".to_string(), "moonshotai/kimi-k3".to_string()))
        );
    }

    #[test]
    fn registry_has_permissioned_archive_tools() {
        let home = tempfile::tempdir().unwrap();
        let profile = ProfilePaths::new(home.path());
        let history = HistoryStore::open_profile(&profile, WorkspaceScope::Global).unwrap();
        let memory = EpisodicMemory::open_scoped(
            profile.memory_database().to_path_buf(),
            WorkspaceScope::Global,
        )
        .unwrap();
        let registry = build_registry(
            profile.todo_file(),
            &MemoryPermissionHandler::new(),
            &history,
            Some(&memory),
        )
        .unwrap();
        for expected in [
            "search_memories",
            "read_memory",
            "search_conversations",
            "read_conversation",
        ] {
            assert!(registry.tool_names().iter().any(|name| name == expected));
        }
        assert!(!include_str!("../SYSTEM_PROMPT.md").contains("enhanced_memory"));
    }
}
