use colored::*;
use generalist::provider::{
    anthropic, openrouter, AnthropicProvider, OpenAiProvider, OpenRouterProvider, Provider,
};
use generalist::tools::*;
use generalist::tui::{TerminalUi, UiAction};
use generalist::{
    history_tool_protocol_is_valid, is_local_command, parse_local_command, truncate_middle, Agent,
    AgentEvent, DeliveryMode, Error, GoalCommand, LocalCommand, MemoryPermissionHandler,
    PermissionBrokerPrompt, PermissionChoice, PermissionRequest, PermissionUiEvent, PromptQueue,
    Result, SavedState, ToolRegistry, TurnControl, TurnOutcome,
};
use std::env;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

const AUTOSAVE_NAME: &str = "autosave";
const PENDING_QUEUE_NAME: &str = "pending-queue";
const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
const DEFAULT_LOCAL_MODEL: &str = "qwen3.6:35b-a3b";

fn home_dir() -> PathBuf {
    #[allow(deprecated)]
    env::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

fn history_dir_for(home: &Path) -> Result<PathBuf> {
    let dir = home.join(".generalist").join("history");
    fs::create_dir_all(&dir).map_err(|error| {
        Error::Other(format!(
            "Failed to create state directory {}: {error}",
            dir.display()
        ))
    })?;
    Ok(dir)
}

fn history_dir() -> Result<PathBuf> {
    history_dir_for(&home_dir())
}

fn state_search_dirs() -> Vec<PathBuf> {
    let home = home_dir();
    let mut dirs = Vec::new();
    if let Ok(current) = history_dir_for(&home) {
        dirs.push(current);
    }
    // Older releases used both paths. `.generalist_history` may also be a
    // regular readline-history file, so only read_dir/read_to_string callers
    // decide whether a particular candidate has the shape they need.
    dirs.push(home.join(".generalist_history"));
    dirs.push(home.join(".chatbot_history"));
    dirs
}

fn save_state(state: &SavedState, filename: &str) -> Result<PathBuf> {
    let filepath = history_dir()?.join(format!("{}.json", filename));
    let json_data = serialize_state(state)?;
    write_atomically(&filepath, &json_data)?;
    Ok(filepath)
}

fn serialize_state(state: &SavedState) -> Result<Vec<u8>> {
    if !history_tool_protocol_is_valid(&state.conversation_history) {
        return Err(Error::Other(
            "Refusing to persist history with an unpaired tool use/result".to_string(),
        ));
    }
    serde_json::to_vec_pretty(state)
        .map_err(|error| Error::Other(format!("Failed to serialize state: {error}")))
}

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

fn load_state(filename: &str) -> Result<SavedState> {
    for dir in state_search_dirs() {
        let filepath = dir.join(format!("{}.json", filename));
        if let Ok(json_data) = fs::read_to_string(&filepath) {
            return SavedState::from_legacy_json(&json_data, anthropic::SUGGESTED_MODELS[0])
                .ok_or_else(|| Error::Other(format!("Failed to parse {}", filepath.display())));
        }
    }
    Err(Error::Other(format!(
        "No saved conversation named '{}'",
        filename
    )))
}

fn list_saved_conversations() -> Vec<String> {
    let mut conversations = Vec::new();
    for dir in state_search_dirs() {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(stem) = name.strip_suffix(".json") {
                        if stem == PENDING_QUEUE_NAME {
                            continue;
                        }
                        if !conversations.contains(&stem.to_string()) {
                            conversations.push(stem.to_string());
                        }
                    }
                }
            }
        }
    }
    conversations.sort();
    conversations
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

fn build_registry(permission_handler: &MemoryPermissionHandler) -> Result<ToolRegistry> {
    let mut registry = ToolRegistry::with_permission_handler(Box::new(permission_handler.clone()));
    registry.register(Arc::new(ReadFileTool))?;
    registry.register(Arc::new(PatchFileTool))?;
    registry.register(Arc::new(ListDirectoryTool))?;
    registry.register(Arc::new(BashTool))?;
    registry.register(Arc::new(WeatherTool))?;
    registry.register(Arc::new(HttpFetchTool))?;
    registry.register(Arc::new(EnhancedMemoryTool::new()?))?;
    registry.register(Arc::new(WikipediaTool))?;
    registry.register(Arc::new(Z3SolverTool))?;
    registry.register(Arc::new(TodoTool))?;
    registry.register(Arc::new(FirecrawlCrawlTool))?;
    registry.register(Arc::new(FirecrawlSearchTool))?;
    registry.register(Arc::new(FirecrawlMapTool))?;
    registry.register(Arc::new(FirecrawlExtractTool))?;
    Ok(registry)
}

fn make_saved_state(
    agent: &Agent,
    handler: &MemoryPermissionHandler,
    queue: &PromptQueue,
) -> SavedState {
    SavedState {
        provider: agent.provider().id().to_string(),
        model: agent.provider().model().to_string(),
        goal: agent.goal().map(str::to_string),
        conversation_history: agent.history.clone(),
        always_allow_tools: handler.always_allow().lock().unwrap().clone(),
        always_deny_tools: handler.always_deny().lock().unwrap().clone(),
        queued_prompts: queue.snapshot(),
    }
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
            history: agent.history.clone(),
        }
    }

    fn save(
        &self,
        permission_handler: &MemoryPermissionHandler,
        queue: &PromptQueue,
    ) -> Result<()> {
        save_state(
            &SavedState {
                provider: self.provider.clone(),
                model: self.model.clone(),
                goal: self.goal.clone(),
                conversation_history: self.history.clone(),
                always_allow_tools: permission_handler.always_allow().lock().unwrap().clone(),
                always_deny_tools: permission_handler.always_deny().lock().unwrap().clone(),
                queued_prompts: queue.snapshot(),
            },
            AUTOSAVE_NAME,
        )?;
        Ok(())
    }
}

fn apply_runtime_event(
    ui: &mut TerminalUi,
    queue: &PromptQueue,
    permission_handler: &MemoryPermissionHandler,
    durable: &mut DurableBoundary,
    event: AgentEvent,
) {
    let steering = matches!(&event, AgentEvent::SteeringCommitted { .. });
    match event {
        AgentEvent::HistoryCheckpoint {
            history,
            context_tokens,
        } => {
            ui.set_context_tokens(context_tokens);
            durable.history = history;
            if let Err(error) = durable.save(permission_handler, queue) {
                ui.error(&format!("Failed to persist runtime checkpoint: {error}"));
            }
        }
        event => ui.handle_agent_event(event),
    }
    if steering {
        ui.sync_queue(queue);
    }
}

struct CliArgs {
    local_model: Option<String>,
}

fn print_usage() {
    println!("Usage: generalist [--local [model]]");
    println!();
    println!("  --local [model]   Run against a local OpenAI-compatible server");
    println!(
        "                    (default {}, override with OPENAI_BASE_URL).",
        OLLAMA_BASE_URL
    );
    println!(
        "                    Model defaults to {} if omitted.",
        DEFAULT_LOCAL_MODEL
    );
    println!("  -h, --help        Show this help");
}

fn parse_args() -> CliArgs {
    let mut local_model = None;
    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("{} {}", "Unknown argument:".red(), other);
                print_usage();
                std::process::exit(1);
            }
        }
    }
    CliArgs { local_model }
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

async fn drive_started_turn(
    agent: &mut Agent,
    ui: &mut TerminalUi,
    queue: &PromptQueue,
    permission_rx: &mut mpsc::UnboundedReceiver<PermissionUiEvent>,
    permission_handler: &MemoryPermissionHandler,
) -> Result<bool> {
    let mut durable = DurableBoundary::from_agent(agent);
    let (cancel_handle, mut control) = TurnControl::for_turn(queue.clone());
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (checkpoint_tx, mut checkpoint_rx) = mpsc::unbounded_channel();
    let mut exit_requested = false;

    ui.set_busy(true, "Thinking");
    ui.draw().map_err(|error| Error::Other(error.to_string()))?;

    let outcome = {
        let mut on_event = move |event: AgentEvent| {
            if matches!(&event, AgentEvent::HistoryCheckpoint { .. }) {
                let _ = checkpoint_tx.send(event);
            } else {
                let _ = event_tx.send(event);
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
                        permission_handler,
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
                            ui.status(&format!(
                                "{} was {} by remembered policy",
                                request.tool_name,
                                if allowed { "auto-allowed" } else { "auto-denied" }
                            ));
                        }
                    }
                }
                _ = ticker.tick() => {
                    ui.tick();
                    ui.draw_if_dirty().map_err(|error| Error::Other(error.to_string()))?;
                }
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
                        if let Err(error) = durable.save(permission_handler, queue) {
                            ui.error(&format!("Failed to persist runtime state: {error}"));
                        }
                    }
                }
                // Keep ordinary display events last in this biased reactor:
                // an unbounded stream of deltas must not starve frame ticks,
                // terminal input, or a live permission decision.
                Some(event) = event_rx.recv() => {
                    apply_runtime_event(
                        ui,
                        queue,
                        permission_handler,
                        &mut durable,
                        event,
                    );
                }
            }
        }
    };

    while let Ok(event) = checkpoint_rx.try_recv() {
        apply_runtime_event(ui, queue, permission_handler, &mut durable, event);
    }
    while let Ok(event) = event_rx.try_recv() {
        apply_runtime_event(ui, queue, permission_handler, &mut durable, event);
    }
    queue.normalize_steers();
    ui.sync_queue(queue);
    if let Err(error) = save_state(
        &make_saved_state(agent, permission_handler, queue),
        AUTOSAVE_NAME,
    ) {
        ui.error(&format!("Failed to persist settled runtime state: {error}"));
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

async fn drive_compaction(
    agent: &mut Agent,
    ui: &mut TerminalUi,
    queue: &PromptQueue,
    permission_rx: &mut mpsc::UnboundedReceiver<PermissionUiEvent>,
    permission_handler: &MemoryPermissionHandler,
) -> Result<bool> {
    let mut durable = DurableBoundary::from_agent(agent);
    let before = agent.context_tokens();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (checkpoint_tx, mut checkpoint_rx) = mpsc::unbounded_channel();
    let mut exit_requested = false;
    ui.set_busy(true, "Compacting context");

    let compacted = {
        let mut on_event = move |event: AgentEvent| {
            if matches!(&event, AgentEvent::HistoryCheckpoint { .. }) {
                let _ = checkpoint_tx.send(event);
            } else {
                let _ = event_tx.send(event);
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
                    permission_handler,
                    &mut durable,
                    event,
                ),
                result = &mut operation => break Some(result),
                Some(permission_event) = permission_rx.recv() => {
                    match permission_event {
                        PermissionUiEvent::Request(request) => {
                            let _ = request.reply.send(PermissionChoice::DenyOnce);
                        }
                        PermissionUiEvent::Automatic { request, allowed } => {
                            ui.status(&format!(
                                "{} was {} by remembered policy",
                                request.tool_name,
                                if allowed { "auto-allowed" } else { "auto-denied" }
                            ));
                        }
                    }
                }
                _ = ticker.tick() => {
                    ui.tick();
                    ui.draw_if_dirty().map_err(|error| Error::Other(error.to_string()))?;
                }
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
                        if let Err(error) = durable.save(permission_handler, queue) {
                            ui.error(&format!("Failed to persist runtime state: {error}"));
                        }
                    }
                }
                // See the active-turn reactor above: display backlog is lower
                // priority than interaction and bounded frame progress.
                Some(event) = event_rx.recv() => apply_runtime_event(
                    ui,
                    queue,
                    permission_handler,
                    &mut durable,
                    event,
                ),
            }
        }
    };

    while let Ok(event) = checkpoint_rx.try_recv() {
        apply_runtime_event(ui, queue, permission_handler, &mut durable, event);
    }
    while let Ok(event) = event_rx.try_recv() {
        apply_runtime_event(ui, queue, permission_handler, &mut durable, event);
    }
    if let Err(error) = save_state(
        &make_saved_state(agent, permission_handler, queue),
        AUTOSAVE_NAME,
    ) {
        ui.error(&format!("Failed to persist compaction state: {error}"));
    }
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

fn replace_goal(agent: &mut Agent, ui: &mut TerminalUi, goal: Option<String>) {
    agent.set_goal(goal);
    ui.set_goal(agent.goal());
    ui.set_context_tokens(agent.context_tokens());
    if let Some(goal) = agent.goal() {
        ui.info(&format!("Active goal set: {}", truncate_middle(goal, 400)));
    } else {
        ui.info("Active goal cleared.");
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_command(
    text: &str,
    agent: &mut Agent,
    ui: &mut TerminalUi,
    queue: &PromptQueue,
    permission_rx: &mut mpsc::UnboundedReceiver<PermissionUiEvent>,
    permission_handler: &MemoryPermissionHandler,
    keys: &ApiKeys,
    available: &[&'static str],
) -> Result<CommandFlow> {
    let command = parse_local_command(text).unwrap_or(LocalCommand::Unknown(text.trim()));
    match command {
        LocalCommand::Exit => return Ok(CommandFlow::Exit),
        LocalCommand::Help => terminal(ui.show_help().await)?,
        LocalCommand::Compact => {
            if drive_compaction(agent, ui, queue, permission_rx, permission_handler).await? {
                return Ok(CommandFlow::Exit);
            }
        }
        LocalCommand::Clear => {
            agent.clear_history();
            ui.clear_conversation();
            ui.set_context_tokens(0);
            ui.info("Conversation cleared. The active goal was preserved.");
        }
        LocalCommand::Save => {
            let default = format!("chat_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
            if let Some(name) = terminal(ui.prompt("Save conversation as", &default).await)? {
                match save_state(&make_saved_state(agent, permission_handler, queue), &name) {
                    Ok(path) => ui.info(&format!("Saved to {}", path.display())),
                    Err(error) => ui.error(&format!("Failed to save: {error}")),
                }
            }
        }
        LocalCommand::Load => {
            let saved = list_saved_conversations();
            if saved.is_empty() {
                ui.info("No saved conversations found.");
            } else if let Some(index) = terminal(ui.select("Load conversation", &saved).await)? {
                match load_state(&saved[index]) {
                    Ok(state) => {
                        let SavedState {
                            provider,
                            model,
                            goal,
                            conversation_history,
                            always_allow_tools,
                            always_deny_tools,
                            queued_prompts,
                        } = state;
                        if !history_tool_protocol_is_valid(&conversation_history) {
                            ui.error(
                                "Saved conversation has an unpaired tool use/result; refusing to load it.",
                            );
                            return Ok(CommandFlow::Continue);
                        }
                        match build_provider(keys, &provider, model) {
                            Ok(provider) => agent.set_provider(provider),
                            Err(error) => ui.error(&format!(
                                "Saved API '{provider}' is unavailable ({error}); keeping the current API."
                            )),
                        }
                        permission_handler.set_always_allow(always_allow_tools);
                        permission_handler.set_always_deny(always_deny_tools);
                        agent.set_goal(goal);
                        agent.replace_history(conversation_history);
                        queue.replace(queued_prompts);
                        ui.load_history(&agent.history);
                        ui.set_goal(agent.goal());
                        ui.sync_queue(queue);
                        ui.set_session(
                            agent.provider().display_name(),
                            agent.provider().model(),
                            agent.registry.tool_names().len(),
                        );
                        ui.set_context_tokens(agent.context_tokens());
                        ui.info(&format!("Loaded {} messages", agent.history.len()));
                    }
                    Err(error) => ui.error(&format!("Failed to load: {error}")),
                }
            }
        }
        LocalCommand::Model => {
            if let Some((provider_name, model)) =
                choose_provider_and_model(ui, keys, available).await?
            {
                match build_provider(keys, &provider_name, model) {
                    Ok(provider) => {
                        agent.set_provider(provider);
                        ui.set_session(
                            agent.provider().display_name(),
                            agent.provider().model(),
                            agent.registry.tool_names().len(),
                        );
                        ui.info("Model switched.");
                    }
                    Err(error) => ui.error(&format!("Failed to switch model: {error}")),
                }
            }
        }
        LocalCommand::Goal(GoalCommand::Edit) => {
            let current = agent.goal().unwrap_or_default();
            if let Some(goal) = terminal(ui.prompt("Active goal (empty clears)", current).await)? {
                replace_goal(agent, ui, Some(goal));
            }
        }
        LocalCommand::Goal(GoalCommand::Show) => {
            if let Some(goal) = agent.goal() {
                ui.info(&format!("Active goal: {goal}"));
            } else {
                ui.info("No active goal. Use /goal <objective> to set one.");
            }
        }
        LocalCommand::Goal(GoalCommand::Clear) => replace_goal(agent, ui, None),
        LocalCommand::Goal(GoalCommand::Set(goal)) => {
            replace_goal(agent, ui, Some(goal.to_string()))
        }
        LocalCommand::Unknown(command) => {
            ui.info(&format!("Unknown local command: {command}. Use /help."));
        }
    }
    Ok(CommandFlow::Continue)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = parse_args();

    let env_path = home_dir().join(".generalist.env");
    if env_path.exists() {
        dotenv::from_path(&env_path).ok();
    }

    let mut keys = ApiKeys::from_env();
    if cli.local_model.is_some() {
        if env::var("OPENAI_BASE_URL").is_err() {
            keys.openai_base_url = OLLAMA_BASE_URL.to_string();
        }
        keys.openai.get_or_insert_with(|| "unused".to_string());
    }
    let available = keys.available_providers();
    if available.is_empty() {
        eprintln!("{}", "No API key found.".red());
        eprintln!("Set at least one of these (in the environment or ~/.generalist.env):");
        eprintln!("  ANTHROPIC_API_KEY=...   for Anthropic models");
        eprintln!("  OPENAI_API_KEY=...      for OpenAI or a compatible server");
        eprintln!("  OPENROUTER_API_KEY=...  for OpenRouter (defaults to Kimi K3)");
        eprintln!("  OPENAI_BASE_URL=...     optional, e.g. {OLLAMA_BASE_URL} for Ollama");
        eprintln!("Or run against a local model directly: generalist --local <model>");
        std::process::exit(1);
    }

    let mut ui = terminal(TerminalUi::start("Starting", "selecting model"))?;
    let provider_and_model = match cli.local_model {
        Some(model) => Some(("openai".to_string(), model)),
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
    let mut registry = build_registry(&permission_handler)?;

    ui.set_busy(true, "Connecting tools");
    terminal(ui.draw())?;
    if let Some(config) = generalist::mcp::McpConfig::load(&home_dir().join(".generalist/mcp.json"))
    {
        for line in generalist::mcp::register_servers(&mut registry, &config).await {
            ui.info(&line);
        }
    }
    ui.set_busy(false, "Ready");

    let mut system_prompt = include_str!("../SYSTEM_PROMPT.md").to_string();
    if let Some(index) = generalist::skills::skills_index(&home_dir().join(".generalist/skills")) {
        system_prompt.push_str(&index);
    }
    for name in ["AGENTS.md", "CLAUDE.md"] {
        if let Ok(notes) = fs::read_to_string(name) {
            system_prompt.push_str(&format!("\n\n## Project notes (./{name})\n\n{notes}"));
            break;
        }
    }

    let mut agent = Agent::new(provider, registry, system_prompt);
    let queue = match load_state(AUTOSAVE_NAME) {
        Ok(mut state) => {
            // The active goal is independent of crash recovery: preserve it
            // across restarts even when there is no queued turn to resume.
            agent.set_goal(state.goal.take());
            if state.queued_prompts.is_empty() {
                PromptQueue::default()
            } else if history_tool_protocol_is_valid(&state.conversation_history) {
                let SavedState {
                    conversation_history,
                    always_allow_tools,
                    always_deny_tools,
                    queued_prompts,
                    ..
                } = state;
                let count = queued_prompts.len();
                permission_handler.set_always_allow(always_allow_tools);
                permission_handler.set_always_deny(always_deny_tools);
                agent.replace_history(conversation_history);
                ui.load_history(&agent.history);
                ui.info(&format!(
                    "Recovered {count} queued message(s) with their conversation context."
                ));
                PromptQueue::from_saved(queued_prompts)
            } else {
                ui.error(
                    "Autosave has an unpaired tool use/result; queued work was not recovered.",
                );
                PromptQueue::default()
            }
        }
        _ => PromptQueue::default(),
    };
    queue.normalize_steers();
    ui.sync_queue(&queue);
    ui.set_goal(agent.goal());
    ui.set_session(
        agent.provider().display_name(),
        agent.provider().model(),
        agent.registry.tool_names().len(),
    );
    ui.set_context_tokens(agent.context_tokens());

    let mut ticker = tokio::time::interval(Duration::from_millis(50));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut exiting = false;

    while !exiting {
        queue.normalize_steers();
        ui.sync_queue(&queue);

        if let Some(claim) = queue.claim_follow_up() {
            let prompt = claim.prompts()[0].clone();
            if is_local_command(&prompt.text) {
                claim.commit();
                exiting = matches!(
                    execute_command(
                        &prompt.text,
                        &mut agent,
                        &mut ui,
                        &queue,
                        &mut permission_rx,
                        &permission_handler,
                        &keys,
                        &available,
                    )
                    .await?,
                    CommandFlow::Exit
                );
            } else {
                agent.begin_turn(&prompt.text);
                claim.commit();
                ui.push_user(prompt.text.trim());
                if let Err(error) = save_state(
                    &make_saved_state(&agent, &permission_handler, &queue),
                    AUTOSAVE_NAME,
                ) {
                    ui.error(&format!("Failed to persist the started turn: {error}"));
                }
                exiting = drive_started_turn(
                    &mut agent,
                    &mut ui,
                    &queue,
                    &mut permission_rx,
                    &permission_handler,
                )
                .await?;
            }
            if let Err(error) = save_state(
                &make_saved_state(&agent, &permission_handler, &queue),
                AUTOSAVE_NAME,
            ) {
                ui.error(&format!("Failed to persist settled runtime state: {error}"));
            }
            continue;
        }

        ui.set_busy(false, "Ready");
        tokio::select! {
            biased;
            Some(permission_event) = permission_rx.recv() => {
                match permission_event {
                    PermissionUiEvent::Request(request) => {
                        let _ = request.reply.send(PermissionChoice::DenyOnce);
                        ui.error("Ignored a stale permission request with no active turn.");
                    }
                    PermissionUiEvent::Automatic { request, allowed } => {
                        ui.status(&format!(
                            "{} was {} by remembered policy",
                            request.tool_name,
                            if allowed { "auto-allowed" } else { "auto-denied" }
                        ));
                    }
                }
            }
            _ = ticker.tick() => {
                ui.tick();
                ui.draw_if_dirty().map_err(|error| Error::Other(error.to_string()))?;
            }
            terminal_event = ui.next_event() => {
                let action =
                    terminal(ui.handle_event(terminal(terminal_event)?, &queue))?;
                let persist_queue = action.requires_queue_persist();
                match action {
                    UiAction::Submit { text, delivery } => {
                        enqueue_submission(&mut ui, &queue, text, delivery, false);
                    }
                    UiAction::Exit => exiting = true,
                    UiAction::None
                    | UiAction::QueueChanged
                    | UiAction::Interrupt
                    | UiAction::Permission { .. } => {}
                }
                if persist_queue {
                    if let Err(error) = save_state(
                        &make_saved_state(&agent, &permission_handler, &queue),
                        AUTOSAVE_NAME,
                    ) {
                        ui.error(&format!("Failed to persist runtime state: {error}"));
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use generalist::{ContentBlock, Message};
    use serde_json::json;

    #[test]
    fn structured_state_does_not_collide_with_legacy_input_history_file() {
        let home = tempfile::tempdir().unwrap();
        let legacy = home.path().join(".generalist_history");
        fs::write(&legacy, "old input history\n").unwrap();

        let directory = history_dir_for(home.path()).unwrap();
        assert_eq!(directory, home.path().join(".generalist/history"));
        assert!(directory.is_dir());
        assert_eq!(fs::read_to_string(&legacy).unwrap(), "old input history\n");

        let autosave = directory.join("autosave.json");
        write_atomically(&autosave, br#"{"version":1}"#).unwrap();
        write_atomically(&autosave, br#"{"version":2}"#).unwrap();
        assert_eq!(fs::read_to_string(autosave).unwrap(), r#"{"version":2}"#);
    }

    #[test]
    fn persistence_rejects_an_invalid_tool_protocol_boundary() {
        let mut state = SavedState::new("openai".into(), "model".into());
        state
            .conversation_history
            .push(Message::assistant(vec![ContentBlock::ToolUse {
                name: "python".into(),
                input: json!({}),
                id: "dangling".into(),
            }]));

        let error = serialize_state(&state).unwrap_err().to_string();
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
}
