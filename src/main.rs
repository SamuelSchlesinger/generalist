use colored::*;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use generalist::chat_ui::ChatUI;
use generalist::provider::{anthropic, AnthropicProvider, OpenAiProvider, Provider};
use generalist::tools::*;
use generalist::{
    Agent, AgentEvent, Error, MemoryPermissionHandler, Result, SavedState, ToolRegistry,
    TurnOutcome,
};
use indicatif::ProgressBar;
use std::cell::RefCell;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

const AUTOSAVE_NAME: &str = "autosave";

fn home_dir() -> PathBuf {
    #[allow(deprecated)]
    env::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

fn history_dir() -> PathBuf {
    let dir = home_dir().join(".generalist_history");
    fs::create_dir_all(&dir).ok();
    dir
}

/// Legacy history location from earlier versions.
fn legacy_history_dir() -> PathBuf {
    home_dir().join(".chatbot_history")
}

fn save_state(state: &SavedState, filename: &str) -> Result<PathBuf> {
    let filepath = history_dir().join(format!("{}.json", filename));
    let json_data = serde_json::to_string_pretty(state)
        .map_err(|e| Error::Other(format!("Failed to serialize state: {}", e)))?;
    fs::write(&filepath, json_data)
        .map_err(|e| Error::Other(format!("Failed to write state file: {}", e)))?;
    Ok(filepath)
}

fn load_state(filename: &str) -> Result<SavedState> {
    for dir in [history_dir(), legacy_history_dir()] {
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
    for dir in [history_dir(), legacy_history_dir()] {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(stem) = name.strip_suffix(".json") {
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
    openai_base_url: String,
}

impl ApiKeys {
    fn from_env() -> Self {
        let openai_base_url = env::var("OPENAI_BASE_URL").ok();
        Self {
            // CLAUDE_API_KEY is the name earlier versions used.
            anthropic: env::var("ANTHROPIC_API_KEY")
                .or_else(|_| env::var("CLAUDE_API_KEY"))
                .ok(),
            // Local servers (Ollama, LM Studio, vLLM) don't check the key, so
            // a configured base URL is enough to enable the provider.
            openai: env::var("OPENAI_API_KEY")
                .ok()
                .or_else(|| openai_base_url.as_ref().map(|_| "unused".to_string())),
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
        providers
    }
}

fn choose_model(provider: &str) -> String {
    match provider {
        "anthropic" => {
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select model")
                .items(anthropic::SUGGESTED_MODELS)
                .default(0)
                .interact()
                .unwrap_or(0);
            anthropic::SUGGESTED_MODELS[selection].to_string()
        }
        _ => {
            let default = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
            Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Model name (e.g. gpt-4o, or an Ollama model like qwen3:30b)")
                .default(default)
                .interact_text()
                .unwrap_or_else(|_| "gpt-4o".to_string())
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
        other => Err(Error::Other(format!("Unknown provider '{}'", other))),
    }
}

fn build_registry(permission_handler: &MemoryPermissionHandler) -> Result<ToolRegistry> {
    let shared = MemoryPermissionHandler::with_shared_state(
        permission_handler.always_allow(),
        permission_handler.always_deny(),
    );
    let mut registry = ToolRegistry::with_permission_handler(Box::new(shared));
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

fn make_saved_state(agent: &Agent, handler: &MemoryPermissionHandler) -> SavedState {
    SavedState {
        provider: agent.provider().id().to_string(),
        model: agent.provider().model().to_string(),
        conversation_history: agent.history.clone(),
        always_allow_tools: handler.always_allow().lock().unwrap().clone(),
        always_deny_tools: handler.always_deny().lock().unwrap().clone(),
    }
}

const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
const DEFAULT_LOCAL_MODEL: &str = "qwen3.6:35b-a3b";

struct CliArgs {
    /// `--local [model]`: skip provider selection, run against a local
    /// OpenAI-compatible server (Ollama by default).
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
                // Model name is optional: `--local` alone uses the default,
                // and a following flag is not a model name.
                let model = match args.peek() {
                    Some(next) if !next.starts_with('-') => args.next().unwrap(),
                    _ => DEFAULT_LOCAL_MODEL.to_string(),
                };
                local_model = Some(model);
            }
            a if a.starts_with("--local=") => {
                local_model = Some(a["--local=".len()..].to_string());
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = parse_args();

    // ~/.generalist.env is optional; plain environment variables work too.
    let env_path = home_dir().join(".generalist.env");
    if env_path.exists() {
        dotenv::from_path(&env_path).ok();
    }

    let mut keys = ApiKeys::from_env();
    if cli.local_model.is_some() {
        // --local needs no API key; point at Ollama unless a base URL was
        // configured explicitly.
        if env::var("OPENAI_BASE_URL").is_err() {
            keys.openai_base_url = OLLAMA_BASE_URL.to_string();
        }
        keys.openai.get_or_insert_with(|| "unused".to_string());
    }
    let available = keys.available_providers();
    if available.is_empty() {
        eprintln!("{}", "No API key found.".red());
        eprintln!("Set at least one of these (in the environment or in ~/.generalist.env):");
        eprintln!("  ANTHROPIC_API_KEY=...   for Anthropic models");
        eprintln!("  OPENAI_API_KEY=...      for OpenAI or any OpenAI-compatible server");
        eprintln!(
            "  OPENAI_BASE_URL=...     optional, e.g. {} for Ollama",
            OLLAMA_BASE_URL
        );
        eprintln!("Or run against a local model directly:  generalist --local <model>");
        std::process::exit(1);
    }

    let ui = ChatUI::new();

    let (provider_name, model) = match &cli.local_model {
        Some(model) => ("openai", model.clone()),
        None => {
            let provider_name = if available.len() == 1 {
                available[0]
            } else {
                let selection = Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("Select provider")
                    .items(&available)
                    .default(0)
                    .interact()
                    .unwrap_or(0);
                available[selection]
            };
            (provider_name, choose_model(provider_name))
        }
    };
    let provider = build_provider(&keys, provider_name, model)?;

    let permission_handler = MemoryPermissionHandler::new();
    let mut registry = build_registry(&permission_handler)?;

    // MCP servers from ~/.generalist/mcp.json; their tools are code-only
    // (callable from scripts via the tools module, not direct tool calls).
    if let Some(config) = generalist::mcp::McpConfig::load(&home_dir().join(".generalist/mcp.json"))
    {
        for line in generalist::mcp::register_servers(&mut registry, &config).await {
            ui.print_info(&line);
        }
    }

    // System prompt = base + skills index + project notes, if present.
    let mut system_prompt = include_str!("../SYSTEM_PROMPT.md").to_string();
    if let Some(index) = generalist::skills::skills_index(&home_dir().join(".generalist/skills")) {
        system_prompt.push_str(&index);
    }
    for name in ["AGENTS.md", "CLAUDE.md"] {
        if let Ok(notes) = fs::read_to_string(name) {
            system_prompt.push_str(&format!("\n\n## Project notes (./{})\n\n{}", name, notes));
            break;
        }
    }

    let mut agent = Agent::new(provider, registry, system_prompt);

    ui.print_welcome(
        agent.provider().id(),
        agent.provider().model(),
        &agent.registry.tool_names(),
    );

    // Exits on Ctrl-C / closed stdin (Err) or the exit command.
    while let Ok(input) = Input::<String>::with_theme(&ColorfulTheme::default())
        .with_prompt("You")
        .interact_text()
    {
        let trimmed = input.trim();

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("\n{}", "Goodbye!".yellow());
            break;
        } else if trimmed.eq_ignore_ascii_case("/help") {
            ui.print_help();
            continue;
        } else if trimmed.eq_ignore_ascii_case("/compact") {
            let before = agent.context_tokens();
            let mut on_note = |event: AgentEvent| {
                if let AgentEvent::Notice(message) = event {
                    ui.print_info(&message);
                }
            };
            match agent.compact(&mut on_note).await {
                Ok(true) => {}
                Ok(false) => ui.print_info("Nothing to compact yet."),
                Err(e) => ui.print_error(&format!("Compaction failed: {}", e)),
            }
            ui.print_info(&format!(
                "Context: ~{}k -> ~{}k tokens",
                before / 1000,
                agent.context_tokens() / 1000
            ));
            continue;
        } else if trimmed.eq_ignore_ascii_case("/clear") {
            agent.history.clear();
            ui.print_info("Conversation cleared.");
            continue;
        } else if trimmed.eq_ignore_ascii_case("/save") {
            let name: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Save conversation as")
                .default(format!(
                    "chat_{}",
                    chrono::Local::now().format("%Y%m%d_%H%M%S")
                ))
                .interact_text()
                .unwrap_or_else(|_| "unnamed".to_string());
            match save_state(&make_saved_state(&agent, &permission_handler), &name) {
                Ok(path) => ui.print_info(&format!("Saved to {}", path.display())),
                Err(e) => ui.print_error(&format!("Failed to save: {}", e)),
            }
            continue;
        } else if trimmed.eq_ignore_ascii_case("/load") {
            let saved = list_saved_conversations();
            if saved.is_empty() {
                ui.print_info("No saved conversations found.");
                continue;
            }
            let Some(idx) = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select conversation to load")
                .items(&saved)
                .interact_opt()
                .ok()
                .flatten()
            else {
                continue;
            };
            match load_state(&saved[idx]) {
                Ok(state) => {
                    match build_provider(&keys, &state.provider, state.model.clone()) {
                        Ok(provider) => agent.set_provider(provider),
                        Err(e) => {
                            ui.print_error(&format!(
                                "Conversation used provider '{}' which is unavailable ({}); keeping current provider.",
                                state.provider, e
                            ));
                        }
                    }
                    permission_handler.set_always_allow(state.always_allow_tools.clone());
                    permission_handler.set_always_deny(state.always_deny_tools.clone());
                    agent.history = state.conversation_history;
                    ui.print_info(&format!(
                        "Loaded {} messages on {} / {}",
                        agent.history.len(),
                        agent.provider().id(),
                        agent.provider().model()
                    ));
                    for msg in &agent.history {
                        let text = msg.text();
                        if !text.is_empty() {
                            ui.print_message(&msg.role, &text);
                        }
                    }
                    println!();
                }
                Err(e) => ui.print_error(&format!("Failed to load: {}", e)),
            }
            continue;
        } else if trimmed.eq_ignore_ascii_case("/model") {
            let provider_name = if available.len() == 1 {
                available[0]
            } else {
                let selection = Select::with_theme(&ColorfulTheme::default())
                    .with_prompt("Select provider")
                    .items(&available)
                    .default(0)
                    .interact()
                    .unwrap_or(0);
                available[selection]
            };
            let model = choose_model(provider_name);
            match build_provider(&keys, provider_name, model) {
                Ok(provider) => {
                    agent.set_provider(provider);
                    ui.print_info(&format!(
                        "Switched to {} / {}",
                        agent.provider().id(),
                        agent.provider().model()
                    ));
                }
                Err(e) => ui.print_error(&format!("Failed to switch: {}", e)),
            }
            continue;
        } else if trimmed.is_empty() {
            continue;
        }

        // Run the turn, rendering events as they arrive. The spinner lives in
        // a RefCell so the event closure can start/stop it; ApiCallFinished is
        // guaranteed to fire even on errors, keeping the state balanced.
        let spinner: RefCell<Option<ProgressBar>> = RefCell::new(None);
        let streaming = RefCell::new(false);
        let clear_spinner = || {
            if let Some(pb) = spinner.borrow_mut().take() {
                pb.finish_and_clear();
            }
        };
        let mut on_event = |event: AgentEvent| match event {
            AgentEvent::ApiCallStarted => {
                *spinner.borrow_mut() = Some(ui.spinner("Thinking..."));
            }
            AgentEvent::ApiCallFinished { .. } => {
                clear_spinner();
                if *streaming.borrow() {
                    ui.stream_end();
                    *streaming.borrow_mut() = false;
                }
            }
            AgentEvent::AssistantTextDelta(text) => {
                clear_spinner();
                let first = !*streaming.borrow();
                ui.stream_delta(first, &text);
                *streaming.borrow_mut() = true;
            }
            AgentEvent::AssistantText(text) => ui.print_message("assistant", &text),
            AgentEvent::ToolCallStarted { name, input } => ui.print_tool_call(&name, &input),
            AgentEvent::ToolCallFinished {
                name,
                outcome,
                content,
            } => ui.print_tool_result(&name, outcome, &content),
            AgentEvent::Retrying {
                attempt,
                max_retries,
                delay_secs,
                error,
            } => {
                ui.print_info(&format!(
                    "Transient API error ({}); retry {}/{} in {}s",
                    error, attempt, max_retries, delay_secs
                ));
            }
            AgentEvent::Notice(message) => ui.print_info(&message),
        };

        match agent.run_turn(trimmed, &mut on_event).await {
            Ok(TurnOutcome::Completed) | Ok(TurnOutcome::Refused) => {}
            Ok(TurnOutcome::PausedOnDenial) => {
                ui.print_info("You can now give new instructions.");
            }
            Ok(TurnOutcome::MaxIterationsReached) => {
                ui.print_info("Ask to continue if you want the agent to keep going.");
            }
            Err(e) => {
                ui.print_error(&e.to_string());
                ui.print_info("The conversation so far is preserved; you can continue or /save.");
            }
        }

        // Best-effort autosave so a crash never loses a session.
        let _ = save_state(
            &make_saved_state(&agent, &permission_handler),
            AUTOSAVE_NAME,
        );
        println!();
    }
    Ok(())
}
