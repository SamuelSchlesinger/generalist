use chrono::Local;
use claude::{
    tools::*, ChatbotState, Claude, ContentBlock, Error, MemoryPermissionHandler, Message, Result,
    ToolRegistry,
};
use colored::*;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::time::Duration;

mod chat_ui;
use chat_ui::ChatUI;

// Conversation history management
fn get_history_dir() -> PathBuf {
    let home_dir = env::home_dir().expect("Unable to determine home directory");
    let history_dir = home_dir.join(".chatbot_history");
    fs::create_dir_all(&history_dir).ok();
    history_dir
}

fn save_state(state: &ChatbotState, filename: &str) -> Result<()> {
    let history_dir = get_history_dir();
    let filepath = history_dir.join(format!("{}.json", filename));

    let json_data = serde_json::to_string_pretty(state)
        .map_err(|e| Error::Other(format!("Failed to serialize state: {}", e)))?;

    fs::write(&filepath, json_data)
        .map_err(|e| Error::Other(format!("Failed to write state file: {}", e)))?;

    println!("{} State saved to: {}", "✓".green(), filepath.display());
    Ok(())
}

fn load_state(filename: &str) -> Result<ChatbotState> {
    let history_dir = get_history_dir();
    let filepath = history_dir.join(format!("{}.json", filename));

    let json_data = fs::read_to_string(&filepath)
        .map_err(|e| Error::Other(format!("Failed to read state file: {}", e)))?;

    // First try to parse as ChatbotState
    match serde_json::from_str::<ChatbotState>(&json_data) {
        Ok(state) => {
            println!("{} State loaded from: {}", "✓".green(), filepath.display());
            Ok(state)
        }
        Err(_) => {
            // Fall back to old format (just conversation history)
            let messages: Vec<Message> = serde_json::from_str(&json_data)
                .map_err(|e| Error::Other(format!("Failed to parse state: {}", e)))?;

            println!(
                "{} Loaded legacy conversation format from: {}",
                "✓".green(),
                filepath.display()
            );
            println!("{} Converting to new state format...", "ℹ".blue());

            // Use default model for legacy files
            Ok(ChatbotState::from_conversation(
                messages,
                "claude-3-7-sonnet-latest".to_string(),
            ))
        }
    }
}

fn list_saved_conversations() -> Vec<String> {
    let history_dir = get_history_dir();
    let mut conversations = Vec::new();

    if let Ok(entries) = fs::read_dir(history_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json") {
                    conversations.push(name.trim_end_matches(".json").to_string());
                }
            }
        }
    }

    conversations.sort();
    conversations
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment from ~/.generalist.env
    let home_dir = env::home_dir().expect("Unable to determine home directory");
    let env_path = home_dir.join(".generalist.env");

    // Check if the file exists
    if !env_path.exists() {
        eprintln!("{}", "Error: ~/.generalist.env file not found".red());
        eprintln!("Please create the file with your API key:");
        eprintln!("  echo 'CLAUDE_API_KEY=your-api-key-here' > ~/.generalist.env");
        std::process::exit(1);
    }

    // Load the env file
    dotenv::from_path(&env_path).expect("Failed to load ~/.generalist.env");

    // Get API key
    let api_key = env::var("CLAUDE_API_KEY").unwrap_or_else(|_| {
        eprintln!(
            "{}",
            "Error: CLAUDE_API_KEY not found in ~/.generalist.env".red()
        );
        eprintln!("Please add your API key to ~/.generalist.env:");
        eprintln!("  echo 'CLAUDE_API_KEY=your-api-key-here' >> ~/.generalist.env");
        std::process::exit(1);
    });

    // Initialize UI
    let ui = ChatUI::new();
    ui.print_welcome();

    // Select model
    let models = vec![
        "claude-3-7-sonnet-latest",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
    ];

    let model_selection = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select Claude model")
        .items(&models)
        .default(0)
        .interact()
        .unwrap();

    let mut model = models[model_selection].to_string();
    println!("{} Using model: {}\n", "✓".green(), model.cyan());

    // Initialize state
    let mut state = ChatbotState::new(model.clone());

    // Initialize permission handler
    let permission_handler = Arc::new(MemoryPermissionHandler::new());

    // Initialize Claude client
    let mut client = Claude::new(api_key.clone(), model.clone());

    // Initialize tool registry with memory permission handler
    println!("{} Using interactive permissions with memory", "🔐".cyan());
    println!(
        "{}",
        "You'll be prompted for each tool execution with options to remember your choice.\n"
            .dimmed()
    );

    // Create a shared handler for the registry
    let shared_handler = MemoryPermissionHandler::with_shared_state(
        permission_handler.always_allow(),
        permission_handler.always_deny(),
    );
    let mut registry = ToolRegistry::with_permission_handler(Box::new(shared_handler));

    registry.register(Arc::new(PatchFileTool))?;
    registry.register(Arc::new(ReadFileTool))?;
    registry.register(Arc::new(ListDirectoryTool))?;
    registry.register(Arc::new(BashTool))?;
    registry.register(Arc::new(SystemInfoTool))?;
    registry.register(Arc::new(CalculatorTool))?;
    registry.register(Arc::new(WeatherTool))?;
    registry.register(Arc::new(HttpFetchTool))?;
    registry.register(Arc::new(EnhancedMemoryTool::new()?))?;
    registry.register(Arc::new(ThinkTool))?;
    registry.register(Arc::new(WikipediaTool))?;
    registry.register(Arc::new(Z3SolverTool))?;
    registry.register(Arc::new(TodoTool))?;
    registry.register(Arc::new(FirecrawlCrawlTool))?;
    registry.register(Arc::new(FirecrawlSearchTool))?;
    registry.register(Arc::new(FirecrawlMapTool))?;
    registry.register(Arc::new(FirecrawlExtractTool))?;

    // Load system prompt
    let system_prompt = include_str!("../SYSTEM_PROMPT.md");
    state.system_prompt = Some(system_prompt.to_string());

    // Main conversation loop
    loop {
        // Get user input
        let input: String = Input::with_theme(&ColorfulTheme::default())
            .with_prompt("You")
            .interact_text()
            .unwrap();

        // Check for special commands
        let input_trimmed = input.trim();
        if input_trimmed.eq_ignore_ascii_case("exit") || input_trimmed.eq_ignore_ascii_case("quit")
        {
            println!("\n{}", "👋 Goodbye! Thanks for chatting!".yellow());
            break;
        } else if input_trimmed.eq_ignore_ascii_case("/save") {
            let name: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Save conversation as")
                .default(format!("chat_{}", Local::now().format("%Y%m%d_%H%M%S")))
                .interact_text()
                .unwrap();

            // Update state with current permissions
            state.always_allow_tools = permission_handler.always_allow().lock().unwrap().clone();
            state.always_deny_tools = permission_handler.always_deny().lock().unwrap().clone();

            if let Err(e) = save_state(&state, &name) {
                ui.print_error(&format!("Failed to save state: {}", e));
            }
            continue;
        } else if input_trimmed.eq_ignore_ascii_case("/load") {
            let saved = list_saved_conversations();
            if saved.is_empty() {
                println!("{}", "No saved conversations found.".yellow());
                continue;
            }

            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select conversation to load")
                .items(&saved)
                .interact_opt()
                .unwrap();

            if let Some(idx) = selection {
                match load_state(&saved[idx]) {
                    Ok(loaded_state) => {
                        // Update state
                        state = loaded_state;
                        println!(
                            "{} Loaded {} messages",
                            "✓".green(),
                            state.conversation_history.len()
                        );

                        // Update model if different
                        if state.model != model {
                            model = state.model.clone();
                            client = Claude::new(api_key.clone(), model.clone());
                            println!("{} Switched to model: {}", "✓".green(), model.cyan());
                        }

                        // Update permissions
                        permission_handler.set_always_allow(state.always_allow_tools.clone());
                        permission_handler.set_always_deny(state.always_deny_tools.clone());
                        println!(
                            "{} Restored {} allowed and {} denied tools",
                            "✓".green(),
                            state.always_allow_tools.len(),
                            state.always_deny_tools.len()
                        );

                        // Display loaded conversation
                        for msg in &state.conversation_history {
                            match msg.role.as_str() {
                                "user" => {
                                    if let Some(ContentBlock::Text { text }) = msg.content.first() {
                                        ui.print_message("user", text);
                                    }
                                }
                                "assistant" => {
                                    for block in &msg.content {
                                        if let ContentBlock::Text { text } = block {
                                            ui.print_message("assistant", text);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        println!();
                    }
                    Err(e) => ui.print_error(&format!("Failed to load state: {}", e)),
                }
            }
            continue;
        } else if input_trimmed.eq_ignore_ascii_case("/model") {
            let models = vec![
                "claude-3-7-sonnet-latest",
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514",
            ];

            // Find current model index
            let current_idx = models.iter().position(|&m| m == model).unwrap_or(0);

            let model_selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select new Claude model")
                .items(&models)
                .default(current_idx)
                .interact()
                .unwrap();

            let new_model = models[model_selection].to_string();
            if new_model != model {
                model = new_model;
                state.model = model.clone();
                client = Claude::new(api_key.clone(), model.clone());
                println!("{} Switched to model: {}", "✓".green(), model.cyan());
            } else {
                println!("{} Already using model: {}", "ℹ".blue(), model.cyan());
            }
            continue;
        } else if input_trimmed.eq_ignore_ascii_case("/help") {
            println!("\n{}", "Available commands:".yellow().bold());
            println!("  {} - Save current conversation", "/save".cyan());
            println!("  {} - Load a saved conversation", "/load".cyan());
            println!("  {} - Switch Claude model", "/model".cyan());
            println!("  {} - Show this help message", "/help".cyan());
            println!(
                "  {} or {} - Exit the chatbot",
                "exit".cyan(),
                "quit".cyan()
            );
            println!();
            continue;
        }

        ui.print_message("user", &input);

        // Add user message to history
        state
            .conversation_history
            .push(Message::user(vec![ContentBlock::Text {
                text: input.clone(),
            }]));

        // Show thinking indicator
        let mut thinking_pb = ui.multi_progress().add(ProgressBar::new_spinner());
        thinking_pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} {msg}")
                .unwrap(),
        );
        thinking_pb.set_message("Claude is thinking...");
        thinking_pb.enable_steady_tick(Duration::from_millis(100));

        // Manual conversation handling for real-time display
        let mut current_messages = state.conversation_history.clone();
        let max_iterations = 100;
        let mut iterations = 0;
        let mut final_response = None;

        loop {
            if iterations >= max_iterations {
                thinking_pb.finish_and_clear();
                ui.print_error("Maximum tool execution iterations reached");
                break;
            }

            // Create request
            let request = claude::MessageRequest {
                model: client.model().to_string(),
                messages: current_messages.clone(),
                tools: registry.get_tool_defs(),
                max_tokens: 1024,
                system: Some(system_prompt.to_string()),
                temperature: None,
            };

            // Send message
            match client.next_message(request).await {
                Ok(response) => {
                    thinking_pb.finish_and_clear();

                    // Process response content in real-time
                    let mut has_tool_uses = false;
                    let mut tool_results = Vec::new();
                    let mut tool_was_denied = false;

                    for block in &response.content {
                        match block {
                            ContentBlock::Text { text } => {
                                // Show text immediately
                                ui.print_message("assistant", text);
                            }
                            ContentBlock::ToolUse { name, input, id } => {
                                has_tool_uses = true;
                                // Don't show tool use until after permission check

                                // Execute tool (permission check happens inside)
                                match registry.execute_tool(name, input.clone(), id.clone()).await {
                                    Ok(result) => {
                                        // Check if this is a permission denial (is_error = true and content contains "denied")
                                        if let ContentBlock::ToolResult {
                                            content,
                                            is_error: Some(true),
                                            ..
                                        } = &result
                                        {
                                            if content.contains("denied") {
                                                // Permission was denied - don't show progress bar
                                                println!(
                                                    "   {} Tool {} was not executed: {}",
                                                    "✗".red(),
                                                    name.cyan(),
                                                    content.dimmed()
                                                );
                                                tool_was_denied = true;
                                            } else {
                                                // Other error during execution - show progress bar
                                                let pb = ui.print_tool_use(name, input);
                                                pb.finish_with_message(format!(
                                                    "✗ {} failed",
                                                    name.red()
                                                ));
                                                println!(
                                                    "   {} Error: {}",
                                                    "→".red(),
                                                    ui.shorten_result_public(content).dimmed()
                                                );
                                            }
                                        } else {
                                            // Success - show progress bar
                                            let pb = ui.print_tool_use(name, input);
                                            pb.finish_with_message(format!(
                                                "✓ {} completed",
                                                name.green()
                                            ));
                                            if let ContentBlock::ToolResult { content, .. } =
                                                &result
                                            {
                                                println!(
                                                    "   {} Result: {}",
                                                    "→".cyan(),
                                                    ui.shorten_result_public(content).dimmed()
                                                );
                                            }
                                        }
                                        tool_results.push(result);
                                    }
                                    Err(e) => {
                                        // Unexpected error (tool not found, etc)
                                        println!(
                                            "   {} Tool {} error: {}",
                                            "✗".red(),
                                            name.cyan(),
                                            e.to_string().dimmed()
                                        );
                                        tool_results.push(ContentBlock::ToolResult {
                                            tool_use_id: id.clone(),
                                            content: format!("Error: {}", e),
                                            is_error: Some(true),
                                        });
                                    }
                                }
                            }
                            ContentBlock::ToolResult { .. } => {
                                // Should not appear in assistant responses
                            }
                        }
                    }

                    // Add assistant response to history
                    current_messages.push((&response).into());

                    if !has_tool_uses {
                        // No more tools, we're done
                        final_response = Some(response);
                        break;
                    }

                    // Add tool results to messages
                    if !tool_results.is_empty() {
                        current_messages.push(Message::user(tool_results));

                        if tool_was_denied {
                            // Tool was denied - stop processing and wait for user input
                            println!(
                                "\n{} {}",
                                "⚠️".yellow(),
                                "Tool execution was denied. The conversation has been paused."
                                    .yellow()
                            );
                            println!("{}", "You can now provide new instructions or continue the conversation.".dimmed());
                            println!();

                            // Save the response with proper conversation history
                            final_response = Some(response);
                            break;
                        }

                        // Show we're waiting for Claude's next response
                        thinking_pb = ui.multi_progress().add(ProgressBar::new_spinner());
                        thinking_pb.set_style(
                            ProgressStyle::default_spinner()
                                .template("{spinner:.blue} {msg}")
                                .unwrap(),
                        );
                        thinking_pb.set_message("Processing tool results...");
                        thinking_pb.enable_steady_tick(Duration::from_millis(100));
                    }

                    iterations += 1;
                }
                Err(e) => {
                    thinking_pb.finish_and_clear();
                    ui.print_error(&format!("{}", e));
                    break;
                }
            }
        }

        // Update conversation history with the full exchange
        if let Some(_final_resp) = final_response {
            state.conversation_history = current_messages;
        }

        println!();
    }
    Ok(())
}
