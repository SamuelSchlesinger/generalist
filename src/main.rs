use claude::{Claude, Message, ContentBlock, ToolRegistry, Result, Error, 
    ToolPermissionHandler, ToolExecutionRequest, PermissionDecision, tools::*, ChatbotState};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::env;
use std::collections::HashSet;
use dialoguer::{theme::ColorfulTheme, Input, Select};
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use console::Term;
use colored::*;
use tokio::time::Duration;
use chrono::Local;
use std::fs;
use std::path::PathBuf;


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
struct MemoryPermissionHandler {
    always_allow: Arc<Mutex<HashSet<String>>>,
    always_deny: Arc<Mutex<HashSet<String>>>,
}

impl MemoryPermissionHandler {
    fn new() -> Self {
        Self {
            always_allow: Arc::new(Mutex::new(HashSet::new())),
            always_deny: Arc::new(Mutex::new(HashSet::new())),
        }
    }
}

/// Wrapper to allow sharing permission handler between registry and state updates
struct MemoryPermissionHandlerWrapper {
    inner: Arc<Mutex<MemoryPermissionHandler>>,
}

#[async_trait]
impl ToolPermissionHandler for MemoryPermissionHandlerWrapper {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        // Clone the handler reference to avoid holding the lock across await
        let handler_clone = Arc::clone(&self.inner);
        let handler = handler_clone.lock().unwrap();
        
        // Check always allow/deny first
        {
            let always_allow = handler.always_allow.lock().unwrap();
            if always_allow.contains(&request.tool_name) {
                eprintln!("{} Automatically allowing '{}' (previously set to always allow)", 
                    "✓".green(), request.tool_name.cyan());
                return PermissionDecision::Allow;
            }
        }
        
        {
            let always_deny = handler.always_deny.lock().unwrap();
            if always_deny.contains(&request.tool_name) {
                eprintln!("{} Automatically denying '{}' (previously set to never allow)", 
                    "✗".red(), request.tool_name.cyan());
                return PermissionDecision::DenyWithReason(
                    "Tool was previously set to never allow".to_string()
                );
            }
        }
        
        // Drop the handler lock before the interactive prompt
        drop(handler);
        
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
                println!("Input: {}", 
                    serde_json::to_string_pretty(&request.input)
                        .unwrap_or_default()
                        .dimmed()
                );
            }
        } else {
            println!("Input: {}", 
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
        
        // Re-acquire the handler to update always_allow/always_deny
        let handler = handler_clone.lock().unwrap();
        
        match selection {
            0 => { // Yes (always)
                let mut always_allow = handler.always_allow.lock().unwrap();
                always_allow.insert(request.tool_name.clone());
                println!("{} Tool '{}' will be automatically allowed in the future", 
                    "✓".green(), request.tool_name.cyan());
                PermissionDecision::Allow
            }
            1 => { // Yes (once)
                PermissionDecision::Allow
            }
            2 => { // No (never)
                let mut always_deny = handler.always_deny.lock().unwrap();
                always_deny.insert(request.tool_name.clone());
                println!("{} Tool '{}' will be automatically denied in the future", 
                    "✗".red(), request.tool_name.cyan());
                PermissionDecision::DenyWithReason(
                    "User chose to never allow this tool".to_string()
                )
            }
            3 => { // No (once)
                PermissionDecision::DenyWithReason(
                    "User denied this tool execution".to_string()
                )
            }
            _ => unreachable!()
        }
    }
}

#[async_trait]
impl ToolPermissionHandler for MemoryPermissionHandler {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        // Check if we have a remembered decision
        {
            let always_allow = self.always_allow.lock().unwrap();
            if always_allow.contains(&request.tool_name) {
                eprintln!("{} Automatically allowing '{}' (previously set to always allow)", 
                    "✓".green(), request.tool_name.cyan());
                return PermissionDecision::Allow;
            }
        }
        
        {
            let always_deny = self.always_deny.lock().unwrap();
            if always_deny.contains(&request.tool_name) {
                eprintln!("{} Automatically denying '{}' (previously set to never allow)", 
                    "✗".red(), request.tool_name.cyan());
                return PermissionDecision::DenyWithReason(
                    "Tool was previously set to never allow".to_string()
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
                println!("Input: {}", 
                    serde_json::to_string_pretty(&request.input)
                        .unwrap_or_default()
                        .dimmed()
                );
            }
        } else {
            println!("Input: {}", 
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
            0 => { // Yes (always)
                let mut always_allow = self.always_allow.lock().unwrap();
                always_allow.insert(request.tool_name.clone());
                println!("{} Tool '{}' will be automatically allowed in the future", 
                    "✓".green(), request.tool_name.cyan());
                PermissionDecision::Allow
            }
            1 => { // Yes (once)
                PermissionDecision::Allow
            }
            2 => { // No (never)
                let mut always_deny = self.always_deny.lock().unwrap();
                always_deny.insert(request.tool_name.clone());
                println!("{} Tool '{}' will be automatically denied in the future", 
                    "✗".red(), request.tool_name.cyan());
                PermissionDecision::DenyWithReason(
                    "User chose to never allow this tool".to_string()
                )
            }
            3 => { // No (once)
                PermissionDecision::DenyWithReason(
                    "User denied permission for this execution".to_string()
                )
            }
            _ => unreachable!()
        }
    }
}

struct ChatUI {
    term: Term,
    multi_progress: MultiProgress,
    max_result_length: usize,
}

impl ChatUI {
    fn new() -> Self {
        Self {
            term: Term::stdout(),
            multi_progress: MultiProgress::new(),
            max_result_length: 200, // Default max length for tool results
        }
    }
    
    fn shorten_result(&self, result: &str) -> String {
        if result.len() <= self.max_result_length {
            result.to_string()
        } else {
            let half_len = (self.max_result_length - 20) / 2;
            format!(
                "{}... [truncated {} chars] ...{}", 
                &result[..half_len],
                result.len() - self.max_result_length,
                &result[result.len() - half_len..]
            )
        }
    }
    
    fn print_welcome(&self) {
        self.term.clear_screen().unwrap();
        println!("{}", "╔═══════════════════════════════════════════════════════════╗".bright_blue());
        println!("{}", "║            🤖 Claude CLI Chatbot with Tools 🛠️            ║".bright_blue());
        println!("{}", "╚═══════════════════════════════════════════════════════════╝".bright_blue());
        println!();
        println!("{}", "Available tools:".yellow());
        println!("  • {} - Apply patches/diffs to files", "patch_file".cyan());
        println!("  • {} - Read content from files", "read_file".cyan());
        println!("  • {} - List directory contents", "list_directory".cyan());
        println!("  • {} - Execute bash commands", "bash".cyan());
        println!("  • {} - Get system information", "system_info".cyan());
        println!("  • {} - Perform mathematical calculations", "calculator".cyan());
        println!("  • {} - Get current weather for any city", "weather".cyan());
        println!("  • {} - Make HTTP requests to fetch data", "http_fetch".cyan());
        println!("  • {} - Store and search persistent memories", "enhanced_memory".cyan());
        println!("  • {} - Generate deeper thinking prompts", "still_thinking".cyan());
        println!("  • {} - Search Wikipedia articles and get summaries", "wikipedia".cyan());
        println!("  • {} - Z3 SMT/SAT constraint solver for logic and optimization", "z3_solver".cyan());
        println!("  • {} - Search recent news articles from RSS feeds", "news_search".cyan());
        println!("  • {} - Search the web using DuckDuckGo API and scraping", "web_search".cyan());
        println!("  • {} - Search academic papers from arXiv and other sources", "academic_search".cyan());
        println!();
        println!("{} {}", "🔐".cyan(), "Tool Permission System Active".yellow().bold());
        println!("{}", "You'll be asked to approve each tool use with these options:".dimmed());
        println!("  • {} - Tool will always be allowed automatically", "Yes (always allow)".green());
        println!("  • {} - Allow just this execution", "Yes (just this once)".green());
        println!("  • {} - Tool will always be denied automatically", "No (never allow)".red());
        println!("  • {} - Deny just this execution", "No (just this once)".red());
        println!();
        println!("{}", "Commands:".yellow());
        println!("  • {} - Save current conversation", "/save".cyan());
        println!("  • {} - Load a saved conversation", "/load".cyan());
        println!("  • {} - Show help message", "/help".cyan());
        println!("  • {} or {} - Exit the chatbot", "exit".cyan(), "quit".cyan());
        println!("{}", "─".repeat(60).dimmed());
        println!();
    }
    
    fn print_message(&self, role: &str, content: &str) {
        let timestamp = Local::now().format("%H:%M:%S");
        match role {
            "user" => {
                println!("{} {} {}", 
                    format!("[{}]", timestamp).dimmed(),
                    "You:".green().bold(),
                    content
                );
            }
            "assistant" => {
                println!("{} {} {}", 
                    format!("[{}]", timestamp).dimmed(),
                    "Claude:".blue().bold(),
                    content
                );
            }
            _ => {}
        }
    }
    
    fn print_tool_use(&self, tool_name: &str, input: &Value) -> ProgressBar {
        let pb = self.multi_progress.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        );
        pb.set_message(format!("🔧 Using tool: {} with input: {}", 
            tool_name.yellow(), 
            serde_json::to_string(input).unwrap_or_default().dimmed()
        ));
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    }
    
    #[allow(dead_code)]
    fn print_tool_result(&self, tool_name: &str, result: &str, pb: ProgressBar) {
        pb.finish_and_clear();
        println!("   {} {} result: {}", 
            "✓".green(),
            tool_name.yellow(),
            result.italic()
        );
    }
    
    fn print_error(&self, error: &str) {
        println!("{} {}", "Error:".red().bold(), error);
    }
}

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
            
            println!("{} Loaded legacy conversation format from: {}", "✓".green(), filepath.display());
            println!("{} Converting to new state format...", "ℹ".blue());
            
            // Use default model for legacy files
            Ok(ChatbotState::from_conversation(messages, "claude-3-7-sonnet-latest".to_string()))
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
    // Load environment from ~/.chatbot.env
    let home_dir = env::home_dir().expect("Unable to determine home directory");
    let env_path = home_dir.join(".chatbot.env");
    
    // Check if the file exists
    if !env_path.exists() {
        eprintln!("{}", "Error: ~/.chatbot.env file not found".red());
        eprintln!("Please create the file with your API key:");
        eprintln!("  echo 'CLAUDE_API_KEY=your-api-key-here' > ~/.chatbot.env");
        std::process::exit(1);
    }
    
    // Load the env file
    dotenv::from_path(&env_path).expect("Failed to load ~/.chatbot.env");

    // Get API key
    let api_key = env::var("CLAUDE_API_KEY")
        .unwrap_or_else(|_| {
            eprintln!("{}", "Error: CLAUDE_API_KEY not found in ~/.chatbot.env".red());
            eprintln!("Please add your API key to ~/.chatbot.env:");
            eprintln!("  echo 'CLAUDE_API_KEY=your-api-key-here' >> ~/.chatbot.env");
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
    let permission_handler = Arc::new(Mutex::new(MemoryPermissionHandler::new()));
    
    // Initialize Claude client
    let mut client = Claude::new(api_key.clone(), model.clone());
    
    // Initialize tool registry with memory permission handler
    println!("{} Using interactive permissions with memory", "🔐".cyan());
    println!("{}", "You'll be prompted for each tool execution with options to remember your choice.\n".dimmed());
    
    let mut registry = ToolRegistry::with_permission_handler(
        Box::new(MemoryPermissionHandlerWrapper {
            inner: Arc::clone(&permission_handler)
        })
    );
    
    registry.register(Arc::new(PatchFileTool))?;
    registry.register(Arc::new(ReadFileTool))?;
    registry.register(Arc::new(ListDirectoryTool))?;
    registry.register(Arc::new(BashTool))?;
    registry.register(Arc::new(SystemInfoTool))?;
    registry.register(Arc::new(CalculatorTool))?;
    registry.register(Arc::new(WeatherTool))?;
    registry.register(Arc::new(HttpFetchTool))?;
    registry.register(Arc::new(EnhancedMemoryTool::new()?))?;
    registry.register(Arc::new(StillThinkingTool))?;
    registry.register(Arc::new(WikipediaTool))?;
    registry.register(Arc::new(Z3SolverTool))?;
    registry.register(Arc::new(NewsSearchTool))?;
    registry.register(Arc::new(WebSearchTool))?;
    registry.register(Arc::new(AcademicSearchTool))?;
    
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
        if input_trimmed.eq_ignore_ascii_case("exit") || input_trimmed.eq_ignore_ascii_case("quit") {
            println!("\n{}", "👋 Goodbye! Thanks for chatting!".yellow());
            break;
        } else if input_trimmed.eq_ignore_ascii_case("/save") {
            let name: String = Input::with_theme(&ColorfulTheme::default())
                .with_prompt("Save conversation as")
                .default(format!("chat_{}", Local::now().format("%Y%m%d_%H%M%S")))
                .interact_text()
                .unwrap();
            
            // Update state with current permissions
            {
                let handler = permission_handler.lock().unwrap();
                state.always_allow_tools = handler.always_allow.lock().unwrap().clone();
                state.always_deny_tools = handler.always_deny.lock().unwrap().clone();
            }
            
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
                        println!("{} Loaded {} messages", "✓".green(), state.conversation_history.len());
                        
                        // Update model if different
                        if state.model != model {
                            model = state.model.clone();
                            client = Claude::new(api_key.clone(), model.clone());
                            println!("{} Switched to model: {}", "✓".green(), model.cyan());
                        }
                        
                        // Update permissions
                        {
                            let handler = permission_handler.lock().unwrap();
                            *handler.always_allow.lock().unwrap() = state.always_allow_tools.clone();
                            *handler.always_deny.lock().unwrap() = state.always_deny_tools.clone();
                            println!("{} Restored {} allowed and {} denied tools", 
                                "✓".green(), 
                                state.always_allow_tools.len(), 
                                state.always_deny_tools.len());
                        }
                        
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
                    Err(e) => ui.print_error(&format!("Failed to load state: {}", e))
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
            println!("  {} or {} - Exit the chatbot", "exit".cyan(), "quit".cyan());
            println!();
            continue;
        }
        
        ui.print_message("user", &input);
        
        // Add user message to history
        state.conversation_history.push(Message::user(vec![
            ContentBlock::Text { text: input.clone() }
        ]));
        
        // Show thinking indicator
        let mut thinking_pb = ui.multi_progress.add(ProgressBar::new_spinner());
        thinking_pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} {msg}")
                .unwrap()
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
                                        if let ContentBlock::ToolResult { content, is_error: Some(true), .. } = &result {
                                            if content.contains("denied") {
                                                // Permission was denied - don't show progress bar
                                                println!("   {} Tool {} was not executed: {}", "✗".red(), name.cyan(), content.dimmed());
                                                tool_was_denied = true;
                                            } else {
                                                // Other error during execution - show progress bar
                                                let pb = ui.print_tool_use(name, input);
                                                pb.finish_with_message(format!("✗ {} failed", name.red()));
                                                println!("   {} Error: {}", "→".red(), ui.shorten_result(content).dimmed());
                                            }
                                        } else {
                                            // Success - show progress bar
                                            let pb = ui.print_tool_use(name, input);
                                            pb.finish_with_message(format!("✓ {} completed", name.green()));
                                            if let ContentBlock::ToolResult { content, .. } = &result {
                                                println!("   {} Result: {}", "→".cyan(), ui.shorten_result(content).dimmed());
                                            }
                                        }
                                        tool_results.push(result);
                                    }
                                    Err(e) => {
                                        // Unexpected error (tool not found, etc)
                                        println!("   {} Tool {} error: {}", "✗".red(), name.cyan(), e.to_string().dimmed());
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
                            println!("\n{} {}", 
                                "⚠️".yellow(), 
                                "Tool execution was denied. The conversation has been paused.".yellow()
                            );
                            println!("{}", "You can now provide new instructions or continue the conversation.".dimmed());
                            println!();
                            
                            // Save the response with proper conversation history
                            final_response = Some(response);
                            break;
                        }
                        
                        // Show we're waiting for Claude's next response
                        thinking_pb = ui.multi_progress.add(ProgressBar::new_spinner());
                        thinking_pb.set_style(
                            ProgressStyle::default_spinner()
                                .template("{spinner:.blue} {msg}")
                                .unwrap()
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
