use chrono::Local;
use colored::*;
use console::Term;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde_json::Value;
use tokio::time::Duration;

pub struct ChatUI {
    term: Term,
    multi_progress: MultiProgress,
    max_result_length: usize,
}

impl ChatUI {
    pub fn new() -> Self {
        Self {
            term: Term::stdout(),
            multi_progress: MultiProgress::new(),
            max_result_length: 200,
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

    pub fn print_welcome(&self) {
        self.term.clear_screen().unwrap();
        println!(
            "{}",
            "╔═══════════════════════════════════════════════════════════╗".bright_blue()
        );
        println!(
            "{}",
            "║            🤖 Claude CLI Chatbot with Tools 🛠️            ║".bright_blue()
        );
        println!(
            "{}",
            "╚═══════════════════════════════════════════════════════════╝".bright_blue()
        );
        println!();
        println!("{}", "Available tools:".yellow());
        println!("  • {} - Apply patches/diffs to files", "patch_file".cyan());
        println!("  • {} - Read content from files", "read_file".cyan());
        println!("  • {} - List directory contents", "list_directory".cyan());
        println!("  • {} - Execute bash commands", "bash".cyan());
        println!("  • {} - Get system information", "system_info".cyan());
        println!(
            "  • {} - Perform mathematical calculations",
            "calculator".cyan()
        );
        println!(
            "  • {} - Get current weather for any city",
            "weather".cyan()
        );
        println!(
            "  • {} - Make HTTP requests to fetch data",
            "http_fetch".cyan()
        );
        println!(
            "  • {} - Store and search persistent memories",
            "enhanced_memory".cyan()
        );
        println!("  • {} - Think more deeply about topics", "think".cyan());
        println!(
            "  • {} - Search Wikipedia articles and get summaries",
            "wikipedia".cyan()
        );
        println!(
            "  • {} - Z3 SMT/SAT constraint solver for logic and optimization",
            "z3_solver".cyan()
        );
        println!(
            "  • {} - Crawl websites and extract content using Firecrawl",
            "firecrawl_crawl".cyan()
        );
        println!(
            "  • {} - Search the web using Firecrawl's search API",
            "firecrawl_search".cyan()
        );
        println!(
            "  • {} - Map website structure using Firecrawl",
            "firecrawl_map".cyan()
        );
        println!(
            "  • {} - Extract structured data from web pages using Firecrawl",
            "firecrawl_extract".cyan()
        );
        println!();
        println!(
            "{} {}",
            "🔐".cyan(),
            "Tool Permission System Active".yellow().bold()
        );
        println!(
            "{}",
            "You'll be asked to approve each tool use with these options:".dimmed()
        );
        println!(
            "  • {} - Tool will always be allowed automatically",
            "Yes (always allow)".green()
        );
        println!(
            "  • {} - Allow just this execution",
            "Yes (just this once)".green()
        );
        println!(
            "  • {} - Tool will always be denied automatically",
            "No (never allow)".red()
        );
        println!(
            "  • {} - Deny just this execution",
            "No (just this once)".red()
        );
        println!();
        println!("{}", "Commands:".yellow());
        println!("  • {} - Save current conversation", "/save".cyan());
        println!("  • {} - Load a saved conversation", "/load".cyan());
        println!("  • {} - Show help message", "/help".cyan());
        println!(
            "  • {} or {} - Exit the chatbot",
            "exit".cyan(),
            "quit".cyan()
        );
        println!("{}", "─".repeat(60).dimmed());
        println!();
    }

    pub fn print_message(&self, role: &str, content: &str) {
        let timestamp = Local::now().format("%H:%M:%S");
        match role {
            "user" => {
                println!(
                    "{} {} {}",
                    format!("[{}]", timestamp).dimmed(),
                    "You:".green().bold(),
                    content
                );
            }
            "assistant" => {
                println!(
                    "{} {} {}",
                    format!("[{}]", timestamp).dimmed(),
                    "Claude:".blue().bold(),
                    content
                );
            }
            _ => {}
        }
    }

    pub fn print_tool_use(&self, tool_name: &str, input: &Value) -> ProgressBar {
        let pb = self.multi_progress.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        pb.set_message(format!(
            "🔧 Using tool: {} with input: {}",
            tool_name.yellow(),
            serde_json::to_string(input).unwrap_or_default().dimmed()
        ));
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    }

    #[allow(dead_code)]
    pub fn print_tool_result(&self, tool_name: &str, result: &str, pb: ProgressBar) {
        pb.finish_and_clear();
        println!(
            "   {} {} result: {}",
            "✓".green(),
            tool_name.yellow(),
            result.italic()
        );
    }

    pub fn print_error(&self, error: &str) {
        println!("{} {}", "Error:".red().bold(), error);
    }

    pub fn multi_progress(&self) -> &MultiProgress {
        &self.multi_progress
    }

    pub fn shorten_result_public(&self, result: &str) -> String {
        self.shorten_result(result)
    }
}
