//! Terminal rendering for the CLI.

use crate::tool::ToolCallOutcome;
use crate::types::truncate_middle;
use chrono::Local;
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde_json::Value;
use std::time::Duration;

pub struct ChatUI {
    multi_progress: MultiProgress,
    /// Display cap for tool inputs/results. Only affects what is printed —
    /// the model always receives the full (history-truncated) content.
    pub max_display_chars: usize,
}

impl Default for ChatUI {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatUI {
    pub fn new() -> Self {
        Self {
            multi_progress: MultiProgress::new(),
            max_display_chars: 300,
        }
    }

    pub fn print_welcome(&self, provider: &str, model: &str, tools: &[String]) {
        println!();
        println!("{}", "generalist — a provider-agnostic CLI agent".bold());
        println!(
            "{} {} {} {}",
            "Provider:".dimmed(),
            provider.cyan(),
            "Model:".dimmed(),
            model.cyan()
        );
        println!("{} {}", "Tools:".dimmed(), tools.join(", ").cyan());
        println!();
        println!(
            "{}",
            "Each tool call asks for permission; 'always allow' is remembered per tool.".dimmed()
        );
        println!(
            "{} {}",
            "Commands:".dimmed(),
            "/save /load /model /compact /clear /help, exit".cyan()
        );
        println!("{}", "─".repeat(60).dimmed());
        println!();
    }

    pub fn print_help(&self) {
        println!("\n{}", "Commands:".yellow().bold());
        println!("  {} - Save the conversation", "/save".cyan());
        println!("  {} - Load a saved conversation", "/load".cyan());
        println!("  {} - Switch model", "/model".cyan());
        println!(
            "  {} - Summarize older history to free context",
            "/compact".cyan()
        );
        println!("  {} - Clear the conversation history", "/clear".cyan());
        println!("  {} - Show this help", "/help".cyan());
        println!("  {} or {} - Exit", "exit".cyan(), "quit".cyan());
        println!();
    }

    pub fn print_message(&self, role: &str, content: &str) {
        let timestamp = Local::now().format("%H:%M:%S");
        let label = match role {
            "user" => "You:".green().bold(),
            _ => "Assistant:".blue().bold(),
        };
        println!(
            "{} {} {}",
            format!("[{}]", timestamp).dimmed(),
            label,
            content
        );
    }

    /// Announce a tool call *before* it is permission-checked or executed.
    pub fn print_tool_call(&self, name: &str, input: &Value) {
        let compact = serde_json::to_string(input).unwrap_or_default();
        println!(
            "{} {} {}",
            "→".cyan(),
            name.yellow(),
            truncate_middle(&compact, self.max_display_chars).dimmed()
        );
    }

    pub fn print_tool_result(&self, name: &str, outcome: ToolCallOutcome, content: &str) {
        let shown = truncate_middle(content, self.max_display_chars);
        match outcome {
            ToolCallOutcome::Success => {
                println!("  {} {} {}", "✓".green(), name.yellow(), shown.dimmed());
            }
            ToolCallOutcome::Failed => {
                println!("  {} {} {}", "✗".red(), name.yellow(), shown.dimmed());
            }
            ToolCallOutcome::Denied => {
                println!("  {} {} {}", "⊘".red(), name.yellow(), "denied".dimmed());
            }
        }
    }

    pub fn spinner(&self, message: &str) -> ProgressBar {
        let pb = self.multi_progress.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} {msg}")
                .expect("static template"),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(Duration::from_millis(100));
        pb
    }

    /// Render one streamed fragment; `first` prints the message prefix.
    pub fn stream_delta(&self, first: bool, text: &str) {
        use std::io::Write;
        if first {
            let timestamp = Local::now().format("%H:%M:%S");
            print!(
                "{} {} ",
                format!("[{}]", timestamp).dimmed(),
                "Assistant:".blue().bold()
            );
        }
        print!("{}", text);
        std::io::stdout().flush().ok();
    }

    /// Close a streamed message.
    pub fn stream_end(&self) {
        println!();
    }

    pub fn print_error(&self, error: &str) {
        println!("{} {}", "Error:".red().bold(), error);
    }

    pub fn print_info(&self, message: &str) {
        println!("{} {}", "ℹ".blue(), message.dimmed());
    }
}
