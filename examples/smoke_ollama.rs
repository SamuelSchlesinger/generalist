//! Live smoke test against a local Ollama server via the OpenAI-compatible
//! provider.
//!
//! Run with: `cargo run --example smoke_ollama [model]`
//! Requires Ollama running on localhost:11434 with a tool-capable model
//! (default: qwen3:30b).

use generalist::provider::OpenAiProvider;
use generalist::tools::CalculatorTool;
use generalist::{Agent, AgentEvent, ToolRegistry, TurnOutcome};
use std::sync::Arc;

#[tokio::main]
async fn main() -> generalist::Result<()> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "qwen3:30b".to_string());

    // Ollama ignores the API key but the header must be present.
    let provider = OpenAiProvider::new(
        "ollama".into(),
        "http://localhost:11434/v1".into(),
        model.clone(),
    )?;

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool))?;

    let mut agent = Agent::new(
        Box::new(provider),
        registry,
        "You are a terse assistant. Use the calculator tool for any arithmetic.",
    );

    let said = std::cell::RefCell::new(String::new());
    let mut on_event = |event: AgentEvent| match event {
        AgentEvent::AssistantText(text) => {
            said.borrow_mut().push_str(&text);
            println!("ASSISTANT: {}", text);
        }
        AgentEvent::AssistantTextDelta(text) => {
            use std::io::Write;
            said.borrow_mut().push_str(&text);
            print!("{}", text);
            std::io::stdout().flush().ok();
        }
        AgentEvent::ApiCallFinished { .. } => println!(),
        AgentEvent::ToolCallStarted { name, input } => println!("TOOL CALL: {} {}", name, input),
        AgentEvent::ToolCallFinished { name, content, .. } => {
            println!("TOOL RESULT: {} -> {}", name, content)
        }
        AgentEvent::Notice(n) => println!("NOTICE: {}", n),
        AgentEvent::Retrying { error, .. } => println!("RETRY: {}", error),
        _ => {}
    };

    println!(
        "Model: {} (first call may be slow while the model loads)\n",
        model
    );

    let outcome = agent
        .run_turn("Use the calculator to compute 111 * 111.", &mut on_event)
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(
        said.borrow().replace([',', ' '], "").contains("12321"),
        "expected 12321 in: {}",
        said.borrow()
    );

    let outcome = agent
        .run_turn(
            "Now add 58 to that result with the calculator.",
            &mut on_event,
        )
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(
        said.borrow().replace([',', ' '], "").contains("12379"),
        "expected 12379 in: {}",
        said.borrow()
    );

    println!(
        "\nSMOKE OK (ollama/{}) — {} messages in history",
        model,
        agent.history.len()
    );
    Ok(())
}
