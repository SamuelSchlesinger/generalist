//! Live smoke test: two turns with a real provider, exercising the tool loop.
//!
//! Run with: `cargo run --example smoke`
//! Requires ANTHROPIC_API_KEY (or CLAUDE_API_KEY) in the environment or
//! ~/.generalist.env. Makes a few small real API calls.

use generalist::provider::AnthropicProvider;
use generalist::tools::CalculatorTool;
use generalist::{Agent, AgentEvent, ToolRegistry, TurnOutcome};
use std::sync::Arc;

#[tokio::main]
async fn main() -> generalist::Result<()> {
    #[allow(deprecated)]
    let env_path = std::env::home_dir().unwrap().join(".generalist.env");
    if env_path.exists() {
        dotenv::from_path(&env_path).ok();
    }
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .or_else(|_| std::env::var("CLAUDE_API_KEY"))
        .expect("set ANTHROPIC_API_KEY");

    let provider = AnthropicProvider::new(api_key, "claude-opus-4-8".into())?;
    let mut registry = ToolRegistry::new(); // AlwaysAllow — fine for a calculator
    registry.register(Arc::new(CalculatorTool))?;

    let mut agent = Agent::new(
        Box::new(provider),
        registry,
        "You are a terse assistant. Use the calculator tool for any arithmetic.",
    );

    let mut on_event = |event: AgentEvent| match event {
        AgentEvent::AssistantText(text) => println!("ASSISTANT: {}", text),
        AgentEvent::AssistantTextDelta(text) => {
            use std::io::Write;
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

    // Turn 1: forces a tool round-trip.
    let outcome = agent
        .run_turn("Use the calculator to compute 111 * 111.", &mut on_event)
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    let answer = agent.history.last().unwrap().text();
    assert!(answer.contains("12321"), "expected 12321 in: {}", answer);

    // Turn 2: replays turn-1 history (including any thinking blocks) on the
    // same model — the part most likely to 400 if serialization is wrong.
    let outcome = agent
        .run_turn(
            "Now add 58 to that result with the calculator.",
            &mut on_event,
        )
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    let answer = agent.history.last().unwrap().text();
    assert!(answer.contains("12379"), "expected 12379 in: {}", answer);

    let thinking_blocks = agent
        .history
        .iter()
        .flat_map(|m| &m.content)
        .filter(|b| matches!(b, generalist::ContentBlock::Thinking { .. }))
        .count();
    println!(
        "\nSMOKE OK — {} messages in history, {} thinking blocks replayed",
        agent.history.len(),
        thinking_blocks
    );
    Ok(())
}
