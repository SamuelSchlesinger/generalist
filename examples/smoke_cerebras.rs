//! Live Cerebras streaming smoke test with two tool-driven turns.
//!
//! Run with: `cargo run --example smoke_cerebras [model]`
//! Requires `CEREBRAS_API_KEY` in the environment or `~/.generalist.env`.

use generalist::provider::{cerebras, CerebrasProvider};
use generalist::tools::CalculatorTool;
use generalist::{Agent, AgentEvent, Provider, ToolRegistry, TurnOutcome};
use std::sync::Arc;

#[tokio::main]
async fn main() -> generalist::Result<()> {
    #[allow(deprecated)]
    let env_path = std::env::home_dir().unwrap().join(".generalist.env");
    if env_path.exists() {
        dotenvy::from_path(&env_path).ok();
    }
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| cerebras::DEFAULT_MODEL.to_string());
    let api_key = std::env::var("CEREBRAS_API_KEY").expect("set CEREBRAS_API_KEY");
    let provider = CerebrasProvider::new(api_key, model.clone())?;
    assert_eq!(provider.id(), "cerebras");

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool))?;
    let mut agent = Agent::new(
        Box::new(provider),
        registry,
        "Use the calculator tool for every arithmetic operation.",
    );

    let mut on_event = |event: AgentEvent| match event {
        AgentEvent::AssistantTextDelta(text) => print!("{text}"),
        AgentEvent::Retrying { error, .. } => eprintln!("RETRY: {error}"),
        AgentEvent::Notice(notice) => eprintln!("NOTICE: {notice}"),
        _ => {}
    };

    let outcome = agent
        .run_turn("Use the calculator to compute 111 * 111.", &mut on_event)
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    let first_answer = agent.history().last().unwrap().text();
    assert!(
        first_answer.replace(',', "").contains("12321"),
        "expected 12321 in: {first_answer}"
    );

    let outcome = agent
        .run_turn(
            "Use the calculator to add 58 to that result.",
            &mut on_event,
        )
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    let second_answer = agent.history().last().unwrap().text();
    assert!(
        second_answer.replace(',', "").contains("12379"),
        "expected 12379 in: {second_answer}"
    );

    assert!(generalist::history_tool_protocol_is_valid(agent.history()));
    let tool_calls = agent
        .history()
        .iter()
        .flat_map(|message| &message.content)
        .filter(|block| matches!(block, generalist::ContentBlock::ToolUse { .. }))
        .count();
    assert!(
        tool_calls >= 2,
        "expected a calculator call in both turns, observed {tool_calls}"
    );
    println!("\nSMOKE OK: Cerebras/{model}");
    Ok(())
}
