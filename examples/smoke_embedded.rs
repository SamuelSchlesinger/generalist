//! Live end-to-end agent turn through the in-process dense Qwen MLX provider.
//!
//! Build the pinned native stack first with `make embedded`, then use the
//! native cache paths documented in `native/mlx/README.md` to build and run
//! this example through Cargo.

use generalist::provider::embedded::{EmbeddedProvider, DEFAULT_EMBEDDED_MODEL};
use generalist::tools::CalculatorTool;
use generalist::{Agent, AgentEvent, ToolRegistry, TurnOutcome, Usage};
use std::sync::Arc;

#[tokio::main]
async fn main() -> generalist::Result<()> {
    let provider = EmbeddedProvider::qwen(DEFAULT_EMBEDDED_MODEL)?;
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool))?;
    let mut agent = Agent::new(
        Box::new(provider),
        registry,
        "You are a precise assistant. Use the calculator for arithmetic and follow the requested final-answer format exactly.",
    );
    agent.code_mode = false;
    agent.max_tokens = Some(512);
    agent.max_retries = 0;

    let mut answer = String::new();
    let mut reasoning_bytes = 0usize;
    let mut first_usages: Vec<Usage> = Vec::new();
    let outcome = agent
        .run_turn(
            "Use the calculator to compute 19 * 23. End your final answer with the exact marker EMBEDDED_OK.",
            &mut |event| match event {
                AgentEvent::ReasoningDelta(reasoning) => reasoning_bytes += reasoning.len(),
                AgentEvent::AssistantTextDelta(text) => answer.push_str(&text),
                AgentEvent::AssistantText(text) => answer.push_str(&text),
                AgentEvent::ApiCallFinished { usage } => first_usages.extend(usage),
                _ => {}
            },
        )
        .await?;

    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(answer.contains("437"), "expected 437 in: {answer}");
    assert!(
        answer.trim_end().ends_with("EMBEDDED_OK"),
        "expected final marker in: {answer}"
    );

    assert!(
        first_usages.len() >= 2,
        "first turn did not execute a calculator tool loop"
    );
    assert!(
        first_usages[1].cache_read_input_tokens.unwrap_or(0) > 0,
        "tool follow-up did not reuse the prompt checkpoint: {:?}",
        first_usages[1]
    );
    let mut cached_answer = String::new();
    let mut second_usage: Option<Usage> = None;
    let second_outcome = agent
        .run_turn(
            "Using the prior result, reply with only the exact marker CACHE_OK.",
            &mut |event| match event {
                AgentEvent::AssistantTextDelta(text) => cached_answer.push_str(&text),
                AgentEvent::AssistantText(text) => cached_answer.push_str(&text),
                AgentEvent::ApiCallFinished { usage } => second_usage = usage,
                _ => {}
            },
        )
        .await?;
    assert_eq!(second_outcome, TurnOutcome::Completed);
    assert_eq!(cached_answer.trim(), "CACHE_OK");
    let second_usage = second_usage.expect("second embedded turn must report usage");
    assert!(
        second_usage.cache_read_input_tokens.unwrap_or(0) > 0,
        "second turn did not reuse the exact prompt checkpoint: {second_usage:?}"
    );
    println!(
        "EMBEDDED SMOKE OK: reasoning_bytes={reasoning_bytes} first_input={} \
         tool_cache_read={} second_input={} second_cache_read={} answer={} cached_answer={}",
        first_usages[0].input_tokens,
        first_usages[1].cache_read_input_tokens.unwrap_or(0),
        second_usage.input_tokens,
        second_usage.cache_read_input_tokens.unwrap_or(0),
        answer.trim(),
        cached_answer.trim(),
    );
    Ok(())
}
