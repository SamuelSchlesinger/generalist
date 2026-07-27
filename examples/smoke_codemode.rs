//! Verifies real "code mode" end to end with a live model: the model should
//! answer a task by writing ONE python script that calls a tool through the
//! generated `tools` bridge, instead of issuing direct tool calls.
//!
//! Run with: `cargo run --example smoke_codemode [model]`
//! Requires Ollama on localhost:11434 (default model: qwen3:30b).

use generalist::provider::OpenAiProvider;
use generalist::tools::CalculatorTool;
use generalist::{Agent, AgentEvent, ToolRegistry, TurnOutcome};
use std::sync::Arc;

#[tokio::main]
async fn main() -> generalist::Result<()> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "qwen3:30b".to_string());

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
        "You are a terse assistant. For multi-step work, write one python script; inside \
         scripts, call tools via `import tools`.",
    );

    let mut python_calls = 0;
    let mut bridged_calc_calls = 0;
    let mut on_event = |event: AgentEvent| match event {
        AgentEvent::AssistantText(text) => println!("ASSISTANT: {}", text),
        AgentEvent::AssistantTextDelta(text) => {
            use std::io::Write;
            print!("{}", text);
            std::io::stdout().flush().ok();
        }
        AgentEvent::ApiCallFinished { .. } => println!(),
        AgentEvent::ToolCallStarted { name, input } => {
            match name.as_str() {
                "python" => {
                    python_calls += 1;
                    println!("SCRIPT:\n{}", input["code"].as_str().unwrap_or(""));
                }
                _ => {
                    if python_calls > 0 {
                        bridged_calc_calls += 1;
                    }
                    println!("  BRIDGED CALL: {} {}", name, input);
                }
            };
        }
        AgentEvent::ToolCallFinished { name, content, .. } => {
            println!(
                "  RESULT ({}): {}",
                name,
                content.lines().next().unwrap_or("")
            );
        }
        AgentEvent::Notice(n) => println!("NOTICE: {}", n),
        _ => {}
    };

    let outcome = agent
        .run_turn(
            "In a single python script: use tools.calculator to evaluate \"3^7\", then double \
             the result in Python and report the final number.",
            &mut on_event,
        )
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(python_calls >= 1, "model never used the python tool");
    assert!(
        bridged_calc_calls >= 1,
        "script never called the calculator through the bridge"
    );
    let answer = agent.history().last().unwrap().text();
    assert!(
        answer.replace([',', ' '], "").contains("4374"),
        "expected 4374 in: {}",
        answer
    );

    println!(
        "\nCODE MODE OK (ollama/{}) — {} script(s), {} bridged tool call(s)",
        model, python_calls, bridged_calc_calls
    );
    Ok(())
}
