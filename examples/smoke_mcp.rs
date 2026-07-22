//! Live MCP smoke test in two stages:
//!
//! 1. Infrastructure: connect to an MCP server over Streamable HTTP,
//!    discover tools, call one directly through the registry.
//! 2. Full loop: a local model answers a question by calling the MCP tool
//!    from inside a code-mode script (`tools.tickerfacts_get_fundamentals`).
//!
//! Run with: `cargo run --example smoke_mcp [model]`
//! Requires Ollama on localhost:11434 (default model: qwen3:30b) and network
//! access to tickerfacts.com.

use generalist::mcp::{register_servers, McpConfig};
use generalist::provider::OpenAiProvider;
use generalist::{Agent, AgentEvent, ToolRegistry, TurnOutcome};

const MCP_URL: &str = "https://tickerfacts.com/mcp";

#[tokio::main]
async fn main() -> generalist::Result<()> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "qwen3:30b".to_string());

    // Stage 1: infrastructure, no model involved.
    let config: McpConfig = serde_json::from_str(&format!(
        r#"{{"servers": {{"tickerfacts": {{"url": "{}"}}}}}}"#,
        MCP_URL
    ))
    .expect("static config parses");

    let mut registry = ToolRegistry::new();
    let report = register_servers(&mut registry, &config).await;
    println!("{}", report.join("\n"));
    assert!(
        registry.has_tool("tickerfacts_get_fundamentals"),
        "expected tickerfacts_get_fundamentals, report: {:?}",
        report
    );
    // Progressive disclosure: MCP tools are code-only.
    assert!(!registry
        .get_tool_defs()
        .iter()
        .any(|d| d.name.starts_with("tickerfacts")));

    let direct = registry
        .execute_tool(
            "tickerfacts_get_fundamentals",
            serde_json::json!({"ticker": "AAPL"}),
            "direct".into(),
        )
        .await;
    assert_eq!(direct.outcome, generalist::ToolCallOutcome::Success);
    match &direct.block {
        generalist::ContentBlock::ToolResult { content, .. } => {
            assert!(content.contains("Apple"), "unexpected: {:.200}", content);
            println!("STAGE 1 OK — direct MCP call returned Apple fundamentals");
        }
        other => panic!("expected tool result, got {:?}", other),
    }

    // Stage 2: model drives the MCP tool through the code-mode bridge.
    let provider = OpenAiProvider::new(
        "ollama".into(),
        "http://localhost:11434/v1".into(),
        model.clone(),
    )?;
    let mut agent = Agent::new(
        Box::new(provider),
        registry,
        "You are a terse assistant. For data lookups, write one python script and call \
         tools via `import tools`.",
    );

    let mut bridged_mcp_calls = 0;
    let mut on_event = |event: AgentEvent| match event {
        AgentEvent::AssistantText(text) => println!("ASSISTANT: {}", text),
        AgentEvent::AssistantTextDelta(text) => {
            use std::io::Write;
            print!("{}", text);
            std::io::stdout().flush().ok();
        }
        AgentEvent::ApiCallFinished { .. } => println!(),
        AgentEvent::ToolCallStarted { name, input } => {
            if name.starts_with("tickerfacts") {
                bridged_mcp_calls += 1;
            }
            if name == "python" {
                println!("SCRIPT:\n{}", input["code"].as_str().unwrap_or(""));
            } else {
                println!("  BRIDGED CALL: {} {}", name, input);
            }
        }
        AgentEvent::ToolCallFinished { name, content, .. } => {
            println!("  RESULT ({}): {:.120}", name, content.replace('\n', " "));
        }
        AgentEvent::Notice(n) => println!("NOTICE: {}", n),
        _ => {}
    };

    let outcome = agent
        .run_turn(
            "Using the tickerfacts tool from a python script, look up Apple (AAPL) and report \
             its most recent fiscal year's revenue in dollars.",
            &mut on_event,
        )
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(
        bridged_mcp_calls >= 1,
        "model never called the MCP tool through the bridge"
    );

    println!(
        "\nMCP SMOKE OK (ollama/{}) — {} bridged MCP call(s)",
        model, bridged_mcp_calls
    );
    Ok(())
}
