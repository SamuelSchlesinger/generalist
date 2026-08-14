//! Live OpenRouter streaming smoke test.
//!
//! Run with: `cargo run --example smoke_openrouter [model]`
//! Requires `OPENROUTER_API_KEY` in the environment or `~/.generalist.env`.

use generalist::provider::{openrouter, OpenRouterProvider};
use generalist::{Agent, AgentEvent, Provider, ToolRegistry, TurnOutcome};

#[tokio::main]
async fn main() -> generalist::Result<()> {
    #[allow(deprecated)]
    let env_path = std::env::home_dir().unwrap().join(".generalist.env");
    if env_path.exists() {
        dotenvy::from_path(&env_path).ok();
    }
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| openrouter::DEFAULT_MODEL.to_string());
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("set OPENROUTER_API_KEY");
    let provider = OpenRouterProvider::new(api_key, model.clone())?;
    assert_eq!(provider.id(), "openrouter");

    let mut agent = Agent::new(
        Box::new(provider),
        ToolRegistry::new(),
        "Follow the user's requested output format exactly.",
    );
    let answer = std::cell::RefCell::new(String::new());
    let mut on_event = |event: AgentEvent| match event {
        AgentEvent::AssistantText(text) | AgentEvent::AssistantTextDelta(text) => {
            answer.borrow_mut().push_str(&text);
        }
        AgentEvent::Retrying { error, .. } => eprintln!("RETRY: {error}"),
        AgentEvent::Notice(notice) => eprintln!("NOTICE: {notice}"),
        _ => {}
    };

    let outcome = agent
        .run_turn(
            "Reply with exactly OPENROUTER_OK and do not call any tools.",
            &mut on_event,
        )
        .await?;
    assert_eq!(outcome, TurnOutcome::Completed);
    assert!(
        answer.borrow().contains("OPENROUTER_OK"),
        "unexpected answer: {}",
        answer.borrow()
    );
    println!("SMOKE OK — OpenRouter/{model}");
    Ok(())
}
