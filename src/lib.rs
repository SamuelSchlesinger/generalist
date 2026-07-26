/*!
# generalist

A provider-agnostic agent library and CLI.

The core is small and deliberate:

- [`types`] — neutral conversation types ([`Message`], [`ContentBlock`], ...)
- [`provider`] — the [`Provider`] trait plus Anthropic and OpenAI-compatible
  implementations
- [`tool`] — the [`Tool`] trait and [`ToolRegistry`], with permission gating
- [`agent`] — [`Agent`], the request → tool → result loop, reporting progress
  through [`AgentEvent`] callbacks
- [`permissions`] — pluggable [`ToolPermissionHandler`] implementations
- [`tools`] — a batteries-included tool set (bash, file ops, web, memory, ...)

## Quick start

```rust,no_run
use generalist::{Agent, AgentEvent, ToolRegistry};
use generalist::provider::AnthropicProvider;
use generalist::tools::CalculatorTool;
use std::sync::Arc;

# async fn example() -> generalist::Result<()> {
let provider = AnthropicProvider::new("api-key".into(), "claude-opus-4-8".into())?;
let mut registry = ToolRegistry::new();
registry.register(Arc::new(CalculatorTool))?;

let mut agent = Agent::new(Box::new(provider), registry, "You are a helpful assistant.");
let outcome = agent
    .run_turn("What is 2 + 2?", &mut |event| {
        if let AgentEvent::AssistantText(text) = event {
            println!("{}", text);
        }
    })
    .await?;
# let _ = outcome;
# Ok(())
# }
```

The full conversation lives in `agent.history` and survives errors, so a
failed request never loses the record of tool calls that already ran.
*/

#[cfg(not(unix))]
compile_error!("generalist supports Unix-like systems only");

pub mod agent;
pub(crate) mod codemode;
pub mod error;
pub mod execution;
pub mod mcp;
pub mod permissions;
pub mod provider;
pub mod runtime;
pub mod skills;
pub mod state;
pub mod tool;
pub mod tools;
pub mod tui;
pub mod types;

pub use agent::{history_tool_protocol_is_valid, Agent, AgentEvent, TurnOutcome};
pub use error::{Error, Result};
pub use execution::{ExecutionState, ToolExecution};
pub use permissions::{
    AlwaysAllowPermissions, AlwaysDenyPermissions, MemoryPermissionHandler, PermissionBrokerPrompt,
    PermissionChoice, PermissionDecision, PermissionPrompt, PermissionRequest, PermissionUiEvent,
    PolicyPermissions, ToolExecutionRequest, ToolPermissionHandler,
};
pub use provider::Provider;
pub use runtime::{
    CancelHandle, DeliveryMode, PromptClaim, PromptId, PromptQueue, QueuedPrompt, TurnControl,
};
pub use state::SavedState;
pub use tool::{Tool, ToolCallOutcome, ToolCallResult, ToolRegistry};
pub use types::{
    estimate_tokens, truncate_middle, CompletionRequest, CompletionResponse, ContentBlock, Message,
    StopReason, ToolDef, ToolUse, Usage,
};
