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
- [`profile`] — one immutable layout for profile-scoped configuration and storage
- [`tools`] — a batteries-included tool set (bash, file ops, web, ...)

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

The full conversation is available through `agent.history()` and survives
errors, so a failed request never loses the record of tool calls that already
ran.
*/

#[cfg(not(unix))]
compile_error!("generalist supports Unix-like systems only");

pub mod agent;
pub mod clipboard;
pub(crate) mod codemode;
pub mod command;
pub mod error;
pub mod execution;
pub mod goal;
pub mod history;
pub mod mcp;
pub mod memory;
pub mod model_trace;
pub mod permissions;
pub mod profile;
pub mod provider;
pub mod runtime;
pub mod scope;
pub mod skills;
pub mod state;
pub(crate) mod storage;
pub(crate) mod subprocess;
pub mod tool;
pub mod tools;
pub mod tui;
pub mod types;

pub use agent::{history_tool_protocol_is_valid, Agent, AgentEvent, TurnOutcome};
pub use clipboard::{conversation_transcript, latest_assistant_reasoning, latest_assistant_text};
pub use command::{
    complete_local_command, is_local_command, parse_local_command, CommandCompletion, CommandSpec,
    CopyCommand, GoalCommand, HistoryCommand, LocalCommand, McpCommand, MemoryCommand,
    PermissionCommand, ToolsCommand, UsageCommand, COMMAND_SPECS,
};
pub use error::{Error, Result};
pub use execution::{ExecutionState, ToolExecution};
pub use goal::{is_goal_continuation_prompt, GOAL_CONTINUATION_PROMPT, UPDATE_GOAL_TOOL_NAME};
pub use history::{
    ArchivedConversation, ArchivedConversationEvent, ConversationSummary, HistoryStore,
};
pub use memory::{
    default_memory_path, CorruptEpisode, Episode, EpisodeEvent, EpisodeExport, EpisodeMatches,
    EpisodeOutcome, EpisodeSummary, EpisodicMemory, ForgetResult, MemoryEvent, MemoryStatus,
};
pub use model_trace::{
    ArchiveModelAction, AsyncModelAction, MemoryModelAction, ModelKind, ModelTrace,
    ModelTraceSnapshot,
};
pub use permissions::{
    remembers_exact_input, AlwaysAllowPermissions, AlwaysDenyPermissions, MemoryPermissionHandler,
    PermissionBrokerPrompt, PermissionChoice, PermissionDecision, PermissionPrompt,
    PermissionRequest, PermissionUiEvent, PolicyPermissions, RememberedPermissionPolicy,
    ToolExecutionRequest, ToolPermissionHandler,
};
pub use profile::ProfilePaths;
pub use provider::Provider;
pub use runtime::{
    CancelHandle, DeliveryMode, PromptClaim, PromptId, PromptQueue, PromptSource, QueuedPrompt,
    TurnControl,
};
pub use scope::{discover_project_root, ScopeFilter, WorkspaceScope};
pub use state::SavedState;
pub use tool::{
    DisclosureCapability, DisclosureGrant, Tool, ToolAuthorization, ToolCallOutcome,
    ToolCallResult, ToolRegistry,
};
pub use types::{
    estimate_tokens, truncate_middle, CompletionDelta, CompletionLimits, CompletionRequest,
    CompletionResponse, ContentBlock, Message, MessageOrigin, StopReason, ToolDef, ToolUse, Usage,
};
