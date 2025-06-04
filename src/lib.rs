/*!
A Rust client library for Anthropic's Claude AI models and API.

This crate provides a simple, type-safe interface for interacting with Claude's API,
with a focus on the Messages API and tool-based interactions. It allows you to:

- Send messages to Claude models
- Define and use tools that Claude can invoke
- Handle multi-turn conversations
- Process tool calls and results
- Control tool execution with permission handlers

## Quick Start

```rust,no_run
# async fn example() -> Result<(), Box<dyn std::error::Error>> {
use claude::{Claude, ToolRegistry};

// Create a Claude client
let client = Claude::new(
    "your-api-key".to_string(),
    "claude-3-haiku-20240307".to_string()
);

// Create a tool registry and register tools
let mut registry = ToolRegistry::new();
// Register your custom tools here
// registry.register(Arc::new(MyCustomTool))?;

// Have a conversation with automatic tool execution
let response = client.run_conversation_turn(
    "What's the weather in London?",
    &mut registry,
    Some("You are a helpful assistant."),
    None,  // No conversation history
    None   // Use default max iterations
).await?;
# Ok(())
# }
```

## Features

- **Easy-to-use client**: Simple API for interacting with Claude models
- **Strongly typed**: Full type safety with Rust's type system
- **Tool support**: Define custom tools that Claude can use during conversations
- **Permission control**: Fine-grained control over tool execution with permission handlers
- **Conversation management**: Track multi-turn conversations with message history
- **Real-time execution**: Process tool calls as they happen
- **Comprehensive error handling**: Detailed error types for debugging

## Main Components

- [`Claude`]: The main client for interacting with the API
- [`Tool`]: Trait for implementing custom tools
- [`ToolRegistry`]: Manages available tools and tracks execution history
- [`Message`] and [`ContentBlock`]: Core types for conversation messages
- [`ToolPermissionHandler`]: Control whether tools can be executed
*/

// Re-export main types from submodules
pub use client::{Claude, MESSAGES_ENDPOINT};
pub use error::{Error, Result};
pub use message::{Message, ContentBlock, ToolUse};
pub use request::{MessageRequest, MessageResponse, ToolDef, Usage};
pub use tool::{Tool, ToolRegistry};
pub use permissions::{
    ToolPermissionHandler, PermissionDecision, ToolExecutionRequest,
    AlwaysAllowPermissions, AlwaysDenyPermissions, LoggingPermissions,
    InteractivePermissions, PolicyPermissions, MemoryPermissionHandler
};
pub use execution::{ExecutionState, ToolExecution};
pub use state::ChatbotState;

// Modules
pub mod client;
pub mod error;
pub mod message;
pub mod request;
pub mod tool;
pub mod permissions;
pub mod execution;
pub mod tools;
pub mod state;