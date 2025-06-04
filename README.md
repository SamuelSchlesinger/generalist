# claude-rs

[![Crates.io](https://img.shields.io/crates/v/claude-rs)](https://crates.io/crates/claude-rs)
[![Documentation](https://docs.rs/claude-rs/badge.svg)](https://docs.rs/claude-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)

A type-safe Rust client library for Anthropic's Claude AI models and API, with a focus on tool-use capabilities and conversation management.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [CLI Chatbot Example](#cli-chatbot-example)
  - [Features](#chatbot-features)
  - [Running the Chatbot](#running-the-chatbot)
  - [Example Session](#example-session)
- [Usage Examples](#example-basic-usage)
  - [Basic Usage](#example-basic-usage)
  - [Using Tool Abstractions](#example-using-tool-abstractions)
- [Tool Permissions and Safety](#tool-permissions-and-safety)
- [Creating Custom Tools](#creating-custom-tools)
- [License](#license)

## CLI Chatbot Example

This repository includes a fully-featured CLI chatbot that demonstrates Claude's tool-use capabilities with complete transparency into what the bot is doing.

### Features

- 🎨 Beautiful terminal UI with colors and progress indicators 
- 🛠️ Multiple built-in tools:
  - **Haskell**: Execute Haskell code
  - **Bash**: Execute bash commands
  - **File operations**: Read, write, and list files
  - **System Info**: Gets current time, date, and OS information
- 🔐 **Permission modes** for tool safety:
  - Default (always allow)
  - Interactive (ask for each tool)
  - Logging (with audit trail)
  - Safe mode (only allow safe tools)
- 📊 Full transparency - see exactly when tools are being used
- ⏱️ Execution timing and status for each tool call
- 💬 Conversation history maintained throughout the session
- 🎯 Model selection (Haiku, Sonnet, or Opus)

### Running the Chatbot

1. Set your Claude API key:
   ```bash
   export CLAUDE_API_KEY='your-api-key-here'
   ```

2. Run the chatbot:
   ```bash
   cargo run --example chatbot
   ```

3. Select your preferred Claude model when prompted

4. The chatbot will prompt you for permission on each tool use with options:
   - **Yes (always allow)** - Always allow this specific tool
   - **Yes (just this once)** - Allow only this execution
   - **No (never allow)** - Never allow this specific tool
   - **No (just this once)** - Deny only this execution

5. Start chatting! Try commands like:
   - "What's 25 * 4?"
   - "What's the weather in London?"
   - "What time is it?"
   - "Calculate (100 + 50) * 2 and tell me the weather in Tokyo"

### Real-time Features

The chatbot now shows Claude's thoughts and tool executions in real-time:
- **Immediate text display** - See Claude's responses as they arrive
- **Live tool execution** - Watch tools being called and executed
- **Progress indicators** - Visual feedback during processing
- **Permission prompts** - Interactive safety controls with memory

### Example Session

```
╔═══════════════════════════════════════════════════════════╗
║            🤖 Claude CLI Chatbot with Tools 🛠️            ║
╚═══════════════════════════════════════════════════════════╝

Available tools:
  • calculator - Perform arithmetic calculations
  • weather - Get weather information
  • system_info - Get system information

You: What's 123 + 456?

[14:23:15] Claude: I'll calculate that for you.
🔧 Using tool: calculator with input: {"expression":"123 + 456"}
   ✓ calculator result: 579

[14:23:16] Claude: 123 + 456 equals 579.

📊 Tools used in this response:
   ✓ calculator (125ms)
```

## Overview {#overview}

`claude-rs` provides a simple, type-safe interface for interacting with Claude's API, 
with a focus on the Messages API and tool-based interactions. It allows you to:

- Send messages to Claude models
- Define and use tools that Claude can invoke
- Handle multi-turn conversations
- Process tool calls and results

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
claude-rs = "0.1.0"
```

## Features {#features}

- **Type-safe API client** for Claude's Messages API
- **Tool abstraction system** for creating and managing tools that Claude can use
- **Automatic tool execution** with conversation turn management  
- **Execution tracking** with detailed history and state management
- **Tool permission system** for safety and transparency
- **Async/await support** using Tokio
- **Comprehensive error handling**

## Example: Basic Usage

```rust
use claude::{Claude, ToolDef, Message, ContentBlock};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new Claude client
    let api_key = std::env::var("CLAUDE_API_KEY")?;
    let client = Claude::new(api_key, "claude-3-sonnet-20240229".to_string());

    // Define a calculator tool
    let calculator_tool = ToolDef {
        name: "calculator".to_string(),
        description: "A simple calculator that evaluates arithmetic expressions".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate"
                }
            },
            "required": ["expression"]
        })
    };

    // Send a message that will likely use the calculator tool
    let response = client.send_message(
        "What is 123 + 456?",
        vec![calculator_tool],
        Some("You're a helpful assistant."),
        None
    ).await?;

    // Process the response
    for block in &response.content {
        match block {
            ContentBlock::Text { text } => println!("Claude: {}", text),
            ContentBlock::ToolUse { name, input, id } => {
                println!("Claude wants to use tool: {} (ID: {})", name, id);
                println!("Input: {}", input);
                
                // In a real app, you would execute the tool here and send the result back
            },
            _ => {}
        }
    }

    Ok(())
}
```

## Example: Using Tool Abstractions

```rust
use claude::{Claude, Tool, ToolRegistry, CalculatorTool, WeatherTool};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client and tool registry
    let api_key = std::env::var("CLAUDE_API_KEY")?;
    let client = Claude::new(api_key, "claude-3-haiku-20240307".to_string());
    let mut registry = ToolRegistry::new();
    
    // Register tools
    registry.register(Arc::new(CalculatorTool))?;
    registry.register(Arc::new(WeatherTool))?;
    
    // Run a conversation turn with automatic tool execution
    let response = client.run_conversation_turn(
        "What's the weather in San Francisco and what is 25 * 4?",
        &mut registry,
        Some("You're a helpful assistant."),
        None,
        None,
    ).await?;
    
    // Claude will automatically use the tools and provide a final response
    for block in &response.content {
        if let ContentBlock::Text { text } = block {
            println!("Claude: {}", text);
        }
    }
    
    // Check execution history
    println!("\nTool execution history:");
    for execution in registry.execution_history() {
        println!("- {} ({}): {:?}", 
            execution.tool_name, 
            execution.id,
            execution.state
        );
    }
    
    Ok(())
}
```

## Tool Permissions and Safety

The library includes a comprehensive permission system that allows you to control tool execution for safety and transparency:

### Permission Handlers

```rust
use claude::{ToolRegistry, InteractivePermissions, PolicyPermissions, LoggingPermissions};
use std::sync::Arc;

// Always allow (default)
let mut registry = ToolRegistry::new();

// Interactive approval
let mut registry = ToolRegistry::with_permission_handler(
    Arc::new(InteractivePermissions::new(|request| {
        println!("Allow tool '{}' to execute?", request.tool_name);
        println!("Input: {:?}", request.input);
        // Return true to allow, false to deny
        prompt_user_for_approval()
    }))
);

// Policy-based permissions (whitelist)
let mut registry = ToolRegistry::with_permission_handler(
    Arc::new(PolicyPermissions::new(
        vec!["calculator".to_string(), "weather".to_string()],
        false // deny unlisted tools
    ))
);

// Logging permissions
let mut registry = ToolRegistry::with_permission_handler(
    Arc::new(LoggingPermissions) // logs all tool requests
);
```

### Built-in Permission Handlers

- **AlwaysAllowPermissions** (default) - Allows all tool executions
- **AlwaysDenyPermissions** - Denies all tool executions
- **InteractivePermissions** - Prompts for user approval via callback
- **PolicyPermissions** - Whitelist/blacklist based on tool names
- **LoggingPermissions** - Logs all requests and allows execution

### Custom Permission Handler

```rust
use claude::{ToolPermissionHandler, ToolExecutionRequest, PermissionDecision};
use async_trait::async_trait;

struct MyCustomPermissions;

#[async_trait]
impl ToolPermissionHandler for MyCustomPermissions {
    async fn check_permission(&self, request: &ToolExecutionRequest) -> PermissionDecision {
        // Custom logic here
        if request.tool_name == "dangerous_tool" {
            PermissionDecision::DenyWithReason("This tool is not allowed".to_string())
        } else {
            PermissionDecision::Allow
        }
    }
}
```

## Creating Custom Tools

```rust
use claude::{Tool, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct MyCustomTool;

#[async_trait]
impl Tool for MyCustomTool {
    fn name(&self) -> &str {
        "my_tool"
    }
    
    fn description(&self) -> &str {
        "A custom tool that does something specific"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Input parameter"
                }
            },
            "required": ["param"]
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let param = input.get("param")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("Missing 'param' field".to_string()))?;
        
        // Implement your tool logic here
        Ok(format!("Processed: {}", param))
    }
}
```

## Tool Usage Flow

The library supports two approaches for tool usage:

### Manual Tool Execution
1. Define tools with name, description, and input schema
2. Send a message to Claude with tool definitions
3. Claude responds with `ToolUse` content blocks
4. Your application executes the tools
5. Send tool results back to Claude
6. Claude provides final response

### Automatic Tool Execution
1. Implement the `Tool` trait for your tools
2. Register tools in a `ToolRegistry`
3. Use `run_conversation_turn()` method
4. The library automatically handles tool execution
5. Get Claude's final response with all tool results processed

## API Documentation

### Core Types

- `Claude` - The main client for interacting with the API
- `Message` - Represents a message in the conversation
- `ContentBlock` - Different types of content (text, tool use, tool result)
- `ToolDef` - Tool definition with name, description, and schema
- `Tool` - Trait for implementing executable tools
- `ToolRegistry` - Manages tool implementations and execution history
- `ToolExecution` - Tracks individual tool execution state and results

## License

This project is licensed under the MIT License.
