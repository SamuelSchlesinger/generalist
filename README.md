# Generalist 🤖

A powerful AI-powered command-line agent built with Rust that combines Claude's reasoning capabilities with 17 specialized tools. Designed for developers, researchers, and power users who need an intelligent assistant with real-world capabilities.

**Key Features:**
- 🧠 **Intelligent Problem Solving** - Uses Claude's advanced reasoning with means-ends analysis
- 🔧 **17 Built-in Tools** - File operations, web scraping, calculations, system administration, and more
- 🔐 **Granular Permissions** - Complete control over what tools can execute
- 💾 **Persistent Memory** - Enhanced memory system with tagging and search
- 📝 **Conversation Management** - Save and resume conversations with full context
- 🎨 **Beautiful CLI** - Real-time tool execution with progress indicators

## Quick Start

### Prerequisites

- **Rust** (latest stable version) - [Install Rust](https://rustup.rs/)
- **Claude API Key** - Get one from [Anthropic Console](https://console.anthropic.com/)
- **Firecrawl API Key** (optional) - Get one from [Firecrawl](https://firecrawl.dev/) for enhanced web scraping
- **Z3 Solver** (optional) - Install for constraint solving: `brew install z3` (macOS) or `apt install z3` (Linux)

### Installation

1. **Clone and build**:
   ```bash
   git clone https://github.com/SamuelSchlesinger/generalist.git
   cd generalist
   cargo build --release
   ```

2. **Create configuration file** `~/.generalist.env`:
   ```bash
   echo "CLAUDE_API_KEY=your-claude-api-key-here" > ~/.generalist.env
   echo "FIRECRAWL_API_KEY=your-firecrawl-api-key-here" >> ~/.generalist.env  # Optional
   ```

3. **Run the agent**:
   ```bash
   cargo run
   # or use the built binary
   ./target/release/generalist
   ```

### First Steps

Try these example interactions to get started:

**📊 Data & Calculations:**
- "What's the weather in Tokyo?"
- "Calculate the solution to: x + 2*y = 10, x - y = 1"
- "Solve for the optimal values: minimize x + y subject to x >= 0, y >= 1, x + 2*y <= 5"

**📁 File Operations:**
- "Show me the files in my home directory"
- "Read the contents of my .bashrc file"
- "Create a simple Python script that prints 'Hello World'"

**🌐 Web & Research:**
- "Search Wikipedia for 'quantum computing' and summarize the key concepts"
- "What are the latest developments in AI from Hacker News?"
- "Extract the main content from https://example.com"

**🧠 Productivity:**
- "Remember that I prefer using tabs over spaces in Python code"
- "Add 'Review quarterly budget' to my todo list"
- "Think deeply about the trade-offs between microservices and monolithic architecture"

## Key Features

- **🛠️ 17 Built-in Tools** - Everything from file operations to web scraping
- **🔐 Permission System** - You control what tools can run
- **💾 Save Conversations** - Resume chats later with `/save` and `/load`
- **🎨 Beautiful UI** - See exactly what the generalist is doing in real-time

## Available Tools

The generalist agent comes with 17 specialized tools organized into functional categories:

### 📁 File Operations
- **`read_file`** - Read content from any file on the system
- **`patch_file`** - Apply diffs/patches to modify files safely
- **`list_directory`** - Browse and explore directory structures

### 💻 System Administration
- **`bash`** - Execute shell commands with full output capture
- **`system_info`** - Get detailed system information and diagnostics

### 🧮 Computing & Mathematics
- **`calculator`** - Evaluate mathematical expressions with support for trigonometry, logarithms, and more
- **`z3_solver`** - Advanced constraint solving, optimization, and theorem proving using Microsoft's Z3 SMT solver

### 🌐 Web & Data Retrieval
- **`http_fetch`** - Make HTTP requests to APIs and web services
- **`weather`** - Get current weather information for any city using Open-Meteo API
- **`wikipedia`** - Search and retrieve Wikipedia content with intelligent summarization

#### Advanced Web Scraping (Firecrawl Integration)
- **`firecrawl_extract`** - Extract clean content from single web pages, handling JavaScript and removing ads
- **`firecrawl_crawl`** - Systematically crawl entire websites with depth control and filtering
- **`firecrawl_map`** - Discover and map website structure, creating comprehensive sitemaps
- **`firecrawl_search`** - Enhanced web search that returns actual page content, not just links

### 🧠 Productivity & Intelligence
- **`enhanced_memory`** - Persistent memory system with tagging, search, and cross-session storage
- **`todo`** - Task management system with JSON persistence and status tracking
- **`think`** - Deep analysis and reasoning prompts for complex problem-solving

### Tool Architecture

Each tool implements a standardized interface with:
- **JSON Schema Validation** - Ensures type safety and clear parameter requirements
- **Permission Control** - Granular execution control with user consent
- **Error Handling** - Comprehensive error messages with usage examples
- **Documentation** - Self-describing tools with built-in help

## Security & Permission System

The generalist agent prioritizes safety through a comprehensive permission system that gives you complete control over tool execution.

### How Permissions Work

Before any tool is executed, you'll see a detailed permission prompt:

```
⚠️  Tool Permission Request
──────────────────────────────────────────────────
Tool: bash
Description: Execute a bash command
Input: {"command": "ls -la"}

Allow this tool to execute?
> Yes (always allow this tool)
  Yes (just this once)
  No (never allow this tool)
  No (just this once)
```

### Permission Options

- **Always Allow** - Trust this tool completely for the session
- **Just This Once** - Allow this specific execution only
- **Never Allow** - Block this tool type entirely
- **Just This Once (No)** - Deny this execution but ask again next time

### Permission Persistence

- Permissions are remembered within a conversation session
- When you save and load conversations, permission settings are preserved
- This allows you to build trusted tool configurations over time
- Each tool request shows complete input parameters for transparency

### Safety Features

- **Full Transparency** - Every tool call shows exact parameters before execution
- **Granular Control** - Approve or deny individual tool operations
- **No Surprises** - The agent can't execute tools without explicit permission
- **Audit Trail** - All tool executions are logged and can be reviewed

## Built-in Commands

The agent supports several slash commands for session management:

- **`/save [name]`** - Save current conversation with optional custom name (defaults to timestamp)
- **`/load`** - Load a previously saved conversation from an interactive menu
- **`/model`** - Switch between available Claude models (Claude-3 Haiku, Sonnet, Opus)
- **`/help`** - Display available commands and usage information
- **`exit` or `quit`** - Safely exit the agent

### Conversation Management

Conversations are automatically saved to `~/.chatbot_history/` as JSON files containing:
- Complete message history
- Tool execution records
- Permission settings
- Model configuration

This allows you to resume complex problem-solving sessions exactly where you left off.

## Advanced Usage

### Using as a Library

The generalist agent can also be used as a Rust library for building custom AI-powered applications.

Add to your `Cargo.toml`:
```toml
[dependencies]
claude = "0.1.0"
```

### Basic Library Example

```rust
use claude::{Claude, ToolRegistry, tools::*};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize Claude client
    let client = Claude::new(
        std::env::var("CLAUDE_API_KEY")?, 
        "claude-3-7-sonnet-latest".to_string()
    );
    
    // Create tool registry and add tools
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool))?;
    registry.register(Arc::new(WeatherTool))?;
    registry.register(Arc::new(WikipediaTool))?;
    
    // Run a conversation turn with tool support
    let response = client.run_conversation_turn(
        "What's the weather like in Paris and what's 25 * 4?",
        &mut registry,
        Some("You are a helpful assistant."),
        None,
        None,
    ).await?;
    
    println!("Response: {}", response);
    Ok(())
}
```

### Creating Custom Tools

Implement the `Tool` trait to create your own tools:

```rust
use claude::{Tool, Result, Error};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct CustomTool;

#[async_trait]
impl Tool for CustomTool {
    fn name(&self) -> &str {
        "custom_tool"
    }

    fn description(&self) -> &str {
        "A custom tool that demonstrates the Tool trait"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A message to process"
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let message = input.get("message")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("Missing message field".to_string()))?;
        
        Ok(format!("Processed: {}", message))
    }
}
```

### Permission Handlers

Customize permission handling for different use cases:

```rust
use claude::{ToolRegistry, AlwaysAllowPermissions, PolicyPermissions};

// Always allow all tools (for trusted environments)
let mut registry = ToolRegistry::with_permission_handler(
    Box::new(AlwaysAllowPermissions)
);

// Or implement custom permission logic
let policy = PolicyPermissions::new()
    .allow_tool("calculator")
    .allow_tool("weather")
    .deny_tool("bash");

let mut registry = ToolRegistry::with_permission_handler(
    Box::new(policy)
);
```

## Architecture & Design Philosophy

### Problem-Solving Methodology

The generalist agent implements a sophisticated problem-solving approach based on **means-ends analysis**, a methodology pioneered in early AI research:

1. **State Assessment** - Analyze the current situation and desired outcome
2. **Gap Identification** - Determine what differs between current and goal states  
3. **Operator Selection** - Choose appropriate tools to reduce the differences
4. **Execution & Iteration** - Apply tools systematically and monitor progress

### Historical Inspiration

This architecture draws from pioneering AI systems:

- **General Problem Solver (GPS)** (Newell & Simon, 1957) - Introduced means-ends analysis for systematic problem decomposition
- **STRIPS** (Stanford Research Institute, 1971) - Advanced automated planning with operator-based state space search
- **SHRDLU** (Winograd, 1970) - Demonstrated sophisticated reasoning about goals and actions

### Modern Implementation

The generalist agent modernizes these classical approaches by:

- **Tool Ecosystem** - 17 specialized tools covering file operations, web scraping, mathematics, and system administration
- **Safety First** - Comprehensive permission system prevents unwanted tool execution
- **Real-world Integration** - Direct integration with APIs, file systems, and external services
- **Conversational Interface** - Natural language interaction with full context preservation

### Core Components

- **`Claude`** - Main API client handling communication with Anthropic's models
- **`ToolRegistry`** - Manages available tools and tracks execution history
- **`PermissionHandler`** - Controls tool execution with user consent
- **`ChatbotState`** - Maintains conversation history and session state
- **`Tool` Trait** - Standardized interface for all tool implementations

## Contributing

We welcome contributions from the community! Here are some ways to get involved:

### Adding New Tools

1. **Implement the `Tool` trait** - Create a new file in `src/tools/`
2. **Add comprehensive tests** - Ensure your tool handles edge cases gracefully
3. **Update documentation** - Include clear examples and usage patterns
4. **Follow the established patterns** - Look at existing tools for architectural guidance

### Improving Core Features

- **Enhanced UI/UX** - Better progress indicators, error handling, or visual design
- **Performance Optimizations** - Faster tool execution or reduced memory usage
- **New Permission Handlers** - More sophisticated access control mechanisms
- **Extended CLI Commands** - Additional slash commands for power users

### Documentation & Examples

- **Tutorial Content** - Step-by-step guides for common use cases
- **Tool-specific Documentation** - Detailed guides for complex tools like Z3 solver
- **Integration Examples** - Demonstrations of using the agent with other systems
- **Video Tutorials** - Screen recordings showing real-world problem solving

### Quality Improvements

- **Test Coverage** - Unit tests, integration tests, and property-based testing
- **Error Handling** - Better error messages and recovery mechanisms
- **Code Organization** - Refactoring for maintainability and extensibility
- **Accessibility** - Making the CLI more accessible to users with different needs

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/SamuelSchlesinger/generalist.git
cd generalist

# Install dependencies (including optional ones for development)
cargo build

# Run tests
cargo test

# Check code formatting
cargo fmt --check

# Run linter
cargo clippy
```

### Submitting Changes

1. **Fork the repository** and create a feature branch
2. **Write tests** for any new functionality
3. **Update documentation** to reflect your changes
4. **Submit a pull request** with a clear description of the changes
5. **Respond to feedback** during the review process

## License

MIT License - see LICENSE file

---

Built with ❤️ using [Claude API](https://www.anthropic.com/) and [Firecrawl](https://firecrawl.com/)
