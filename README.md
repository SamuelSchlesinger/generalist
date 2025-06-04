# Generalist 🤖

A powerful command-line generalist command-line agent powered by Claude AI and Firecrawl, with 17 built-in tools and a focus on safety and transparency.

## Quick Start

1. **Create your config file** `~/.generalist.env`:
   ```bash
   CLAUDE_API_KEY=your-claude-api-key
   FIRECRAWL_API_KEY=your-firecrawl-api-key
   ```

2. **Run the generalist**:
   ```bash
   cargo run
   ```

3. **Start chatting!** Try these examples:
   - "What's the weather in Tokyo?"
   - "Calculate 25 * 4"
   - "Search Wikipedia for Rust programming"
   - "List files in the current directory"

## Key Features

- **🛠️ 17 Built-in Tools** - Everything from file operations to web scraping
- **🔐 Permission System** - You control what tools can run
- **💾 Save Conversations** - Resume chats later with `/save` and `/load`
- **🎨 Beautiful UI** - See exactly what the generalist is doing in real-time

## Available Tools

### 📁 Files
- `read_file` - Read any file
- `patch_file` - Modify files with diffs
- `list_directory` - Browse folders

### 💻 System
- `bash` - Run shell commands
- `system_info` - Get system details

### 🧮 Computing
- `calculator` - Math expressions
- `z3_solver` - Solve complex constraints

### 🌐 Web
- `http_fetch` - Make HTTP requests
- `weather` - Get weather info
- `wikipedia` - Search Wikipedia
- `firecrawl_*` - Advanced web scraping (4 tools)

### 🧠 Productivity
- `enhanced_memory` - Store information
- `todo` - Manage tasks
- `think` - Deep analysis

## Permission System

When the generalist wants to use a tool, you'll see:

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

Your choices are remembered during the session.

## Commands

- `/save [name]` - Save conversation
- `/load` - Load conversation
- `/model` - Change Claude model
- `/help` - Show help
- `exit` - Quit

## Installation

### From Source

```bash
git clone https://github.com/SamuelSchlesinger/generalist.git
cd generalist
cargo build --release
./target/release/generalist
```

### As a Library

Add to your `Cargo.toml`:
```toml
[dependencies]
claude = "0.1.0"
```

## Basic Library Usage

```rust
use claude::{Claude, ToolRegistry, tools::*};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Claude::new(
        std::env::var("CLAUDE_API_KEY")?, 
        "claude-3-7-sonnet-latest".to_string()
    );
    
    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool))?;
    
    let response = client.run_conversation_turn(
        "What's 25 * 4?",
        &mut registry,
        None,
        None,
        None,
    ).await?;
    
    // Print response
    Ok(())
}
```

## Historical Inspiration

This project draws inspiration from pioneering AI systems that established foundational problem-solving methodologies:

- **General Problem Solver (GPS)** - Developed by Allen Newell and Herbert A. Simon (1957), GPS introduced means-ends analysis as a systematic approach to problem solving, breaking down complex problems by identifying differences between current and goal states.
- **STRIPS** - The Stanford Research Institute Problem Solver (1971) advanced automated planning with operator-based state space search.
- **SHRDLU** - Terry Winograd's natural language understanding system (1970) demonstrated sophisticated reasoning about goals and actions in constrained domains.

The generalist agent architecture continues this tradition by implementing means-ends analysis with modern AI capabilities and a rich set of tools for real-world problem solving.

### References

- Newell, A., & Simon, H. A. (1972). *Human Problem Solving*. Prentice-Hall.
- Fikes, R. E., & Nilsson, N. J. (1971). STRIPS: A new approach to the application of theorem proving to problem solving. *Artificial Intelligence*, 2(3-4), 189-208.
- Winograd, T. (1972). *Understanding Natural Language*. Academic Press.

## Contributing

We welcome contributions! Feel free to:
- Add new tools
- Improve the UI
- Fix bugs
- Add features

## License

MIT License - see LICENSE file

---

Built with ❤️ using [Claude API](https://www.anthropic.com/) and [Firecrawl](https://firecrawl.com/)
