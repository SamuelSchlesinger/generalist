# generalist

A provider-agnostic CLI agent in Rust. Works with the Anthropic Messages API or any
OpenAI-compatible endpoint (OpenAI, Ollama, Groq, Mistral, vLLM, LM Studio). The
library is small: neutral conversation types, a `Provider` trait, a tool registry with
permission gating, and an agent loop that reports progress through event callbacks.

## Install and run

```bash
cargo build --release

# Keys go in the environment or ~/.generalist.env:
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> ~/.generalist.env
echo 'OPENAI_API_KEY=sk-...'        >> ~/.generalist.env   # and/or
echo 'FIRECRAWL_API_KEY=fc-...'     >> ~/.generalist.env   # optional, web tools

./target/release/generalist

# Local models need no key:
./target/release/generalist --local                    # qwen3.6:35b-a3b
./target/release/generalist --local qwen2.5-coder:32b  # or name one
```

`--local [model]` skips provider selection and uses `http://localhost:11434/v1`; set
`OPENAI_BASE_URL` for other local servers. Tool calling requires a tool-capable model
(`qwen3.6`, `qwen3`, `qwen2.5-coder`, `devstral`).

Optional binaries: `z3` (constraint solver), `patch` (file editing; present on any Unix).

Smoke tests, all live end-to-end: `cargo run --example smoke` (Anthropic),
`smoke_ollama` (local tool loop), `smoke_codemode` (local code-mode bridge).

## Usage

Type a request; the agent calls tools and reports back. On Unix, code mode is on by
default and `python` is the only model-facing tool. Scripts reach all registered
capabilities through `import tools`: bash, file read/patch, directory listing, HTTP
fetch, web search/scrape/crawl (Firecrawl), Wikipedia, weather, Z3, persistent memory,
and todo list. (Calculator, system-info, and think tools were retired from the CLI:
python and bash subsume them.)

Responses stream as they generate. Commands: `/save`, `/load`, `/model` (switch
provider or model mid-conversation), `/compact` (summarize older history to free
context), `/clear`, `/help`, `exit`. Every turn autosaves to
`~/.generalist_history/autosave.json`. `/load` also reads the legacy
`~/.chatbot_history` directory.

## Permissions

Every tool call prompts for approval and shows the full input (diffs rendered as
diffs). Choices: allow always, allow once, deny always, deny once. Decisions persist
across save/load.

Caveats:

- "Always allow" is per tool name. Always-allowing `bash` approves every future
  command; the command is still printed before it runs.
- The prompts are not a sandbox. An agent that can write and run code can circumvent
  in-process fences; use a container or dedicated user for real isolation.
- Fetched web content is untrusted input; the approval step exists mainly to catch
  prompt injection acting on it.
- `http_fetch` rejects localhost, private, and link-local addresses, including
  redirect hops and DNS results. Best effort, not a guarantee.

## Agent loop

Follows what pi, opencode, and Claude Code converged on:

- History survives mid-turn API errors, including tool calls that already ran.
- Transient API errors retry 3 times with exponential backoff.
- Tool results are truncated before entering history. Bash and python keep the tail
  (where errors are) and spill full output to a temp file the model can read back.
- Tool calls in a response that hit the output-token limit are failed, not executed —
  their arguments may be silently incomplete. The model is asked to re-issue them.
- A denied tool call ends the turn so you can redirect. Denial is a structured
  outcome, never inferred from result text.
- On Anthropic, the system prompt and conversation prefix carry prompt-cache
  breakpoints, which cuts input cost substantially in long sessions.
- Responses stream (SSE) on both providers; a stream that dies mid-message is
  retried rather than treated as a complete answer.
- When context passes a threshold (default 150k tokens, configurable), older
  history is summarized in place and recent turns stay verbatim. `/compact`
  triggers it manually. Local models with small context windows may want a much
  lower `compaction_threshold_tokens`.

## Code mode

The agent advertises exactly one tool, `python`, when code mode is enabled (the Unix
default). Every registered tool is available only to scripts through a generated
`tools` module:

```python
import tools
pages = tools.firecrawl_search(query="rust async runtimes")
# result stays in the script; the model never sees it unless printed
print(extract_urls(pages))
```

Ordinary tool descriptions and schemas are folded into the `python` tool description,
so the model can use them in its first script without a discovery round-trip. Calls are
served over a Unix socket and pass through the same permission gate as direct mode.
Results return to the script, not the model, so one script can perform a long sequence,
process megabytes of output, validate the result, and print only the conclusion. Script
errors come back as tool results, so the model can fix and re-run. This is the pattern
from CodeAct, Cloudflare's Code Mode, Anthropic's code-execution-with-MCP, and the
"Code as Agent Harness" survey (arXiv 2605.18747).

Library users can opt back into independently advertised direct tools with
`agent.code_mode = false`. Registering a custom tool named `python` also overrides the
built-in runner.

Relation to CaMeL: tool output that stays inside a script cannot prompt-inject the
model, and per-call approval acts as the policy check. There is no data-flow/taint
tracking, and scripts run unsandboxed with your privileges.

## MCP

Configure servers in `~/.generalist/mcp.json`; both Streamable HTTP and stdio
transports are supported:

```json
{
  "servers": {
    "tickerfacts": { "url": "https://tickerfacts.com/mcp" },
    "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] }
  }
}
```

Discovered tools register as `<server>_<tool>` with progressive disclosure. Like every
registered tool in code mode, they are callable only from scripts; unlike ordinary
tools, their heavy schemas (often 10k+ tokens per server) are omitted from the
model-facing `python` description. Full schemas remain in the generated module's
docstrings (`print(tools.tickerfacts_get_fundamentals.__doc__)`). Context cost scales
with what a script uses, not what a server offers. Bridged MCP calls pass through the
permission gate like any other tool. A failed server logs a warning and is skipped.
`cargo run --example smoke_mcp` verifies the stack live.

## Library use

```rust,no_run
use generalist::{Agent, AgentEvent, ToolRegistry};
use generalist::provider::OpenAiProvider;
use generalist::tools::CalculatorTool;
use std::sync::Arc;

#[tokio::main]
async fn main() -> generalist::Result<()> {
    let provider = OpenAiProvider::new(
        "unused".into(),
        "http://localhost:11434/v1".into(),
        "qwen3.6:35b-a3b".into(),
    )?;

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(CalculatorTool))?;

    let mut agent = Agent::new(Box::new(provider), registry, "You are a helpful assistant.");
    agent.run_turn("What is 17 * 43?", &mut |event| match event {
        AgentEvent::AssistantTextDelta(text) => print!("{text}"), // streamed
        AgentEvent::AssistantText(text) => println!("{text}"),    // non-streaming fallback
        _ => {}
    })
    .await?;
    Ok(())
}
```

Custom tools implement the `Tool` trait: name, description, JSON schema, async
execute. Put the trigger condition in the description — in code mode that text is
included in the `python` tool's bridge-function documentation. Permission policy is
pluggable via `ToolPermissionHandler`:
`AlwaysAllow`, `AlwaysDeny`, name-based `PolicyPermissions`, or the interactive
`MemoryPermissionHandler` the CLI uses.

## Skills and project notes

Drop instruction folders in `~/.generalist/skills/<name>/SKILL.md` (optional
`name:`/`description:` frontmatter). Only a one-line index enters the system prompt;
the agent reads the full file when a task matches. A `./AGENTS.md` or `./CLAUDE.md`
in the working directory is appended to the system prompt at startup.

## Limits

- The OpenAI provider sends no `max_tokens`, for compatibility across servers.
- Compaction uses a chars/4 token estimate between provider measurements; treat
  thresholds as approximate.
- Code-mode scripts run unsandboxed (see Permissions).

## License

MIT.
