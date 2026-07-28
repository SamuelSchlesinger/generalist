# generalist

A provider-agnostic terminal agent in Rust with a full-screen Ratatui interface. Works
with the Anthropic Messages API, OpenRouter, or any OpenAI-compatible endpoint
(OpenAI, Ollama, Groq, Mistral, vLLM, LM Studio). The library is small: neutral
conversation types, a `Provider` trait, a tool registry with permission gating, and
an agent loop that reports progress through event callbacks.

## Install and run

```bash
cargo build --release

# Keys go in the environment or ~/.generalist.env:
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> ~/.generalist.env
echo 'OPENAI_API_KEY=sk-...'        >> ~/.generalist.env   # and/or
echo 'OPENROUTER_API_KEY=sk-or-...' >> ~/.generalist.env   # and/or
echo 'FIRECRAWL_API_KEY=fc-...'     >> ~/.generalist.env   # optional, web tools

./target/release/generalist

# Local models need no key:
./target/release/generalist --local                    # qwen3.6:35b-a3b
./target/release/generalist --local qwen2.5-coder:32b  # or name one
```

When `OPENROUTER_API_KEY` is configured, normal remote startup defaults to
OpenRouter's `moonshotai/kimi-k3`. Use `/model` to switch to another configured
API. `--local` always takes precedence and keeps the local-model behavior.

`--local [model]` skips provider selection and uses `http://localhost:11434/v1`; set
`OPENAI_BASE_URL` for other local servers. Tool calling requires a tool-capable model
(`qwen3.6`, `qwen3`, `qwen2.5-coder`, `devstral`).

Optional binaries: `z3` (constraint solver), `patch` (file editing; present on any Unix).

Smoke tests, all live end-to-end: `cargo run --example smoke` (Anthropic),
`smoke_openrouter` (Kimi K3), `smoke_ollama` (local tool loop), and
`smoke_codemode` (local code-mode bridge).

## Usage

Type a request; the agent calls tools and reports back. Generalist targets Unix-like
systems; code mode is on by default and `python` is the only model-facing
capability tool. Scripts reach all registered capabilities through `import tools`: bash, file
read/patch, directory listing, HTTP
fetch, web search/scrape/crawl (Firecrawl), Wikipedia, weather, Z3, and todo
list. Permission-gated archive tools can explicitly search/read sanitized
episodic memories and saved conversations in selected scopes. (Calculator,
system-info, think, and the former model-controlled memory *write* tool were
retired from the CLI: python and bash subsume the first three, while episodic
capture remains host-owned.)

Responses stream as they generate. Type `/` to enter visible command mode; the
footer lists the available slash commands from the same catalog used by the
parser and help window. `/goal <objective>` sets durable instruction context,
starts working, and automatically prompts another turn after each normal answer
until the model calls the host-owned `update_goal(status="complete")` control.
`/goal edit` opens the editor and resumes the loop, `/goal show` displays the
objective, and `/goal clear` stops and removes it. Escape, a provider error,
refusal, or permission denial pauses automatic continuation without discarding
the objective; edit the goal or send another prompt to resume. Other commands
are `/save`, `/load`, `/model`, `/compact`, `/clear`, `/memory`, `/help`, and
`/exit`.

History-valid boundaries, the active goal, queue, named sessions, and remembered
tool decisions are isolated by default to the canonical Git worktree root (or
canonical working directory outside Git). Scoped state lives beneath
`~/.generalist/history/scopes/<scope-id>/`; the project autosave never falls
back to another project or to global history. The goal survives restart even
with no queued work, and startup schedules its next automatic continuation. If
the process exits with queued work, the next run also recovers that queue
together with its conversation context. Current save files must carry their
scope explicitly; unscoped state is rejected rather than treated as global.

Run `generalist --global` to select the explicit cross-project namespace.
Pre-scoping flat history and the legacy `~/.chatbot_history` and
`~/.generalist_history` paths are left untouched but are not read by the new
stores; there is no implicit migration or compatibility fallback. Set
`GENERALIST_HOME` to an alternate directory for an isolated profile or
reproducible harness run; all of the paths above, plus `.generalist.env`, MCP
configuration, and skills, are then resolved beneath that directory.

## Terminal UI

The Ratatui dashboard keeps conversation, live model status, context usage, the
prompt queue, and recent tool activity visible at once. A single current-thread
Tokio reactor polls the active model/tool future, terminal events, permission
requests, and frame ticks together. You can keep editing and scrolling while a
response is in flight. The header keeps the active goal visible and says
`code mode / N bridges`: `python` remains the sole model-facing capability
tool, while nested bridge activity is shown as `↳ tools.<name>`. While a goal
is active, the separate permission-free `update_goal` host control is also
advertised.

Keyboard and mouse controls:

- While idle, `Enter` starts a turn. While busy, `Enter` queues a steer for the
  next history-valid boundary.
- `Tab` or `Alt+Enter` queues a separate follow-up. `Shift+Enter` or `Ctrl+J`
  inserts a newline.
- `Up`/`Down` browse input history. `Ctrl+A`/`Ctrl+E`, `Ctrl+U`, and `Ctrl+W`
  provide familiar shell-style editing.
- `PageUp`/`PageDown` or the mouse wheel scroll the conversation. A paused
  viewport stays anchored while new text streams; scrolling to the bottom
  resumes follow-latest.
- `F2` opens the queue manager: edit, delete, change steer/follow-up mode,
  reorder, or restore a queued message. The mouse wheel moves long queue
  selections. `Alt+Up` restores the latest queued message directly to an empty
  composer; restore never overwrites an unsent draft.
- `F3` enters native terminal copy mode. Mouse capture and redraws pause so you
  can select text and use the terminal's normal copy shortcut. Provider/tool
  work continues in memory; press `F3` again to resume and redraw the latest
  state. Normal terminal paste works in the composer after copy mode is closed.
- `F4` opens the live model-reasoning inspector. It shows only inspectable
  reasoning fields actually supplied by the provider, separately from answer
  text, and says so explicitly when a request supplies none. Provider
  signatures and redacted payloads are never displayed.
- `F1` opens help. With no modal, `Esc`/`Ctrl+C` interrupts a busy turn safely;
  while idle, `Esc` clears the editor and `Ctrl+C` exits. A permission modal
  consumes its own keys first.

The exact async semantics, TLA+ models, and maintained model-to-Rust review are
documented in [the architecture note](docs/async-tui.md) and
[runtime traceability matrix](docs/runtime-traceability.md).
`make conformance` additionally runs deterministic paths through the real Rust
queue, agent, permission, history, and memory code, then asks TLC to consume
those exact abstract action sequences through the original model operators. It
also verifies that three deliberately invalid traces are rejected. This is a
sampled executable refinement check, not a proof over every Rust execution.

## Memory architecture status

The model-controlled `enhanced_memory` bridge has been removed completely. The
old `~/.generalist_memory.json` and `~/.claude_memory.json` files are neither
read nor modified; delete or archive them manually after inspecting any data
you want to keep.

Generalist now has a deliberately small host-owned episodic prototype:

- capture is paused by default and enabled independently for the active project
  or explicit global scope with `/memory resume`;
- a dedicated worker per Generalist process owns its SQLite connection at
  `~/.generalist/memory/scoped-episodes.sqlite3`, keeping database work off the
  TUI reactor. The pre-scoping `episodes.sqlite3` is left untouched and ignored;
- only settled user/assistant text, tool names, and tool success/error metadata
  are retained; tool inputs/results, provider reasoning, signatures, and
  redacted payloads are structurally omitted. Host-authored goal-continuation
  text is also omitted rather than misclassified as user-authored. If in-turn context compaction
  moves the exact turn boundary, the record safely degrades to a
  `prompt_only` episode rather than retaining a generated compaction summary.
  In code mode, this smallest slice records the outer `python` call, not each
  nested bridge call;
- `/memory status`, `pause`, `resume`, `search <query>`, `show <id>`, `export`,
  and `forget <id>` are explicit local commands; and
- `search_memories`/`read_memory` and
  `search_conversations`/`read_conversation` are read-only model capabilities.
  Each call requires the ordinary tool permission flow and an explicit
  `current`, `global`, `other_projects`, or `all` scope. Results omit provider
  reasoning and tool payloads and are labeled as untrusted historical context.
  Reads repeat the search filter and exact returned scope label; long
  transcripts use bounded pages with `next_offset`. After permission, the
  registry mints an exact-input disclosure grant that the cross-scope storage
  API must recheck; direct archive-tool execution without that grant is
  rejected. No episode or conversation is automatically retrieved or injected.

`/memory forget` removes the current-scope row from the live SQLite store and
attempts a truncating WAL checkpoint, reporting separately if truncation
remains pending. It does not claim erasure from prior exports, backups, or
filesystem snapshots. User and assistant text can itself contain secrets, so
pause capture before sensitive work. The worker rejects symlinked
database/directory targets. There is no automatic expiry or storage quota in
this experiment; inspect and delete retained rows explicitly.
The `0700` directory and `0600` database protect against other Unix users, not
same-UID tools; this prototype is not a sandbox or a multi-agent security
boundary. In particular, code mode's unsandboxed Python process can access the
database as an ordinary local file if directed to its path. Permission-gated
archive reads are not a same-UID sandbox and do not weaken this caveat.
Simultaneous Generalist processes share settings/rows through SQLite locking,
but have no cross-process queue ordering or collaboration protocol.

The source-grounded design, adversarial safety review, multi-agent analysis,
and possible future lifecycle are checked in under
[the agent-memory research corpus](docs/research/agent-memory/index.md). The
current prototype intentionally implements much less than that design. Its
actual scoped FIFO capture/deletion and permissioned-disclosure semantics are modeled in
[`MemoryRuntime.tla`](spec/MemoryRuntime.tla), while shared conversation/memory
scope routing is modeled in
[`ArchiveScopeRuntime.tla`](spec/ArchiveScopeRuntime.tla); product value must be measured
before adding automatic retrieval, candidate promotion, consolidation,
simulation, or collaboration.

## Permissions

New tool calls open a permission modal showing the full input (patches are rendered as
colored diffs). Choices are allow always, allow once, deny always, and deny once.
Decisions persist across save/load; remembered decisions are surfaced in the status
bar while every execution remains visible in the tool-activity panel.

All model, tool, queue, provider, and MCP text is control-character-sanitized at
the display boundary so untrusted content cannot emit terminal escape commands.
The raw text retained in conversation history and passed to tools is unchanged.

Caveats:

- "Always allow" is per tool name. Always-allowing `bash` approves every future
  command; each command is still shown in tool activity before it runs.
- The same rule applies to archive tools: always-allowing
  `search_conversations` or `search_memories` covers future inputs including
  broader scope selectors. Use allow-once when that breadth is not intended.
  Each allowed invocation still receives a new capability bound to that exact
  tool name, complete input, and scope; allow-always does not grant a reusable
  unscoped storage handle.
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
  visibly marked uncommitted and retried rather than treated as a complete
  answer or persisted as assistant history.
- Provider-supplied reasoning streams independently from answer text into the
  `F4` inspector. Anthropic thinking deltas and common OpenAI-compatible string
  extensions (`reasoning_content`, `reasoning`, or `thinking`) are supported.
  Endpoints that expose no such field produce no invented reasoning.
- When context passes a threshold (default 150k tokens, configurable), older
  history is summarized in place and recent turns stay verbatim. `/compact`
  triggers it manually. Local models with small context windows may want a much
  lower `compaction_threshold_tokens`.

## Code mode

The agent advertises exactly one capability tool, `python`, when code mode is
enabled (the default). Every registered tool is available only to scripts
through a generated `tools` module. An active goal additionally advertises the
native `update_goal` host control; it is not a registered capability, is never
available through the Python bridge, and does not ask for execution permission.

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

Some OpenAI-compatible models return a bridge expression such as
`tools.firecrawl_search` as an undeclared native call despite receiving only the
`python` capability schema. Generalist treats that as a provider-protocol violation: it records a
paired error so history stays valid, but does not request permission or execute the
named tool.

Library users can opt back into independently advertised direct tools with
`agent.code_mode = false`. Registering a custom tool named `python` also overrides the
built-in runner. The name `update_goal` is reserved for the host controller and
cannot be registered as a custom capability.

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

## Development

Run `make setup` once to install the pinned TLA+ tools and checked-in Git hooks,
then `make check` before contributing. The full methodology—including the
required TUI-to-TLA+ trace review—is in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT.
