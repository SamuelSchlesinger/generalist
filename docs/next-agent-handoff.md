# Next-Agent Handoff

## Current stop point

Generalist now contains the smallest explicit episodic-memory prototype. It is
not the full memory/consolidation/collaboration architecture described in the
research corpus.

The previously completed product baseline remains:

- asynchronous Ratatui interaction, stable queued steering/follow-ups, exact
  wrapped scrolling, copy mode, reasoning inspection, and durable `/goal`
  autorun through the host-owned `update_goal` completion control;
- code mode as the only model-facing capability boundary, with native host
  controls kept separate; and
- OpenRouter `moonshotai/kimi-k3` as the default remote model when
  `OPENROUTER_API_KEY` exists.

The OpenRouter key in `~/.generalist.env` was detected without printing it. The
last live Kimi smoke request reached OpenRouter but returned HTTP 402
“Insufficient credits.” Do not copy the key into source, logs, tests, or this
document.

## What the episodic prototype implements

- The `enhanced_memory` tool, registration, module, and system-prompt
  instruction are gone. Existing `~/.generalist_memory.json` and
  `~/.claude_memory.json` files are left untouched but are never read.
- Conversation history, goals, queues, named saves, and remembered permissions
  are isolated beneath a deterministic scope directory. Normal startup selects
  the canonical current Git project/worktree (or canonical non-Git working
  directory); `--global` is an explicit scope and never a fallback. Legacy flat
  conversations are left untouched and ignored.
- Capture is paused by default. `/memory resume` opts in persistently for the
  active project or explicit global scope; `/memory pause` opts out.
- One named blocking worker thread per Generalist process owns a bundled
  SQLite 3.51.3 connection at
  `~/.generalist/memory/scoped-episodes.sqlite3`. The containing directory is
  `0700` and database file is `0600`; the old `episodes.sqlite3` is ignored.
- A protocol-valid settled turn queues one immutable episode without awaiting
  SQLite. The FIFO is flushed before normal process exit.
- Episodes retain user/assistant text, provider/model/outcome, tool names, and
  tool success/error metadata. Tool inputs/results, provider reasoning,
  signatures, and redacted-reasoning payloads are structurally omitted. An
  automatic goal-continuation prompt is also omitted rather than retained as
  user-authored text. An in-turn compaction that moves the exact boundary
  degrades capture to the
  original prompt with `capture_quality = prompt_only`. Because capture is
  derived from committed history, code-mode metadata names the outer `python`
  call rather than each nested bridge call.
- `/memory status|pause|resume|search <query>|show <id>|export|forget <id>` are
  typed idle commands. While a command awaits SQLite, the TUI still polls
  terminal input, queue mutations, memory events, stale permissions, and frame
  ticks.
- Local operations are bound to the worker's current byte-valued scope key.
  `search_memories`/`read_memory` and
  `search_conversations`/`read_conversation` are read-only bridged
  capabilities under the ordinary permission gate. They require an explicit
  current/global/other/all scope, filter before opening conversation files or
  matching memory content/IDs, check returned scope labels, and expose only
  sanitized historical text/tool metadata in bounded, resumable pages.
- `MemoryRuntime.tla` models one immutable current-scope handle, FIFO settlement, complete
  insert/skip/failure, immutable live rows, live-store deletion, and
  permissioned explicit disclosure. `ArchiveScopeRuntime.tla` separately
  models project/global selection, same-scope writes, and filtered archive
  reads. Neither permits automatic retrieval.
- Cross-scope history and memory APIs require an exact-input `DisclosureGrant`
  that only `ToolRegistry` can mint after the permission policy allows the
  matching tool call. Direct archive-tool execution is rejected.
- `make conformance` generates payload-free traces through the real queue,
  agent, registry, history, and memory implementations. TLC must consume each
  observed sequence through the original model actions and reject three
  deliberate ordering/lifecycle/scope mutations. This is sampled safety
  conformance, not exhaustive Rust refinement.

`/memory forget` deletes from the live SQLite store with `secure_delete=ON`,
then attempts a truncating WAL checkpoint and reports if truncation remains
pending. It deliberately does not claim erasure from prior exports, backups,
or filesystem snapshots.

## What remains deliberately unimplemented

- automatic memory retrieval or prompt injection;
- model-authored durable memory writes (generic same-UID Python remains
  unsandboxed and can access local files);
- trusted secret redaction or capture admission beyond structural payload
  omission;
- FTS/embeddings, ranking, summaries, candidates, approval, lineage, or
  consolidation;
- retention quotas or automatic expiry;
- external tombstone ledger, backup non-resurrection, or encryption hierarchy;
- a process-isolated supervisor or protection from same-UID tools;
- simulations, predictions, procedures, dreaming, or weight updates;
- durable multi-agent tasks/messages/delegation and
  `CollaborationRuntime.tla`.

Concurrent processes receive SQLite transaction/locking semantics only. They
do not share an in-memory FIFO, so cross-process capture/pause ordering is
outside `MemoryRuntime.tla` and must not be presented as agent collaboration.

The reviewed future design remains under
[agent-memory](research/agent-memory/index.md), but it is an options and safety
corpus rather than an implementation backlog.

## Exact next action

Evaluate the prototype before expanding it:

1. run real multi-session Generalist workflows with capture paused (B0) and
   explicit episodic search enabled (B1);
2. measure whether search recovers useful project facts that saved
   conversation history does not, along with latency, retained volume, false
   matches, and deletion behavior;
3. exercise `/memory` during a stalled provider and under a deliberately held
   SQLite write lock, confirming typing and queue management remain live;
4. inject process exit around episode enqueue/insert and confirm each attempt
   is absent or one complete immutable row; and
5. stop at explicit search unless measured value justifies a separately
   reviewed next milestone.

Do not proceed directly to automatic retrieval or consolidation. If the
prototype does not outperform ordinary saved-history workflows, retain only
the useful inspection/export pieces or remove it.

## Validation

Run:

```sh
make memory-research
make conformance
make check
git diff --check
```

The established `AsyncRuntime.tla` baseline is 470,086 generated states,
117,750 distinct states, and depth 27. The current-handle `MemoryRuntime.tla`
configuration generates 7,627 states (1,690 distinct, depth 16), and
`ArchiveScopeRuntime.tla` generates 163,652 states (20,341 distinct, depth 11).
The exact Rust test count can change with dependency and target updates; rely on
the current `make check` output rather than this handoff. All three base models,
the three observed implementation traces, and the three deliberately invalid
traces must be repeated after any relevant change.
