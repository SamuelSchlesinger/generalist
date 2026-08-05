# Next-Agent Handoff

## Current stop point

Generalist now contains and has evaluated the smallest explicit
episodic-memory prototype. It is not the full
memory/consolidation/collaboration architecture described in the research
corpus.

The previously completed product baseline remains:

- asynchronous Ratatui interaction, stable queued steering/follow-ups, exact
  wrapped scrolling, clipboard/native selection, live conversation search,
  catalog-backed slash completion, reasoning inspection, and durable `/goal`
  autorun through the host-owned `update_goal` completion control;
- deterministic progressive MCP startup through the sole TUI reactor, with a
  durable pre-agent queue, live F2 editing, explicit skip, no false steering
  during background work, typed `/mcp status`, and targeted/all failed-server
  retry into the live registry;
- host-owned `/permissions` inspection/reset/clear, deterministic policy
  display, and deny-first normalization of contradictory remembered state;
- host-owned `/tools` list/search/show over the finalized bridge catalog, with
  stable bounded output and no provider, permission, execution, or history
  effect;
- host-owned process-local `/usage show|reset`, with per-provider/model attempt
  counts, explicit missing-report/cache-field coverage, and no cost claim or
  provider/durable-state effect;
- host-owned `/history` list/search/show/forget over sanitized current-scope
  saves, with confirmation-gated durable deletion, direct `/save <name>` and
  `/load <name>`, race-safe no-clobber creation, confirmed replacement,
  autosave protection, and no provider request, tool permission, or cross-scope
  read;
- no fixed ordinary-completion token ceiling: Anthropic resolves and caches
  the selected model's advertised maximum, compatible endpoints own their
  default, and `--max-tokens` is an explicit override. Independent logical
  byte/block/tool/wire limits reject oversized or malformed completions before
  history or execution; committed chat/reasoning rendering is capped at 64 Ki
  characters with full `/copy last|all|reasoning` access;
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

## Evaluation result and decision

`benchmarks/episodic_memory` now makes the prior gate reproducible. The
authoritative local run is
`results/20260805T014941Z-local-explicit-memory.jsonl`; it records exact binary,
evaluator, corpus, Git, platform, Python, and SQLite provenance. One earlier
JSONL intentionally preserves a probe-classification failure, and the next run
shows the corrected check; neither was overwritten.

- B0 retained zero episodes. B1 and deliberately named saves both achieved
  recall `1.0`, mean reciprocal rank `1.0`, and precision `0.875`; ordinary
  latest autosave achieved recall `1/7`. The one false positive was the older
  `PREF-CARBON` preference returned after the newer preference, as expected
  from literal chronological search.
- On this machine, 200 explicit B1 searches measured p50 `0.097 ms`, p95
  `0.191 ms`; the serialized eight-episode logical export was 4,892 bytes.
  SQLite allocation is reported separately because WAL/checkpoint timing makes
  live file size non-monotonic.
- Current-scope search excluded a deliberate same-token record in another
  project. Delete disappeared from live and restarted search while the prior
  in-memory export still retained it, matching the documented boundary.
- The worker timed out under `BEGIN IMMEDIATE` after about 2.18 seconds while
  the current-thread executor advanced 218 ten-millisecond ticks and the
  setting remained unchanged.
- Eight abrupt post-enqueue exits produced two absent and six complete rows;
  three durable acknowledgements produced three complete rows; three children
  killed behind a writer lock produced three absent rows. No duplicate or
  partial row appeared, the immutable update failed, and `integrity_check` was
  `ok`.
- The exact TUI kept input visible in 80 ms during a stalled provider, 59 ms
  during the SQLite stall, and 73 ms after 20,000 of 30,000 one-byte provider
  deltas. The flood's committed tail rendered and the turn settled normally. A
  real paused-capture turn was answered but not retained; queued B1 turns
  dispatched later, explicit search found them, and
  normal exit preserved exactly the expected rows. `/history search` found the
  B0 fact in ordinary autosave and `/history show autosave` inspected it with
  zero provider requests and no autosave-content change. Direct named save/load
  worked; default-Cancel preserved an existing checkpoint byte-for-byte,
  explicit replacement changed it, and manual `autosave` was refused.
  Default-Cancel then retained the checkpoint, confirmed deletion removed it,
  and deleting `autosave` was refused; all lifecycle commands still made zero
  provider requests. `/usage` then reported the fixture's four attempts and
  four input/four output tokens with zero-of-four cache-field coverage;
  `/usage reset` produced an empty ledger. Both commands made zero provider
  requests and preserved autosave bytes. A fail-once stdio MCP server also
  moved from visible startup failure through `/mcp retry flaky` to connected
  status and a live `flaky_ping` bridge with zero provider calls; connected
  and unknown retry targets were refused.

The measured value is convenient automatic capture across otherwise replaced
autosaves, not better retrieval than disciplined named saves and not temporal
truth resolution. Retain explicit Stage 1 capture/search/show/export/forget.
Do not proceed to automatic retrieval, promotion, or consolidation on this
evidence.

## Exact next action

Token fragments and maximum committed chat/reasoning snapshots now have bounded
display projections without making the UI a history authority. Provider output
length policy is separate from host safety, including sticky callback failures
and final validation for custom providers. Keep the million-fragment proof, the
30,000-fragment PTY probe, and the new oversized-response/tool-burst/projection
tests. Ordered structural events and checkpoints remain lossless. The next
reliability pass should adversarially measure structural-event volume and decide
whether backpressure can preserve every tool, permission, and history-boundary
event without creating a second authority. Keep the saved-session lifecycle and
Stage 1 memory stable. Any automatic-memory proposal still requires a separately
reviewed, paired real-model longitudinal study.

## Validation

Run:

```sh
make memory-research
make memory-evaluation
make conformance
make check
git diff --check
```

The established `AsyncRuntime.tla` baseline is 470,086 generated states,
117,750 distinct states, and depth 27. The current-handle `MemoryRuntime.tla`
configuration generates 7,627 states (1,690 distinct, depth 16), and
`ArchiveScopeRuntime.tla` generates 519,516 states (59,725 distinct, depth 13).
The exact Rust test count can change with dependency and target updates; rely on
the current `make check` output rather than this handoff. All three base models,
the three observed implementation traces, and the three deliberately invalid
traces must be repeated after any relevant change.
