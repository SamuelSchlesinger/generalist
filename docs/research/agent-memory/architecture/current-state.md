# Current-State Gap Analysis

> **Implementation update (2026-07-27):** this file preserves the historical
> audit at commit `db900fa`. The model-facing `EnhancedMemoryTool` described
> below has since been removed. The current checkout implements an opt-in,
> host-owned, scope-local settled-turn SQLite prototype, project-scoped
> conversation storage, and permission-gated read-only cross-scope archive
> tools with no automatic retrieval. See the repository
> `README.md`, `docs/next-agent-handoff.md`, and
> `docs/runtime-traceability.md` for current behavior.

## Scope and evidence

This audit covers Generalist commit `db900fa` (“Fix TUI observability and
terminal interaction”). The relevant implementation is
`src/tools/enhanced_memory.rs`, with registration in `src/main.rs`, model
guidance in `SYSTEM_PROMPT.md`, conversation ownership in `src/agent/`,
durable session state in `src/state.rs`, and the async protocol documented in
`docs/async-tui.md` and `spec/AsyncRuntime.tla`.

## What exists

`EnhancedMemoryTool` is a script-callable CRUD service over one JSON file,
`~/.generalist_memory.json`. An entry contains:

- a UUID;
- free-form content;
- free-form tags;
- creation and update timestamps;
- string-to-string metadata.

The model chooses when to call `store`, `search`, `update`, `delete`, or
`list_tags`. Search is case-insensitive substring matching over content, tags,
and metadata, optionally filtered by any matching tag, then sorted by most
recent update. Ten entries are returned by default. The in-process store is a
`tokio::RwLock`; each mutation serializes the complete map and writes it with
`fs::write`.

The system prompt asks the model to store “durable facts worth remembering” and
to check memory when history might help. In built-in code mode the model must
call this service indirectly from a `python` script. The call passes through
the same per-tool permission mechanism as every other bridged capability.

## What it is not

Despite its name, this is not an agent memory runtime:

- It does not record episodes or link a memory to the conversation, turn,
  observation, action, tool result, environment, user, or project that produced
  it.
- It has no automatic write policy, post-turn extraction, retrieval injection,
  prospective reminder, or consolidation schedule. Recall succeeds only if the
  model remembers to search and prints the right result back out of code mode.
- It has no separation between observed events, user statements, model
  inferences, generated counterfactuals, semantic facts, preferences,
  procedures, or summaries.
- It has no provenance, evidence pointer, confidence, temporal validity,
  supersession, contradiction set, verification state, access scope, sensitivity
  label, or use history.
- It cannot express derived knowledge or trace a derived item back to episodes.
- It has no reflection/replay/consolidation process and no notion of a candidate
  awaiting promotion. A model assertion becomes durable immediately after one
  approved tool call.
- It has no decay, compaction, duplicate detection, merge policy, contradiction
  repair, quarantine, or rollback beyond deleting an entry by UUID.
- It has no first-class TUI for inspection, correction, export, or deletion.
- It has no retrieval-quality, longitudinal-task, poisoning, privacy, or
  consolidation tests.

It therefore provides persistent notes, not episodic memory or learning.

## Safety and durability gaps

### Durable injection and epistemic collapse

Content is an undifferentiated string authored by the model. Retrieved strings
can contain instructions, untrusted web content, guesses, or stale facts with
no machine-readable distinction. There is no policy that prevents a single
compromised episode from becoming an instruction-like memory, and no
independent evidence check before later use.

### Privacy and scope

All memories share one home-directory file. There is no project/user/session
namespace, capability scope, secret classification, retention policy, or
selective export. File permissions are inherited from the process umask rather
than set deliberately.

### Crash consistency and corruption

The store rewrites the target directly rather than flushing a temporary file,
atomically renaming it, and syncing its directory as session autosave does.
Interruption can leave malformed JSON, and one parse error prevents the tool
from being constructed at startup. There is no schema version, migration
journal, backup, checksum, or recovery path.

### Index integrity and quality

Tags are not normalized or deduplicated. Empty tag-index buckets survive
updates/deletes. Retrieval is an `O(n)` scan and recency sort with no lexical
ranking, semantic ranking, diversity, temporal filtering, or context-budget
control. Metadata updates replace the entire map. None of these behaviors has
unit coverage.

## Runtime constraints a replacement must respect

1. `Agent` is the sole owner of committed conversation history during a turn.
   Memory capture may observe history-valid checkpoints but must not insert
   blocks between a tool use and its result.
2. The current-thread UI reactor must remain responsive. Retrieval,
   consolidation, indexing, and storage I/O need explicit async ownership and
   cancellation behavior; they cannot hide a blocking second runtime.
3. Code mode advertises only `python` as a capability tool. The separate
   `update_goal` completion control is host-owned, reserved, and
   permission-free. A host-native memory lifecycle should not require
   additional model-facing tool calls merely to remember or recall, but
   user-visible memory operations may still be exposed through the generated
   `tools` module.
4. Durable queue/history autosave is atomic and has a documented TLA+
   refinement. Memory transactions must define whether they are coupled to a
   history checkpoint or independently recoverable.
5. Local commands run only while the agent is idle. Explicit memory
   inspect/edit/delete/consolidate commands can reuse this ownership guard.
6. Provider text and exposed reasoning are untrusted payloads. F4 observability
   does not make reasoning faithful evidence, and reasoning must not silently
   become memory.
7. The project supports Unix only, so SQLite, file locking, restrictive modes,
   Unix sockets, and background child processes are available design options;
   cross-platform abstractions are unnecessary.

## Integration points worth evaluating

- `AgentEvent::HistoryCheckpoint`: a complete, protocol-valid episode boundary
  with a cloned history and context estimate.
- `TurnOutcome`: distinguishes completion, denial, interruption, refusal, and
  iteration cap for episode outcome metadata.
- tool start/finish events and the registry execution log: possible structured
  action/outcome evidence without parsing assistant prose.
- the authoritative prompt queue and active goal: prospective context that must
  not be confused with completed experience.
- the idle outer controller: a safe point for bounded consolidation jobs and
  explicit user commands.
- the TUI event stream: status, review queues, conflicts, and memory-management
  views without a second terminal reader.
- compaction: currently a destructive summary of older conversation messages,
  not a memory promotion mechanism. The two lifecycles should coordinate but
  remain distinguishable.

## Preliminary non-decisions

The audit does not justify choosing embeddings, a graph database, autonomous
reflection, generated dream episodes, or weight updates. It establishes only
that the current tool lacks the information and lifecycle controls needed to
evaluate any of them safely.

## Multi-agent readiness

Generalist is reusable as one worker but has no collaboration layer. The CLI
constructs one `Agent`; `runtime.rs` intentionally uses single-threaded
`Rc<RefCell<_>>` queue ownership; prompt IDs and claims are process-local; and
one autosave contains the complete conversation, queue, goal, and permission
state. There is no durable agent/task identity, mailbox, dependency graph,
lease/fencing protocol, capability delegation, peer control, shared-memory
scope, worktree isolation, or artifact integration.

Launching multiple copies would therefore provide accidental filesystem
parallelism, not coordinated agents. See the
[multi-agent architecture](multi-agent.md) for the proposed separation between
SQLite-backed control/message/memory planes and isolated source-tree mutation.
