# Coding-Agent Memory: pi, Claude Code, and OpenAI Codex

Coding agents expose three easily confused forms of persistence:

1. a saved transcript can resume a session;
2. a compacted summary can keep a long session inside the context window; and
3. a curated cross-session artifact can change how future sessions begin.

Only the third is durable learned memory in the narrow sense used here.
Transcript persistence and compaction are still valuable, but they solve
recovery and context pressure rather than automatic cross-session learning.
The contracts below are verified against official sources on 2026-07-26.

| Agent | Source record | Active-context projection | Cross-session learned artifact | Automatic consolidation |
| --- | --- | --- | --- | --- |
| pi | append-only JSONL session tree | LLM compaction summary plus recent entries; branch summary when changing paths | none in core; `AGENTS.md`/`CLAUDE.md` and skills are authored context | within-session compaction only |
| Claude Code | conversation state plus local project files | `/compact`-managed context | auto-memory `MEMORY.md` and topic files; human-authored `CLAUDE.md` | Claude decides which notes are worth saving |
| OpenAI Codex | immutable rollout records and staged DB records | ordinary session context management | task-grouped `MEMORY.md`, compact `memory_summary.md`, rollout summaries, optional skills | two-phase startup extraction and global consolidation when enabled |

## pi: an append-only event tree, not a learned-memory system

### Session data model

pi stores sessions as JSONL. Every entry has an `id` and `parentId`, so one
file represents a tree rather than a single overwritten transcript. Sessions
auto-save under `~/.pi/agent/sessions/`, organized by working directory.
`/tree` moves to an earlier point and continues on a new branch without
discarding the old branch. `/fork` and `/clone` create new session files from
selected active paths.[pi-readme][pi-readme]

This append-only shape is excellent provenance for what the agent saw, said,
and did. It supports audit, branch recovery, and alternative continuations.
There is no semantic ranking across past sessions and no core process that
extracts stable preferences or procedures from them.

### Compaction

Compaction triggers manually through `/compact` or automatically when context
usage crosses the configured threshold:

\[
\text{contextTokens} >
\text{contextWindow} - \text{reserveTokens}.
\]

The implementation walks backward to retain a recent-token budget, summarizes
the older span with an LLM, appends a `CompactionEntry`, and reloads the active
context as that summary plus the recent entries. The entry records its ID,
parent, timestamp, summary, first retained entry, tokens before compaction,
generation usage, and details including read and modified files. Repeated
compactions supply the previous summary as input.[pi-compact][pi-compact]

When `/tree` leaves a branch, pi can summarize the abandoned path from its
common ancestor and append a `BranchSummaryEntry`. Both mechanisms use a
structured summary of goal, constraints, progress, decisions, next steps,
critical context, and cumulative file operations.[pi-compact][pi-compact]

The critical interpretation is stated by pi’s own documentation: **compaction
is lossy while the full JSONL history remains**.[pi-readme][pi-readme] The
summary is the model’s active-context projection, not a replacement source of
truth. It may omit a qualification or error even though an auditor can recover
the source from the session tree.

### Authored context and learning verdict

At startup, pi concatenates global, ancestor-directory, and current-directory
`AGENTS.md` or `CLAUDE.md` files. Skills are separate on-demand packages.
These files can carry procedural knowledge across sessions, but a person,
repository, or extension authors them; core pi does not automatically learn
them from completed sessions.[pi-readme][pi-readme]

Therefore pi provides durable session memory and within-session compression,
not built-in durable learned memory. Extensions could add one, but that would
be an extension-specific contract.

## Claude Code: authored instructions plus auto memory

### Two distinct stores

Claude Code’s official documentation distinguishes:

- **`CLAUDE.md` instructions**, written by users at organization, user,
  project, local, and path-specific scopes; and
- **auto memory**, notes Claude writes based on corrections, preferences,
  debugging findings, architecture, commands, and workflow patterns.

Both enter future prompts as context, not enforced configuration. The
documentation explicitly recommends hooks for behavior that must execute and
warns that conflicting instruction files may be resolved arbitrarily by the
model.[claude-memory][claude-memory]

### Auto-memory data model and trigger

Auto memory is on by default. Each Git repository gets a machine-local
directory under `~/.claude/projects/<project>/memory/`, shared by its worktrees.
`MEMORY.md` is a concise index; Claude may create topical Markdown files for
detail. Claude decides what is useful enough to write and does not necessarily
save something every session.[claude-memory][claude-memory]

The first 200 lines or 25 KB of `MEMORY.md`, whichever comes first, load at the
start of each conversation. Topic files load on demand through ordinary file
tools. When the index approaches its limit, Claude is instructed to merge or
drop stale entries and move details to topic files. Newer versions can attach a
`modified` timestamp in YAML frontmatter when rewriting a memory file.

There is no documented vector ranker, explicit episodic store, source-message
pointer, confidence score, or truth-maintenance algorithm. The division is
primarily procedural (`CLAUDE.md`) versus learned notes (auto memory), with
semantics determined by free-form Markdown.

### Correction, deletion, and compaction

Users can inspect, edit, or delete auto-memory files and toggle the feature
through `/memory` or settings.[claude-memory][claude-memory] This is a clear
human correction surface. The documentation does not claim deletion from
filesystem history, backups, telemetry, or derived copies, so plain-file
deletion should not be inflated into a general erasure guarantee.

Claude Code also compacts long conversational context. The troubleshooting
guide treats lost post-compaction instructions separately from auto memory.
Like pi’s compaction, this is active-context maintenance; it is not evidence
that a durable lesson was extracted.

### Learning verdict

Claude Code genuinely rewrites an external cross-session note base
automatically. It does not update base-model weights. Its advantages are
inspectability, simple scoping, and user-editability. Its weaknesses are
model-chosen writes, missing source lineage, free-form conflicts, startup
truncation, machine-local drift, and the possibility that poisoned repository
content is promoted into a trusted-looking note.

## OpenAI Codex: staged rollout extraction and global consolidation

### Current status

The open-source Codex feature registry marks startup memory extraction and
file-backed consolidation as `Stage::Stable` but disabled by default as of
2026-07-26.[codex-features][codex-features] Individual app-server controls such
as `memory/reset` and per-thread memory mode are documented as experimental.
Those two status labels refer to different surfaces and should both be
preserved.[codex-server][codex-server]

The pipeline runs asynchronously when a root session starts only if the session
is non-ephemeral, the feature is enabled, the session is not a subagent, and a
state database is available.[codex-readme][codex-readme]

### Phase 1: per-rollout normalization

Phase 1 finds a bounded set of recent, idle, eligible interactive rollouts. It
leases each job in the state DB, extracts memory-relevant response items, and
runs model calls in parallel under a concurrency cap. Successful structured
output contains:

- detailed `raw_memory`;
- a `rollout_summary`; and
- an optional filesystem-safe slug.

Outputs are secret-redacted and persisted as stage-one records. A valid
low-signal rollout can return `succeeded_no_output`; failures receive retry
backoff rather than a hot loop.[codex-readme][codex-readme]

The extraction prompt supplies an explicit epistemic contract. Raw rollouts are
immutable evidence, third-party content is data rather than instruction,
secrets must be removed, and a no-op is preferred over filler. It classifies
each task as success, partial, fail, or uncertain and emphasizes exact
commands, validation, failure shields, working-directory scope, and evidence
for user-preference inferences.[codex-stageone][codex-stageone]

This is a stronger provenance discipline than an unstructured “remember this”
summary, but it is still LLM-derived. A prompt rule lowers error probability; it
does not make extraction deterministic or sound.

### Phase 2: selected global consolidation

Phase 2 takes a global lock before touching the shared memories root. It selects
a bounded set of stage-one records, dropping those outside a maximum-unused
window and ranking the rest first by `usage_count`, then by recent
`last_usage` or `generated_at`. It materializes a stable `raw_memories.md` and
per-rollout summaries, prunes unselected or expired inputs, and computes a
Git-style workspace diff against the previous successful baseline.
[codex-readme][codex-readme]

If the workspace changed, a dedicated consolidation agent runs with no network,
no approvals, local write access only, and collaboration disabled. After
success, the system records which stage-one snapshots were consumed and resets
the baseline. Parallel extraction scales across rollouts; the serialized phase
prevents concurrent global writers.

### Consolidated data model

The consolidation contract defines a hierarchy:

- `memory_summary.md` is a compact cross-task routing and preference layer;
- `MEMORY.md` is a grep-oriented handbook grouped by project, working
  directory, or workflow;
- rollout summaries retain task evidence; and
- optional `skills/` packages capture procedures worth invoking as workflows.

Every `MEMORY.md` block must state `applies_to`, including a primary `cwd` and a
reuse rule. Task blocks link rollout-summary paths, timestamps, thread IDs, and
retrieval keywords. The prompt keeps user preferences, validated reusable
knowledge, and failures in distinct subsections.[codex-consolidate][codex-consolidate]

This is not a cognitive episodic/semantic ontology, but it separates evidence,
retrieval summary, durable facts/preferences, and procedural packages more
explicitly than a flat note file.

### Conflict, provenance, and forgetting

The consolidation prompt treats validation strength and `updated_at` together:
fresh validated evidence usually wins; unresolved conflicts remain explicit;
working-directory boundaries must be preserved. When an input disappears from
the selection diff, the agent should remove only memory supported solely by
that input and retain claims with surviving evidence.[codex-consolidate][codex-consolidate]

This supplies source-aware retirement rather than mere recency decay. It is not
a formal dependency graph: the consolidation model must correctly identify
which clauses have mixed support. Usage-based selection can also remove a rare
but critical lesson from the active corpus.

The app-server `memory/reset` operation clears the current Codex home’s memory
directory and staged SQLite data while preserving existing thread memory-mode
settings. Per-thread mode controls future eligibility separately.
[codex-server][codex-server] The public contract describes local reset, not
remote backup, telemetry, or training-data erasure.

### Learning verdict and limitations

Codex changes an external, cross-session, file-backed handbook and may synthesize
procedural skills. It does not train the base model. The two phases add useful
minimum-signal, evidence, concurrency, selection, conflict, and deletion
contracts, but no published benchmark here establishes how accurately they
recover preferences or avoid harmful promotion.

The principal risks are LLM-generated misattribution, poisoned rollout content
surviving the data/instruction boundary, global-root cross-project leakage if
scope metadata is mishandled, usage ranking that suppresses rare safety facts,
and lossy consolidation that outlives its nuance.

## Cross-agent conclusion

pi preserves the strongest raw session lineage but performs no core
cross-session learning. Claude Code provides the simplest automatic editable
note memory but little formal provenance. Codex performs the richest staged
consolidation and source-aware retirement, at the price of more moving parts
and more LLM-mediated transformations.

None updates model weights. For all three, a durable source log, an
active-context summary, an instruction file, and a learned memory artifact are
different objects and should have different deletion and trust policies.

## Local References

[claude-memory]: Anthropic. “How Claude remembers your project,” Claude Code documentation. https://code.claude.com/docs/en/memory (accessed 2026-07-26).

[codex-consolidate]: OpenAI. “Memory Writing Agent: Phase 2 Consolidation,” Codex source template. https://github.com/openai/codex/blob/main/codex-rs/memories/write/templates/memories/consolidation.md (accessed 2026-07-26).

[codex-features]: OpenAI. `codex-rs/features/src/lib.rs`, Codex feature registry. https://github.com/openai/codex/blob/main/codex-rs/features/src/lib.rs (accessed 2026-07-26).

[codex-readme]: OpenAI. “Memories,” Codex source documentation. https://github.com/openai/codex/blob/main/codex-rs/memories/README.md (accessed 2026-07-26).

[codex-server]: OpenAI. “Codex App Server,” source protocol documentation. https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md (accessed 2026-07-26).

[codex-stageone]: OpenAI. “Memory Writing Agent: Phase 1,” Codex source template. https://github.com/openai/codex/blob/main/codex-rs/memories/write/templates/memories/stage_one_system.md (accessed 2026-07-26).

[pi-compact]: earendil-works contributors. “Compaction & Branch Summarization,” pi source documentation. https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md (accessed 2026-07-26).

[pi-readme]: earendil-works contributors. “pi coding agent,” official source documentation. https://github.com/earendil-works/pi/blob/main/packages/coding-agent/README.md (accessed 2026-07-26).
