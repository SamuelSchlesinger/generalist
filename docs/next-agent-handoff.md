# Next-Agent Handoff

## Current stop point

This handoff intentionally stops before the episodic-memory implementation.
The main branch already contains:

- `992f62f`: asynchronous Ratatui runtime;
- `0e8ddd4`, `fa7a2ea`, and `db900fa`: wrapped-scroll, durable `/goal`, copy,
  paste, reasoning-inspection, direct-tool-boundary, persistence, and UI-jank
  fixes;
- `732f5f3`: OpenRouter provider with `moonshotai/kimi-k3` as the default remote
  model when `OPENROUTER_API_KEY` exists.

The OpenRouter key in `~/.generalist.env` was detected without printing it. A
live Kimi smoke request reached OpenRouter but returned HTTP 402 “Insufficient
credits.” That is the current account-state blocker for live generation, not a
provider-selection or authentication failure. Do not copy the key into source,
logs, tests, or this document.

The complete episodic-memory, consolidation, and multi-agent research corpus is
under [agent-memory](research/agent-memory/index.md). It passed parallel
accuracy, security, and coherence review through three revision cycles. The
final reviewers reported no remaining Error or Gap. Structural validation
reports 33 Markdown pages, 310 citation uses, 119 canonical sources, and 19
threat-control rows.

## What is deliberately not implemented

- SQLite/rusqlite or a memory supervisor/client protocol
- immutable episode drafts/finalization
- `/memory` commands
- automatic memory retrieval
- model-generated candidate promotion or consolidation
- simulations, predictions, procedures, or weight updates
- durable tasks/messages/delegation for multiple agents
- `MemoryRuntime.tla` or `CollaborationRuntime.tla`

`src/tools/enhanced_memory.rs` remains the legacy flat JSON CRUD tool and is
still registered. Do not describe it as episodic memory.

## Exact next action

Read the
[implementation handoff](research/agent-memory/architecture/implementation-handoff.md)
and implement M0 only:

1. memory disabled by default;
2. supervisor/client message contracts and a fake supervisor;
3. principal/project-bound controller-session contract;
4. schema/measurement/degraded-status scaffolding;
5. `MemoryRuntime.tla` and `CollaborationRuntime.tla` skeletons/configurations;
6. shared-interface traceability to the existing `AsyncRuntime.tla`;
7. CI, hook acknowledgement, and failing threat fixtures.

Do not start M1 capture in the same commit. M0 is the reviewable boundary.

## Non-obvious review constraints

- Same-UID `0600`/`0700` files do not isolate model-controlled tool processes.
- Exact capture means exact admitted canonical bytes after
  redaction/omission—not raw secrets.
- Provider reasoning and generated simulations are not evidence.
- Domain-local FTS isolates ranks/content/cache keys, not shared-resource
  timing.
- Prompt memory epochs need a recheck immediately before provider dispatch.
- Remote effects may be `sent_unknown`; SQLite cannot promise exactly once.
- The external deletion ledger is fsynced before SQLite applies its high-water
  mark; startup reconciles ledger-ahead state and fails closed on DB-ahead.
- M1 `/forget` covers raw episodes only. Descendant invalidation and
  stale-promotion races belong to M2/M3.
- Worker attempt sessions are implemented at C2, separately from the M0/M1
  controller session.
- M3 offline consolidation requires both M2 and C2. Only a task/attempt-bound
  fenced worker session may publish proposals; a controller session cannot.

## Validation

Run:

```sh
make memory-research
make check
git diff --check
```

When M0 adds the two new TLA+ specifications, extend `make tla`,
`scripts/check-runtime-traceability.sh`, CI, the Git-hook acknowledgement, and
the contribution methodology in the same commit. Preserve the current
442,870-generated / 113,190-distinct / depth-27 `AsyncRuntime.tla` TLC baseline
until a deliberate model change explains a new state-space count.
