# Implementation Handoff and Unified Milestones

> **Implementation update (2026-07-27):** after reviewing the product value
> question, Generalist implemented a deliberately smaller experiment than the
> roadmap below. The legacy model-controlled tool is gone; opt-in settled-turn
> episodes, explicit local commands, one in-process SQLite worker, and a
> concrete `MemoryRuntime.tla` now exist. Automatic retrieval, trusted
> admission/redaction, supervisor isolation, candidates, consolidation, and
> collaboration remain unimplemented. The repository
> `docs/next-agent-handoff.md` is the authoritative current stop point; this
> document remains the safety envelope for any later expansion.

## Handoff status

This corpus is the reviewed design input for the next implementation agent.
It is **not** evidence that the memory runtime or collaboration runtime exists.
At repository commit `732f5f3`:

- `EnhancedMemoryTool` is still a model-controlled flat JSON CRUD tool and is
  still registered;
- there is no SQLite dependency, supervisor/client protocol, episode schema,
  `/memory` command family, memory TLA+ model, collaboration TLA+ model, or
  memory-specific traceability matrix;
- the async Ratatui reactor and `AsyncRuntime.tla` are the implementation
  baseline and must remain live; and
- OpenRouter Kimi K3 is the default remote model when its key is configured.

The next agent should implement only the first gated slice below. It should not
silently jump to automatic consolidation, generated “dreams,” executable
procedures, or a peer swarm.

## Frozen decisions

1. The host retains immutable admitted episodes before deriving abstractions.
2. Payload disposition is explicit: `retained`, `redacted`, `secret_ref`, or
   `omitted`. “Exact” means exact retained canonical bytes, not pre-redaction
   secrets.
3. Provider reasoning is observability data and never memory evidence.
4. Facts, preferences, summaries, reflections, procedures, and predictions are
   typed candidates. Initial model-generated candidates cannot auto-promote.
5. `SimulationArtifact`, `Prediction`, and observed `VerificationEvent` are
   distinct types and tables. Generated material is never observation or
   corroboration.
6. Automatic retrieval is off in the first episodic release. Raw episode
   search is explicit and read-only.
7. A trusted Unix supervisor is the sole database, index, key, and external
   tombstone-ledger owner. Same-UID mode bits are not isolation.
8. Multi-agent memory remains disabled unless worker sandboxes or distinct OS
   identities cannot directly open supervisor state.
9. SQLite serializes storage; authenticated identities, typed capability
   attenuation, CAS epochs, fencing, idempotency, and tombstone/revocation
   precedence define semantic coordination.
10. Write-capable workers use isolated worktrees and return a commit/patch
    manifest. SQLite does not merge files.
11. No base-model weight updates are part of this project.

## Canonical evaluation variants

Use the same IDs everywhere:

- B0: no persistent memory;
- B1: immutable episodic, chronological selection;
- B2: immutable episodic, production retrieval;
- B3: unguarded consolidation, evaluation-only;
- B4: guarded candidate/promotion pipeline;
- B5-off/B5-on: B4 without/with quarantined generated counterfactuals.

No consolidation claim is valid unless B4 beats B1/B2 under matched budgets.

## Unified milestone DAG

```text
M0 contracts/models/measurement
├── M1 single-agent immutable episodic capture + explicit search
│   └── M2 user-authored/approved memory
└── C1 two observable cancellable read-only workers
    └── C2 durable tasks/messages/leases/fences/typed delegation
        └── C3 isolated worktrees + reviewed artifact integration

M2 + C2
└── M3 manual offline candidate proposals
    └── M4 conservative scheduling
        └── M5 quarantined simulations / inert procedures

M1 + C2 + C3 + supervisor OS isolation
└── MC1 private-by-default team episodic memory
    └── optional peer/self-claim experiments after security evaluation
```

M0 is first. M1 and C1 may proceed independently after it. M3 requires both M2
and C2 because a consolidator must claim a task and lease, then publish through
a fenced worker session. MC1 cannot inherit a same-UID development shortcut.

## TLA+ composition contract

Keep three specifications rather than one unreviewable cross product:

- `AsyncRuntime.tla` owns one TUI reactor, prompt queue, permission,
  cancellation, history validity, and copy-mode liveness.
- `MemoryRuntime.tla` owns draft/finalize, capture quality, source class,
  candidate/review/promotion, domain-scoped retrieval, correction,
  tombstone/restore, and consolidation cancellation.
- `CollaborationRuntime.tla` owns authenticated agent/task/attempt identity,
  immutable message envelopes plus events, dependency DAGs, typed delegation,
  leases/fences, effect intents, cancellation, and worker recovery.

Shared interface actions are:

1. `BindPromptMemory`: bind ordered memory versions, rendered hash, and current
   retrieval/tombstone/policy epochs to a prompt draft.
2. `RecheckPrompt`: immediately before provider I/O, reject a stale binding and
   re-retrieve/re-render or abort. Provider-send state is `prepared`,
   `dispatched`, or `sent_unknown`; dispatch is the exposure linearization
   point.
3. `RegisterActorSnapshot`: bind a host-authenticated actor/task/attempt and
   capability epoch to memory operations.
4. `CreateEffectIntent`: record contributing memory IDs/epochs, delegation,
   and idempotency key.
5. `RecheckEffect`: immediately before dispatch, reject stale memory,
   tombstone, revocation, task-fence, or capability epochs.
6. `InvalidateTurn`: notify the async controller to cancel an affected local
   turn; record already-sent provider exposure.

TLC checks each finite model. Deterministic Rust integration schedules exercise
the shared interface; a green model does not prove SQL, filesystem isolation,
factual truth, or Rust refinement.

## Requirements traceability

| Requirement | First milestone | Model owner | Required executable evidence |
| --- | --- | --- | --- |
| R1 immutable typed episodes | M1 | Memory | schema/class tests; reasoning exclusion; immutable update rejection |
| R2 crash-recoverable construction | M1 | Memory + Async interface | crash at every begin/event/finalize boundary; draft never retrieved |
| R3 explicit classes | M0/M1 | Memory | generated/observed type separation and migration tests |
| R4 evidence-bearing derivation | M3 | Memory | span/causal-root DAG properties; copied-root laundering tests |
| R5 candidate/promotion separation | M2/M3 | Memory | model has no promote capability; CAS review transaction |
| R6 scoped authorized retrieval | M1/M2 | Memory | per-domain rank/count/content/cache isolation; residual timing measurement |
| R7 correction/deletion/non-resurrection | M1 raw episodes; M2/M3 derivatives | Memory + Async/Collaboration interfaces | M1 tombstone/live purge/ledger restore; later descendant invalidation, stale-promotion races, forensic scans |
| R8 bounded offline consolidation | M3 after C2 | Memory + Collaboration + Async | worker session, quotas/cancel/lease loss; prompt remains live |
| R9 quarantined simulation | M5 | Memory | generated item cannot satisfy evidence query; observed verification stays separate |
| R10 async runtime fit | every milestone | Async | current-thread tests and stalled-provider PTY regression |
| R11 durable inspectable storage | M0/M1 | Memory | patched SQLite version; modes/owner/symlink; migration/integrity/degraded start |
| R12 multi-agent coordination | C1/C2/MC1 | Collaboration + Memory interface | identity/delegation/fence/CAS races; supervisor bypass; worktree isolation |
| R13 formal/executable correspondence | M0 then every milestone | all three | action/invariant matrices, TLC, deterministic cross-interface traces |

## Exact first implementation slice

### 1. Preserve a reviewable stop point

Create M0 without serving or capturing memory:

- feature flag memory off by default;
- supervisor/client message types, an implemented principal/project-bound
  controller session, a schema/model-only worker-session type, and a fake
  in-memory supervisor for protocol tests;
- schema migrations and a command that reports disabled/degraded status;
- `MemoryRuntime.tla` and `CollaborationRuntime.tla` skeletons/configurations,
  the existing `AsyncRuntime.tla` shared-interface rows, cross-model
  traceability skeleton, CI invocation, and contribution attestation coverage;
- threat fixtures for secret admission, generated/observed separation, and
  supervisor bypass.

Do not register a new model-facing memory write tool in M0.

### 2. Choose SQLite deliberately

The reviewed candidate is `rusqlite` with bundled SQLite. At the research
cutoff, `rusqlite` 0.39.0 bundled SQLite 3.51.3, which includes the WAL-reset
fix [rusqlite][rusqlite] [sqlite-3513][sqlite-3513]. Pin through `Cargo.lock`
and assert `sqlite_version() >= 3.51.3` at runtime and in tests. Re-check before
implementation because crate and SQLite releases are change-sensitive.

Use local filesystems only, short supervisor-owned transactions, bounded busy
handling, and no model/provider call inside a transaction. A same-UID developer
sidecar is useful for semantics but must display `isolation: unsafe-dev` and
must not unlock shared-agent memory or automatic retrieval.

### 3. Integrate M1 at existing trusted boundaries

- Replace `EnhancedMemoryTool` registration in `src/main.rs`; preserve its raw
  JSON only as a future `legacy_model_note` quarantine import. Never
  auto-retrieve it.
- Add `/memory status|pause|search|show|export|forget` to the typed
  `COMMAND_SPECS` parser and idle command controller. Add no model write path.
- M1 `/forget` covers raw-episode tombstoning, immediate live-store exclusion,
  resumable live purge, and external-ledger restore enforcement. Because M1 has
  no promoted descendants, correction propagation and stale-promotion races
  remain M2/M3 gates.
- Immediately before `Agent::begin_turn`, create a non-retrievable draft after
  any retrieval bundle is fixed.
- Derive essential admitted events only from host `AgentEvent`,
  `HistoryCheckpoint`, structured tool outcomes, and `TurnOutcome` boundaries.
  Do not capture provider reasoning.
- Finalize only after a history-valid checkpoint and settled outcome. An API
  error is a typed outcome, not inferred success.
- Poll memory status/events in every idle, active-turn, compaction, permission,
  and copy-mode reactor path. SQLite work never runs on the current-thread UI
  reactor.
- Keep explicit episode search out of automatic prompt context. A future
  code-mode read bridge may return typed records, but no mutable connection or
  database path.

### 4. Tests that must fail before implementation

- crash/fault injection at every draft/event/finalize transition;
- omitted-essential-event episodes are promotion-ineligible;
- credential and exclusion canaries never appear in DB, WAL, FTS, audit, or
  ordinary hashes;
- copied imported spans never gain authority through user/assistant/peer
  quoting;
- unauthorized rows cannot change authorized FTS ranks, counts, returned
  content, cache keys, or observable result class; latency is measured as a
  residual side channel rather than asserted invariant;
- generated simulation/prediction IDs cannot enter evidence edges;
- restored DB fails closed without the external ledger high-water mark;
- ledger-ahead/database-ahead and crash-before/after-fsync/commit schedules
  reconcile or fail closed;
- per-artifact and whole-scope key destruction cannot erase unrelated records
  or revive through key/database backup restore;
- model-controlled tool direct-open/copy/lock/socket replay is denied in
  isolated mode;
- controller-session replay, PID reuse, expiry, stolen material, and
  tool-subprocess FD/environment inheritance are rejected;
- terminal review escapes ANSI/OSC, bidi, invisibles, deceptive links, and
  oversized evidence;
- typing, queue editing, scrolling, copy mode, reasoning inspection,
  permissions, and cancellation remain live during a stalled memory request.

Before M2/M3 exits, add correction/descendant invalidation, stale-promotion,
affected effect-intent, and full derived-lineage forensic schedules. They are
not falsely claimed by M1’s raw-only deletion tests.

Before C2 exits, implement and test worker sessions: task/attempt/fence binding,
stale-fence and lease-loss revocation, replay/PID reuse, stolen material, and
tool-subprocess FD/environment inheritance. M0 models their contract but does
not falsely claim a running worker authentication path.

## External effects

SQLite can make one host authorization/effect intent idempotent. It cannot make
an arbitrary remote effect exactly once. Use provider idempotency keys when
supported. Otherwise persist `prepared`, `sent_unknown`, `confirmed`, or
`reconciled` and expose possible duplicates; never convert an uncertain send
into success.

## Stop rules for the next agent

Stop and leave evidence rather than broadening scope if:

- the supervisor cannot be kept off the TUI reactor;
- the chosen environment cannot isolate worker tools from supervisor files;
- capture admission would persist excluded raw values before redaction;
- the implementation needs global FTS ranking before authorization;
- the TLA interface and Rust transition cannot be traced action by action;
- M1 requires automatic model-generated promotion; or
- the existing async/runtime/PTTY/TLC baseline regresses.

## Validation commands

From the main repository after this corpus is imported:

```sh
python3 docs/research/agent-memory/systems/data/check_corpus.py
python3 docs/research/agent-memory/safety-evaluation/data/validate-branch.py
PYTHONDONTWRITEBYTECODE=1 \
  python3 docs/research/agent-memory/data/validate_corpus.py
make check
git diff --check
```

The first three validate research structure, not implementation truth. `make
check` must grow to run all checked-in TLA+ models and memory traceability once
M0 begins.

## Local References

[rusqlite]: rusqlite contributors. “Ergonomic bindings to SQLite for Rust.” Official repository documentation, accessed 2026-07-26. https://github.com/rusqlite/rusqlite

[sqlite-3513]: SQLite Project. “SQLite Release 3.51.3 On 2026-03-13.” Official release notes. https://www.sqlite.org/releaselog/3_51_3.html
