# Candidate Memory and Consolidation Protocol

## Status

This protocol is the preferred architecture for review. It intentionally
implements less autonomy than many “dreaming” proposals: durable episodes and
user-controlled recall arrive first; offline consolidation can initially
create only quarantined candidates.

## Components and ownership

```text
UI/controller ── commands/status ──> MemoryHandle ── authenticated UDS ─┐
Agent events ── structured capture ─> MemoryHandle                    │
ConsolidationWorker ── proposals ───> MemoryHandle                    ▼
                                                        MemorySupervisor
                                                  (sole DB/key/ledger owner)
                                                     │              │
                                              authoritative DB   job manifests
                                                     │
                                               domain-local FTS
```

`MemoryHandle` is a non-blocking, clonable client. A trusted Unix supervisor is
the only process that opens the database, WAL, deletion ledger, encryption
keys, or FTS files. It authenticates each Unix-socket request with peer process
credentials and an operation-appropriate host-issued controller or worker
session. The UI reactor still owns
terminal input and `Agent` owns conversation history. A consolidation worker
receives an immutable, scope-filtered snapshot and publishes through the
supervisor; it never holds a transaction across a model call.

Mode `0600` does not isolate two processes running as the same Unix user. Shared
multi-agent memory therefore remains disabled unless an OS sandbox or distinct
service identity prevents workers and their tools from opening, copying,
locking, or replacing the database and socket credentials directly. A
same-process development mode may test semantics, but it is not a security
boundary and cannot satisfy the multi-agent release gate.

The supervisor creates two operation-disjoint session types through a one-time
challenge-response:

```text
ControllerSession:
session_id, principal_id, controller_run_id
peer_pid, peer_process_start_nonce
project_id, capability_set_id, policy_epoch
issued_at, expires_at, request_sequence, socket_session_nonce

WorkerSession:
session_id, agent_id, run_id
peer_pid, peer_process_start_nonce
project_id, task_id, attempt_id, fencing_generation
capability_set_id, policy_epoch
issued_at, expires_at, request_sequence
socket_session_nonce
```

Controller sessions authorize user-facing capture and idle `/memory` operations
for one authenticated principal/project; they cannot claim tasks, publish
worker artifacts, or exercise a fence. Worker sessions authorize only the
named task/attempt/fence and cannot impersonate the controller.

The host verifies Unix peer credentials plus the process-start nonce to prevent
PID reuse, MACs each framed request over the session nonce and monotonic request
sequence, and rejects replay/out-of-order frames. Session material is
host-memory only: never prompt text, environment, argv, logs, or inherited tool
subprocess state. Supervisor/socket descriptors are `CLOEXEC`; tool subprocesses
receive separate narrowed capability channels. Process exit, policy change, or
expiry revokes either session; lease loss or task cancellation additionally
revokes a worker session before another request or effect. A peer credential
proves the process, while the session binds the relevant principal/controller
or logical agent/task/attempt/fence.

## Authoritative stores

The schema is relational even if some payloads are JSON:

- `episode_drafts`: non-retrievable construction records;
- `episodes`: immutable finalized headers and outcome;
- `episode_events`: ordered, typed event payloads and hashes;
- `artifacts`: content-addressed large payloads, with sensitivity metadata;
- `key_inventory`: wrapped DEK, wrapped per-artifact KEK and scope-KEK IDs,
  algorithm/version, storage-copy inventory, and
  `active|destroy_pending|destroyed` state;
- `artifact_spans`: byte ranges with origin, taint, transformation, and causal
  root identity; mixed spans inherit the least-authoritative class;
- `candidates`: versioned semantic, reminder, summary, reflection, or procedure
  proposals;
- `candidate_sources`: exact evidence edges and optional byte spans;
- `memories`: approved, versioned records;
- `revisions`: supersession, contradiction, correction, and invalidation edges;
- `simulation_artifacts`: generated counterfactuals only;
- `predictions`: generated claims with evaluation deadlines and calibration
  state;
- `verification_events`: separately observed tool/test outcomes that may link
  to, but never become part of, a simulation or prediction;
- `procedures`: inert proposals and separately approved manifests;
- `jobs`: input snapshot, state, quotas, policy, result, and cancellation;
- `leases`: expiring ownership plus monotonically increasing fencing tokens;
- `tombstones`: deletion roots and monotonic invalidation state;
- `retrieval_audit`: authorization, rank, render, and use decisions;
- `effect_intents`: memory IDs/epochs, current capability decision, and
  idempotency key carried to the final external-effect check;
- `policy_versions` and `schema_migrations`.

FTS5 is a rebuildable lexical index, not an authority store
[fts5][fts5]. Ranking never runs over a global unauthorized corpus: the first
release has one index per exact security domain. A query spanning several
authorized domains ranks each independently and merges their bounded results
with a deterministic host rule, so unauthorized rows cannot change IDF,
candidate counts, returned content, or cache keys for an allowed domain. Shared
supervisor CPU, queue, SQLite pages, WAL/checkpoints, and I/O still create a
measurable timing side channel. Exact timing noninterference requires stronger
per-domain process/database/resource partitioning and padding; it is not
claimed by FTS partitioning alone. The first release does not require
embeddings. If semantic retrieval is later added, vectors are partitioned by
the same domains.

The supervisor-owned database and sidecar files use mode `0600`; their parent
directory uses `0700` and is inaccessible to worker identities. SQLite WAL is
used only on a local filesystem. Connections use a busy timeout, short
transactions, and a patched SQLite release; WAL versions affected by the 2026
reset race are rejected [sqlite-wal][sqlite-wal]. Startup performs migration,
integrity, owner/mode, symlink, and runtime-version checks before memory is
enabled; a failure disables memory visibly rather than preventing the whole
agent from starting.

Canonical retained payloads use application-layer envelope encryption:
`payload_ciphertext = AEAD_DEK(payload)`,
`wrapped_DEK = AEAD_artifact_KEK(DEK)`, and
`wrapped_artifact_KEK = AEAD_scope_KEK(artifact_KEK)`. Only ciphertexts,
wrapped keys, and IDs persist; plaintext DEKs/KEKs exist only inside the
supervisor key boundary. Record deletion destroys the artifact KEK and appends
and fsyncs its key ID in the external destruction ledger; whole-scope erasure
destroys the scope KEK. Restore replays destroyed key IDs before unwrapping
anything. Every DB, spill, export, and backup copy carries an inventory key ID.
Exceptional sensitive payloads have no FTS projection; ordinary admitted search
projections remain subject to secure-delete/purge and forensic scans.

## Episode lifecycle

### 1. Begin

Immediately before dispatching prompt `p`, the controller asks the supervisor to
create draft `d` with:

- prompt/queue identity and delivery mode;
- session/project/user/purpose scope;
- active goal reference;
- retrieval-bundle ID, ordered memory-version IDs, retrieval/tombstone/policy
  epochs, rendered-bundle hash, and final provider-prompt hash;
- provider and runtime versions; and
- start time.

If draft creation fails, the turn may proceed with memory visibly degraded.
The runtime must not claim the episode was captured.

### 2. Observe

Before durable capture, trusted host logic applies a declared admission policy:
drop excluded fields, replace ordinary secrets with typed redactions, and route
the rare explicitly retained sensitive payload to a separately encrypted store.
The episode records the exact **admitted** bytes plus the admission-policy
version, never an unqualified claim that the original bytes were persisted.
Low-entropy secrets are not placed in ordinary content hashes; correlation uses
a keyed opaque digest whose key is held only by the supervisor.

During the turn, the host records admitted, typed events already available at
trusted runtime boundaries: queue claim, committed user messages, tool name and
normalized arguments, permission outcome, tool result hash/status, model
response fragments needed for the final transcript, retry, and cancellation.
Span records retain origin through copying and transformation. Quoting imported
text inside a user, assistant, tool, or peer message does not upgrade it; an
untraceable or mixed span receives the least-authoritative applicable class.

Essential capture consists of the admitted current user/steering messages,
committed assistant blocks, tool-use/result pairs, permission outcomes, final
turn outcome/checkpoint, and the retrieval/prompt binding above. It has reserved
channel capacity or a bounded append-only spill journal. Every omitted
nonessential event gets an explicit marker. Losing an essential event sets
`capture_quality = incomplete`; such an episode may support diagnostics but is
excluded from evidentiary promotion.

Events sent by the provider are never upgraded beyond provider-originated text.
Reasoning deltas are excluded entirely. Imported file, web, or tool content
keeps its untrusted origin even when quoted by the assistant.

### 3. Finalize

After `Agent` emits a protocol-valid `HistoryCheckpoint` and returns a
`TurnOutcome`, the controller sends one finalize command containing the
checkpoint identity, settled outcome, end time, and expected ordered event
hash. The supervisor transaction:

1. verifies the draft is open and scoped correctly;
2. verifies event order/hash and required boundary fields;
3. inserts immutable episode/event records;
4. marks the draft finalized; and
5. commits all changes atomically.

Only after commit can the episode appear in explicit episode search or a
consolidation snapshot. Finalize is idempotent by episode ID and content hash.
Crash recovery never infers missing success from assistant text. Executable
failure-injection tests must verify that the schema and transaction usage
actually preserve this boundary; the SQLite atomic-commit protocol is necessary
context, not an application-level proof [sqlite-atomic][sqlite-atomic].

## Retrieval lifecycle

### Explicit episode search

The user or a code-mode script can search raw episodes. Host logic first
selects exact authorized security domains, then each domain-local FTS index
ranks only its allowed corpus. Results are typed records with snippets and
provenance, not executable instructions. Code mode lets a script inspect large
result sets without injecting all rows into model context.

### Automatic promoted-memory retrieval

Before a provider request, the controller may request a small bundle of
approved non-procedural memories. The supervisor:

1. resolves effective user/project/purpose scope;
2. excludes tombstoned, invalid, expired, conflicting-without-policy, and
   capability-incompatible records;
3. ranks the authorized set lexically and by explicit priority/recency;
4. applies diversity and token budgets;
5. returns provenance and validity metadata; and
6. writes an audit decision and returns the record IDs plus retrieval,
   tombstone, revocation, and policy epochs that downstream effect checks must
   carry.

Immediately before provider I/O, the controller asks the supervisor to
`RecheckPrompt` against those epochs. A stale binding is aborted and
re-retrieved/re-rendered (or the turn fails visibly); it is never dispatched.
The provider-send state is `prepared`, `dispatched`, or `sent_unknown`, with
`dispatched` as the exposure linearization point. A deletion that commits first
wins; a send that linearizes first is recorded as an exposure that cannot be
recalled.

The renderer identifies the bundle as untrusted remembered data, below current
system/developer/user instructions. Memory cannot alter permission policy or
tool schemas. The user can inspect which item influenced a turn.

## Consolidation lifecycle

### Trigger

Jobs may be started manually, during idle time, or by a conservative
recurrence/salience scheduler. Interactive prompts take priority. A trigger
creates an idempotent manifest with tenant/project scope, immutable episode
cutoff, candidate types allowed, policy and prompt hashes, budgets, and
cancellation token. One worker claims an expiring lease and fencing token
through a task/attempt-bound `WorkerSession`. M3 therefore depends on C2. A
controller session may create or cancel job manifests and review candidates,
but it cannot publish worker proposals.

Recurrence is a scheduling signal, not evidence. Rare one-off events can enter
review through an explicit user “remember this” path. Forced consolidation
after every turn is excluded because repeated lossy rewriting can degrade
performance [faultymemory][faultymemory].

### Snapshot and replay

The supervisor selects authorized immutable episodes and existing approved memories
at the recorded cutoff. It never includes another job’s unreviewed candidates
as evidence. Source diversity, temporal coverage, and unresolved
contradictions are represented explicitly. The worker receives the snapshot
after the read transaction closes.

### Proposal

The model may return typed proposals:

- a fact or preference with validity/scope;
- a supersession or contradiction link;
- a prospective reminder with a trigger and expiry;
- a lossy summary;
- a reflection/hypothesis;
- an inert procedure with preconditions and capability manifest;
- a counterfactual simulation; or
- no change.

Every non-simulation proposal cites exact source IDs and spans. The proposal
format contains no field capable of granting authority or changing source
class.

### Validation

Host checks reject malformed schemas, missing/out-of-range spans, cycles,
deleted sources, cross-scope sources, unsupported content classes, excess
lineage depth, undeclared capabilities, secrets outside policy, and incomplete
transactions. Contradiction and temporal checks create review findings rather
than silently choosing the newest string.

An optional verifier can inspect independent trusted tools or tests. Re-asking
the same model does not count as independent corroboration. Candidate
confidence is never raised by descendant count or generated simulations.
Evidence roots additionally carry a causal/correlation identity for the
upstream artifact and version. Copies reached through different agents, URLs,
tools, or summaries collapse to one root when they share an origin; unknown
dependence is conservatively counted as one, not guessed independent.

### Publish and review

The supervisor atomically publishes the complete checked batch to quarantine with
the job manifest only if the worker still holds the current fencing token.
Cancellation, lease loss, or failure before commit publishes nothing.
The TUI review view presents proposal, type, scope, evidence excerpts,
conflicts, verifier results, and expected retrieval effect.

User actions are:

- approve as a new version;
- edit, producing a user-authored revision with retained lineage;
- reject with reason;
- defer;
- mark sources for deletion; or
- approve a reminder without approving a broader inferred fact.

Approval creates a `memory` version and audit event. Procedures require a
second, capability-specific approval and remain inert in the initial release.

A correction or material edit atomically marks every descendant
`needs_revalidation`. Descendants supported only by invalidated roots disappear
from retrieval immediately; mixed-support descendants are recomputed before
they may be served. A material user edit is a new user-authored assertion with
fresh validation, not inherited verifier approval.

## Simulation, prediction, and “dreaming”

The implementation should avoid the overloaded type name `Dream`. A
`SimulationArtifact` is a generated counterfactual that may be useful for:

- producing adversarial regression cases from real failures;
- rehearsing queue/cancellation/storage interleavings;
- finding missing procedure preconditions;
- exploring alternative plans before execution.

Simulation IDs, storage tables, UI styling, retrieval APIs, and query types are
separate from episodes. The evidence-edge schema rejects simulation sources for
fact promotion. Simulation output may cause a test to be run; only the
separately captured `VerificationEvent` can support a candidate, linked by
`evaluates_simulation` without entering the simulation namespace. A generated
`Prediction` likewise remains separate from the later observation that
evaluates it. Generated test cases must still be reviewed for secret leakage
and destructive actions.

This preserves the useful part of world-model “dreaming”—cheap offline
hypothesis generation—without manufacturing history. Systems that change
weights through replay solve a different continual-learning problem and are
out of scope.

## Correction and deletion

Correction adds a revision and invalidates the prior version for current-time
retrieval while retaining historical validity where appropriate. Contradictory
sources remain visible until policy or review resolves them.

Deletion uses an ordered two-store protocol, not cross-store atomicity:

1. authorize and allocate an idempotent `tombstone_id`;
2. append a hash-chained, authenticated ledger record with the next monotonic
   sequence, then `fsync` the ledger and its directory;
3. in one short SQLite transaction, apply that tombstone/lineage closure,
   exclude it from reads/promotion/effects, stop affected jobs, create the purge
   manifest, and set `applied_ledger_high_water` to the sequence; and
4. acknowledge only after the SQLite commit.

On every ordinary start and restore, before any memory read, the supervisor
verifies the ledger chain/signature and compares ledger high water `L` with
database-applied high water `D`. If `D < L`, it idempotently replays missing
tombstones into SQLite. `D > L`, an invalid chain, an anchor that expects a
missing tail, or a conflicting repeated `tombstone_id` fails closed. A crash
before ledger fsync has no accepted deletion; after ledger fsync but before DB
commit it replays; after DB commit but before reply a retry returns the same
result. Fault injection covers every boundary.

Resumable idempotent purge jobs then move affected artifact keys through
`destroy_pending` to `destroyed`, remove authoritative payloads, use SQLite/FTS
secure-delete controls, clear temporary
spill state, remove index/cache entries, checkpoint/truncate WAL, run the
declared vacuum procedure, and reconcile exports/backups. Each tier records
completed, deferred, excepted, or failed. No cross-tier atomicity is claimed.

Reconsolidation filters tombstoned roots before snapshot creation. Restoring an
old database requires replaying newer tombstones before serving reads. The
ledger is kept in a separate authenticated failure domain (or an operator
recovery export); if its signed high-water mark is missing, stale, or corrupt,
the restored database fails closed. Without that external anchor the product
must describe deletion as live-store logical deletion and must not claim
backup-resurrection safety.

SQLite forensic tests scan database pages, WAL/SHM, FTS shadow tables, temporary
files, backups, and restored copies for synthetic canaries. Append-only audit
records retain only erasable or non-sensitive identifiers required by policy;
they never silently preserve deleted payloads.

## Concurrency and cancellation

Interactive commands and memory messages share no in-process database state.
Across clients, the supervisor uses SQLite snapshot reads and one writer at a
time; application versions and fencing tokens supply semantic concurrency. Two
agents can append episodes independently. Consolidators produce candidate
branches rather than mutating one shared summary. Promotion verifies the
candidate version, live evidence, review decision, and tombstone epoch in the
same short write transaction; conflicts remain reviewable instead of becoming
last-writer-wins.

Every rendered prompt and proposed effect carries the contributing memory IDs
and epochs. Deletion, scope narrowing, or revocation cancels affected local
turns where possible. The host rechecks those epochs immediately before
provider dispatch, tool authorization, and effect dispatch. Data already sent
to a provider cannot be recalled; that exposure is recorded as an explicit
deletion receipt exception.

Priority classes are:

1. shutdown, deletion, and permission resolution;
2. episode finalization and explicit UI commands;
3. retrieval needed for an active prompt;
4. capture append;
5. consolidation and index maintenance.

Priorities govern local dequeue choice, not transaction preemption. Transactions
are short; model calls happen outside them. Consolidation observes cancellation
and lease renewal at bounded intervals and settles to `completed`, `failed`,
`cancelled`, or `lease_lost`. Queueing a prompt cancels or suspends local
low-priority work but never discards a finalized episode.

Backpressure produces a visible degraded state. The controller may omit only
declared nonessential events and must append their omission markers. Essential
events use reserved capacity/spill; losing one marks the episode incomplete and
promotion-ineligible. Backpressure may not block terminal input or silently
report durability.

## Formal model boundary

`MemoryRuntime.tla` should model IDs and content classes abstractly. Required
safety invariants include:

- finalized episodes originate from exactly one open draft and settled turn;
- promotion-eligible episodes have complete essential capture and an immutable
  prompt/retrieval binding;
- drafts and simulations are never retrievable as observations;
- a generated artifact cannot become a verification event or evidence root;
- promoted memories have approved candidates and live evidence roots;
- evidence is acyclic and cannot self-corroborate;
- no read or job crosses scope;
- capabilities never increase through memory transitions;
- publication is all-or-nothing;
- at most one fencing generation may publish a job result;
- stale promotion loses to a newer review, source revision, or tombstone;
- tombstones survive rollback and prevent resurrection.

Liveness under weak/strong fairness should cover:

- accepted finalizations eventually settle;
- cancelled jobs eventually stop;
- an expired lease can eventually be reclaimed;
- review decisions eventually publish or reject;
- a queued prompt eventually dispatches despite background work.

TLC establishes state-machine properties under the finite model. It does not
prove factual truth, privacy of unmodeled storage, SQL correctness, model
faithfulness, or legal compliance; executable traceability tests cover the
implementation boundary.

## Local References

[faultymemory]: Dylan Zhang et al. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1 (2026). https://arxiv.org/abs/2605.12978

[fts5]: SQLite Consortium. “SQLite FTS5 Extension.” https://www.sqlite.org/fts5.html (accessed 2026-07-26).

[sqlite-atomic]: SQLite Consortium. “Atomic Commit In SQLite.” https://www.sqlite.org/atomiccommit.html (accessed 2026-07-26).

[sqlite-wal]: SQLite Consortium. “Write-Ahead Logging.” https://www.sqlite.org/wal.html (accessed 2026-07-26).
