# Requirements and Non-goals

## Decision frame

Generalist needs durable experience that can improve later work without turning
model-generated prose into an invisible source of authority. The replacement
for `EnhancedMemoryTool` must therefore be a host-owned evidence and lifecycle
system, not a larger scratchpad.

The foundations review supports a conservative transfer from consolidation
research: keep high-fidelity episodes, derive slower abstractions, and make
derived state reversible. An immutable episodic-only mode is also a required
baseline because continuous abstraction can amplify error in the evaluated
settings [faultymemory][faultymemory]. This does not support treating biological
sleep as an implementation specification. Existing systems demonstrate useful
mechanisms but leave provenance, contradiction handling, scope, and complete
deletion to the application. PROV-O supplies a useful general vocabulary for
explicit derivation and invalidation [provo][provo]. The threat model makes
every retained item a potentially persistent and sometimes compositional attack
surface [mempoison][mempoison].

## Functional requirements

### R1 — Immutable, typed episodes

Every settled turn may produce one immutable episode containing:

- stable episode and session identifiers;
- project, user, and purpose scope;
- a canonical admitted view of user inputs and model-visible committed
  messages, with each payload or span marked `retained`, `redacted`,
  `secret_ref`, or `omitted`;
- structured tool invocation and outcome references;
- turn start, event, and record times;
- `TurnOutcome`;
- active goal and queue references, explicitly marked as prospective state;
- the ordered promoted-memory versions that influenced the prompt, their
  retrieval-bundle and policy epochs, and the rendered-bundle/provider-prompt
  hashes;
- provider and policy versions; and
- content hashes and lineage metadata.

Exactness applies only to retained canonical bytes. Redaction and omission are
explicit loss markers, not silently hashed originals; low-entropy sensitive
values use opaque keyed correlation IDs or a secret-manager reference.
Provider reasoning is excluded. A provider refusal, interruption, permission
denial, and iteration cap remain valid outcomes rather than being rewritten as
successes.

### R2 — Crash-recoverable episode construction

The host creates a non-retrievable draft before or at turn dispatch and
atomically finalizes it only after a protocol-valid history checkpoint and
settled outcome exist. A crash may leave a draft, but a draft is never evidence
and never appears in normal retrieval. Recovery may discard it or present it as
an incomplete diagnostic record.

### R3 — Explicit memory classes

At minimum the schema distinguishes:

- observed user messages;
- observed tool or environment results;
- imported untrusted content;
- immutable episodes;
- semantic fact and preference candidates;
- prospective reminders;
- summary and reflection candidates;
- procedure candidates and approved procedures;
- predictions; and
- counterfactual simulations.

Content class is structural, not a label embedded in prose. Generated material
can never be reclassified as an observation merely because it is repeated.

### R4 — Evidence-bearing derivation

Every derived candidate names exact source records or source spans, extractor
and policy versions, scope, valid time when known, and its derivation type.
Evidence graphs are acyclic. A summary cannot corroborate itself through a
descendant, and multiple summaries of one source do not count as independent
support.

### R5 — Candidate/promotion separation

Model-assisted consolidation writes only to quarantine. Deterministic checks,
independent evidence where available, policy checks, regression evaluation, and
user review are distinct gates. The initial release has no automatic semantic
or procedural promotion.

### R6 — Scoped, authorized retrieval

Authorization happens before ranking and again before rendering. Retrieval
honors tenant, project, user, purpose, sensitivity, temporal validity,
supersession, and capability constraints. It returns a bounded data bundle with
provenance and conflict annotations, never an instruction-tier prompt fragment.

Automatic host retrieval initially contains only explicitly approved,
non-procedural memories. Raw episode search remains an explicit read-only
operation so untrusted historical content is not silently injected on every
turn.

### R7 — Correction, deletion, and non-resurrection

Users can inspect, correct, supersede, reject, forget, export, and pause memory.
Deletion walks lineage through derived records, indexes, caches, queued jobs,
and exports under host control. Tombstones prevent later consolidation,
recovery, or rollback from recreating deleted content. Backup erasure semantics
must be documented separately from live-store deletion.

### R8 — Bounded offline consolidation

Offline jobs are cancellable, quota-bound, scope-bound, versioned, observable,
and transactional. A valid result is often “no change.” They may propose
abstractions, contradictions, reminders, procedures, or simulations, but may
not:

- grant credentials, permissions, or capabilities;
- change source class or authority;
- publish directly to promoted memory;
- train or fine-tune model weights;
- execute proposed procedures;
- mix tenants or projects; or
- block interactive prompt dispatch.

### R9 — Quarantined simulation

Counterfactual “dreams” live in a separate simulation namespace, visibly marked
as generated. They may supply test cases or planning hypotheses. They cannot
support promotion, raise confidence, satisfy recurrence, or be returned as past
experience. A prediction becomes informative only after a separately observed
outcome is linked to it.

### R10 — Async runtime fit

The TUI remains interactive during storage, retrieval, and consolidation.
Typing, queue management, cancellation, goal editing, copy mode, reasoning
inspection, and permission resolution retain their current ownership and
liveness contracts. Memory clients use bounded channels to the sole supervisor
connection owner; no blocking database work runs on the current-thread UI
reactor.

### R11 — Durable, inspectable storage

The initial backend is local and project-scoped. It provides schema migrations,
transactions, crash recovery, restrictive Unix permissions, a dedicated owner,
and reconstructable indexes. Lexical retrieval must work without a remote
embedding service. Audit records identify who or what proposed, checked,
approved, retrieved, corrected, or deleted each item.

### R12 — Multi-agent coordination

Several Generalist processes may work in the same project. They can append
episodes concurrently, read pinned snapshots, and propose independent candidate
branches through a trusted supervisor that is the sole database and key owner.
Worker sandboxes cannot open the database directly. Database locks protect
pages; the protocol additionally uses authenticated instances, attenuated typed
capabilities, idempotency keys, version checks, leases with fencing tokens,
monotonic tombstones, and atomic promotion so a stale worker cannot overwrite
newer review or deletion state. A process crash cannot retain exclusive logical
ownership indefinitely.

### R13 — Formal and executable correspondence

Two compositional TLA+ models keep the state spaces reviewable:
`MemoryRuntime.tla` covers capture, provenance class, retrieval, correction,
promotion, deletion, restore, and consolidation; `CollaborationRuntime.tla`
covers agent/task identity, messages, delegation, leases/fences, cancellation,
and external-effect intent. Their explicit shared interface is prompt
construction with memory epochs, effect intent/recheck, revocation/tombstone
notification, and authenticated actor/capability snapshots. The existing
`AsyncRuntime.tla` continues to own one-TUI prompt/permission/cancellation
liveness. A cross-model traceability matrix assigns every invariant to an owner
and executable integration test. CI and contribution review run all models and
require an explicit architecture-model attestation.

## Quality requirements

- **Evidence fidelity:** raw episodes remain available until a documented
  retention or deletion event; incomplete capture is marked and ineligible for
  evidentiary promotion.
- **Defeasibility:** a newer observation may supersede rather than erase an
  older time-bounded fact.
- **Least retention:** secret-like and high-sensitivity fields are excluded or
  redacted by default; users can opt out of capture.
- **Determinism at boundaries:** source-span checks, access control, state
  transitions, transaction publication, and deletion closure are host logic.
- **Observability:** the UI exposes job state, candidate count, conflicts,
  recent retrieval use, and failures without exposing hidden provider state.
- **Version pinning:** extractor, prompts, schemas, ranking, and evaluation
  policies are stored with their outputs.
- **Rebuildability:** derived indexes can be reconstructed from authoritative
  records and tombstones.

## Non-goals

The initial system will not:

- update base-model weights;
- claim human-like cognition, dreaming, or consciousness;
- automatically decide open-world truth;
- use provider reasoning as evidence;
- treat recurrence or embedding similarity as independent corroboration;
- auto-execute learned procedures;
- share memory globally across unrelated projects or users;
- promise erasure from infrastructure outside the documented storage boundary;
- make remote embeddings a dependency;
- use memory to relax tool permissions or current user intent; or
- claim safety from prompt delimiters, classifiers, or benchmark scores alone.

## Acceptance criteria

The episodic foundation is acceptable only when:

1. a crash at every episode transition produces either a finalized episode or a
   non-retrievable draft, never a partial promoted record;
2. current-thread PTY tests show typing and queue operations continue during a
   slow consolidation job;
3. scope tests demonstrate pre-ranking tenant/project isolation;
4. raw-episode deletion tests show immediate exclusion, live-store purge, and
   fail-closed external-ledger restore; descendant invalidation and stale
   promotion are later M2/M3 gates;
5. a generated simulation cannot satisfy any evidence or promotion query;
6. the single-agent episodic milestone passes supervisor-bypass, capture
   quality, secret-admission, and prompt-influence reconstruction tests;
7. the no-memory and episodic-only baselines remain available; and
8. TLC, Rust tests, lints, hooks, and implementation/model trace review pass.

Multi-process finalize/promotion/deletion races are a later collaboration gate,
not a hidden prerequisite for the first episodic capture milestone. The unified
milestone and R1–R13 mapping is in
[the implementation handoff](implementation-handoff.md).

## Local References

[faultymemory]: Dylan Zhang et al. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1 (2026). https://arxiv.org/abs/2605.12978

[mempoison]: Jifeng Gao et al. “MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents.” arXiv:2607.14651v1 (2026). https://arxiv.org/abs/2607.14651

[provo]: World Wide Web Consortium. “PROV-O: The PROV Ontology.” W3C Recommendation (2013). https://www.w3.org/TR/prov-o/
