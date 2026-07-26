# Alternatives and Self-critique

## Decision summary

The preferred design is a host-owned, project-scoped SQLite event and lineage
store with immutable episodes, explicit candidate review, lexical retrieval,
and quarantined simulation. This is not the most autonomous design. It is the
smallest architecture that makes consolidation effects measurable and
reversible.

## Alternatives considered

### Keep and harden the flat JSON tool

Atomic rename, restrictive permissions, provenance fields, and better search
would fix immediate corruption and scope defects.

**Why not sufficient:** the model would still choose arbitrary writes into one
undifferentiated note type. Episode capture, source spans, candidate promotion,
transactional derivation, lineage deletion, and async jobs would all be
reinvented around a whole-file rewrite. This is a migration source, not a
foundation.

### Append-only JSONL event log plus rebuilt indexes

JSONL makes raw episode capture simple, auditable, and crash-tolerant if records
are checksummed and synced.

**Strengths:** minimal dependencies, transparent recovery, immutable by
construction.

**Why not selected alone:** candidate/evidence graphs, revisions, scope-filtered
queries, tombstone closure, atomic multi-record promotion, and review views
need a transactional materialization layer. A log could remain an optional
export or audit format, but dual authoritative stores would create recovery
ambiguity.

### Git-backed Markdown memory

Versioned text files are inspectable, diffable, and familiar. Letta Code
demonstrates that Git history can make self-edited context auditable.

**Why not selected:** Git records byte history, not epistemic source class,
scope authorization, temporal validity, lineage closure, safe retrieval, or
complete erasure. Secrets persist in history and remotes. Concurrent background
updates and transactional multi-record review become merge problems.

### Vector database first

Embeddings improve paraphrase recall and are common in agent-memory systems.

**Why deferred:** similarity is neither truth nor authorization. A remote
embedding dependency creates privacy, cost, versioning, and availability
questions before the lifecycle is sound. Transparent FTS establishes a
reproducible baseline. Semantic indexes can later be evaluated as derived,
rebuildable state.

### Temporal knowledge graph first

Graphiti shows the value of retaining episodes, source edges, validity time,
and invalidation rather than overwriting a profile.

**Why not selected as the initial storage abstraction:** Generalist first needs
correct episode boundaries and a small set of typed revisions. LLM entity
resolution and relation extraction add another unverified model boundary, and
graph deletion can leave summaries or invalidation effects requiring replay.
Relational lineage tables preserve the transferable contract without adopting
graph extraction as a prerequisite.

### Always-in-context mutable profile

A small profile offers predictable recall and low retrieval complexity.

**Why limited to an optional rendered view:** rewriting a profile repeatedly
loses conditions, creates last-write-wins conflicts, and makes complete
provenance awkward. An approved profile can be generated as a versioned view of
atomic records, never the sole authority.

### Automatic consolidation after every turn

This maximizes freshness and imitates common “reflection” agents.

**Why rejected:** it adds latency/cost, makes every interaction a write attack
surface, and repeatedly compresses sparse evidence. Recent controlled work
shows that continuous natural-language consolidation can degrade performance
and that immutable episodic management is a strong baseline
[faultymemory][faultymemory]. “No change” and delayed batching must be normal.

### Fully autonomous promotion

An LLM could extract, verify, and immediately publish memories or skills.

**Why rejected initially:** the same model’s agreement is not independent
evidence; poisoning can be compositional or dormant; rare but harmful errors
persist across sessions. User review is expensive, but silent durable authority
is more expensive before evaluation establishes a safe automation envelope.

### Parametric sleep or continual fine-tuning

Offline replay can distill recent episodes into model weights, as some
continual-learning research explores.

**Why out of scope:** weight changes have a different rollback, privacy,
evaluation, and provenance boundary. They are not deletable with a row
tombstone, and the current provider API does not expose a local, reversible
training loop. External memory must first prove value.

## Why SQLite

SQLite provides transactions, schema migrations, crash recovery, and FTS5 in
one local artifact. WAL permits cross-process readers alongside one writer, so
several authenticated clients can share a project database through one trusted
supervisor connection owner off every current-thread UI reactor.
SQLite locks serialize physical writes; leases, idempotency keys, fencing
tokens, and compare-and-swap versions must serialize semantic decisions.

SQLite’s documentation warns that WAL requires a local filesystem and that
long readers can starve checkpoints. It also documents a rare multi-connection
WAL-reset corruption race fixed in SQLite 3.51.3 and selected backports. The
runtime must bundle or verify a fixed release, manage checkpoints, and never
bypass SQLite locking [sqlite-wal][sqlite-wal]
[sqlite-corruption][sqlite-corruption].

SQLite is not a truth engine. A committed hallucination remains a hallucination.
The value is that the system can atomically record what was observed, proposed,
checked, and approved.

## Self-critique of the preferred design

### Raw episodes increase privacy and injection exposure

High-fidelity history improves provenance but retains more sensitive and
adversarial material. Hashing content is not anonymization, and encryption at
rest does not protect against an authorized-but-overbroad retrieval.

**Mitigation and residual risk:** default project scope, sensitivity
classification, redaction, retention controls, explicit raw search, and
pre-ranking authorization reduce exposure. They do not make indefinite capture
acceptable. Capture must be pausable and visibly degraded or disabled.

### Human review can become an unusable queue

If every recurrence creates candidates, users will rubber-stamp or ignore them.

**Mitigation and residual risk:** strict budgets, deduplication, “no change,”
grouped evidence, explicit high-value triggers, and batched review. The initial
system should prefer missing a memory over flooding review. Automation may
expand only for low-risk, empirically precise candidate classes.

### Recurrence misses rare but crucial events

RecMem-style recurrence is efficient for repeated patterns but can suppress a
one-off constraint, incident, or preference [recmem][recmem].

**Mitigation:** explicit `/memory remember`, prospective reminders, risk/salience
rules, and episode search. Recurrence schedules review; it never defines
importance or truth.

### Same-model consolidation is correlated

A model that misunderstood an episode may repeat the error in extraction and
verification. Multiple prompts or sampled answers are not independent sources.

**Mitigation:** exact spans, deterministic schema/source checks, executable
tests, trusted tool re-observation where possible, user decisions, and honest
“unverified hypothesis” status. Open-world truth remains unsolved.

### Lexical retrieval has known recall limits

FTS misses paraphrases and cross-lingual relations; recency can crowd out older
requirements.

**Mitigation:** structured tags, entities only when deterministically supplied,
temporal/scope filters, hybrid explicit priorities, and a later version-pinned
embedding experiment. The evaluation must measure incremental benefit rather
than assume semantic search wins.

### Delimiter-based prompt rendering is not isolation

XML or “untrusted memory” labels can help the model but do not enforce
instruction precedence.

**Mitigation:** only approved low-authority data enters automatic retrieval;
host authorization and tool permissions remain outside model text; procedures
cannot execute from retrieval alone. Residual content-level steering must be
tested adversarially.

### Lineage deletion is operationally expensive

Deep derivations, exports, backups, and external tool side effects cannot be
erased by one database transaction.

**Mitigation:** bounded lineage depth, rebuildable indexes, deletion manifests,
tombstone replay, export accounting, and precise boundary documentation. The
system must say “pending external deletion” rather than falsely report success.

### One database supervisor can become a bottleneck

Supervisor serial ownership and worker isolation simplify reactor safety, but
SQLite still allows only one WAL writer. Consolidation, FTS maintenance, and
large exports can starve active writes; long snapshot readers can starve
checkpoints.

**Mitigation:** short transactions, immutable snapshots copied before model
calls, busy timeouts with bounded retry, idempotent commands, checkpoint
telemetry, and admission backpressure. A connection pool does not create more
SQLite writers and adds interleavings without solving semantic conflicts.

### Locks do not merge agent beliefs

Two agents can derive incompatible candidates from the same snapshot. A
successful commit only proves one transaction acquired the writer lock; it does
not prove the winner was more accurate or current.

**Mitigation:** append candidates as branches, keep contradictions visible, and
make promotion conditional on expected versions, live sources, review state,
and tombstone epoch. A different coordinator backend remains possible if
contention or governance outgrows local SQLite.

### Formalization can model the wrong boundary

TLA+ may prove an abstract lifecycle while Rust code maps events incorrectly,
or leave SQL/filesystem failure modes unmodeled.

**Mitigation:** action-by-action traceability, PTY/concurrency/crash-injection
tests, model/implementation review as a contribution requirement, and explicit
claims about what TLC cannot prove.

### Quarantined simulations may still leak or bias

Generated tests can contain secrets copied from episodes, propose destructive
actions, or lead developers to overfit to imagined failures.

**Mitigation:** minimize/redact inputs, keep simulation separate, never count it
as evidence, require normal permission for any execution, and compare generated
tests against observed failure coverage. Deleting simulation is not equivalent
to deleting its source episode unless lineage is tracked.

## Rejected slogans

- “More memory is always better.”
- “The model saw it twice, so confidence increased.”
- “Recent means true.”
- “The verifier agreed, so evidence is independent.”
- “Encrypted means safe to retrieve.”
- “Git/SQLite provides semantic rollback.”
- “The dream felt plausible, so it probably happened.”

## Local References

[faultymemory]: Dylan Zhang et al. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1 (2026). https://arxiv.org/abs/2605.12978

[recmem]: Zijie Dai et al. “RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents.” *Findings of ACL 2026*. https://aclanthology.org/2026.findings-acl.1619/

[sqlite-corruption]: SQLite Consortium. “How To Corrupt An SQLite Database File.” https://www.sqlite.org/howtocorrupt.html (accessed 2026-07-26).

[sqlite-wal]: SQLite Consortium. “Write-Ahead Logging.” https://www.sqlite.org/wal.html (accessed 2026-07-26).
