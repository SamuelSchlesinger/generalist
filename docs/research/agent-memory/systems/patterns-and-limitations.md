# Transferable Patterns and Unresolved Limitations

## Executive finding

The systems converge on a durable design principle:

> Keep evidence, active context, and consolidated knowledge as different
> artifacts.

Raw evidence is needed for audit and correction. Active context must be small
and is necessarily selective or lossy. Consolidated knowledge should be
reusable but must retain enough lineage to be revised. A single mutable vector
store cannot cleanly satisfy all three roles.

The systems do **not** converge on a safe universal write trigger, ranker, or
forgetting rule. Every trigger drops some useful one-offs; every consolidation
step can invent or flatten information; and every deletion API faces derived
copies. The findings below are current to the primary-source pass completed on
2026-07-26.

## 1. Use a layered data model

A robust architecture needs at least four logically distinct layers:

| Layer | Purpose | Representative systems | Required invariant |
| --- | --- | --- | --- |
| Evidence log | recover exactly what happened | pi JSONL, Graphiti episodes, Codex rollouts, RecMem subconscious turns | immutable or versioned; tenant and source identity |
| Working set | fit the current model context | pi/Claude compaction, Letta blocks, LangGraph thread state | explicitly lossy; never mistaken for the source |
| Durable semantic records | facts, preferences, temporal state | RecMem facts, Mem0 memories, Graphiti edges, Claude/Codex notes | source links, validity, revision and deletion policy |
| Procedural records | reusable actions and checks | Voyager code skills, ExpeL insights, Codex/Letta skills | validation evidence, dependencies, versioning, revocation |

RecMem’s raw/episodic/semantic split shows why an unpromoted raw turn should
remain recallable.[recmem][recmem] Graphiti shows why derived facts should name
their supporting episodes and separate valid time from ingestion time.
[graphiti][graphiti] pi shows why a compact summary can be discarded and
regenerated without destroying the event tree.[pi][pi]

The layers can share storage technology, but not semantics. A vector index is
an access path, not a provenance model.

## 2. Separate persistence, compaction, and learning

Three questions should be answered independently:

- **Can the system resume?** A transcript or checkpoint is enough.
- **Can a long session continue?** A compact summary plus recent state may be
  enough.
- **Does a later independent session behave differently?** This requires a
  cross-session record and a read policy.

pi deliberately answers the first two but has no built-in third mechanism.
Claude Code auto memory and Codex consolidation answer the third by writing
plain files.[claude][claude] [codex][codex] Conflating these leads to claims
that an agent “learned” merely because it shortened its current prompt.

The same rule applies to metrics. Reduced context tokens demonstrate
compression efficiency. Higher future-task success can support system-level
learning. Neither demonstrates a weight update.

## 3. Make write triggers plural and explicit

Observed triggers occupy a spectrum:

- unconditional raw capture: Generative Agents observations, RecMem
  subconscious turns, Graphiti episodes;
- outcome feedback: Reflexion after a trial, Voyager after successful
  verification;
- accumulated importance: Generative Agents reflection;
- semantic recurrence: RecMem consolidation;
- batch comparison: ExpeL, Mistake Notebook, and ReMe;
- application events: LangGraph nodes or APIs; and
- asynchronous background extraction: Codex phase one and Letta sleep-time
  compute.

No one trigger is sufficient. Recurrence saves extraction cost but misses a
critical one-off. Success-only admission avoids known failures but cannot learn
from why attempts failed. Eager extraction captures one-offs but is expensive
and produces duplicates. Importance scores are uncalibrated predictions.

A safer policy combines:

1. cheap raw persistence for eligible events;
2. an immediate high-impact path for explicit user requests, security
   constraints, irreversible decisions, and verified corrections;
3. recurrence- or batch-triggered abstraction for ordinary patterns;
4. outcome-gated procedural promotion; and
5. periodic background review for contradiction, expiry, and orphaned
   derivations.

The trigger decision itself should be logged, including which rule fired and
which model or policy version made the judgment.

## 4. Rank by utility, not similarity alone

The systems demonstrate a useful menu of ranking signals:

- embedding relevance: nearly universal;
- lexical relevance: Mem0 and Graphiti;
- recency and access recency: Generative Agents;
- predicted importance: Generative Agents;
- usage count and last use: Codex consolidation;
- entity overlap and graph distance: Graphiti;
- temporal validity: Graphiti; and
- outcome or utility: Voyager’s success gate, ExpeL votes, ReMe refinement.

A production score should preserve these components separately rather than hide
them behind one opaque number. Hard filters—tenant, authorization, deletion
state, validity interval, and data class—must run before soft ranking.
Similarity is never permission.

Ranking also needs diversity and contradiction awareness. Top-\(k\) nearest
neighbors can return five paraphrases of the same stale fact. Graphiti’s
documented fusion and optional maximal marginal relevance provide useful access
patterns, but even a sophisticated ranker cannot repair a poisoned source.
[graphiti-search][graphiti-search]

## 5. Consolidate with merge, evidence, and no-op paths

Good consolidation is not “summarize everything.”

- RecMem searches for an existing episode before creating a new one.
- ExpeL can add, edit, upvote, downvote, and remove an insight.
- Graphiti keeps source episodes while invalidating temporal fact edges.
- Codex first normalizes rollouts independently, then serializes global
  consolidation; both stages allow low-signal no-ops.[expel][expel]
  [codex-stage][codex-stage]

The safest common shape is:

1. identify a candidate cluster;
2. retrieve its existing abstractions and contradictions;
3. generate a proposed patch rather than an unconstrained rewrite;
4. attach source record IDs and validation status;
5. no-op when evidence does not justify change;
6. test procedural memories before promotion; and
7. retain the prior version for rollback.

“Merge first” avoids duplicate summaries, but in-place rewriting can erase
nuance. Versioned patches and evidence-linked clauses are preferable to one
mutable prose blob.

## 6. Treat semantic, episodic, and procedural memory as schemas

The cognitive labels are useful only if they change behavior:

- episodic records need source time, actor, environment, outcome, and raw
  evidence;
- semantic records need normalized subject/predicate, validity, confidence,
  authority, and supporting episodes; and
- procedural records need preconditions, ordered actions, dependencies,
  expected outcome, verification, failure modes, and version.

Voyager comes closest to an executable procedural artifact but lacks a
paper-level conflict and revocation contract.[voyager][voyager] LangGraph names
all three categories while intentionally leaving schemas to applications.
[langgraph][langgraph] A free-form note labeled “semantic memory” does not gain
truth maintenance by name.

## 7. Model conflict as time and evidence

“Newest wins” is sometimes right for a changed address and wrong for an
uncorroborated correction. A useful record distinguishes:

- event or world-valid time;
- ingestion time;
- last verification time;
- source authority;
- supporting and contradicting evidence; and
- whether a newer record supersedes, narrows, or merely disagrees.

Graphiti’s `valid_at`, `invalid_at`, `created_at`, and `expired_at` fields
provide the strongest concrete starting point in this survey.
[graphiti-source][graphiti-source] Codex’s consolidation prompt combines
freshness with validation and preserves uncertainty when evidence is unclear.
[codex-consolidate][codex-consolidate] Mem0 v3’s add-only path is a cautionary
counterexample: retaining both changed facts is safe from accidental overwrite
but delegates the conflict to retrieval.[mem0][mem0]

Contradiction handling should produce an auditable state transition, not
silently edit the old sentence.

## 8. Define five different forms of forgetting

The word “delete” hides distinct operations:

1. **attenuation**: lower retrieval rank through decay;
2. **eviction**: drop an item from a bounded working buffer;
3. **selection pruning**: exclude low-use or stale records from an active
   consolidated set;
4. **logical deletion or invalidation**: suppress a record while retaining
   audit history; and
5. **physical erasure**: remove raw data, embeddings, derivations, histories,
   replicas, backups, and exports according to a declared contract.

Generative Agents implements attenuation; Reflexion uses buffer eviction;
Codex prunes selected inputs and exposes a local reset; Graphiti exposes graph
deletion plus temporal invalidation; Mem0 explicitly distinguishes decay,
reversible expiration, and deletion.[mem0-expiration][mem0-expiration] None of
those facts alone proves complete physical erasure across histories, replicas,
backups, and exports.

Derived-memory deletion is the hardest case. Managed Zep’s documentation warns
that deleting an episode may leave shared node summaries or invalidation effects.
[graphiti-delete][graphiti-delete] A complete system needs a derivation graph
and one of two contracts:

- cascade and regenerate every affected abstraction; or
- tombstone the source and rebuild the memory projection from remaining
  evidence.

Retention, user correction, and legal erasure should not share one ambiguous
`delete()` method.

## 9. Preserve provenance through every abstraction

Useful provenance minimally records:

- source record IDs and tenant/project scope;
- actor and ingestion channel;
- event and ingestion timestamps;
- extractor, embedding, and prompt versions;
- the trigger and consolidation job;
- validation evidence and outcome;
- prior versions and supersession edges; and
- every active index or skill derived from the record.

Graphiti episode links and Codex rollout-summary metadata are useful partial
models. Generative Agents reflection pointers show the same idea in a smaller
prototype.[generative][generative] ExpeL’s insight vote count is not enough:
it records aggregate model judgment but can lose which trajectory supports
which clause.

Source text must remain data during consolidation. Codex’s prompt explicitly
requires this for third-party rollout content.[codex-stage][codex-stage] That
is necessary but not sufficient against indirect prompt injection; ingestion
also needs taint labels, restricted consolidation tools, and review for
privileged procedural changes.

## 10. Treat project memory as multi-writer state

Several surveyed systems already expose pieces of the shared-project problem.
Claude Code’s auto memory is shared by worktrees of one repository, while an
ordinary subagent’s auto memory is separate.[claude][claude] Letta blocks can
be attached to and shared by multiple agents.[letta-blocks][letta-blocks]
LangGraph gives applications tuple namespaces but does not prescribe their
identity hierarchy.[langgraph][langgraph] Graphiti records a `group_id`, and
Codex associates consolidated tasks with working-directory and rollout
provenance.[graphiti-source][graphiti-source] [codex-consolidate][codex-consolidate]

A future shared-project design should use explicit identities at every layer:

- project, tenant, repository, and checkout or worktree;
- agent, process, and session;
- human or service principal that authorized the work;
- source record and consolidation job; and
- artifact scope: private scratch, agent-local, project-shared, or global.

Promotion into project-shared memory should be a separate event from writing
agent-local scratch. Otherwise one exploratory agent can silently turn an
unverified hypothesis into every other agent’s starting context.

### Storage locking is not semantic coordination

Codex demonstrates useful storage coordination: Phase 1 leases rollout jobs,
and Phase 2 takes one global lock before updating shared artifacts.
[codex][codex] Those mechanisms prevent duplicate extraction and concurrent
filesystem mutation. They do **not** establish that:

- two successful agents’ conclusions are compatible;
- the last writer had greater authority or fresher world-valid evidence;
- a shared procedure works in every represented checkout;
- a memory deleted by one agent should disappear for all others; or
- an agent’s generated summary faithfully represents another agent’s evidence.

Storage coordination needs leases, transactions or compare-and-swap versions,
idempotent jobs, parallel snapshot/proposal workers, one supervisor that
serializes physical commits, and CAS/fenced promotion. “One owner” applies to
publication, not to offline analysis branches. Semantic coordination needs
source-linked proposals, scope and authority rules, explicit
contradiction states, validation gates, and a policy for merge, supersession,
or human escalation. A mutex supplies none of the latter.

For shared artifacts, each write should carry the version it read. A stale
writer should rebase its proposed patch over intervening changes rather than
overwrite them. Conflicting claims should coexist as attributed alternatives
until evidence resolves them; “last write wins” is suitable only when that is
the declared domain rule.

Deletion also needs scope. Detaching a Letta block from one agent is different
from deleting a block shared by several agents. Removing a project fact must
invalidate every agent cache and derived summary that used it; deleting
agent-local scratch should not damage shared evidence. A tombstone plus
derivation-aware rebuild is safer than one process erasing a shared file while
others still hold the old prompt.

Finally, pi’s append-only session tree is strong within-session lineage, but
its documentation does not promise that several processes can safely append to
the same JSONL file.[pi][pi] Branch structure should not be mistaken for a
multi-writer transaction protocol.

## 11. Evaluate lifecycle properties, not just answer accuracy

Existing results mainly show task success, retrieval accuracy, believability,
or construction-token savings under bounded datasets. A long-running-agent
evaluation should separately measure:

- recall of rare one-off constraints;
- precision and contradiction rate of promoted facts;
- provenance completeness;
- time-to-correction after the world changes;
- stale-memory influence on actions;
- cascading deletion completeness;
- cross-user and cross-project isolation;
- poisoning resistance;
- procedural validation and rollback;
- construction, retrieval, and consolidation cost; and
- improvement over sessions with the same frozen base model.

RecMem’s ablation is especially instructive: the raw subconscious tier, not
the polished episode tier, produced the largest accuracy loss when removed.
[recmem][recmem] That is evidence against discarding raw data solely because a
summary exists. It is not permission to retain raw user data indefinitely.
Privacy retention and model utility are separate objectives.

## What the surveyed systems actually learn

| Category | Systems | Changing artifact |
| --- | --- | --- |
| Retrieval only or runtime persistence | core LangGraph store/checkpoints, pi sessions | saved state and selected prompt context |
| External semantic or episodic revision | Generative Agents, Reflexion, RecMem, Mem0, Graphiti, Claude Code auto memory, Codex | text records, facts, temporal edges, summaries |
| External procedural revision | Voyager, ExpeL/ReMe, Letta Code, Codex skills | programs, rules, prompts, skill packages |
| Parametric learning in the core memory path | none surveyed | no model weights change |

“External” does not mean trivial. These artifacts can durably alter future
behavior, and executable skills can be more operationally consequential than a
small weight update. The correct claim is precise: the **agent system adapts by
editing and retrieving non-parametric memory while the base model remains
fixed**.

## Practical design target

A defensible offline consolidation system should require:

- append-only, scope-isolated evidence;
- actor-, process-, project-, and authorization-scoped records;
- cheap raw indexing before abstraction;
- both recurrence and high-impact one-off triggers;
- two-stage extraction, parallel proposal branches, and serialized fenced
  publication;
- version-checked multi-writer proposals with explicit semantic conflict states;
- source-linked semantic and procedural patches;
- validation-gated skill promotion;
- hybrid retrieval behind authorization and validity filters;
- explicit uncertainty and temporal supersession;
- user-visible inspect, correct, export, and delete controls;
- derivation-aware regeneration after deletion;
- versioned rollback; and
- lifecycle evaluations alongside accuracy and token cost.

No surveyed system implements this complete contract. The synthesis is a design
target assembled from complementary strengths, not a claim that combining
their components automatically yields a safe lifelong agent.

## Local References

[claude]: Anthropic. “How Claude remembers your project,” Claude Code documentation. https://code.claude.com/docs/en/memory (accessed 2026-07-26).

[codex]: OpenAI. “Memories,” Codex source documentation. https://github.com/openai/codex/blob/main/codex-rs/memories/README.md (accessed 2026-07-26).

[codex-consolidate]: OpenAI. “Memory Writing Agent: Phase 2 Consolidation,” Codex source template. https://github.com/openai/codex/blob/main/codex-rs/memories/write/templates/memories/consolidation.md (accessed 2026-07-26).

[codex-stage]: OpenAI. “Memory Writing Agent: Phase 1,” Codex source template. https://github.com/openai/codex/blob/main/codex-rs/memories/write/templates/memories/stage_one_system.md (accessed 2026-07-26).

[expel]: Andrew Zhao et al. “ExpeL: LLM Agents Are Experiential Learners.” AAAI 2024. https://arxiv.org/abs/2308.10144 (accessed 2026-07-26).

[generative]: Joon Sung Park et al. “Generative Agents: Interactive Simulacra of Human Behavior.” UIST 2023. https://arxiv.org/abs/2304.03442 (accessed 2026-07-26).

[graphiti]: Preston Rasmussen et al. “Zep: A Temporal Knowledge Graph Architecture for Agent Memory.” 2025. https://arxiv.org/abs/2501.13956 (accessed 2026-07-26).

[graphiti-delete]: Zep. “Deleting Data from the Graph,” managed Zep documentation. https://help.getzep.com/deleting-data-from-the-graph (accessed 2026-07-26).

[graphiti-search]: Zep. “Searching the Graph,” Graphiti documentation. https://help.getzep.com/graphiti/working-with-data/searching (accessed 2026-07-26).

[graphiti-source]: Zep AI. `graphiti_core/edges.py`, Graphiti source. https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py (accessed 2026-07-26).

[langgraph]: LangChain. “Memory overview,” LangGraph documentation. https://docs.langchain.com/oss/python/concepts/memory (accessed 2026-07-26).

[letta-blocks]: Letta. “Memory Blocks,” V1 SDK documentation (legacy). https://docs.letta.com/v1-sdk/memory/memory-blocks (accessed 2026-07-26).

[mem0]: Mem0. “Open Source v2 to v3 Migration Guide.” https://docs.mem0.ai/migration/oss-v2-to-v3 (accessed 2026-07-26).

[mem0-expiration]: Mem0. “Memory Expiration in Mem0,” product and OSS documentation. https://docs.mem0.ai/platform/features/memory-expiration (accessed 2026-07-26).

[pi]: earendil-works contributors. “pi coding agent,” official source documentation. https://github.com/earendil-works/pi/blob/main/packages/coding-agent/README.md (accessed 2026-07-26).

[recmem]: Zijie Dai et al. “RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents.” Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.1619/ (accessed 2026-07-26).

[voyager]: Guanzhi Wang et al. “Voyager: An Open-Ended Embodied Agent with Large Language Models.” 2023. https://arxiv.org/abs/2305.16291 (accessed 2026-07-26).
