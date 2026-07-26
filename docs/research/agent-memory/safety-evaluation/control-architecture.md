# Control Architecture

## Status

Everything in this document is a **design proposal** unless a sentence is
explicitly attributed to a source. The cited papers motivate threat coverage;
they do not validate this architecture. The objective is to make unsafe state
transitions explicit, reviewable, and reversible.

## Architectural stance

Persistent memory is a data plane. It does not join the instruction hierarchy.
The model may propose writes and transformations, but a host-controlled policy
engine owns admission, promotion, retrieval authorization, capability
assignment, deletion, and rollback.

The same rule applies to peer agents. An authenticated agent may propose data or
request work; it cannot confer trust, transfer its authority through text, or
publish merely because it holds a database connection.

In the multi-process design, agents never hold a database connection. A trusted
Unix supervisor is the sole opener of the database, WAL, indexes, deletion
ledger, and keys. It authenticates Unix-socket peers and task/attempt-bound
challenge-response sessions.
Modes `0600`/`0700` are not isolation between same-UID processes; shared memory
stays disabled unless a sandbox or separate service identity prevents workers
and their tools from direct-open, copy, lock, symlink, and socket-replay bypass.

The core pattern is:

`immutable episode → typed candidate → independent checks → atomic promotion
→ capability-scoped retrieval → monitored use`.

Offline “dreaming” runs the same candidate path. It has no direct write access
to promoted memory, raw provenance, access policy, or executable procedure
stores.

## Stores and boundaries

| Store | Contents | Mutability | Model access |
|---|---|---|---|
| Episode log | Exact user/tool/environment events and metadata | Append-only; authorized tombstone/erasure | Read through scoped retrieval |
| Candidate quarantine | Proposed facts, summaries, reflections, predictions, procedures | Versioned until accept/reject/expire | Consolidator writes; verifier reads |
| Promoted fact graph | Defeasible claims, preferences, temporal state, contradiction sets | New revisions; no silent in-place overwrite | Read through scoped retrieval |
| Simulation namespace | Counterfactuals, dream traces, predicted outcomes | Expiring and isolated | Offline processes only by default |
| Procedure registry | Reviewed inert artifacts plus capability/environment manifests | Signed versions | Resolver may propose; runner executes |
| Provenance graph | Generation, derivation, revision, invalidation, policy decisions | Append-only | Read-only views |
| Coordination ledger | Agent instances, task envelopes, delegations, leases, fencing counters, idempotency keys, current epochs | Host-controlled and transactional | No model write |
| Authorization policy | Tenant, purpose, sensitivity, role, capability rules | Operator-controlled versioning | Never writable from memory |
| Audit ledger | Reads, decisions, tool effects, deletion and rollback receipts | Append-only and integrity-protected | No model write |

W3C PROV-O supplies a general vocabulary for entities, activities, agents,
generation, derivation, primary sources, revisions, and invalidation
[provo13][provo13]. The schema below extends that vocabulary with agent-specific
source, authority, sensitivity, and capability fields; it is not a claim of
PROV-O conformance.

## Record schema

Every record should carry at least:

```text
record_id, version_id
tenant_id, principal_scope, project_scope, purpose_scope
visibility_scope, owner_principal, sharing_decision
source_class, origin_principal, origin_system, origin_locator
actor_agent_id, actor_run_id, task_id, delegation_id
content_hash, captured_bytes_ref, extraction_method
payload_disposition, admission_policy, span_provenance[]
artifact_dek_id, artifact_kek_id, scope_kek_id, key_state
event_time, recorded_time, valid_from, valid_until, checked_at
sensitivity, retention_class, consent_or_policy_basis
instruction_authority, capability_ceiling
derived_from[], primary_sources[], contradicts[], revises[]
candidate_state, promotion_decision, policy_version
verifier_results[], confidence_vector, calibration_version
expires_at, tombstone_state, lineage_epoch
expected_parent_epoch, fencing_token, idempotency_key
causal_root_id, upstream_artifact_version, capture_quality
```

`event_time` says when the asserted event occurred; `recorded_time` says when
the system learned it. `valid_from` and `valid_until` describe the modeled
fact’s interval. Conflating these fields makes stale records appear current and
makes late-arriving corrections indistinguishable from malicious overwrites.

### Evidence roots

`primary_sources` must resolve to provenance roots, not other generated
summaries. A support count is the number of policy-eligible independent causal
roots. Artifact/version correlation IDs collapse copies that arrive through
different agents, URLs, tools, or summaries; unknown dependence counts
conservatively as one. Two descendants of one episode count once. A reflection
cannot corroborate its own ancestor. A cycle in `derived_from` is invalid.

### Confidence is a vector

Do not collapse these into one opaque score:

- source reliability for the specific claim type;
- extraction or parsing uncertainty;
- number and diversity of independent supporting roots;
- contradiction mass and source authority;
- temporal freshness and validity;
- verifier agreement;
- calibration cohort and uncertainty interval.

Embedding score, retrieval frequency, model verbal certainty, and the number of
descendants are never evidence-strength components.

## Candidate-to-promotion transaction

### 1. Capture

Before durable storage, trusted host admission drops excluded fields, replaces
ordinary secrets with typed redactions or secret-manager references, and routes
explicit exceptional retention to a separately encrypted store. Then store the
exact **admitted canonical view** before asking a model to interpret it. Each
payload/span is marked `retained`, `redacted`, `secret_ref`, or `omitted`;
low-entropy sensitive values never enter ordinary hashes, only supervisor-keyed
opaque correlation IDs where needed. Bind tool identity, admitted arguments and
response bytes, endpoint, execution status, tenant, and time. Treat all content
as data regardless of transport authentication.

For inter-agent input, also bind the host-authenticated logical agent,
process-instance identity, task envelope, delegation reference, and message
hash. Sender identity proves who sent the message, not that its contents are
instructions or facts.

Origin is span-level as well as record-level. Copying or paraphrasing imported
content into a user, assistant, tool, or peer message retains transformation
lineage; mixed or untraceable spans inherit the least-authoritative applicable
class.

### 2. Type

Assign source class, claim type, sensitivity, possible instruction content,
tenant/project/purpose scope, private-versus-shared visibility, and retention
class. Same-project does not imply shared. If any required field is unknown,
use the more restrictive class and quarantine.

### 3. Propose

The consolidator emits a structured candidate with:

- exact claim or procedure;
- source-class ceiling;
- evidence roots and relevant spans;
- proposed validity and scope;
- known counterevidence;
- expected future utility;
- risk tier; and
- machine-checkable tests where possible.

The proposer cannot choose its own final authority or capability.

### 4. Deterministic validation

Reject or quarantine candidates with:

- missing or cross-tenant evidence;
- provenance cycles or invalid hashes;
- generated material presented as observation;
- capability or sensitivity escalation;
- invalid temporal intervals;
- references to deleted/tombstoned roots;
- unbounded procedure inputs, output sinks, or endpoints;
- recursive lineage beyond the configured depth; or
- malformed schemas and non-idempotent migration.

### 5. Independent verification

Verification must be independent in **evidence**, not merely a second sample
from the same model. Depending on claim type:

- rerun an authoritative read-only tool against a fresh source;
- compare against signed configuration or a user-confirmed preference;
- replay the raw trace in a clean sandbox;
- run executable assertions and negative tests;
- retrieve counterevidence from a separately selected evidence set;
- use a different model only as a fallible reviewer, never as independent fact;
- require human approval for high-impact preferences, credentials, endpoint
  changes, or procedures.

The finding that intrinsic self-correction often failed in the reasoning tasks
studied by Huang and colleagues makes model-only self-review weak evidence
[selfcorrect24][selfcorrect24]. It does not imply that every multi-model review
is useless; it implies that external evidence and deterministic checks must be
distinguished from another opinion.

### 6. Contradiction and temporal analysis

Compare the candidate against all active and recent claims for the same
entity/property/scope, not only nearest neighbors. Outcomes are:

- `compatible`: add support or narrower scope;
- `historical_transition`: close an old validity interval and add a new one;
- `authoritative_revision`: link `revises`, retaining prior history;
- `unresolved_contradiction`: create or extend a contradiction set;
- `suspected_poison`: quarantine and alert;
- `duplicate_lineage`: retain one support root, not another vote.

Never use newest-wins unless the property has an authenticated, total ordering
that defines the newest record as authoritative. A newer webpage is not
automatically more authoritative than an older user-approved configuration.

### 7. Policy and risk decision

The policy engine decides whether the system may retain the content, for which
purpose and duration, and at what promotion tier:

- `episodic_only`;
- `promoted_fact`;
- `promoted_preference`;
- `prediction_pending`;
- `approved_procedure`;
- `reject`;
- `needs_user`;
- `needs_operator`.

Content cannot modify the policy used to judge it. The decision records policy
version and all check results.

### 8. Canary and regression evaluation

Before publishing a consolidation batch, run relevant clean and adversarial
tests on a snapshot:

- current task utility;
- abstention when unsupported;
- direct, compositional, and dormant poison triggers;
- privacy and cross-tenant canaries;
- procedural negative tests and capability denial;
- deletion and rollback preservation;
- latency, prompt size, and retrieval budget.

A high-risk batch that lacks required tests fails closed. Low-risk candidates
may be deferred; incomplete evaluation is not a pass.

### 9. Atomic promote

Publish candidate versions, indexes, contradiction updates, and the rollback
manifest as one transaction. Readers see either the old epoch or the new epoch.
If index construction or verification fails, the old epoch remains current.

## Multi-agent identity, delegation, and database coordination

This section is a **design proposal** for the case where several agents or
processes share one project memory database.

### Identity and cross-agent messages

Give each logical agent a stable `agent_id` and each process incarnation a
unique `run_id`. A host-authenticated task envelope binds both to tenant,
project, principal, purpose, task, policy version, and starting capability. Do
not accept an agent name, role, or delegation asserted only inside message or
memory text.

Every inter-agent message enters as `imported_content` with its exact envelope
and sender recorded. It may be evidence that the sender made an assertion. It
is not automatically:

- a current user instruction;
- an authoritative fact;
- consent to retain or share data;
- an independent verifier decision; or
- a transferable tool capability.

The receiver applies the same candidate, source, authorization, and
instruction-isolation checks used for files and tool output.

### Private and shared project scopes

At minimum, distinguish:

- private to a principal;
- private to a task or logical agent;
- shared with a named set of principals/agents; and
- shared project-wide.

Default to the narrow scope established by capture. A private-to-shared change
is a new, authorized promotion decision that records the owner, requested
audience, purpose, sensitivity review, policy version, and evidence. Copying
content into a shared summary does not bypass that decision. Project sharing
never widens tool capability or turns one agent’s private conversation into
another agent’s instruction.

### Attenuated delegation and confused deputies

A host-issued delegation object should bind:

```text
delegation_id, issuer, subject_agent_id, subject_run_id
tenant, project, task, audience, purpose
allowed_operations, resource_predicates, tool_capabilities
not_before, expires_at, max_redelegation_depth
parent_delegation_hash, policy_version, revocation_epoch
```

A child delegation is valid only if every dimension is a subset or stricter
restriction of its parent. Effective authority is the intersection of the
receiver’s current host authorization, the complete valid delegation chain,
current user intent, task purpose, and current policy. Memory text cannot add a
grant. Delegated credentials are audience-bound and non-transferable.

“Subset” is defined over a closed host-owned algebra, never arbitrary model
predicates. File grants use pre-opened handles, allowed operations, and
no-follow resolution; network grants use canonical scheme/host/port/audience
tuples rechecked after DNS and every redirect; project, worktree, recipient,
and credential audiences are opaque registered handles. Each type has a
deterministic subset function and explicit deny precedence. Wildcards, aliases,
symlinks, redirects, and message text cannot widen a grant.

Before acting for a peer, the receiver checks both that the sender was allowed
to request the operation and that the receiver is allowed to perform it for
that sender, task, purpose, and resource. Checking only the receiver’s stronger
capability creates a confused deputy. Revocation and expiry are rechecked at
effect time, not only when the message was received.

### Supervisor and SQLite coordination

Only the trusted supervisor owns SQLite connections. SQLite ordinarily
isolates separate connections, allows only one concurrent
writer, and serializes writes. In WAL mode, readers can continue on an older
snapshot while a writer commits [sqliteisolation26][sqliteisolation26]
[sqlitetransactions26][sqlitetransactions26]. This prevents certain torn or
interleaved database writes. It does **not** establish:

- which agent or delegation authorized a write;
- whether a lease holder is still current;
- whether an old snapshot may publish;
- which concurrent semantic update should win;
- whether a record is private or project-shared;
- whether deletion or revocation dominates promotion; or
- whether a retried tool effect is a duplicate.

Use SQLite transactions to enforce explicit semantic preconditions. A
publication transaction should read current tombstone, revocation, sharing,
parent-epoch, and fencing state and conditionally advance the epoch only if all
still match. `BEGIN IMMEDIATE` can acquire the single writer slot before those
checks when a write is planned; it is not itself the policy.

The supervisor boundary is tested with direct-open, forged-row, symlink,
WAL-copy, lock-denial, socket-replay, and stolen-token attempts. If worker tools
share its filesystem authority, database rows are not protected from the model
and multi-agent mode fails closed.

Supervisor sessions use challenge-response, Unix peer PID plus process-start
identity, socket-session nonce, monotonic request sequence/MAC, and expiry.
A principal/project-bound controller session authorizes capture and explicit UI
memory operations but no worker fence. A distinct worker session binds logical
agent/run, project/task/attempt, fencing generation, capability set, and policy
epoch. Session material is absent from prompts, environment, argv, logs, and
tool subprocesses; descriptors are close-on-exec. Lease loss/cancellation
revokes the worker session; exit, policy change, or expiry revokes either type.

### Leases, fencing, retries, and races

A lease is a liveness hint. A paused process can resume after expiry while
still believing it is leader. Assign a monotonically increasing fencing token
whenever publication ownership changes. Every promotion, index swap, procedure
publish, and destructive maintenance operation includes the token; the commit
checks it against the current value in the same transaction. A stale token is
rejected even if its process still holds local state.

Also require:

- compare-and-swap on the expected parent epoch or record version;
- a unique idempotency key for each logical write and effect request;
- deterministic handling of duplicate delivery and `SQLITE_BUSY` retries;
- revalidation after a losing worker rebases on a newer epoch;
- no automatic merge of concurrent facts, sharing decisions, or procedures;
- a monotonic tombstone and revocation ledger outside ordinary rollback;
- promotion failure if any evidence root is tombstoned at commit;
- deletion or scope narrowing to dominate a concurrent promotion or share; and
- durable effect intent plus an idempotent tool key or explicit reconciliation,
  because a SQLite transaction cannot atomically commit a remote tool effect.

Concurrent candidate creation may be harmless when candidates remain
quarantined and deduplicated by lineage. Concurrent promotion is a state-machine
decision: only the transaction whose parent epoch and fencing token remain
current may publish. Losing work is stale, not implicitly approved.

## Why admission filtering is insufficient

AgentPoison and MINJA establish persistent poisoning through different attacker
capabilities [agentpoison24][agentpoison24] [minja25][minja25]. The recent
MemPoison preprint adds an especially relevant structural result: its L2 records
can be benign in isolation and harmful when jointly retrieved, while L3 records
activate in a later trigger context [mempoison26][mempoison26]. Its pointwise
write defenses reduced direct attacks more effectively than these cases.

The architecture therefore adds:

- set-level retrieval checks;
- trigger-context re-evaluation before high-impact use;
- counterfactual removal diagnostics during testing;
- limits on how many untrusted roots can jointly influence a response;
- no authority aggregation from multiple low-authority records; and
- tool-side capability enforcement even if model-level checks fail.

These are design responses, not demonstrated complete defenses.

## Retrieval control

### Authorize, then rank

There is no global FTS corpus whose unauthorized rows influence IDF, returned
content, cache keys, or query planning. The first implementation maintains
separate indexes for exact security domains. Multi-domain queries rank each
authorized domain independently and merge bounded results with a deterministic
host rule; every statistic and cache carries the same domain key. Shared
supervisor/SQLite/CPU/I/O timing remains a measured residual side channel, not
an exact noninterference claim. Strong timing isolation would require separate
process/database/resource domains plus a declared padding/admission regime.

The retrieval pipeline is:

1. authenticate principal and resolve tenant/project/purpose;
2. authenticate the acting agent instance and validate its task/delegation
   chain;
3. filter by private/shared audience, access, consent/policy, sensitivity,
   source class, validity, and deletion state;
4. determine allowed memory types and capability ceiling for the task;
5. rank the already authorized set;
6. enforce top-k, token, source-diversity, and contradiction budgets;
7. attach provenance and conflict metadata;
8. render memories as quoted data with explicit class labels; and
9. immediately before provider I/O, recheck retrieval/tombstone/revocation/
   policy epochs and abort/re-render a stale prompt; and
10. log actor, delegation, record IDs, ranking reasons, policy version,
    provider-send state, and downstream use.

Filtering after a global similarity search can leak through content, scores,
timing, or cache state. Tenant keys must be present in primary storage,
secondary indexes, caches, deduplication, offline queues, and telemetry.

### Retrieval bundles

Return a bundle, not a flat string:

```text
claim
status: supported | stale | disputed | prediction | simulation
source class and origin
validity interval and last check
supporting roots
counterevidence / contradiction set
allowed use: inform | ask-user | plan-only | procedure-reference
why retrieved
```

When the query asks for current state, historical claims may be included as
history but not silently substituted. When uncertainty exceeds the calibrated
action threshold, retrieval should cause abstention, a fresh tool check, or a
user question.

### Set-level instruction isolation

All retrieved memory remains untrusted as instruction. The prompt renderer
separates host instructions, current user instructions, memory claims, raw
evidence, and tool data. XML tags or labels are clarity aids, not a security
boundary. CaMeL’s preprint illustrates a stronger principle: trusted control
flow should be separated from untrusted data, and capabilities should constrain
data exfiltration [camel25][camel25]. CaMeL’s AgentDojo result does not prove
this memory design secure; it motivates keeping control flow and capability
assignment out of free-form retrieved text.

## Temporal validity, contradiction, and decay

### Contradiction sets

A contradiction set preserves alternatives with:

- normalized subject/property/scope;
- each claim’s value and interval;
- source authority by claim type;
- support and counterevidence roots;
- whether the conflict is logical, temporal, scoped, or merely uncertain;
- resolution status and resolver identity.

Resolution adds a new decision record; it does not erase losing evidence unless
retention or deletion policy separately requires that.

### Freshness

Freshness policy is field-specific. A package version, access token, endpoint,
calendar state, address, user preference, and mathematical theorem have
different invalidation behavior. The runtime should support:

- explicit `valid_until`;
- tool-specific refresh intervals;
- event subscriptions where available;
- refresh-on-consequential-use;
- stale rendering and abstention;
- revocation events that bypass ordinary batch schedules.

### Decay and forgetting

Decay lowers retrieval priority or triggers review; it does not rewrite origin
or evidence. Suggested factors are age, last validated time, task utility,
redundancy across independent roots, sensitivity cost, contradiction, and user
retention choices. Rare high-impact records should not disappear simply
because they are infrequent. Deletion, expiry, archival, and decay are distinct
state transitions.

MemoryAgentBench explicitly includes selective forgetting/conflict scenarios,
which is useful capability coverage [memoryagentbench26][memoryagentbench26].
It does not determine a safe production decay function.

## Confidence and calibration

For each claim type and action-risk tier:

- create held-out cases with known truth, staleness, contradiction, and
  unsupported conditions;
- record the system’s probability or support score before observing outcomes;
- report Brier score, log loss where appropriate, expected calibration error
  with bin definitions, and reliability diagrams;
- report risk-coverage curves for abstaining systems;
- stratify by source class, age, contradiction, generated/observed origin, and
  retrieval depth;
- calibrate action thresholds separately from answer thresholds; and
- freeze the calibrator before the final evaluation.

An uncalibrated numeric confidence should be treated as metadata for analysis,
not permission for consequential action.

## Procedures and skills

Agent Workflow Memory shows that reusable workflows can be induced and reused
[awm25][awm25]. MPBench treats experience-to-procedure synthesis as a distinct,
high-impact write path [untrusted26][untrusted26]. The procedure registry must
therefore be stricter than factual memory.

An approved procedure needs:

- immutable source trace and candidate version;
- explicit preconditions, postconditions, and failure modes;
- parameter schema with no embedded secrets;
- allowed tools, methods, paths, domains, recipients, and resource ceilings;
- denied capabilities and negative tests;
- environment/version fingerprint and expiry;
- deterministic tests plus sandbox replay;
- human approval for high-impact effects;
- signature over artifact and manifest; and
- runtime confirmation for effects that normally require it.

Procedure text cannot mint credentials, widen paths, add network recipients, or
override current user intent. Retrieval only selects a procedure candidate;
the runner independently checks the manifest and current authorization.

## Offline dreaming

### Allowed jobs

- replay immutable episodes to test retrieval;
- propose summaries, links, contradiction candidates, or refresh requests;
- generate counterfactual test cases in the simulation namespace;
- predict an outcome with an evaluation deadline;
- propose a procedure for sandbox testing;
- run canary and adversarial regression suites.

### Forbidden direct effects

- writing promoted facts or user preferences;
- relabeling generated material as observation;
- using a generated item to corroborate its ancestor;
- granting tool capability;
- importing another tenant’s episode;
- deleting or rewriting raw evidence;
- training model weights without a separate, explicit governance process; or
- publishing a partial batch.

### Counterfactual lifecycle

A counterfactual remains `counterfactual`. If later observation matches it, the
new observation is stored separately and linked by `evaluates_prediction`.
Prediction accuracy may update the simulator’s calibration; it does not convert
the generated trace into a historical event.

### Episodic-only baseline

Useful Memories Become Faulty reports that raw episodic management often
matched or exceeded its natural-language abstraction systems, and that forced
continuous consolidation could amplify error [usefulmem26][usefulmem26].
Although limited to the paper’s evaluated tasks and models, this makes an
immutable episodic-only system a required baseline and a safe fallback.
Consolidation must demonstrate incremental value over it.

## Privacy and minimization

### Before storage

- classify secrets, credentials, authentication artifacts, health/financial
  data, third-party data, and user-designated exclusions;
- default-deny raw secret values from semantic or procedural promotion;
- store references to secret managers rather than secrets where possible;
- collect only fields needed for a declared purpose;
- keep tenant and purpose tags inseparable from content; and
- avoid model-visible logs of raw sensitive values.

### During retrieval

- require principal, tenant, purpose, and task risk;
- minimize fields and redact at the data layer;
- prevent secret-bearing records from entering generic generation contexts;
- use non-secret canaries to detect boundary failures;
- rate-limit and audit probing patterns;
- make export and sharing separate authorized operations.

MEXTRA’s black-box extraction results make output-only safety review
insufficient [mextra25][mextra25]. Retrieval and store policies must be tested
under adaptive extraction attempts.

### User controls

The user interface should expose:

- what is stored and its source class;
- evidence, transformations, current status, and why it was retrieved;
- validity and retention;
- contradiction and uncertainty;
- edit as an explicit revision;
- delete with expected affected derivatives and backup handling;
- pause capture, retrieval, or offline consolidation; and
- export in a form that preserves provenance and scope.

Review rendering is itself a hostile-input boundary. Host-owned chrome keeps
source class, scope, status, and action keys outside evidence text. The renderer
escapes ANSI/OSC (including OSC-52), control bytes, and deceptive hyperlinks;
visibly annotates bidirectional and invisible Unicode; bounds excerpts; and
requires a separate gated raw-view/copy action. Adversarial PTY fixtures cover
escape sequences, bidi filenames, zero-width text, huge payloads, and fake
approval/status chrome.

GDPR may require access, rectification, and erasure in applicable contexts
[gdpr16][gdpr16]. The interface is still useful outside that legal scope.

## Deletion and rollback

### Deletion

An ordered two-store protocol:

1. authorizes and allocates an idempotent `tombstone_id`;
2. appends a hash-chained authenticated record at the next monotonic sequence
   and fsyncs the separate ledger plus directory;
3. commits one short SQLite transaction that applies the lineage tombstone,
   freezes derivations/effects, excludes reads, stops affected jobs, creates the
   purge manifest, and advances `applied_ledger_high_water`; and
4. acknowledges only after the database commit.

Every ordinary start and restore verifies the ledger chain/signature and
compares ledger high water `L` with database-applied high water `D` before any
read. `D < L` idempotently replays missing tombstones. `D > L`, invalid or
conflicting IDs, or an external anchor expecting a missing tail fails closed.
Crashes before ledger fsync, between ledger and database commit, and after
commit but before reply are separate modeled and fault-injected states.

Resumable idempotent jobs then traverse raw records, promoted derivatives,
simulations, procedures, embeddings, lexical indexes, caches, exports, pending
jobs, and backups. Canonical payloads use a random per-artifact DEK and
independently erasable artifact KEK:
`payload_ciphertext = AEAD_DEK(payload)`,
`wrapped_DEK = AEAD_artifact_KEK(DEK)`, and
`wrapped_artifact_KEK = AEAD_scope_KEK(artifact_KEK)`. Only ciphertexts,
wrapped keys, and IDs persist. Inventory records bind every database, spill,
export, and backup copy to its key ID and
`active|destroy_pending|destroyed` state. Record deletion destroys the artifact
KEK and fsyncs its destroyed ID to the external ledger; whole-scope erasure
destroys the scope KEK; the ledger prevents a restored key backup from reviving
destroyed IDs. Exceptional
sensitive payloads are never indexed. Purge jobs apply
SQLite/FTS secure-delete, temporary-file, checkpoint/WAL-truncation, and vacuum
procedures; rebuild indexes/contradictions; and scan raw database pages,
WAL/SHM, FTS shadow tables, temporary files, and restored backups for synthetic
canaries. The receipt lists every tier as completed, deferred, excepted, failed,
or impossible. No cross-tier atomicity is claimed.

Deletion also advances the monotonic tombstone epoch. A concurrent promoter
must recheck this epoch and every referenced root inside its commit; a writer
that started before deletion cannot publish a descendant afterward. Scope
narrowing and delegation revocation use the same dominance rule.

Every rendered prompt and effect intent carries contributing memory IDs and
epochs. Deletion or revocation cancels an affected local turn when possible and
is rechecked immediately before tool authorization and effect dispatch. Data
already sent to a provider or external system cannot be recalled; the receipt
records that exposure explicitly.

Backups may require expiry or key destruction rather than immediate physical
rewrite. The system must state this instead of claiming instantaneous universal
erasure. Restore serves no memory until the external signed ledger reaches its
expected high-water mark; missing, stale, corrupt, or partially restored ledgers
fail closed. Without a separate recovery anchor, the system claims only
live-store logical deletion. If records trained model weights, external-store
deletion does not establish unlearning. Append-only audit records retain only
non-sensitive or independently erasable identifiers, never deleted payloads.

### Correction

A correction creates a revision linked to the original, updates validity and
contradiction state, and atomically marks downstream derived objects
`needs_revalidation`. A descendant supported only by invalidated roots is
immediately excluded; mixed support is recomputed before serving. A material
user edit is a new user-authored assertion and cannot inherit old verifier
approval. This does not falsify the historical fact that the old assertion was
stored or used.

### Rollback

Every promotion batch produces:

- parent epoch and candidate manifest;
- content and index hashes;
- policy/model/prompt/tool versions;
- additions, invalidations, and deletions;
- provenance and contradiction changes;
- test results;
- known downstream actions.

Rollback switches readers to the parent epoch, invalidates poisoned additions,
rebuilds indexes, cancels pending procedures, and preserves an incident record.
User-requested deletions, sharing restrictions, and capability revocations must
not be undone by operational rollback.

## Operational controls

- Default-off consolidator with per-tenant enablement.
- Per-source write, candidate, and token quotas.
- Maximum derivation depth and batch fan-out.
- Two-person rule for policy changes and high-impact procedure promotion.
- Immutable policy and model-version capture.
- Shadow mode and canary tenants before wider release.
- Alerts for promotion spikes, lineage fan-out, contradiction growth, unusual
  secret access, cross-scope denials, and deletion resurrection.
- Alerts for invalid agent instances, delegation amplification, stale fencing
  tokens, repeated idempotency keys, epoch conflicts, and private-to-shared
  transitions.
- Emergency stop for retrieval, procedure execution, and consolidation
  independently.
- Restore drills that include poison tombstones and deletion state.

NIST AI RMF’s Govern/Map/Measure/Manage cycle and the NIST Generative AI Profile
support continuous monitoring and explicit risk ownership
[nistairmf23][nistairmf23] [nistgenai24][nistgenai24]. NIST SP 800-53 Rev. 5
offers broader access enforcement, audit protection, shared-resource, and
information-management controls [nist80053][nist80053]. These are governance
anchors, not memory-specific assurance or product certification.

## Safety properties to formalize

These properties are assigned across `MemoryRuntime.tla` and
`CollaborationRuntime.tla`, with prompt/effect integration schedules against
the existing `AsyncRuntime.tla`:

- **Tenant noninterference:** no returned record has a tenant outside the
  authenticated authorized set.
- **Actor authenticity:** every state transition is bound to a
  host-authenticated agent instance and task.
- **Delegation attenuation:** effective child authority is a subset of every
  ancestor grant and current host policy.
- **Private-by-default sharing:** same-project access does not imply visibility;
  widening scope requires an authorized transition.
- **Origin preservation:** a generated record never transitions to an observed
  source class.
- **No self-corroboration:** every counted support root is acyclic and
  independent under the declared relation.
- **Capability monotonicity:** memory-derived capability is a subset of current
  host authorization.
- **Promotion atomicity:** readers observe only fully committed epochs.
- **Fenced publication:** no state transition with a token older than the
  current project fence can commit.
- **Idempotent host intent:** replaying the same logical operation cannot
  duplicate a host authorization, state transition, or durable effect intent.
  A remote effect is exactly-once only when its provider honors the idempotency
  key; otherwise the modeled outcomes include `sent_unknown`, reconciliation,
  and a possible duplicate rather than asserting an impossible guarantee.
- **Tombstone persistence:** a deleted lineage cannot re-enter an active epoch
  through consolidation, reindex, or restore.
- **Deletion precedence:** a promotion derived before a concurrent tombstone
  cannot publish after that tombstone.
- **Rollback respects deletion:** switching epochs cannot revive a
  user-deleted record.
- **Procedure mediation:** every external effect is checked against the current
  signed procedure manifest and principal authorization.

Formalizing these properties can find state-machine defects. It cannot prove
that an LLM summary is true, an access policy is legally correct, or every
implementation component matches the model.

## Local References

[provo13]: World Wide Web Consortium. *PROV-O: The PROV Ontology.* W3C Recommendation (2013). https://www.w3.org/TR/prov-o/

[sqliteisolation26]: SQLite Project. “Isolation In SQLite.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/isolation.html

[sqlitetransactions26]: SQLite Project. “Transaction.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/lang_transaction.html

[selfcorrect24]: Huang, Jie; Chen, Xinyun; Mishra, Swaroop; Zheng, Huaixiu Steven; Yu, Adams; Song, Xinying; Zhou, Denny. “Large Language Models Cannot Self-Correct Reasoning Yet.” *International Conference on Learning Representations* (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html

[agentpoison24]: Chen, Zhaorun; Xiang, Zhen; Xiao, Chaowei; Song, Dawn; Li, Bo. “AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.” *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html

[minja25]: Dong, Shen; Xu, Shaochen; He, Pengfei; Li, Yige; Tang, Jiliang; Liu, Tianming; Liu, Hui; Xiang, Zhen. “Memory Injection Attacks on LLM Agents via Query-Only Interaction.” *Advances in Neural Information Processing Systems 38* (NeurIPS 2025). https://papers.nips.cc/paper_files/paper/2025/file/42a97bbd9844d2bf68596730af80bcdf-Paper-Conference.pdf

[mempoison26]: Gao, Jifeng; Xia, Kang; Zhang, Yi; Hong, Xiaobin; Lin, Mingkai; Wei, Xingshen; Li, Wenzhong; Lu, Sanglu. “MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents.” arXiv:2607.14651v1, preprint (2026). https://arxiv.org/abs/2607.14651

[camel25]: Debenedetti, Edoardo; Shumailov, Ilia; Fan, Tianqi; Hayes, Jamie; Carlini, Nicholas; Fabian, Daniel; Kern, Christoph; Shi, Chongyang; Terzis, Andreas; Tramèr, Florian. “Defeating Prompt Injections by Design.” arXiv:2503.18813v2, preprint (2025). https://arxiv.org/abs/2503.18813

[memoryagentbench26]: Hu, Yuanzhe; Wang, Yu; McAuley, Julian. “Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.” *International Conference on Learning Representations* (ICLR 2026). https://openreview.net/forum?id=DT7JyQC3MR

[awm25]: Wang, Zora Zhiruo; Mao, Jiayuan; Fried, Daniel; Neubig, Graham. “Agent Workflow Memory.” *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267 (2025). https://proceedings.mlr.press/v267/wang25bx.html

[untrusted26]: Dash, Pritam; Ge, Tongyu; Jain, Aditi; Shah, Tanmay; Shang, Zhiwei. “From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents.” arXiv:2606.04329v2, preprint (2026). https://arxiv.org/abs/2606.04329

[usefulmem26]: Zhang, Dylan; Lin, Yanshan; Wu, Zhengkun; Sun, Yihang; Li, Bingxuan; Li, Dianqi; Peng, Hao. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1, preprint (2026). https://arxiv.org/abs/2605.12978

[mextra25]: Wang, Bo; He, Weiyi; Zeng, Shenglai; Xiang, Zhen; Xing, Yue; Tang, Jiliang; He, Pengfei. “Unveiling Privacy Risks in LLM Agent Memory.” *Proceedings of ACL 2025*, 25241–25260. https://aclanthology.org/2025.acl-long.1227/

[gdpr16]: European Parliament and Council. *Regulation (EU) 2016/679 (General Data Protection Regulation).* Official Journal of the European Union (2016). https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng/

[nistairmf23]: Tabassi, Elham. *Artificial Intelligence Risk Management Framework (AI RMF 1.0).* NIST AI 100-1 (2023). https://doi.org/10.6028/NIST.AI.100-1

[nistgenai24]: Autio, Chloe; Schwartz, Reva; Dunietz, Jesse; Jain, Shomik; Stanley, Martin; Tabassi, Elham; Hall, Patrick; Roberts, Kamie. *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile.* NIST AI 600-1 (2024). https://doi.org/10.6028/NIST.AI.600-1

[nist80053]: Joint Task Force. *Security and Privacy Controls for Information Systems and Organizations.* NIST SP 800-53 Rev. 5, Release 5.2.0 (2025). https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final
