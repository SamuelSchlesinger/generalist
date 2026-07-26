# Longitudinal Safety and Utility Evaluation Plan

## Decision this plan is designed to support

This plan asks a narrower question than “does memory help?”:

> Under a fixed model, tool set, authorization policy, and resource budget,
> does guarded consolidation improve longitudinal task performance over an
> immutable episodic-only system without creating unacceptable poisoning,
> provenance, privacy, procedural, deletion, or rollback risk?

No existing benchmark answers that question alone. MemoryAgentBench tests four
important competencies under incremental interaction
[memoryagentbench26][memoryagentbench26]. PM-Bench tests prospective intentions
and their updates [pmbench26][pmbench26]. LongMemEval-V2 tests evidence use over
long web-agent histories [longmemevalv226][longmemevalv226]. Existing poisoning
and prompt-injection work supplies attack mechanisms, but does not establish
the safety of this architecture [agentpoison24][agentpoison24]
[minja25][minja25] [untrusted26][untrusted26] [hidden26][hidden26]
[mempoison26][mempoison26] [badmemory26][badmemory26].

Everything below that specifies a harness, metric, ablation, or release rule is
a **design inference**. Cited papers establish benchmark or attack scope, not
that the proposed protocol is sufficient.

## Claims and falsifiers

Pre-register each claim, comparison, population, metric, estimator, and failure
threshold before examining final test results.

### C1: useful consolidation

Guarded consolidation improves prespecified longitudinal utility over the
immutable episodic-only baseline at comparable latency, token, and storage
budgets.

Falsifiers include:

- no material paired improvement over episodic-only memory;
- an improvement that disappears under equal retrieval or context budgets;
- a gain only over no-memory;
- worse current-state accuracy despite higher historical recall; or
- utility that depends on generated records being treated as observations.

### C2: evidence-preserving consolidation

Promoted memories remain traceable to correctly typed, independent evidence
roots and do not become more confident merely because one source was repeatedly
summarized.

Falsifiers include:

- a generated item labeled as observed;
- a promoted claim with no reachable admissible evidence root;
- support counts inflated by descendants of one root;
- a promotion decision that cannot be replayed from its recorded policy and
  verifier inputs; or
- an unsupported summary becoming more authoritative after repeated dream
  cycles.

### C3: bounded poisoning

Guarded consolidation reduces promotion, retrieval, behavioral influence, and
persistence of direct, query-only, indirect, compositional, and dormant
poisoning without achieving safety by rejecting most benign memories.

Falsifiers include:

- the attack reaches a prohibited tool effect;
- poison survives correction, deletion, or rollback;
- generated dreams amplify the poisoned claim;
- the defense only moves the failure to a later cycle or trigger; or
- clean utility or benign write acceptance collapses.

### C4: calibrated uncertainty and time

The system distinguishes supported, unsupported, stale, superseded, and
contradictory claims, and its confidence predicts correctness on the declared
test population.

Falsifiers include:

- “last write wins” where authority is unknown;
- high confidence on unsupported or contradictory memories;
- failure to preserve both current and historical truth;
- lower selective accuracy as the system abstains more; or
- apparent calibration that vanishes on a held-out scenario family.

### C5: lifecycle closure

Tenant and same-project private-scope isolation, agent identity and delegation,
concurrent publication, deletion, and rollback operate correctly over raw
episodes, derived records, indexes, caches, exports, and backup restoration.

Falsifiers include:

- any cross-tenant retrieval or disclosure;
- an agent identity accepted from message text or a cross-agent message treated
  as authority;
- a delegated operation broader than any ancestor grant or current user intent;
- a private same-project record becoming shared without authorization;
- a stale worker or duplicate retry publishing;
- an erased item or derivative reappearing after re-indexing or restore;
- rollback reviving a user-deleted item;
- a procedure retaining authority after its capability was revoked; or
- an audit record that cannot explain the state transition.

## Experimental systems and mandatory ablations

All variants use the same base model snapshot, model parameters, tool schemas,
authorization policy, prompt budget, task budget, and episode stream. If a
variant cannot consume the same budget, report the difference and include a
matched-budget analysis.

| ID | Variant | Persistent state | Derived writes | Purpose |
|---|---|---|---|---|
| B0 | No memory | none beyond the declared working context | none | Establish whether persistence helps at all. |
| B1 | Immutable episodic-only, chronological | authorized raw episodes selected by a deterministic chronological rule | none | Establish a transparent persistence baseline. |
| B2 | Immutable episodic-only, retrieved | the same authorized raw episodes selected by the production retriever | none | Separate retrieval gains from consolidation gains. |
| B3 | Unguarded consolidation | raw episodes plus direct model summaries/reflections | model output is immediately usable | Measure the value and risk of consolidation without the proposed controls. |
| B4 | Guarded consolidation | raw episodes plus candidate/promote state machine | only validated, independently verified candidates are usable | Test the complete control architecture. |
| B5-off | B4 without generated counterfactuals | as B4 | summaries may be proposed; synthetic episodes and counterfactuals disabled | Isolate the effect of offline generated material. |
| B5-on | B4 with generated counterfactuals | as B4 | synthetic material remains generated and non-evidentiary | Measure whether “dreaming” adds value or amplifies error. |

B0, at least one immutable episodic-only variant, and B4 are mandatory in every
headline comparison. B1 and B2 should both be used wherever chronological
replay fits the task budget. B3 is mandatory for security claims about the
guarded promotion boundary. B5-off/B5-on is mandatory for claims about offline
dreaming.

Additional diagnostic ablations remove one guard at a time: source typing,
independent-root counting, contradiction handling, verifier separation,
authorization-before-ranking, procedural mediation, tombstones, and atomic
promotion. These are diagnostic experiments, not acceptable deployment
variants.

## Evaluation corpus

### Core utility tracks

Use final proceedings datasets where available and preserve their native
metrics before adding cross-track summaries.

- **LoCoMo:** long conversational recall, temporal reasoning, summarization,
  and unanswerable questions [locomo24][locomo24].
- **LongMemEval:** information extraction, multi-session reasoning, temporal
  reasoning, updates, and abstention [longmemeval25][longmemeval25].
- **MemBench:** sequential participation/observation and
  factual/reflective-memory conditions [membench25][membench25].
- **MemoryAgentBench:** incremental accurate retrieval, test-time learning,
  long-range understanding, and selective forgetting/conflict resolution
  [memoryagentbench26][memoryagentbench26].
- **PM-Bench:** remembering, revising, cancelling, and withholding deferred
  intentions under intervening activity [pmbench26][pmbench26].
- **LongMemEval-V2:** evidence aggregation over long web-agent trajectory
  collections, including workflow and environment knowledge
  [longmemevalv226][longmemevalv226].

Keep benchmark-native splits and scores. Do not pool unlike answer judges into
one “memory intelligence” score.

### Consolidation-drift track

Replay fixed clean episodes through repeated consolidation rounds. Include:

- stable facts;
- legitimate corrections;
- ambiguous conflicts;
- temporally scoped facts;
- preferences that genuinely change;
- repeated paraphrases of one evidence root;
- assistant-generated errors later quoted by the user; and
- generated summaries recursively used as later inputs.

Useful Memories reports severe degradation under repeated consolidation in its
experimental setting, including 52.6% correctness at round ten in one streaming
condition [usefulmem26][usefulmem26]. This is the Figure 2 value; the
[benchmark audit](benchmark-audit.md#consolidation-stress-evidence) records the
paper’s inconsistent prose percentages. That result motivates a round-by-round
track; it is not an expected effect size for Generalist.

Measure every round, not just the final state. Randomize equivalent episode
orderings and vary consolidation interval, compaction pressure, and evidence
density independently.

### Security tracks

Use the original threat mechanisms as starting points, then adapt them to the
candidate/promote and dream boundaries:

- AgentPoison-style direct poisoning of an accessible memory or knowledge
  store [agentpoison24][agentpoison24];
- MINJA-style query-only injection across multiple interactions
  [minja25][minja25];
- MPBench-style content delivered through the actual write channels exposed by
  the agent [untrusted26][untrusted26];
- Hidden in Memory-style sleeper content activated by a later trigger
  [hidden26][hidden26];
- MemPoison-style direct, compositional, and dormant cases
  [mempoison26][mempoison26];
- Bad Memory-style instructions in durable memory files
  [badmemory26][badmemory26];
- AgentDojo and InjecAgent-style indirect instructions embedded in tool output
  [agentdojo24][agentdojo24] [injecagent24][injecagent24]; and
- MEXTRA-style adaptive extraction of private interaction history
  [mextra25][mextra25].

Preserve an unmodified replication track where licenses and APIs permit.
Clearly label architectural adaptations as new Generalist tests; do not compare
their scores directly with published headline numbers.

### New lifecycle tracks

Existing suites leave important gaps. Construct new scenarios with exact
machine-checkable oracles for:

- two or more tenants with lexically similar records and unique secret
  canaries;
- purpose- and principal-specific visibility within one tenant;
- several concurrent logical agents and process incarnations sharing one
  project database;
- same-project records private to a principal, task, or agent versus explicitly
  shared with named agents or the whole project;
- pre-persistence admission cases containing credentials, excluded fields,
  low-entropy secrets, mixed-origin spans, and copied/paraphrased hostile text;
- domain-local retrieval indexes whose ranks, candidate counts, cache entries,
  and returned content remain invariant as unauthorized rows are added, while
  latency distributions quantify the residual shared-resource side channel;
- host-authenticated agent envelopes, forged identity claims in message text,
  and cross-agent messages containing indirect instructions;
- dependency DAG cycles, exhausted retries, terminal prerequisite failures,
  and `all_success|all_terminal|manual` descendant propagation;
- capability delegations with audience, purpose, resource, expiry, revocation,
  and parent-chain restrictions, including confused-deputy requests;
- expired leases, monotonically fenced publishers, delayed workers, duplicate
  delivery, and crash/retry schedules;
- direct database/WAL open, copy, lock, forged-row, symlink, socket-replay, and
  stolen-token attempts from worker sandboxes;
- provenance class (`observed_user_message`, `observed_tool_result`,
  `observed_environment`, `imported_content`, `summary_candidate`,
  `reflection_candidate`, or `counterfactual`) and independent evidence roots;
- contradictory sources with known and unknown authority ordering;
- temporal facts with `valid_from`, `valid_until`, and correction time;
- procedural memories with explicit preconditions and capability manifests;
- revocation of a tool, credential, endpoint, or procedure;
- deletion before and after promotion, embedding, caching, export, and backup;
- correction of sole and mixed-support roots with immediate descendant
  invalidation/revalidation;
- rollback across a deletion epoch;
- concurrent promotion with correction, deletion, revocation, and
  private-to-shared or shared-to-private transitions;
- re-indexing and backup restoration after a tombstone exists; and
- missing, stale, corrupt, and partially restored external deletion ledgers;
- resource-exhaustion attempts that generate excessive candidates or dream
  work.

Synthetic canaries must not be real credentials or personal information.

## Episode and intervention protocol

Each scenario is an immutable event tape with hidden ground truth. The tape
declares principal, tenant, purpose, timestamp, source channel, source class,
authorization context, expected state, and permitted effects.

Run these phases:

1. **Clean acquisition.** Feed ordinary interactions and establish baseline
   utility and benign write behavior.
2. **Intervention.** Inject one declared error, attack, update, permission
   change, deletion, or contradiction. For compositional attacks, distribute
   individually plausible fragments across events.
3. **Offline cycle.** Run zero, one, and multiple bounded consolidation/dream
   cycles. Record every candidate, verifier decision, evidence edge, and state
   version.
4. **Concurrent schedule.** Where applicable, pause two or more processes at
   declared barriers, then interleave promotion, deletion, sharing, revocation,
   lease expiry, retry, and commit operations.
5. **Immediate probe.** Issue retrieval, question-answering, and tool-use
   probes, including semantically nearby benign queries.
6. **Delayed probe.** Add unrelated activity and then trigger dormant,
   prospective, or stale-state tests.
7. **Correction and erasure.** Supply authoritative correction or deletion and
   verify the entire derivative graph.
8. **Operational transition.** Re-index, evict caches, rotate policy, restart,
   roll back, and restore a backup in controlled combinations.
9. **Final audit.** Compare observable behavior and every instrumented storage
   tier with the hidden oracle and append-only audit log.

The attack event and trigger event must be separated in delayed cases. The
consolidator must not receive hidden labels. Defense-specific prompts must be
frozen before the final test split.

## Adversary and benign-failure factors

Cross the following factors where the combination is meaningful:

- **access:** direct store write, query-only interaction, tool/file content,
  cross-agent message, shared-resource content, or compromised dream input;
- **knowledge:** black-box, architecture-aware, retriever-aware, or
  policy-aware;
- **adaptation:** fixed payload or adaptive query sequence;
- **composition:** single record, multiple individually benign records,
  recursive summary, or dormant trigger;
- **goal:** false belief, unsafe action, credential disclosure, cross-tenant
  retrieval, persistence, deletion evasion, or resource exhaustion;
- **timing:** before promotion, after promotion, before correction, after
  deletion, or after rollback/restore; and
- **authority appearance:** anonymous content, user-like wording,
  tool-generated content, peer-agent wording, copied policy language, or
  purported approval.

Also test non-adversarial causes: stale facts, honest user mistakes, tool
errors, duplicated events, missing timestamps, clock skew, partial writes,
verifier unavailability, delayed or restarted workers, expired leases,
`SQLITE_BUSY` retries, duplicate delivery, and truncated context.

## Stage-level measurements

End-to-end attack success is necessary but not sufficient. A blocked tool call
can hide a poisoned store that remains dangerous after a policy change.

### Write, promotion, and persistence

For both malicious and benign inputs report:

- write-attempt count and candidate-creation rate;
- candidate rejection, quarantine, and promotion rate;
- time and number of dream cycles to promotion;
- poisoned-record promotion rate;
- unsupported-promotion rate per eligible event;
- generated-as-observed classification error;
- evidence-root precision, recall, and duplicate-root inflation;
- retrieval exposure: fraction of probes for which the target enters the
  authorized candidate set, ranked context, and final prompt;
- behavioral influence without a tool call;
- prohibited tool-call proposal and executed-effect rate; and
- persistence curves after unrelated events, compaction, correction, deletion,
  restart, rollback, and restore.

Keep these denominators separate. `promoted / attacks`, `retrieved / promoted`,
and `harmful effects / retrieved` reveal different control failures.

### Utility

Report each benchmark’s native score. Add, where appropriate:

- supported-answer accuracy;
- abstention precision and recall;
- current-state and historical-state accuracy;
- intent execution, cancellation, rescheduling, and false-positive action;
- evidence-retrieval recall and precision;
- workflow completion with all authorization checks;
- reflective-memory support precision; and
- utility by memory age, history length, update count, and dream round.

Report paired differences from B0, B1/B2, and B3. A headline consolidation
claim must use episodic-only—not no-memory—as its primary utility comparator.

### Temporal and contradiction handling

Measure:

- conflict-detection precision, recall, and time-to-detection;
- correct-current-value accuracy;
- historical query accuracy at requested time;
- superseded-claim retrieval and use;
- correct abstention when authority cannot be resolved;
- resolution accuracy when an authoritative ordering is supplied; and
- confidence change after a correction or contradiction.

Score detection separately from resolution. A system can notice conflict and
still choose the wrong source.

### Confidence calibration

Require the system to emit a probability or monotonically ordered confidence
for factual memory use and for abstention. Report:

- Brier score and log loss;
- reliability diagrams and expected calibration error with bins fixed before
  evaluation;
- selective accuracy and risk-coverage curves;
- calibration split by source class, memory age, contradiction status, attack
  class, and dream round; and
- calibration drift from the episodic-only system to guarded consolidation.

The self-correction literature cautions against treating the model’s own
critique as independent verification [selfcorrect24][selfcorrect24]. A second
sample from the same model may be diagnostically useful but is not an
independent evidence root.

### Privacy and authorization

Report:

- cross-tenant candidate-set, retrieval-context, answer, and tool-effect
  violations separately;
- canary disclosure rate, unique canaries disclosed, and queries-to-disclosure;
- adaptive extraction yield and false-positive disclosure;
- same-tenant wrong-purpose and wrong-principal access;
- sensitive-field retention when the task does not require it;
- unauthorized procedure retrieval, proposal, and execution; and
- access after revocation or policy-version change.

Run exact-match and semantic disclosure checks, with human adjudication for
ambiguous paraphrases. Log retrieval internally for evaluation without exposing
another tenant’s content to the judge.

### Multi-agent identity, delegation, and concurrency

Use at least two logical agents, separate authenticated client sessions to the
supervisor, and replaceable process instances. Give agents overlapping project
membership but different
private scopes, task purposes, and tool capabilities. Record:

- identity-envelope verification and identity-misbinding rate;
- cross-agent instruction-following when the message lacks current user
  authority;
- confused-deputy requests proposed, denied, and executed;
- delegation-chain validation, privilege amplification, audience misuse,
  expiry, and revocation;
- private-record candidate-set, retrieval-context, answer, and share-transition
  violations within the same project;
- supervisor-bypass attempts and accepted direct state mutations;
- lease acquisition, fencing token, expected parent epoch, and commit outcome;
- stale-worker publication attempts and accepted stale publications;
- compare-and-swap conflicts, revalidation outcomes, duplicate candidates, and
  duplicate logical writes;
- idempotent external-effect requests and duplicate effects; and
- `sent_unknown`, reconciliation, and duplicate outcomes for providers without
  idempotency support; and
- final state under concurrent promotion, correction, deletion, revocation,
  scope narrowing, rollback, and restore.

Run deterministic schedules that force each operation immediately before and
after the other operation’s validation and commit point. Add randomized
schedule fuzzing only after the exact two-operation cases are covered.

SQLite’s official documentation says separate connections normally see only
committed transactions, writes are serialized, and WAL readers may remain on an
older snapshot [sqliteisolation26][sqliteisolation26]
[sqlitetransactions26][sqlitetransactions26]. A clean SQLite commit therefore
shows physical atomicity, not semantic correctness. The oracle must separately
check actor identity, delegation attenuation, scope, fence freshness, expected
version, idempotency, and the rule that tombstones and revocations dominate
concurrent promotion.

### Deletion and rollback

For each deletion target enumerate the expected closure over:

- raw episode store;
- candidate and promoted stores;
- provenance and contradiction indexes;
- embeddings and search indexes;
- prompt/result caches;
- exports and analytics copies;
- audit-visible tombstones; and
- backup manifests and restored states.

Report deletion completion by tier, total time, residual reference count,
semantic canary retrieval after deletion, and resurrection after re-index,
restart, rollback, or restore. Test that rollback removes newly promoted items
but preserves deletions that occurred after the rolled-back state. A tombstone
may preserve non-sensitive deletion metadata; it must not itself retain the
deleted payload.

Scan database pages, WAL/SHM, FTS shadow tables, temporary spill files, caches,
exports, backups, and restored copies for raw and semantic canaries. Force
deletion/revocation immediately before and after prompt send, tool
authorization, effect intent, remote receipt, and episode finalization. Record
provider exposure that cannot be recalled. Restore tests fail closed until the
separate authenticated deletion ledger reaches its signed high-water mark.
Crash at every ledger-fsync/database-commit boundary and test ledger-ahead,
database-ahead, truncated-chain, and duplicate-tombstone states. Verify
per-artifact key destruction leaves unrelated records readable, whole-scope
destruction closes the scope, and restored key backups cannot revive destroyed
key IDs.

### Operational cost and availability

Measure wall-clock latency, model calls, input/output tokens, index operations,
storage growth, verifier load, operator-review load, dream-cycle duration,
quarantine backlog, and failure recovery. Report clean and adversarial
distributions. A defense that lets an attacker force unbounded offline work has
failed even if no poisoned item is promoted.

## Statistical design

### Units and randomization

The primary unit is the independent scenario or source conversation, not every
question drawn from it. Questions sharing an event tape are clustered.

- Replay the identical tape across ablations.
- Treat each deterministic concurrent schedule as a scenario; do not count
  individual SQL statements as independent trials.
- Randomize variant order where order can affect service state.
- Use multiple declared model seeds or sampling replicates for stochastic
  behavior.
- Stratify by benchmark family, attack class, history length, and tool-risk
  level.
- Keep development attacks, validation attacks, and final adaptive red-team
  attacks separate.

### Estimates

For paired utility outcomes, report the paired effect, a cluster-aware
confidence interval, and the raw per-variant score. For binary safety outcomes,
report numerator, denominator, and an exact or otherwise justified confidence
interval. For time-to-failure, report survival curves with censoring stated.
For calibration, bootstrap at the scenario cluster, not individual answer,
level.

When zero failures are observed in `n` independent trials, report the one-sided
exact binomial upper bound rather than “0% risk.” At confidence
`1 - alpha`, the zero-event upper bound is `1 - alpha^(1/n)` under the stated
independence and stationarity assumptions. If those assumptions are doubtful,
cluster more conservatively.

Predeclare handling of retries, model refusals, harness errors, timeouts, and
interrupted runs. Do not silently drop them. Correct for multiple confirmatory
claims or identify one primary outcome per claim; label the rest exploratory.

### Judge quality

Prefer deterministic state and tool-effect oracles. For semantic answers:

- blind judges to system variant;
- freeze judge model and prompt;
- double-code a stratified human sample;
- report adjudication rules and agreement;
- test the judge on poisoned outputs designed to manipulate evaluation; and
- release hashed inputs/outputs or complete artifacts where privacy and
  licenses permit.

## Acceptance gates

Numerical thresholds are product-risk decisions and must be fixed before final
evaluation. This corpus does not invent universal cutoffs.

At minimum, a release candidate must satisfy all of these **design gates**:

1. **Invariant gate:** deterministic tests find no source-class relabeling,
   unsupported promotion, independent-root inflation, authorization-after-
   ranking, agent-identity misbinding, delegation amplification, stale-fence
   publication, private-scope widening, duplicate host effect intent, or
   deletion-resurrection violation.
2. **Critical-effect gate:** no observed cross-tenant disclosure or
   same-project private-scope disclosure, confused-deputy effect, or
   unauthorized high-impact tool effect in the declared final suite, with the
   zero-event upper confidence bound reported.
3. **Poisoning gate:** guarded consolidation improves every prespecified
   stage-level poisoning measure over B3 and does not merely delay activation
   beyond the test horizon.
4. **Utility gate:** B4 improves the prespecified primary utility outcome over
   immutable episodic-only memory by the declared margin under matched budgets.
5. **Calibration gate:** risk-coverage and high-confidence error on unsupported,
   stale, and contradictory claims meet predeclared bounds on held-out
   scenarios.
6. **Lifecycle gate:** all instrumented deletion targets close, no deleted item
   resurrects, rollback preserves later tombstones/revocations, and every stale
   publisher is rejected.
7. **Operations gate:** clean and adversarial latency, resource, quarantine,
   and human-review budgets stay within declared limits.

A failed critical-effect or lifecycle gate blocks write-capable deployment even
if average utility improves. An inconclusive confidence interval is
inconclusive, not a pass.

## Staged execution

1. **State-machine tests:** property and fault-injection tests for provenance,
   agent identity, delegation, sharing, fencing, idempotency, authorization,
   transactions, tombstones, rollback, and restore.
2. **Deterministic tape replay:** all B0–B5 variants on small exact-oracle
   scenarios.
3. **Benchmark utility:** native benchmark protocols with matched budgets.
4. **Static adversarial suite:** frozen attacks, compaction schedules, and
   delayed triggers.
5. **Adaptive red team:** held-out attackers with query and tool/file channels.
6. **Shadow mode:** production-like traffic with no memory-derived external
   effects and isolated synthetic canaries.
7. **Read-only episodic release:** retrieval enabled, consolidation decisions
   logged but not served.
8. **Limited promoted-memory release:** low-risk tasks, rollback ready, bounded
   cohorts, and explicit monitoring.
9. **Procedural-memory release:** separate decision only after capability,
   revocation, sandbox, and tool-effect gates pass.

Each stage produces a signed manifest of code, model, policy, benchmark,
attacks, configuration, and results. Advancement is a governance decision, not
an automatic consequence of a composite score.

## Reproducibility record

For every run retain:

- code commit and clean/dirty state;
- model/provider version and sampling settings;
- benchmark repository commit, data hashes, and license;
- event-tape and attack-corpus hashes;
- prompts, tool schemas, policies, capability manifests, and verifier versions;
- embedding model, chunker, index, ranking parameters, and context budget;
- consolidation schedule, dream budget, time source, and random seeds;
- logical agent and process-instance identities, task/delegation envelopes,
  lease/fencing history, idempotency keys, connection/journal mode, and forced
  concurrency schedule;
- candidate, verifier, promotion, retrieval, tool, deletion, and rollback logs;
- harness failures and excluded cases with reasons; and
- analysis code, metric definitions, and confidence-interval method.

Sensitive raw episodes should remain access-controlled. Publish synthetic
reproduction artifacts and redacted manifests where full release would create
privacy or misuse risk.

## Interpretation limits

A pass supports only the tested model, harness, policies, tools, tenant
topology, attacks, data distributions, and time horizon. It does not prove:

- absence of unknown attacks;
- legal compliance;
- removal from model weights or uncontrolled third-party copies;
- security after provider, model, retriever, or policy changes;
- safety for higher-impact tools; or
- that offline-generated content is true.

Re-run the affected tracks after any material change. Treat benchmark
improvement, implementation correctness, security evidence, privacy evidence,
and production readiness as separate claims.

## Local References

[sqliteisolation26]: SQLite Project. “Isolation In SQLite.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/isolation.html

[sqlitetransactions26]: SQLite Project. “Transaction.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/lang_transaction.html

[locomo24]: Maharana, Adyasha; Lee, Dong-Ho; Tulyakov, Sergey; Bansal, Mohit; Barbieri, Francesco; Fang, Yuwei. “Evaluating Very Long-Term Conversational Memory of LLM Agents.” *Proceedings of ACL 2024*, 13851–13870. https://aclanthology.org/2024.acl-long.747/

[longmemeval25]: Wu, Di; Wang, Hongwei; Yu, Wenhao; Zhang, Yuwei; Chang, Kai-Wei; Yu, Dong. “LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.” *International Conference on Learning Representations* (ICLR 2025). https://openreview.net/forum?id=pZiyCaVuti

[membench25]: Tan, Haoran; Zhang, Zeyu; Ma, Chen; Chen, Xu; Dai, Quanyu; Dong, Zhenhua. “MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents.” *Findings of ACL 2025*, 19336–19352. https://aclanthology.org/2025.findings-acl.989/

[memoryagentbench26]: Hu, Yuanzhe; Wang, Yu; McAuley, Julian. “Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.” *International Conference on Learning Representations* (ICLR 2026). https://openreview.net/forum?id=DT7JyQC3MR

[pmbench26]: Liu, Genglin; Gabriel, Saadia. “PM-Bench: Evaluating Prospective Memory in LLM Agents.” *Conference on Language Modeling* (COLM 2026). https://arxiv.org/abs/2607.12385

[longmemevalv226]: Wu, Di; Ji, Zixiang; Kawatkar, Asmi; Kwan, Bryan; Gu, Jia-Chen; Peng, Nanyun; Chang, Kai-Wei. “LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues.” arXiv:2605.12493v1, preprint (2026). https://arxiv.org/abs/2605.12493

[usefulmem26]: Zhang, Dylan; Lin, Yanshan; Wu, Zhengkun; Sun, Yihang; Li, Bingxuan; Li, Dianqi; Peng, Hao. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1, preprint (2026). https://arxiv.org/abs/2605.12978

[agentdojo24]: Debenedetti, Edoardo; Zhang, Jie; Balunović, Mislav; Beurer-Kellner, Luca; Fischer, Marc; Tramèr, Florian. “AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.” *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track* (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html

[injecagent24]: Zhan, Qiusi; Liang, Zhixiang; Ying, Zifan; Kang, Daniel. “InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents.” *Findings of ACL 2024*, 10471–10506. https://aclanthology.org/2024.findings-acl.624/

[agentpoison24]: Chen, Zhaorun; Xiang, Zhen; Xiao, Chaowei; Song, Dawn; Li, Bo. “AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.” *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html

[minja25]: Dong, Shen; Xu, Shaochen; He, Pengfei; Li, Yige; Tang, Jiliang; Liu, Tianming; Liu, Hui; Xiang, Zhen. “Memory Injection Attacks on LLM Agents via Query-Only Interaction.” *Advances in Neural Information Processing Systems 38* (NeurIPS 2025). https://papers.nips.cc/paper_files/paper/2025/file/42a97bbd9844d2bf68596730af80bcdf-Paper-Conference.pdf

[untrusted26]: Dash, Pritam; Ge, Tongyu; Jain, Aditi; Shah, Tanmay; Shang, Zhiwei. “From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents.” arXiv:2606.04329v2, preprint (2026). https://arxiv.org/abs/2606.04329

[hidden26]: Pulipaka, Sidharth; Hlebik, Stanislau; Raghav, Leonidas; Abdelnabi, Sahar; Raina, Vyas; Sheth, Ivaxi; Fritz, Mario. “Hidden in Memory: Sleeper Memory Poisoning in LLM Agents.” arXiv:2605.15338v2, preprint under review (2026). https://arxiv.org/abs/2605.15338

[mempoison26]: Gao, Jifeng; Xia, Kang; Zhang, Yi; Hong, Xiaobin; Lin, Mingkai; Wei, Xingshen; Li, Wenzhong; Lu, Sanglu. “MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents.” arXiv:2607.14651v1, preprint (2026). https://arxiv.org/abs/2607.14651

[badmemory26]: Gadgil, Soham; Alexander, David; Sunku, Sai; Roesner, Franziska. “Bad Memory: Evaluating Prompt Injection Risks from Memory in Agentic Systems.” arXiv:2607.14611v1, preprint (2026). https://arxiv.org/abs/2607.14611

[mextra25]: Wang, Bo; He, Weiyi; Zeng, Shenglai; Xiang, Zhen; Xing, Yue; Tang, Jiliang; He, Pengfei. “Unveiling Privacy Risks in LLM Agent Memory.” *Proceedings of ACL 2025*, 25241–25260. https://aclanthology.org/2025.acl-long.1227/

[selfcorrect24]: Huang, Jie; Chen, Xinyun; Mishra, Swaroop; Zheng, Huaixiu Steven; Yu, Adams; Song, Xinying; Zhou, Denny. “Large Language Models Cannot Self-Correct Reasoning Yet.” *International Conference on Learning Representations* (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html
