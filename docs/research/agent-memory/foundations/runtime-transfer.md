# Transfer to a Non-Weight-Updating Agent Runtime

## Claim type

This document is an **architectural inference**, not a report that cognitive
science or continual-learning papers have already validated an LLM-agent memory
design. It asks what can be implemented when:

- the base model's parameters are fixed;
- durable state lives in a host runtime;
- retrieval changes the model's prompt-time context;
- tools can observe or mutate an external world; and
- offline jobs must not silently acquire new authority.

Primary language-model evidence does show that non-parametric memory can affect
fixed-model behavior. A nearest-neighbor datastore improved language-model
predictions and enabled domain adaptation without further model training
[khandelwal20][khandelwal20]. Retrieval-augmented generation combined a
parametric generator with retrieved documents for knowledge-intensive tasks
[lewis20][lewis20]. Those results establish the viability of retrieval, not the
reliability of autonomous consolidation.

## A type system for memory

The strongest transfer from the evidence is to keep memory classes distinct:

| Type | Meaning | May be rewritten? | May support an answer directly? |
|---|---|---|---|
| Raw event | Exact input, tool output, state snapshot, or runtime event | No; append correction metadata | Yes, with provenance |
| Episode | Grouped temporal trace over raw events | Grouping may be versioned; events remain intact | Yes |
| Semantic claim | Derived proposition with scope and evidence links | Supersede or retract, do not erase lineage | Yes, if current and supported |
| Procedure | Reusable action pattern plus preconditions and tests | Versioned and regression-gated | Yes, subject to policy |
| Counterfactual | Generated possible trajectory | Immutable as a prediction; never retyped as observation | No, except as a planning hypothesis |
| Index/cache | Embedding, summary view, retrieval score, compiled context | Freely rebuildable | Only as a pointer to source records |

The names “episodic” and “semantic” are functional. They do not claim a
biological homology.

## Evidence from recent LLM memory systems

### Recurrence can gate expensive consolidation

RecMem uses a lightweight raw-interaction layer and triggers LLM extraction of
episodic and semantic memories only after semantically similar interactions
recur. It then performs a semantic-refinement pass intended to recover detail
omitted from episodic abstraction. On the two conversational-memory benchmarks
tested, the authors report a favorable construction-cost and task-performance
trade-off relative to eager baselines [dai26][dai26].

This is peer-reviewed Findings of ACL 2026 evidence for **one selective
consolidation policy**. It does not establish recurrence as salience. The paper
itself identifies the risk that rare but critical one-off information may never
cross the recurrence threshold.

The transferable pattern is:

- cheaply retain raw interactions;
- delay lossy LLM processing;
- use an inspectable trigger;
- retrieve from both raw and derived layers; and
- check which details abstraction omitted.

### Repeated rewriting can reduce utility

Zhang and colleagues' May 2026 preprint directly tests continuously updated
textual memories. In their evaluated settings, memory utility was non-monotonic,
the same trajectories produced different derived memories under different
update schedules, and episodic-only controls remained competitive
[zhang26][zhang26].

This new result should be treated as a warning and a testable hypothesis, not a
universal theorem. It supports:

- preserving raw episodes as first-class evidence;
- evaluating repeated update cycles rather than one-shot summaries;
- making consolidation optional and reversible; and
- refusing to equate a cleaner summary with a better memory.

As noted in [the continual-learning analysis](continual-learning-replay.md), the
paper's abstract and project page describe the selected ARC result with
different 54% wordings. No numeric accuracy conclusion is used here.

### Anticipatory compute is narrower than dreaming

Sleep-time compute transforms known context before a future query and works
best in the paper's experiments when that query is predictable
[lin25][lin25]. For a host runtime, this supports precomputing indexes,
dependency maps, candidate invariants, or likely-query views. It does not
support unbounded synthetic episode generation.

## The consolidation transaction

The following transaction is a synthesis of the evidence, with all normative
controls added as engineering requirements.

### 1. Snapshot

Choose an immutable cutoff in the event log and record:

- episode identifiers, content hashes, and user or tenant boundary;
- runtime, model, prompt, tool, and policy versions; and
- external-resource versions and the consolidation trigger.

This prevents a job from silently reasoning over a moving target.

### 2. Select

Build a bounded evidence set using multiple signals:

- recurrence, novelty, surprise, staleness, and explicit user importance;
- unresolved failure, high consequence, or conflict with an active claim; and
- anticipated near-term query value and coverage of rare domains.

Recurrence alone misses rare critical instructions [dai26][dai26]. Retrieval
frequency alone creates popularity feedback. The selector and its omissions
must be logged.

### 3. Propose

The model may propose:

- a scoped semantic claim or procedure with preconditions;
- a supersession relation, retrieval index entry, or counterfactual test; or
- no change.

The output is a candidate, never a committed memory. Every proposition should
cite supporting episode spans, state its scope, and distinguish observation
from inference.

### 4. Challenge

Run independent queries for:

- disconfirming episodes and older versions of the same claim;
- temporal changes, source-authority differences, and unsupported steps; and
- cross-user or cross-project leakage and prompt-injection markers.

The challenger should not receive only the evidence chosen by the proposer.
Replay that samples only supporting cases can reinforce a false abstraction.

### 5. Verify

Use the strongest available verifier:

- deterministic schema, content-hash, source, or software checks;
- read-only queries against current external state or held-out longitudinal
  tasks; and
- independent model critique or explicit human approval.

Model agreement is weaker than external verification. If no suitable verifier
exists, retain the item as an unverified hypothesis with restricted retrieval.

### 6. Regression gate

Compare the active memory set with the candidate set on:

- prior and current cases, especially likely conflicts and rare,
  high-consequence episodes; and
- repeated future consolidation cycles.

This is the fixed-weight analogue of replay and “do not increase old loss”
constraints, not an application of their gradient guarantees.

### 7. Commit or quarantine

A commit is atomic and versioned. It adds the derived record, evidence edges,
scope, verifier result, and supersession edges while leaving raw events
unchanged. A failed or ambiguous candidate is quarantined, not silently merged.

### 8. Observe

Log retrievals and downstream outcomes. Success can raise utility estimates but
must not turn popularity into truth. Later corrections supersede the claim while
preserving the earlier version and the episodes that motivated it.

## Addressing the four target problems

### Abstraction

Mechanism:

- cluster related episodes and contrast positive with negative cases;
- propose the narrowest common claim with scope and lineage; and
- test on held-out episodes while retaining exemplars after semanticization.

The schema and representational-overlap findings support the possibility of
integration, while also suggesting a detail trade-off
[tompary17][tompary17]. They do not validate LLM summarization.

### Contradiction

Mechanism:

- normalize propositions for conflict retrieval while retaining validity time
  and environment scope;
- distinguish temporal change from present conflict and rank source authority
  under explicit policy; and
- preserve unresolved alternatives and supersede rather than overwrite.

Replay and consolidation research does not solve truth maintenance. This entire
mechanism is an architectural requirement.

### Forgetting

Separate three policies:

1. **Retention**: whether source records remain stored.
2. **Accessibility**: whether indexes and retrieval policies surface them.
3. **Influence**: whether retrieved items may change an answer or action.

Decay can reduce accessibility without deleting evidence. Deletion may still
be required for consent, privacy, or policy reasons and should propagate to
derived memories. Rare, consequential items need protected retention even if
they recur infrequently.

### Counterfactual generation

Mechanism:

- root generation in a declared observed state, label generated transitions,
  and bound horizon and resource use;
- prohibit side effects and use deterministic or external verifiers; and
- record predictions before tests, promoting only resulting observations.

This captures the useful distinction in model-based RL between real and
simulated experience without importing its training guarantees.

## Non-weight learning that is actually available

A fixed-model runtime can still learn by changing:

- episode retention and indexing, retrieval ranking, and query expansion;
- derived facts, validity intervals, procedures, task decompositions, and
  checklists; and
- cached environment representations, regression suites, and tool/model
  routing.

These changes can materially alter behavior. They are host-state learning, not
parametric learning. Evaluation must attribute gains and failures to the
runtime state rather than claiming the base model learned.

## Invariants for safe offline work

1. Consolidation never rewrites raw evidence.
2. Every derived item has machine-readable lineage.
3. Generated content is never typed as observation.
4. A consolidation job has no authority to execute external side effects.
5. Commits are atomic, versioned, reversible, and tenant-scoped.
6. Deletion and consent constraints propagate through derivation edges.
7. Current external facts are revalidated when cheap and relevant.
8. A failed verifier cannot be replaced by the proposing model's confidence.
9. Retrieval and update policies are versioned with the memories they affect.
10. “No change” is a successful consolidation outcome.

These invariants are design hypotheses for later architecture and safety
analysis. The foundations literature motivates them but does not prove them.

## Transfer scoreboard

| Claim | Status |
|---|---|
| Separate rapid episodes from slower abstractions | Strong computational motivation; partial biological support |
| Preserve raw episodes after abstraction | Strong architectural inference; supported by new fixed-weight experiments |
| Gate consolidation on recurrence | Peer-reviewed 2026 benchmark evidence; rare-event limitation remains |
| Replay retained cases before an update | Strong continual-learning evidence when weights change; regression-gate analogue for host state |
| Generate counterfactual tool trajectories while idle | Plausible only in a typed sandbox; no general safety evidence |
| Let synthetic “dreams” update deployed procedures automatically | Unsupported |
| Let an offline job act with the online agent's authority | Unsupported and outside the memory-learning evidence |
| Claim continual neural learning without weight updates | Category error |

## Uncertainty and access status

The two 2026 agent-memory sources are unusually recent. RecMem is published in
Findings of ACL 2026; Zhang et al. remained arXiv v1 at the checked URL.
Sleep-time compute remained an arXiv preprint. Model and benchmark behavior can
drift, and production agent histories differ from conversational-memory
benchmarks. URLs and source status were checked on 2026-07-26.

## Local References

[khandelwal20]: Khandelwal, Urvashi; Levy, Omer; Jurafsky, Dan; Zettlemoyer, Luke; Lewis, Mike. “Generalization Through Memorization: Nearest Neighbor Language Models.” *International Conference on Learning Representations* (2020). https://openreview.net/forum?id=HklBjCEKvH

[lewis20]: Lewis, Patrick; Perez, Ethan; Piktus, Aleksandra; Petroni, Fabio; Karpukhin, Vladimir; Goyal, Naman; Küttler, Heinrich; Lewis, Mike; Yih, Wen-tau; Rocktäschel, Tim; Riedel, Sebastian; Kiela, Douwe. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” *Advances in Neural Information Processing Systems 33* (2020). https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

[dai26]: Dai, Zijie; Deng, Shiyuan; Guan, Sheng; Tian, Yizhou; Yao, Xin; Yan, Xiao; Cheng, James. “RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents.” *Findings of the Association for Computational Linguistics: ACL 2026*, 32353–32376 (2026). https://doi.org/10.18653/v1/2026.findings-acl.1619

[zhang26]: Zhang, Dylan; Lin, Yanshan; Wu, Zhengkun; Sun, Yihang; Li, Bingxuan; Li, Dianqi; Peng, Hao. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1 (2026). https://arxiv.org/abs/2605.12978

[lin25]: Lin, Kevin; Snell, Charlie; Wang, Yu; Packer, Charles; Wooders, Sarah; Stoica, Ion; Gonzalez, Joseph E. “Sleep-time Compute: Beyond Inference Scaling at Test-time.” arXiv:2504.13171v1 (2025). https://arxiv.org/abs/2504.13171

[tompary17]: Tompary, Alexa; Davachi, Lila. “Consolidation Promotes the Emergence of Representational Overlap in the Hippocampus and Medial Prefrontal Cortex.” *Neuron* 96(1), 228–241.e5 (2017). https://doi.org/10.1016/j.neuron.2017.09.005
