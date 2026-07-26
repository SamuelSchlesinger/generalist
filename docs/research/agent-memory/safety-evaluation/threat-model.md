# Threat Model for Persistent Agent Memory and Offline “Dreaming”

## Scope

This document covers a tool-using language-model agent with:

- durable raw interaction and tool episodes;
- retrieval over prior episodes and derived facts;
- an offline job that summarizes, reflects, links, decays, or proposes
  procedures and counterfactuals;
- cross-session use of promoted memory;
- user-facing inspection, correction, and deletion;
- optional multi-user or multi-tenant operation; and
- multiple concurrent agent instances or processes sharing one project memory
  database.

The protected system includes the writer, candidate store, promotion service,
retriever, prompt renderer, procedure runner, indexes, caches, audit log,
exports, backups, and offline consolidator. Model weights are outside the
external-memory deletion boundary unless an implementation explicitly trains
on stored episodes. Agent identity, capability delegation, project sharing,
leases, fencing, and publication coordination are inside the boundary.

## Evidence status

Persistent poisoning is not hypothetical. AgentPoison demonstrated poisoning of
agents’ memory or knowledge bases without retraining the model; in its evaluated
systems it reported at least 80% average attack success, under 1% benign
performance degradation, and poison rates below 0.1% [agentpoison24][agentpoison24].
Those numbers belong to its three evaluated agents and attack construction, not
to arbitrary production systems.

MINJA showed a different route: a query-only adversary can induce an agent to
write attacker-chosen records without directly modifying the memory database
[minja25][minja25]. MEXTRA showed black-box extraction of private interactions
from two representative memory-augmented agents [mextra25][mextra25].
AgentDojo and InjecAgent establish that untrusted tool data can steer tool-using
agents and enable privacy harms even before persistence is added
[agentdojo24][agentdojo24] [injecagent24][injecagent24].

Four 2026 papers materially expand the test space, but are recent preprints:

- MPBench models four write channels—explicit writes, policy-driven inferred
  writes, compaction-driven writes, and experience-to-procedure writes—and
  evaluates 3,240 attack cases plus 2,997 benign cases on OpenClaw and HERMES
  with one model [untrusted26][untrusted26].
- Bad Memory plants malicious instructions in a synthetic workspace’s
  auto-loaded or referenced memory files and observes model-, target-, and
  sequence-dependent persistence across Claude Code and Codex
  [badmemory26][badmemory26]. Its attacker already controls a memory file; it
  does not demonstrate reliable ingestion from external content.
- Hidden in Memory studies a single adversarial document that induces a
  fabricated memory, which is later retrieved and used in a new session
  [hidden26][hidden26]. Its provider pipelines are only partly observable and
  some system prompts are reconstructed.
- MemPoison-Bench contains 1,227 hand-validated cases over direct,
  compositional, and context-triggered dormant corruption, three injection
  channels, and three textual memory substrates [mempoison26][mempoison26].
  Its reported defense results do not cover multimodal memory, temporal decay,
  or access-control systems.

These sources justify adversarial tests. They do not estimate real-world attack
frequency or prove that the controls proposed here work.

## Assets and security objectives

| Asset | Objective | Representative harm |
|---|---|---|
| User intent and policy | Integrity and precedence | A retrieved “preference” silently overrides a current instruction |
| Agent identity and delegated authority | Authenticity, attenuation, revocation | A peer makes a more privileged agent act as its confused deputy |
| Factual and temporal state | Accuracy, history, and freshness | A former endpoint or address is treated as current |
| Raw episodes | Fidelity, provenance, privacy | Tool output is rewritten as a user assertion or exposed cross-tenant |
| Derived memories | Traceability and defeasibility | A summary loses a precondition and becomes a false general rule |
| Procedures and skills | Integrity and least privilege | A poisoned trace becomes a reusable credential-upload workflow |
| Secrets and personal data | Confidentiality and minimization | A memory extractor recovers private interactions |
| Tenant boundary | Noninterference | Similarity search crosses user or project scope |
| Same-project private scope | Confidentiality and authorized sharing | A project peer retrieves another principal’s private task memory |
| Deletion and correction state | Completeness and non-resurrection | A deleted record returns through an embedding or re-consolidation |
| Audit trail | Integrity and accountability | A harmful promotion cannot be reconstructed or rolled back |
| Availability and cost | Bounded resource use | Poison records crowd out useful memory or trigger unbounded dreaming |

## Trust is multidimensional

A single `trusted` bit is inadequate. Each item needs independent answers to:

1. **Origin:** which principal, tool, environment, or model process produced it?
2. **Content class:** observation, assertion, inference, counterfactual,
   prediction, or procedure?
3. **Integrity:** was the captured byte sequence altered after ingestion?
4. **Epistemic support:** what independent evidence supports the claim?
5. **Instruction authority:** may this origin direct behavior at all?
6. **Capability authority:** which, if any, tools and resources may it influence?
7. **Scope:** for which tenant, user, project, purpose, and time is it valid?
8. **Sensitivity:** who may retrieve or render it?
9. **Execution actor:** which authenticated agent instance proposed, verified,
   promoted, retrieved, or acted on it?
10. **Delegation:** which host-issued grant authorized that actor, and was the
    grant current, purpose-bound, and no broader than its parent?

An authenticated web response has transport integrity but remains untrusted as
an instruction. A user statement has conversational authority but can still be
mistaken. A generated summary may be useful while having neither observation
status nor independent support. Embedding similarity establishes relevance,
not truth, permission, or current validity.

## Source classes

The following taxonomy is a **design inference**:

| Class | Meaning | Promotion ceiling without new evidence |
|---|---|---|
| `observed_user_message` | Exact user-authored content and metadata | User assertion, never objective fact merely by repetition |
| `observed_tool_result` | Exact tool response, endpoint, invocation, and time | Tool-attested observation within tool scope |
| `observed_environment` | Runtime state captured by trusted instrumentation | Observation within sensor and clock limits |
| `imported_content` | Document, webpage, email, repository, or inter-agent input | Untrusted data |
| `inferred_fact_candidate` | Model or deterministic extraction from evidence | Candidate fact |
| `summary_candidate` | Lossy compression of identified inputs | Candidate summary |
| `reflection_candidate` | Model-generated diagnosis or lesson | Hypothesis |
| `counterfactual` | Generated event that did not occur | Simulation only |
| `prediction` | Counterfactual with outcome and evaluation deadline | Testable hypothesis |
| `procedure_candidate` | Proposed reusable action sequence | Inert text |
| `approved_procedure` | Reviewed version with capability and environment manifest | Executable only inside declared bounds |

No amount of replay or descendant agreement changes an item’s origin class.

## Adversaries and failure sources

### External content adversary

Controls a webpage, document, repository, email, tool result, or inter-agent
message processed during a legitimate task. The adversary may try to induce a
write, wait for later retrieval, and steer a tool action. This is the threat
modeled most directly by Hidden in Memory and MPBench
[hidden26][hidden26] [untrusted26][untrusted26].

An inter-agent message is in this class. Authentication can establish which
agent sent bytes; it does not make the bytes policy, user intent, independent
evidence, or a transferable capability.

### Query-only interlocutor

Can converse with the agent but cannot directly access the store or system
prompt. The adversary uses multi-step interaction to make a malicious record
appear useful or user-approved, as in MINJA [minja25][minja25].

### Memory-file or store writer

Can modify a workspace instruction or knowledge file, import/restore path, or
raw store record. Bad Memory studies a limited version of this stronger
capability [badmemory26][badmemory26]. Supply-chain compromise, insecure file
permissions, or restore tampering can create it.

### Cross-tenant attacker

Is a legitimate user in another scope and attempts to influence shared indexes,
infer existence through timing/ranking, or retrieve another tenant’s memory.
This threat is mostly absent from current memory-utility benchmarks.

### Opportunistic insider or operator

Has some administrative access and attempts unauthorized browsing, export, or
mutation. This requires access control and audit protection beyond prompt-level
defenses.

### Peer agent, confused deputy, or stale worker

A peer agent may be benign, compromised, less privileged, or assigned a
different purpose within the same project. It may ask another agent to use a
capability it does not hold, label private memory as project-shared, or send
instruction-bearing content. A crashed or partitioned worker may also resume
after its lease expired and publish an obsolete consolidation result. Physical
serialization of database writes does not authenticate the actor or resolve
these semantic conflicts.

### Non-adversarial generator and environment

The model can hallucinate, summarize incorrectly, overgeneralize, or misread a
tool result. External facts can change. Tool outputs can be partial, stale, or
duplicated. Safety cannot depend on malicious intent being detectable.

## Threats by lifecycle stage

### T1 — Persistent prompt injection and memory poisoning

**Path.** Untrusted content influences a direct, inferred, compaction, or
procedure write; the item survives; retrieval presents it as authoritative;
the agent follows it or calls a tool.

**Variants.**

- Direct single-record corruption.
- Query-only induction without store access.
- Indirect injection through tool or document content.
- Compositional corruption distributed across individually plausible records.
- Dormant corruption activated only by a later natural context.
- A payload stored in an auto-loaded instruction or behavior file.
- Poison transferred into a higher-authority summary during compaction.

MemPoison reports that its write-time consistency check reduced direct L1
behavioral corruption much more than L2/L3; its combined `MIXed` baseline still
reported nonzero corruption in every tier [mempoison26][mempoison26]. The
specific defense implementations and numbers are not a universal lower bound,
but the structural lesson is strong: per-record admission checks do not observe
all future co-retrieval sets and triggers.

### T2 — Self-reinforcing hallucination

**Path.** A generated reflection states unsupported claim `x`; consolidation
stores it; later retrieval treats `x` as evidence; another reflection cites the
first; frequency and descendant count raise confidence; behavior generates
episodes consistent with `x`, closing a false feedback loop.

Useful Memories Become Faulty reports degradation when natural-language memory
is repeatedly consolidated. On a 19-problem ARC-AGI Stream slice where GPT-5.4
solved all problems without memory, its ground-truth-solution streaming setup
fell to 52.6% by round ten, while the static whole-pool condition retained
94.7% [usefulmem26][usefulmem26]. These are Figure 2 values; the
[benchmark audit](benchmark-audit.md#consolidation-stress-evidence) records the
preprint’s inconsistent prose percentages. The authors identify grouping errors,
interference, overgeneralization, and stripped preconditions, and their
episodic-management-only conditions often matched or exceeded abstraction-heavy
ones. This May 2026 preprint uses textual/synthetic settings, small repeat
counts, and no formal uncertainty analysis; it is not proof that all
consolidation is harmful.

Huang and colleagues found that intrinsic self-correction without external
feedback often failed or degraded reasoning in their evaluated tasks
[selfcorrect24][selfcorrect24]. This does not establish universal impossibility.
This supports treating same-model critique as weak evidence. Independently, the
architecture does not count another model sample as an independent evidence
root.

### T3 — Generated episode masquerades as observation

**Path.** Offline replay invents a plausible episode or fills missing fields;
the runtime stores it in the same schema as a tool trace; future retrieval,
evaluation, or consolidation cannot distinguish it from an event that occurred.

Generative Agents explicitly stores experiences and synthesizes reflections;
Reflexion stores model/feedback-derived reflections in an episodic buffer
[generativeagents23][generativeagents23] [reflexion23][reflexion23]. These
systems demonstrate the usefulness of generated memory objects, not their
truth. A hard source-class boundary is a design inference from that ambiguity.

### T4 — Stale facts and contradictions

**Path.** A current fact changes; both versions remain; similarity ranking
returns the older one, or consolidation merges them into a timeless statement.
Alternatively, a malicious “update” wins because it is newer.

LongMemEval explicitly tests knowledge updates and temporal reasoning, while
MemoryAgentBench’s FactConsolidation data exercises version conflict/selective
forgetting [longmemeval25][longmemeval25] [memoryagentbench26][memoryagentbench26].
These are capability evaluations under benchmark conditions, not authorization
or adversarial-source tests. Safe handling requires event time, record time,
scope, source authority, revision links, and explicit contradiction sets.

### T5 — Privacy, secret retention, and extraction

**Path.** The writer stores unnecessary secrets or third-party data; a broad
retrieval query surfaces them; prompt injection or black-box probing elicits
them; generated summaries create additional copies.

MEXTRA demonstrates black-box extraction against two memory-augmented agents
[mextra25][mextra25]. Work on RAG database leakage further shows that retrieval
stores can expose private source data [ragprivacy24][ragprivacy24]. Neither paper
quantifies Generalist’s risk. Both make secret canaries, extraction attempts,
minimization, and retrieval authorization necessary tests.

### T6 — Cross-user or cross-project leakage

**Path.** A global embedding index ranks an item before access control;
identifiers collide; caches omit tenant keys; deduplication links same-text
records across tenants; an offline batch groups episodes from several users; an
operator export loses policy metadata.

Current utility benchmarks rarely exercise mutually distrustful tenants. This
is an architecture-derived threat. Authorization must happen before semantic
ranking and again before rendering; offline jobs must use tenant-scoped inputs
and outputs.

### T7 — Over-broad retrieval

**Path.** The retriever returns too many records, wrong source classes, or
irrelevant sensitive content. The model follows instructions embedded in
relevant-looking data or loses the authoritative item in a large context.

Hidden in Memory found its poisoned memories were much more likely to be
retrieved and used for goal-adjacent than goal-distant queries in its external
manager setup [hidden26][hidden26]. This makes relevance a potential attacker
tool, not a safety property. Retrieval must jointly enforce scope, minimum
necessary disclosure, source class, freshness, and capability restrictions.

### T8 — Unsafe procedural memory

**Path.** A successful but unsafe trace is summarized as a reusable workflow;
preconditions disappear; secrets or attacker endpoints become constants; later
matching auto-executes it with broader capability.

Agent Workflow Memory demonstrates automatic induction and reuse of workflows
offline and online [awm25][awm25]. MPBench’s experience-to-procedure channel
specifically treats a reusable skill as a high-impact poisoning target
[untrusted26][untrusted26]. This motivates separate procedural admission and
execution boundaries; it does not establish that the proposed boundary is
sufficient.

### T9 — Deletion, correction, and rollback failure

**Path.** A raw item is deleted but remains in summaries, embeddings, caches,
exports, or backups; a later job recreates it from another derivative; a
correction hides the old value but cannot identify downstream actions; rollback
restores poisoned state or discards legitimate writes.

GDPR Articles 15–17 establish access, rectification, and erasure rights where
applicable, subject to the regulation’s conditions and exceptions
[gdpr16][gdpr16]. W3C PROV-O defines useful relations such as generation,
derivation, primary source, revision, and invalidation
[provo13][provo13]. Neither specifies a complete agent-memory deletion
algorithm. Dependency closure, tombstones, and rollback manifests are design
requirements.

### T10 — Poisoned dreams and consolidation denial of service

**Path.** An attacker creates many salient episodes to dominate replay;
cross-topic batching induces interference; recursive summaries multiply
records; expensive verification or counterfactual generation exhausts compute;
the offline job publishes a partial update after failure.

Useful Memories reports worse outcomes for heterogeneous groupings and repeated
updates in its evaluated settings [usefulmem26][usefulmem26]. MemPoison reports
cross-agent and tool-return channels as more corrupting than user input on its
benchmark [mempoison26][mempoison26]. Rate limits, diversity-aware sampling,
bounded lineage depth, transactional publish, and resource quotas are proposed
controls, not source-demonstrated solutions.

### T11 — Multi-agent identity, delegation, and concurrency failure

**Path.** Several agent processes share one project database. A receiver trusts
an authenticated peer’s message as instruction; a less-privileged agent induces
a more-privileged agent to call a tool; a delegation is copied or widened; a
worker publishes after its lease expires; or promotion races correction,
deletion, revocation, or a private-to-shared scope transition. A same-UID worker
may also bypass the intended protocol by directly opening, copying, locking, or
replacing supervisor database/WAL/socket state.

**Variants.**

- Agent identity is taken from message text instead of a host-authenticated
  connection or signed task envelope.
- A same-project record defaults to shared even though it is private to a
  principal or agent task.
- A receiver acts as a confused deputy because it checks its own capability but
  not the delegator, purpose, audience, or current user intent.
- A child delegation contains a capability, resource, duration, or
  re-delegation right absent from its parent.
- An expired, restarted, or partitioned worker publishes using a stale lease.
- Two consolidators derive from the same parent epoch and both believe they won.
- Promotion validates a root, then a concurrent transaction deletes it before
  publication.
- Retry after `SQLITE_BUSY`, timeout, or crash duplicates a write or tool
  effect.
- A worker forges rows, replays a socket token, swaps a symlink, copies WAL
  state, or holds a denial-of-service lock because filesystem modes do not
  isolate it from the supervisor.

SQLite normally isolates separate connections and serializes writes; WAL mode
gives readers snapshot isolation, and only one write transaction is active at a
time [sqliteisolation26][sqliteisolation26]
[sqlitetransactions26][sqlitetransactions26]. Those are physical transaction
properties. They do not decide whether a peer was authorized, which delegation
wins, whether a lease holder is stale, whether private content may become
shared, or whether deletion must dominate a concurrent promotion. Identity,
attenuation, version preconditions, fencing tokens, tombstone precedence, and
idempotency are separate semantic controls.
The design therefore requires a sole trusted supervisor plus an OS sandbox or
distinct service identity; shared multi-agent memory fails closed when worker
tools retain direct filesystem authority over supervisor state.

## Harm paths that controls must break

The central attack chain is:

`untrusted source → write influence → candidate admission → promotion → retention
→ authorized retrieval failure → instruction assimilation → capability use`.

Security evaluation must report each stage separately. A low final attack
success rate can hide a poisoned store that simply was not retrieved during the
test. Conversely, a retrieved poison may not affect a response. MemPoison
separates admission, retrieval, counterfactual causality, and behavioral
corruption; Hidden in Memory reports injection, retrieval, and conditional
adversarial use [mempoison26][mempoison26] [hidden26][hidden26]. Generalist
should additionally record authorization denial and tool-side enforcement.

## Assumptions

- The host runtime and cryptographic primitives are not fully compromised.
- Trusted clocks, tenant identities, tool schemas, and policy versions are
  available.
- Agent instances have host-authenticated identities; delegations and fencing
  counters cannot be forged by memory text.
- Raw episodes can be kept immutable apart from authorized tombstoning or
  cryptographic erasure.
- Tools enforce capabilities independently of model text.
- Operators can stop consolidation and restore a known-good manifest.

If these assumptions fail, prompt-layer memory controls cannot recover the
security boundary.

## Explicit non-goals

- Proving the foundation model itself is free of backdoors or memorized secrets.
- Guaranteeing truth for open-world claims.
- Treating anomaly or injection classifiers as complete detectors.
- Allowing memory content to grant new credentials or tool capabilities.
- Automatically executing model-written code because it passed benchmark
  tests.
- Claiming legal compliance from technical controls alone.

## Open questions

- Which sources can independently attest user preferences versus objective
  environmental facts?
- What risk tier requires a human promotion decision?
- How should a multi-user collaboration distinguish shared project facts from
  personal memories?
- Which same-project records may be shared among agents, and who can authorize
  the private-to-shared transition?
- Which service allocates monotonically increasing fencing tokens when several
  processes share the SQLite project database?
- Can counterfactual dreaming improve tests or planning without creating a
  harmful feedback loop?
- What backup erasure and audit commitments are operationally supportable?
- How should confidence decay when evidence is old but no fresher source exists?

## Local References

[sqliteisolation26]: SQLite Project. “Isolation In SQLite.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/isolation.html

[sqlitetransactions26]: SQLite Project. “Transaction.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/lang_transaction.html

[agentpoison24]: Chen, Zhaorun; Xiang, Zhen; Xiao, Chaowei; Song, Dawn; Li, Bo. “AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.” *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html

[minja25]: Dong, Shen; Xu, Shaochen; He, Pengfei; Li, Yige; Tang, Jiliang; Liu, Tianming; Liu, Hui; Xiang, Zhen. “Memory Injection Attacks on LLM Agents via Query-Only Interaction.” *Advances in Neural Information Processing Systems 38* (NeurIPS 2025). https://papers.nips.cc/paper_files/paper/2025/file/42a97bbd9844d2bf68596730af80bcdf-Paper-Conference.pdf

[mextra25]: Wang, Bo; He, Weiyi; Zeng, Shenglai; Xiang, Zhen; Xing, Yue; Tang, Jiliang; He, Pengfei. “Unveiling Privacy Risks in LLM Agent Memory.” *Proceedings of ACL 2025*, 25241–25260. https://aclanthology.org/2025.acl-long.1227/

[agentdojo24]: Debenedetti, Edoardo; Zhang, Jie; Balunović, Mislav; Beurer-Kellner, Luca; Fischer, Marc; Tramèr, Florian. “AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents.” *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track* (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets_and_Benchmarks_Track.html

[injecagent24]: Zhan, Qiusi; Liang, Zhixiang; Ying, Zifan; Kang, Daniel. “InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents.” *Findings of ACL 2024*, 10471–10506. https://aclanthology.org/2024.findings-acl.624/

[untrusted26]: Dash, Pritam; Ge, Tongyu; Jain, Aditi; Shah, Tanmay; Shang, Zhiwei. “From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents.” arXiv:2606.04329v2, preprint (2026). https://arxiv.org/abs/2606.04329

[badmemory26]: Gadgil, Soham; Alexander, David; Sunku, Sai; Roesner, Franziska. “Bad Memory: Evaluating Prompt Injection Risks from Memory in Agentic Systems.” arXiv:2607.14611v1, preprint (2026). https://arxiv.org/abs/2607.14611

[hidden26]: Pulipaka, Sidharth; Hlebik, Stanislau; Raghav, Leonidas; Abdelnabi, Sahar; Raina, Vyas; Sheth, Ivaxi; Fritz, Mario. “Hidden in Memory: Sleeper Memory Poisoning in LLM Agents.” arXiv:2605.15338v2, preprint under review (2026). https://arxiv.org/abs/2605.15338

[mempoison26]: Gao, Jifeng; Xia, Kang; Zhang, Yi; Hong, Xiaobin; Lin, Mingkai; Wei, Xingshen; Li, Wenzhong; Lu, Sanglu. “MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents.” arXiv:2607.14651v1, preprint (2026). https://arxiv.org/abs/2607.14651

[usefulmem26]: Zhang, Dylan; Lin, Yanshan; Wu, Zhengkun; Sun, Yihang; Li, Bingxuan; Li, Dianqi; Peng, Hao. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1, preprint (2026). https://arxiv.org/abs/2605.12978

[selfcorrect24]: Huang, Jie; Chen, Xinyun; Mishra, Swaroop; Zheng, Huaixiu Steven; Yu, Adams; Song, Xinying; Zhou, Denny. “Large Language Models Cannot Self-Correct Reasoning Yet.” *International Conference on Learning Representations* (ICLR 2024). https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html

[generativeagents23]: Park, Joon Sung; O'Brien, Joseph C.; Cai, Carrie J.; Morris, Meredith Ringel; Liang, Percy; Bernstein, Michael S. “Generative Agents: Interactive Simulacra of Human Behavior.” *Proceedings of UIST 2023*. https://doi.org/10.1145/3586183.3606763

[reflexion23]: Shinn, Noah; Cassano, Federico; Gopinath, Ashwin; Narasimhan, Karthik; Yao, Shunyu. “Reflexion: Language Agents with Verbal Reinforcement Learning.” *Advances in Neural Information Processing Systems 36* (NeurIPS 2023). https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html

[longmemeval25]: Wu, Di; Wang, Hongwei; Yu, Wenhao; Zhang, Yuwei; Chang, Kai-Wei; Yu, Dong. “LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.” *International Conference on Learning Representations* (ICLR 2025). https://openreview.net/forum?id=pZiyCaVuti

[memoryagentbench26]: Hu, Yuanzhe; Wang, Yu; McAuley, Julian. “Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.” *International Conference on Learning Representations* (ICLR 2026). https://openreview.net/forum?id=DT7JyQC3MR

[ragprivacy24]: Zeng, Shenglai; Zhang, Jiankun; He, Pengfei; Xing, Yue; Liu, Yiding; Xu, Han; Ren, Jie; Wang, Shuaiqiang; Yin, Dawei; Chang, Yi; Tang, Jiliang. “The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG).” *Findings of ACL 2024*, 4505–4524. https://aclanthology.org/2024.findings-acl.267/

[awm25]: Wang, Zora Zhiruo; Mao, Jiayuan; Fried, Daniel; Neubig, Graham. “Agent Workflow Memory.” *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267 (2025). https://proceedings.mlr.press/v267/wang25bx.html

[gdpr16]: European Parliament and Council. *Regulation (EU) 2016/679 (General Data Protection Regulation).* Official Journal of the European Union (2016). https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng/

[provo13]: World Wide Web Consortium. *PROV-O: The PROV Ontology.* W3C Recommendation (2013). https://www.w3.org/TR/prov-o/
