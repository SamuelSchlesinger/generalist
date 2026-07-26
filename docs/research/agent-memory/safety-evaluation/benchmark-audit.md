# Benchmark Audit

## Bottom line

The benchmark landscape is now strong enough to test several distinct memory
capabilities, but not to prove safe longitudinal learning.

- LoCoMo, LongMemEval, and MemBench are primarily conversational/retrospective
  evaluations.
- MemoryAgentBench improves ecological validity by feeding information
  incrementally and covers retrieval, test-time learning, long-range
  understanding, and selective forgetting.
- PM-Bench evaluates whether an intention is executed, revised, or withheld at
  the right future cue while other activity continues.
- LongMemEval-V2 evaluates evidence gathering from large collections of
  pre-recorded web-agent trajectories, including workflow and environment
  knowledge.
- Existing security suites test important prompt-injection and poisoning paths,
  but no one suite covers cross-tenant isolation, secret minimization,
  generated-versus-observed provenance, calibration, deletion closure,
  rollback, agent identity/delegation, same-project private scopes,
  stale-worker publication, concurrent deletion/promotion, and poisoning across
  repeated offline consolidation cycles.

A benchmark score establishes performance on a declared artifact and protocol.
It is not evidence that a memory system is private, authorized, calibrated,
deletable, or safe under a different model, harness, tenant architecture, or
adversary.

## Audit method

This audit uses final proceedings where available and labels arXiv-only work as
preprint. Repository statements are used only where they clarify current
release or acceptance status. Scope and publication status were checked on
2026-07-26.

Coverage symbols below are our interpretation:

- `yes`: directly and materially evaluated;
- `partial`: present in a narrow or indirect form;
- `no`: not a stated evaluation target.

They do not score benchmark quality.

## Core utility and longitudinal benchmarks

### LoCoMo

LoCoMo is an ACL 2024 benchmark built from 10 long human-verified,
machine-generated conversations. The final paper reports an average of 588.2
turns, 27.2 sessions, and 16,618.1 tokens per conversation, with up to 32
sessions [locomo24][locomo24]. It provides:

- question answering over single-hop, multi-hop, temporal, open-domain, and
  adversarial/unanswerable cases;
- event-graph summarization; and
- multimodal dialogue generation.

What it supports: retrospective conversational recall, temporal/causal
comprehension, and consistency over months-long synthetic narratives.

What it does not support: incremental write-policy evaluation, tool actions,
adversarial memory writes, source authority, cross-user isolation, deletion, or
rollback. “Adversarial” QA in LoCoMo is an unanswerable-question category, not
persistent prompt-injection testing.

### LongMemEval

LongMemEval is an ICLR 2025 benchmark with 500 manually created questions
covering information extraction, multi-session reasoning, temporal reasoning,
knowledge updates, and abstention [longmemeval25][longmemeval25]. Its standard
histories include:

- LongMemEval-S, approximately 115k tokens per question; and
- LongMemEval-M, 500 sessions and approximately 1.5 million tokens.

The benchmark decomposes a memory system into indexing, retrieval, and reading,
which is useful for stage-level diagnosis.

What it supports: recall of user- and assistant-provided facts, updates,
temporal metadata, evidence retrieval, and abstention under long conversational
histories.

What it does not support: malicious write sources, tool-action safety, tenant
boundaries, privacy extraction, procedure synthesis, deletion, or repeated
consolidation. Correctly resolving its updates does not show that the system can
distinguish an authoritative correction from a malicious newer claim.

### MemBench

MemBench appeared in Findings of ACL 2025. It combines participation
(user-agent dialogue) and observation (passively received user messages)
scenarios, and distinguishes factual from reflective memory
[membench25][membench25]. The released construction starts from 500 user/entity
graphs. Its full data statistics include:

- participation/reflective: 3.5k sessions and 3.5k questions;
- participation/factual: 51k sessions and 39k questions;
- observation/reflective: 2k sessions and 2k questions; and
- observation/factual: 8.5k sessions and 8.5k questions.

It reports accuracy, recall, capacity, and temporal efficiency and evaluates
memory under sequential input rather than giving the complete history at once.
The paper’s own limitation is important: its evaluation centers on structured
profile/entity data, and reflective memory remains narrowly explored.

What it supports: factual versus inferred preference-like content,
participation versus observation, sequential input, scale/capacity, and
latency.

What it does not support: whether a “reflection” is evidentially justified,
source provenance, malicious content, tool actions, privacy isolation,
deletion, or rollback.

### MemoryAgentBench

MemoryAgentBench is an ICLR 2026 paper that reformulates existing long-context
data as incremental chunks and adds EventQA and FactConsolidation
[memoryagentbench26][memoryagentbench26]. The current paper reports 2,071
questions over contexts from 103k to 1.44M tokens and defines four
competencies:

1. accurate retrieval;
2. test-time learning;
3. long-range understanding; and
4. selective forgetting.

The authors evaluate long-context, RAG, and agentic-memory systems under a
shared protocol. FactConsolidation supplies single- and multi-hop version
conflicts, making this the most relevant core benchmark for incremental update
dynamics.

There is terminology drift worth recording: the paper calls the fourth
competency **selective forgetting**, while the current repository overview
labels the same slot **conflict resolution** [memoryagentrepo26][memoryagentrepo26].
Adaptations must pin the paper/repository version and dataset configuration.

What it supports: incremental ingestion, diverse memory mechanisms,
retrieval-versus-learning distinctions, long-range synthesis, and controlled
fact conflicts.

What it does not support: adversarial write authority, compositional or dormant
poisoning, privacy or tenant isolation, procedural capability safety, generated
provenance, deletion closure, or rollback.

### PM-Bench

PM-Bench is published at COLM 2026 and targets **prospective memory**: carrying
an intention through ongoing activity and acting only when its future cue or
time condition is satisfied [pmbench26][pmbench26]. The released synthetic week
has:

- seven days and 80 steps;
- 83 task definitions, 81 scored executable tasks;
- event- and time-based, regular and non-regular, cross-day, and
  channel-triggered intentions;
- 11 update events covering cancellation, override, and rescheduling;
- 11 hidden state channels requiring active monitoring; and
- 74 lure actions.

Its primary Set-F1 penalizes both missed due actions and false-positive action
spam. The paper evaluates eight models under eight configurations and shows a
real precision/recall/monitoring tradeoff: more reminders or queries do not
uniformly improve behavior.

What it supports: deferred intention maintenance, cancellation and
rescheduling, action timing, proactive state checks, false-positive action
cost, and cross-day interference.

What it does not support: persistent storage implementation, provenance,
malicious intention insertion, capability authorization, privacy, deletion, or
rollback. The reported experiments use the same fixed, deterministic,
synthetic benchmark week; generalization to real workflows must be separately
tested.

### LongMemEval-V2

LongMemEval-V2 is a May 2026 arXiv preprint, not yet a peer-reviewed proceedings
paper as of the access date [longmemevalv226][longmemevalv226]. It contains 451
manually curated questions over pre-collected web-agent trajectories and tests:

1. static state recall;
2. dynamic state tracking;
3. workflow knowledge;
4. environment gotchas; and
5. premise awareness.

Its Small and Medium tiers contain 100/500 trajectories and about 25M/115M
tokens. The memory system returns compact evidence to a fixed downstream reader.
The paper explicitly notes that this does not directly measure live online
learning or end-to-end task execution, and that browser environments do not
cover coding, computer-use, or enterprise agents.

What it supports: retrieval from real action/observation histories,
failed-trajectory lessons, workflow/environment knowledge, premise changes,
accuracy-latency tradeoffs, and evidence traceability.

What it does not support: online behavioral feedback, poisoning, privacy,
tenant boundaries, execution authorization, deletion, or rollback.

## Consolidation stress evidence

Useful Memories Become Faulty When Continuously Updated by LLMs is a May 2026
preprint, not a general-purpose benchmark [usefulmem26][usefulmem26]. Its
experiments are nevertheless a required diagnostic template:

- compare static whole-pool, grouped, and streaming update schedules;
- preserve raw episodic traces;
- compare forced consolidation, model-controlled consolidation,
  episodic-management-only, and abstraction-only conditions;
- vary whether traces contain ground-truth or model-running solutions; and
- evaluate after every consolidation round.

On the paper’s 19-problem ARC-AGI Stream slice, GPT-5.4 had 100% no-memory
accuracy, the stream of ground-truth solutions reached 52.6% at round ten, and
the refreshed static whole-pool condition retained 94.7%. These are Figure 2
values. The paper’s abstract says the system “fails on 54%,” page-one prose
says “fails on 46%,” and later prose rounds the plotted result differently; the
corpus therefore uses the figure values and does not reconcile those prose
percentages. The paper reports
grouping errors, interference, overgeneralization, and lost preconditions.
Raw-episodic or episodic-management-only conditions were competitive with or
better than consolidation-heavy conditions in several studied tasks.

Scope limits are substantial: natural-language consolidators, text benchmarks
plus a synthetic stream, current models, small repeat counts, and no formal
error bars for the highlighted small-n result. The result justifies an
episodic-only baseline and round-by-round evaluation; it does not establish
that all abstraction, streaming, or offline processing is unsafe.

## Security and privacy benchmarks

### AgentDojo and InjecAgent

AgentDojo is a NeurIPS 2024 dynamic environment with 97 tasks and 629 security
cases for agents using tools over untrusted data [agentdojo24][agentdojo24].
InjecAgent is a Findings of ACL 2024 suite with 1,054 cases over 17 user tools
and 62 attacker tools, targeting direct harm and private-data exfiltration
[injecagent24][injecagent24].

Both are valuable for testing what happens after a poisoned memory is rendered
as untrusted input. Neither is inherently longitudinal: the attack payload is
normally present in the current interaction. Adaptations must split injection
and trigger across sessions and inspect memory state between them.

### AgentPoison

AgentPoison is a NeurIPS 2024 attack, not a broad governance benchmark
[agentpoison24][agentpoison24]. It optimizes embedding-space triggers and
poisons memory/knowledge bases for three evaluated agents without model
training. It should seed tests for targeted retrieval, trigger transfer, low
poison rates, and benign-utility preservation. Its direct store/knowledge-base
poisoning assumption is stronger than query-only or indirect-write attacks.

### MINJA

MINJA is a NeurIPS 2025 query-only memory-injection attack
[minja25][minja25]. It removes the assumption that the attacker can directly
edit the store, using bridging steps and progressive shortening to cause
malicious records to be written and later retrieved. It should seed multi-turn,
black-box write induction tests. Its exact agents, models, and optimized
queries do not establish universal attack success.

### MPBench

MPBench is introduced by a June 2026 arXiv preprint
[untrusted26][untrusted26]. It contains 3,240 attack cases spanning six attack
classes and seven domain types plus 2,997 benign cases for false-positive
measurement. It separates memory-write success from retrieval-session success
and targets four write channels:

- explicit instruction-executed write;
- system-policy-driven inferred write;
- compaction-driven write; and
- experience-to-procedure write.

The study evaluates OpenClaw and HERMES with one model, GPT-OSS-120B. For some
email, Slack, and browser domains, external payloads are represented as labeled
untrusted blocks rather than arriving through a full tool pipeline. MPBench is
therefore strong write-channel coverage but not a production prevalence
estimate.

### Hidden in Memory

Hidden in Memory is a May 2026 arXiv preprint under review
[hidden26][hidden26]. Its black-box adversary controls a single external
document, not the store, and the attack is absent from the later trigger
session. The study separately reports injection, retrieval, and adversarial use
across tool-managed and external-manager memory regimes. It tests semantic,
manager-selected, and all-memory retrieval variants.

Its limitations include LLM-judge disagreement on borderline influence,
partially opaque provider pipelines, a single adversarial document per session,
and no systematic deletion, correction, user review, or provenance-aware
defense evaluation. It is especially useful for dormant, future-session, and
goal-adjacent retrieval tests.

### MemPoison-Bench

MemPoison-Bench is a July 2026 arXiv preprint with 1,227 hand-validated textual
cases [mempoison26][mempoison26]. It crosses:

- fact, instruction, preference, and state corruption;
- user input, tool return, and cross-agent injection;
- flat chunks, fact stores, and hierarchical notes; and
- L1 direct, L2 compositional multi-record, and L3 context-triggered dormant
  attacks.

It reports clean accuracy and behavioral corruption, then decomposes defended
failures into write-blocked, admitted/not retrieved, retrieved/noncausal, and
residual-causal stages. Its key value is structural: a pointwise write filter
cannot see every future record composition or trigger. The paper is a first
preprint, focuses on text, and does not evaluate access-control or decay
architectures.

### Bad Memory

Bad Memory is a July 2026 arXiv preprint using a sandboxed synthetic workspace
across Claude Code and Codex [badmemory26][badmemory26]. It plants credential
exfiltration, unauthorized tool-use, and brand-targeting instructions in
auto-loaded or referenced memory files and measures attack success and payload
persistence over short multi-session sequences.

The adversary is assumed already able to control a persistent workspace file.
The authors report that external ingestion was difficult in their preliminary
setup, and their main results use ten trials per condition in a synthetic
workspace. It is therefore evidence that durable, apparently authoritative
files are dangerous—not evidence that every agent will self-write the payload.

### MEXTRA

MEXTRA is an ACL 2025 black-box memory-extraction attack evaluated on two
representative memory-augmented agents [mextra25][mextra25]. It motivates
adaptive extraction tests and stage-level disclosure metrics. It does not by
itself test cross-tenant search, deletion, minimization, or a particular
Generalist store.

## Coverage matrix

| Benchmark | Incremental writes | Updates / conflicts | Prospective intentions | Tool/action use | Adversarial write | Compositional / dormant | Privacy extraction | Deletion / rollback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | partial | no | no | no | no | no | no | no |
| LongMemEval | partial | yes | no | no | no | no | no | no |
| MemBench | yes | partial | no | no | no | no | no | no |
| MemoryAgentBench | yes | yes | no | no | no | no | no | no |
| PM-Bench | sequential intentions | yes | yes | structured action selection | no | no | no | no |
| LongMemEval-V2 | pre-collected | yes | no | trajectory evidence only | no | no | no | no |
| AgentDojo | no | no | no | yes | current-context only | adaptive injection, not persistent | partial exfiltration | no |
| InjecAgent | no | no | no | yes | current-context only | no | yes | no |
| AgentPoison | poisoned store | no | no | partial | yes | optimized trigger | no | no |
| MINJA | yes | no | no | partial | yes, query-only | multi-step injection | no | no |
| MPBench | yes | partial | no | yes | yes | write-channel taxonomy | no | no |
| Hidden in Memory | yes | no | no | yes | yes, external document | sleeper | no | no |
| MemPoison-Bench | yes | partial | no | yes | yes | yes | no | no |
| Bad Memory | file persistence | no | no | yes | planted file | short sequence | credential target | no |
| MEXTRA | no | no | no | no | no | no | yes | no |

No row directly tests generated-versus-observed source-class confusion,
independent-evidence counting, calibrated memory confidence, complete
cross-tenant noninterference, lineage-wide deletion, or rollback that preserves
user deletion. It also does not test authenticated agent identity, confused
deputies, attenuated delegation, same-project private-versus-shared scopes,
fencing of stale workers, or semantic races between promotion and deletion.

## Required Generalist composition

The evaluation should not report one blended “memory score.” It should compose:

1. LoCoMo/LongMemEval/MemBench for conversational recall, update, reflection,
   capacity, and latency;
2. MemoryAgentBench for incremental retrieval, test-time learning, long-range
   synthesis, and controlled forgetting/conflicts;
3. PM-Bench for deferred intentions, cancellation, rescheduling, monitoring,
   and false-positive action;
4. LongMemEval-V2-style trajectory evidence for environment and workflow
   memory;
5. Useful Memories-style round-by-round consolidation schedules;
6. AgentDojo/InjecAgent adaptations for tool-side harm and exfiltration;
7. AgentPoison, MINJA, MPBench, Hidden in Memory, MemPoison, and Bad Memory
   threat variants; and
8. new tests for privacy scope, provenance, calibration, deletion, rollback,
   dream-origin confusion, agent identity/delegation, and concurrent-process
   state transitions.

Every task family must compare at least:

- no memory;
- immutable episodic-only memory;
- episodic retrieval without consolidation;
- unguarded consolidation;
- guarded candidate/promote consolidation; and
- guarded consolidation with generated counterfactuals disabled/enabled where
  relevant.

Without the first three controls, a gain cannot be attributed to consolidation.
Without unguarded versus guarded comparison, a safety improvement can be
confounded with simply disabling useful writes.

## Reporting cautions

- Pin benchmark commit, data version, model, prompt, tool, retriever, index, and
  judge.
- Separate fresh inference from replay-based ablations.
- Do not compare scores across tasks with different answer judges as if they
  shared one scale.
- Report question- or scenario-level uncertainty, not just trial-to-trial model
  variance.
- Treat interrupted or failed agent runs as declared outcomes, not silently
  dropped samples.
- Evaluate both clean utility and attack outcomes. A defense that never stores
  or acts can look secure while being useless.
- Report write, promotion, retrieval, behavioral influence, tool denial, and
  persistence separately.
- Preserve exact poisoned records and expected clean/poison behavior as
  reproducible certificates where release is safe.

## Local References

[locomo24]: Maharana, Adyasha; Lee, Dong-Ho; Tulyakov, Sergey; Bansal, Mohit; Barbieri, Francesco; Fang, Yuwei. “Evaluating Very Long-Term Conversational Memory of LLM Agents.” *Proceedings of ACL 2024*, 13851–13870. https://aclanthology.org/2024.acl-long.747/

[longmemeval25]: Wu, Di; Wang, Hongwei; Yu, Wenhao; Zhang, Yuwei; Chang, Kai-Wei; Yu, Dong. “LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.” *International Conference on Learning Representations* (ICLR 2025). https://openreview.net/forum?id=pZiyCaVuti

[membench25]: Tan, Haoran; Zhang, Zeyu; Ma, Chen; Chen, Xu; Dai, Quanyu; Dong, Zhenhua. “MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents.” *Findings of ACL 2025*, 19336–19352. https://aclanthology.org/2025.findings-acl.989/

[memoryagentbench26]: Hu, Yuanzhe; Wang, Yu; McAuley, Julian. “Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.” *International Conference on Learning Representations* (ICLR 2026). https://openreview.net/forum?id=DT7JyQC3MR

[memoryagentrepo26]: Hu, Yuanzhe; Wang, Yu; McAuley, Julian. *MemoryAgentBench official code repository and release notes.* Accessed 2026-07-26. https://github.com/HUST-AI-HYZ/MemoryAgentBench

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
