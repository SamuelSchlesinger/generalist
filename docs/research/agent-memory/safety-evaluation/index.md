# Safety, Governance, and Evaluation

## Verdict

An episodic-memory runtime can improve continuity only if memory is treated as
an **untrusted, versioned evidence system**, not as an extension of the system
prompt. Offline “dreaming” adds a second trust boundary: model-generated
reflections, summaries, counterfactuals, and procedures are candidates for
testing, not observations and not self-authenticating evidence.

The safety case therefore rests on six separations:

1. immutable raw episodes from revisable derived memories;
2. observations from model-generated material;
3. candidate writes from promoted memories;
4. relevance from authority and permission;
5. stored information from executable procedure or tool capability; and
6. authenticated agent identity from delegated authority and trusted content.

No single content filter closes the threat model. Peer-reviewed work already
shows persistent memory/knowledge-base poisoning and query-only memory
injection [agentpoison24][agentpoison24] [minja25][minja25]. Four recent
preprints broaden the empirical warning to write-channel attacks, memory-file
instructions, dormant activation, and harm that emerges only when plausible
records are composed [untrusted26][untrusted26] [badmemory26][badmemory26]
[hidden26][hidden26] [mempoison26][mempoison26]. Those 2026 results are useful
threat hypotheses, not settled prevalence estimates.

## Evidence discipline

Every substantive statement in this branch is one of:

- **Sourced result** — a result reported by a cited primary paper or official
  standard, with venue/status and important scope limits stated nearby.
- **Design inference** — a control, architecture, test, or acceptance rule
  proposed here. It may be motivated by sources but has not thereby been
  validated for Generalist.
- **Open question** — a claim that requires implementation evidence,
  longitudinal experiments, or a policy/legal decision.

Preprints are called preprints. A benchmark score is never treated as a
production-security guarantee. A successful deletion from an external memory
store is not evidence that information was removed from model weights or every
backup.

## Documents

- [Threat model](threat-model.md) defines assets, adversaries, trust boundaries,
  failure paths, and the special risks created by consolidation, generated
  episodes, and concurrent agents sharing project state.
- [Benchmark audit](benchmark-audit.md) identifies what LoCoMo, LongMemEval,
  MemoryAgentBench, newer agent-memory benchmarks, and adversarial suites do and
  do not test.
- [Control architecture](control-architecture.md) specifies provenance,
  candidate/promote transactions, independent verification, temporal and
  contradiction handling, scoped retrieval, procedural isolation, privacy,
  deletion, rollback, agent delegation, fencing, and semantic concurrency.
- [Evaluation plan](evaluation-plan.md) turns the threat model into longitudinal
  utility, security, privacy, calibration, and lifecycle tests.
- [Threat-control matrix](data/threat-control-matrix.csv) is a machine-readable
  coverage ledger. It identifies preventive, detective, corrective, and
  evaluation controls without implying that a listed control is sufficient.
- [Branch validator](data/validate-branch.py) checks local links, lowercase
  citation keys, citation resolution, cross-document metadata consistency,
  index coverage, and the presence of local bibliographies.

## Non-negotiable design invariants

These are **design requirements**, not sourced empirical findings:

- A generated item can never be relabeled as an observation.
- Capture exactness applies to retained canonical bytes after trusted admission;
  redacted, secret-reference, omitted, and incomplete spans stay explicit.
- Promotion never mutates or deletes its supporting raw episodes.
- Evidence support is counted by independent provenance roots, not by the
  number of summaries descended from one root.
- Untrusted text may supply data but cannot grant tool authority, change tenant
  scope, or promote itself.
- Cross-agent messages are untrusted input even when the sender is
  authenticated.
- Delegation can only attenuate capability, resource, audience, purpose, and
  lifetime; a receiver rechecks current user intent and host policy.
- Same-project memory is not automatically shared: private-to-shared promotion
  is an explicit authorized state transition.
- Retrieval is authorized before ranking and is bounded by tenant, principal,
  purpose, sensitivity, time, source class, and capability.
- A procedural memory is inert text until separately approved, sandboxed, and
  supplied with an explicit capability manifest.
- Contradictory claims remain jointly inspectable; “last write wins” is allowed
  only for fields whose authoritative ordering is known.
- Deletion creates a resurrection-blocking tombstone and traverses all known
  derivatives, indexes, caches, exports, and backup-retention paths.
- Every promoted item can be traced to a promotion decision, policy version,
  verifier outputs, and evidence roots.
- Offline consolidation has a kill switch, bounded budget, atomic commit, and
  rollback manifest.
- Every publisher presents a current fencing token and expected parent version;
  stale workers and duplicate retries cannot commit.
- A trusted supervisor is the sole database/key/ledger opener; same-UID file
  modes do not authorize multi-agent sharing.
- A tombstone or revocation dominates a concurrent promotion or rollback.
- SQLite transaction locks provide storage serialization, not agent identity,
  delegation, sharing policy, or deletion precedence
  [sqliteisolation26][sqliteisolation26]
  [sqlitetransactions26][sqlitetransactions26].

## Release gates

Memory should remain read-only and episodic until the evaluation plan shows:

- no cross-tenant retrieval or secret disclosure in the tested threat model;
- no unauthorized high-impact tool action attributable to memory;
- deletion and rollback completion across every instrumented storage tier;
- no identity confusion, privilege amplification, stale-worker publication, or
  private-to-shared leakage or supervisor bypass under concurrent-process
  schedules;
- calibrated abstention on unsupported, stale, and contradictory claims;
- utility gains over an immutable episodic-only baseline, not merely over
  no-memory;
- bounded poisoning persistence under direct, query-only, indirect,
  compositional, and dormant attacks; and
- acceptable latency, token, storage, and operator-review costs.

The zero-event gates above must be reported with confidence bounds over a
declared test population. “Zero observed” does not mean impossible.

## Governance baseline

NIST AI RMF 1.0 frames risk work as continuous Govern, Map, Measure, and Manage
functions [nistairmf23][nistairmf23]. Its Generative AI Profile is voluntary
guidance, not a certification regime [nistgenai24][nistgenai24]. GDPR
principles and data-subject rights motivate purpose limitation, minimization,
accuracy, storage limitation, access, rectification, and erasure where the
regulation applies [gdpr16][gdpr16]. These sources justify governance questions;
they do not decide Generalist’s jurisdiction, lawful basis, retention period,
or product policy.

## Current status

Source and publication status were checked on 2026-07-26. The documents specify
a defensible design and test program; they do **not** establish that a deployed
Generalist memory implementation is secure, private, legally compliant, or
better than an episodic-only system.

## Local References

[sqliteisolation26]: SQLite Project. “Isolation In SQLite.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/isolation.html

[sqlitetransactions26]: SQLite Project. “Transaction.” Official documentation, accessed 2026-07-26. https://www.sqlite.org/lang_transaction.html

[agentpoison24]: Chen, Zhaorun; Xiang, Zhen; Xiao, Chaowei; Song, Dawn; Li, Bo. “AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases.” *Advances in Neural Information Processing Systems 37* (NeurIPS 2024). https://papers.nips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html

[minja25]: Dong, Shen; Xu, Shaochen; He, Pengfei; Li, Yige; Tang, Jiliang; Liu, Tianming; Liu, Hui; Xiang, Zhen. “Memory Injection Attacks on LLM Agents via Query-Only Interaction.” *Advances in Neural Information Processing Systems 38* (NeurIPS 2025). https://papers.nips.cc/paper_files/paper/2025/file/42a97bbd9844d2bf68596730af80bcdf-Paper-Conference.pdf

[untrusted26]: Dash, Pritam; Ge, Tongyu; Jain, Aditi; Shah, Tanmay; Shang, Zhiwei. “From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents.” arXiv:2606.04329v2, preprint (2026). https://arxiv.org/abs/2606.04329

[badmemory26]: Gadgil, Soham; Alexander, David; Sunku, Sai; Roesner, Franziska. “Bad Memory: Evaluating Prompt Injection Risks from Memory in Agentic Systems.” arXiv:2607.14611v1, preprint (2026). https://arxiv.org/abs/2607.14611

[hidden26]: Pulipaka, Sidharth; Hlebik, Stanislau; Raghav, Leonidas; Abdelnabi, Sahar; Raina, Vyas; Sheth, Ivaxi; Fritz, Mario. “Hidden in Memory: Sleeper Memory Poisoning in LLM Agents.” arXiv:2605.15338v2, preprint under review (2026). https://arxiv.org/abs/2605.15338

[mempoison26]: Gao, Jifeng; Xia, Kang; Zhang, Yi; Hong, Xiaobin; Lin, Mingkai; Wei, Xingshen; Li, Wenzhong; Lu, Sanglu. “MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents.” arXiv:2607.14651v1, preprint (2026). https://arxiv.org/abs/2607.14651

[nistairmf23]: Tabassi, Elham. *Artificial Intelligence Risk Management Framework (AI RMF 1.0).* NIST AI 100-1 (2023). https://doi.org/10.6028/NIST.AI.100-1

[nistgenai24]: Autio, Chloe; Schwartz, Reva; Dunietz, Jesse; Jain, Shomik; Stanley, Martin; Tabassi, Elham; Hall, Patrick; Roberts, Kamie. *Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile.* NIST AI 600-1 (2024). https://doi.org/10.6028/NIST.AI.600-1

[gdpr16]: European Parliament and Council. *Regulation (EU) 2016/679 (General Data Protection Regulation).* Official Journal of the European Union (2016). https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng/
