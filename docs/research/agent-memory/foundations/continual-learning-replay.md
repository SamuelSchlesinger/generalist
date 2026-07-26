# Continual Learning, Rehearsal, and Generative Replay

## The problem is parameter interference

In machine learning, **catastrophic forgetting** normally means a sharp loss on
previously learned tasks after a parameterized model is trained on later,
non-stationary data. McCloskey and Cohen demonstrated the sequential-learning
problem in connectionist simulations and traced it to new updates changing
weights that also represent older items [mccloskey89][mccloskey89].

That definition matters. A fixed-weight language model with an external memory
store does not undergo catastrophic forgetting merely because an episode falls
out of retrieval. It can still suffer important but different failures:

- **retention failure**: an episode is deleted or overwritten;
- **retrieval failure**: the right item exists but is not selected;
- **compression failure**: a summary omits or changes relevant detail;
- **interference in context**: retrieved items distract or contradict one
  another; and
- **policy drift through memory**: changed instructions or lessons alter future
  behavior even though model parameters are fixed.

The continual-learning literature is evidence about the first problem only
when parameters are updated. Its mechanisms can inspire host-side controls, but
their benchmark results do not transfer automatically.

## What the major mechanism families establish

### Rehearsal or experience replay

The simplest defense is to mix stored old examples with new examples during
training. Replay changes the effective training distribution so gradients are
not computed from the newest data alone.

In continual reinforcement learning, CLEAR combines replayed transitions,
behavioral cloning on past behavior, and on-policy learning; the reported
experiments found substantially less forgetting across the studied multi-task
environments without requiring task identity [rolnick19][rolnick19]. This is
evidence that replay can be a strong, relatively simple baseline in those
settings, not that replay eliminates forgetting generally.

Memory selection matters. Maximally Interfered Retrieval chooses examples whose
predictions would be most harmed by a prospective update, and its experiments
improved over random replay on the tested online continual-learning benchmarks
[aljundi19][aljundi19]. This supplies a useful computational idea:
**prioritize evidence at risk of being contradicted by the next change**.

Replay has costs and assumptions:

- it retains training examples, with privacy and storage consequences;
- a bounded buffer underrepresents something by construction;
- old examples may be stale or poisoned;
- repeated examples can dominate a skewed stream; and
- an objective that preserves old predictions can also preserve old errors.

### Gradient constraints

Gradient Episodic Memory (GEM) stores examples from prior tasks and projects a
new gradient when it would increase loss on those memories
[lopezpaz17][lopezpaz17]. It operationalizes “do not make remembered tasks
worse” at update time.

This is an empirical parameter-training method. For a fixed-weight host, the
closest analogue is not gradient projection. It is a **pre-commit regression
gate**: test a proposed memory or procedure against retained old cases, and
reject or scope the update if performance regresses.

### Parameter regularization

Elastic Weight Consolidation (EWC) estimates which parameters matter for prior
tasks and penalizes changing them during later training
[kirkpatrick17][kirkpatrick17]. Synaptic Intelligence similarly accumulates an
online importance measure and discourages changes to important parameters
[zenke17][zenke17].

These methods establish that selective plasticity can reduce forgetting on
their evaluated task sequences. They do not identify facts, retain raw
evidence, detect contradictions, or generate abstractions. A host-side analogue
is field- or record-level immutability: make evidence-bearing fields harder to
rewrite than caches or derived summaries.

### Architectural isolation

Progressive Neural Networks allocate a new column for each task, freeze prior
columns, and use lateral connections for transfer
[rusu16][rusu16]. This avoids forgetting by spending capacity and assuming task
segmentation.

The host-side lesson is structural: **append or version before overwrite**.
Preserving an old memory version makes rollback possible, but unbounded
versioning creates retrieval and storage problems. Isolation relocates the
capacity trade-off; it does not solve it.

### Generative replay

Generative replay replaces stored old examples with samples from a learned
generator. Deep Generative Replay trains a new solver on new-task data
interleaved with generated pseudo-examples representing previous tasks
[shin17][shin17]. Later work combined a generative model with replay and
parameter-importance mechanisms on class-incremental benchmarks
[vandeven20][vandeven20].

The method can reduce raw-example storage, but its evidence type is different:

- generated samples approximate an earlier training distribution;
- the generator can itself forget or omit low-probability cases;
- errors can be replayed into the next generation;
- generated labels or targets come from an older model, not external truth; and
- distributional resemblance does not preserve provenance.

Generative replay is therefore evidence that synthetic rehearsal can preserve
task performance under tested objectives. It is not evidence that fabricated
episodes may safely enter an agent's autobiographical record.

## What replay does and does not address

| Mechanism | Forgetting | Abstraction | Contradiction | Counterfactuals |
|---|---|---|---|---|
| Raw rehearsal | Directly mitigates parameter interference | Only through the learner | May preserve both sides without resolving them | No |
| Risk-based selection | Focuses replay on vulnerable items | No guarantee | Can surface likely interference | No |
| Gradient constraint | Bounds regression on stored examples | No | Encodes “do not worsen,” not truth | No |
| Parameter regularization | Slows important parameter changes | No | No proposition-level logic | No |
| Architectural isolation | Prevents overwrite of frozen components | Transfer through added structure | Keeps versions separate | No |
| Generative replay | Rehearses an approximate old distribution | Generator may capture regularities | Can reproduce or amplify inconsistency | Yes, but synthetic by construction |

None of these methods supplies semantic contradiction resolution. The training
objective defines what counts as preservation. If the loss, labels, buffer, or
generator are wrong, successful replay can faithfully preserve the wrong thing.

## Replay selection is a policy

Uniform random replay is only one policy. Plausible selectors include:

- recency;
- rarity;
- surprise or prediction error;
- reward or failure severity;
- expected future use;
- expected interference from a proposed update;
- coverage across tasks, users, tools, or time; and
- explicit protected status.

The Aljundi et al. results justify testing interference-aware selection
[aljundi19][aljundi19], but not treating it as universally optimal. A selector
can create a self-confirming loop: memories retrieved often appear useful, so
they are replayed more, while never-retrieved counterevidence disappears.
Evaluation must therefore measure coverage and neglected cases as well as
average downstream performance.

## Translation to a fixed-weight agent

### Transfers cleanly

- **Interleave evidence**: evaluate a proposed abstraction against old and new
  episodes, not only the latest cluster.
- **Keep exemplars**: summaries should not be the sole surviving record.
- **Select deliberately**: make replay priority inspectable and test its
  coverage.
- **Protect high-value state**: immutable raw results, user-approved
  constraints, and rollback points deserve stronger update rules than caches.
- **Regression-test consolidation**: query a held-out longitudinal case set
  before committing a derived memory.

### Transfers only as analogy

- Replaying text into a prompt is not weight rehearsal.
- Preventing a memory rewrite is not EWC.
- Versioning a procedure is not a progressive neural network.
- Asking an LLM to invent examples is not evidence that generative replay
  preserved a true distribution.

### Does not transfer without a training subsystem

- gradient projection;
- Fisher-information or path-integral parameter penalties;
- distillation into network weights;
- policy-gradient updates; and
- claims about resistance to parameter forgetting.

## Direct evidence for external textual memory

The 2026 preprint *Useful Memories Become Faulty When Continuously Updated by
LLMs* studies the fixed-weight setting more directly than classic continual
learning. Across its tested agent tasks, repeatedly rewriting trajectories into
textual lessons produced non-monotonic utility: early gains could be followed by
regression, while retaining raw episodes remained competitive. In a controlled
memory-management environment, agents preferred retaining episodes, and forced
consolidation performed worse [zhang26][zhang26].

This is new preprint evidence, not an established general law. It supports three
design hypotheses:

1. raw trajectories should remain first-class evidence;
2. consolidation should be gated rather than triggered on every interaction;
3. a derived memory should be evaluated across repeated future updates, not
   only immediately after creation.

It does not prove that all consolidation is harmful or that episodic-only memory
is sufficient. The results depend on the tested tasks, prompts, models, update
schedules, and retrieval policies.

There is also a reporting ambiguity worth preserving rather than smoothing
over: the arXiv abstract says the model “fails on 54%” of a selected ARC-AGI
set, while the authors' project page describes accuracy falling to 54%. This
document therefore does not convert that result into a numeric accuracy claim.

## A defensible non-weight replay loop

The following is architectural inference from the evidence above:

1. Select a bounded set of raw episodes using an explicit, logged policy.
2. Separate observations, tool outputs, model interpretations, and rewards.
3. Ask the model to propose—not commit—facts, procedures, scopes, and
   counterfactuals.
4. Retrieve supporting and disconfirming episodes independently.
5. Run old and new longitudinal test cases with and without the candidate.
6. Commit a versioned derived item only if it meets declared gates.
7. Keep evidence links and the prior version; allow later supersession without
   rewriting history.

This loop can improve retrieval and procedures without changing model weights.
It is consolidation in a functional sense, not continual neural learning.

## Uncertainty and access status

Continual-learning results are protocol-sensitive: task boundaries, task IDs,
buffer sizes, class order, model capacity, and evaluation timing change the
problem. Conference, DOI, and arXiv URLs were checked on 2026-07-26.
Zhang et al. is arXiv v1 and has not, from the source checked, been established
as peer reviewed.

## Local References

[mccloskey89]: McCloskey, Michael; Cohen, Neal J. “Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.” *Psychology of Learning and Motivation* 24, 109–165 (1989). https://doi.org/10.1016/S0079-7421(08)60536-8

[rolnick19]: Rolnick, David; Ahuja, Arun; Schwarz, Jonathan; Lillicrap, Timothy; Wayne, Gregory. “Experience Replay for Continual Learning.” *Advances in Neural Information Processing Systems 32* (2019). https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html

[aljundi19]: Aljundi, Rahaf; Belilovsky, Eugene; Tuytelaars, Tinne; Charlin, Laurent; Caccia, Massimo; Lin, Min; Page-Caccia, Lucas. “Online Continual Learning with Maximally Interfered Retrieval.” *Advances in Neural Information Processing Systems 32* (2019). https://proceedings.neurips.cc/paper/2019/hash/15825aee15eb335cc13f9b559f166ee8-Abstract.html

[lopezpaz17]: Lopez-Paz, David; Ranzato, Marc'Aurelio. “Gradient Episodic Memory for Continual Learning.” *Advances in Neural Information Processing Systems 30* (2017). https://papers.nips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html

[kirkpatrick17]: Kirkpatrick, James; Pascanu, Razvan; Rabinowitz, Neil; et al. “Overcoming Catastrophic Forgetting in Neural Networks.” *Proceedings of the National Academy of Sciences* 114(13), 3521–3526 (2017). https://doi.org/10.1073/pnas.1611835114

[zenke17]: Zenke, Friedemann; Poole, Ben; Ganguli, Surya. “Continual Learning Through Synaptic Intelligence.” *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 3987–3995 (2017). https://proceedings.mlr.press/v70/zenke17a.html

[rusu16]: Rusu, Andrei A.; Rabinowitz, Neil C.; Desjardins, Guillaume; Soyer, Hubert; Kirkpatrick, James; Kavukcuoglu, Koray; Pascanu, Razvan; Hadsell, Raia. “Progressive Neural Networks.” arXiv:1606.04671 (2016). https://arxiv.org/abs/1606.04671

[shin17]: Shin, Hanul; Lee, Jung Kwon; Kim, Jaehong; Kim, Jiwon. “Continual Learning with Deep Generative Replay.” *Advances in Neural Information Processing Systems 30* (2017). https://papers.nips.cc/paper_files/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html

[vandeven20]: van de Ven, Gido M.; Siegelmann, Hava T.; Tolias, Andreas S. “Brain-Inspired Replay for Continual Learning with Artificial Neural Networks.” *Nature Communications* 11, 4069 (2020). https://doi.org/10.1038/s41467-020-17866-2

[zhang26]: Zhang, Dylan; Lin, Yanshan; Wu, Zhengkun; Sun, Yihang; Li, Bingxuan; Li, Dianqi; Peng, Hao. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1 (2026). https://arxiv.org/abs/2605.12978
