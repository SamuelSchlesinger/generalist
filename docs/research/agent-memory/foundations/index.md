# Foundations: Cognition, Replay, and Offline Learning

## Bottom line

The evidence supports a conservative, two-level principle: retain
high-fidelity, temporally grounded episodes and derive slower abstractions from
multiple episodes. It does **not** support deleting the episodes, treating a
summary as truth, or allowing unconstrained generated “dreams” to update
deployed behavior.

Three literatures reach that conclusion for different reasons:

- Complementary learning systems gives a computational account of why rapid
  event storage and slow distributed integration can coexist
  [mcclelland95][mcclelland95].
- Biological replay studies show structured reactivation and task-specific
  causal roles for replay-associated events, not a general consolidation
  algorithm [wilson94][wilson94].
- Machine-learning replay can reduce parameter forgetting, while model-based
  imagination can improve control; both rely on explicit objectives, update
  rules, and evaluation [shin17][shin17] [ha18][ha18].

For a fixed-weight tool-using LLM, the closest transfer is a **versioned,
evidence-preserving host transaction**. Retrieval, derived facts, procedures,
indexes, tests, and routing may change; the base model does not thereby learn in
its weights.

## Documents

- [Cognitive consolidation and replay](cognitive-consolidation.md)
  distinguishes biological observation, causal intervention, computational
  account, and metaphor.
- [Continual learning, rehearsal, and generative replay](continual-learning-replay.md)
  explains catastrophic parameter forgetting and what its mitigation
  mechanisms do and do not transfer.
- [World-model “dreaming” and offline reinforcement learning](dreaming-world-models.md)
  separates simulated rollouts, fixed-dataset policy learning, anticipatory
  compute, and host-side consolidation.
- [Transfer to a non-weight-updating agent runtime](runtime-transfer.md)
  derives a memory type system, consolidation transaction, and evidence
  invariants without claiming biological homology.

## Evidence ladder

| Level | What it can establish | Typical example here |
|---|---|---|
| Biological intervention | A manipulated neural event matters for a specific species, task, and outcome | Ripple disruption impairs a tested spatial-memory behavior |
| Biological observation | A neural pattern covaries with experience or later behavior | Place-cell pattern reactivation |
| Computational account | A mechanism can explain a trade-off under modeled assumptions | Fast sparse traces plus slow interleaved learning |
| ML experiment | An algorithm improves specified metrics on specified datasets and protocols | Replay reducing benchmark forgetting |
| Architectural inference | A defensible runtime mechanism suggested by the above | Preserve evidence and regression-gate derived memory |
| Metaphor | A naming analogy with no transferred guarantee | Calling a cron job “sleep” |

Evidence does not move up this ladder merely because a term is
neuroscience-inspired.

## What is established

### Biological and cognitive

- Event-specific and decontextualized knowledge are partially dissociable, but
  normally interactive [varghakhadem97][varghakhadem97].
- Experience-related neural patterns reactivate during sleep and quiet waking
  [wilson94][wilson94].
- Disrupting sharp-wave-ripple events impairs some spatial learning and memory
  tasks [girardeau09][girardeau09] [jadhav12][jadhav12].
- Consolidation can integrate overlap and accelerate learning when prior
  schemas exist [tse07][tse07] [tompary17][tompary17].
- Some replay-like sequences are constructive or goal-directed rather than
  literal copies [gupta10][gupta10] [pfeiffer13][pfeiffer13].

### Machine learning

- Sequential weight updates can catastrophically interfere with older tasks
  [mccloskey89][mccloskey89].
- Rehearsal, generative replay, selective regularization, gradient constraints,
  and architectural isolation can mitigate forgetting under tested protocols.
- Learned dynamics can support planning or policy learning from simulated
  trajectories [hafner25][hafner25].
- Offline RL is vulnerable to out-of-distribution actions and model or value
  exploitation; support constraints and pessimism are important
  [fujimoto19][fujimoto19].
- External retrieval can alter and sometimes improve a fixed model's outputs
  without further parameter training [khandelwal20][khandelwal20].

### New agent-memory evidence

RecMem, published in Findings of ACL 2026, reports that retaining cheap raw
interactions and delaying LLM consolidation until recurrence can reduce
construction work while preserving benchmark performance
[dai26][dai26]. A separate May 2026 preprint reports that repeatedly rewriting
textual memories can produce non-monotonic utility and regress on previously
solved cases [zhang26][zhang26].

The two results are complementary, not conclusive: **gate lossy consolidation,
preserve raw evidence, and test repeated updates**. Recurrence is only one
trigger and can miss rare critical events.

## What is not established

- Human memory has a clean database schema that should be copied.
- Sleep is required for an artificial system to consolidate memory.
- Repetition implies importance, correctness, or user consent.
- Semantic memory is more truthful than episodic memory.
- Replay resolves contradictions.
- Synthetic trajectories are observations.
- Offline execution is safe merely because the user is absent.
- Continual learning occurs in a fixed base model when only external text
  changes.
- A benchmark gain authorizes self-modification or external side effects.

## Mechanisms for the four target problems

| Problem | Supported ingredient | Required architectural addition |
|---|---|---|
| Abstraction | Slow integration across overlapping episodes | Scoped claims, evidence links, negative cases, held-out tests |
| Contradiction | Retrieval can reopen or juxtapose memories | Temporal truth model, source authority, supersession, quarantine |
| Forgetting | Replay and selective protection reduce interference | Separate retention, accessibility, influence, consent-driven deletion |
| Counterfactuals | Biological and learned models can construct novel trajectories | Generated-data type, evidence root, bounded horizon, verifier |

## Transfer rules

1. Preserve raw tool results and interaction events as immutable evidence.
2. Treat every summary, fact, procedure, and imagined trajectory as derived.
3. Record lineage, scope, time, model/prompt identity, and verifier status.
4. Challenge a proposal with disconfirming episodes selected independently.
5. Regression-test old and new cases before an atomic, reversible commit.
6. Never promote a counterfactual into the observed episode stream.
7. Keep offline work within the online agent's existing authority; default to no
   side effects.
8. Count “no change” as a valid consolidation result.

These are architectural inferences to be tested, not results proved by the
foundational papers.

## Known limitations

- Neuroscience results vary by species, task, sleep state, and replay
  definition.
- Continual-learning comparisons vary with task boundaries, model capacity,
  buffers, and evaluation protocol.
- World-model and offline-RL results assume state, action, and reward
  structures unlike open-ended tool use.
- The 2026 fixed-weight memory evidence is extremely recent; one source is a
  peer-reviewed benchmark paper and one remains a preprint.
- None of these literatures alone settles provenance, privacy, prompt injection,
  consent, deletion, or deployment authority.

## Supplementary code

No data artifact is included in this branch. It reports no independently
recomputed benchmark statistic; quantitative paper results are either avoided
or attributed directly to the primary source. The relevant validation work for
an agent runtime is a future longitudinal and adversarial evaluation, not a
recalculation from paper abstracts.

## Local References

URLs and publication status were checked on 2026-07-26.

[mcclelland95]: McClelland, James L.; McNaughton, Bruce L.; O'Reilly, Randall C. “Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights from the Successes and Failures of Connectionist Models of Learning and Memory.” *Psychological Review* 102(3), 419–457 (1995). https://doi.org/10.1037/0033-295X.102.3.419

[wilson94]: Wilson, Matthew A.; McNaughton, Bruce L. “Reactivation of Hippocampal Ensemble Memories During Sleep.” *Science* 265(5172), 676–679 (1994). https://doi.org/10.1126/science.8036517

[varghakhadem97]: Vargha-Khadem, Faraneh; Gadian, David G.; Watkins, Kate E.; Connelly, Alan; Van Paesschen, Wim; Mishkin, Mortimer. “Differential Effects of Early Hippocampal Pathology on Episodic and Semantic Memory.” *Science* 277(5324), 376–380 (1997). https://doi.org/10.1126/science.277.5324.376

[girardeau09]: Girardeau, Gabrielle; Benchenane, Karim; Wiener, Sidney I.; Buzsáki, György; Zugaro, Michaël B. “Selective Suppression of Hippocampal Ripples Impairs Spatial Memory.” *Nature Neuroscience* 12, 1222–1223 (2009). https://doi.org/10.1038/nn.2384

[jadhav12]: Jadhav, Shantanu P.; Kemere, Caleb; German, P. Walter; Frank, Loren M. “Awake Hippocampal Sharp-Wave Ripples Support Spatial Memory.” *Science* 336(6087), 1454–1458 (2012). https://doi.org/10.1126/science.1217230

[tse07]: Tse, Dorothy; Langston, Rosamund F.; Kakeyama, Masaki; Bethus, Ingrid; Spooner, Patrick A.; Wood, Emma R.; Witter, Menno P.; Morris, Richard G. M. “Schemas and Memory Consolidation.” *Science* 316(5821), 76–82 (2007). https://doi.org/10.1126/science.1135935

[tompary17]: Tompary, Alexa; Davachi, Lila. “Consolidation Promotes the Emergence of Representational Overlap in the Hippocampus and Medial Prefrontal Cortex.” *Neuron* 96(1), 228–241.e5 (2017). https://doi.org/10.1016/j.neuron.2017.09.005

[gupta10]: Gupta, Anoopum S.; van der Meer, Matthijs A. A.; Touretzky, David S.; Redish, A. David. “Hippocampal Replay Is Not a Simple Function of Experience.” *Neuron* 65(5), 695–705 (2010). https://doi.org/10.1016/j.neuron.2010.01.034

[pfeiffer13]: Pfeiffer, Brad E.; Foster, David J. “Hippocampal Place-Cell Sequences Depict Future Paths to Remembered Goals.” *Nature* 497, 74–79 (2013). https://doi.org/10.1038/nature12112

[mccloskey89]: McCloskey, Michael; Cohen, Neal J. “Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.” *Psychology of Learning and Motivation* 24, 109–165 (1989). https://doi.org/10.1016/S0079-7421(08)60536-8

[shin17]: Shin, Hanul; Lee, Jung Kwon; Kim, Jaehong; Kim, Jiwon. “Continual Learning with Deep Generative Replay.” *Advances in Neural Information Processing Systems 30* (2017). https://papers.nips.cc/paper_files/paper/2017/hash/0efbe98067c6c73dba1250d2beaa81f9-Abstract.html

[ha18]: Ha, David; Schmidhuber, Jürgen. “Recurrent World Models Facilitate Policy Evolution.” *Advances in Neural Information Processing Systems 31* (2018). https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution

[hafner25]: Hafner, Danijar; Pasukonis, Jurgis; Ba, Jimmy; Lillicrap, Timothy. “Mastering Diverse Control Tasks Through World Models.” *Nature* 640, 647–653 (2025). https://doi.org/10.1038/s41586-025-08744-2

[fujimoto19]: Fujimoto, Scott; Meger, David; Precup, Doina. “Off-Policy Deep Reinforcement Learning without Exploration.” *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 2052–2062 (2019). https://proceedings.mlr.press/v97/fujimoto19a.html

[khandelwal20]: Khandelwal, Urvashi; Levy, Omer; Jurafsky, Dan; Zettlemoyer, Luke; Lewis, Mike. “Generalization Through Memorization: Nearest Neighbor Language Models.” *International Conference on Learning Representations* (2020). https://openreview.net/forum?id=HklBjCEKvH

[dai26]: Dai, Zijie; Deng, Shiyuan; Guan, Sheng; Tian, Yizhou; Yao, Xin; Yan, Xiao; Cheng, James. “RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents.” *Findings of the Association for Computational Linguistics: ACL 2026*, 32353–32376 (2026). https://doi.org/10.18653/v1/2026.findings-acl.1619

[zhang26]: Zhang, Dylan; Lin, Yanshan; Wu, Zhengkun; Sun, Yihang; Li, Bingxuan; Li, Dianqi; Peng, Hao. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1 (2026). https://arxiv.org/abs/2605.12978
