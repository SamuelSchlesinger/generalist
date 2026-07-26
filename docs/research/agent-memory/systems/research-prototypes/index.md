# Research Prototypes

Publication and source status in this section was checked on 2026-07-26.
The research systems differ less in whether they “have memory” than in what
they promote out of an interaction and when they pay for abstraction. The
table below names the actual persistent artifact rather than the authors’
cognitive metaphor.

| System | Durable artifact | Write or consolidation trigger | Recall | Revision or forgetting | What improves |
| --- | --- | --- | --- | --- | --- |
| Generative Agents | timestamped natural-language observations, reflections, and plans | every observation; reflection after accumulated importance crosses a threshold | weighted recency, importance, and embedding relevance | no specified deletion; reflection adds derived records | prompts receive selected past records |
| Reflexion | short verbal reflections over failed trials | task feedback after a trial | a small recent episodic buffer | sliding-window eviction | retries receive verbal feedback |
| Voyager | executable code skills indexed by generated descriptions | successful self-verification | embedding search, top five | no paper-level conflict or deletion policy | plans reuse procedures |
| ExpeL | successful trajectories plus a voted list of distilled insights | offline training trajectories and LLM comparison | nearest successful examples plus the full insight list | add, edit, upvote, downvote, delete at zero votes | prompts receive cases and rules |
| RecMem | raw turns, episodic narratives, atomic semantic facts | raw write every turn; abstraction only after semantic recurrence | fixed-budget retrieval from all three stores | semantic refinement; deletion left as future operational work | prompts receive multi-level records |

All five keep the base model fixed. “Learning” means changing external records
and the next prompt, not updating model parameters. That distinction matters:
the systems can improve empirically while still inheriting the generator’s
hallucinations, evaluator errors, and context sensitivity.[reflexion][reflexion]
[voyager][voyager] [expel][expel]

## Reading path

- [Generative Agents](generative-agents.md) established the
  observation–retrieval–reflection loop.
- [Reflexion, Voyager, and ExpeL](reflexion-voyager-expel.md) compares verbal
  feedback, executable skills, and cross-trajectory insight extraction, then
  traces later refinements.
- [RecMem](recmem.md) makes consolidation conditional on repeated semantic
  evidence and retains unpromoted turns as a queryable safety net.

## What these prototypes establish

The strongest transferable result is architectural: a frozen LLM can use
external experience at several abstraction levels. The studies do **not**
establish a universal memory policy. Their environments, evaluators, write
budgets, and retrieval budgets differ; most evidence comes from bounded
benchmarks or simulations. A production design still needs access control,
provenance, contradiction handling, deletion semantics, and poisoning
resistance that the prototype papers either omit or treat as future work.

## Local References

[expel]: Andrew Zhao et al. “ExpeL: LLM Agents Are Experiential Learners.” AAAI 2024. https://arxiv.org/abs/2308.10144 (accessed 2026-07-26).

[reflexion]: Noah Shinn et al. “Reflexion: Language Agents with Verbal Reinforcement Learning.” NeurIPS 2023. https://arxiv.org/abs/2303.11366 (accessed 2026-07-26).

[voyager]: Guanzhi Wang et al. “Voyager: An Open-Ended Embodied Agent with Large Language Models.” 2023. https://arxiv.org/abs/2305.16291 (accessed 2026-07-26).
