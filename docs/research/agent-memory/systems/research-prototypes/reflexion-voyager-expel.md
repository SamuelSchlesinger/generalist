# Reflexion, Voyager, ExpeL, and Later Experience Distillation

Publication and source status in this report was checked on 2026-07-26.
These systems form a useful progression. Reflexion retains a short critique of
one agent’s recent failures. Voyager promotes successful behavior into
executable procedures. ExpeL compares many trajectories offline and maintains
cross-task natural-language rules. Later work adds selective retrieval,
validation, and explicit pruning, but still generally changes external context
rather than model weights.

## Reflexion: bounded verbal feedback

### Record and trigger

Reflexion separates an actor, an evaluator, and a self-reflection model. After a
trial, the evaluator produces a scalar, binary, or free-form signal. The
self-reflection model turns the trajectory and feedback into a concise verbal
lesson, which is placed in an episodic memory buffer and supplied to the actor
on a later trial.[reflexion][reflexion]

The write trigger is therefore explicit task feedback, usually failure or
suboptimal performance—not elapsed time, similarity, or a background process.
The paper’s HotPotQA setup uses a memory size of three, but that is an
experiment-specific configuration rather than a universal Reflexion contract.
In general, the buffer is a small sliding window: recent reflections are
included directly rather than vector-ranked.

### Semantics and limits

The memory is episodic in the operational sense that it is tied to recent
attempts, although the text itself may state a general rule. There is no
separate semantic fact store or executable procedural store. Eviction provides
bounded forgetting; there is no provenance graph, confidence reconciliation,
or contradiction resolver.

The authors call the verbal feedback a “semantic gradient,” but no gradient is
backpropagated and the model parameters are unchanged.[reflexion][reflexion]
The approach can stall when the evaluator is wrong, the reflection rationalizes
a bad outcome, or the actor cannot explore a genuinely different policy. The
paper reports such local-minimum and exploration limitations, notably in
WebShop. A successful retry is empirical evidence for a lesson in that task,
not proof that its text is causal or transferable.

## Voyager: a procedural skill library

### Record, write gate, and retrieval

Voyager’s durable memory is a library of executable JavaScript skills for
Minecraft. When the iterative code-generation loop completes a task and a
self-verification module judges it successful, the skill is admitted to the
library. A generated natural-language description of the program is embedded
as its retrieval key; the program is the value.[voyager][voyager]

For a new task, Voyager embeds a generated task plan or suggestion together
with environment feedback and retrieves the five most similar skills. The
prompt also receives execution errors and observations, allowing the LLM to
repair code iteratively. This is a semantic index over procedural artifacts,
not a replay store of complete autobiographical episodes.

### Curriculum and memory boundaries

An automatic curriculum proposes increasingly difficult goals using the current
state, completed and failed tasks, and optional context. The task history helps
choose what to attempt; it should not be conflated with the executable skill
library used for procedural recall.

The success-only write gate is stronger than unconditional trajectory storage,
but its guarantee is only as good as the self-verifier and test environment.
The paper does not define semantic versioning, duplicate-skill merging,
revocation, dependency tracking, or a conflict policy when two retrieved
programs encode incompatible assumptions. It also reports practical dependence
on costly frontier-model calls and failures caused by hallucinated APIs or
impossible goals.[voyager][voyager] The evidence is compelling within
Minecraft, not a general safety case for self-written tools.

## ExpeL: offline cross-trajectory consolidation

### Three phases

ExpeL runs a distinct experience-gathering and consolidation phase before
inference:

1. An agent gathers successful and failed trajectories on training tasks,
   using Reflexion-style retries.
2. An LLM compares trajectories and edits a global list of distilled insights.
3. At inference, the agent receives relevant successful trajectories as
   few-shot examples and the insight list as general guidance.[expel][expel]

The experience pool is thus episodic; the insight list is semantic or
procedural guidance expressed as text. They are separate records with separate
recall policies.

### Ranking

ExpeL embeds tasks with `all-mpnet-base-v2`, indexes successful trajectories in
FAISS, and retrieves nearest examples by maximum inner-product similarity. It
does not retrieve failed trajectories at inference. Failure matters during
consolidation, where paired success–failure comparison can isolate a useful
distinction. The global insight list is included rather than selectively
retrieved in the reported experiments.[expel][expel]

### Revision, votes, and deletion

The consolidation LLM emits operations over the insight list: `ADD`, `EDIT`,
`UPVOTE`, and `DOWNVOTE`. An added insight begins with a count; corroborating
comparisons can raise it, contrary evidence can lower it, and a count reaching
zero removes the insight. The interface also leaves the natural-language list
inspectable and manually editable.[expel][expel]

This is an unusually concrete conflict-and-forgetting mechanism for its period,
but the “votes” are not independent measurements. The same LLM family interprets
trajectories, proposes text, and judges agreement. Counts therefore represent
repeated model judgments, not calibrated truth or independent provenance. Edits
can also blur which trajectories support which wording.

### Limits

The reported studies cover HotpotQA, ALFWorld, WebShop, and transfer to FEVER.
The tasks are textual and comparatively bounded. The full insight list fit in
context; the authors identify selective insight retrieval as necessary for a
truly lifelong setting.[expel][expel] ExpeL supplies no user-data deletion
contract, access-control model, or poisoning defense.

## What later work changes

“Successor” is a family resemblance, not a single official ExpeL v2.
MetaReflection moves past per-trial reflections toward an offline semantic
memory of instructions distilled across trials and evaluates it across several
domains.[metareflection][metareflection] Two Findings of ACL 2026 systems expose
the next two pressures especially clearly:

- Mistake Notebook Learning clusters failures, derives generalized mistake
  notes, and updates external memory only when batch performance improves. This
  replaces ExpeL’s LLM vote with an outcome-based admission check, although the
  check is still benchmark-specific.[mnl][mnl]
- ReMe distills success patterns, failure triggers, and comparative insights;
  performs scenario-adaptive reuse; and uses utility to add validated memories
  and prune outdated ones. Its BFCL-V3 and AppWorld results support a dynamic
  procedural-memory lifecycle, not universal continual learning.[reme][reme]

The lineage is from append/retry, to cross-trajectory abstraction, to selective
admission and utility-based retirement. The persistent weakness is semantic
credit assignment: a higher downstream score says a memory set helped in
aggregate, but rarely proves which generated rule was correct or safe.

## Learning verdict

Reflexion, Voyager, ExpeL, MetaReflection, MNL, and ReMe all produce
deployment-time adaptation without changing the underlying LLM parameters.
Their “learned” artifact is respectively a reflection, program, insight,
instruction, mistake note, or procedure. This is genuine system-level learning
under a broad behavioral definition, but it is external-memory revision—not
parametric learning—and can be inspected, removed, or bypassed at prompt time.

## Local References

[expel]: Andrew Zhao et al. “ExpeL: LLM Agents Are Experiential Learners.” AAAI 2024. https://arxiv.org/abs/2308.10144 (accessed 2026-07-26).

[metareflection]: Priyanshu Gupta et al. “MetaReflection: Learning Instructions for Language Agents using Past Reflections.” EMNLP 2024. https://aclanthology.org/2024.emnlp-main.477/ (accessed 2026-07-26).

[mnl]: Xuanbo Su et al. “Mistake Notebook Learning: Batch-Clustered Failures for Training-Free Agent Adaptation.” Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.719/ (accessed 2026-07-26).

[reflexion]: Noah Shinn et al. “Reflexion: Language Agents with Verbal Reinforcement Learning.” NeurIPS 2023. https://arxiv.org/abs/2303.11366 (accessed 2026-07-26).

[reme]: Zouying Cao et al. “Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution.” Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.829/ (accessed 2026-07-26).

[voyager]: Guanzhi Wang et al. “Voyager: An Open-Ended Embodied Agent with Large Language Models.” 2023. https://arxiv.org/abs/2305.16291 (accessed 2026-07-26).
