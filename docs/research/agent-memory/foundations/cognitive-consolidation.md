# Cognitive Consolidation and Replay

## Scope and evidence labels

This document asks what biological memory research establishes before borrowing
its vocabulary for an artificial agent. It uses three labels:

- **Biological observation**: neural or behavioral association without a
  selective intervention.
- **Biological intervention**: an experiment that manipulates a candidate
  mechanism. Causality is still limited to the manipulated event, species, and
  task.
- **Computational account**: a model that explains observations or makes useful
  predictions. It is not, by itself, a discovered biological implementation.

“Hippocampal,” “cortical,” “sleep,” and “dreaming” should not be used as
component names in an agent architecture unless the analogy adds an operational
constraint. A database is not a hippocampus, and an overnight job is not sleep.

## Complementary learning systems

The complementary learning systems (CLS) account starts from a stability versus
plasticity problem. Rapidly changing a shared distributed representation to
encode one event can interfere with older structure. McClelland, McNaughton, and
O'Reilly modeled a fast-learning hippocampal system that stores sparse,
pattern-separated traces and a slower neocortical system that learns distributed
structure through interleaved reinstatement [mcclelland95][mcclelland95].

What this establishes is narrower than the slogan “the brain has a cache and a
database”:

- The original work gives a computational reason for separating rapid,
  event-specific storage from slow integration.
- Its connectionist simulations show why interleaving old and new patterns can
  reduce catastrophic interference.
- Neuropsychological observations are compatible with the division. For
  example, three people with early bilateral hippocampal pathology had severe
  everyday episodic amnesia while acquiring substantial language, literacy, and
  factual knowledge [varghakhadem97][varghakhadem97].
- The lesion result supports partial dissociation, not a clean two-table
  ontology. Semantic learning was impaired relative to normal development, and
  both memory forms depend on interacting systems.

The most defensible transferable principle is therefore **different update
rates and representations for event records and cross-event abstractions**.
Nothing in CLS makes the slow store automatically true, consistent, or safe.

## Episodic and semantic memory

For this corpus, an **episode** is a temporally and causally situated record:
who or what acted, the state available at the time, the action, tool result,
outcome, and provenance. A **semantic memory** is a claim, relation, preference,
or procedure abstracted away from a single occurrence.

The distinction is functional, not a guarantee about physical location.
Vargha-Khadem and colleagues' cases show that rich factual learning can coexist
with profound episodic impairment [varghakhadem97][varghakhadem97]. Conversely,
semantic knowledge normally draws on many experiences and can retain links to
particular episodes. An agent should therefore allow a semantic item to cite
episodes instead of treating semanticization as evidence deletion.

## Replay: observation, intervention, and limits

### Reactivation exists

Wilson and McNaughton recorded rat hippocampal place-cell ensembles and found
that patterns expressed during spatial behavior were re-expressed during
subsequent slow-wave sleep [wilson94][wilson94]. Diba and Buzsáki later observed
forward and reverse place-cell sequences around sharp-wave ripples during awake
pauses [diba07][diba07].

These are biological observations. They show structured, experience-related
reactivation; they do not alone show that replay caused durable memory or that
every replay event copied a past episode faithfully.

### Ripple events matter in some tasks

Selective disruption makes the causal claim stronger. Suppressing hippocampal
sharp-wave ripples during post-training rest impaired later spatial learning in
rats [girardeau09][girardeau09]. Interrupting awake ripples during a spatial
alternation task produced a learning and performance deficit while leaving
place fields and post-experience reactivation intact
[jadhav12][jadhav12].

These interventions support a role for ripple-associated activity in particular
forms of spatial learning and memory-guided choice. They do **not** isolate the
informational content of a decoded replay sequence from every other function of
the ripple event. Nor do they establish that replay should be exhaustive,
uniform, or scheduled only during sleep.

### Sleep can modulate consolidation

In humans, re-presenting a learning-associated odor during slow-wave sleep
improved retention for a hippocampus-dependent spatial task under the tested
conditions, while the same cue in REM sleep or wakefulness did not have the same
effect [rasch07][rasch07]. This is intervention evidence that targeted
reactivation during a particular sleep state can affect later recall.

It is not evidence that:

- all memories improve during sleep;
- unconstrained generation is beneficial;
- an artificial system needs a circadian schedule; or
- “offline” computation is intrinsically safer than online computation.

Replay is also observed during quiet wakefulness, and awake ripple disruption
can impair learning [diba07][diba07] [jadhav12][jadhav12]. The transferable
feature is **decoupling expensive integration from the latency-critical action
loop**, not biological sleep itself.

## Abstraction and semanticization

Two primary experiments support cautious claims about integration:

1. In a rat paired-associate task, an established schema allowed new
   flavor-place associations to become hippocampal-independent much faster than
   the traditional slow-consolidation story predicts
   [tse07][tse07]. Prior structure changed the rate of integration.
2. In a multi-day human fMRI experiment, neural representations of overlapping
   associations became more similar in hippocampus and medial prefrontal cortex
   after a week. Greater overlap representation was inversely related to unique
   episodic reinstatement [tompary17][tompary17].

Together, these results support transformation and regularity extraction, but
also expose a cost: integration can trade event detail for shared structure.
They do not show that repetition alone identifies a valid general rule.
Correlated errors, duplicated text, or a repeated prompt injection are also
regularities.

For an agent, **abstraction must remain a derived, defeasible object**. A useful
semantic item needs:

- links to the episodes from which it was inferred;
- a scope that says where it was observed;
- a method or rationale for the generalization;
- a timestamp or validity interval where facts can change; and
- a way to retrieve disconfirming as well as supporting episodes.

These controls are architectural inference, not claims about human memory.

## Updating is not contradiction resolution

Retrieval can make some consolidated memories labile. Nader, Schafe, and LeDoux
showed that reactivated conditioned fear memories in rats again required
amygdala protein synthesis to persist [nader00][nader00]. This supports a
reconsolidation process in that preparation.

It does not establish a generic algorithm that:

- detects two propositions as contradictory;
- decides which source is authoritative;
- preserves temporal changes such as “address was A, now B”;
- distinguishes correction from adversarial instruction; or
- merges two textual summaries without information loss.

Reactivation may enable updating, but it can also enable distortion or loss.
Explicit contradiction detection, provenance comparison, temporal modeling, and
non-destructive supersession are therefore engineering requirements, not
features inherited from the biological metaphor.

## Replay can be constructive

“Replay” is not always a literal recording:

- Gupta and colleagues observed replay of remote and infrequently experienced
  paths, plus sequences corresponding to physically possible paths the rats had
  not traversed [gupta10][gupta10].
- Before goal-directed navigation, hippocampal sequences were biased toward
  paths from the current location to a remembered goal and predicted behavior
  even for novel start-goal combinations [pfeiffer13][pfeiffer13].

These results support a constructive or planning role for some awake sequences.
They do not show human-like imagination, causal-world-model accuracy, or safe
counterfactual reasoning. A never-experienced sequence can be useful, wrong, or
both.

For an agent, the safe translation is a type distinction:

- **observed episode** — grounded in a recorded interaction or tool result;
- **derived abstraction** — inferred from identified observations;
- **counterfactual** — a generated possibility, explicitly not observed; and
- **prediction** — a counterfactual with a testable expectation.

No promotion between these classes should occur merely because an item was
replayed often.

## Mechanism-to-problem ledger

| Problem | Biological evidence supports | It does not supply |
|---|---|---|
| Abstraction | Related experiences can become integrated; schemas can accelerate integration | Sound rule induction or truth certification |
| Contradiction | Retrieval can reopen some memories to modification | Source authority, temporal truth maintenance, or safe merge logic |
| Forgetting | Replay-associated activity can support retention in specific tasks | A universal retention schedule or a reason never to forget |
| Counterfactuals | Some sequences represent novel but possible paths and future goals | Calibrated simulation or permission to treat generation as evidence |

## Transfer verdict

### Strong functional transfers

- Keep high-fidelity episodes separate from slower derived knowledge.
- Consolidate selectively and preserve links back to evidence.
- Mix older and newer evidence when revising a generalization.
- Treat reactivation as an opportunity for review, not automatic overwrite.
- Permit counterfactual construction only in a provenance-distinct namespace.

### Useful hypotheses requiring agent experiments

- Recurrence, surprise, reward, error, and anticipated utility may be useful
  consolidation triggers.
- Interleaving dissimilar episodes may produce more robust abstractions than
  summarizing one topical cluster in isolation.
- Re-checking a derived memory against raw episodes may reduce detail loss.

### Metaphor only

- Vector search is “hippocampal recall.”
- A nightly cron job is biologically sleep-like.
- A summary is semantic memory and therefore more stable or true.
- Generated episodes are dreams and therefore intrinsically creative or useful.
- Human replay research licenses autonomous self-training.

## Uncertainty and access status

The biological literature is heterogeneous across species, tasks, recording
methods, and definitions of replay. The cited intervention studies establish
local causal roles, not a single consensus algorithm for consolidation. DOI and
publisher URLs below were checked on 2026-07-26.

## Local References

[mcclelland95]: McClelland, James L.; McNaughton, Bruce L.; O'Reilly, Randall C. “Why There Are Complementary Learning Systems in the Hippocampus and Neocortex: Insights from the Successes and Failures of Connectionist Models of Learning and Memory.” *Psychological Review* 102(3), 419–457 (1995). https://doi.org/10.1037/0033-295X.102.3.419

[varghakhadem97]: Vargha-Khadem, Faraneh; Gadian, David G.; Watkins, Kate E.; Connelly, Alan; Van Paesschen, Wim; Mishkin, Mortimer. “Differential Effects of Early Hippocampal Pathology on Episodic and Semantic Memory.” *Science* 277(5324), 376–380 (1997). https://doi.org/10.1126/science.277.5324.376

[wilson94]: Wilson, Matthew A.; McNaughton, Bruce L. “Reactivation of Hippocampal Ensemble Memories During Sleep.” *Science* 265(5172), 676–679 (1994). https://doi.org/10.1126/science.8036517

[diba07]: Diba, Kamran; Buzsáki, György. “Forward and Reverse Hippocampal Place-Cell Sequences During Ripples.” *Nature Neuroscience* 10, 1241–1242 (2007). https://doi.org/10.1038/nn1961

[girardeau09]: Girardeau, Gabrielle; Benchenane, Karim; Wiener, Sidney I.; Buzsáki, György; Zugaro, Michaël B. “Selective Suppression of Hippocampal Ripples Impairs Spatial Memory.” *Nature Neuroscience* 12, 1222–1223 (2009). https://doi.org/10.1038/nn.2384

[jadhav12]: Jadhav, Shantanu P.; Kemere, Caleb; German, P. Walter; Frank, Loren M. “Awake Hippocampal Sharp-Wave Ripples Support Spatial Memory.” *Science* 336(6087), 1454–1458 (2012). https://doi.org/10.1126/science.1217230

[rasch07]: Rasch, Björn; Büchel, Christian; Gais, Steffen; Born, Jan. “Odor Cues During Slow-Wave Sleep Prompt Declarative Memory Consolidation.” *Science* 315(5817), 1426–1429 (2007). https://doi.org/10.1126/science.1138581

[tse07]: Tse, Dorothy; Langston, Rosamund F.; Kakeyama, Masaki; Bethus, Ingrid; Spooner, Patrick A.; Wood, Emma R.; Witter, Menno P.; Morris, Richard G. M. “Schemas and Memory Consolidation.” *Science* 316(5821), 76–82 (2007). https://doi.org/10.1126/science.1135935

[tompary17]: Tompary, Alexa; Davachi, Lila. “Consolidation Promotes the Emergence of Representational Overlap in the Hippocampus and Medial Prefrontal Cortex.” *Neuron* 96(1), 228–241.e5 (2017). https://doi.org/10.1016/j.neuron.2017.09.005

[nader00]: Nader, Karim; Schafe, Glenn E.; LeDoux, Joseph E. “Fear Memories Require Protein Synthesis in the Amygdala for Reconsolidation After Retrieval.” *Nature* 406, 722–726 (2000). https://doi.org/10.1038/35021052

[gupta10]: Gupta, Anoopum S.; van der Meer, Matthijs A. A.; Touretzky, David S.; Redish, A. David. “Hippocampal Replay Is Not a Simple Function of Experience.” *Neuron* 65(5), 695–705 (2010). https://doi.org/10.1016/j.neuron.2010.01.034

[pfeiffer13]: Pfeiffer, Brad E.; Foster, David J. “Hippocampal Place-Cell Sequences Depict Future Paths to Remembered Goals.” *Nature* 497, 74–79 (2013). https://doi.org/10.1038/nature12112
