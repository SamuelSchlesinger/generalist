# World-Model “Dreaming” and Offline Reinforcement Learning

## Five different meanings of offline

The word “dreaming” collapses several technically different operations:

| Term | Input | Output | Parameters change? | Evidence status |
|---|---|---|---|---|
| Biological replay | Neural activity after experience | Reactivated or constructive sequences | Biological plasticity may occur | Neural event |
| Model-based imagination | Learned dynamics plus state/action | Simulated trajectory | Usually, during training | Model sample |
| Offline reinforcement learning | Fixed logged transition dataset | Learned value function or policy | Yes | Policy learned from observations |
| Sleep-time compute | Known context before a query | Precomputed representation or inference | Not necessarily | Derived text/state |
| Host-side consolidation | Stored agent episodes | Facts, summaries, procedures, indexes | Base LLM need not change | Derived memory |

Confusing these categories creates false transfers. In particular, offline RL is
“offline” because it learns from a fixed dataset without new environment
interaction. It is not automatically an idle-time job, a memory system, or a
safety mechanism.

## What world-model imagination establishes

### Dyna: simulated experience can support planning

Dyna integrates direct reinforcement learning with a learned environment model.
Real transitions update both the value estimates and the model; simulated
transitions sampled from the model provide additional planning updates
[sutton90][sutton90]. The enduring contribution is a separation between:

- observed experience;
- a learned predictive model; and
- computation performed on simulated experience.

It does not make simulated and observed transitions epistemically equivalent.

### World Models: train a controller inside a learned simulator

Ha and Schmidhuber learned a compressed visual representation and recurrent
dynamics model, then optimized a compact controller using the learned model. In
one reported setup, a controller trained inside the model transferred back to
the environment [ha18][ha18].

This is primary evidence that a learned simulator can sometimes be useful
enough for policy search. It is not evidence that arbitrary hallucination
improves an agent. The “dream” is constrained by a learned transition model, an
action space, an objective, and subsequent environment evaluation.

### Dreamer: optimize behavior through latent imagination

Dreamer trains a latent dynamics model and learns actor and value networks from
imagined latent trajectories [hafner20][hafner20]. The 2025 peer-reviewed
DreamerV3 work reports a single configuration across a broad set of control
benchmarks. Its world model, actor, and critic are trained concurrently from
replayed environment experience [hafner25][hafner25].

This establishes that latent imagination can be a powerful training substrate
for control. It still depends on:

- a defined observation and action interface;
- rewards or value targets;
- many parameter updates;
- repeated environment interaction for online training; and
- benchmark-specific evaluators.

A fixed-weight CLI agent has none of those properties merely because it can
generate text while idle.

### A useful model need not reconstruct the whole world

MuZero learns a representation, dynamics, reward, policy, and value model
sufficient for tree search, without being given the environment's transition
rules [schrittwieser20][schrittwieser20]. It demonstrates that a “world model”
can be **decision-equivalent** rather than a faithful generative simulator.

That distinction matters for memory. A compact procedure can improve decisions
without being a true factual description of the environment. Such an artifact
should be evaluated as a policy aid and should not silently enter a semantic
fact store.

## Model bias is the central boundary

Model-generated data are cheap, so optimization can exploit their errors.
Janner and colleagues analyzed this trade-off and found that short model
rollouts branched from real data could retain the benefit of model-based policy
optimization while reducing the damage from compounding model error in their
experiments [janner19][janner19].

For an agent-memory runtime, this suggests four controls:

1. **Branch from evidence**: attach every counterfactual to a real episode or an
   explicitly supplied hypothetical state.
2. **Limit rollout depth**: uncertainty and unsupported assumptions grow with
   each generated step.
3. **Check against independent evidence**: use tools, tests, or held-out
   episodes where available.
4. **Never relabel a rollout as observation**: a plausible continuation remains
   synthetic.

These are architectural inferences. MBPO's guarantees and benchmark results do
not apply directly to natural-language agents.

## What offline reinforcement learning establishes

### The fixed-data problem

Offline RL learns a policy from a static batch of logged transitions. It is
attractive when exploration is costly or dangerous, but the learned policy can
choose actions that are poorly supported by the dataset.

Fujimoto, Meger, and Precup demonstrated extrapolation error in standard
off-policy deep RL under fixed data and proposed constraining actions toward
the dataset's support [fujimoto19][fujimoto19]. Conservative Q-Learning instead
regularizes values so unsupported actions are not assigned spuriously high
value [kumar20][kumar20]. Implicit Q-Learning avoids directly evaluating unseen
actions during value learning and extracts a policy through
advantage-weighted behavioral cloning [kostrikov22][kostrikov22].

The shared lesson is **pessimism under distribution shift**. A candidate action
that looks excellent only in unsupported regions should not be trusted.

### Model-based offline RL

MOPO learns a dynamics model from the offline batch and penalizes model
uncertainty in the reward used for policy optimization
[yu20][yu20]. It demonstrates a concrete way to generate counterfactual
rollouts while discouraging exploitation of uncertain model regions.

The guarantee is tied to the paper's assumptions and penalized Markov decision
process. For a tool-using LLM agent, embedding similarity, model self-reported
confidence, or verbal hedging is not a calibrated dynamics uncertainty measure.

### Offline is not safe by default

Offline RL avoids live exploration during training, but can still:

- optimize a misspecified reward;
- inherit confounding and coverage gaps from logs;
- overestimate unsupported actions;
- learn unsafe behavior present in the dataset; and
- fail after environment drift.

The defensible transfer is not “train on all old trajectories while nobody is
watching.” It is: keep learning or search inside a sandbox, use conservative
support checks, and require independent evaluation before changing deployed
behavior.

## Recent LLM uses of “sleep”

### Sleep-time compute is anticipatory precomputation

Lin and colleagues define sleep-time compute as processing a known context
before the eventual query arrives. Their method prompts a model to rewrite the
context into inferences likely to help anticipated queries, then reuses that
representation at query time [lin25][lin25].

The experiments support a compute-amortization claim in constructed stateful
reasoning tasks and a software-engineering case study. The paper also finds the
method more useful when the future query is predictable from the context.

It is **not** evidence for unconstrained synthetic dreaming:

- generation is conditioned on a known context;
- the target is anticipated query utility;
- the output is a re-representation used at inference;
- the base setting does not require autonomous environment action; and
- unpredictable queries benefit less.

For a fixed-weight agent, this is closer to precomputing indexes, candidate
invariants, dependency maps, or likely questions than to learning a world
model.

### “Language Models Need Sleep” changes model parameters

The June 2026 preprint by Behrouz and colleagues proposes a two-stage process:
knowledge seeding through distillation and reinforcement-learning-based
imitation, followed by “dreaming” that generates a synthetic curriculum for
reinforcement-learning self-improvement [behrouz26][behrouz26]. The current
arXiv version is explicitly a proof of concept and was revised in July 2026.

This is relevant as new parameter-learning research, but it does not directly
transfer to a host-side fixed-weight runtime:

- its durable memory is parametric;
- it requires training infrastructure and optimizer access;
- synthetic examples affect weights through reinforcement learning;
- its benchmark results do not establish safe open-ended self-modification; and
- the biological terminology is inspiration, not validation.

The paper can motivate a separately governed research track. It does not justify
allowing an idle CLI agent to fine-tune itself or execute model-generated tasks.

## Counterfactual generation protocol

A safe host-side analogue of model-based imagination would produce a typed
record:

```text
counterfactual {
  root_episode_ids
  assumed_state
  proposed_action
  predicted_observations
  model_and_prompt_identity
  uncertainty_or_support_notes
  verifier
  verification_status
  expiry
}
```

Generated observations must never be copied into the episode stream. If a tool
or later interaction tests the prediction, the actual result becomes a new
episode linked to the counterfactual. The original prediction remains immutable
for calibration and audit.

Useful offline jobs include:

- enumerating likely failure branches for a known procedure;
- generating tests whose outcomes are determined by a trusted checker;
- precomputing repository maps or dependency summaries from a fixed snapshot;
- contrasting a proposed rule with near-miss episodes; and
- estimating which memories would matter for anticipated queries.

Unsafe default jobs include:

- inventing user preferences;
- simulating tool success and storing it as fact;
- generating self-praise as a reward signal;
- updating deployed procedures solely because generated trials favored them;
- executing side effects while the user is absent.

## Mechanism-to-problem ledger

| Mechanism | Abstraction | Contradiction | Forgetting | Counterfactuals |
|---|---|---|---|---|
| World model | Compresses predictive structure | May encode inconsistency | Replay can preserve training distribution | Core capability |
| Short branched rollout | No direct guarantee | Limits compounding, not logical conflict | No direct guarantee | Constrains generation near evidence |
| Conservative offline RL | Learns pessimistic values | No proposition-level repair | No direct guarantee | Penalizes unsupported actions |
| Sleep-time compute | Re-represents known context | Can compare context if prompted | Does not preserve omitted evidence by itself | Anticipates likely queries |
| Host consolidation | Can create facts or procedures | Must add explicit provenance and supersession | Can retain episodes and versions | Must type generated records separately |

## Uncertainty and access status

World-model and offline-RL results are empirical and task-distribution
dependent. “Dreaming” is not a standardized technical term. The DreamerV3
publication is peer reviewed; the 2025 sleep-time-compute paper and the 2026
Behrouz et al. paper were preprints at the checked URLs. Source and official
proceedings URLs were checked on 2026-07-26.

## Local References

[sutton90]: Sutton, Richard S. “Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming.” In *Proceedings of the Seventh International Conference on Machine Learning*, 216–224 (1990). https://doi.org/10.1016/B978-1-55860-141-3.50030-4

[ha18]: Ha, David; Schmidhuber, Jürgen. “Recurrent World Models Facilitate Policy Evolution.” *Advances in Neural Information Processing Systems 31* (2018). https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution

[hafner20]: Hafner, Danijar; Lillicrap, Timothy; Ba, Jimmy; Norouzi, Mohammad. “Dream to Control: Learning Behaviors by Latent Imagination.” *International Conference on Learning Representations* (2020). https://openreview.net/forum?id=S1lOTC4tDS

[hafner25]: Hafner, Danijar; Pasukonis, Jurgis; Ba, Jimmy; Lillicrap, Timothy. “Mastering Diverse Control Tasks Through World Models.” *Nature* 640, 647–653 (2025). https://doi.org/10.1038/s41586-025-08744-2

[schrittwieser20]: Schrittwieser, Julian; Antonoglou, Ioannis; Hubert, Thomas; et al. “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.” *Nature* 588, 604–609 (2020). https://doi.org/10.1038/s41586-020-03051-4

[janner19]: Janner, Michael; Fu, Justin; Zhang, Marvin; Levine, Sergey. “When to Trust Your Model: Model-Based Policy Optimization.” *Advances in Neural Information Processing Systems 32* (2019). https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html

[fujimoto19]: Fujimoto, Scott; Meger, David; Precup, Doina. “Off-Policy Deep Reinforcement Learning without Exploration.” *Proceedings of the 36th International Conference on Machine Learning*, PMLR 97, 2052–2062 (2019). https://proceedings.mlr.press/v97/fujimoto19a.html

[kumar20]: Kumar, Aviral; Zhou, Aurick; Tucker, George; Levine, Sergey. “Conservative Q-Learning for Offline Reinforcement Learning.” *Advances in Neural Information Processing Systems 33* (2020). https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html

[kostrikov22]: Kostrikov, Ilya; Nair, Ashvin; Levine, Sergey. “Offline Reinforcement Learning with Implicit Q-Learning.” *International Conference on Learning Representations* (2022). https://openreview.net/forum?id=68n2s9ZJWF8

[yu20]: Yu, Tianhe; Thomas, Garrett; Yu, Lantao; Ermon, Stefano; Zou, James Y.; Levine, Sergey; Finn, Chelsea; Ma, Tengyu. “MOPO: Model-Based Offline Policy Optimization.” *Advances in Neural Information Processing Systems 33* (2020). https://proceedings.neurips.cc/paper_files/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html

[lin25]: Lin, Kevin; Snell, Charlie; Wang, Yu; Packer, Charles; Wooders, Sarah; Stoica, Ion; Gonzalez, Joseph E. “Sleep-time Compute: Beyond Inference Scaling at Test-time.” arXiv:2504.13171v1 (2025). https://arxiv.org/abs/2504.13171

[behrouz26]: Behrouz, Ali; Hashemi, Farnoosh; Javanmard, Adel; Mirrokni, Vahab. “Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.” arXiv:2606.03979v2 (2026). https://arxiv.org/abs/2606.03979
