# Evaluation and Staged Rollout

## Evaluation question

The system succeeds only if it improves longitudinal task performance and user
control without creating unacceptable persistent errors, leakage, latency, or
review burden. Store size, retrieval count, model-rated plausibility, and demo
quality are diagnostics—not success criteria.

## Required baselines

The canonical IDs match the safety evaluation:

| ID | Mode |
| --- | --- |
| B0 | No memory beyond declared working context |
| B1 | Immutable episodic, deterministic chronological selection |
| B2 | Immutable episodic, production retrieval but no derived writes |
| B3 | Unguarded model consolidation, evaluation-only and never deployable |
| B4 | Guarded candidate/promotion pipeline |
| B5-off | B4 with generated counterfactuals disabled |
| B5-on | B4 with generated counterfactuals quarantined from evidence/answer paths |

B0, B1/B2, B3, and B4 are mandatory for the claims they isolate. B5-off/on is
mandatory for any simulation claim. The episodic-only condition is essential
because recent work finds it surprisingly competitive with continuous
abstraction in some settings [faultymemory][faultymemory].

## Experimental unit and reporting

The unit is a complete longitudinal trajectory with a fixed task order, memory
policy, provider/model version, tool environment, random seed where exposed,
and storage snapshot. Report paired results and confidence intervals across
trajectories, not individual question counts as independent samples.

Interrupted jobs, provider failures, permission denials, and timeouts remain
separate outcomes. A stopped evaluation is `unknown`, not evidence that the
architecture is safe or useful.

For each mode report:

- end-task correctness and constraint adherence;
- prospective-memory success and false reminders;
- stale/conflicting-fact selection;
- source attribution accuracy;
- correction and deletion completeness;
- poison admission, retrieval, causal use, and tool-impact rates separately;
- private-data exposure and cross-scope leakage;
- p50/p95 interactive latency and UI input lag;
- storage growth and consolidation token/time cost;
- candidate precision, rejection/edit rate, and review minutes;
- retrieval precision/recall under a fixed context budget; and
- rollback/recovery success after injected crashes.

## Utility suites

### Longitudinal conversation

Use LongMemEval-style information extraction, multi-session reasoning, temporal
reasoning, knowledge updates, and abstention [longmemeval][longmemeval].
Supplement with LoCoMo-style long conversational histories and explicit source
questions [locomo][locomo]. Preserve benchmark licensing and avoid claiming that
one aggregate score covers privacy or security.

### Incremental memory operations

Use MemoryAgentBench-style retrieval, test-time learning, long-range
understanding, and conflict/forgetting tasks
[memoryagentbench][memoryagentbench]. Add exact checks for source spans,
version selection, contradiction display, and “no supported answer.”

### Generalist-native workflows

Build reproducible multi-session tasks around:

- repository conventions and later code changes;
- a user preference that changes with an effective date;
- a failed command whose precondition matters later;
- an unresolved goal and later trigger;
- a rare one-off safety constraint;
- two projects with deliberately similar identifiers;
- a procedure that succeeds once but must not generalize; and
- a manual correction followed by rollback and restart;
- two concurrent agents proposing compatible and conflicting project memories;
  and
- a worker crash followed by lease expiry and safe job reclamation.

Score repository state and tool outputs, not only model judgment.

## Adversarial suites

### Persistent injection and poisoning

Exercise direct, compositional, and dormant records across user input, imported
documents, tool results, compaction, procedure proposals, memory files, and
cross-agent content. Inspired by MPBench and MemPoison's distinction between
direct, compositional, and dormant memory threats [mempoison][mempoison],
instrument the full chain:

`influence → candidate → promotion → retention → authorized retrieval →
causal response change → capability attempt → tool-side allow/deny`.

A defense that blocks tool execution but leaves a promoted poison is not a
clean store; a defense that prevents promotion but leaks the source during
review is not private.

### Privacy and scope

Use unique canaries for user, project, episode, index, cache, export, and backup
layers. Attempt exact, semantic, partial, and timing-based extraction from
another scope. Verify authorization occurs before ranking by instrumenting
candidate counts, not merely inspecting final text.

### Staleness and contradiction

Generate event-time/record-time permutations, malicious newer claims,
equivalent paraphrases, mutually scoped facts, and facts that were historically
valid. Score correct current selection, historical reconstruction, conflict
visibility, and abstention.

### Deletion and resurrection

Delete roots with descendants in summaries, candidates, procedures,
simulations, FTS, caches, active job snapshots, exports, and backups. Restart,
restore an old snapshot, run consolidation again, rebuild indexes, and assert
that tombstones prevent serving or regenerating the data. Report external
deletion work separately.

### Consolidation failure

Inject malformed model output, missing spans, cyclic evidence, same-source
“corroboration,” partial database failure, cancellation at every state,
provider retry, quota exhaustion, and a user prompt arriving during each job
phase. Verify atomic publish and interactive liveness.

### Simulation isolation

Generate a plausible but false counterfactual sharing names and text with a real
episode. Assert it cannot appear in episode search defaults, evidence edges,
confidence counts, automatic retrieval, or historical answers. Running a test
derived from it may create a new observed result with a fresh lineage edge.

## Formal and implementation validation

`MemoryRuntime.tla` uses small finite sets for scopes, drafts, episodes,
candidates, sources, jobs, simulations, reviews, tombstones, and prompts.
`CollaborationRuntime.tla` owns authenticated agents/tasks, immutable message
envelopes plus events, delegation, leases/fences, and effect intents.
`AsyncRuntime.tla` continues to own TUI prompt/permission/cancellation
liveness. Their shared interface—prompt memory epochs, effect recheck,
tombstone/revocation notification, and authenticated capability snapshot—is
traced in [the implementation handoff](implementation-handoff.md). TLC checks
each finite model plus executable cross-interface schedules.

Implementation tests include:

- property tests over state transitions and lineage closure;
- SQLite failure/crash injection around begin/finalize/publish/delete;
- multi-process races over finalization, job claim, lease renewal, candidate
  publication, promotion, correction, tombstoning, and checkpointing;
- schema migration and corrupt-store degraded-start tests;
- scope-first query-plan/instrumentation tests;
- deterministic FTS rebuild tests;
- current-thread async tests with a deliberately slow worker;
- PTY tests for typing, queue editing, review, cancellation, status, copy mode,
  and terminal restoration;
- direct-code-mode memory search tests; and
- a painstaking trace from every TLA+ action to Rust event, transaction, and
  test.

Reviewers must record mismatches in `docs/runtime-traceability.md`; “the tests
pass” does not discharge the model-correspondence review.

## Rollout stages

These memory stages are nodes in the unified memory/collaboration milestone DAG
in [the implementation handoff](implementation-handoff.md); they are not an
independent roadmap.

### Stage 0 — Measurement and migration safety

- Add feature flags and telemetry with memory disabled by default.
- Define schemas, all three TLA+ ownership/interface mappings, the
  supervisor/client protocol, and crash tests.
- Import existing enhanced-memory JSON entries as `legacy_model_note`
  candidates in quarantine only after admission. Leave the legacy file
  byte-for-byte untouched for reversible migration; do not copy excluded raw
  secrets into the new database.
- Never inject legacy entries automatically.

**Exit gate:** migration is lossless/reversible; corrupt legacy data cannot
prevent startup; model and implementation trace review pass.

### Stage 1 — Episodic foundation

- Capture immutable typed episodes.
- Add `/memory status`, `pause`, `search`, `show`, `export`, and `forget`.
- Keep all recall explicit.
- Add retention, sensitivity, and project-scope controls.

**Exit gate:** crash/scope/UI tests and raw-episode tombstone, live-purge, and
external-ledger restore tests pass; the episodic-only baseline does not regress
interactive performance beyond the agreed budget. Derived-descendant and
promotion-race deletion tests begin only when those states exist in M2/M3.

### Stage 2 — User-authored and approved memory

- Add `/memory remember`, review/edit/approve/reject, temporal supersession,
  reminders, and small automatic retrieval bundles.
- No model-generated candidate jobs yet.

**Exit gate:** retrieval improves paired native tasks, poison/use and
cross-scope tests remain within stated thresholds, and users can identify and
remove every automatically rendered item.

### Stage 3 — Offline candidate consolidation

- Enable manual, cancellable jobs that propose facts, reminders, conflicts, and
  summaries from a frozen episode snapshot.
- Require the C2 task, lease, fence, and task/attempt-bound worker-session
  boundary; controller sessions cannot publish proposals.
- Require exact source spans and user approval.
- Keep “no change” visible and expected.

**Exit gate:** consolidation beats approved-retrieval and episodic-only
baselines on preregistered tasks after cost/review burden, without statistically
or operationally material degradation on poisoning, staleness, privacy,
deletion, and liveness suites.

### Stage 4 — Conservative scheduling

- Evaluate recurrence and explicit salience triggers during idle periods.
- Add quotas, backpressure, cancellation, and candidate deduplication.
- Permit narrowly defined auto-promotion only if a low-risk class has measured
  precision, reversible impact, and an opt-out. Default remains review.

**Exit gate:** longitudinal shadow evaluation shows sustained net benefit and
the review queue remains bounded.

### Stage 5 — Quarantined simulations and procedures

- Generate counterfactual regression tests in the simulation namespace.
- Evaluate inert procedure proposals with explicit preconditions and capability
  manifests.
- Keep execution under ordinary permission and user approval.

**Exit gate:** simulations find independently reproducible failures without
contaminating evidence; procedure reuse improves verified outcomes without
capability escalation. Parametric training remains a separate project.

## Stop rules

Pause rollout and preserve evidence if any of the following occurs:

- a cross-scope retrieval;
- a deleted item resurrects;
- a simulation is accepted as observation;
- memory grants or broadens capability;
- a partial batch becomes visible after failure;
- input/queue interaction stalls behind consolidation;
- poisoned candidates are silently promoted;
- review burden grows without measured task benefit; or
- repeated consolidation underperforms the episodic-only baseline.

The correct response is rollback, root-cause analysis, model/test update, and a
new preregistered evaluation—not a prompt-only patch presented as resolution.

## Local References

[faultymemory]: Dylan Zhang et al. “Useful Memories Become Faulty When Continuously Updated by LLMs.” arXiv:2605.12978v1 (2026). https://arxiv.org/abs/2605.12978

[locomo]: Adyasha Maharana et al. “Evaluating Very Long-Term Conversational Memory of LLM Agents.” arXiv:2402.17753 (2024). https://arxiv.org/abs/2402.17753

[longmemeval]: Di Wu et al. “LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.” ICLR 2025. https://openreview.net/forum?id=pZiyCaVuti

[memoryagentbench]: Yuanzhe Hu, Yu Wang, and Julian McAuley. “Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions.” ICLR 2026. https://openreview.net/forum?id=DT7JyQC3MR

[mempoison]: Jifeng Gao et al. “MemPoison: Uncovering Persistent Memory Threats and Structural Blind Spots in LLM Agents.” arXiv:2607.14651v1 (2026). https://arxiv.org/abs/2607.14651
