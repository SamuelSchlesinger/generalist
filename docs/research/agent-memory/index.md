# Episodic Memory and Safe Offline Consolidation for Generalist

> **Repository status (2026-07-27):** Generalist has implemented only a small
> opt-in episodic-search experiment. The model-facing legacy tool was removed;
> automatic retrieval, consolidation, dreaming, and collaboration remain
> absent. See `../../next-agent-handoff.md` for the current implementation
> boundary. This corpus remains research and future design input.

This corpus investigates whether and how Generalist should replace its current
enhanced-memory tool with an agent-native memory runtime that learns across
episodes. “Dreaming” is treated as offline replay and consolidation under
explicit evidence, provenance, evaluation, and rollback controls—not as
unbounded self-training.

## Research questions

1. What do cognitive science and continual-learning research actually support
   about episodic replay, semantic consolidation, abstraction, forgetting, and
   sleep-like offline processing?
2. Which mechanisms transfer usefully to tool-using language-model agents, and
   which are only biological metaphors?
3. How do deployed/open agent systems represent, retrieve, reflect on,
   consolidate, edit, and forget memory?
4. What can the host runtime learn without changing model weights? When, if
   ever, is prompt/procedure/skill synthesis justified?
5. How do we prevent persistent prompt injection, poisoned memories,
   self-confirming false beliefs, privacy leakage, stale facts, and
   contradiction accumulation?
6. What evaluation would demonstrate improved longitudinal performance rather
   than merely more retrieval or benchmark overfitting?
7. What architecture fits Generalist’s current async runtime, code-mode
   boundary, durable history, TLA+ methodology, and Unix-only scope?
8. Which identity, delegation, messaging, database, and workspace protocols are
   required before multiple agents can safely share one project?

## Corpus

- [Foundations: cognition and continual learning](foundations/index.md)
  - complementary learning systems and biological replay
  - rehearsal, generative replay, consolidation, and forgetting
  - world-model “dreaming” and offline reinforcement learning
- [Existing agent memory systems](systems/index.md)
  - research agents and memory architectures
  - current open-source/product implementations and their contracts
  - transferable design patterns and unresolved limitations
- [Safety, governance, and evaluation](safety-evaluation/index.md)
  - persistent-injection and memory-poisoning threat model
  - provenance, contradiction repair, deletion, privacy, and rollback
  - longitudinal benchmarks and adversarial evaluation
- [Generalist architecture synthesis](architecture/index.md)
  - current-state gap analysis
  - requirements and non-goals
  - candidate runtime and consolidation protocol
  - unified memory/collaboration implementation handoff
  - self-critique, alternatives, and staged validation plan
- [Master bibliography](sources.md)

## Design conclusions

- Raw admitted episodes are immutable evidence. Retained, redacted,
  secret-reference, omitted, and incomplete payloads remain explicit. A draft
  is not retrievable and becomes an episode only after a history-valid
  checkpoint and settled outcome.
- Model summaries, reflections, preferences, procedures, and predictions are
  typed candidates. They never promote themselves or increase their authority
  by being summarized repeatedly.
- Generated counterfactuals live in a separate simulation namespace. They are
  never observational evidence or independent corroboration.
- Offline work is a bounded `ConsolidationJob`, not an authority-bearing
  “dream.” The first release proposes candidates but does not auto-promote.
- Approved facts and preferences may eventually be retrieved automatically;
  procedures remain inert until a separate capability-aware approval path.
- Provider reasoning is an observability surface, not memory evidence.
- Supervisor-owned SQLite is the proposed local coordination substrate, but
  transaction locks do not replace OS isolation, authenticated agent
  identities, attenuated delegations, CAS epochs, fencing tokens, idempotency
  keys, or tombstone precedence.
- Same-project agents do not automatically share private memories. Parallel
  writers need isolated worktrees and an explicit patch/commit integration
  protocol in addition to database coordination.

## Remaining unknowns

- Whether guarded consolidation improves utility over immutable episodic
  retrieval under equal latency, token, and storage budgets.
- Which claim classes can be verified independently enough to permit promotion
  without user review.
- What decay policy improves relevance without suppressing rare, high-impact
  evidence.
- Whether generated simulations produce measurable test value once they are
  prevented from acting as evidence.
- Which multi-agent operations justify the coordination cost beyond the first
  read-only worker milestone.
- How much of the full storage and collaboration protocol remains tractable in
  TLA+ without abstracting away implementation-critical races.

## Method

The corpus follows parallel authoring, whole-corpus review through accuracy,
completeness, coherence, source-quality, and staleness lenses, and up to three
review/revision cycles. Significant claims must use primary sources. Product
claims are current only as of the access date. Quantitative or structural
claims should include small validation artifacts where practical.

## Supplementary code

- [Corpus bibliography compiler](data/compile_sources.py)
- [Corpus-wide structural validator](data/validate_corpus.py)
- [Systems corpus validator](systems/data/check_corpus.py)
- [Systems source audit](systems/data/source-audit.md)
- [Safety branch validator](safety-evaluation/data/validate-branch.py)
- [Threat-control coverage matrix](safety-evaluation/data/threat-control-matrix.csv)
