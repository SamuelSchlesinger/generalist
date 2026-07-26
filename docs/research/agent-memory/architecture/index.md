# Generalist Architecture Synthesis

This branch translates the research into a concrete, reviewable design for the
Generalist repository. It must preserve the async TUI and history-valid runtime
boundaries while replacing model-managed flat notes with host-owned memory.

## Documents

- [Current-state gap analysis](current-state.md)
- [Requirements and non-goals](requirements.md)
- [Candidate memory and consolidation protocol](protocol.md)
- [Multi-agent coordination architecture](multi-agent.md)
- [Implementation handoff and unified milestones](implementation-handoff.md)
- [Alternatives and self-critique](alternatives.md)
- [Evaluation and staged rollout plan](evaluation-rollout.md)

## Decision status

The candidate architecture is a review target, not yet an implementation
decision. It is source-grounded in repository commit `db900fa` and deliberately
stages immutable episodic capture before automated consolidation. Promotion,
procedural reuse, and simulation remain disabled until their corresponding
evaluation gates pass.
