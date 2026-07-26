# Existing Agent Memory Systems

This branch examines concrete memory contracts in research prototypes, open
memory runtimes, and coding agents. The central distinction is between systems
that retrieve or rewrite external records and systems that actually update a
learned policy or model.

## Scope and questions

- What is stored, and which component decides to write?
- How are memories retrieved, ranked, reflected on, consolidated, revised, and
  deleted?
- Are episodic, semantic, and procedural records separated?
- Which claims are supported by implementation contracts rather than product
  language?
- Which patterns are safe enough to transfer into offline consolidation for
  Generalist?

## Research prototypes

- [Research prototypes overview](research-prototypes/index.md)
- [Generative Agents](research-prototypes/generative-agents.md)
- [Reflexion, Voyager, and ExpeL](research-prototypes/reflexion-voyager-expel.md)
- [RecMem](research-prototypes/recmem.md)

## Open memory runtimes

- [Runtime overview](runtimes/index.md)
- [MemGPT and Letta](runtimes/memgpt-letta.md)
- [LangGraph memory](runtimes/langgraph.md)
- [Mem0](runtimes/mem0.md)
- [Zep and Graphiti](runtimes/zep-graphiti.md)

## Coding agents

- [pi, Claude Code, and OpenAI Codex](coding-agents/index.md)

## Synthesis

- [Transferable design patterns and unresolved limitations](patterns-and-limitations.md)

## Status

Phase 1 research complete. Product, documentation, repository, and publication
status was verified against primary sources on 2026-07-26. This is a dated
architecture review, not a guarantee that mutable product behavior remains
current after that date. The [source-quality audit](data/source-audit.md)
records the final metadata and status checks.

## Local References

This navigation page contains no source-dependent claims beyond the linked
reports; each report carries its own local primary-source bibliography.
