# Open Memory Runtimes

These projects expose memory as application infrastructure rather than one
fixed cognitive architecture. Their most important differences are who owns
the schema and lifecycle.

| Runtime | Primary durable unit | Built-in write policy | Retrieval | Correction and deletion | Status as of 2026-07-26 |
| --- | --- | --- | --- | --- | --- |
| Letta | mutable always-in-context blocks plus archival passages, files, and messages | agent tool calls or external API writes; newer Letta Code also rewrites context and skills | block injection, search over external tiers | API lifecycle; current MemFS offers Git history, but shared-block and derived-data scope still matter | MemGPT is the research lineage; V1 SDK docs are explicitly legacy; active agent work is in Letta Agent/Code |
| LangGraph | JSON graph-state checkpoints and namespaced JSON store items | application-defined graph nodes | exact get and optional semantic search | item delete, thread checkpoint delete, optional TTL in supporting stores | active orchestration primitives, not an automatic memory curator |
| Mem0 | atomic facts plus embeddings and entity metadata | current OSS v3 one-pass extraction is `ADD`-only | fused semantic/BM25/entity ranking | explicit update/delete; reversible expiration hides records, and Platform decay reranks; no automatic conflict replacement in the add path | current pipeline differs materially from the 2025 paper |
| Graphiti | raw episodes, entities, communities, and temporal fact edges | LLM extraction and graph integration on episode ingest | semantic and BM25 fusion, optionally graph and rerank recipes | hard-delete nodes/edges; managed Zep adds episode cleanup that may leave shared summaries or invalidation artifacts | open-source temporal knowledge-graph engine; managed Zep is a distinct product surface |

The status and current write-path distinctions in the table are taken from each
project’s official documentation or repository.[letta][letta]
[langgraph][langgraph] [mem0][mem0] [graphiti][graphiti]

The table should not be read as a benchmark ranking. Letta and LangGraph are
agent runtimes, Mem0 is a memory service/library, and Graphiti is a temporal
graph engine. Their abstractions can be composed, and each leaves the
application responsible for security boundaries that a memory benchmark does
not test.

## Reading path

- [MemGPT and Letta](memgpt-letta.md): virtual-context paging, mutable memory
  blocks, and newer sleep-time/context-rewriting work.
- [LangGraph](langgraph.md): checkpointed thread state versus cross-thread
  namespaced stores.
- [Mem0](mem0.md): the important split between the older
  add/update/delete-on-ingest architecture and the 2026 add-only extraction
  pipeline.
- [Zep and Graphiti](zep-graphiti.md): provenance-bearing episodic records,
  temporal facts, hybrid graph search, and nontrivial deletion.

## Shared interpretation rule

In every runtime here, “learning” normally means an external record changed or
was selected for the next prompt. None of the core memory paths described here
updates the base LLM’s weights. Application code may separately fine-tune a
model, but persistence or retrieval alone is not parametric learning.

## Local References

[graphiti]: Zep AI. “Graphiti” official source repository. https://github.com/getzep/graphiti (accessed 2026-07-26).

[langgraph]: LangChain. “Memory” and “Persistence” documentation. https://docs.langchain.com/oss/python/concepts/memory (accessed 2026-07-26).

[letta]: Letta. “Letta” official source repository. https://github.com/letta-ai/letta (accessed 2026-07-26).

[mem0]: Mem0. “Open Source v2 to v3 Migration Guide.” https://docs.mem0.ai/migration/oss-v2-to-v3 (accessed 2026-07-26).
