# Zep and Graphiti

## Scope and status

Graphiti is Zep’s open-source framework for constructing and querying temporally
aware knowledge graphs from episodes. The Graphiti paper and repository define
the open engine; Zep’s managed service exposes related but not identical APIs
and operational guarantees.[paper][paper] [repo][repo] This page describes the
open data model and official Graphiti documentation as verified on 2026-07-26.
Where managed Zep adds a service-level lifecycle contract, notably episode
deletion, that distinction is explicit.

## Graph data model

Graphiti separates source material from derived knowledge.

### Nodes

- An `EpisodicNode` retains an ingested message, JSON object, text, or fact
  triple, along with source description, timestamps, and links to extracted
  entities.
- An `EntityNode` holds an entity name, embedding, evolving summary, labels,
  and application attributes.
- A `CommunityNode` summarizes a cluster of related entities.

Base nodes carry a UUID, name, group/tenant identifier, labels, and creation
time.[nodes][nodes]

### Edges

An `EntityEdge` stores a natural-language fact, its embedding, the episodes
that support it, and multiple notions of time:

- `valid_at` and `invalid_at` represent when the fact is true in the modeled
  world;
- `created_at` records system ingestion time; and
- `expired_at` records system-side invalidation.

`EpisodicEdge` records that an episode mentions an entity. The precise fields
are visible in the source definitions rather than inferred from a product
diagram.[edges][edges]

This is a meaningful semantic/episodic distinction. Episodes are provenance
records; entities and fact edges are derived semantic memory; community
summaries are a higher abstraction. Graphiti does not provide a distinct
executable procedural-skill store.

## Ingestion and consolidation

Adding an episode invokes LLM-based extraction and entity resolution. The
system creates or links entity nodes, adds fact edges, connects them to source
episodes, and detects facts that should be invalidated by new information. It
does not simply overwrite the current profile; temporal fields can preserve
that an earlier assertion was once valid.[paper][paper]

This bitemporal shape is stronger than “last write wins.” It distinguishes
world validity from database ingestion and retains source episode IDs. The
quality of the graph still depends on LLM entity resolution, relation
extraction, and contradiction judgment. Two aliases may split one entity;
similar statements may duplicate; a new assertion can incorrectly invalidate
an old one.

## Search and ranking

Graphiti’s documented search combines semantic similarity with BM25 full-text
search and fuses rankings using reciprocal rank fusion. Search recipes can add
graph-distance constraints, temporal filters, maximal marginal relevance, or a
cross-encoder reranker, and can return edges, nodes, or communities.[search][search]

Hybrid recall is helpful for exact identifiers and paraphrases. Graph proximity
and temporal filters introduce signals absent from flat vector stores. None of
those signals by itself establishes factual correctness: a poisoned or
incorrect fact can be lexically exact, semantically close, graph-central, and
recent.

## Conflict handling

When new evidence contradicts a fact, Graphiti can mark the earlier edge
invalid or expired rather than erasing it. Query-time temporal constraints can
then select what was believed at a specified time or what is currently valid.
The supporting episode list gives an auditor a route back to raw evidence.

This is the most explicit provenance and conflict model among the runtimes in
this branch, but it remains generated. Source identity, authority, confidence,
and extraction-model version are not automatically equivalent to the episode
IDs. Applications should store those fields if decisions depend on them.

## Deletion is graph-aware but not perfect retraction

Open Graphiti’s core node and edge classes expose hard-delete methods; deleting
a node detaches its incident relationships.[crud][crud] Managed Zep adds
service-level edge, node, episode, and thread deletion. Its node deletion
cascades to connected edges, while episode deletion removes nodes and edges
only when no other episode supports them.[delete][delete]

Managed Zep’s documentation warns that deleting an episode does not regenerate
shared node names or summaries and may leave temporal invalidation state
produced during the deleted episode’s ingestion. This is a crucial distinction:
removing a provenance record is not always enough to reverse every derived
mutation. Safe erasure may require rebuilding affected summaries or replaying
the graph from retained episodes.

As with other systems, “hard delete” at the API data-model level does not by
itself prove erasure from logs, replicas, backups, or exports.

## Learning verdict and evidence

Graphiti learns no base-model parameters. It performs non-parametric learning
in the broad system sense: ingestion changes a temporal graph, and search
changes the prompt context. The Graphiti paper reports strong results on the
Deep Memory Retrieval benchmark and describes low-latency incremental graph
construction.[paper][paper] Those experiments support the retrieval
architecture under the reported setup, not a universal privacy, consistency,
or real-time guarantee.

The transferable pattern is **retain source episodes, derive temporal facts,
and invalidate rather than overwrite**. The hard parts left to deployment are
source trust, extraction correctness, tenant isolation, correction propagation,
and fully reconstructable deletion.

## Local References

[crud]: Zep. “CRUD Operations,” Graphiti documentation. https://help.getzep.com/graphiti/working-with-data/crud-operations (accessed 2026-07-26).

[delete]: Zep. “Deleting Data from the Graph,” managed Zep documentation. https://help.getzep.com/deleting-data-from-the-graph (accessed 2026-07-26).

[edges]: Zep AI. `graphiti_core/edges.py`, Graphiti source. https://github.com/getzep/graphiti/blob/main/graphiti_core/edges.py (accessed 2026-07-26).

[nodes]: Zep AI. `graphiti_core/nodes.py`, Graphiti source. https://github.com/getzep/graphiti/blob/main/graphiti_core/nodes.py (accessed 2026-07-26).

[paper]: Preston Rasmussen et al. “Zep: A Temporal Knowledge Graph Architecture for Agent Memory.” 2025. https://arxiv.org/abs/2501.13956 (accessed 2026-07-26).

[repo]: Zep AI. “Graphiti” official source repository. https://github.com/getzep/graphiti (accessed 2026-07-26).

[search]: Zep. “Searching the Graph,” Graphiti documentation. https://help.getzep.com/graphiti/working-with-data/searching (accessed 2026-07-26).
