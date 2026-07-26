# LangGraph Memory

## Status and scope

As of 2026-07-26, LangGraph exposes two persistence primitives:

- a **checkpointer** persists short-term graph state within a thread; and
- a **store** persists arbitrary JSON documents across threads.

The framework supplies durable state, namespaces, search hooks, replay, and
deletion. It does not autonomously decide which conversation facts are true or
worthy of retention.[memory][memory] [persistence][persistence]

## Short-term memory: checkpoints

A graph is compiled with a checkpointer and invoked with a `thread_id`. At each
super-step, the checkpointer records a checkpoint of the graph’s channel
values, pending work, configuration, and metadata. A later invocation can
resume the thread from persisted state.[persistence][persistence]

This is operational or episodic memory at thread granularity. It preserves the
state machine, not merely a transcript, and supports human review, fault
recovery, and time travel. Replaying from a checkpoint reuses earlier state but
re-executes later graph steps; it should not be described as a side-effect-free
database snapshot.[timetravel][timetravel]

`delete_thread` removes a thread’s checkpoints. Otherwise checkpoint history
can grow indefinitely unless the application configures a retention process.
This deletion is narrower than deleting facts copied from the thread into a
cross-thread store.

## Long-term memory: namespaced items

The store data model is deliberately generic. An item has:

- a tuple namespace;
- a key unique within that namespace;
- a JSON `value`;
- `created_at`; and
- `updated_at`.

The base interface provides `get`, `put`, `search`, and `delete`. Namespaces are
commonly organized by user, application, or memory type. That convention is
part of the application’s authorization boundary: using a globally broad
namespace can leak memories across users even if the store behaves exactly as
documented.[store][store]

Semantic search is optional. A store may embed selected fields and rank items
for a text query; applications may also filter by metadata. Exact scoring,
reranking, and index consistency depend on the selected store implementation,
not on a universal LangGraph algorithm. TTL is likewise available only through
supporting backends and configuration and is not the default semantic
forgetting policy.[store][store]

## Semantic, episodic, and procedural patterns

LangGraph’s conceptual documentation distinguishes:

- **semantic memory**: facts, commonly kept as one mutable profile or a
  collection of atomic documents;
- **episodic memory**: past actions or examples, often used for few-shot
  prompting; and
- **procedural memory**: instructions or behavior encoded in prompts, code, or
  model weights.[memory][memory]

These are design patterns, not enforced schemas. A `BaseStore` item does not
declare its cognitive class. The profile-versus-collection choice creates a
real tradeoff: one profile is easy to inject but must be rewritten as a whole;
a collection supports narrow updates and search but can accumulate duplicates
and contradictions.

## Write triggers and consolidation

The documentation names two placement strategies:

- **hot-path writes** happen during the user interaction and make updates
  immediately available, at the cost of latency and possible response-path
  failure; and
- **background writes** extract or consolidate memory asynchronously, reducing
  latency but introducing delay and concurrency questions.[memory][memory]

LangGraph does not choose the extractor, salience trigger, conflict rule, or
consolidation prompt. A graph node can call an LLM to update a profile, append
an episode, or revise instructions, but that behavior belongs to the
application. The runtime’s success means a write was durably executed—not that
the resulting sentence is correct.

## Conflict, provenance, and deletion

For store items, `put` under the same namespace and key provides deterministic
replacement and `delete` provides deterministic logical removal. Neither
operation automatically:

- resolves two conflicting items under different keys;
- stores the source messages behind an extracted fact;
- retracts summaries derived from a deleted checkpoint;
- guarantees erasure from backend logs, replicas, or backups; or
- establishes that an LLM-generated update is newer in real-world validity
  rather than merely newer in storage time.

An application can add source IDs, validity intervals, version fields, and
confidence to JSON values. The important point is that it **must** do so if it
needs those guarantees.

## Learning verdict

Core LangGraph remembers graph state and retrieves documents. It does not learn
by itself. An application may implement reflective or procedural updates atop
the store, and a checkpointed graph can make that process reproducible, but
base-model weights remain unchanged unless a separate training system changes
them.

The most transferable lesson is separation of scopes: thread state should not
silently become global user memory, and cross-thread facts should not be
deleted merely by truncating one conversation. The largest limitation is the
mirror image of LangGraph’s flexibility: lifecycle, provenance, and semantics
are application obligations.

## Local References

[memory]: LangChain. “Memory overview,” LangGraph documentation. https://docs.langchain.com/oss/python/concepts/memory (accessed 2026-07-26).

[persistence]: LangChain. “Persistence,” LangGraph documentation. https://docs.langchain.com/oss/python/langgraph/persistence (accessed 2026-07-26).

[store]: LangChain. “BaseStore,” LangGraph Python reference. https://reference.langchain.com/python/langgraph.store/base/BaseStore (accessed 2026-07-26).

[timetravel]: LangChain. “Use time travel,” LangGraph documentation. https://docs.langchain.com/oss/python/langgraph/use-time-travel (accessed 2026-07-26).
