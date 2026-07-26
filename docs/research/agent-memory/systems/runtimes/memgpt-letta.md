# MemGPT and Letta

## Lineage and current status

MemGPT introduced an operating-system analogy for LLM context: a small “main
context” acts like RAM, external stores act like slower storage, and the agent
uses function calls to move information between them. The model remains fixed;
the novelty is self-directed virtual context management.[memgpt][memgpt]

The project became Letta. As of 2026-07-26, the `letta` repository identifies
itself as the legacy V1 server and points active development to newer Letta
Agent and Letta Code surfaces.[letta-repo][letta-repo] The V1 documentation is
still useful because it states the memory contract precisely, but it must not
be presented as the only current product architecture.

## V1 memory hierarchy

### Memory blocks

A memory block is a typed mutable text region that is always placed in the
agent’s context. Its key fields are:

- `label`, unique within an agent’s set of attached blocks;
- `description`, which tells the agent how the block should be used;
- `value`, the actual remembered text; and
- `limit`, the maximum block size.

Blocks are read-write by default and may be marked read-only. They can be
attached, detached, or shared between agents. An external update replaces the
block value rather than semantically merging it.[blocks][blocks]

Agents can use tools such as `memory_insert`, `memory_replace`, and
`memory_rethink` to edit blocks. The tool name indicates the edit shape, not a
truth guarantee: an LLM can still preserve a stale fact or rewrite away a
qualification. A shared block also complicates deletion because detaching it
from one agent is different from destroying the shared object.

### External tiers

The V1 context hierarchy distinguishes:

- **messages and agent state**, persisted across interactions;
- **memory blocks**, always injected and agent-editable;
- **files**, partially loaded and read-only to the agent, with open, close,
  semantic-search, and grep operations; and
- **archival memory**, an external read-write passage store reached through
  insert and search tools.[hierarchy][hierarchy]

These are engineering tiers, not a mandatory semantic/episodic/procedural
ontology. A developer can place a user profile, a procedure, or an event
summary in a block. The framework controls placement and access; prompts and
application code control meaning.

## Triggers, ranking, and consolidation

Classic MemGPT lets the LLM decide when to call memory-management functions as
context pressure and conversation demand change. Archival recall is search
driven; always-in-context blocks need no ranking. This gives the agent agency
over writes but also makes retention sensitive to tool-call behavior and prompt
quality.[memgpt][memgpt]

Letta’s sleep-time work explicitly moves some state maintenance off the
interactive path. A background “sleep-time” agent can process experience and
rewrite shared memory while the primary agent responds with lower latency.
The paper evaluates this separation in stateful math and software-engineering
settings.[sleep][sleep] It is offline context optimization, not offline weight
training.

## Newer Letta Code behavior

The current Letta Code repository describes agents that can rewrite their
context, system prompt, and skills; periodically “dream” through `/sleeptime`;
audit state through `/doctor`; and inspect it through `/palace`. Its MemFS
design keeps context artifacts, including memory blocks, under Git and can sync
them to GitHub.[letta-code][letta-code]

Git history materially improves audit and rollback compared with an opaque
mutable block. It does not by itself solve:

- whether a generated change is semantically correct;
- whether secrets or poisoned instructions should have entered the repository;
- which upstream episode justified a generalized skill;
- whether deleting the visible current value also erases Git history, remote
  copies, indices, and backups; or
- whether a rollback restores all external side effects.

Current Letta Code and legacy V1 should therefore be described separately:
they share the persistent-agent lineage, but their state and operational
contracts are not interchangeable.

## Retrieval versus learning

MemGPT/Letta is more than pure retrieval because the agent may rewrite its
external state. It still does not update the base model’s weights in the memory
path. The learned artifact is a block, passage, prompt, or skill; inference
improves only when the runtime injects or searches that artifact.

This makes deletion and correction technically possible but semantically hard.
Deleting one passage is straightforward. Retracting every summary, skill, Git
commit, and shared block derived from it requires explicit lineage that the
basic block API does not enforce.

## Strengths and limitations

The architecture’s durable strength is that memory is inspectable and
tool-addressable. Always-in-context blocks provide predictable recall for a
small profile; archival stores prevent every detail from consuming the prompt;
background compute can consolidate without increasing every response’s
latency.

The corresponding risks are agent-controlled self-editing, context-injection
attacks, mutable summaries without mandatory source links, shared-state
authorization, and lifecycle ambiguity across product generations. The MemGPT
paper and sleep-time evaluations establish useful mechanisms and bounded task
improvements, not production privacy, truth maintenance, or complete erasure.

## Local References

[blocks]: Letta. “Memory Blocks,” V1 SDK documentation (legacy). https://docs.letta.com/v1-sdk/memory/memory-blocks (accessed 2026-07-26).

[hierarchy]: Letta. “Context Hierarchy,” V1 SDK documentation (legacy). https://docs.letta.com/v1-sdk/memory/context-hierarchy (accessed 2026-07-26).

[letta-code]: Letta. “Letta Code” official source repository. https://github.com/letta-ai/letta-code (accessed 2026-07-26).

[letta-repo]: Letta. “Letta” official source repository and V1 status notice. https://github.com/letta-ai/letta (accessed 2026-07-26).

[memgpt]: Charles Packer et al. “MemGPT: Towards LLMs as Operating Systems.” 2023. https://arxiv.org/abs/2310.08560 (accessed 2026-07-26).

[sleep]: Kevin Lin et al. “Sleep-time Compute: Beyond Inference Scaling at Test-time.” 2025. https://arxiv.org/abs/2504.13171 (accessed 2026-07-26).
