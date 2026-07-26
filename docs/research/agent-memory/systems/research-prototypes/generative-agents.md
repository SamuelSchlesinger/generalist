# Generative Agents

## Status and actual claim

Publication and source status in this report was checked on 2026-07-26.
Generative Agents is a UIST 2023 research prototype in which 25 LLM-driven
characters inhabit a Smallville simulation. Its contribution is an agent
architecture—memory stream, retrieval, reflection, and planning—and a
believability study, not a production personal-memory service or a learned
long-term model.[paper][paper] The official repository supplies the simulation
implementation and serialized-agent examples.[repo][repo]

## Data model

An agent’s memory stream is an append-only list of natural-language records.
An ordinary record contains:

- a textual description of an observation or event;
- a creation timestamp;
- the timestamp of its most recent retrieval; and
- an importance score assigned when it is written.

Reflections and plans are written into the same stream. A reflection additionally
points to the lower-level memories from which it was inferred. This gives a
limited derivation trail: the architecture can name supporting record IDs, but
it does not record a calibrated confidence, source authority, or a reproducible
proof that the inference follows.[paper][paper]

The categories resemble episodic and semantic memory but are not enforced as
separate stores. Observations are event-like; reflections are higher-level
generalizations; plans are future-oriented procedural context. All remain text
records in one stream.

## Write, retrieval, and ranking

Every perceived event can be appended as an observation. At query time the
system scores a candidate memory \(m\) with three normalized terms:

\[
\operatorname{score}(m)
= \alpha_r\,\operatorname{recency}(m)
+ \alpha_i\,\operatorname{importance}(m)
+ \alpha_q\,\operatorname{relevance}(m,q).
\]

The paper’s implementation gives the terms equal weight. Recency decays
exponentially with elapsed simulation time since last access. Importance is a
one-time LLM rating of the record’s likely long-term significance. Relevance is
cosine similarity between embedding vectors for the query and memory. The
highest-scoring records that fit the prompt budget are returned.[paper][paper]

This is a useful hybrid, but each term has a distinct failure mode:

- repeated retrieval refreshes “last access,” creating a rich-get-richer loop;
- an uncalibrated LLM assigns importance before later evidence is known; and
- embedding similarity favors lexical or topical proximity, not truth,
  authority, or causal usefulness.

No component verifies a memory against the environment before retrieval.

## Reflection is threshold-triggered consolidation

The agent periodically sums the importance of recent memories. When the total
crosses a fixed threshold, an LLM generates salient questions over the recent
stream, retrieves relevant records for each question, and produces higher-level
insights. Those insights, with pointers to their source records, are appended
as new memories. Reflections can later support further reflections, producing a
tree of increasingly abstract statements.[paper][paper]

This is not a recurrence test. A single highly rated event can help trip the
threshold; several mundane but repeated facts may never do so. The trigger
therefore measures predicted salience, not evidence multiplicity. RecMem’s
recurrence gate is a materially different answer to when consolidation should
happen.

## Planning and action

The agent creates broad daily plans and recursively decomposes them into
time-bounded actions. Plans are conditioned on the current state and retrieved
memories, then placed back in memory so later action selection can consult
them. New observations can cause replanning. This closes the loop:

1. perceive and append an event;
2. retrieve relevant past records;
3. occasionally derive reflections;
4. plan using retrieved and derived records; and
5. create new events that enter the stream.

The loop is behaviorally adaptive, but its model weights remain fixed.

## Conflict, deletion, and provenance

The paper specifies neither deletion nor contradiction resolution. Exponential
recency is attenuation, not erasure: old records remain in the stream and can
resurface through importance or relevance. A later contradictory observation
does not invalidate an earlier one, and a reflection can preserve a conclusion
after its premises have become stale. Source pointers help inspect reflection
lineage but do not repair these cases.

For a production memory system, the missing contracts are consequential:
identity and tenant boundaries, user-visible correction, expiry, cascaded
retraction of derived statements, treatment of malicious observations, and
hard-delete behavior are all outside the prototype.

## Evidence and limits

The authors evaluate individual architecture components by interviewing agents
and compare the full architecture with ablations. They also demonstrate
emergent social diffusion in the 25-agent simulation. This supports the claim
that retrieval, reflection, and planning improve perceived believability in
that environment.[paper][paper] It does not establish factual consistency over
months, adversarial robustness, privacy, or reliable transfer to tool-using
agents.

The official repository further warns, through its research-oriented setup and
serialized simulation workflow, that this is an experimental artifact rather
than a maintained general memory API.[repo][repo]

## Bottom line

Generative Agents supplied the canonical “store raw experience, retrieve a
small context, sometimes reflect” pattern. Its durable state is an external
text log plus derived text—not learned parameters. Its most reusable ideas are
multi-signal ranking and source-linked reflection. Its largest omissions are
precisely the hard lifecycle questions: correction, conflict, deletion,
provenance quality, and security.

## Local References

[paper]: Joon Sung Park et al. “Generative Agents: Interactive Simulacra of Human Behavior.” UIST 2023. https://arxiv.org/abs/2304.03442 (accessed 2026-07-26).

[repo]: Joon Sung Park et al. “generative_agents” official source repository. https://github.com/joonspk-research/generative_agents (accessed 2026-07-26).
