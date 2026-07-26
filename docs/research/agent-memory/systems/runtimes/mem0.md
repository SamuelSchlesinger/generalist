# Mem0

## The version boundary matters

Mem0’s 2025 research paper and its current open-source pipeline are not the same
memory algorithm. The paper describes an LLM-driven decision among add, update,
delete, and no-op while integrating new conversation facts.[paper][paper] The
official v2-to-v3 migration guide says the April 2026 OSS pipeline uses a
single-pass, **add-only** extractor and no longer performs automatic update or
delete during ingestion.[migration][migration]

The analysis below treats the migration guide and current API as authoritative
for status as of 2026-07-26. The paper remains evidence for the earlier design
and its benchmark, not for the exact current write path.

## Current write path

For a new interaction, current OSS v3:

1. retrieves up to ten related existing memories to give the extraction LLM
   deduplication context;
2. calls the LLM once to emit distinct new atomic facts;
3. embeds the batch;
4. applies an MD5 exact-duplicate check;
5. inserts the new facts; and
6. extracts and links entities.[migration][migration]

The write operation is `ADD` only. If the user first says “I live in Vienna”
and later “I moved to Graz,” both facts may coexist. Retrieval is expected to
surface the useful current fact; ingestion no longer guarantees replacement of
the stale one. Exact-hash deduplication prevents byte-identical repeats, not
semantic contradictions or near-duplicates.

This choice reduces LLM calls and avoids destructive model-generated updates,
but transfers truth maintenance to retrieval, explicit application logic, or
manual API operations.

## Retrieval and ranking

The current retrieval path preprocesses a query through lemmatization and
entity extraction, then runs semantic, BM25, and entity-based retrieval in
parallel and fuses their scores. A subtle implementation contract in the
migration guide is that BM25 and entity matching boost candidates found by
semantic retrieval; they do not independently expand recall. Optional
dependencies can degrade the pipeline to semantic search only.[migration][migration]

That hybrid is well suited to preferences containing rare exact names while
retaining paraphrase recall. It does not rank truth, source reliability,
validity time, or safety importance unless the application encodes and uses
those signals.

## Graph change

OSS v3 removes external graph-store dependencies. Entity data lives in a
parallel vector collection; the old explicit relation representation is gone
and entities are not exposed as a directly traversable knowledge graph in the
same way.[migration][migration] Therefore claims based on the paper’s graph
variant or older provider integrations should be labeled historical.

## Manual correction, deletion, and history

The current API still exposes explicit update and delete operations. A caller
can update one memory, delete one memory, or delete a set of memories under an
identity scope.[update][update] [delete][delete] [delete-many][delete-many]

The history endpoint records events such as `ADD`, `UPDATE`, and `DELETE` with
inputs, old and new memory values, metadata, and timestamps.[history][history]
This is useful audit provenance at the mutation level. It is not necessarily
source provenance for the semantic claim: unless application metadata links a
fact to exact messages and extractor versions, the history only shows how the
stored record changed.

Nor should an API `DELETE` response be equated with complete erasure. The
public endpoint documentation does not establish deletion from audit history,
logs, backups, embeddings, or downstream copies. Applications with erasure
requirements must verify those storage-specific semantics.

Mem0 also supports an explicit `expiration_date` in both Platform and OSS.
Once the UTC date has passed, normal `search()` and `get_all()` calls hide the
memory, but direct lookup by ID still returns it and clearing the date makes it
visible again. The default is no expiration. The same documentation
distinguishes Platform-only decay, which only reranks by recent use, from
expiration and deletion.[expiration][expiration] This is a particularly clear
example of soft hiding, attenuation, and record removal being separate
lifecycle operations.

## Memory classes and consolidation

The core unit is an atomic fact; categories such as semantic user preference,
episode, or procedure are not strongly enforced by the base record. Entity
metadata adds structure but is not a complete episodic/semantic/procedural
taxonomy.

Current automatic ingestion performs extraction and deduplication, not
cross-episode reflection or scheduled consolidation. There is no built-in
semantic forgetting decision in the add pipeline. Explicit expiration, manual
CRUD, and application-defined retention are distinct from an algorithm that
autonomously decides a stale fact should be retired.

## Learning verdict and evidence limits

Mem0 extracts, stores, and retrieves external facts; it does not update the
LLM’s weights. Calling this “memory learning” is reasonable at a system level
only if the external store is named as the changing artifact.

The 2025 paper reports results on the LOCOMO benchmark for its then-current
architecture.[paper][paper] Those results do not validate the April 2026
add-only pipeline. Conversely, performance claims in a vendor migration guide
are product-reported figures, not a substitute for an independently reviewed
benchmark. Version-pinned evaluation is essential.

## Operational limitations

The current design’s central unresolved problem is temporal conflict. Add-only
facts preserve history and avoid accidental overwrites, but a downstream model
may retrieve both sides or confidently select the wrong one. Other gaps include
poisoned conversation input, extractor hallucination, identity-scope mistakes,
hard-delete verification, embedding migration, and derived entity cleanup.

Mem0 is best understood as an opinionated extraction and retrieval service with
manual lifecycle APIs—not as an autonomous truth-maintenance system.

## Local References

[delete]: Mem0. “Delete a Memory,” API reference. https://docs.mem0.ai/api-reference/memory/delete-memory (accessed 2026-07-26).

[delete-many]: Mem0. “Delete Memories,” API reference. https://docs.mem0.ai/api-reference/memory/delete-memories (accessed 2026-07-26).

[expiration]: Mem0. “Memory Expiration in Mem0,” product and OSS documentation. https://docs.mem0.ai/platform/features/memory-expiration (accessed 2026-07-26).

[history]: Mem0. “Get Memory History,” API reference. https://docs.mem0.ai/api-reference/memory/history-memory (accessed 2026-07-26).

[migration]: Mem0. “Open Source v2 to v3 Migration Guide.” https://docs.mem0.ai/migration/oss-v2-to-v3 (accessed 2026-07-26).

[paper]: Prateek Chhikara et al. “Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.” 2025. https://arxiv.org/abs/2504.19413 (accessed 2026-07-26).

[update]: Mem0. “Update a Memory,” API reference. https://docs.mem0.ai/api-reference/memory/update-memory (accessed 2026-07-26).
