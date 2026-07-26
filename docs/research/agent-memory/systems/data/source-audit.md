# Source-Quality Audit

## Scope

This audit covers only the `systems/` branch and was completed on 2026-07-26.
It checks whether system claims rest on primary sources, whether publication
metadata matches the canonical paper page, and whether change-sensitive
product claims are explicitly dated.

## Verification checks

| Target | Authoritative source | Check performed | Result |
| --- | --- | --- | --- |
| RecMem | ACL Anthology paper page and official repository | exact lead author, title, venue, pages, DOI, and implementation status | corrected to Zijie Dai et al.; exact Findings of ACL 2026 title and metadata now used |
| Generative Agents, Reflexion, Voyager, ExpeL, MemGPT, sleep-time compute, Mem0, Zep | arXiv paper pages, plus official repositories where implementation claims are made | lead author and exact title | passed after correcting the Mem0 lead author to Prateek Chhikara and attributing the Zep paper to Preston Rasmussen et al. |
| MetaReflection, Mistake Notebook Learning, ReMe | ACL Anthology paper pages | lead author, exact title, venue, and year | passed |
| Letta | official repositories and V1 SDK documentation | legacy/current boundary | V1 server and documentation are labeled legacy; newer Letta Agent/Code claims are kept separate |
| Mem0 | official v2-to-v3 migration guide and current API/lifecycle reference | paper/current algorithm boundary, CRUD, expiration, and decay scope | April 2026 add-only ingestion is separated from the older paper architecture; deletion is kept distinct from reversible expiration and Platform-only decay |
| pi | current official `earendil-works/pi` repository | session, branch, and compaction contract | current package/repository path used; compaction explicitly described as lossy active-context projection |
| Claude Code | official product documentation | auto-memory scope, load limit, edit/delete, and conflict behavior | claims dated to 2026-07-26 |
| OpenAI Codex | official source repository | trigger, two phases, feature status, prompt contracts, and reset scope | memory feature is currently `Stable` and disabled by default; experimental app-server controls are labeled separately |
| LangGraph | official concepts, persistence, time-travel, and API reference | framework guarantees versus application policy | automatic learning or conflict resolution is not attributed to the core runtime |
| Graphiti | official paper, source definitions, and documentation | node/edge schema, ranking, temporal invalidation, and deletion caveat | open-source Graphiti hard deletion is kept distinct from managed Zep’s episode-deletion contract |

## Corrections made during audit

Three bibliography defects were found and corrected:

- RecMem’s lead author and full paper title, verified against the ACL
  Anthology record.[recmem][recmem]
- Mem0’s lead author.[mem0-paper][mem0-paper]
- The Zep paper’s author attribution.[zep-paper][zep-paper]

The audit also caught a status drift: Codex’s current feature registry marks
the memory feature stable but disabled by default, rather than categorically
experimental.[codex-features][codex-features] The `memory/reset` and thread
memory-mode app-server methods remain experimental surfaces.
[codex-server][codex-server]

## Remaining source limitations

- Product documentation and repository `main` branches are mutable. Access
  dates make the snapshot explicit, but most links are not commit-pinned.
- Official vendor benchmarks are primary evidence for what the vendor
  evaluated, not independent replication.
- Repository documentation can lead or lag a released package. This branch
  reports the source contract visible on the audit date, not behavior verified
  across every packaged version.
- A structural validator can check references and links; it cannot prove the
  semantic truth of an LLM-generated summary. The whole-corpus review should
  still spot-check high-impact numbers and deletion claims against the cited
  page or source file.

## Local References

[codex-features]: OpenAI. `codex-rs/features/src/lib.rs`, Codex feature registry. https://github.com/openai/codex/blob/main/codex-rs/features/src/lib.rs (accessed 2026-07-26).

[codex-server]: OpenAI. “Codex App Server,” source protocol documentation. https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md (accessed 2026-07-26).

[mem0-paper]: Prateek Chhikara et al. “Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.” 2025. https://arxiv.org/abs/2504.19413 (accessed 2026-07-26).

[recmem]: Zijie Dai et al. “RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents.” Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.1619/ (accessed 2026-07-26).

[zep-paper]: Preston Rasmussen et al. “Zep: A Temporal Knowledge Graph Architecture for Agent Memory.” 2025. https://arxiv.org/abs/2501.13956 (accessed 2026-07-26).
