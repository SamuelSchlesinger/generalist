# Preserved results

Each result file is append-only JSONL with a provenance record followed by one
evaluation result. Generated files include the exact corpus, binary, and
evaluator hashes and whether the worktree was dirty. Do not compare latency
across machines without accounting for platform and load.

The deterministic provider fixture uses only loopback HTTP and returns
prompt-derived acknowledgements. It is suitable for runtime/lifecycle checks,
not model-quality claims.

- `20260805T000737Z-local-explicit-memory.jsonl` preserves the first failed run;
  a probe sampled provider requests after the memory command had already timed
  out and misclassified the queued dispatch as concurrent with the command.
- `20260805T000829Z-local-explicit-memory.jsonl` passes the corrected sampling
  point.
- `20260805T000944Z-local-explicit-memory.jsonl` adds an exact paused-capture
  TUI turn and separates logical retention from SQLite/WAL allocation.
- `20260805T001720Z-local-explicit-memory.jsonl` adds exact `/history` search
  and show coverage after the host inspection feature was implemented.
- `20260805T001838Z-local-explicit-memory.jsonl` adds exact `/history`
  search/show coverage and records zero provider requests with byte-for-byte
  unchanged autosave content.
- `20260805T003227Z-local-explicit-memory.jsonl` adds
  direct named save/load, default-Cancel and confirmed deletion, on-disk
  absence, and live-autosave refusal while preserving zero provider requests
  and the autosave content hash.
- `20260805T004243Z-local-explicit-memory.jsonl` adds default-Cancel and
  confirmed replacement plus manual-autosave refusal.
- `20260805T004549Z-local-explicit-memory.jsonl` repeats
  that complete lifecycle after final manual-name normalization and observed
  trace updates; all lifecycle predicates pass with zero provider requests.
- `20260805T004820Z-local-explicit-memory.jsonl` is the last pre-usage run. It
  rebuilds after the storage API itself also reserved the live autosave name;
  every retrieval, liveness, replacement, and deletion predicate passes.
- `20260805T010103Z-local-explicit-memory.jsonl` adds
  exact `/usage show|reset` coverage: four deterministic reports total four
  input/four output tokens, absent cache fields retain zero-of-four coverage,
  reset reaches an empty ledger, and the commands make zero provider requests
  while preserving autosave bytes.
- `20260805T011743Z-local-explicit-memory.jsonl` adds the first fail-once MCP
  recovery probe.
- `20260805T012248Z-local-explicit-memory.jsonl` is authoritative. It preserves
  every prior predicate and adds fail-once MCP recovery: visible startup
  failure, targeted retry, connected status, and immediate recovered-tool
  inspection, plus refusal of connected and unknown retry targets, with zero
  provider requests.
