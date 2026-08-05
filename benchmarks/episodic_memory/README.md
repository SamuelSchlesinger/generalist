# Explicit episodic-memory evaluation

This benchmark evaluates the shipped Stage 1 prototype before any automatic
retrieval or consolidation is considered. It deliberately separates three
kinds of evidence:

1. deterministic storage/retrieval behavior through the real Rust APIs and
   permission-gated archive tools;
2. exact-binary TUI responsiveness under a stalled provider and a held SQLite
   write lock; and
3. model answer quality, which this local fixture does **not** claim to
   measure.

The deterministic corpus covers repository conventions, a changing
preference, a failed-command precondition, a prospective trigger, a rare
safety constraint, a non-generalizable one-off procedure, a correction, an
unsupported query, and a same-token collision in another project.

## Conditions

- `b0_paused`: episodic capture remains paused.
- `b1_episodic`: every settled session is retained and search is explicit.
- `history_autosave`: only the latest ordinary autosave remains searchable.
- `history_named`: the user deliberately names and retains every session.

The last two are important controls. Episodic capture should beat ordinary
latest-state autosave on long-range recall, but it has not created semantic
value if disciplined named saves produce the same retrieval result.

The corpus marks only the newer `PREF-CARBON` record as current. Literal search
is therefore expected to return one stale false positive alongside the newer
record. This is evidence that Stage 1 surfaces conflict but does not resolve
truth or time.

## Run

```sh
python3 benchmarks/episodic_memory/run.py
```

The runner rebuilds the exact binary and evaluator, then writes an exclusive,
mode-`0600`, two-record JSONL file beneath `results/`: one provenance record
and one result record. It records hashes, Git state, platform versions,
retrieval precision/recall/rank, p50/p95/max latency, allocated and logical
volume, scope isolation, live deletion/restart behavior, SQLite-lock liveness,
and subprocess-exit schedules.

SQLite allocation includes the database, WAL, and shared-memory sidecars and
is recorded both before and after deletion. It is intentionally labeled
separately from the serialized logical export: page allocation and checkpoint
timing mean a zero-row database can occupy more live bytes than a recently
checkpointed nonempty database.

`tmux` is required for the exact-TUI probe. A storage/crash-only run is
available for constrained environments:

```sh
python3 benchmarks/episodic_memory/run.py --skip-ui
```

The TUI liveness budget is 750 ms from sending a short probe to seeing it in
the compositor/queue. The threshold is a regression tripwire, not a claim
about human-perceived end-to-end model latency.

The same PTY leg sends 30,000 one-byte streaming deltas. It waits until 20,000
have crossed the loopback provider boundary, types a composer probe while the
remaining stream is live, and requires both sub-budget input visibility and the
committed tail of the response. A Rust unit flood separately proves that one
million alternating text/reasoning callbacks occupy one 16 KiB preview record;
the PTY measurement is a responsiveness check, not a heap measurement.

The exact-TUI leg also exercises the complete named-session lifecycle without
provider assistance: direct save and load with a space-bearing name,
atomic no-clobber creation, default-Cancel and confirmed replacement,
default-Cancel and confirmed durable deletion, absence from later inspection,
and refusal to manually save or delete the live autosave. It checks the named
file and autosave content hash on disk and requires zero additional provider
calls.

Before model traffic, a deterministic stdio MCP fixture exits on its first
handshake and succeeds on its second. The probe requires visible failed status,
targeted `/mcp retry flaky`, 1/1 connected status, and immediate inspection of
the recovered `flaky_ping` bridge with zero provider requests. It also requires
refusal of a redundant connected-server retry and an unknown server name.

The same PTY process checks `/usage` after four deterministic calls (four input
and four output tokens), verifies that absent cache fields are labeled with
zero-of-four report coverage, resets the ledger, and observes the empty state.
Those host commands must make zero provider calls and leave autosave bytes
unchanged. This validates accounting/control plumbing, not vendor token
accuracy or monetary cost.

Run the static harness tests with:

```sh
python3 -m unittest discover -s benchmarks/episodic_memory -p 'test_*.py'
```

## Exit-boundary interpretation

The evaluator launches itself as a child and exits without running Rust
destructors at several delays after `enqueue_settled_turn`. A row may be absent
or present, because enqueue is intentionally not a durable acknowledgement;
any present row must be one complete immutable episode. Separately, every
awaited `record_settled_turn` must be present and complete. Gated children are
killed while an external `BEGIN IMMEDIATE` prevents insertion, and must leave
no row. Every schedule ends with `PRAGMA integrity_check` and an attempted
immutable-row update.

This covers the current one-statement episode insert. It does not prove the
future draft/event/finalize protocol described in the research corpus, because
that protocol has not been implemented.
