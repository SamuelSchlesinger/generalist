# Contributing

Generalist supports Unix-like systems only. Install the development tools and
the checked-in Git hooks once:

```sh
make setup
make doctor
```

Use `make check` before sending a change. It runs formatting, Clippy with
warnings denied, ShellCheck, the documentation traceability lint, TLC, and the
complete Rust test suite. The pre-commit hook first runs `make format-staged`
(rustfmt on fully staged `.rs` files, re-staged so the commit and the working
tree cannot diverge), then `make lint`; the pre-push hook runs `make check`.
If a Rust file has both staged and unstaged changes, the hook stops before
formatting it so unstaged work cannot be pulled into the commit accidentally.

## Runtime model review

Changes to the TUI, conversation runtime, episodic-memory lifecycle, or archive
scope routing are not complete until the Rust control flow has been traced
against the affected `spec/AsyncRuntime.tla`, `spec/MemoryRuntime.tla`, and
`spec/ArchiveScopeRuntime.tla` actions. The living state/action/invariant
mappings are in
`docs/runtime-traceability.md`. Treat that file as review evidence, not as
architecture prose that can be updated from memory.

For every change touching `src/main.rs`, `src/command.rs`, `src/goal.rs`, `src/tui.rs`,
`src/runtime.rs`, `src/agent.rs`, `src/permissions.rs`, `src/codemode.rs`,
`src/tool.rs`, `src/provider/`, `src/types.rs`, or queue/goal-bearing
persistence in `src/state.rs`, plus every change to `src/memory.rs`,
`src/history.rs`, `src/scope.rs`, or `src/tools/archive.rs`:

1. Read the changed Rust paths and the entire TLA+ action they refine. Do not
   infer equivalence from names.
2. Write down the precondition, authoritative state mutation, await or
   cancellation boundary, rollback path, durable checkpoint, and visible TUI
   effect.
3. Update the model before weakening or adding a transition. If the code is a
   deliberate refinement of an over-approximating model action, record the
   refinement boundary in `docs/runtime-traceability.md`.
4. Update every affected row of the state, action, and invariant matrices.
   `make traceability` catches missing names but cannot judge whether a row is
   truthful.
5. Add deterministic Rust evidence for the concrete path. Exercise stable IDs,
   duplicate text, claim rollback, FIFO order, safe steering, correlated
   permissions, cancellation repair, iteration limits, and history-valid
   checkpoints as applicable.
6. If the changed transition is represented by the opt-in trace vocabulary,
   update `src/model_trace.rs` and
   `examples/model_conformance.rs` at the same time. Run `make conformance`;
   `scripts/check-model-conformance.sh` must accept the real implementation
   traces and reject all deliberate mutations. The renderer invokes the
   original TLA+ actions, so do not replace a missing action with a look-alike
   predicate in the wrapper.
7. Re-run every TLC model and the implementation traces with `make tla`. A
   green finite model check plus green sampled traces establishes neither
   exhaustive Rust refinement nor concrete liveness. Record untraced paths in
   `docs/runtime-traceability.md` rather than implying coverage.
8. For interaction changes, build the exact source under review with
   `cargo build --bin generalist --locked`, then run that binary in a PTY
   against a deliberately stalled fake provider. While it is stalled, type in
   the composer, enqueue both delivery modes, edit/reorder/delete the queue,
   rapidly scroll a long transcript, answer or cancel a permission request, and
   interrupt the turn. Confirm that display-only scrolling does not change the
   autosave, that the scrollbar thumb reaches the end precisely when the final
   content row is visible, then confirm the subsequent provider request and
   queue-changing autosave, not only the rendered frame. Never infer PTY
   coverage from a pre-existing `target/` binary.
   Include a bursty streaming response, a queue longer than the modal, a draft
   beneath restore, and a resize below the normal layout. On exit, verify raw
   mode, echo/canonical input, bracketed paste, and the alternate screen were
   restored. Exercise `F3` while the fake provider advances: mouse selection
   must work, the captured frame must stay frozen, and the accumulated state
   must appear after `F3` resumes. Inject a bracketed Unicode paste after
   resuming and inspect the intercepted request. Exercise `F4` both with live
   provider reasoning and with no reasoning field; answer text must stay out of
   the inspector, reasoning must stay out of conversation text, and provider
   signatures/redacted payloads must never render.
9. Run `make check`, inspect the complete diff, and repeat the trace for any
   fix made during validation.

Pay particular attention to the races that have found real defects before:

- terminal input becoming ready at the same time as a provider response;
- cancellation becoming ready with the final permission reply;
- cancellation during the running and not-yet-started members of a tool batch;
- a denial inside a code-mode script that catches the Python exception;
- an OpenAI-compatible model returning a native tool name that was not
  advertised; it must be paired but never permission-checked or executed;
- steering after a final answer, refusal, denial, or the last iteration;
- process failure before and after a prompt claim or history checkpoint;
- stale permission IDs and duplicate prompt text.
- display-event floods starving terminal input, paused scrolling drifting during
  streaming, selection leaving a long queue viewport, and modal input leaking
  into the obscured conversation;
- partial streamed text appearing committed after cancellation, untrusted
  control bytes reaching the terminal, and startup-error paths skipping terminal
  cleanup;
- copy mode accidentally pausing the runtime, redrawing over native selection,
  or consuming pasted text; reasoning fields leaking into answer text, replaying
  unsigned reasoning to a different provider, or exposing signature/redaction
  payloads.

The checked-in pre-commit hook always presents this exact acknowledgement:

> Yes, I have updated the TLA+ model to reflect the current architecture

Answering yes means either the model changed or you painstakingly confirmed
that the current model and traceability matrix still cover the change. The
prompt is an acknowledgement, not a substitute for the review. Deliberate
non-interactive automation may set `GENERALIST_TLA_ACK=1`; ordinary local
commits must answer in a terminal.

## Memory and collaboration handoff

The model-authored `EnhancedMemoryTool` write path remains removed. The
implemented prototype is intentionally narrower than the research architecture:
opt-in scope-local settled-turn capture, one FIFO SQLite worker, scoped
conversation storage, explicit local search/export/live deletion, and
permission-gated read-only model search with no automatic prompt retrieval.
Before expanding memory, multi-agent coordination, scope policy, or offline
consolidation, read the
current [runtime traceability](docs/runtime-traceability.md), the reviewed
[research index](docs/research/agent-memory/index.md) and
[implementation handoff](docs/research/agent-memory/architecture/implementation-handoff.md).
Run `make memory-research` after editing that corpus.

Do not represent the prototype as the research design. In particular, its
same-UID in-process worker is not a supervisor security boundary, its physical
delete is only a live-store operation, and it has no trusted secret redaction,
lineage, candidates, tombstone ledger, automatic retrieval, collaboration, or
consolidation. Product-value evaluation of explicit episodic search is the next
gate; do not add those later layers merely because the schema could hold them.

Review memory changes against these boundaries:

- capture stays opt-in, visible, pausable, and bound to the active project or
  explicitly selected global scope;
- provider reasoning, signatures, redacted payloads, tool inputs, and tool
  result content are structurally omitted before the worker request;
- model-facing archive access stays read-only, permission-gated, explicitly
  scoped, sanitized, and separate from automatic prompt construction; no
  model-authored memory write or automatic retrieval may reappear without a
  new design/evaluation milestone;
- only `ToolRegistry` may mint authorization after a policy allow, and every
  public cross-scope storage operation must continue requiring an exact-input
  `DisclosureGrant`; a remembered allow-always decision creates a fresh grant
  per call rather than a scope-free storage handle;
- same-UID file modes are not worker isolation;
- every SQLite request remains off the current-thread reactor while its command
  driver continues polling terminal, queue, permission, and frame events;
- local commands bind the worker-owned current-scope key before content/ID
  matching; cross-scope model searches apply the selected scope predicate
  before content/ID matching;
- immutable-row insertion is atomic and failures create no retrievable row;
- `/memory forget` must continue to say that prior exports, backups, and
  filesystem snapshots are outside its live-store guarantee; and
- `spec/AsyncRuntime.tla`, `spec/MemoryRuntime.tla`,
  `spec/ArchiveScopeRuntime.tla`, Rust transitions, tests, CI, and this review
  methodology change together whenever their boundary changes.

If future work introduces automatic retrieval, derived records, external
workers, or multi-agent sharing, the deferred provenance, epoch, tombstone,
worker-session, and `CollaborationRuntime.tla` requirements in the research
handoff become active gates rather than optional polish.

## Change hygiene

- Keep one authoritative queue and one terminal reader.
- Do not persist a history checkpoint between an assistant tool use and its
  matching result.
- Keep local commands out of model-visible history and execute them only while
  no turn owns the agent.
- Keep slash-command parsing, composer discovery, and help sourced from
  `COMMAND_SPECS`; test subcommands such as `/goal edit` explicitly.
- Preserve user changes in a dirty worktree and keep commits focused.
- New shell scripts must be POSIX `sh`, executable, and included in ShellCheck.
- Pin downloaded development artifacts and verify their checksums.
