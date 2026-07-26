# Async runtime model traceability

This document is the refinement review between the executable Unix TUI and
`spec/AsyncRuntime.tla`. It must be read with the source. TLC proves properties
of the finite model in `spec/AsyncRuntime.cfg`; the tables below are the
separate, human-reviewed argument that the Rust paths implement those modeled
transitions.

## Scope and refinement direction

The model covers one conversation's asynchronous prompt queue, one active
model turn, safe-point steering, tool-result pairing, permission correlation,
bounded continuation, and cooperative cancellation. A concrete Rust execution
must map to a model execution after hiding rendering, text payloads, provider
wire formats, disk writes, and tool side effects.

The model deliberately permits `Enqueue` in every runtime phase. Permission,
help, and queue modals temporarily accept a smaller set of terminal keys, so
the TUI refines that environment by disabling some user transitions; it never
adds an enqueue transition forbidden by the model. Local commands, goal
editing, manual save/load/compaction, startup tool discovery, and provider
selection execute only outside a mutation-capable turn and are reviewed
separately for that ownership guard. Goal text and other provider payloads are
hidden data in this model: an idle goal mutation refines a TLA+ stutter step.
Reasoning-inspector contents and scroll are also hidden display data. Copy-mode
ownership itself is modeled because it gates all concrete terminal actions:
while `copyMode` is true, provider/tool progress remains enabled but queue,
cancel, and permission-choice input is disabled.

## State mapping

| TLA+ state | Authoritative Rust representation | Review note |
| --- | --- | --- |
| `copyMode` | `tui::AppState::copy_mode`, changed only by `TerminalUi::toggle_copy_mode` | It controls terminal ownership/redraw only; no queue, history, permission ID, or turn state moves with the toggle. |
| `phase` | Control location in `main::drive_started_turn`, `Agent::run_started_turn`, and the permission broker | It is a model program counter, not a second mutable Rust enum that could drift. |
| `activeTurn` | The one prompt synchronously recorded by `Agent::begin_turn` before the controller pins the sole `&mut Agent` future | The current-thread reactor cannot start another turn until that future returns. |
| `queue`, `delivery` | `runtime::PromptQueue` containing `QueuedPrompt { id, text, delivery }` | `Rc<RefCell<_>>` is the sole store; `tui::AppState::queue` is a render snapshot only. |
| `lifecycle`, `claimedSteers` | `PromptClaim` ownership plus visible membership in `PromptQueue` | `lifecycle` is a ghost variable. `Drop` requeues uncommitted IDs; `commit` consumes them. |
| `settledTurns`, `interruptedTurns`, `committedOrder` | `TurnOutcome`, `SteeringCommitted`, and the ordering of queue claim events | These are proof-history variables; committed prompt IDs are not duplicated in model-visible message text. |
| `toolUses`, `toolResults` | `tool_uses`, `results`, and `index` in `Agent::run_started_turn` | The result message is appended only after every ID has a real or synthetic result. |
| `permission`, `permissionOwner`, `usedRequests` | `PermissionRequest::id`, its one-shot sender, and `pending_permission` in the reactor | The one-shot channel is the ownership link; the modal ID prevents a stale key action from answering another request. |
| `continuationNeeded`, `terminalReason` | Response tool presence, `denied`, refusal handling, and the iteration-limit branch | The TLA+ terminal reasons distinguish answer, refusal, denial, and cap because they accept steering differently. |
| `roundsLeft` | `Agent::max_iterations` minus the `for iteration` program counter | Every provider response consumes one iteration. |
| `failuresLeft` | Finite TLC bound for `PromptClaim` rollback branches | Rust does not count failures; RAII permits any individual uncommitted claim to roll back. |

## Action mapping

| TLA+ action | Concrete Rust transition | Deterministic evidence |
| --- | --- | --- |
| `Enqueue` | `tui::submission_delivery` selects steer/follow-up; `main::enqueue_submission` forces idle and local-command submissions to follow-up; `PromptQueue::enqueue` assigns the ID. | `composer_keys_distinguish_busy_steering_from_followups`, `duplicate_text_keeps_distinct_stable_ids` |
| `DeleteQueued` | Queue modal deletion routes the selected stable ID to `PromptQueue::delete`. Restore is the same modeled removal plus a display-only move into an empty composer; it is refused over an existing draft. | `queue_manager_mutations_address_stable_ids`, `restoring_queue_text_never_overwrites_a_draft` |
| `ReclassifyQueued` | Queue modal `s` calls `PromptQueue::toggle_delivery`; idle cannot create a steer. | `queue_manager_mutations_address_stable_ids`, `ending_turn_normalizes_undelivered_steers` |
| `MoveQueuedEarlier` | Queue modal Ctrl+Up/Down calls `PromptQueue::move_by` by ID. | `queue_manager_mutations_address_stable_ids` |
| `EnterCopyMode` | F3 is intercepted before modal/composer dispatch, disables mouse capture, sets `copy_mode`, draws the paused banner once, and then suppresses application input/redraws. | `copy_mode_banner_explains_that_rendering_is_paused`; exact-binary PTY copy test |
| `ExitCopyMode` | The next F3 reenables mouse capture, clears `copy_mode`, and performs one immediate redraw of state accumulated by the still-polled runtime. | `copy_mode_banner_explains_that_rendering_is_paused`; exact-binary PTY progress test |
| `DispatchFollowUp` | The idle outer loop calls `claim_follow_up` once; it never scans past a non-follow-up head. | `followups_dispatch_one_at_a_time_in_fifo_order` |
| `CommitStart` | Without an await, the controller calls `Agent::begin_turn`, commits the `PromptClaim`, and atomically writes the started history plus remaining queue. | `dropped_claim_rolls_back_and_commit_removes`; source-order review in `main` |
| `RequeueStart` | Dropping an uncommitted `PromptClaim` restores the same ID at the front. Production start is currently infallible between claim and commit; the action over-approximates unwinding and future fallible setup. | `dropped_claim_rolls_back_and_commit_removes` |
| `ProviderAnswer` | `complete_with_retry` commits a complete assistant response, emits no partial response into history, then reaches the no-tools boundary. Text and provider-supplied reasoning are separate display deltas; aborted streams receive separate display-only uncommitted markers. | `steering_queued_during_final_response_gets_another_model_call`, `provider_cancellation_commits_no_partial_assistant_message`, `cancelling_a_partial_stream_marks_the_visible_text_uncommitted`, `streamed_text_is_not_double_emitted`, `provider_reasoning_stays_out_of_chat_and_has_a_live_inspector` |
| `ProviderRefusal` | Refusal pairs any anomalous tool uses with synthetic errors, checkpoints, and returns without accepting steering. | `refusal_with_tool_uses_is_repaired_before_checkpointing` |
| `ProviderToolBatch` | A committed assistant response is scanned into a finite `tool_uses` vector; one loop iteration is consumed. | `tool_results_are_truncated_in_history`, `iteration_limit_leaves_late_steering_for_controller_normalization` |
| `CompleteTool` | Tools run sequentially; each outcome becomes one `ToolResult`. Truncation and unknown tools also return structured results. In code mode, an undeclared native call is never started or permission-checked and receives a synthetic error result. | `tool_results_are_truncated_in_history`, `history_survives_api_errors_after_tool_execution`, `code_mode_rejects_unadvertised_direct_tool_calls` |
| `AskPermission` | `PermissionBrokerPrompt::choose` allocates a monotonic request ID, sends one `PermissionRequest`, and awaits its one-shot. | `broker_correlates_the_ui_reply_with_its_request` |
| `AllowPermission` | The reactor sends the choice only when the modal ID equals the live request ID; the handler records `AllowAlways` before execution. | `broker_request_ids_keep_out_of_order_replies_correlated`, `memory_handler_remembers_decisions_without_prompting` |
| `DenyPermission` | A denial becomes a structured denied result. Denials inside code-mode bridge calls propagate through `ScriptResult::denied` even if Python exits successfully. | `dropped_broker_reply_denies_instead_of_hanging`, `denial_inside_code_mode_pauses_the_outer_turn` |
| `PermissionResolution` (fairness action) | This is the union of allow/deny, not another Rust transition. Both choices are unavailable while F3 owns input and become available again on resume. | broker correlation tests; exact-binary permission-during-copy PTY trace |
| `ClaimSteering` | At a history-valid boundary, `PromptQueue::claim_steering` removes all visible steers, preserves their relative order, and leaves follow-ups. | `steering_claim_preserves_relative_order_and_followups` |
| `CommitSteering` | `Agent::commit_steering` appends claimed text to the valid user boundary, commits IDs, emits `SteeringCommitted`, and checkpoints with no await between those operations. | `steering_queued_during_final_response_gets_another_model_call` |
| `RequeueSteering` | Dropping an uncommitted steering claim restores the same IDs at the front. Normal commit contains no fallible/await boundary. | `dropped_steering_claim_restores_the_same_ids_at_the_front` |
| `ContinueAfterTools` | A complete, non-denied tool-result batch with capacity remaining loops to the next provider request. | `tool_results_are_truncated_in_history`, `history_survives_api_errors_after_tool_execution` |
| `SettleTurn` | Answer, refusal, denial, or iteration cap returns a `TurnOutcome`; the controller releases `&mut Agent`, converts remaining steers to follow-ups, and writes one atomic autosave containing history and queue. | `refusal_with_tool_uses_is_repaired_before_checkpointing`, `denial_inside_code_mode_pauses_the_outer_turn`, `iteration_limit_leaves_late_steering_for_controller_normalization` |
| `RequestCancel` | Esc/Ctrl+C retires a live permission with deny-once, sets the turn-scoped watch flag, and keeps polling the controlled future until repair completes. | `cancellation_wins_over_a_ready_permission_before_steering`, `provider_cancellation_commits_no_partial_assistant_message` |
| `RepairCancelledTool` | Cancellation drops the running tool future and emits synthetic error results for it and every unstarted tool use. | `interruption_pairs_the_running_and_unstarted_tool_uses`, `history_tool_protocol_is_valid` debug assertion |
| `FinishCancellation` | After results are appended, the agent checkpoints and returns `Interrupted`; the controller retires nested TUI activity and normalizes undelivered steers. | `interruption_pairs_the_running_and_unstarted_tool_uses`, `interrupted_turn_retires_all_nested_activity`, `ending_turn_normalizes_undelivered_steers` |
| `IdleWait` | With no follow-up, the outer `tokio::select!` continues polling terminal input, stale permission events, and frame ticks without owning `Agent` mutably. | `composer_keys_distinguish_busy_steering_from_followups`, source-order review in `main` |

## Property mapping

| TLA+ property | Rust enforcement and review evidence |
| --- | --- |
| `TypeOK` | Rust enums and ownership constrain concrete values; TLC separately checks every modeled variable over the configured state space. |
| `QueueIdentity` | `PromptQueue` owns one vector, saved duplicate IDs are filtered, all mutations address IDs, and claims remove before returning. Covered by the runtime queue tests. |
| `SingleTurnOwnership` | A current-thread reactor pins one future borrowing `&mut Agent`; the outer loop cannot dispatch again until it returns. No `Arc<Mutex<Agent>>` or background runtime exists. |
| `DeliveryIsWellFormed` | `DeliveryMode` has only two variants; idle submissions are forced to follow-up and all residual steers are normalized when ownership ends. |
| `SafeSteeringBoundary` | `commit_steering` is called only after a complete assistant answer or after the full result vector is appended. It is not called on refusal, cancellation, or an exhausted iteration budget. |
| `TerminalReasonIsWellFormed` | Distinct Rust branches implement final answer, refusal, structured denial, and cap. The cancellation/permission race test prevents a cancelled turn from taking the denial-to-steer branch. |
| `ToolHistoryIsValid` | `history_tool_protocol_is_valid` checks exact adjacent ID sets; every emitted `HistoryCheckpoint` debug-asserts it, every persistence path refuses an invalid history, load rejects invalid saves, and cancellation/refusal tests inspect checkpoint histories. |
| `PermissionIsCorrelated` | IDs are monotonic, choices travel over the request's own one-shot, mismatched modal IDs are ignored, and dropping a reply denies once. |
| `SettledPromptsAreCommitted` | The controller records the initial user message before `PromptClaim::commit`; interrupted and ordinary outcomes therefore refer to a committed follow-up. |
| `HistoryOrderHasStableIds` | Queue claims and `SteeringCommitted` preserve stable-ID order. `committedOrder` is a model ghost variable, verified through claim/event tests rather than stored in model-visible text. |
| `EveryBusyPeriodSettles` | TLC checks settlement for the finite bounds under weakly fair agent progress, weakly fair copy exit, and strongly fair permission resolution when its UI is available infinitely often. Rust bounds provider rounds/retries and keeps polling them during copy mode; permission input waits for resume. External tools/providers, a user who never resumes, or a user who never answers remain environmental blockers. |
| `CopyModeEventuallyResumes` | `WF_vars(ExitCopyMode)` makes the environmental assumption explicit rather than silently proving liveness through a permanently disabled terminal. The PTY test verifies the concrete exit path; it cannot force a real user to press F3. |

## Durable-boundary refinement

Disk state is an implementation strengthening, not a TLA+ variable. The
controller keeps a clone of the latest history-valid boundary and active goal
while the agent future owns `&mut Agent`. Queue edits write that boundary, goal,
and current queue together to `~/.generalist/history/autosave.json` using
flush, atomic rename, and parent-directory flush. `HistoryCheckpoint` replaces
the history boundary only after `history_tool_protocol_is_valid` holds. On
restart, the goal is restored independently; queued work is recovered only
together with that autosaved conversation. Because no turn survives a process
exit, residual steers are normalized to follow-ups. The
`structured_state_does_not_collide_with_legacy_input_history_file` and
`persistence_rejects_an_invalid_tool_protocol_boundary` tests exercise the
filesystem and protocol guards; the state round-trip test covers the optional
goal. `UiAction::QueueChanged` and `Submit` are the only terminal-event effects
that request an autosave; idle local commands are persisted by the controller
after execution. Display-only actions are covered by
`only_queue_mutations_request_terminal_event_persistence`.

## Deliberate abstraction boundaries and residual risks

- `phase`, lifecycle sets, and committed order are model ghost state. Adding a
  mirrored mutable Rust state machine would create a second authority; review
  instead traces control locations and RAII ownership.
- Streaming text/reasoning deltas, reasoning-modal scroll, and Ratatui frames
  are hidden display state. Copy-mode ownership is modeled separately because
  it disables user transitions; rendering beneath that ownership remains
  hidden. Only a complete provider response enters model history. The event
  channel is unbounded, but
  permission, frame, and terminal branches precede display-event draining and
  frames are batched by the 50 ms tick; a pathological provider can consume
  memory but cannot indefinitely starve terminal input outside the user's
  explicit copy-mode pause.
- Provider tool-name validation is an implementation strengthening outside the
  model's payload abstraction. Code mode advertises only `python`; an undeclared
  response is paired with an error for `ToolHistoryIsValid` but is never exposed
  as executable tool activity.
- Cancellation repairs protocol history but cannot roll back external side
  effects. The running tool's synthetic result explicitly says completion is
  unknown.
- Permission/help/queue modals temporarily block ordinary composer keys. This
  is the permitted subset refinement of model-level `Enqueue`, not a claim that
  every key is accepted in every modal.
- Code-mode bridge calls are flattened into the active tool batch. Their
  permission denials are propagated, but their intermediate payloads and
  subprocess/socket mechanics are outside the model.
- Typed local commands, including `/goal edit`, and `/load` mutate session
  state only while idle. They are outside the active-turn model and must retain
  that guard. The active goal is host instruction state, not conversation
  history; `active_goal_is_injected_without_entering_conversation_history`
  checks that boundary.
- Provider reasoning is payload, not control state. The OpenAI-compatible and
  Anthropic adapters normalize only inspectable text, the TUI keeps it out of
  conversation rendering, and redacted/signature material is never rendered.
  `redacted_reasoning_payload_never_reaches_the_inspector` and provider parser
  tests cover that boundary. This adds no TLA+ action or invariant.
- TLC's state space is finite and its fingerprint collision probability is
  nonzero. A green run is model evidence, not a proof of Rust refinement or
  external tool termination.

## 2026-07-26 refinement audit

The implementation and model were traced action by action during the Ratatui
transition. The review found and corrected:

- a start-rollback path that could strand a steer in an idle queue;
- model continuation after denial and incorrect iteration accounting;
- cancellation racing a final permission answer and committing steering;
- code-mode scripts swallowing a nested permission denial;
- refusals checkpointing anomalous tool uses without results;
- split queue/history persistence that could recover a duplicate or lose a
  committed steer;
- structured state using `~/.generalist_history` as a directory even though
  older interactive input history used that path as a regular file, producing
  repeated `Not a directory` persistence failures. Structured state now lives
  under `~/.generalist/history`, and a regression test preserves the legacy
  file across repeated atomic saves;
- scroll and composer events performing an atomic write with both file and
  parent-directory `fsync`, plus an immediate full redraw, for every event.
  Terminal actions now identify queue mutations explicitly, and all ordinary
  display updates are coalesced on the 50 ms frame tick;
- an OpenAI-compatible model emitting the undeclared native name
  `tools.firecrawl_search` even though the request advertised only `python`.
  The pre-existing execution guard prevented the call from running, but the TUI
  presented it as attempted tool activity and generated a malformed
  `tools.tools.firecrawl_search` retry hint. Protocol violations are now rejected
  before tool-start activity and produce an exact bridge expression;
- scroll state measured only as a distance from the newest line, so a supposedly
  paused viewport drifted during streaming and could accumulate overscroll far
  beyond the oldest line. It now uses a clamped absolute viewport with explicit
  follow-latest state, and modal mouse events no longer leak into the transcript.
  A later manual review found that the scroll bound still estimated wrapped
  rows by total character width, while Ratatui wraps at word boundaries. Both
  conversation and permission-detail bounds now use Ratatui's exact
  `WordWrapper` line count, with a regression that scrolls away from and back to
  a final marker. Screenshot review then exposed a separate presentation bug:
  the scrollbar received total wrapped rows even though its position was a
  bounded top-row offset. This left visible track below the thumb at the true
  bottom. It now receives the number of legal offsets and the viewport length;
  the regression checks both the final marker and the bottom thumb cell;
- long queue selections disappearing below a non-stateful list viewport,
  queued-text restore overwriting a composer draft, queue-editor control keys
  inserting literal letters, and same-named saved tool calls displaying swapped
  results. Stable-ID/index-based rendering and focus-aware controls cover those
  cases;
- cancelled provider deltas remaining on screen as an apparently committed
  assistant response. `AssistantStreamAborted` now labels them uncommitted while
  durable history remains unchanged;
- terminal mouse capture preventing native selection/copy, and no inspection
  path for reasoning fields already exposed by providers. F3 now releases mouse
  capture and freezes redraws while the same reactor continues progressing;
  F4 renders provider-supplied reasoning separately, explicitly represents its
  absence, and never renders signatures or redacted payload data. Reasoning is
  a hidden payload refinement. Copy ownership is now explicit in TLA+ because
  the review found that it gates cancellation and permission input and therefore
  changes the liveness assumptions;
- idle and spinner-only ticks rebuilding the complete transcript unnecessarily,
  and ordinary agent-event backlog taking priority over terminal input. Dirty
  frames, a 10 FPS spinner, and reactor branch ordering bound the display work;
- untrusted text reaching terminal cells with raw control bytes, and partial
  startup failures leaving terminal modes enabled. Display-only sanitization
  and a shared cleanup path now cover every rendered source and initialization
  error boundary.

After copy-mode ownership was added, the first TLC run found a liveness
counterexample: repeatedly entering and exiting copy mode could keep a
permission unresolved because permission input was not continuously enabled,
so weak fairness of combined agent progress did not apply. The model now makes
both environmental assumptions explicit: weak fairness for exiting copy mode,
and strong fairness for resolving a permission when its UI is available
infinitely often. TLC then explored 442,870 states (113,190 distinct, depth 27)
with no error under `spec/AsyncRuntime.cfg`. This is the current baseline to
repeat, not a permanent certification or a claim that the program can force a
human response.

The final PTY review rebuilt the binary and ran it against a deliberately
stalled local OpenAI-compatible SSE server. While the first response was
blocked, the composer accepted input, `Enter` and `Tab` created the two
delivery classes, F2 edits and reordering changed the later intercepted
requests, a dedicated deletion run proved deleted text never reached the
server, and PageUp/PageDown changed scroll state while the spinner continued.
The captured request sequence showed the steer in the second request and the
edited follow-up in the third. A separate interruption run left no partial
assistant message, then dispatched the surviving follow-up; a tool-call run
showed a live permission modal, deny-once produced the exactly paired error
result, and the atomic autosaves had valid history and an empty queue. The
first attempt also caught a stale pre-build executable, so the contribution
methodology now requires rebuilding the exact source before PTY review.

A follow-up jank run sent 600 streamed fragments while steering, exercised an
18-item queue, attempted restore over a draft, resized the live TUI to 24×8,
and cancelled after a visible partial response. Steering reached the next
request without display-event starvation; the selected queue tail stayed
visible; restore made no autosave and preserved the draft; the tiny layout
survived; the partial response was visibly uncommitted and absent from the
autosave; and normal exit restored canonical/echo modes, bracketed paste, and
the alternate screen.

The goal-command review rebuilt the exact binary and drove it through a PTY
against a loopback OpenAI-compatible SSE server. It set a goal directly,
opened `/goal edit`, replaced the prefilled value with Ctrl+U, ran `/goal
show`, and intercepted the next provider request. The edited value appeared
only in the system message, every request still advertised exactly `python`,
and no `/goal` command entered conversation history. The atomic autosave held
the edited value; a fresh process with no queued work rendered the recovered
goal, and `/goal clear` persisted `null`. Preparing this run found that generic
prompt modals treated Ctrl+U as a literal `u`; prompt editing now shares the
composer's shell-style replacement controls and has deterministic coverage.

The observability review rebuilt the exact binary again and drove it in an
isolated 100×30 tmux PTY against a two-stage loopback SSE response. F4 displayed
the first `reasoning_content` fragment as live. Entering F3 emitted terminal
mouse-capture disable sequences and drew the paused banner once; while that
frame remained byte-for-byte unchanged, the provider emitted more reasoning
plus the final answer and the controller committed it to the atomic autosave.
Keystrokes sent during copy mode did not enter the composer. Exiting F3 emitted
mouse-capture enable sequences and immediately revealed the accumulated,
completed reasoning, while the answer remained hidden behind the reasoning
modal and appeared only after it closed. A subsequent bracketed Unicode
multiline paste reached the next intercepted user message exactly, and every
intercepted request still advertised only `python`. The no-reasoning response
produced the explicit inspector placeholder. Normal exit disabled mouse capture
and bracketed paste. A final staged response delivered a `python` tool call
while F3 owned the terminal: the frame stayed frozen and no continuation
request occurred until F3 resumed, exposed the permission modal, and allow-once
was answered; only then did the tool result and next provider request appear.
This concretely exercises the permission/copy-mode fairness boundary identified
by TLC. A separate 80×24 PTY regression reconfirmed that the final wrapped
marker and scrollbar thumb reach the real bottom, PageUp pauses, and PageDown
restores follow-latest.
