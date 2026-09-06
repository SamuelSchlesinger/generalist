# Runtime model traceability

This document is the refinement review between the executable Unix TUI and
`spec/AsyncRuntime.tla`, `spec/MemoryRuntime.tla`, and
`spec/ArchiveScopeRuntime.tla`. It must be read with the source. TLC proves
properties of the finite models under their checked-in configurations; the
tables below are the separate, human-reviewed argument that the Rust paths
implement those modeled transitions.

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
selection execute only outside a mutation-capable turn. During MCP discovery,
composer and queue-manager mutations refine the existing idle queue actions;
discovery progress/cancellation is hidden and no dispatch occurs until the
registry is finalized or explicitly skipped. Remembered permission-policy
contents and `/permissions` list/reset/clear are hidden idle state: no live
request exists while they run, so a policy mutation is a TLA+ stutter and only
changes whether a later tool use reaches `AskPermission`. Interactive
allow-always choices for `bash` and `python` are narrower hidden state: they
authorize only the exact JSON input for the current process, are excluded from
saved policy, and are cleared by `/load`, reset, or clear. The stable, bounded
`/tools` list/search/show projection is also hidden idle state: it reads the
finalized registry and effective advertised definitions but cannot request a
completion, permission, or execution, and does not mutate history. `/copy last`
and `/copy all` are hidden, explicit write-only terminal effects; `/copy select`
is also a display-only stutter.
`/mcp status` is a hidden read of typed process-local connection state.
`/mcp retry [server]` performs an explicit external handshake only while the
agent is idle, using the same responsive discovery reactor; it may add
recovered bridge definitions to the existing registry but cannot start a
provider completion, execute a capability, or mutate model history. Tool
definitions and MCP wire state are hidden payloads in this model, so a
successful or failed retry is a stutter; queued composer/F2 work during retry
continues to refine ordinary idle `Enqueue`/queue-edit actions.
The bounded `/history` list/search/show projection is another hidden idle
read: it is hard-bound to the active store's current scope, returns the same
sanitized event shape as archive tools, never loads a session, and cannot
request a provider completion or tool execution. `/history forget` is not
hidden: after a default-Cancel host confirmation, its successful durable unlink
refines `ForgetHistory`. It cannot target the live autosave or another scope.
Cross-scope access remains in the modeled permissioned archive path.
Process-local `/usage show|reset` is also hidden idle state. Provider-attempt
events update a display-only, saturating per-provider/model ledger; missing
usage payloads and optional cache-field coverage remain explicit. Reading or
resetting that ledger cannot enqueue, call a provider/tool, mutate model
history, or write durable state, so both commands refine a TLA+ stutter.
Goal text, active/completed status, conversation-search query/selection,
prompt/message provenance, and other provider payloads are hidden data in this
model. Catalog-backed slash completion is likewise a hidden composer edit:
when Tab is consumed it cannot enqueue, while an unrecognized/full argument
falls through to the existing `Enqueue` path. `Ctrl+F` scans only sanitized
in-memory `ChatEntry` values and computes
one wrapped-row viewport jump; it cannot mutate or disclose model history, the
prompt queue, tool payloads, reasoning, or archived sessions. Setting or
clearing only the objective refines a TLA+ stutter step; scheduling a
host-authored continuation linearizes to `Enqueue`, and the permission-free
completion control linearizes to `CompleteTool` plus a hidden goal-state
mutation. Reasoning-inspector contents and scroll are also hidden display data.
Copy mode is a display-only abstraction. Entry and exit do not change queue,
history, or permission identity. Concrete input handling suppresses interactive
choices until the user resumes, while remembered decisions and host goal
scheduling can proceed. `WF_vars(AgentProgress)` assumes eventual permission
resolution; no model property claims to force user input. Existing copy-mode
Rust/PTY tests check terminal behavior independently.

## Machine-checked implementation traces

`src/model_trace.rs` adds an opt-in, payload-free event sink at existing Rust
linearization points. Ordinary application construction installs no sink. The
events carry stable IDs, counts, categorical scope filters, and trace-local
opaque scope labels. Queue observations also record actual ordered IDs and
delivery modes after an atomic mutation, claim/rollback, or normalization.
Prompt text, archive content,
tool input, provider payloads, project paths, and user-chosen saved-session
names never enter the serialized trace. Saved names are replaced consistently
across save, search, and deletion by trace-local history IDs.

`examples/model_conformance.rs` executes deterministic paths through the real
`PromptQueue`, `Agent`, `ToolRegistry`, `HistoryStore`, and `EpisodicMemory`
implementations. `scripts/render-model-traces.py` normalizes their opaque IDs
into each checked model's finite domain and generates a temporary wrapper for
each original TLA+ module. Every wrapper step requires both the original
module's `Next` relation and the exact named source action with the observed
arguments. Its `TraceCompletes` property requires TLC to consume the complete
Rust-produced sequence; an unrelated model execution cannot satisfy the gate.

`scripts/check-model-conformance.sh`, also available as `make conformance`,
checks independent observed scenarios and targeted invalid traces. The wrapper
configurations inherit every invariant from the original `.cfg` files; finite
identity domains are derived from the trace and `MaxRounds` is supplied by the
Rust scenario's actual configuration. Claimed follow-up IDs and the complete
ordered list of claimed steering IDs are observed from the real queue and
constrain the corresponding model post-state. Empty suites, unknown actions,
missing initial/final queue observations, and missing model coverage are
rejected. Snapshots use the action trace's ID mapping and constrain the modeled
queue and queued delivery modes after the recorded number of actions. Moves
and bulk removals are observed after the complete Rust mutation and its modeled
substeps. The renderer does not reconstruct those snapshots from action names.
TLC must report a temporal
counterexample for an invalid trace, rather than a parse or infrastructure
failure.

Checked async paths include allowed execution, refusal, terminal provider
failure, cancellation before a request and during a permission wait, batch and
outer/nested Python denial, unadvertised native calls, truncated tool batches,
cancellation at a truncated boundary, the iteration cap, steering after an
answer, duplicate-text queue edits, goal-continuation reconciliation, and start
claim rollback. The storage scenario covers paused capture, enabled capture,
forgetting, same-scope writes, and allowed current-scope history/memory search.
Mutations cover early continuation, missing results/repair, lost denial,
replayed permission replies, lost queue deletion/steering commit, forgotten
memory disclosure, and an incorrect scope filter. Three more mutations keep
every action intact and corrupt only observed queue membership, order, or
delivery mode. The action-only mutations run without snapshots so their
rejection still checks the original protocol violation. Three model faults
must violate the named ownership, lifecycle, and scope invariants.

This is sampled safety evidence with partial queue-state observations. Other
components predominantly record actions; missing or misleading instrumentation
can still conceal a defect. Queue text/provenance and the full prompt lifecycle
are not independently observed, and restored queues remain outside the fixture.
The models are checked separately, with no product-model proof linking async,
memory, and archive executions. Copy-mode input ownership, permission-broker
IDs, steering rollback, storage failures, global startup, foreign archive rows,
multiple archive results, concrete provider behavior, and external effects
require their respective tests and review; they are not certified by the
current sample. The trace-local permission IDs distinguish invocations but do
not independently observe the broker's own reply correlation. Memory turn-start
and settlement events are emitted together at capture submission, so those
traces do not observe a live provider/worker overlap. Archive snapshots require
fresh fixtures; arbitrary restored databases are outside this renderer's
initial-state contract.

The permission witness is also concrete now. `ToolRegistry` privately creates
a `ToolAuthorization` only after the policy allows the exact tool name and full
JSON input. An archive tool can narrow that to a `DisclosureGrant`, which binds
the operation, scope filter, query or ID, and expected scope. Public cross-scope
history and memory APIs require that grant and recheck it before opening the
archive. A remembered allow-always policy still mints a new exact grant for
each invocation; it does not create a scope-free storage handle. This is an
in-process capability boundary, not protection from separately authorized
same-UID Python, shell, file access, or direct filesystem access.

## State mapping

| TLA+ state | Authoritative Rust representation | Review note |
| --- | --- | --- |
| `phase` | Control location in `main::drive_started_turn`, `Agent::run_started_turn`, and the permission broker | It is a model program counter, not a second mutable Rust enum that could drift. `AppState::turn_active` is only its TUI input-routing projection; unlike `busy`, it is set precisely around the sole mutation-capable agent future. |
| `activeTurn` | The one prompt synchronously recorded by `Agent::begin_turn` before the controller pins the sole `&mut Agent` future | The current-thread reactor cannot start another turn until that future returns. |
| `queue`, `delivery` | `runtime::PromptQueue` containing `QueuedPrompt { id, text, delivery, source }` | `Rc<RefCell<_>>` is the sole store; `tui::AppState::queue` is a render snapshot only. `PromptSource` is hidden payload, not a second modeled lifecycle. |
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
| `Enqueue` | `tui::submission_delivery` selects steer/follow-up from actual turn ownership; `main::enqueue_submission` forces idle, background-operation, and local-command submissions to follow-up; `PromptQueue::enqueue` assigns user IDs. MCP discovery uses this path before `Agent` construction and queues a complete pre-agent boundary for the autosave worker. After a normally settled active-goal turn, `reconcile_goal_continuation(true)` appends at most one host-authored follow-up with the same fresh-ID transition. | `composer_keys_distinguish_turn_steering_from_background_queueing`, `background_busy_state_does_not_offer_or_create_steering`, `duplicate_text_keeps_distinct_stable_ids`, `goal_continuation_is_unique_removable_and_becomes_user_text_when_edited`, `active_goal_continues_only_after_normal_settlement`; exact-binary stalled-MCP PTY test |
| `DeleteQueued` | Queue modal deletion routes the selected stable ID to `PromptQueue::delete`. Restore is the same modeled removal plus a display-only move into an empty composer; it is refused over an existing draft. | `queue_manager_mutations_address_stable_ids`, `restoring_queue_text_never_overwrites_a_draft` |
| `ReclassifyQueued` | Queue modal `s` calls `PromptQueue::toggle_delivery` with `turn_active`, not presentation-level `busy`; idle and background work cannot create a steer. | `queue_manager_mutations_address_stable_ids`, `background_busy_state_does_not_offer_or_create_steering`, `ending_turn_normalizes_undelivered_steers` |
| `MoveQueuedEarlier` | Queue modal Ctrl+Up/Down calls `PromptQueue::move_by` by ID. | `queue_manager_mutations_address_stable_ids` |
| `DispatchFollowUp` | The idle outer loop calls `claim_follow_up` once; it never scans past a non-follow-up head. | `followups_dispatch_one_at_a_time_in_fifo_order` |
| `CommitStart` | Without an await, the controller calls `Agent::begin_turn`, commits the `PromptClaim`, and queues one complete snapshot containing the started history plus remaining queue. The dedicated writer performs the later atomic file replacement. | `dropped_claim_rolls_back_and_commit_removes`, `autosave_flush_persists_the_latest_queued_snapshot`; source-order review in `main` |
| `RequeueStart` | Dropping an uncommitted `PromptClaim` restores the same ID at the front. Production start is currently infallible between claim and commit; the action over-approximates unwinding and future fallible setup. | `dropped_claim_rolls_back_and_commit_removes` |
| `ProviderAnswer` | `complete_with_retry` commits a complete assistant response only after host byte/block/tool validation, then reaches the no-tools boundary. Text and provider-supplied reasoning are separate display deltas; aborted streams receive separate display-only uncommitted markers. Committed Ratatui projections are bounded without changing history. | observed tool-to-answer trace; `steering_queued_during_final_response_gets_another_model_call`, `provider_cancellation_commits_no_partial_assistant_message`, `custom_provider_oversized_final_response_never_enters_history`, `custom_stream_stops_on_host_callback_limit_and_marks_preview_uncommitted`, `ignored_callback_limit_error_still_prevents_custom_provider_commit`, `committed_chat_and_reasoning_use_bounded_exact_projections` |
| `ProviderRefusal` | Refusal pairs any anomalous tool uses with synthetic errors, checkpoints, and returns without accepting steering. | observed refusal trace; `refusal_with_tool_uses_is_repaired_before_checkpointing` |
| `ProviderFailure` | A non-retryable error or exhausted retry sequence commits no partial assistant response and returns `Err`; the controller preserves prior valid history, normalizes queued steers to follow-ups, and returns idle without consuming them. | observed non-retryable-failure trace; `non_retryable_errors_surface_immediately`, `history_survives_api_errors_after_tool_execution` |
| `ProviderToolBatch` | A host-bounded committed assistant response is scanned into a finite `tool_uses` vector; one loop iteration is consumed. A burst over the completion limit is rejected before any tool-start event or execution. | observed tool-to-answer trace; `tool_results_are_truncated_in_history`, `iteration_limit_leaves_late_steering_for_controller_normalization`, `oversized_tool_burst_is_rejected_before_execution_or_commit` |
| `CompleteTool` | Tools run sequentially; each outer outcome becomes one `ToolResult`, including denied calls and synthetic results for calls skipped after denial or rejected as undeclared. Truncation and unknown tools also return structured results. In code mode, an undeclared native call is never started or permission-checked and receives a synthetic error result. The reserved `update_goal` host control validates exact completion input, clears hidden goal state without capability permission, and produces a normal paired result at this same boundary. | `tool_results_are_truncated_in_history`, `history_survives_api_errors_after_tool_execution`, `code_mode_rejects_unadvertised_direct_tool_calls`, `update_goal_completes_without_capability_permission`, `invalid_goal_completion_keeps_the_goal_active` |
| `AskPermission` | `PermissionBrokerPrompt::choose` allocates a monotonic request ID, sends one `PermissionRequest`, and awaits its one-shot. | `broker_correlates_the_ui_reply_with_its_request` |
| `AllowPermission` | The reactor sends the choice only when the modal ID equals the live request ID. For ordinary tools, `AllowAlways` records a persisted name-level allow and removes an opposite deny. For `bash` and `python`, the interactive choice records only a session-scoped exact-input grant; it never enters `SavedState`, and loading a session clears it. Deliberately configured or loaded name-level policy remains authoritative. | `broker_request_ids_keep_out_of_order_replies_correlated`, `interactive_always_allow_persists_and_undenies_ordinary_tools`, `interactive_always_allow_on_exec_tools_covers_only_the_exact_input`, `replacing_loaded_policy_clears_session_exact_input_grants` |
| `DenyPermission` | A denial retires the permission request and marks the turn denied; `CompleteTool` separately accounts for the outer result, including a Python call containing multiple denied bridge operations. `DenyAlways` removes an opposite remembered allow, and contradictory legacy/shared state is checked deny-first. Denials inside code-mode bridge calls propagate through `ScriptResult::denied` even if Python exits successfully. | `contradictory_remembered_permission_is_loaded_as_deny`, `contradictory_shared_policy_denies_fail_closed`, `dropped_broker_reply_denies_instead_of_hanging`, `denial_inside_code_mode_pauses_the_outer_turn` |
| `ClaimSteering` | At a history-valid boundary, `PromptQueue::claim_steering` removes all visible steers, preserves their relative order, and leaves follow-ups. | `steering_claim_preserves_relative_order_and_followups` |
| `CommitSteering` | `Agent::commit_steering` appends claimed text to the valid user boundary, commits IDs, emits `SteeringCommitted`, and checkpoints with no await between those operations. | `steering_queued_during_final_response_gets_another_model_call` |
| `RequeueSteering` | Dropping an uncommitted steering claim restores the same IDs at the front. Normal commit contains no fallible/await boundary. | `dropped_steering_claim_restores_the_same_ids_at_the_front` |
| `ContinueAfterTools` | A complete, non-denied tool-result batch with capacity remaining loops to the next provider request. | `tool_results_are_truncated_in_history`, `history_survives_api_errors_after_tool_execution` |
| `SettleTurn` | Answer, refusal, provider failure, denial, or iteration cap releases `&mut Agent`; the controller converts remaining steers to follow-ups, reconciles automatic goal work, queues one complete autosave snapshot, and then offers protocol-valid settled history to the independent memory FIFO. That is source order between two queues, not a disk-before-memory guarantee: the autosave worker may coalesce snapshots and performs the atomic file replacement when scheduled. Completed/capped active goals linearize to `SettleTurn` followed by modeled `Enqueue`; interruption, refusal, denial, error, or exit removes automatic entries and pauses without clearing the goal. | observed refusal/failure traces; `refusal_with_tool_uses_is_repaired_before_checkpointing`, `history_survives_api_errors_after_tool_execution`, `denial_inside_code_mode_pauses_the_outer_turn`, `iteration_limit_leaves_late_steering_for_controller_normalization`, `active_goal_continues_only_after_normal_settlement`, `episodes_omit_reasoning_and_tool_payloads` |
| `RequestCancel` | Esc/Ctrl+C retires a live permission with deny-once, sets the turn-scoped watch flag, and keeps polling the controlled future until repair completes. | observed pre-provider cancellation trace; `cancellation_wins_over_a_ready_permission_before_steering`, `provider_cancellation_commits_no_partial_assistant_message` |
| `RepairCancelledTool` | Cancellation drops the running tool future and emits synthetic error results for it and every unstarted tool use. | `interruption_pairs_the_running_and_unstarted_tool_uses`, `history_tool_protocol_is_valid` debug assertion |
| `FinishCancellation` | After results are appended, the agent checkpoints and returns `Interrupted`; the controller retires nested TUI activity and normalizes undelivered steers. | observed pre-provider cancellation trace; `interruption_pairs_the_running_and_unstarted_tool_uses`, `interrupted_turn_retires_all_nested_activity`, `ending_turn_normalizes_undelivered_steers` |
| `IdleWait` | With no follow-up, the outer `tokio::select!` continues polling terminal input, stale permission events, and frame ticks without owning `Agent` mutably. The pre-agent MCP reactor is a concrete strengthening: it permits the same idle queue transitions while withholding dispatch until registry ownership is settled. | `composer_keys_distinguish_turn_steering_from_background_queueing`, startup-recovery tests, source-order review in `main`; exact-binary stalled-MCP PTY test |

## Property mapping

| TLA+ property | Rust enforcement and review evidence |
| --- | --- |
| `TypeOK` | Rust enums and ownership constrain concrete values; TLC separately checks every modeled variable over the configured state space. |
| `QueueIdentity` | `PromptQueue` owns one vector, saved duplicate IDs are filtered, all mutations address IDs, and claims remove before returning. Covered by the runtime queue tests. |
| `SingleTurnOwnership` | A current-thread reactor pins one future borrowing `&mut Agent`; the outer loop cannot dispatch again until it returns. No `Arc<Mutex<Agent>>` or background runtime exists. |
| `DeliveryIsWellFormed` | `DeliveryMode` has only two variants; idle/background submissions and host goal continuations are follow-ups, only `turn_active` permits creating a steer, and all residual steers are normalized when ownership ends. `PromptSource` does not alter delivery semantics. |
| `SafeSteeringBoundary` | `commit_steering` is called only after a complete assistant answer or after the full result vector is appended. It is not called on refusal, cancellation, or an exhausted iteration budget. |
| `TerminalReasonIsWellFormed` | Distinct Rust branches implement final answer, refusal, terminal provider error, structured denial, and cap. Error/refusal/cap cannot consume steering. The cancellation/permission race test prevents a cancelled turn from taking the denial-to-steer branch. |
| `ToolHistoryIsValid` | `history_tool_protocol_is_valid` checks exact adjacent ID sets; every emitted `HistoryCheckpoint` debug-asserts it, every persistence path refuses an invalid history, load rejects invalid saves, and cancellation/refusal tests inspect checkpoint histories. |
| `PermissionIsCorrelated` | IDs are monotonic, choices travel over the request's own one-shot, mismatched modal IDs are ignored, and dropping a reply denies once. Remembered policy is concrete hidden state: persisted name-level snapshots are disjoint, reset/clear runs only without a live request, and deny wins any contradictory loaded/shared entry. Session exact-input allows never enter a snapshot and are cleared on load, reset, or clear. |
| `SettledPromptsAreCommitted` | The controller records the initial user message before `PromptClaim::commit`; interrupted and ordinary outcomes therefore refer to a committed follow-up. |
| `HistoryOrderHasStableIds` | Queue claims and `SteeringCommitted` preserve stable-ID order. `committedOrder` is a model ghost variable, verified through claim/event tests rather than stored in model-visible text. |
| `EveryBusyPeriodSettles` | TLC checks settlement for the finite bounds under weakly fair agent progress, including eventual permission resolution. Rust bounds provider rounds/retries and keeps polling them during copy mode; permission input waits for resume. External tools/providers, a user who never resumes, or a user who never answers remain environmental blockers. |

## Episodic-memory model mapping

`MemoryRuntime.tla` models only the prototype that exists: opt-in capture of
settled turns in explicit scopes, a FIFO SQLite worker, immutable live rows,
explicit live-store deletion, and permission-gated explicit archive
disclosure. It does not model candidates, automatic prompt retrieval,
consolidation, cross-agent authorization, backup erasure, or the broader
research architecture. Its FIFO is one Generalist process's worker channel.
Simultaneous processes share SQLite rows/settings and lock arbitration but not
that channel; their cross-process ordering is outside this model.

The async and memory models intentionally have different clocks.
`AsyncRuntime.SettleTurn` returns the conversation controller to idle after it
enqueues a memory request; `MemoryRuntime.pendingEpisodes` may still be
draining. The controller can start another turn while the sole worker performs
SQLite I/O. All capture failures are display-only `MemoryEvent`s and never
alter conversation history.

### Memory state mapping

| TLA+ state | Authoritative Rust representation | Review note |
| --- | --- | --- |
| `captureEnabled` | The current handle's `memory_settings.capture_enabled` row, read by `MemoryWorker::record` in FIFO order | Each project or global handle owns one immutable scope/key and starts at `0`; only explicit `/memory resume` opts that handle in. Typed v1 scope keys live in `scoped-episodes.sqlite3`; the pre-scope database is ignored. |
| `activeEpisode` | Ghost identity for the prompt owned between `Agent::begin_turn` and the terminal `TurnOutcome` | Rust allocates the UUID when building the settled record; because no draft is stored or observable, late allocation refines the ghost choice. |
| `activeScope`, `episodeScope` | The immutable `WorkspaceScope` shared by `HistoryStore`, `EpisodicMemory`, and the settled `Episode.project_root` label | This lifecycle model follows one current-scope worker. Normal startup discovers the canonical project; `--global` is the explicit global selector covered jointly with `ArchiveScopeRuntime`. An episode cannot move scopes. |
| `pendingEpisodes` | `std::sync::mpsc::Receiver<Request>` owned by the named `generalist-memory` thread | `enqueue_settled_turn` performs a non-awaiting send. A later `flush` is a FIFO barrier. |
| `settledEpisodes` | Ghost set of protocol-valid outcomes offered to the worker | `drive_started_turn` checks `history_tool_protocol_is_valid` before enqueueing; malformed history produces a visible error and no request. |
| `liveEpisodes` | Rows in `episodes`, each carrying its byte-valued scope key | Local commands bind the worker-owned current key. Permissioned search uses one SQL predicate for `current`, `global`, `other_projects`, or `all` before text/ID matching. |
| `skippedEpisodes` | A successful `Record` request returning no ID while capture is paused | Skips are intentionally not persisted; the set is proof-history state. |
| `failedEpisodes` | A failed atomic insert plus `MemoryEvent::CaptureFailed` | No row is made live. The set is proof-history state, not a retry queue. |
| `forgottenEpisodes` | Successful live-row deletion observed during the current model execution | This is ghost state only. The implementation explicitly makes no non-resurrection claim across prior exports, restored backups, or filesystem snapshots. |
| `pendingSearch` | One `search_memories` or `read_memory` registry call awaiting a permission-policy decision | An interactive decision uses the correlated broker; a remembered allow/deny is automatic but still a registry policy decision. Denial returns a structured result and discloses no archive row. |
| `authorizedByFilter`, `disclosedEpisodes` | Sanitized results returned through a registry-minted `DisclosureGrant` for the exact tool call | The grant binds the operation, complete input, and parsed filter. Results can enter code-mode computation/conversation only after an allow-once, allow-always, or remembered allow-always transition. They are never appended to the system prompt or used as instruction state automatically. |

`Agent::history_revision` is a hidden payload-boundary guard, not modeled
memory lifecycle state. Appends preserve the recorded turn-start index;
replacement, clearing, and compaction increment the revision. A changed
revision therefore selects `prompt_only` instead of risking capture from a
coincidentally identical later user message.

### Memory action mapping

| TLA+ action | Concrete Rust transition | Deterministic evidence |
| --- | --- | --- |
| `StartTurn` | The idle controller commits one follow-up with `Agent::begin_turn`; memory has no persisted or retrievable draft. `PromptSource` does not change the modeled lifecycle. | existing prompt-claim tests; source-order review in `main` |
| `SettleTurn` | After the ordinary runtime settles and queues its complete autosave snapshot, `drive_started_turn` maps `TurnOutcome` (or error) to `EpisodeOutcome` and sends the history tail anchored immediately before `begin_turn`. The autosave worker need not have committed that snapshot before memory capture begins. Duplicate steering text cannot move the boundary. Host-authored goal continuations are filtered rather than retained as user text. If `history_revision` shows that in-turn compaction relocated the boundary, capture degrades to the original prompt and labels the record `prompt_only`. | `compaction_summarizes_old_history_and_preserves_recent`, `episodes_omit_reasoning_and_tool_payloads`, `host_goal_continuations_are_not_retained_as_user_authored_text`, `duplicate_steering_text_does_not_move_the_episode_boundary`, `a_relocated_history_boundary_degrades_to_prompt_only` |
| `RecordEpisode` | `MemoryWorker::record` rechecks the project capture setting and performs one immutable-row `INSERT`; SQLite atomicity chooses a whole row or no row. | `episodes_are_immutable_but_can_be_forgotten_from_the_live_store` |
| `SkipEpisode` | The worker observes disabled capture and returns `Ok(None)` without executing an insert. | `capture_is_paused_by_default` |
| `FailEpisode` | Any setting/serialization/SQLite error returns no live ID; asynchronous captures emit `MemoryEvent::CaptureFailed`. | schema/immutability tests plus explicit error branch review |
| `PauseCapture` | `/memory pause` queues a `SetCapture(false)` request and awaits its one-row settings update while the TUI keeps polling. | parser tests; `capture_and_setting_changes_observe_fifo_order` |
| `ResumeCapture` | `/memory resume` queues `SetCapture(true)`; prior capture requests on the same channel complete first. | parser tests; `capture_is_paused_by_default`, `capture_and_setting_changes_observe_fifo_order` |
| `ForgetEpisode` | `/memory forget <id>` resolves a unique current-scope prefix and deletes exactly that row. It then attempts a truncating WAL checkpoint and distinguishes completed from still-pending truncation without misreporting the committed delete as a failure. | `project_handles_cannot_search_or_delete_each_others_episodes`, `episodes_are_immutable_but_can_be_forgotten_from_the_live_store` |
| `RequestSearch` | The model calls `search_memories`/`read_memory` with an explicit scope selector; `ToolRegistry` submits that exact input to the permission policy. Only an allow decision creates the exact `ToolAuthorization` from which the tool derives its `DisclosureGrant`. | `archive_tools_run_through_the_registry_permission_gate`, `disclosure_grants_are_bound_to_the_exact_authorized_call` |
| `DenySearch` | `MemoryPermissionHandler` returns deny and `ToolRegistry` does not call the SQLite-backed tool. | `archive_tools_run_through_the_registry_permission_gate` |
| `ApproveSearch` | After an interactive or remembered allow, `search_scoped` or `show_scoped` requires and rechecks the exact `DisclosureGrant`, then runs on the sole memory worker. SQL applies the selected scope predicate before content/ID matching and rechecks it on the final row read; the read tool also checks the expected returned scope label. Long text is exposed in bounded, resumable pages. | `global_scope_is_explicit_and_cross_scope_search_is_bounded_by_filter`, `tools_require_explicit_scope_and_return_scope_labels`, `disclosure_grants_are_bound_to_the_exact_authorized_call`, `conversation_reads_are_bounded_and_resumably_paginated` |

Local status, search, show, and export are read-only TLA+ stutter steps. Their
concrete operations still run on the worker; `drive_memory_command` continues
polling terminal input, queue edits, stale permissions, memory events, and
frame ticks while awaiting the reply. Tool inputs/results and all
`Thinking`/`RedactedThinking` payloads are structurally omitted before the
record request is sent. A message with host goal-continuation provenance is
likewise omitted before storage rather than misclassified as user-authored
memory; matching text without that provenance is retained normally. This
smallest slice derives tool metadata from committed history, so code mode
records the outer `python` use/result rather than nested bridge activity.

### Memory property mapping

| TLA+ property | Rust enforcement and review evidence |
| --- | --- |
| `TypeOK` | Rust enums define outcomes/events/requests; SQLite `STRICT` tables and checks constrain persistent scalar fields. |
| `EpisodeIdentity` | UUIDs identify records, the primary key rejects reuse, and the FIFO processes each `Record` request once. |
| `SettledLifecycleIsTotal` | A worker request returns inserted, skipped, or failed; a live record can later be forgotten. No partial state is returned as an episode. |
| `EpisodeLifecycleIsDisjoint` | One insert creates live state; skip/failure create none; deletion removes the live row. The corresponding model history sets cannot overlap. |
| `NoAutomaticRetrieval` | The only modeled disclosure set equals records authorized by an explicit permission transition. Rust has no automatic retrieval hook: `Agent` and provider prompt construction accept no episode bundle, while archive reads exist only as ordinary permission-gated tools. |
| `SearchDisclosureIsScoped` | Each approved result is constrained by the requested filter before disclosure. SQL tests exercise project/global isolation and all four filter interpretations; the read path checks the expected scope label. |
| `EveryPendingEpisodeResolves` | TLC assumes weak fairness for the enabled FIFO-head processor. Concretely, a dedicated OS thread drains requests in the order exercised by `capture_and_setting_changes_observe_fifo_order`, and SQLite has a two-second busy timeout; an indefinitely blocked kernel/filesystem or terminated process remains outside the liveness claim. |

## Archive-scope model mapping

`ArchiveScopeRuntime.tla` isolates the routing contract shared by conversation
history and episodic memory from the more detailed FIFO lifecycle. It models
one immutable startup scope, explicit global selection, same-scope writes, and
permissioned representative disclosures from current/global/other/all filters.
A representative record abstracts one member of a bounded result batch; the
Rust SQL and catalog tests separately check every returned member.

### Archive-scope state mapping

| TLA+ state | Authoritative Rust representation | Review note |
| --- | --- | --- |
| `activeScope`, `globalWasExplicit` | One `WorkspaceScope` selected in `main` before either store or the model context is built | `WorkspaceScope` has no default. Project discovery cannot return global, unscoped state does not deserialize, and only `--global` or the explicitly named library constructor selects global. |
| `histories`, `memories` | Live scoped state files under manifest-bearing scope directories and scope-keyed rows in `scoped-episodes.sqlite3` | These sets include archives written by prior runs and remove a named history after durable deletion. The model hides their text and disk representation. |
| `writtenHistories`, `capturedMemories` | Files/rows created through the current `HistoryStore` and `MemoryWorker` | Both clients own an immutable scope and reject mismatched payload labels. |
| `pendingKind`, `pendingFilter` | The archive tool name plus required `scope` input owned by one `ToolRegistry::execute_tool` invocation | Interactive policy checks additionally have a monotonic broker request ID and one-shot; remembered policy decisions do not open a modal. |
| `disclosedHistory`, `historyDisclosureFilter` | The representative conversation result and categorical filter returned from an allowed search/read | Conversation manifests are filtered before any state file in an unselected scope is opened. Reads repeat both the filter and exact returned scope label. |
| `authorizedHistoryDisclosure` | Ghost witness refined by the registry-minted `DisclosureGrant` that produced the last representative history disclosure | The concrete grant binds operation, filter, and complete input. The model records only the last witness to keep finite CI practical; it is not a persistent permission set. |
| `disclosedMemory`, `memoryDisclosureFilter` | The representative episode result and categorical filter returned from an allowed search/read | SQL applies the filter before text/ID matching and on the final row fetch. |
| `authorizedMemoryDisclosure` | Ghost witness refined by the registry-minted `DisclosureGrant` that produced the last representative memory disclosure | Allow-always state remains concrete Rust policy state outside this model; each automatic allow still mints a new exact grant and refines another approve transition. |

### Archive-scope action mapping

| TLA+ action | Concrete Rust transition | Deterministic evidence |
| --- | --- | --- |
| `SelectProjectScope` | Default startup calls `WorkspaceScope::discover`, choosing the nearest canonical Git root or canonical working directory. The resulting value is passed unchanged to `HistoryStore`, `EpisodicMemory`, tool construction, saved state, and the model system context. | `project_discovery_uses_the_nearest_git_root`, `global_is_explicit_and_storage_keys_are_stable_and_distinct` |
| `SelectGlobalScope` | Only CLI `--global` or the explicitly named library constructor creates `WorkspaceScope::Global`; there is no `Default` implementation and unscoped state is rejected. Global startup also omits project-local `AGENTS.md`/`CLAUDE.md`. Pre-scope flat files are not searched. | `unscoped_state_is_rejected_instead_of_becoming_global` plus source-order review in `main` |
| `SaveHistory` | Autosave checkpoints, queue edits, and compaction queue complete snapshots to one dedicated writer, which may coalesce pending snapshots so the newest dominates; every actual replacement still routes through the active `HistoryStore` and is atomic. `/save` routes directly through the same scoped store. A fresh manual name uses atomic no-clobber creation; concurrent creators have exactly one winner. Replacing a valid named save requires a separate default-Cancel host confirmation, while invalid/symlinked targets and manual `autosave` are refused. Every path rejects state claiming another scope and writes beneath the deterministic scope directory. | `autosave_flush_persists_the_latest_queued_snapshot`, `project_autosaves_are_isolated_and_global_does_not_fallback`, `save_rejects_scope_mismatch_and_path_traversal`, `save_if_absent_never_clobbers_an_existing_named_session`, `concurrent_save_if_absent_has_exactly_one_winner`, `persistence_rejects_an_invalid_tool_protocol_boundary`, exact-binary saved-session lifecycle probe |
| `ForgetHistory` | `/history forget <name>` verifies a current-scope saved state, requires an explicit host confirmation with Cancel selected by default, rejects `autosave`, removes only that regular file, and syncs the scope directory before reporting success. Historical `writtenHistories` evidence remains monotone, and saving the same name later is allowed. | `forgetting_is_current_scope_only_durable_and_allows_resave`, `forgetting_rejects_reserved_traversal_and_symlinked_targets`, `successful_forget_is_recorded_in_the_archive_model_trace`, exact-binary saved-session lifecycle probe |
| host history inspection (hidden stutter) | `/history list|search|show` calls only `HistoryStore::list`, `search_current_archives`, or `inspect_current_archive`; each is bound to the handle's current scope and returns sanitized metadata/events without replacing `Agent` history. | `history_commands_preserve_names_and_require_explicit_arguments`, `host_inspection_is_current_scope_only_sanitized_and_non_loading`, `history_catalog_is_sorted_searchable_sanitized_and_non_mutating` |
| `CaptureMemory` | `EpisodicMemory::build_episode` stamps the handle's immutable scope label; `MemoryWorker::record` rejects another label and inserts with the worker-owned scope key. | `project_handles_cannot_search_or_delete_each_others_episodes`, `global_scope_is_explicit_and_cross_scope_search_is_bounded_by_filter` |
| `RequestSearch` | Each archive tool requires a non-empty query/ID and categorical scope selector; each read also repeats the exact scope label from search. `ToolRegistry` submits that input to the ordinary permission policy and privately constructs `ToolAuthorization` only after allow. | `archive_tools_run_through_the_registry_permission_gate`, `tools_require_explicit_scope_and_return_scope_labels`, `disclosure_grants_are_bound_to_the_exact_authorized_call` |
| `DenySearch` | A denied registry decision returns a structured denied tool result without calling the history catalog or SQLite worker. | `archive_tools_run_through_the_registry_permission_gate` |
| `ApproveEmptySearch` | An allowed search with no match returns an empty JSON result without inventing a record or falling back to another scope. | project/global isolation tests in `history` and `memory` |
| `ApproveHistorySearch` | After an interactive or remembered allow, the archive tool derives a `DisclosureGrant`; `HistoryStore` requires and rechecks it before filtering scope manifests and opening conversation files. `read_archive` resolves only an opaque ID inside that same filter and requires the exact expected scope label; outputs omit reasoning/tool payloads and paginate long text. | `archive_search_requires_an_explicit_scope_filter_and_reads_by_opaque_id`, `archived_conversations_omit_reasoning_and_tool_payloads`, `conversation_search_and_read_are_sanitized`, `disclosure_grants_are_bound_to_the_exact_authorized_call`, `conversation_reads_are_bounded_and_resumably_paginated` |
| `ApproveMemorySearch` | After an interactive or remembered allow, the archive tool derives a `DisclosureGrant`; the worker API requires and rechecks it before SQL applies the selected scope predicate to text/ID matching and the final fetch. `ReadMemoryTool` requires a full returned UUID and exact expected scope label. | `global_scope_is_explicit_and_cross_scope_search_is_bounded_by_filter`, `tools_require_explicit_scope_and_return_scope_labels`, `disclosure_grants_are_bound_to_the_exact_authorized_call` |

### Archive-scope property mapping

| TLA+ property | Rust enforcement and review evidence |
| --- | --- |
| `TypeOK` | Rust enums constrain scope/filter values, typed store clients own their scope, and TLC checks every modeled routing/authorization witness value in the finite configuration. |
| `GlobalScopeIsExplicit` | `WorkspaceScope::discover` returns only `Project`; `WorkspaceScope` has no default; unscoped persisted state is rejected; `Global` is constructed from parsed `--global` or an explicitly named library API. |
| `WritesStayInActiveScope` | Both storage clients own immutable scope values. History rejects mismatched `SavedState.scope`; memory rejects mismatched episode labels and binds inserts to its key. |
| `PermissionGatesDisclosure` | Archive capabilities are ordinary registered tools under `MemoryPermissionHandler`; only the registry can mint `ToolAuthorization`, and public cross-scope store APIs require the narrower exact-input `DisclosureGrant`. The model keeps a last-disclosure authorization witness. Allow-always can approve later calls without another modal, but every call still passes the policy and receives a new exact grant. Direct host `/memory` commands are explicit user actions, not model-initiated retrieval. |
| `DisclosureMatchesRequestedScope` | One shared `ScopeFilter` defines current/global/other/all semantics. History filters before content matching; memory embeds the same logic in SQL before content/ID matching; reads check returned scope labels. |
| `PendingSearchIsCorrelated` | One registry invocation owns one kind/filter pair. For interactive decisions, the broker additionally correlates a monotonic ID and one-shot response; stale/mismatched replies cannot authorize a different archive call. |

## Durable-boundary refinement

Disk state is an implementation strengthening, not a TLA+ variable. The
controller keeps a clone of the latest history-valid boundary and active goal
while the agent future owns `&mut Agent`. Every checkpoint or queue mutation
constructs a complete snapshot of that boundary, goal, permission policy, and
current queue and sends it to one dedicated autosave writer. If the writer
falls behind, it coalesces pending snapshots so only the newest complete state
is written. Each actual write beneath
`~/.generalist/history/scopes/<scope-id>/autosave.json` still uses file flush,
atomic rename, and parent-directory flush. `HistoryCheckpoint` replaces the
authoritative controller boundary and its corresponding goal only after
`history_tool_protocol_is_valid` holds, then queues a snapshot; the event is not
a claim that disk has already advanced. Carrying history and goal in one event
prevents the separate display notification for `GoalCompleted` from racing the
snapshot contents. Orderly shutdown flushes the writer and reports any write
failure before exit. On restart, the goal is restored independently from the
same scope and its next continuation is reconciled; queued work is recovered
only together with that autosaved conversation. Because no turn survives a
process exit, residual steers are normalized to follow-ups. The
`structured_state_does_not_collide_with_legacy_input_history_file` and
`persistence_rejects_an_invalid_tool_protocol_boundary` tests exercise the
filesystem and protocol guards; the state round-trip test covers the mandatory
scope, optional goal, and queued prompt provenance;
`autosave_flush_persists_the_latest_queued_snapshot` covers the async handoff.
`UiAction::QueueChanged` and `Submit` are the only terminal-event effects that
request an autosave; idle local commands and host goal scheduling are persisted
by the controller after execution. Display-only actions are covered by
`only_queue_mutations_request_terminal_event_persistence`.

## Deliberate abstraction boundaries and residual risks

- `phase`, lifecycle sets, and committed order are model ghost state. Adding a
  mirrored mutable Rust state machine would create a second authority; review
  instead traces control locations and RAII ownership.
- A process crash after the controller queues a snapshot but before the writer
  replaces the file can lose that newest accepted boundary. The prior autosave
  remains atomic and protocol-valid; graceful exit closes the window with a
  checked flush. The TLA+ models neither this disk queue nor crash recovery.
- Streaming text/reasoning deltas, reasoning-modal scroll, and Ratatui frames
  are hidden display state. Copy-mode ownership and its user-input suppression are also
  hidden; automatic permission decisions and host queue actions may progress
  while rendering is paused. Only a complete provider response enters model history. The concrete
  display pump coalesces consecutive text/reasoning fragments into one record,
  caps their combined pending preview at 16 KiB, reports omitted byte counts,
  and reconciles a successful attempt from a post-commit `StreamCommitted`
  snapshot. Each committed chat/reasoning projection is capped at 64 Ki
  characters while full accepted content remains in history and explicit copy
  paths. Permission, frame, and terminal branches precede display draining, and
  frames are batched by the 50 ms tick. Lossless structural events and
  checkpoints are not globally capacity-limited; adversarial structural volume
  remains outside the display abstraction.
- Provider tool-name validation is an implementation strengthening outside the
  model's payload abstraction. Code mode advertises only `python` as a
  capability plus the reserved `update_goal` control while a goal is active; any
  other undeclared response is paired with an error for `ToolHistoryIsValid`
  but is never exposed as executable tool activity.
- `NoAutomaticRetrieval` is not by itself a same-UID filesystem sandbox. Normal
  CLI `python`, `bash`, `patch_file`, and Z3 subprocesses now run in the
  project-scoped OS sandbox, but approved
  host file/archive tools, configured MCP servers, `--yolo`, and unconfigured
  library registries remain outside that subprocess boundary. Hard
  confidentiality from those trusted-host components still requires a VM,
  container, or dedicated user, not this model.
- The macOS Seatbelt policy remains inherited if a child uses `posix_spawn` to
  enter another process group, but the host's process-group cleanup can no
  longer identify that child after reparenting. Denying `posix_spawn` also
  denies common runtime launchers, so guaranteed adversarial resource cleanup
  remains outside this defense-in-depth boundary.
- Subprocess ownership is RAII-based: timeout, cancellation, bridge failure,
  normal leader exit, and dropped tool futures kill the complete process group.
  Python bridge frames and serialization are bounded, and a two-second host
  write deadline prevents a script from wedging the bridge by refusing a
  response. Every restricted child disables core dumps and has a hard
  1,024-descriptor ceiling. These are per-call controls, not aggregate
  memory/CPU/process/disk quotas.
- The memory request channel and live episode store have no quota in this
  experiment. One request is enqueued per settled turn and ordinary SQLite lock
  contention is bounded by the busy timeout, but a hung filesystem or sustained
  production faster than the worker can drain can grow process/disk usage.
- Cancellation repairs protocol history but cannot roll back external side
  effects. The running tool's synthetic result explicitly says completion is
  unknown.
- Permission/help/queue modals temporarily block ordinary composer keys. This
  is the permitted subset refinement of model-level `Enqueue`, not a claim that
  every key is accepted in every modal.
- Code-mode bridge permissions occur within the outstanding outer Python call.
  Only its outer result increments the modeled tool-result count. Their
  permission denials are propagated, but their intermediate payloads and
  subprocess/socket mechanics are outside the model. Archive reads return
  bounded, resumable transcript pages so the bridge does not receive one
  unbounded result; building/searching the local archive still has no storage
  quota.
- MCP connection state, schemas, per-server progress, and discovery cancellation
  are hidden startup data. Servers are attempted in deterministic name order;
  Esc retains only the completed prefix and drops the in-progress transport.
  The model sees queue edits accepted during that interval as ordinary idle
  transitions. Exact autosave checks and the pre-agent `DurableBoundary` are
  concrete refinement evidence; the model does not prove transport cleanup or
  successful connection to an external server.
- Typed local commands, including `/goal edit`, and `/load` mutate session
  state only while idle. They are outside the active-turn model and must retain
  that guard. The active objective is host instruction state, not conversation
  history; `active_goal_is_injected_without_entering_conversation_history`
  checks that boundary. The short host continuation is conversation protocol
  rather than user-authored text: it carries `PromptSource::GoalContinuation`
  while queued and `MessageOrigin::GoalContinuation` after dispatch, renders as
  info, and is omitted from episodic user text. Matching text without that
  provenance remains ordinary user content.
- Provider reasoning is payload, not control state. The OpenAI-compatible and
  Anthropic adapters normalize only inspectable text, the TUI keeps it out of
  conversation rendering, and redacted/signature material is never rendered.
  `redacted_reasoning_payload_never_reaches_the_inspector` and provider parser
  tests cover that boundary. This adds no TLA+ action or invariant.
- TLC's state space is finite and its fingerprint collision probability is
  nonzero. A green model run plus green sampled implementation traces is
  stronger evidence than either alone, not a proof of complete Rust refinement
  or external tool termination.

## Current audit and maintenance contract

The live matrices above replace the accumulated historical audit narratives;
prior versions remain in Git history. New changes should update the applicable
row and the executable scenario rather than append another dated architecture.
Review is driven by changes to ownership, ordering, authorization, cancellation,
or capture/deletion rules. New model state needs a consequential property, a
fault the check must reject, and a concrete implementation observation. Rendering
and other payload/platform behavior retain their applicable Rust and PTY checks.

The September 2026 review expanded the original three trace wrappers into
independent scenarios. Running those scenarios against the old actions exposed
four failures: multi-call denial, outer Python denial, nested bridge denial,
and undeclared native-call rejection. These were specification/instrumentation
defects: Rust already paired those results. `DenyPermission` now retires only
the permission request; `CompleteTool` records each outer result, including
synthetic results. Python and registered tools now share the same permission
entry point and trace emission. This does not change the permission policy.

Two separate runtime corrections have deterministic Rust regressions:

- `cancellation_at_truncated_batch_boundary_preserves_steering`: a cancelled
  truncated response repairs its tool pairs and leaves steering queued. The
  earlier path consumed the steer before noticing cancellation.
- `provider_measurements_expire_when_history_or_provider_changes`: new user or
  tool messages, steering, and a provider switch invalidate the previous
  provider's context measurement. A completion without usage falls back to
  the current history estimate; reported counter addition saturates.

Copy-mode state was removed from the protocol model after checking the
controller paths. Its old guards disabled remembered permission resolution
and host enqueues while the real controller kept polling both. Its sole
copy-specific temporal property restated the assumed eventual resume action;
no protocol safety invariant depended on it. Display toggles now stutter, and
permission progress is stated directly as an environmental assumption.

The conformance gate also injects one fault into each original model and
requires a specific invariant counterexample: premature idle ownership,
publication without dequeueing an episode, and history disclosure outside the
requested scope. This checks that the invariants reject concrete mistakes;
it does not establish that the invariant set is complete. Parser errors and
unrelated TLC failures do not satisfy a negative test.

Commit-time `make lint` covers non-formal checks; pre-push and CI retain the
complete `make check` gate. `make tla` explicitly runs all finite state spaces,
observed traces, and mutations during protocol development. The technical whitepaper in
`whitepaper/` describes this evidence and its limits. Its validation record
identifies the checked source snapshot, commands, bounds, and results.
