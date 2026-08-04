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
selection execute only outside a mutation-capable turn and are reviewed
separately for that ownership guard. `/copy last` and `/copy all` are hidden,
explicit write-only terminal effects; `/copy select` refines `EnterCopyMode`.
Goal text, active/completed status, conversation-search query/selection,
prompt/message provenance, and other provider payloads are hidden data in this
model. `Ctrl+F` scans only sanitized in-memory `ChatEntry` values and computes
one wrapped-row viewport jump; it cannot mutate or disclose model history, the
prompt queue, tool payloads, reasoning, or archived sessions. Setting or
clearing only the objective refines a TLA+ stutter step; scheduling a
host-authored continuation linearizes to `Enqueue`, and the permission-free
completion control linearizes to `CompleteTool` plus a hidden goal-state
mutation. Reasoning-inspector contents and scroll are also hidden display data.
Copy-mode ownership itself is modeled because it gates all concrete terminal
actions: while `copyMode` is true, provider/tool progress remains enabled but
queue, cancel, and permission-choice input is disabled.

## Machine-checked implementation traces

`src/model_trace.rs` adds an opt-in, payload-free event sink at existing Rust
linearization points. Ordinary application construction installs no sink. The
events carry only abstract action data: stable IDs, counts, categorical scope
filters, and trace-local opaque scope labels. Prompt text, archive content,
tool input, provider payloads, and project paths never enter the trace.

`examples/model_conformance.rs` executes deterministic paths through the real
`PromptQueue`, `Agent`, `ToolRegistry`, `HistoryStore`, and `EpisodicMemory`
implementations. `scripts/render-model-traces.py` normalizes their opaque IDs
into each checked model's finite domain and generates a temporary wrapper for
each original TLA+ module. Every wrapper step requires both the original
module's `Next` relation and the exact named source action with the observed
arguments. Its `TraceCompletes` property requires TLC to consume the complete
Rust-produced sequence; an unrelated model execution cannot satisfy the gate.

`scripts/check-model-conformance.sh`, also available as `make conformance`,
checks these three observed traces and then checks that TLC rejects three
deliberately corrupted refinements:

- async continuation moved before its tool result;
- memory disclosure of an episode already forgotten from the live store; and
- a project history result returned for a request mutated to global scope.

This closes part of the previous manual-only refinement gap. It is sampled
safety conformance, not an exhaustive proof that the Rust program refines TLA+
and not a concrete liveness proof. The current deterministic traces exercise an
allowed tool round followed by an answer, enabled memory capture, same-scope
history and memory writes, and allowed current-scope archive searches. The
event vocabulary also covers provider refusal/failure, steering rollback,
permission denial, capture skip/failure, deletion, global startup, empty
search, and cancellation repair, but those paths do not become checked merely
because a variant exists. Queue editing, copy-mode input ownership, PTY
behavior, persistence failures, batches beyond the representative archive
result, providers, and external tool effects remain covered by the model
checker, focused Rust/PTY tests, and the review tables to different degrees.

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
| `copyMode` | `tui::AppState::copy_mode`, changed only by `TerminalUi::toggle_copy_mode` | It controls terminal ownership/redraw only; no queue, history, permission ID, or turn state moves with the toggle. |
| `phase` | Control location in `main::drive_started_turn`, `Agent::run_started_turn`, and the permission broker | It is a model program counter, not a second mutable Rust enum that could drift. |
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
| `Enqueue` | `tui::submission_delivery` selects steer/follow-up; `main::enqueue_submission` forces idle and local-command submissions to follow-up; `PromptQueue::enqueue` assigns user IDs. After a normally settled active-goal turn, `reconcile_goal_continuation(true)` appends at most one host-authored follow-up with the same fresh-ID transition. | `composer_keys_distinguish_busy_steering_from_followups`, `duplicate_text_keeps_distinct_stable_ids`, `goal_continuation_is_unique_removable_and_becomes_user_text_when_edited`, `active_goal_continues_only_after_normal_settlement` |
| `DeleteQueued` | Queue modal deletion routes the selected stable ID to `PromptQueue::delete`. Restore is the same modeled removal plus a display-only move into an empty composer; it is refused over an existing draft. | `queue_manager_mutations_address_stable_ids`, `restoring_queue_text_never_overwrites_a_draft` |
| `ReclassifyQueued` | Queue modal `s` calls `PromptQueue::toggle_delivery`; idle cannot create a steer. | `queue_manager_mutations_address_stable_ids`, `ending_turn_normalizes_undelivered_steers` |
| `MoveQueuedEarlier` | Queue modal Ctrl+Up/Down calls `PromptQueue::move_by` by ID. | `queue_manager_mutations_address_stable_ids` |
| `EnterCopyMode` | F3 is intercepted before modal/composer dispatch; idle `/copy select` calls the same transition without relying on a function key. Both disable mouse capture, set `copy_mode`, draw the paused banner once, and then suppress application input/redraws. OSC 52 `/copy last|all` requests do not change this state. | `copy_mode_banner_explains_that_rendering_is_paused`, `copy_mode_can_always_resume_with_escape_without_stealing_idle_escape`; exact-binary PTY copy test |
| `ExitCopyMode` | F3 or Esc reenables mouse capture, clears `copy_mode`, and performs one immediate redraw of state accumulated by the still-polled runtime. Idle Esc retains its existing editor behavior. | `copy_mode_banner_explains_that_rendering_is_paused`, `copy_mode_can_always_resume_with_escape_without_stealing_idle_escape`; exact-binary PTY progress test |
| `DispatchFollowUp` | The idle outer loop calls `claim_follow_up` once; it never scans past a non-follow-up head. | `followups_dispatch_one_at_a_time_in_fifo_order` |
| `CommitStart` | Without an await, the controller calls `Agent::begin_turn`, commits the `PromptClaim`, and atomically writes the started history plus remaining queue. | `dropped_claim_rolls_back_and_commit_removes`; source-order review in `main` |
| `RequeueStart` | Dropping an uncommitted `PromptClaim` restores the same ID at the front. Production start is currently infallible between claim and commit; the action over-approximates unwinding and future fallible setup. | `dropped_claim_rolls_back_and_commit_removes` |
| `ProviderAnswer` | `complete_with_retry` commits a complete assistant response, emits no partial response into history, then reaches the no-tools boundary. Text and provider-supplied reasoning are separate display deltas; aborted streams receive separate display-only uncommitted markers. | `steering_queued_during_final_response_gets_another_model_call`, `provider_cancellation_commits_no_partial_assistant_message`, `cancelling_a_partial_stream_marks_the_visible_text_uncommitted`, `streamed_text_is_not_double_emitted`, `provider_reasoning_stays_out_of_chat_and_has_a_live_inspector` |
| `ProviderRefusal` | Refusal pairs any anomalous tool uses with synthetic errors, checkpoints, and returns without accepting steering. | `refusal_with_tool_uses_is_repaired_before_checkpointing` |
| `ProviderFailure` | A non-retryable error or exhausted retry sequence commits no partial assistant response and returns `Err`; the controller preserves prior valid history, normalizes queued steers to follow-ups, and returns idle without consuming them. | `non_retryable_errors_surface_immediately`, `history_survives_api_errors_after_tool_execution` |
| `ProviderToolBatch` | A committed assistant response is scanned into a finite `tool_uses` vector; one loop iteration is consumed. | `tool_results_are_truncated_in_history`, `iteration_limit_leaves_late_steering_for_controller_normalization` |
| `CompleteTool` | Tools run sequentially; each outcome becomes one `ToolResult`. Truncation and unknown tools also return structured results. In code mode, an undeclared native call is never started or permission-checked and receives a synthetic error result. The reserved `update_goal` host control validates exact completion input, clears hidden goal state without capability permission, and produces a normal paired result at this same boundary. | `tool_results_are_truncated_in_history`, `history_survives_api_errors_after_tool_execution`, `code_mode_rejects_unadvertised_direct_tool_calls`, `update_goal_completes_without_capability_permission`, `invalid_goal_completion_keeps_the_goal_active` |
| `AskPermission` | `PermissionBrokerPrompt::choose` allocates a monotonic request ID, sends one `PermissionRequest`, and awaits its one-shot. | `broker_correlates_the_ui_reply_with_its_request` |
| `AllowPermission` | The reactor sends the choice only when the modal ID equals the live request ID; the handler records `AllowAlways` before execution. | `broker_request_ids_keep_out_of_order_replies_correlated`, `memory_handler_remembers_decisions_without_prompting` |
| `DenyPermission` | A denial becomes a structured denied result. Denials inside code-mode bridge calls propagate through `ScriptResult::denied` even if Python exits successfully. | `dropped_broker_reply_denies_instead_of_hanging`, `denial_inside_code_mode_pauses_the_outer_turn` |
| `PermissionResolution` (fairness action) | This is the union of allow/deny, not another Rust transition. Both choices are unavailable while copy mode owns input and become available again on F3/Esc resume. | broker correlation tests; exact-binary permission-during-copy PTY trace |
| `ClaimSteering` | At a history-valid boundary, `PromptQueue::claim_steering` removes all visible steers, preserves their relative order, and leaves follow-ups. | `steering_claim_preserves_relative_order_and_followups` |
| `CommitSteering` | `Agent::commit_steering` appends claimed text to the valid user boundary, commits IDs, emits `SteeringCommitted`, and checkpoints with no await between those operations. | `steering_queued_during_final_response_gets_another_model_call` |
| `RequeueSteering` | Dropping an uncommitted steering claim restores the same IDs at the front. Normal commit contains no fallible/await boundary. | `dropped_steering_claim_restores_the_same_ids_at_the_front` |
| `ContinueAfterTools` | A complete, non-denied tool-result batch with capacity remaining loops to the next provider request. | `tool_results_are_truncated_in_history`, `history_survives_api_errors_after_tool_execution` |
| `SettleTurn` | Answer, refusal, provider failure, denial, or iteration cap releases `&mut Agent`; the controller converts remaining steers to follow-ups, reconciles automatic goal work, writes one atomic autosave containing history, goal, prompt sources, and queue, and only then offers protocol-valid settled history to the independent memory FIFO. Completed/capped active goals linearize to `SettleTurn` followed by modeled `Enqueue`; interruption, refusal, denial, error, or exit removes automatic entries and pauses without clearing the goal. | `refusal_with_tool_uses_is_repaired_before_checkpointing`, `history_survives_api_errors_after_tool_execution`, `denial_inside_code_mode_pauses_the_outer_turn`, `iteration_limit_leaves_late_steering_for_controller_normalization`, `active_goal_continues_only_after_normal_settlement`, `episodes_omit_reasoning_and_tool_payloads` |
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
| `DeliveryIsWellFormed` | `DeliveryMode` has only two variants; idle submissions and host goal continuations are follow-ups, and all residual steers are normalized when ownership ends. `PromptSource` does not alter delivery semantics. |
| `SafeSteeringBoundary` | `commit_steering` is called only after a complete assistant answer or after the full result vector is appended. It is not called on refusal, cancellation, or an exhausted iteration budget. |
| `TerminalReasonIsWellFormed` | Distinct Rust branches implement final answer, refusal, terminal provider error, structured denial, and cap. Error/refusal/cap cannot consume steering. The cancellation/permission race test prevents a cancelled turn from taking the denial-to-steer branch. |
| `ToolHistoryIsValid` | `history_tool_protocol_is_valid` checks exact adjacent ID sets; every emitted `HistoryCheckpoint` debug-asserts it, every persistence path refuses an invalid history, load rejects invalid saves, and cancellation/refusal tests inspect checkpoint histories. |
| `PermissionIsCorrelated` | IDs are monotonic, choices travel over the request's own one-shot, mismatched modal IDs are ignored, and dropping a reply denies once. |
| `SettledPromptsAreCommitted` | The controller records the initial user message before `PromptClaim::commit`; interrupted and ordinary outcomes therefore refer to a committed follow-up. |
| `HistoryOrderHasStableIds` | Queue claims and `SteeringCommitted` preserve stable-ID order. `committedOrder` is a model ghost variable, verified through claim/event tests rather than stored in model-visible text. |
| `EveryBusyPeriodSettles` | TLC checks settlement for the finite bounds under weakly fair agent progress, weakly fair copy exit, and strongly fair permission resolution when its UI is available infinitely often. Rust bounds provider rounds/retries and keeps polling them during copy mode; permission input waits for resume. External tools/providers, a user who never resumes, or a user who never answers remain environmental blockers. |
| `CopyModeEventuallyResumes` | `WF_vars(ExitCopyMode)` makes the environmental assumption explicit rather than silently proving liveness through a permanently disabled terminal. The PTY test verifies both concrete resume keys; it cannot force a real user to resume. |

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
| `SettleTurn` | After the ordinary runtime settles and its autosave is attempted, `drive_started_turn` maps `TurnOutcome` (or error) to `EpisodeOutcome` and sends the history tail anchored immediately before `begin_turn`. Duplicate steering text cannot move that boundary. Host-authored goal continuations are filtered rather than retained as user text. If `history_revision` shows that in-turn compaction relocated the boundary, capture degrades to the original prompt and labels the record `prompt_only`. | `compaction_summarizes_old_history_and_preserves_recent`, `episodes_omit_reasoning_and_tool_payloads`, `host_goal_continuations_are_not_retained_as_user_authored_text`, `duplicate_steering_text_does_not_move_the_episode_boundary`, `a_relocated_history_boundary_degrades_to_prompt_only` |
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
| `histories`, `memories` | Scoped state files under manifest-bearing scope directories and scope-keyed rows in `scoped-episodes.sqlite3` | These sets include archives written by prior runs. The model hides their text and disk representation. |
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
| `SaveHistory` | Every autosave, checkpoint, queue edit, `/save`, and compaction routes through the active `HistoryStore::save`, which rejects a state claiming another scope and writes beneath the deterministic scope directory. | `project_autosaves_are_isolated_and_global_does_not_fallback`, `save_rejects_scope_mismatch_and_path_traversal`, `persistence_rejects_an_invalid_tool_protocol_boundary` |
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
while the agent future owns `&mut Agent`. Queue edits write that boundary, goal,
and current queue together beneath
`~/.generalist/history/scopes/<scope-id>/autosave.json` using
flush, atomic rename, and parent-directory flush. `HistoryCheckpoint` replaces
the history boundary and its corresponding goal only after
`history_tool_protocol_is_valid` holds. Carrying both in one event prevents the
separate display notification for `GoalCompleted` from racing persistence. On
restart, the goal is restored independently from the same scope and its next continuation is
reconciled; queued work is recovered only together with that autosaved
conversation. Because no turn survives a process exit, residual steers are
normalized to follow-ups. The
`structured_state_does_not_collide_with_legacy_input_history_file` and
`persistence_rejects_an_invalid_tool_protocol_boundary` tests exercise the
filesystem and protocol guards; the state round-trip test covers the mandatory
scope, optional goal, and queued prompt provenance. `UiAction::QueueChanged`
and `Submit` are the only terminal-event effects that request an autosave; idle
local commands and host goal scheduling are persisted by the controller after
execution. Display-only actions are covered by
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
  model's payload abstraction. Code mode advertises only `python` as a
  capability plus the reserved `update_goal` control while a goal is active; any
  other undeclared response is paired with an error for `ToolHistoryIsValid`
  but is never exposed as executable tool activity.
- `NoAutomaticRetrieval` is not a same-UID filesystem sandbox. The dedicated
  archive APIs are permission-gated and the host performs no prompt injection,
  but the advertised `python`, `bash`, and file tools can access the same local
  files if separately authorized. Hard confidentiality between project
  archives requires an OS sandbox or a dedicated user, not this model.
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
- Code-mode bridge calls are flattened into the active tool batch. Their
  permission denials are propagated, but their intermediate payloads and
  subprocess/socket mechanics are outside the model. Archive reads return
  bounded, resumable transcript pages so the bridge does not receive one
  unbounded result; building/searching the local archive still has no storage
  quota.
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

## 2026-07-27 scoped-archive and full refinement audit

After implementing project/global isolation, the three models and their live
Rust mappings were re-read action by action. The audit corrected these actual
drifts rather than merely updating prose:

- `WorkspaceScope::default()` and unscoped state could silently select global.
  The type now has no default, `SavedState` requires a scope, and unscoped
  files are rejected.
- `read_conversation` repeated an exact scope label but not the modeled
  categorical filter. Both conversation and memory reads now repeat the search
  filter and exact label.
- conversation catalog construction opened every scoped state file before
  filtering results. Scope manifests are now filtered before an unselected
  conversation file is opened; SQLite already applies its predicate before
  memory matching, and the final episode fetch now repeats that predicate.
- the memory lifecycle model allowed one worker's scope to change between
  turns. It now follows one immutable `CurrentScope`; the archive-routing model
  separately covers pre-existing cross-scope rows and explicit global startup.
- the archive permission invariant correlated only sentinel/filter values. It
  now carries a last-disclosure authorization witness and explicitly treats a
  remembered allow-always decision as a permission-policy approval, not a new
  interactive prompt.
- code-mode bridge calls could receive an unbounded archive body before the
  outer agent result cap. Read tools now return bounded, resumable sanitized
  transcript pages with `next_offset`.
- terminal provider failure existed in Rust but not `AsyncRuntime.tla`.
  `ProviderFailure` now settles without committing partial output or consuming
  queued steering.
- valid non-UTF paths had lossy, potentially colliding scope labels, and
  control-bearing paths could enter model context. Unsafe path labels now use
  an exact byte-hex representation.
- the traceability lint searched the entire document, so a stale historical
  paragraph could satisfy it. It now requires every model variable, action,
  invariant, and temporal property in the appropriate live mapping section.

TLC then completed all checked configurations without error:

- `AsyncRuntime.cfg`: 470,086 generated states, 117,750 distinct, depth 27;
- `MemoryRuntime.cfg`: 7,627 generated states, 1,690 distinct, depth 16; and
- `ArchiveScopeRuntime.cfg`: 163,652 generated states, 20,341 distinct,
  depth 11.

The complete locked Rust validation passed 144 library tests, 6 binary tests,
all example targets, and 2 documentation tests. Formatting, Clippy with
warnings denied, ShellCheck, the strengthened traceability lint, research
corpus validators, hook tests, and `git diff --check` also passed.

That audit still left a manual-only connection between Rust executions and TLA+
actions. The machine-checked trace bridge described above now validates
representative real executions and rejects targeted invalid mutations. The
remaining boundary is still explicit: this is finite model checking, sampled
trace conformance, a human-reviewed refinement map, and deterministic Rust
evidence, not a proof that every Rust execution refines TLA+. Archive batches
use a representative-record abstraction, external provider/tool termination is
an environmental assumption, and same-UID `python`/`bash`/file capabilities can
read local archive files if separately authorized. Those are residual limits,
not claims discharged by the green checks.

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
with no error under `spec/AsyncRuntime.cfg`. That historical
pre-`ProviderFailure` baseline is superseded by the 2026-07-27 audit above; neither
run is a permanent certification or a claim that the program can force a human
response.

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

## 2026-07-27 episodic-memory refinement audit

This section records the pre-scoped episodic baseline. Its observation that no
model-facing memory tool existed was true for that exact rebuilt binary and is
superseded by the permission-gated, read-only archive design mapped above.

The prototype was traced from a settled TUI turn through
`enqueue_settled_turn`, the sole FIFO worker, the project-scoped SQLite row,
and each explicit `/memory` read/delete path. The exact rebuilt binary ran in
an isolated 100×30 tmux PTY against a delayed loopback OpenAI-compatible SSE
server. Capture started paused with zero rows; after explicit resume, a prompt
typed and queued during the delayed response ran only after that response
settled. Every intercepted provider request advertised exactly `python`, with
no model-facing memory tool.

Explicit search found the captured prompt, show rendered its stored
user/assistant text, and export produced a four-row JSON file with mode `0600`
inside a `0700` directory. The database itself was `0600`. A separate process
then held `BEGIN IMMEDIATE`: the memory setting update failed after the
configured busy timeout with `database is locked`, but text entered during
that wait was retained and dispatched, and the failed update left the prior
paused setting unchanged. Forget removed exactly the selected row from the
live store, search no longer found it, and the warning preserved the explicit
export/backup/snapshot limitation. Normal `/exit` closed the PTY session and left
three valid rows on disk. Deterministic tests separately inject reasoning,
redacted reasoning, tool arguments, and tool-result bodies and confirm that
none enter the episode or export representation.

TLC explored 1,433 states (694 distinct, depth 15) with no error under
`spec/MemoryRuntime.cfg`. This establishes the checked finite lifecycle and
no-retrieval invariants, not the Rust refinement by itself; the source trace,
fault injection, and exact-binary PTY exercise are the separate refinement
evidence above.

After the adversarial source pass corrected busy-checkpoint reporting,
tool-ID retention, and duplicate-prompt boundary selection, the exact source
was rebuilt and the isolated PTY cycle was repeated. A single key burst queued
status, resume, a delayed turn, and a follow-up; both settled rows were present
after clean exit, explicit search found the first, file modes remained
`0700`/`0600`, and both intercepted provider requests still advertised only
`python`.

## 2026-07-27 active-goal autorun refinement audit

The exact rebuilt debug binary ran in an isolated 100×30 tmux PTY with
`GENERALIST_HOME` pointing at a fresh profile and a loopback OpenAI-compatible
SSE fixture. `/goal finish the loop` entered no local-command text into model
history. The fixture first returned `partial progress` with a normal stop; the
settled controller assigned a fresh ID to one host continuation and issued a
second request without terminal input. The second response called
`update_goal({"status":"complete"})`; Generalist executed it without a
permission modal, paired the tool use, cleared the header, and made one final
provider request. That response returned `goal done`, after which the UI
remained idle with no queued continuation.

The intercepted first and second requests both contained the exact objective in
system instructions and advertised `python` plus `update_goal`. The third
request omitted the active-goal system section and advertised only `python`,
while retaining the paired `update_goal` use/result in conversation history.
The final atomic autosave had `goal: null`, an empty queue, two user-role
continuations marked `origin: goal_continuation`, and a protocol-valid final
assistant message. Focused tests separately cover invalid completion input,
permission-policy bypass, unique continuation reconciliation, provenance
sanitization on load, omission from episodic user text, and the outcome table
that pauses automatic work after interruption, denial, refusal, error, or
exit.

No new TLA+ state variable is required for objective text or provenance: those
remain hidden payload. The host enqueue is a concrete `Enqueue` transition with
fresh identity and follow-up delivery; `update_goal` is `CompleteTool` plus a
hidden state mutation; normal settlement followed by scheduling linearizes to
`SettleTurn` then `Enqueue`. The updated action comments and matrices record
those refinement points explicitly.

## 2026-08-04 clipboard interaction refinement audit

The exact rebuilt debug binary ran in an isolated 100×30 PTY against a local
OpenAI-compatible SSE fixture. After one committed Unicode response, `/copy
last` emitted one complete OSC 52 request whose decoded payload exactly matched
the assistant text. `/copy all` decoded to the exact labeled user/assistant
transcript. The fixture received no extra model request for either local
command. Deterministic tests separately confirm that reasoning, redacted
reasoning, tool arguments, tool results, and host-authored goal continuations
do not enter clipboard payloads, while matching manually authored text remains.

`/copy select` emitted the full mouse-capture disable sequence and froze the
selection frame. A bare Esc emitted the mouse-capture enable sequence and
redrew the live UI, exercising the same modeled `ExitCopyMode` transition as
F3 without changing idle Esc behavior. After resumption, a bracketed multiline
Unicode paste reached the second intercepted provider request byte-for-byte.
Normal exit disabled bracketed paste and left the alternate screen. These
checks establish emitted bytes and concrete TUI transitions; they do not prove
that an arbitrary terminal's OSC 52 policy accepts a clipboard write.

## 2026-08-04 conversation-search refinement audit

The exact rebuilt debug binary ran in an isolated 100×30 PTY against a
loopback OpenAI-compatible SSE fixture. An early response was long enough to
leave the viewport and a second response streamed a duplicate Unicode marker.
While that second stream was stalled, `Ctrl+F` accepted a bracketed multiline
Unicode paste as one single-line query, reported both matching assistant
entries, and moved selection between them. The scoped autosave was byte-for-byte
unchanged while search owned the terminal and the provider was stalled.

The fixture then completed the response while the search modal remained open;
the settled assistant text reached the autosave without closing the modal.
Selecting the earlier result restored its first wrapped row to the physical
terminal. Exactly two provider requests occurred, the deliberately mixed-case
search spelling appeared in neither request nor final autosave, the queue was
empty, and normal exit disabled bracketed paste and left the alternate screen.
Deterministic rendering tests separately calculate the selected entry's start
with Ratatui's actual wrapping and assert the viewport lands on that exact row;
other tests cover case-insensitive matching, stable entry identity during live
appends, modal mouse ownership, and Unicode paste normalization.

Conversation-search query, selection, and the derived viewport jump remain
hidden display data in `AsyncRuntime.tla`: input ownership suppresses concrete
queue/cancel actions while the modal is open, but provider/tool transitions
continue. Search reads only sanitized in-memory `ChatEntry` values and has no
path to provider history, queued prompts, tool payloads, reasoning, archived
sessions, or durable writes.
