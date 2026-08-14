# Asynchronous TUI architecture

This note defines the interaction model for Generalist's Ratatui frontend. It
exists because a spinner drawn from another thread is not an asynchronous UI:
the composer, transcript, queue, permissions, and cancellation must all remain
responsive while a provider request or tool is running.

The design was checked against the current implementations of:

- OpenAI Codex (`openai/codex` at
  `61a44880a85d2fd0d8770908dea5733495e571c8`), especially its TUI input queues,
  `turn/steer` protocol, and turn-local pending-input queue.
- pi (`badlogic/pi-mono` at
  `5bc1c2c0a6f07e00e8c240304182f213ab8d311f`), especially its separate steering
  and follow-up queues and the safe-point checks in `agent-loop.ts`.
- Claude Code's public changelog and documentation. Claude Code has accepted
  input while busy since 0.2.75, but its public material does not expose an
  implementation boundary comparable to the two open-source agents.

Primary references:

- <https://github.com/openai/codex/blob/main/docs/tui-chat-composer.md>
- <https://github.com/openai/codex/blob/main/codex-rs/tui/src/chatwidget/input_queue.rs>
- <https://github.com/openai/codex/blob/main/codex-rs/tui/src/chatwidget/input_flow.rs>
- <https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs>
- <https://github.com/badlogic/pi-mono/blob/main/packages/agent/src/agent-loop.ts>
- <https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md#message-queue>
- <https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md>

## User-visible semantics

The UI distinguishes presentation activity from mutation-capable turn
ownership. Submission therefore has three intentional modes:

| Action | Idle | Background work (startup, compaction, memory) | Active model turn |
| --- | --- | --- | --- |
| `Enter` | Start a turn | Queue a follow-up for when the operation settles | Steer at the next safe boundary |
| `Tab` | Complete a recognized slash prefix; otherwise start a turn | Complete a slash prefix; otherwise queue a follow-up | Complete a slash prefix; otherwise queue a follow-up |
| `Alt+Enter` | Start a turn | Queue a follow-up | Queue a follow-up after the turn settles |
| `Shift+Enter` or `Ctrl+J` | Insert a newline | Insert a newline | Insert a newline |
| `Alt+Up` | Restore the latest queued item into an empty composer | Restore the latest queued item | Restore the latest queued item |
| `F2` | Open queue manager | Open queue manager | Open queue manager |
| `F3` | Toggle native terminal copy mode | Toggle copy mode without stopping the operation | Toggle copy mode without stopping provider/tool progress |
| `F4` | Inspect provider-supplied reasoning | Inspect the current reasoning log | Inspect live provider-supplied reasoning |
| `Ctrl+F` | Search visible conversation entries | Search while the operation continues | Search while provider/tool work continues |
| `/copy last`, `/copy all`, or `/copy reasoning` | Send the selected committed field to the terminal clipboard | Queue the local command | Queue the local command until the turn settles |
| `/copy select` | Enter native terminal copy mode | Queue the local command; F3 remains immediate | Queue the local command; F3 remains immediate |
| `Esc` | Clear/close the current UI layer | Apply the operation's policy: skip discovery, cancel compaction, or report a non-cancellable memory transaction | Interrupt the active turn when no modal owns the key |

Modal input takes priority. In particular, Esc on a permission modal means
deny-once; that denial settles the turn unless queued steering redirects it.

A steering message is not an immediate interruption. It is inserted after the
current assistant response and its complete tool-call batch, before another
model request. A follow-up is a separate user turn and runs only once the
current turn has completed, failed, been denied, or been interrupted.

The queue manager shows every unclaimed item and supports edit, delete,
reorder, and changing an item between steer and follow-up. Follow-ups are
started one at a time. Multiple steering messages present at one safe boundary
are delivered together, in FIFO order, to avoid unnecessary model round trips.
Its viewport follows the stable-ID selection even when the queue is longer than
the modal. Restore is refused while the composer contains a draft, so moving
text out of the queue cannot silently destroy unsent input.

Local commands are parsed through the typed catalog in `src/command.rs` before
dispatch; queueing `/clear` or `/goal edit` must never accidentally turn it
into model-visible text. Typing `/` gives the composer a distinct command
border/title and lists catalog entries in the footer. The help window renders
the same catalog, so parser and discovery cannot silently drift. Help may open
immediately. Commands that require mutable agent state wait in the same visible
queue until the active turn releases that state.

`/tools`, `/tools search <query>`, and `/tools show <name>` project the
finalized registry into bounded host-only display text. Listing is stable and
search is case-insensitive over names/descriptions. Detail distinguishes
ordinary bridge definitions, progressive schema-on-demand definitions, the
synthetic `python` runner, and the goal-only `update_goal` control. Inspection
does not issue a provider request, ask permission, execute a tool, alter
conversation history, or change durable state (the outer idle boundary may
still checkpoint an unchanged snapshot).

`/history`, `/history search <query>`, and `/history show <name>` similarly
project only the active `HistoryStore` scope. They reuse the sanitized archive
event representation, omit provider reasoning and tool payloads, and never
load the selected session into `Agent`. `/history forget <name>` is a separate
current-scope mutation: it first verifies the named state, opens a confirmation
with Cancel selected, durably unlinks only that file, and reports that prior
copies remain. It refuses the live `autosave`. None of these host controls
issues a provider request or asks model-tool permission; the separate
cross-scope archive tools still require an exact permission-derived disclosure
grant. `/save <name>` and `/load <name>` bypass only the prompt/picker; bare
`/save` and `/load` retain the interactive paths. Manual save first attempts an
atomic no-clobber create; an existing valid checkpoint requires a separate
default-Cancel replacement confirmation, while invalid/symlinked targets and
the controller-owned `autosave` name are refused.

`/usage` and `/usage show` read a process-local ledger populated from
`ApiCallStarted` and `ApiCallFinished`. Attempts are grouped by the provider and
model active when each call began. Only an explicit provider `Usage` payload
adds input/output tokens; missing payloads remain counted as unreported
attempts, and optional cache fields retain report coverage. `/usage reset`
clears the ledger without changing model context, conversation history,
provider state, or durable files. The command does not estimate price and does
not conflate cumulative reports with the header's current-context estimate.

Command completion is derived from the same `COMMAND_SPECS` entries and typed
subcommand metadata. At the end of the composer, unmodified Tab canonicalizes
a unique command/subcommand prefix or reports all ambiguous matches without
submitting. Once a value-bearing command has argument text, for ordinary
prose, with a mid-string cursor, or with modified Tab, the existing follow-up
path is unchanged. Completion itself is editor-only state and is neither
queued nor persisted.

### Interactive MCP discovery

When `.generalist/mcp.json` exists, startup connects servers in sorted name
order before constructing `Agent`. The discovery future owns the mutable tool
registry, while the ordinary UI reactor continues polling terminal input,
frame ticks, stale permission messages, memory-worker events, and per-server
progress. A single terminal reader remains authoritative.

Recovered goal/queue state is classified before discovery. A queued,
protocol-valid autosave restores its matching history and permission policy;
goal state survives independently, exactly as in ordinary crash recovery.
Composer submissions and F2 mutations during discovery operate on that same
authoritative queue and queue a complete autosave with the admitted pre-agent
history boundary. They are always follow-ups because no turn exists to steer.
Help, search, and copy mode remain display-only. `Esc` drops the pending discovery
future and continues with the deterministic prefix of servers that already
reported completion; an in-progress stdio child has `kill_on_drop`, and an
in-progress HTTP request is cancelled with its future. No provider request can
start until discovery finishes or is explicitly skipped and the registry is
moved into `Agent`.

The controller retains typed per-server outcomes after startup. `/mcp status`
renders them in stable server-name order without opening a connection.
`/mcp retry` selects every failed/skipped server; `/mcp retry <server>` selects
one exact configured name. Retry reuses the same responsive discovery reactor
at an idle boundary while temporarily borrowing the existing `Agent` registry.
Already connected servers are refused rather than duplicated. Successfully
recovered tools enter that registry and therefore the next dynamically built
code-mode bridge surface; conversation history, provider state, permissions,
and the loaded configuration are not replaced. An interrupted retry leaves its
unfinished targets explicitly skipped and retryable again.

### Active goal

`/goal <objective>` sets a user-authored objective and queues a host-authored
follow-up. `/goal edit` (or bare `/goal`) opens a prefilled editor and resumes
that loop, `/goal show` displays the objective, and `/goal clear` stops and
removes it. `Agent` is the authority for the raw goal. It appends the objective
and exact completion contract to the effective system instructions for every
provider request without copying the objective into a conversation message.
Changing it invalidates cached context accounting, while `/clear` deliberately
preserves it. The prefilled editor supports the same Ctrl+A/E/U/K/W replacement
controls as the composer.

A normal final answer settles one turn but does not clear an active goal. After
`Completed` or `MaxIterationsReached`, the controller ensures exactly one
`PromptSource::GoalContinuation` follow-up exists; the ordinary queue then
dispatches it with a fresh stable ID. This repeats until the model calls the
native `update_goal({"status":"complete"})` host control. The control is
validated and executed synchronously without capability permission, clears the
authoritative goal, produces the required paired tool result, and emits a
checkpoint containing both the new goal state and valid history. The next
provider response can therefore summarize completion without receiving the old
goal again.

Interruption, refusal, denial, provider error, or exit removes pending automatic
continuations but leaves the objective available for inspection or editing.
Submitting an ordinary prompt or editing the goal resumes continuation after
the next normal settlement. This makes Escape an effective pause instead of
letting the idle controller immediately restart the turn.

The TUI holds only a sanitized render copy and displays it on the second header
row. Saved sessions and autosave carry the raw optional goal. Startup restores
it even when there is no queued turn and schedules one continuation; named
`/load <name>` replaces the current goal with the loaded session's value and reconciles
its automatic queue entry. Once dispatched, a continuation carries
`MessageOrigin::GoalContinuation`; matching text without that provenance
remains ordinary user input in rendering and episodic capture.

## Ownership model

The program has one UI reactor and one active agent future. It does not have a
render thread and does not put the entire `Agent` or terminal behind a mutex.

```text
terminal input ─┐
frame tick ─────┤
agent events ───┼──> UI reactor ───> Ratatui draw
permission req ─┤        │
turn completion ┘        ├──> authoritative prompt queue
                         └──> permission replies

                              safe-point claim
authoritative prompt queue ─────────────────────> active Agent future
```

The UI reactor is the only owner allowed to read terminal events or draw.
Provider and tool work remains a future polled by the same Tokio runtime task;
this preserves the existing `Provider` trait's `?Send` contract. `tokio::select!`
keeps terminal events, frame ticks, permission requests, and the active turn
progressing together.

Startup MCP discovery uses the same shape before `Agent` exists: the pinned
discovery future exclusively borrows the registry while the reactor owns the
terminal and pre-agent queue. `AppState::busy` controls presentation and
interrupt routing; the separate `turn_active` projection controls whether Enter
may request steering and whether F2 may reclassify a follow-up as a steer. The
controller sets that projection immediately before polling
`Agent::run_started_turn` and clears it with the busy state on every outcome.
The actual `&mut Agent` future remains the authority for turn ownership.

The active future emits events into a same-task display pump. Ordered structural
events remain lossless. Consecutive answer and provider-reasoning fragments in
one structural segment coalesce into one preview record whose combined retained
payload is capped at 16 KiB, preserving order within each of the two separately
rendered streams. If more bytes arrive before the reactor drains that preview,
the UI reports the omitted counts. After a successful response enters
`Agent` history, `StreamCommitted` replaces the bounded preview with text and
inspectable reasoning derived from that authoritative response. A failed or
interrupted stream has no committed snapshot, so its cap marker remains beside
the existing uncommitted-stream label. `HistoryCheckpoint` continues through a
separate lossless, high-priority path.

The authoritative snapshot is still projected before Ratatui sees it. Each
conversation or reasoning entry retains at most 64 Ki characters, keeping its
beginning and end around an explicit omission marker. The complete accepted
response remains in `Agent` history and is available through `/copy last`,
`/copy all`, or `/copy reasoning`. Live entries stop growing after they reach
the display cap and are replaced from committed history on success.

Generation policy and host safety are separate. `Agent::max_tokens` is an
optional ordinary-completion request: Anthropic resolves the selected model's
advertised maximum when it is absent, while OpenAI-compatible adapters omit the
non-portable field. `CompletionLimits` independently bounds logical bytes,
content blocks, tool uses, and transport framing. Built-in adapters enforce
these during accumulation; `Agent` revalidates every final response and keeps a
stream-limit error sticky even if a custom provider violates the callback
contract and ignores it.

Drawing happens on the 50 ms frame tick rather than once per token or terminal
event. Rapid key-repeat and mouse-wheel bursts therefore update in-memory
display state immediately but produce at most one frame per tick. Dirty
tracking avoids rebuilding an idle transcript, and spinner-only frames run at
10 FPS. Permission, frame, and terminal branches are polled ahead of ordinary
display events. The pump is intentionally not a second committed-history
authority: committed history remains inside `Agent`.

Conversation scrolling stores an absolute top line while follow-latest is
paused. New streamed lines therefore do not move the user's viewport.
PageDown/mouse-down clamps at the real bottom and resumes follow-latest; PageUp
cannot accumulate an invisible overscroll debt. Scroll bounds come from
Ratatui's own word-wrapping layout rather than a character-width estimate, so
moving a whole word onto the next row cannot make the rendered bottom
unreachable. The scrollbar is given the count of valid top-row positions
(`max_scroll + 1`) plus the visible-row count, rather than the total wrapped
rows. Its thumb therefore reaches the end of the track exactly when
follow-latest reaches the real bottom. Mouse input belongs to the active modal:
it scrolls permission details or a long queue selection instead of changing
the obscured conversation.

`Ctrl+F` opens a display-only, case-insensitive search over the sanitized
conversation entries already held by the TUI: user and assistant text plus
visible informational/error events. The query is a single-line Unicode editor;
pasted line breaks become spaces. Match identity is the entry index rather than
the preview text, so duplicate bodies remain distinct and appending a new event
does not move the selected result. Enter closes the modal and schedules one
render-time jump. That render uses Ratatui's word wrapper on the exact entry
prefix to compute the target top row, preserving the same wrapping semantics as
ordinary scrolling. Search query, match selection, and pending jump are not
conversation history, queue state, episodic memory, or autosave data. Tool
activity and provider reasoning retain their separate inspectors.

### Copy mode and reasoning inspection

`/copy` (equivalently `/copy last`) sends the latest committed assistant text
to the host terminal with an OSC 52 clipboard request; `/copy all` sends a
plain transcript of committed user/assistant text. `/copy reasoning` sends the
latest human-readable `Thinking` fields from committed assistant history.
Provider signatures and `RedactedThinking` data are excluded. Tool payloads and
host-authored goal continuations are structurally excluded from conversation
copy.
Text is base64 encoded before entering the control sequence and requests are
size-bounded. The command is a user-triggered write only: Generalist never
reads ambient clipboard contents. A successful write proves that the request
reached the terminal, not that terminal policy accepted it.

`F3` or idle `/copy select` enters a terminal-ownership mode, not a runtime
pause. On entry the UI releases mouse capture, draws a visible `display paused`
banner, hides the composer cursor, and then freezes Ratatui redraws. The
terminal can therefore perform native selection and its usual copy shortcut
without a frame erasing the selection. All application input except explicit
resume (`F3` or `Esc`) is suspended while this mode owns the screen. The same
reactor continues polling the provider/tool future, permission channel, and
frame clock; events update in-memory state and mark the frame dirty. On resume,
mouse capture is restored and one frame redraws the accumulated state.
Bracketed paste remains enabled and is accepted by the composer after copy mode
closes.

`F4` opens a scrollable, follow-latest view of model reasoning. Its data boundary
is deliberately narrower than “the agent's thoughts”: it contains only fields
the configured provider sent as inspectable output. Anthropic `thinking_delta`
text and common OpenAI-compatible string extensions are normalized into
`CompletionDelta::Reasoning`; answer text uses
`CompletionDelta::Text`. A provider attempt always gets an inspector entry, so
an endpoint that exposes no reasoning produces an explicit “no inspectable
reasoning” message rather than fabricated content. Answer text alone enters the
conversation rendering. Anthropic signatures are retained only for required
provider replay, and redacted-thinking data is represented by a placeholder;
neither is emitted to the inspector. Unsigned reasoning from an
OpenAI-compatible endpoint is retained for inspection/history loading but is
not sent as an Anthropic thinking block after a provider switch.

## Prompt queue

The prompt queue is a single shared state store. Each entry has:

- a monotonically increasing `PromptId`;
- the original text;
- a delivery class (`Steer` or `FollowUp`); and
- a source (`User` or `GoalContinuation`).

The TUI renders snapshots of this store; it does not maintain a second queue.
The agent atomically claims steering entries at a safe boundary. The controller
atomically claims one follow-up when no turn is active. Editing, deleting, or
reordering loses a race cleanly if the item was already claimed.

Stable IDs matter. Matching queue updates by message text is ambiguous when a
user submits the same prompt twice, and maintaining separate display and
execution queues invites drift on error paths.

Queue lifecycle:

```text
draft -> queued -> claimed -> committed
                   │
                   └── failed before commit -> queued again
```

An item is removed from the visible queue only when it is atomically claimed.
If submission fails before the agent records it, it is returned to the front
of the queue. Undelivered steering entries become follow-ups when their target
turn ends. Goal continuations are labeled separately in the queue. Editing one
converts it to an ordinary user prompt; loading a forged goal source whose text
does not exactly match the host prompt also demotes it to user source.

## Safe steering boundary

The agent checks for steering only at a history-valid boundary:

1. A provider response has completed and its assistant message is recorded.
2. Every tool use in that response has a corresponding tool result, including
   synthetic error results for truncation, denial, or cancellation.
3. Pending steering entries are claimed and recorded as user messages.
4. If either tools or steering require continuation, the next provider request
   begins.

Steering is also checked after a response with no tool calls. A message typed
while a final answer is streaming therefore causes another model request
instead of being stranded for a later turn.

Steering is never inserted:

- into an in-flight provider request;
- between one assistant message's tool uses and their tool results;
- into manual compaction or another operation whose transcript protocol does
  not support it.

## Code-mode boundary

When built-in code mode is active, every provider request advertises exactly one
native capability tool, `python`. While a goal is active it also advertises the
host-owned `update_goal` control. That reserved name cannot be registered or
called through the Python bridge. Bridge names such as `tools.firecrawl_search`
occur only inside `python`'s `code` string. OpenAI-compatible servers are not
trusted to honor the advertised set: if a model emits another native name, the
agent pairs the anomalous use with a synthetic error, reports a
provider-protocol violation, and asks the model to retry through `python`. It
emits no tool-start event, requests no permission, and executes no registry
tool for that response.

Keeping the anomalous use and its synthetic result in history is intentional.
Silently dropping either side would produce a transcript that does not match the
provider's preceding response; translating the call into executable Python
would bypass the code-mode boundary.

The `/tools` view does not weaken that boundary: registered entries are
descriptive snapshots only. In particular, showing a progressive MCP schema
does not load it into model context or authorize its use.

## Permissions

Permission prompting becomes asynchronous. `MemoryPermissionHandler` awaits a
broker response instead of synchronously reading terminal input. The broker
sends a request carrying a stable request ID and a one-shot reply channel to
the UI reactor.

The UI owns the modal. A stale response whose turn or request ID no longer
matches is ignored. Interrupting a turn resolves or drops its pending request
before closing the modal. Remembered name-level or exact-input decisions emit
lightweight status events without opening a modal.

The modal renders `python.code` as control-sanitized, line-numbered,
Python-highlighted source and `bash.command` as line-numbered command text instead
of JSON strings with escaped newlines. The lightweight Python lexer distinguishes
keywords, definition and call names, builtins, strings (including multiline triple
strings), comments, numbers, decorators, and operators without changing the source.
Explicit options remain visible below the source, patches retain their colored diff
view, and other tool inputs use pretty JSON. The activity panel applies the same
Python highlighting to its first two source lines after completion and adds the
result as a separate summary, so execution does not erase the invocation being
reviewed.

The remembered policy is host-owned, inspectable, and revocable while idle.
For ordinary tools, allow-always and deny-always are persisted by name. An
interactive allow-always for `bash` or `python` instead records the exact
`(tool, input)` only for the current process; it is excluded from `SavedState`.
Explicit loaded/configured name-level allows remain honored, and deny-always
remains name-level. `/permissions` renders persisted sets in stable tool-name
order plus per-tool counts for session-only exact grants without rendering
their inputs. `/permissions reset <tool>` removes both kinds for one tool, and
`/permissions clear` removes all decisions. The outer controller queues the
persisted mutation at the same local-command boundary as goal/model/history
changes. Loading a named save replaces both persisted sets and clears all
session-only exact grants. `SavedState::from_json`, policy replacement, live
checks, and snapshots resolve a contradictory name-level allow/deny entry in
favor of deny; new persisted snapshots are disjoint.

This avoids two terminal readers and avoids deadlocking an agent future behind
a terminal mutex held by a blocking input loop.

## Cancellation and history validity

Cancellation is cooperative and turn-scoped. Merely dropping `run_turn` is not
safe: the history may already contain an assistant tool-use message without its
required tool-result message.

The controlled agent loop observes cancellation around provider calls, retry
delays, permission waits, and tool execution:

- Before an assistant response is committed, cancellation drops the provider
  future and leaves no partial assistant message in model history.
- During a tool batch, the running tool future is dropped, remaining calls are
  not started, and every unfinished tool use receives a synthetic cancelled
  result before the turn returns `Interrupted`.
- Completed tool results remain in history.
- Streaming text already shown in the TUI is marked interrupted; it is display
  state, not silently treated as a committed assistant response. Streamed
  reasoning receives its own aborted-attempt label. Both remain inspection
  evidence, while neither partial stream enters durable model history.

The long-running Bash and code-mode Python subprocesses use
`kill_on_drop(true)`. Protocol repair does not claim to roll back external side
effects: the synthetic result says completion is unknown. Nested activity
entries whose futures disappear with a code-mode cancellation are retired by
the controller when the turn returns.

## Persistence

The controller queues a complete autosave snapshot after every committed
boundary, local state command (including remembered-permission reset/clear),
and visible queue edit, not only after a whole multi-turn queue drains. It
retains a clone of the latest history-valid boundary and active goal while the
agent future owns the live history. One dedicated writer performs filesystem
I/O off the UI reactor; if it falls behind, it coalesces pending complete
snapshots so the newest one dominates. Each actual write stores history, goal,
persisted permission policy, and queue together using file flush, atomic
rename, and parent-directory flush. A restart recovers queued work only with
the conversation history from the same atomic file; residual steers become
follow-ups because their target turn no longer exists. Goal restoration is
independent of queued-work recovery. `HistoryCheckpoint` carries the goal along
with history so the cross-channel display event for completion cannot race a
snapshot that still contains the old objective. Orderly exit flushes the writer
and reports write failures; a crash after enqueue but before replacement can
lose the newest accepted boundary while leaving the previous file valid.

The same rule begins before MCP discovery: recovery first admits a goal and,
only when queued work has a protocol-valid boundary, its matching history and
permission policy. A `DurableBoundary` containing that pre-agent state queues a
complete snapshot for every submission or F2 mutation accepted while discovery
is pending. The authoritative process-local queue updates immediately; disk
follows asynchronously under the crash window above. Once discovery finishes
or is skipped, the queue and admitted history are moved unchanged into the
ordinary controller.

Terminal actions explicitly report whether they changed the queue. Submission,
edit, delete, reclassification, reorder, and restore queue a complete snapshot;
composer editing, scrolling, resize, help navigation, and other display-only
input do not. In particular, a mouse-wheel burst requests neither a file nor
parent-directory `fsync`.

The transcript remains the source of truth for committed model context. Queue
previews are not rendered as user chat messages until claimed; otherwise a
deleted queued item would appear to have been sent.

## Terminal hygiene

The alternate screen, mouse capture, bracketed paste, cursor, and raw mode have
one cleanup path. Startup failures after any partial terminal initialization use
that path too, rather than returning with the terminal stranded in raw mode.
Copy mode temporarily disables only mouse capture; terminal cleanup disables it
again unconditionally, so exiting from either copy state is safe. OSC 52 copy
requests do not alter terminal input modes or the conversation runtime.
Ratatui display strings are sanitized at their entry boundary: ESC and other
control bytes become visible control pictures, while newlines remain newlines.
This applies to provider/model labels, assistant and tool output, queue text,
permission details, MCP descriptions, and editor previews. Conversation and
tool data are not rewritten.

## State invariants

1. At most one mutation-capable agent turn is active for a conversation.
2. Only the UI reactor reads terminal events or draws.
3. Only the agent mutates committed conversation history.
4. A prompt ID is in exactly one lifecycle state.
5. Follow-ups start FIFO, one turn at a time.
6. Tool-use and tool-result blocks are never left unpaired by a controlled
   interruption.
7. A permission response is applied only to its live request.
8. Outside explicit copy mode, queue, turn, token, and terminal-display changes
   are rendered on the bounded frame tick; no individual delta or scroll event
   forces an immediate frame. Copy mode accumulates those changes until its
   single exit redraw.
9. The named memory worker is the sole SQLite connection owner in one process.
   Settled-turn capture is sent over a FIFO and never blocks terminal polling
   or enters model-visible prompt construction.
10. At most one host-authored goal continuation is visible in the queue, and
    none remains after goal completion or an outcome that pauses autorun.
11. Spinner-visible background work cannot create steering: only a live agent
    turn sets `turn_active`, and the registry is finalized before such a turn
    can begin.
12. Effective and newly persisted name-level permission policy is disjoint;
    deny takes precedence over a contradictory legacy or externally shared
    entry. Session exact-input allows are excluded from persistence and cleared
    on load, reset, or clear.

## Alternatives rejected

### Background render thread plus `Arc<Mutex<Terminal>>`

This can animate a spinner, but a blocking input loop can hold the same mutex,
preventing agent events and permission prompts from progressing. Even with
shorter lock scopes, multiple terminal readers are fragile.

### `Arc<Mutex<Agent>>` with a spawned turn

The turn holds the mutex for its full duration, so save/load/model operations
still block. It also fights the provider abstraction's deliberately non-`Send`
futures and makes permission re-entry prone to deadlock.

### Dedicated OS thread and second Tokio runtime

This can work, but it adds cross-runtime shutdown, terminal-broker, and state
snapshot complexity without providing concurrency we need. The model and tools
remain sequential; the UI only needs concurrent polling. The episodic
prototype does use one plain blocking OS thread as the sole SQLite owner, but
it has no terminal access, no `Agent`, and no second Tokio runtime.

### One undifferentiated FIFO

It cannot express “correct the work before the next model call” separately from
“do this after the current task is complete.” pi and Codex both demonstrate
that users need the distinction.

### Injecting text as soon as a key is pressed

Provider requests are immutable once sent, and inserting between a tool use and
its result corrupts the protocol history. The next valid model boundary is the
earliest safe delivery point.

## Formal model and review

`spec/AsyncRuntime.tla` models the controller protocol and copy-mode terminal
ownership, while reasoning text and Ratatui layout remain hidden
payload/display state. `spec/MemoryRuntime.tla` separately models opt-in
settled-turn capture, the FIFO SQLite worker, failure/skip outcomes, immutable
live episodes, explicit live-store deletion, and the absence of automatic
retrieval. Their checked-in `.cfg` files supply the finite CI bounds. TLC checks queue
identity, single-turn ownership, delivery modes, safe steering, terminal
reasons, tool-result pairing, permission correlation, committed settlement,
stable ID ordering, weakly fair release of busy ownership, and eventual copy
mode exit under an explicit user-resumption fairness assumption. Because copy
mode makes permission keys intermittently unavailable, permission resolution
has its own strong-fairness assumption when the permission UI is available
infinitely often; the first version without that assumption produced a TLC
liveness counterexample. The memory model's worker fairness similarly assumes
that the OS and filesystem eventually return; Rust bounds SQLite lock waiting
but cannot force a failed kernel or disk to progress.

The model is not accepted as a proxy for inspecting Rust. The maintained
state/action/invariant refinement is in
[`docs/runtime-traceability.md`](runtime-traceability.md), and
[`CONTRIBUTING.md`](../CONTRIBUTING.md) requires contributors to repeat the
trace for runtime changes. `make traceability` ensures every checked action and
invariant remains represented; `make tla` runs TLC; `make check` runs both plus
the Rust and shell validation.

## Self-critique and remaining limits

- The model phase is the Rust program counter, not a mirrored runtime enum.
  This avoids a second authority but makes the source trace a necessary human
  review step.
- Token fragmentation cannot multiply same-task queue records: each consecutive
  preview is capped at 16 KiB and successful output is reconciled from committed
  history. Each committed chat/reasoning projection is independently capped at
  64 Ki characters. Structural events and checkpoints deliberately remain
  lossless rather than globally capacity-limited, so adversarial structural-event
  volume can still create display work.
- The default 1 MiB logical completion limit is a host safety policy, not a
  model-token claim. Library users asking models for exceptionally large output
  must raise `CompletionLimits` intentionally as well as any provider token
  request, accepting the larger memory and transport bounds.
- A permission modal temporarily owns keyboard input. The agent future and
  animation continue, but ordinary composition resumes after the decision.
- Copy mode intentionally suspends application input and visual updates so the
  host terminal can own selection. Runtime work continues, but a permission
  request that arrives in copy mode cannot be answered until `F3` or `Esc`
  resumes the display. TLA+ liveness assumes the user eventually resumes; the
  implementation cannot force that environmental action.
- OSC 52 is intentionally a best-effort terminal request. Generalist can prove
  it emitted one complete, bounded request but cannot observe whether the
  terminal accepted, ignored, truncated, or redirected the clipboard write.
- Conversation search intentionally covers the TUI's visible conversation
  entries, not tool-activity payloads, provider reasoning, or archived sessions
  outside the currently loaded conversation.
- Reasoning inspection is provider-dependent and is not a faithful transcript
  of hidden model computation. Providers may omit, summarize, redact, or
  transform reasoning before exposing it.
- MCP retry is explicit rather than a background health monitor. It recovers
  startup failures/skips within the configuration loaded for this process, but
  does not hot-reload edited configuration or replace a bridge whose transport
  fails after it was registered; those cases still require restart.
- Cooperative cancellation repairs history and kills subprocesses on drop
  where supported; it cannot establish whether an arbitrary external action
  completed before its future was dropped.
