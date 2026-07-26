# Multi-agent Coordination Architecture

## Current verdict

Generalist is well factored as a **single-agent component**, but it is not
currently a multi-agent runtime.

Useful foundations already exist:

- `Agent` encapsulates provider, history, tool registry, goal, and one turn
  state machine;
- `AgentEvent` makes execution observable without coupling it to the TUI;
- `PromptQueue` has reversible local claims and safe steering boundaries;
- cancellation, permission decisions, and tool outcomes are structured;
- the current-thread reactor remains usable during one active turn; and
- code mode can orchestrate many tool calls without repeated model round trips.

The gaps are structural, not cosmetic:

- `main.rs` constructs exactly one `Agent`;
- `runtime.rs` explicitly describes its primitives as single-threaded and uses
  `Rc<RefCell<_>>`;
- `PromptId` is a process-local `u64`;
- one autosave owns history, goal, permissions, and queue;
- there is no durable agent/run/task identity;
- no mailbox, addressing, acknowledgment, causal reply, or delivery contract;
- no task DAG, assignment, lease, fencing, heartbeat, or orphan recovery;
- no capability delegation or per-agent permission provenance;
- no shared-artifact ownership, worktree isolation, patch handoff, or merge
  review;
- no peer status, steering, interruption, shutdown, or cost quota protocol; and
- the TLA+ model covers one active conversation owner, not a team.

Running several binaries in one directory would therefore create independent
agents contending over files and autosave paths, not a coordinated team.

## Lessons from current coding agents

The products surveyed expose three distinct patterns:

1. **Caller/subagent:** Codex and Claude Code can spawn isolated-context workers
   that return a result to a parent. This limits main-context pollution and
   works well for bounded read-heavy tasks [codex-subagents][codex-subagents]
   [claude-subagents][claude-subagents].
2. **Peer team:** Claude Code’s experimental agent teams add a lead, shared task
   list, mailboxes, direct peer messages, and independent contexts. Its
   documented limitations—lagging task state, weak resumption, slow shutdown,
   fixed leadership, and no nesting—show that messaging alone does not make
   lifecycle management trivial [claude-teams][claude-teams].
3. **Isolated workspace:** Claude Code recommends Git worktrees for parallel
   edits. Codex likewise cautions that parallel write-heavy workflows increase
   conflicts. pi’s core remains a small agent/session substrate; subagent
   orchestration is supplied through SDK/extensions rather than a shared
   team-state protocol [claude-worktrees][claude-worktrees]
   [pi-sdk][pi-sdk].

Generalist should implement caller/subagent first, but choose IDs and storage
that do not rule out peer communication later.

## Separation of planes

Do not put every concern into one “agent bus.”

### Control plane

Authoritative lifecycle and policy:

- agent/run registration and heartbeat;
- task creation, dependencies, claim/lease, completion, and cancellation;
- capability delegation and revocation;
- quotas and concurrency limits;
- review and merge decisions; and
- team shutdown and recovery.

### Message plane

Immutable addressed envelopes plus append-only lifecycle events:

- task delegation and clarification;
- progress, findings, warnings, and result summaries;
- follow-up, interruption, and cancellation notices;
- acknowledgment and causal reply links; and
- artifact/episode/candidate references rather than large copied payloads.

### Evidence and memory plane

The memory protocol defined elsewhere:

- private agent episodes;
- shared project episodes explicitly eligible for sharing;
- derived candidates and promoted project memory;
- provenance across messages and artifacts; and
- tombstones and scope-aware retrieval.

### Workspace plane

Source-tree and external side effects:

- read-only shared checkout for exploration;
- isolated Git worktree/branch for write-capable tasks;
- declared file/module ownership hints;
- commits or patch bundles as handoff artifacts;
- deterministic integration/review; and
- ordinary tool permissions for every worker.

SQLite can coordinate the first three planes. It cannot lock arbitrary
filesystem meaning or merge two edits safely.

## Identity and scope

Use opaque UUIDs rather than display names:

- `project_id`: stable local project identity, not just a path string;
- `team_id`: one coordinated campaign;
- `agent_id`: durable participant identity for a run lineage;
- `run_id`: one process/session incarnation;
- `task_id`: durable unit in the dependency graph;
- `attempt_id`: one leased execution of a task;
- `message_id`, `artifact_id`, and `episode_id`;
- `parent_agent_id` and `delegated_by`; and
- `policy_version` and `capability_set_id`.

Display names are mutable labels. Authorization, idempotency, lineage, and
message routing use IDs.

`project_id` comes from explicit supervisor-owned project registration, not a
model-supplied path or hash. Worktrees that resolve to the same registered Git
common directory inherit that ID and record a distinct checkout ID. A fresh
clone receives a new project ID unless the user explicitly links it. Canonical
paths, repository identity, and worktree membership are revalidated on each
write-capable task so symlink or directory replacement cannot silently change
scope.

Memory scopes are explicit:

- `agent_private`: not retrievable by peers;
- `team_shared`: visible only inside one team;
- `project_shared`: eligible across later teams after promotion;
- `user_private`: reusable for one principal across projects only by policy;
- `imported_untrusted`: shareable as data but never instructions.

An agent cannot promote its private reasoning or unreviewed notes into shared
memory merely by messaging another agent.

## Task state machine

Task state and attempt state are separate. Dependencies form an acyclic graph
validated transactionally at task creation; they freeze when the task first
becomes `ready`.

| Current task | Event and guard | Task result | Attempt result |
| --- | --- | --- | --- |
| `blocked` | last prerequisite completes | `ready` | none |
| `blocked` | all prerequisites are terminal and frozen policy is `all_terminal` | `ready` | none |
| `blocked` | prerequisite ends non-successfully under `all_success` or `manual` | `skipped`, `cancelled`, or `review_required` by frozen policy | none |
| `ready` | valid claim within retry/cost budget | `leased` | new `active` attempt and incremented fence |
| `leased` | current-fence heartbeat | `leased` | lease renewed |
| `leased` | current-fence success | `completed` | `succeeded` |
| `leased` | retryable failure with budget left | `ready` | `failed` |
| `leased` | terminal failure or exhausted budget | `failed` | `failed` |
| `leased` | lease expiry | `ready` or `failed` by retry budget | `orphaned`; fence advances |
| nonterminal | authorized cancellation | `cancelled` | active attempt cancelled; fence advances |
| `review_required` | authorized replan/resolution satisfies frozen policy | `ready` | none |
| `review_required` | authorized reject/cancel | `skipped` or `cancelled` | none |

Adding or changing dependencies after `ready` is rejected rather than moving a
task backward. A cancellation transaction advances the task fence, records a
reason, cancels active descendants whose policy requires ancestor success, and
leaves independent descendants blocked or reviewable. Retry policy fixes
maximum attempts, retryable classes, backoff, and cost before the first claim.
Dependency policy is fixed at creation: `all_success` skips/cancels descendants
after terminal prerequisite failure, `all_terminal` may proceed after all
prerequisites settle, and `manual` becomes `review_required`. Each terminal
failure transaction propagates or marks every affected descendant, so no
blocked node waits forever on a prerequisite that can no longer succeed.

A claim transaction checks:

- state is `ready`;
- the frozen dependency policy is satisfied;
- agent capability is a superset of task requirements;
- concurrency and cost quota permit another attempt; and
- no unexpired lease exists.

It increments a fencing generation and returns the generation with the lease.
Every progress, artifact, and completion write names that generation. A late
worker whose lease expired may append a diagnostic, but cannot complete the
task, publish shared memory, or merge artifacts.

Completion is idempotent. Dependencies unblock in the same transaction that
records a valid completion. A failed, orphaned, or cancelled attempt never
implies the task itself succeeded.

## Message contract

Messages are durable and at-least-once visible. Consumers deduplicate by
`message_id`; effects use separate idempotency keys. Each immutable envelope
contains:

- sender and recipient agent/team;
- task, attempt, and fencing generation where applicable;
- message kind and source/content class;
- creation time, optional expiry, and priority;
- causal parent and correlation ID;
- bounded inline body plus artifact references;
- sensitivity and capability-use ceiling.

Delivery, read, acknowledgment, rejection, expiry, and retry are separate
append-only events keyed by `message_id`. Current status is a projection over
those events; no consumer mutates the original envelope.

Message text is untrusted input. A peer cannot grant authority by writing “the
user approved this.” Approval references must resolve to host-owned policy
events. Cancellation and lease loss are control-plane records, not natural
language messages alone.

Backpressure is explicit. Senders receive `accepted`, `duplicate`, `quota_full`,
`recipient_closed`, or `policy_denied`; they do not assume delivery from a
successful local enqueue.

## Capability delegation

Child capability is the intersection of:

- parent’s current capability set;
- task-required maximum;
- agent-role policy;
- user-approved overrides; and
- workspace isolation policy.

Delegation can only narrow capability. Revocation increments a policy epoch;
later tool checks reject stale epochs. A worker may request additional
capability, but only the normal user-facing permission path can grant it.

Capabilities use a closed, host-owned algebra rather than free-form resource
predicates. File authority is expressed as pre-opened directory/file handles
plus allowed operations and no-follow resolution; network authority is a
canonical scheme/host/port/audience tuple rechecked after DNS and every
redirect; recipients and project/worktree IDs are opaque registered handles.
Subset checks are defined per type with explicit deny precedence. Aliases,
symlinks, wildcard expansion, and text asserted by an agent cannot widen a
grant.

Permission prompts show agent, task, worktree, requested action, and delegated
ceiling. Approving one worker’s tool name does not silently approve every
worker or future task.

## Workspace and merge protocol

Read-heavy agents may share the main checkout. A write-capable task receives:

1. a pinned base commit;
2. an isolated worktree and branch;
3. declared target paths as coordination hints, not security boundaries;
4. a task-scoped permission/capability set; and
5. an artifact manifest of changed paths, commit/patch, tests, and dirty state.

The integrator verifies the base, overlap, tests, generated artifacts, and user
changes before applying anything. It never uses `git add .`, destructive
checkout/reset, or implicit last-writer-wins. Same-file parallel editing is
serialized or deliberately merged under review.

Shared external systems need their own idempotency and authorization; a Git
worktree does not isolate API calls, databases, email, deployments, or secrets.

## Supervisor and SQLite concurrency protocol

One supervisor-owned local project database in WAL mode is adequate for a
bounded same-host team if:

- every process uses a patched SQLite version and normal SQLite locking;
- only the supervisor opens the database, WAL, FTS, ledger, and keys;
- workers authenticate over a Unix socket with peer credentials and
  task/attempt-bound session credentials;
- an OS sandbox or distinct Unix identity prevents worker tools from directly
  opening or locking supervisor state;
- write transactions are short and contain no model/tool wait;
- busy handling is bounded, jittered, visible, and idempotent;
- task leases use database time plus fencing generations;
- promotion and deletion verify source/tombstone epochs atomically;
- long snapshots close before model calls;
- checkpoint progress and WAL size are monitored; and
- the database never lives on a network filesystem.

WAL permits readers alongside one writer, not parallel writers. SQLite
documents both this limit and a now-fixed 2026 multi-connection WAL-reset race,
so version verification is part of startup [sqlite-wal][sqlite-wal].

Direct-open, forged-row, symlink replacement, WAL-copy, socket replay, and
lock-denial attempts are release tests. If workers share the supervisor’s Unix
identity and filesystem view, modes `0600`/`0700` are not isolation and shared
multi-agent memory stays disabled.

The worker credential is not a bearer string exposed to the model. A
challenge-response binds a worker supervisor session to Unix
peer PID/process-start identity, logical agent/run, project/task/attempt, current fencing
generation, capability set, policy epoch, expiry, socket nonce, and monotonic
request sequence. Requests are MACed and replay/out-of-order frames fail.
Session/socket descriptors are close-on-exec and stripped from tool subprocess
environments and file-descriptor tables. Lease loss, cancellation, exit,
policy change, or expiry revokes the session immediately.

The UI/controller uses a separate principal/project-bound controller session
without task, attempt, or fence authority. It may perform capture and explicit
idle `/memory` operations but cannot claim or complete worker tasks. Both
session types share peer/process-start, nonce, sequence, expiry, and
close-on-exec rules.

If write contention, distributed hosts, or operational governance outgrow this
contract, migrate the supervisor’s control plane. Stable message/task semantics
make that a backend change rather than a protocol rewrite.

## UI requirements

The TUI needs:

- team overview with active/idle/blocked/failed agents;
- task DAG/list with owner, dependencies, lease age, and result;
- per-agent thread inspection and unread/error badges;
- composer routing to main agent, selected worker, or team;
- steer, interrupt, cancel task, and shutdown controls;
- permission overlays labeled by agent/task;
- artifact/diff and candidate-memory review;
- queueing while any agents respond; and
- aggregate plus per-agent token/time/cost status.

Terminal input remains owned by one reactor. Agent/provider futures can be
polled on a `LocalSet`, but process-per-worker is the safer first write-capable
isolation on Unix. Worker crashes then do not take down the TUI supervisor.

## TLA+ scope

`CollaborationRuntime.tla` covers at least two agents, tasks, messages, lease
generations, capabilities, prompts, artifacts, and tombstones. Required
properties include:

- one live task attempt may publish;
- stale generations cannot complete or promote;
- frozen dependency policy is satisfied before claim and completion;
- terminal prerequisite failure cannot strand a dependent task;
- delegation never increases capability;
- cancellation and revocation eventually prevent new effects;
- acknowledged messages were durably accepted exactly once by ID;
- deletion wins against stale publication;
- project/team/private memory scope never widens implicitly;
- a queued user prompt eventually receives controller attention; and
- a crashed lease can eventually be reclaimed.

TLC will not prove that agents interpret messages correctly, patches merge
semantically, a lease duration fits real work, or peer content is harmless.
Traceability and adversarial multi-process tests remain mandatory.
`MemoryRuntime.tla` owns record lifecycle and epochs; `AsyncRuntime.tla` owns
terminal/prompt liveness. Their shared actions and integration tests are
specified in [the implementation handoff](implementation-handoff.md).

## Staged delivery

The collaboration work is sequenced in the unified milestone DAG in
[the implementation handoff](implementation-handoff.md), rather than an
independent roadmap.

The first useful milestone is not an autonomous swarm. It is two observable,
cancellable, read-only workers whose results are durably attributed and whose
failure cannot corrupt the main conversation or project.

## Local References

[claude-subagents]: Anthropic. “Create custom subagents.” Claude Code documentation. https://code.claude.com/docs/en/sub-agents (accessed 2026-07-26).

[claude-teams]: Anthropic. “Orchestrate teams of Claude Code sessions.” Claude Code documentation. https://code.claude.com/docs/en/agent-teams (accessed 2026-07-26).

[claude-worktrees]: Anthropic. “Run parallel sessions with worktrees.” Claude Code documentation. https://code.claude.com/docs/en/worktrees (accessed 2026-07-26).

[codex-subagents]: OpenAI. “Subagents.” Codex documentation. https://learn.chatgpt.com/docs/agent-configuration/subagents (accessed 2026-07-26).

[pi-sdk]: earendil-works contributors. “pi coding-agent SDK.” https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/sdk.md (accessed 2026-07-26).

[sqlite-wal]: SQLite Consortium. “Write-Ahead Logging.” https://www.sqlite.org/wal.html (accessed 2026-07-26).
