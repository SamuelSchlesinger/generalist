--------------------------- MODULE AsyncRuntime ---------------------------
\* A finite-state model of Generalist's asynchronous conversation runtime.
\*
\* The model deliberately describes the controller protocol rather than
\* terminal rendering or provider payloads.  It captures the places where the
\* Rust implementation crosses an await boundary:
\*
\*   * prompts are queued, atomically claimed, then committed or requeued;
\*   * steering is committed only at a history-valid boundary;
\*   * at most one conversation-mutating turn is active;
\*   * permission replies are correlated with a live request;
\*   * cancellation pairs every outstanding tool use before returning idle.
\*
\* Idle local commands (including goal edits, remembered-permission
\* list/reset/clear, and OSC 52 clipboard writes), goal text, terminal
\* rendering, conversation-search query/selection, reasoning payloads, and
\* other provider payloads are hidden data represented by stuttering steps.
\* Permission-policy mutation occurs with no live request and only changes
\* whether a later concrete tool use reaches AskPermission. `/copy select`
\* instead refines EnterCopyMode. Copy mode is
\* the one terminal-rendering state modeled
\* explicitly: it gates user input and therefore participates in the liveness
\* argument, while agent progress remains independent of it.
\*
\* MCP startup discovery and its cancellation/progress payloads are also
\* hidden. Before an Agent exists, accepted composer and queue-manager actions
\* refine the ordinary idle Enqueue/DeleteQueued/ReclassifyQueued/
\* MoveQueuedEarlier transitions; no steer can be created and DispatchFollowUp
\* remains disabled concretely until the registry is finalized or discovery is
\* explicitly skipped.
\*
\* See docs/async-tui.md for the corresponding implementation architecture.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    PromptIds,
    RequestIds,
    NoPrompt,
    NoRequest,
    MaxTools,
    MaxRounds,
    MaxFailures

ASSUME /\ PromptIds # {}
       /\ RequestIds # {}
       /\ NoPrompt \notin PromptIds
       /\ NoRequest \notin RequestIds
       /\ MaxTools \in Nat \ {0}
       /\ MaxRounds \in Nat \ {0}
       /\ MaxFailures \in Nat

Phases ==
    {"idle", "starting", "provider", "tools", "permission",
     "boundary", "claiming", "cancelling"}

DeliveryModes == {"steer", "followup"}
LifecycleStates == {"fresh", "queued", "claimed", "committed", "discarded"}
TerminalReasons == {"none", "answer", "refusal", "error", "denial", "limit"}

VARIABLES
    copyMode,
    phase,
    activeTurn,
    queue,
    delivery,
    lifecycle,
    claimedSteers,
    settledTurns,
    interruptedTurns,
    committedOrder,
    toolUses,
    toolResults,
    permission,
    permissionOwner,
    usedRequests,
    continuationNeeded,
    terminalReason,
    roundsLeft,
    failuresLeft

protocolVars ==
    <<phase, activeTurn, queue, delivery, lifecycle, claimedSteers,
      settledTurns, interruptedTurns, committedOrder, toolUses,
      toolResults, permission, permissionOwner, usedRequests,
      continuationNeeded, terminalReason, roundsLeft, failuresLeft>>

vars ==
    <<copyMode, phase, activeTurn, queue, delivery, lifecycle, claimedSteers,
      settledTurns, interruptedTurns, committedOrder, toolUses,
      toolResults, permission, permissionOwner, usedRequests,
      continuationNeeded, terminalReason, roundsLeft, failuresLeft>>

SeqSet(s) == {s[i] : i \in 1..Len(s)}

Unique(s) ==
    \A i, j \in 1..Len(s) : s[i] = s[j] => i = j

RECURSIVE KeepMode(_, _)
KeepMode(s, wanted) ==
    IF Len(s) = 0
    THEN <<>>
    ELSE (IF delivery[Head(s)] = wanted THEN <<Head(s)>> ELSE <<>>)
         \o KeepMode(Tail(s), wanted)

RECURSIVE DropMode(_, _)
DropMode(s, unwanted) ==
    IF Len(s) = 0
    THEN <<>>
    ELSE (IF delivery[Head(s)] = unwanted THEN <<>> ELSE <<Head(s)>>)
         \o DropMode(Tail(s), unwanted)

RemoveAt(s, index) ==
    [i \in 1..(Len(s) - 1) |->
        IF i < index THEN s[i] ELSE s[i + 1]]

SwapWithPrevious(s, index) ==
    [i \in 1..Len(s) |->
        IF i = index - 1
        THEN s[index]
        ELSE IF i = index THEN s[index - 1] ELSE s[i]]

QueuedSteers == KeepMode(queue, "steer")

ExpectedClaims ==
    IF phase = "starting"
    THEN {activeTurn}
    ELSE IF phase = "claiming" THEN SeqSet(claimedSteers) ELSE {}

NormalizeQueuedSteers ==
    [p \in PromptIds |->
        IF lifecycle[p] = "queued" /\ delivery[p] = "steer"
        THEN "followup"
        ELSE delivery[p]]

Init ==
    /\ copyMode = FALSE
    /\ phase = "idle"
    /\ activeTurn = NoPrompt
    /\ queue = <<>>
    /\ delivery = [p \in PromptIds |-> "unset"]
    /\ lifecycle = [p \in PromptIds |-> "fresh"]
    /\ claimedSteers = <<>>
    /\ settledTurns = {}
    /\ interruptedTurns = {}
    /\ committedOrder = <<>>
    /\ toolUses = 0
    /\ toolResults = 0
    /\ permission = NoRequest
    /\ permissionOwner = NoPrompt
    /\ usedRequests = {}
    /\ continuationNeeded = FALSE
    /\ terminalReason = "none"
    /\ roundsLeft = 0
    /\ failuresLeft = MaxFailures

\* Enqueue covers both user submissions and host-authored active-goal
\* continuations and is intentionally available in every runtime phase. A
\* concrete modal may temporarily accept only its own keys, so the TUI is a
\* refinement with a subset of the user transitions. The host continuation is
\* inserted after settlement with a fresh stable ID and "followup" delivery,
\* so it is also a concrete instance of this action. An idle enqueue is
\* necessarily a follow-up because there is no live turn to steer.
Enqueue(p, requestedMode) ==
    /\ p \in PromptIds
    /\ requestedMode \in DeliveryModes
    /\ lifecycle[p] = "fresh"
    /\ phase = "idle" => requestedMode = "followup"
    /\ queue' = Append(queue, p)
    /\ delivery' = [delivery EXCEPT ![p] = requestedMode]
    /\ lifecycle' = [lifecycle EXCEPT ![p] = "queued"]
    /\ UNCHANGED <<copyMode, phase, activeTurn, claimedSteers, settledTurns,
                   interruptedTurns, committedOrder, toolUses, toolResults,
                   permission, permissionOwner, usedRequests,
                   continuationNeeded, terminalReason, roundsLeft, failuresLeft>>

\* Queue-manager actions operate only on entries that are still visible.  If
\* an item has already been claimed, none of these guards can match it.
DeleteQueued(index) ==
    /\ index \in 1..Len(queue)
    /\ LET p == queue[index] IN
       /\ queue' = RemoveAt(queue, index)
       /\ delivery' = [delivery EXCEPT ![p] = "unset"]
       /\ lifecycle' = [lifecycle EXCEPT ![p] = "discarded"]
    /\ UNCHANGED <<copyMode, phase, activeTurn, claimedSteers, settledTurns,
                   interruptedTurns, committedOrder, toolUses, toolResults,
                   permission, permissionOwner, usedRequests,
                   continuationNeeded, terminalReason, roundsLeft, failuresLeft>>

ReclassifyQueued(index, requestedMode) ==
    /\ index \in 1..Len(queue)
    /\ requestedMode \in DeliveryModes
    /\ phase = "idle" => requestedMode = "followup"
    /\ LET p == queue[index] IN
       delivery' = [delivery EXCEPT ![p] = requestedMode]
    /\ UNCHANGED <<copyMode, phase, activeTurn, queue, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   toolUses, toolResults, permission, permissionOwner,
                   usedRequests, continuationNeeded, terminalReason, roundsLeft,
                   failuresLeft>>

MoveQueuedEarlier(index) ==
    /\ index \in 2..Len(queue)
    /\ queue' = SwapWithPrevious(queue, index)
    /\ UNCHANGED <<copyMode, phase, activeTurn, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   toolUses, toolResults, permission, permissionOwner,
                   usedRequests, continuationNeeded, terminalReason, roundsLeft,
                   failuresLeft>>

\* Native terminal copy mode releases mouse capture and freezes application
\* input/redraws. F3 and idle `/copy select` are concrete entry gestures; F3
\* and Esc are concrete exit gestures. It never owns or changes protocol state.
\* The UI reactor keeps polling AgentProgress, and weak fairness for
\* ExitCopyMode states the user assumption required by the liveness property.
EnterCopyMode ==
    /\ ~copyMode
    /\ copyMode' = TRUE
    /\ UNCHANGED protocolVars

ExitCopyMode ==
    /\ copyMode
    /\ copyMode' = FALSE
    /\ UNCHANGED protocolVars

\* The controller removes one follow-up before handing mutable conversation
\* ownership to an agent turn.
DispatchFollowUp ==
    /\ phase = "idle"
    /\ Len(queue) > 0
    /\ delivery[Head(queue)] = "followup"
    /\ LET p == Head(queue) IN
       /\ phase' = "starting"
       /\ activeTurn' = p
       /\ queue' = Tail(queue)
       /\ lifecycle' = [lifecycle EXCEPT ![p] = "claimed"]
    /\ roundsLeft' = MaxRounds
    /\ UNCHANGED <<copyMode, delivery, claimedSteers, settledTurns,
                   interruptedTurns, committedOrder, toolUses, toolResults,
                   permission, permissionOwner, usedRequests,
                   continuationNeeded, terminalReason, failuresLeft>>

CommitStart ==
    /\ phase = "starting"
    /\ phase' = "provider"
    /\ lifecycle' = [lifecycle EXCEPT ![activeTurn] = "committed"]
    /\ committedOrder' = Append(committedOrder, activeTurn)
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, claimedSteers,
                   settledTurns, interruptedTurns, toolUses, toolResults,
                   permission, permissionOwner, usedRequests,
                   continuationNeeded, terminalReason, roundsLeft, failuresLeft>>

\* Submission can fail before it is recorded in history.  The stable ID is
\* returned to the front and can be retried without duplicating the prompt.
RequeueStart ==
    /\ phase = "starting"
    /\ failuresLeft > 0
    /\ phase' = "idle"
    /\ queue' = <<activeTurn>> \o queue
    /\ delivery' = NormalizeQueuedSteers
    /\ lifecycle' = [lifecycle EXCEPT ![activeTurn] = "queued"]
    /\ activeTurn' = NoPrompt
    /\ roundsLeft' = 0
    /\ failuresLeft' = failuresLeft - 1
    /\ UNCHANGED <<copyMode, claimedSteers, settledTurns, interruptedTurns,
                   committedOrder, toolUses, toolResults, permission,
                   permissionOwner, usedRequests, continuationNeeded, terminalReason>>

\* A provider response is committed as a unit.  MaxRounds counts provider
\* responses, exactly like Agent::max_iterations.  Refusal is separate because
\* it settles immediately instead of accepting steering.
ProviderAnswer ==
    /\ phase = "provider"
    /\ roundsLeft > 0
    /\ phase' = "boundary"
    /\ continuationNeeded' = FALSE
    /\ terminalReason' = "answer"
    /\ roundsLeft' = roundsLeft - 1
    /\ toolUses' = 0
    /\ toolResults' = 0
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   permission, permissionOwner, usedRequests, failuresLeft>>

ProviderRefusal ==
    /\ phase = "provider"
    /\ roundsLeft > 0
    /\ phase' = "boundary"
    /\ continuationNeeded' = FALSE
    /\ terminalReason' = "refusal"
    /\ roundsLeft' = roundsLeft - 1
    /\ toolUses' = 0
    /\ toolResults' = 0
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   permission, permissionOwner, usedRequests, failuresLeft>>

\* A provider/API failure after retries commits no partial assistant response
\* and settles the current turn without consuming queued steering.
ProviderFailure ==
    /\ phase = "provider"
    /\ roundsLeft > 0
    /\ phase' = "boundary"
    /\ continuationNeeded' = FALSE
    /\ terminalReason' = "error"
    /\ roundsLeft' = roundsLeft - 1
    /\ toolUses' = 0
    /\ toolResults' = 0
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   permission, permissionOwner, usedRequests, failuresLeft>>

ProviderToolBatch(count) ==
    /\ phase = "provider"
    /\ roundsLeft > 0
    /\ count \in 1..MaxTools
    /\ phase' = "tools"
    /\ toolUses' = count
    /\ toolResults' = 0
    /\ continuationNeeded' = TRUE
    /\ roundsLeft' = roundsLeft - 1
    /\ terminalReason' =
        IF roundsLeft' = 0 THEN "limit" ELSE "none"
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   permission, permissionOwner, usedRequests, failuresLeft>>

CompleteTool ==
    /\ phase = "tools"
    /\ toolResults < toolUses
    /\ toolResults' = toolResults + 1
    /\ phase' = IF toolResults' = toolUses THEN "boundary" ELSE "tools"
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   toolUses, permission, permissionOwner, usedRequests,
                   continuationNeeded, terminalReason, roundsLeft, failuresLeft>>

AskPermission(requestId) ==
    /\ phase = "tools"
    /\ toolResults < toolUses
    /\ requestId \in RequestIds \ usedRequests
    /\ phase' = "permission"
    /\ permission' = requestId
    /\ permissionOwner' = activeTurn
    /\ usedRequests' = usedRequests \cup {requestId}
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   toolUses, toolResults, continuationNeeded, terminalReason, roundsLeft,
                   failuresLeft>>

AllowPermission(requestId) ==
    /\ ~copyMode
    /\ phase = "permission"
    /\ permission = requestId
    /\ permissionOwner = activeTurn
    /\ phase' = "tools"
    /\ permission' = NoRequest
    /\ permissionOwner' = NoPrompt
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   toolUses, toolResults, usedRequests, continuationNeeded,
                   terminalReason, roundsLeft, failuresLeft>>

\* A denied tool still gets a result block.  The model abstracts its contents
\* but keeps the same pairing rule as a successfully completed tool.
DenyPermission(requestId) ==
    /\ ~copyMode
    /\ phase = "permission"
    /\ permission = requestId
    /\ permissionOwner = activeTurn
    /\ permission' = NoRequest
    /\ permissionOwner' = NoPrompt
    /\ toolResults' = toolResults + 1
    /\ terminalReason' = "denial"
    /\ phase' = IF toolResults' = toolUses THEN "boundary" ELSE "tools"
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   toolUses, usedRequests, continuationNeeded, roundsLeft,
                   failuresLeft>>

\* Claim every steering item that exists at this safe boundary, preserving its
\* relative FIFO order while leaving follow-ups in the authoritative queue.
ClaimSteering ==
    /\ phase = "boundary"
    /\ Len(QueuedSteers) > 0
    /\ roundsLeft > 0
    /\ terminalReason \notin {"refusal", "error", "limit"}
    /\ phase' = "claiming"
    /\ claimedSteers' = QueuedSteers
    /\ queue' = DropMode(queue, "steer")
    /\ lifecycle' =
        [p \in PromptIds |->
            IF p \in SeqSet(QueuedSteers) THEN "claimed" ELSE lifecycle[p]]
    /\ UNCHANGED <<copyMode, activeTurn, delivery, settledTurns, interruptedTurns,
                   committedOrder, toolUses, toolResults, permission,
                   permissionOwner, usedRequests, continuationNeeded,
                   terminalReason, roundsLeft, failuresLeft>>

CommitSteering ==
    /\ phase = "claiming"
    /\ Len(claimedSteers) > 0
    /\ phase' = "provider"
    /\ lifecycle' =
        [p \in PromptIds |->
            IF p \in SeqSet(claimedSteers) THEN "committed" ELSE lifecycle[p]]
    /\ committedOrder' = committedOrder \o claimedSteers
    /\ claimedSteers' = <<>>
    /\ toolUses' = 0
    /\ toolResults' = 0
    /\ continuationNeeded' = FALSE
    /\ terminalReason' = "none"
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, settledTurns,
                   interruptedTurns, permission, permissionOwner,
                   usedRequests, roundsLeft, failuresLeft>>

RequeueSteering ==
    /\ phase = "claiming"
    /\ failuresLeft > 0
    /\ phase' = "boundary"
    /\ queue' = claimedSteers \o queue
    /\ lifecycle' =
        [p \in PromptIds |->
            IF p \in SeqSet(claimedSteers) THEN "queued" ELSE lifecycle[p]]
    /\ claimedSteers' = <<>>
    /\ failuresLeft' = failuresLeft - 1
    /\ UNCHANGED <<copyMode, activeTurn, delivery, settledTurns, interruptedTurns,
                   committedOrder, toolUses, toolResults, permission,
                   permissionOwner, usedRequests, continuationNeeded,
                   terminalReason, roundsLeft>>

ContinueAfterTools ==
    /\ phase = "boundary"
    /\ Len(QueuedSteers) = 0
    /\ continuationNeeded
    /\ terminalReason = "none"
    /\ roundsLeft > 0
    /\ phase' = "provider"
    /\ toolUses' = 0
    /\ toolResults' = 0
    /\ continuationNeeded' = FALSE
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   permission, permissionOwner, usedRequests, terminalReason,
                   roundsLeft, failuresLeft>>

SettleTurn ==
    /\ phase = "boundary"
    /\ terminalReason # "none"
    /\ \/ Len(QueuedSteers) = 0
       \/ roundsLeft = 0
       \/ terminalReason \in {"refusal", "error"}
    /\ phase' = "idle"
    /\ settledTurns' = settledTurns \cup {activeTurn}
    /\ activeTurn' = NoPrompt
    /\ delivery' = NormalizeQueuedSteers
    /\ roundsLeft' = 0
    /\ toolUses' = 0
    /\ toolResults' = 0
    /\ continuationNeeded' = FALSE
    /\ terminalReason' = "none"
    /\ UNCHANGED <<copyMode, queue, lifecycle, claimedSteers,
                   interruptedTurns, committedOrder, permission,
                   permissionOwner, usedRequests, failuresLeft>>

\* Escape is turn-scoped.  A live permission request is retired immediately,
\* while outstanding tool uses are repaired in subsequent cancelling steps.
RequestCancel ==
    /\ phase \in {"provider", "tools", "permission", "boundary"}
    /\ phase' = "cancelling"
    /\ permission' = NoRequest
    /\ permissionOwner' = NoPrompt
    /\ continuationNeeded' = FALSE
    /\ terminalReason' = "none"
    /\ UNCHANGED <<copyMode, activeTurn, queue, delivery, lifecycle, claimedSteers,
                   settledTurns, interruptedTurns, committedOrder,
                   toolUses, toolResults, usedRequests, roundsLeft,
                   failuresLeft>>

RepairCancelledTool ==
    /\ phase = "cancelling"
    /\ toolResults < toolUses
    /\ toolResults' = toolResults + 1
    /\ UNCHANGED <<copyMode, phase, activeTurn, queue, delivery, lifecycle,
                   claimedSteers, settledTurns, interruptedTurns,
                   committedOrder, toolUses, permission, permissionOwner,
                   usedRequests, continuationNeeded, terminalReason, roundsLeft,
                   failuresLeft>>

FinishCancellation ==
    /\ phase = "cancelling"
    /\ toolResults = toolUses
    /\ phase' = "idle"
    /\ interruptedTurns' = interruptedTurns \cup {activeTurn}
    /\ activeTurn' = NoPrompt
    /\ delivery' = NormalizeQueuedSteers
    /\ roundsLeft' = 0
    /\ toolUses' = 0
    /\ toolResults' = 0
    /\ continuationNeeded' = FALSE
    /\ terminalReason' = "none"
    /\ UNCHANGED <<copyMode, queue, lifecycle, claimedSteers, settledTurns,
                   committedOrder, permission, permissionOwner,
                   usedRequests, failuresLeft>>

\* Stale replies are absent from the state-changing relation: AllowPermission
\* and DenyPermission both require the exact live request and owning turn.

IdleWait ==
    /\ phase = "idle"
    /\ Len(queue) = 0
    /\ UNCHANGED vars

UserAction ==
    \/ \E p \in PromptIds, requestedMode \in DeliveryModes :
        Enqueue(p, requestedMode)
    \/ \E index \in 1..Len(queue) : DeleteQueued(index)
    \/ \E index \in 1..Len(queue), requestedMode \in DeliveryModes :
        ReclassifyQueued(index, requestedMode)
    \/ \E index \in 2..Len(queue) : MoveQueuedEarlier(index)
    \/ RequestCancel

AgentProgress ==
    \/ DispatchFollowUp
    \/ CommitStart
    \/ RequeueStart
    \/ ProviderAnswer
    \/ ProviderRefusal
    \/ ProviderFailure
    \/ \E count \in 1..MaxTools : ProviderToolBatch(count)
    \/ CompleteTool
    \/ \E requestId \in RequestIds : AskPermission(requestId)
    \/ \E requestId \in RequestIds : AllowPermission(requestId)
    \/ \E requestId \in RequestIds : DenyPermission(requestId)
    \/ ClaimSteering
    \/ CommitSteering
    \/ RequeueSteering
    \/ ContinueAfterTools
    \/ SettleTurn
    \/ RepairCancelledTool
    \/ FinishCancellation

\* Permission choices are user input. They are unavailable while copy mode owns
\* the terminal, so weak fairness of the combined AgentProgress action is not
\* enough: repeatedly entering copy mode could otherwise postpone a reply
\* forever. Strong fairness states the environmental assumption that a choice is
\* eventually made if the permission UI is available infinitely often.
PermissionResolution ==
    \E requestId \in RequestIds :
        AllowPermission(requestId) \/ DenyPermission(requestId)

Next ==
    \/ /\ ~copyMode
       /\ UNCHANGED copyMode
       /\ UserAction
    \/ /\ UNCHANGED copyMode
       /\ AgentProgress
    \/ IdleWait
    \/ EnterCopyMode
    \/ ExitCopyMode

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ WF_vars(AgentProgress)
    /\ WF_vars(ExitCopyMode)
    /\ SF_vars(PermissionResolution)

\* -------------------------------------------------------------------------
\* Safety properties checked by TLC

TypeOK ==
    /\ copyMode \in BOOLEAN
    /\ phase \in Phases
    /\ activeTurn \in PromptIds \cup {NoPrompt}
    /\ queue \in Seq(PromptIds)
    /\ delivery \in [PromptIds -> DeliveryModes \cup {"unset"}]
    /\ lifecycle \in [PromptIds -> LifecycleStates]
    /\ claimedSteers \in Seq(PromptIds)
    /\ settledTurns \subseteq PromptIds
    /\ interruptedTurns \subseteq PromptIds
    /\ committedOrder \in Seq(PromptIds)
    /\ toolUses \in 0..MaxTools
    /\ toolResults \in 0..MaxTools
    /\ permission \in RequestIds \cup {NoRequest}
    /\ permissionOwner \in PromptIds \cup {NoPrompt}
    /\ usedRequests \subseteq RequestIds
    /\ continuationNeeded \in BOOLEAN
    /\ terminalReason \in TerminalReasons
    /\ roundsLeft \in 0..MaxRounds
    /\ failuresLeft \in 0..MaxFailures

QueueIdentity ==
    /\ Unique(queue)
    /\ Unique(claimedSteers)
    /\ \A p \in PromptIds :
        (lifecycle[p] = "queued") <=> (p \in SeqSet(queue))
    /\ \A p \in PromptIds :
        (lifecycle[p] = "claimed") <=> (p \in ExpectedClaims)
    /\ SeqSet(queue) \cap SeqSet(claimedSteers) = {}

SingleTurnOwnership ==
    /\ (phase = "idle") <=> (activeTurn = NoPrompt)
    /\ (phase = "starting" => lifecycle[activeTurn] = "claimed")
    /\ (phase \notin {"idle", "starting"}
        => lifecycle[activeTurn] = "committed")
    /\ activeTurn \notin settledTurns \cup interruptedTurns
    /\ settledTurns \cap interruptedTurns = {}

DeliveryIsWellFormed ==
    /\ \A p \in PromptIds :
        lifecycle[p] \in {"queued", "claimed", "committed"}
        => delivery[p] \in DeliveryModes
    /\ \A p \in PromptIds :
        lifecycle[p] \in {"fresh", "discarded"}
        => delivery[p] = "unset"
    /\ phase = "idle"
        => \A p \in SeqSet(queue) : delivery[p] = "followup"
    /\ phase = "claiming"
        => \A p \in SeqSet(claimedSteers) : delivery[p] = "steer"

SafeSteeringBoundary ==
    phase = "claiming"
    => /\ toolResults = toolUses
       /\ permission = NoRequest
       /\ permissionOwner = NoPrompt
       /\ Len(claimedSteers) > 0
       /\ roundsLeft > 0
       /\ terminalReason \notin {"refusal", "error", "limit"}

TerminalReasonIsWellFormed ==
    /\ phase \in {"idle", "starting", "provider", "cancelling"}
        => terminalReason = "none"
    /\ terminalReason \in {"answer", "refusal", "error"}
        => ~continuationNeeded
    /\ terminalReason \in {"denial", "limit"}
        => continuationNeeded
    /\ terminalReason = "limit" => roundsLeft = 0
    /\ phase = "boundary" /\ terminalReason = "none"
        => continuationNeeded

ToolHistoryIsValid ==
    /\ toolResults <= toolUses
    /\ phase \in {"tools", "permission"} => toolResults < toolUses
    /\ phase \in {"idle", "starting", "provider", "boundary", "claiming"}
        => toolResults = toolUses
    /\ phase = "idle" => toolUses = 0

PermissionIsCorrelated ==
    /\ (permission # NoRequest)
        <=> (phase = "permission" /\ permissionOwner = activeTurn)
    /\ permission = NoRequest => permissionOwner = NoPrompt
    /\ permission # NoRequest => permission \in usedRequests

SettledPromptsAreCommitted ==
    /\ settledTurns \cup interruptedTurns
        \subseteq {p \in PromptIds : lifecycle[p] = "committed"}
    /\ \A p \in settledTurns \cup interruptedTurns :
        delivery[p] = "followup"

HistoryOrderHasStableIds ==
    /\ Unique(committedOrder)
    /\ SeqSet(committedOrder)
        = {p \in PromptIds : lifecycle[p] = "committed"}

\* Under weak fairness for controller progress and the finite CI bounds, every
\* busy period eventually releases conversation ownership.
EveryBusyPeriodSettles ==
    [](phase # "idle" => <>(phase = "idle"))

\* Copy mode deliberately suspends user input, but it is not a permanent
\* runtime state under the explicit user-resumption fairness assumption.
CopyModeEventuallyResumes ==
    [](copyMode => <>(~copyMode))

=============================================================================
