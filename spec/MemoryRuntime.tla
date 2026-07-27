-------------------------- MODULE MemoryRuntime --------------------------
\* A finite-state model of the explicit episodic-memory prototype.
\*
\* The async conversation controller can begin another turn while the
\* single-process SQLite worker drains settled episodes in FIFO order. Capture
\* is opt-in per project. A worker transaction either publishes one complete
\* immutable episode, skips it because capture is paused, or reports failure;
\* no partial record becomes live. Local memory mutations execute only while
\* no turn is active and after earlier capture requests have drained.
\*
\* Prompt retrieval is deliberately absent. promptMemories is ghost state
\* fixed to the empty set so future prompt injection cannot be added without
\* changing this model and its traceability review.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS EpisodeIds, NoEpisode

ASSUME /\ EpisodeIds # {}
       /\ NoEpisode \notin EpisodeIds

VARIABLES
    captureEnabled,
    activeEpisode,
    pendingEpisodes,
    settledEpisodes,
    liveEpisodes,
    skippedEpisodes,
    failedEpisodes,
    forgottenEpisodes,
    promptMemories

vars ==
    <<captureEnabled, activeEpisode, pendingEpisodes, settledEpisodes,
      liveEpisodes, skippedEpisodes, failedEpisodes, forgottenEpisodes,
      promptMemories>>

SeqSet(sequence) == {sequence[index] : index \in 1..Len(sequence)}

Unique(sequence) ==
    \A left, right \in 1..Len(sequence) :
        sequence[left] = sequence[right] => left = right

TerminalEpisodes ==
    liveEpisodes \cup skippedEpisodes \cup failedEpisodes \cup forgottenEpisodes

Init ==
    /\ captureEnabled = FALSE
    /\ activeEpisode = NoEpisode
    /\ pendingEpisodes = <<>>
    /\ settledEpisodes = {}
    /\ liveEpisodes = {}
    /\ skippedEpisodes = {}
    /\ failedEpisodes = {}
    /\ forgottenEpisodes = {}
    /\ promptMemories = {}

\* begin_turn owns a fresh episode ID but does not publish a retrievable draft.
StartTurn(episode) ==
    /\ episode \in EpisodeIds
    /\ activeEpisode = NoEpisode
    /\ episode \notin settledEpisodes
    /\ episode \notin SeqSet(pendingEpisodes)
    /\ activeEpisode' = episode
    /\ UNCHANGED <<captureEnabled, pendingEpisodes, settledEpisodes,
                   liveEpisodes, skippedEpisodes, failedEpisodes,
                   forgottenEpisodes, promptMemories>>

\* A protocol-valid terminal outcome queues exactly one complete host record.
SettleTurn ==
    /\ activeEpisode # NoEpisode
    /\ pendingEpisodes' = Append(pendingEpisodes, activeEpisode)
    /\ settledEpisodes' = settledEpisodes \cup {activeEpisode}
    /\ activeEpisode' = NoEpisode
    /\ UNCHANGED <<captureEnabled, liveEpisodes, skippedEpisodes,
                   failedEpisodes, forgottenEpisodes, promptMemories>>

RecordEpisode ==
    /\ Len(pendingEpisodes) > 0
    /\ captureEnabled
    /\ LET episode == Head(pendingEpisodes) IN
       liveEpisodes' = liveEpisodes \cup {episode}
    /\ pendingEpisodes' = Tail(pendingEpisodes)
    /\ UNCHANGED <<captureEnabled, activeEpisode, settledEpisodes,
                   skippedEpisodes, failedEpisodes, forgottenEpisodes,
                   promptMemories>>

SkipEpisode ==
    /\ Len(pendingEpisodes) > 0
    /\ ~captureEnabled
    /\ LET episode == Head(pendingEpisodes) IN
       skippedEpisodes' = skippedEpisodes \cup {episode}
    /\ pendingEpisodes' = Tail(pendingEpisodes)
    /\ UNCHANGED <<captureEnabled, activeEpisode, settledEpisodes,
                   liveEpisodes, failedEpisodes, forgottenEpisodes,
                   promptMemories>>

FailEpisode ==
    /\ Len(pendingEpisodes) > 0
    /\ LET episode == Head(pendingEpisodes) IN
       failedEpisodes' = failedEpisodes \cup {episode}
    /\ pendingEpisodes' = Tail(pendingEpisodes)
    /\ UNCHANGED <<captureEnabled, activeEpisode, settledEpisodes,
                   liveEpisodes, skippedEpisodes, forgottenEpisodes,
                   promptMemories>>

\* The command channel is FIFO. An idle pause/resume request cannot overtake
\* captures that the controller enqueued after earlier settled turns.
PauseCapture ==
    /\ activeEpisode = NoEpisode
    /\ pendingEpisodes = <<>>
    /\ captureEnabled
    /\ captureEnabled' = FALSE
    /\ UNCHANGED <<activeEpisode, pendingEpisodes, settledEpisodes,
                   liveEpisodes, skippedEpisodes, failedEpisodes,
                   forgottenEpisodes, promptMemories>>

ResumeCapture ==
    /\ activeEpisode = NoEpisode
    /\ pendingEpisodes = <<>>
    /\ ~captureEnabled
    /\ captureEnabled' = TRUE
    /\ UNCHANGED <<activeEpisode, pendingEpisodes, settledEpisodes,
                   liveEpisodes, skippedEpisodes, failedEpisodes,
                   forgottenEpisodes, promptMemories>>

\* /memory forget is a live-store deletion. forgottenEpisodes is ghost state
\* for this process/model run, not a claim about backups or snapshot restore.
ForgetEpisode(episode) ==
    /\ episode \in liveEpisodes
    /\ activeEpisode = NoEpisode
    /\ pendingEpisodes = <<>>
    /\ liveEpisodes' = liveEpisodes \ {episode}
    /\ forgottenEpisodes' = forgottenEpisodes \cup {episode}
    /\ UNCHANGED <<captureEnabled, activeEpisode, pendingEpisodes,
                   settledEpisodes, skippedEpisodes, failedEpisodes,
                   promptMemories>>

ProcessHead == RecordEpisode \/ SkipEpisode \/ FailEpisode

Next ==
    \/ \E episode \in EpisodeIds : StartTurn(episode)
    \/ SettleTurn
    \/ ProcessHead
    \/ PauseCapture
    \/ ResumeCapture
    \/ \E episode \in EpisodeIds : ForgetEpisode(episode)

Spec == Init /\ [][Next]_vars /\ WF_vars(ProcessHead)

TypeOK ==
    /\ captureEnabled \in BOOLEAN
    /\ activeEpisode \in EpisodeIds \cup {NoEpisode}
    /\ pendingEpisodes \in Seq(EpisodeIds)
    /\ settledEpisodes \subseteq EpisodeIds
    /\ liveEpisodes \subseteq EpisodeIds
    /\ skippedEpisodes \subseteq EpisodeIds
    /\ failedEpisodes \subseteq EpisodeIds
    /\ forgottenEpisodes \subseteq EpisodeIds
    /\ promptMemories \subseteq EpisodeIds

EpisodeIdentity ==
    /\ Unique(pendingEpisodes)
    /\ activeEpisode # NoEpisode => activeEpisode \notin settledEpisodes
    /\ SeqSet(pendingEpisodes) \subseteq settledEpisodes

SettledLifecycleIsTotal ==
    settledEpisodes =
        SeqSet(pendingEpisodes) \cup TerminalEpisodes

EpisodeLifecycleIsDisjoint ==
    /\ liveEpisodes \cap skippedEpisodes = {}
    /\ liveEpisodes \cap failedEpisodes = {}
    /\ liveEpisodes \cap forgottenEpisodes = {}
    /\ skippedEpisodes \cap failedEpisodes = {}
    /\ skippedEpisodes \cap forgottenEpisodes = {}
    /\ failedEpisodes \cap forgottenEpisodes = {}
    /\ SeqSet(pendingEpisodes) \cap TerminalEpisodes = {}

NoAutomaticRetrieval == promptMemories = {}

EveryPendingEpisodeResolves ==
    \A episode \in EpisodeIds :
        (episode \in SeqSet(pendingEpisodes))
            ~> (episode \notin SeqSet(pendingEpisodes))

=============================================================================
