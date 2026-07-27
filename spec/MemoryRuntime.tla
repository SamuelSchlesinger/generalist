-------------------------- MODULE MemoryRuntime --------------------------
\* A finite-state model of the explicit episodic-memory prototype.
\*
\* The async conversation controller can begin another turn while the
\* single-process SQLite worker drains settled episodes in FIFO order. One
\* handle has one immutable CurrentScope; capture is opt-in for that scope. A
\* worker transaction either publishes one complete immutable episode, skips it
\* because capture is paused, or reports failure;
\* no partial record becomes live. Local memory mutations execute only while
\* no turn is active and after earlier capture requests have drained.
\*
\* No archive is retrieved automatically. A model-requested search names
\* current, global, other-project, or all scopes; the permission policy must
\* authorize a request before any disclosure. Permissioned results are ghost
\* state for text made available to code mode/conversation history, never
\* instruction state. Cross-scope pre-existing rows and explicit global startup
\* are modeled in ArchiveScopeRuntime; this module follows lifecycle rows
\* produced by one current-scope worker.

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS EpisodeIds, ScopeIds, CurrentScope, GlobalScope, NoEpisode, NoScope

ASSUME /\ EpisodeIds # {}
       /\ ScopeIds # {}
       /\ CurrentScope \in ScopeIds
       /\ GlobalScope \in ScopeIds
       /\ CurrentScope # GlobalScope
       /\ NoEpisode \notin EpisodeIds
       /\ NoScope \notin ScopeIds

SearchFilters == {"current", "global", "other_projects", "all"}

VARIABLES
    captureEnabled,
    activeEpisode,
    activeScope,
    episodeScope,
    pendingEpisodes,
    settledEpisodes,
    liveEpisodes,
    skippedEpisodes,
    failedEpisodes,
    forgottenEpisodes,
    pendingSearch,
    authorizedByFilter,
    disclosedEpisodes

vars ==
    <<captureEnabled, activeEpisode, activeScope, episodeScope,
      pendingEpisodes, settledEpisodes,
      liveEpisodes, skippedEpisodes, failedEpisodes, forgottenEpisodes,
      pendingSearch, authorizedByFilter, disclosedEpisodes>>

SeqSet(sequence) == {sequence[index] : index \in 1..Len(sequence)}

Unique(sequence) ==
    \A left, right \in 1..Len(sequence) :
        sequence[left] = sequence[right] => left = right

TerminalEpisodes ==
    liveEpisodes \cup skippedEpisodes \cup failedEpisodes \cup forgottenEpisodes

MatchesFilter(episode, filter) ==
    \/ filter = "all"
    \/ /\ filter = "current"
       /\ episodeScope[episode] = CurrentScope
    \/ /\ filter = "global"
       /\ episodeScope[episode] = GlobalScope
    \/ /\ filter = "other_projects"
       /\ episodeScope[episode] \notin {CurrentScope, GlobalScope}

AuthorizedEpisodes ==
    UNION {authorizedByFilter[filter] : filter \in SearchFilters}

Init ==
    /\ captureEnabled = FALSE
    /\ activeEpisode = NoEpisode
    /\ activeScope = NoScope
    /\ episodeScope = [episode \in EpisodeIds |-> NoScope]
    /\ pendingEpisodes = <<>>
    /\ settledEpisodes = {}
    /\ liveEpisodes = {}
    /\ skippedEpisodes = {}
    /\ failedEpisodes = {}
    /\ forgottenEpisodes = {}
    /\ pendingSearch = "none"
    /\ authorizedByFilter = [filter \in SearchFilters |-> {}]
    /\ disclosedEpisodes = {}

\* begin_turn owns a fresh episode ID but does not publish a retrievable draft.
StartTurn(episode) ==
    /\ episode \in EpisodeIds
    /\ activeEpisode = NoEpisode
    /\ episode \notin settledEpisodes
    /\ episode \notin SeqSet(pendingEpisodes)
    /\ episodeScope[episode] = NoScope
    /\ activeEpisode' = episode
    /\ activeScope' = CurrentScope
    /\ episodeScope' = [episodeScope EXCEPT ![episode] = CurrentScope]
    /\ UNCHANGED <<captureEnabled, pendingEpisodes, settledEpisodes,
                   liveEpisodes, skippedEpisodes, failedEpisodes,
                   forgottenEpisodes, pendingSearch, authorizedByFilter,
                   disclosedEpisodes>>

\* A protocol-valid terminal outcome queues exactly one complete host record.
SettleTurn ==
    /\ activeEpisode # NoEpisode
    /\ pendingEpisodes' = Append(pendingEpisodes, activeEpisode)
    /\ settledEpisodes' = settledEpisodes \cup {activeEpisode}
    /\ activeEpisode' = NoEpisode
    /\ activeScope' = NoScope
    /\ UNCHANGED <<captureEnabled, episodeScope, liveEpisodes, skippedEpisodes,
                   failedEpisodes, forgottenEpisodes, pendingSearch,
                   authorizedByFilter, disclosedEpisodes>>

RecordEpisode ==
    /\ Len(pendingEpisodes) > 0
    /\ captureEnabled
    /\ LET episode == Head(pendingEpisodes) IN
       liveEpisodes' = liveEpisodes \cup {episode}
    /\ pendingEpisodes' = Tail(pendingEpisodes)
    /\ UNCHANGED <<captureEnabled, activeEpisode, activeScope, episodeScope,
                   settledEpisodes,
                   skippedEpisodes, failedEpisodes, forgottenEpisodes,
                   pendingSearch, authorizedByFilter, disclosedEpisodes>>

SkipEpisode ==
    /\ Len(pendingEpisodes) > 0
    /\ ~captureEnabled
    /\ LET episode == Head(pendingEpisodes) IN
       skippedEpisodes' = skippedEpisodes \cup {episode}
    /\ pendingEpisodes' = Tail(pendingEpisodes)
    /\ UNCHANGED <<captureEnabled, activeEpisode, activeScope, episodeScope,
                   settledEpisodes,
                   liveEpisodes, failedEpisodes, forgottenEpisodes,
                   pendingSearch, authorizedByFilter, disclosedEpisodes>>

FailEpisode ==
    /\ Len(pendingEpisodes) > 0
    /\ LET episode == Head(pendingEpisodes) IN
       failedEpisodes' = failedEpisodes \cup {episode}
    /\ pendingEpisodes' = Tail(pendingEpisodes)
    /\ UNCHANGED <<captureEnabled, activeEpisode, activeScope, episodeScope,
                   settledEpisodes,
                   liveEpisodes, skippedEpisodes, forgottenEpisodes,
                   pendingSearch, authorizedByFilter, disclosedEpisodes>>

\* The command channel is FIFO. An idle pause/resume request cannot overtake
\* captures that the controller enqueued after earlier settled turns.
PauseCapture ==
    /\ activeEpisode = NoEpisode
    /\ pendingEpisodes = <<>>
    /\ captureEnabled
    /\ captureEnabled' = FALSE
    /\ UNCHANGED <<activeEpisode, activeScope, episodeScope,
                   pendingEpisodes, settledEpisodes,
                   liveEpisodes, skippedEpisodes, failedEpisodes,
                   forgottenEpisodes, pendingSearch, authorizedByFilter,
                   disclosedEpisodes>>

ResumeCapture ==
    /\ activeEpisode = NoEpisode
    /\ pendingEpisodes = <<>>
    /\ ~captureEnabled
    /\ captureEnabled' = TRUE
    /\ UNCHANGED <<activeEpisode, activeScope, episodeScope,
                   pendingEpisodes, settledEpisodes,
                   liveEpisodes, skippedEpisodes, failedEpisodes,
                   forgottenEpisodes, pendingSearch, authorizedByFilter,
                   disclosedEpisodes>>

\* /memory forget is a live-store deletion. forgottenEpisodes is ghost state
\* for this process/model run, not a claim about backups or snapshot restore.
ForgetEpisode(episode) ==
    /\ episode \in liveEpisodes
    /\ activeEpisode = NoEpisode
    /\ pendingEpisodes = <<>>
    /\ episodeScope[episode] = CurrentScope
    /\ liveEpisodes' = liveEpisodes \ {episode}
    /\ forgottenEpisodes' = forgottenEpisodes \cup {episode}
    /\ UNCHANGED <<captureEnabled, activeEpisode, activeScope, episodeScope,
                   pendingEpisodes,
                   settledEpisodes, skippedEpisodes, failedEpisodes,
                   pendingSearch, authorizedByFilter, disclosedEpisodes>>

\* A model tool request alone discloses nothing. The next transition is an
\* permission-policy decision; denial clears the request unchanged. A remembered
\* allow-always policy refines another ApproveSearch without an interactive
\* modal.
RequestSearch(filter) ==
    /\ filter \in SearchFilters
    /\ pendingSearch = "none"
    /\ pendingSearch' = filter
    /\ UNCHANGED <<captureEnabled, activeEpisode, activeScope, episodeScope,
                   pendingEpisodes, settledEpisodes, liveEpisodes,
                   skippedEpisodes, failedEpisodes, forgottenEpisodes,
                   authorizedByFilter, disclosedEpisodes>>

DenySearch ==
    /\ pendingSearch \in SearchFilters
    /\ pendingSearch' = "none"
    /\ UNCHANGED <<captureEnabled, activeEpisode, activeScope, episodeScope,
                   pendingEpisodes, settledEpisodes, liveEpisodes,
                   skippedEpisodes, failedEpisodes, forgottenEpisodes,
                   authorizedByFilter, disclosedEpisodes>>

ApproveSearch(results) ==
    /\ pendingSearch \in SearchFilters
    /\ results \subseteq liveEpisodes
    /\ \A episode \in results : MatchesFilter(episode, pendingSearch)
    /\ authorizedByFilter' =
        [authorizedByFilter EXCEPT ![pendingSearch] = @ \cup results]
    /\ disclosedEpisodes' = disclosedEpisodes \cup results
    /\ pendingSearch' = "none"
    /\ UNCHANGED <<captureEnabled, activeEpisode, activeScope, episodeScope,
                   pendingEpisodes, settledEpisodes, liveEpisodes,
                   skippedEpisodes, failedEpisodes, forgottenEpisodes>>

ProcessHead == RecordEpisode \/ SkipEpisode \/ FailEpisode

Next ==
    \/ \E episode \in EpisodeIds : StartTurn(episode)
    \/ SettleTurn
    \/ ProcessHead
    \/ PauseCapture
    \/ ResumeCapture
    \/ \E episode \in EpisodeIds : ForgetEpisode(episode)
    \/ \E filter \in SearchFilters : RequestSearch(filter)
    \/ DenySearch
    \/ \E results \in SUBSET liveEpisodes : ApproveSearch(results)

Spec == Init /\ [][Next]_vars /\ WF_vars(ProcessHead)

TypeOK ==
    /\ captureEnabled \in BOOLEAN
    /\ activeEpisode \in EpisodeIds \cup {NoEpisode}
    /\ activeScope \in ScopeIds \cup {NoScope}
    /\ episodeScope \in [EpisodeIds -> ScopeIds \cup {NoScope}]
    /\ pendingEpisodes \in Seq(EpisodeIds)
    /\ settledEpisodes \subseteq EpisodeIds
    /\ liveEpisodes \subseteq EpisodeIds
    /\ skippedEpisodes \subseteq EpisodeIds
    /\ failedEpisodes \subseteq EpisodeIds
    /\ forgottenEpisodes \subseteq EpisodeIds
    /\ pendingSearch \in SearchFilters \cup {"none"}
    /\ authorizedByFilter \in [SearchFilters -> SUBSET EpisodeIds]
    /\ disclosedEpisodes \subseteq EpisodeIds

EpisodeIdentity ==
    /\ Unique(pendingEpisodes)
    /\ activeEpisode # NoEpisode => activeEpisode \notin settledEpisodes
    /\ (activeEpisode = NoEpisode) <=> (activeScope = NoScope)
    /\ activeEpisode # NoEpisode
        => episodeScope[activeEpisode] = activeScope
    /\ SeqSet(pendingEpisodes) \subseteq settledEpisodes
    /\ \A episode \in EpisodeIds :
        (episodeScope[episode] = NoScope)
            <=> (episode \notin settledEpisodes /\ episode # activeEpisode)

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

NoAutomaticRetrieval ==
    disclosedEpisodes = AuthorizedEpisodes

SearchDisclosureIsScoped ==
    \A filter \in SearchFilters :
        /\ authorizedByFilter[filter] \subseteq settledEpisodes
        /\ \A episode \in authorizedByFilter[filter] :
            MatchesFilter(episode, filter)

EveryPendingEpisodeResolves ==
    \A episode \in EpisodeIds :
        (episode \in SeqSet(pendingEpisodes))
            ~> (episode \notin SeqSet(pendingEpisodes))

=============================================================================
