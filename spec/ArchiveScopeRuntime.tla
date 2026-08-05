---------------------- MODULE ArchiveScopeRuntime ----------------------
\* Scope-routing model for saved conversations and permissioned archive reads.
\*
\* Startup chooses either a project scope or the explicit global scope once.
\* Conversation saves and episodic captures remain in that active scope.
\* Existing archives may contain records from every scope, but one
\* representative disclosed record can change only after the permission policy
\* authorizes a request with a matching explicit filter. Authorization may be
\* an interactive allow-once/always choice or a remembered allow-always policy;
\* both are permission-gate decisions. Batch results refine repeated
\* representative choices; Rust tests separately check every returned row's
\* storage/SQL predicate.

EXTENDS FiniteSets, TLC

CONSTANTS
    ScopeIds,
    HistoryIds,
    MemoryIds,
    ProjectScope,
    GlobalScope,
    OtherScope,
    NoScope,
    NoHistory,
    NoMemory,
    GlobalHistory,
    OtherHistory,
    GlobalMemory,
    OtherMemory

SearchKinds == {"history", "memory"}
SearchFilters == {"current", "global", "other_projects", "all"}

ASSUME /\ ScopeIds # {}
       /\ HistoryIds # {}
       /\ MemoryIds # {}
       /\ ProjectScope \in ScopeIds
       /\ GlobalScope \in ScopeIds
       /\ OtherScope \in ScopeIds
       /\ ProjectScope # GlobalScope
       /\ ProjectScope # OtherScope
       /\ GlobalScope # OtherScope
       /\ NoScope \notin ScopeIds
       /\ NoHistory \notin (ScopeIds \X HistoryIds)
       /\ NoHistory \notin ((ScopeIds \X HistoryIds) \X SearchFilters)
       /\ NoMemory \notin (ScopeIds \X MemoryIds)
       /\ NoMemory \notin ((ScopeIds \X MemoryIds) \X SearchFilters)
       /\ GlobalHistory \in HistoryIds
       /\ OtherHistory \in HistoryIds
       /\ GlobalMemory \in MemoryIds
       /\ OtherMemory \in MemoryIds

VARIABLES
    activeScope,
    globalWasExplicit,
    histories,
    memories,
    writtenHistories,
    capturedMemories,
    pendingKind,
    pendingFilter,
    disclosedHistory,
    historyDisclosureFilter,
    authorizedHistoryDisclosure,
    disclosedMemory,
    memoryDisclosureFilter,
    authorizedMemoryDisclosure

vars ==
    <<activeScope, globalWasExplicit, histories, memories,
      writtenHistories, capturedMemories, pendingKind, pendingFilter,
      disclosedHistory, historyDisclosureFilter,
      authorizedHistoryDisclosure, disclosedMemory, memoryDisclosureFilter,
      authorizedMemoryDisclosure>>

MatchesFilter(item, filter) ==
    \/ filter = "all"
    \/ /\ filter = "current"
       /\ item[1] = activeScope
    \/ /\ filter = "global"
       /\ item[1] = GlobalScope
    \/ /\ filter = "other_projects"
       /\ item[1] \notin {activeScope, GlobalScope}

Init ==
    /\ activeScope = NoScope
    /\ globalWasExplicit = FALSE
    /\ histories =
        {<<GlobalScope, GlobalHistory>>, <<OtherScope, OtherHistory>>}
    /\ memories =
        {<<GlobalScope, GlobalMemory>>, <<OtherScope, OtherMemory>>}
    /\ writtenHistories = {}
    /\ capturedMemories = {}
    /\ pendingKind = "none"
    /\ pendingFilter = "none"
    /\ disclosedHistory = NoHistory
    /\ historyDisclosureFilter = "none"
    /\ authorizedHistoryDisclosure = NoHistory
    /\ disclosedMemory = NoMemory
    /\ memoryDisclosureFilter = "none"
    /\ authorizedMemoryDisclosure = NoMemory

SelectProjectScope(scope) ==
    /\ scope \in ScopeIds \ {GlobalScope}
    /\ activeScope = NoScope
    /\ activeScope' = scope
    /\ globalWasExplicit' = FALSE
    /\ UNCHANGED <<histories, memories, writtenHistories, capturedMemories,
                   pendingKind, pendingFilter, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

SelectGlobalScope ==
    /\ activeScope = NoScope
    /\ activeScope' = GlobalScope
    /\ globalWasExplicit' = TRUE
    /\ UNCHANGED <<histories, memories, writtenHistories, capturedMemories,
                   pendingKind, pendingFilter, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

SaveHistory(history) ==
    /\ activeScope \in ScopeIds
    /\ history \in HistoryIds
    /\ LET item == <<activeScope, history>> IN
       /\ histories' = histories \cup {item}
       /\ writtenHistories' = writtenHistories \cup {item}
    /\ UNCHANGED <<activeScope, globalWasExplicit, memories, capturedMemories,
                   pendingKind, pendingFilter, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

ForgetHistory(history) ==
    /\ activeScope \in ScopeIds
    /\ history \in HistoryIds
    /\ LET item == <<activeScope, history>> IN
       /\ item \in histories
       /\ histories' = histories \ {item}
    /\ UNCHANGED <<activeScope, globalWasExplicit, memories,
                   writtenHistories, capturedMemories, pendingKind,
                   pendingFilter, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

CaptureMemory(memory) ==
    /\ activeScope \in ScopeIds
    /\ memory \in MemoryIds
    /\ LET item == <<activeScope, memory>> IN
       /\ memories' = memories \cup {item}
       /\ capturedMemories' = capturedMemories \cup {item}
    /\ UNCHANGED <<activeScope, globalWasExplicit, histories, writtenHistories,
                   pendingKind, pendingFilter, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

RequestSearch(kind, filter) ==
    /\ activeScope \in ScopeIds
    /\ kind \in SearchKinds
    /\ filter \in SearchFilters
    /\ pendingKind = "none"
    /\ pendingKind' = kind
    /\ pendingFilter' = filter
    /\ UNCHANGED <<activeScope, globalWasExplicit, histories, memories,
                   writtenHistories, capturedMemories, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

DenySearch ==
    /\ pendingKind \in SearchKinds
    /\ pendingKind' = "none"
    /\ pendingFilter' = "none"
    /\ UNCHANGED <<activeScope, globalWasExplicit, histories, memories,
                   writtenHistories, capturedMemories, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

ApproveEmptySearch ==
    /\ pendingKind \in SearchKinds
    /\ pendingFilter \in SearchFilters
    /\ pendingKind' = "none"
    /\ pendingFilter' = "none"
    /\ UNCHANGED <<activeScope, globalWasExplicit, histories, memories,
                   writtenHistories, capturedMemories, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure,
                   disclosedMemory, memoryDisclosureFilter,
                   authorizedMemoryDisclosure>>

ApproveHistorySearch(item) ==
    /\ pendingKind = "history"
    /\ pendingFilter \in SearchFilters
    /\ item \in histories
    /\ MatchesFilter(item, pendingFilter)
    /\ disclosedHistory' = item
    /\ historyDisclosureFilter' = pendingFilter
    /\ authorizedHistoryDisclosure' = <<item, pendingFilter>>
    /\ pendingKind' = "none"
    /\ pendingFilter' = "none"
    /\ UNCHANGED <<activeScope, globalWasExplicit, histories, memories,
                   writtenHistories, capturedMemories, disclosedMemory,
                   memoryDisclosureFilter, authorizedMemoryDisclosure>>

ApproveMemorySearch(item) ==
    /\ pendingKind = "memory"
    /\ pendingFilter \in SearchFilters
    /\ item \in memories
    /\ MatchesFilter(item, pendingFilter)
    /\ disclosedMemory' = item
    /\ memoryDisclosureFilter' = pendingFilter
    /\ authorizedMemoryDisclosure' = <<item, pendingFilter>>
    /\ pendingKind' = "none"
    /\ pendingFilter' = "none"
    /\ UNCHANGED <<activeScope, globalWasExplicit, histories, memories,
                   writtenHistories, capturedMemories, disclosedHistory,
                   historyDisclosureFilter, authorizedHistoryDisclosure>>

Next ==
    \/ \E scope \in ScopeIds : SelectProjectScope(scope)
    \/ SelectGlobalScope
    \/ \E history \in HistoryIds : SaveHistory(history)
    \/ \E history \in HistoryIds : ForgetHistory(history)
    \/ \E memory \in MemoryIds : CaptureMemory(memory)
    \/ \E kind \in SearchKinds, filter \in SearchFilters :
        RequestSearch(kind, filter)
    \/ DenySearch
    \/ ApproveEmptySearch
    \/ \E item \in histories : ApproveHistorySearch(item)
    \/ \E item \in memories : ApproveMemorySearch(item)

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ activeScope \in ScopeIds \cup {NoScope}
    /\ globalWasExplicit \in BOOLEAN
    /\ histories \subseteq (ScopeIds \X HistoryIds)
    /\ memories \subseteq (ScopeIds \X MemoryIds)
    /\ writtenHistories \subseteq (ScopeIds \X HistoryIds)
    /\ capturedMemories \subseteq (ScopeIds \X MemoryIds)
    /\ pendingKind \in SearchKinds \cup {"none"}
    /\ pendingFilter \in SearchFilters \cup {"none"}
    /\ disclosedHistory \in (ScopeIds \X HistoryIds) \cup {NoHistory}
    /\ historyDisclosureFilter \in SearchFilters \cup {"none"}
    /\ authorizedHistoryDisclosure
        \in ((ScopeIds \X HistoryIds) \X SearchFilters) \cup {NoHistory}
    /\ disclosedMemory \in (ScopeIds \X MemoryIds) \cup {NoMemory}
    /\ memoryDisclosureFilter \in SearchFilters \cup {"none"}
    /\ authorizedMemoryDisclosure
        \in ((ScopeIds \X MemoryIds) \X SearchFilters) \cup {NoMemory}

GlobalScopeIsExplicit ==
    (activeScope = GlobalScope) <=> globalWasExplicit

WritesStayInActiveScope ==
    /\ activeScope = NoScope
        => writtenHistories = {} /\ capturedMemories = {}
    /\ activeScope \in ScopeIds
        => /\ writtenHistories \subseteq ({activeScope} \X HistoryIds)
           /\ capturedMemories \subseteq ({activeScope} \X MemoryIds)

PermissionGatesDisclosure ==
    /\ (disclosedHistory = NoHistory)
        <=> (historyDisclosureFilter = "none")
    /\ (disclosedHistory = NoHistory)
        <=> (authorizedHistoryDisclosure = NoHistory)
    /\ disclosedHistory # NoHistory
        => authorizedHistoryDisclosure =
            <<disclosedHistory, historyDisclosureFilter>>
    /\ (disclosedMemory = NoMemory)
        <=> (memoryDisclosureFilter = "none")
    /\ (disclosedMemory = NoMemory)
        <=> (authorizedMemoryDisclosure = NoMemory)
    /\ disclosedMemory # NoMemory
        => authorizedMemoryDisclosure =
            <<disclosedMemory, memoryDisclosureFilter>>

DisclosureMatchesRequestedScope ==
    /\ disclosedHistory # NoHistory
        => MatchesFilter(disclosedHistory, historyDisclosureFilter)
    /\ disclosedMemory # NoMemory
        => MatchesFilter(disclosedMemory, memoryDisclosureFilter)

PendingSearchIsCorrelated ==
    (pendingKind = "none") <=> (pendingFilter = "none")

=============================================================================
