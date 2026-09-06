//! Opt-in, payload-free events for checking implementation traces against TLA+.
//!
//! The trace is deliberately not a second runtime state machine. Components
//! emit an event only at their existing linearization points; TLC remains the
//! authority that decides whether the resulting action sequence is legal.
//! Prompt text, archive contents, tool inputs, and provider payloads never
//! enter this trace.

use crate::runtime::{DeliveryMode, QueuedPrompt};
use crate::scope::{ScopeFilter, WorkspaceScope};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// One of the checked TLA+ runtime models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    AsyncRuntime,
    MemoryRuntime,
    ArchiveScopeRuntime,
}

#[derive(Debug, Clone, Copy)]
struct EnabledModels {
    async_runtime: bool,
    memory_runtime: bool,
    archive_scope_runtime: bool,
}

impl EnabledModels {
    fn contains(self, kind: ModelKind) -> bool {
        match kind {
            ModelKind::AsyncRuntime => self.async_runtime,
            ModelKind::MemoryRuntime => self.memory_runtime,
            ModelKind::ArchiveScopeRuntime => self.archive_scope_runtime,
        }
    }
}

/// Observable actions in `spec/AsyncRuntime.tla`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum AsyncModelAction {
    Enqueue {
        prompt_id: u64,
        delivery: DeliveryMode,
    },
    DeleteQueued {
        index: usize,
    },
    ReclassifyQueued {
        index: usize,
        delivery: DeliveryMode,
    },
    MoveQueuedEarlier {
        index: usize,
    },
    DispatchFollowUp {
        prompt_id: u64,
    },
    CommitStart,
    RequeueStart,
    ProviderAnswer,
    ProviderRefusal,
    ProviderFailure,
    ProviderToolBatch {
        count: usize,
    },
    CompleteTool,
    AskPermission {
        request_id: String,
    },
    AllowPermission {
        request_id: String,
    },
    DenyPermission {
        request_id: String,
    },
    ClaimSteering {
        prompt_ids: Vec<u64>,
    },
    CommitSteering,
    RequeueSteering,
    ContinueAfterTools,
    SettleTurn,
    RequestCancel,
    RepairCancelledTool,
    FinishCancellation,
}

/// Observable actions in `spec/MemoryRuntime.tla`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum MemoryModelAction {
    StartTurn { episode_id: String },
    SettleTurn,
    RecordEpisode { episode_id: String },
    SkipEpisode { episode_id: String },
    FailEpisode { episode_id: String },
    PauseCapture,
    ResumeCapture,
    ForgetEpisode { episode_id: String },
    RequestSearch { filter: ScopeFilter },
    DenySearch,
    ApproveSearch { episode_ids: Vec<String> },
}

/// Observable actions in `spec/ArchiveScopeRuntime.tla`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ArchiveModelAction {
    SelectProjectScope { scope: String },
    SelectGlobalScope,
    SaveHistory { history_id: String },
    ForgetHistory { history_id: String },
    CaptureMemory { memory_id: String },
    RequestSearch { kind: String, filter: ScopeFilter },
    DenySearch,
    ApproveEmptySearch,
    ApproveHistorySearch { scope: String, history_id: String },
    ApproveMemorySearch { scope: String, memory_id: String },
}

/// Serializable trace bundle consumed by the TLC conformance checker.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelTraceSnapshot {
    pub async_runtime: Vec<AsyncModelAction>,
    /// Actual queue contents after the indicated number of async actions.
    #[serde(default)]
    pub async_queue: Vec<QueueObservation>,
    pub memory_runtime: Vec<MemoryModelAction>,
    pub archive_scope_runtime: Vec<ArchiveModelAction>,
}

/// A partial state observation, projected directly from the authoritative queue.
/// Only stable IDs and delivery modes are retained, in their actual order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueObservation {
    pub after_action: usize,
    pub prompts: Vec<(u64, DeliveryMode)>,
}

/// Cloneable event sink shared by the runtime, permission gate, and memory
/// worker during deterministic conformance runs.
#[derive(Debug, Clone)]
pub struct ModelTrace {
    enabled: EnabledModels,
    state: Arc<Mutex<TraceState>>,
}

#[derive(Debug, Default)]
struct TraceState {
    snapshot: ModelTraceSnapshot,
    scope_ids: HashMap<String, String>,
    history_ids: HashMap<String, String>,
    selected_archive_scope: Option<String>,
    next_request_id: u64,
}

impl ModelTrace {
    /// Observe only the selected models. Ordinary application construction
    /// does not install a trace at all.
    pub fn for_models(models: &[ModelKind]) -> Self {
        Self {
            enabled: EnabledModels {
                async_runtime: models.contains(&ModelKind::AsyncRuntime),
                memory_runtime: models.contains(&ModelKind::MemoryRuntime),
                archive_scope_runtime: models.contains(&ModelKind::ArchiveScopeRuntime),
            },
            state: Arc::new(Mutex::new(TraceState::default())),
        }
    }

    pub fn snapshot(&self) -> ModelTraceSnapshot {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .snapshot
            .clone()
    }

    pub(crate) fn records(&self, kind: ModelKind) -> bool {
        self.enabled.contains(kind)
    }

    pub(crate) fn record_async(&self, action: AsyncModelAction) {
        if self.records(ModelKind::AsyncRuntime) {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .snapshot
                .async_runtime
                .push(action);
        }
    }

    pub(crate) fn observe_queue(&self, items: &[QueuedPrompt]) {
        if self.records(ModelKind::AsyncRuntime) {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let after_action = state.snapshot.async_runtime.len();
            state.snapshot.async_queue.push(QueueObservation {
                after_action,
                prompts: items.iter().map(|item| (item.id, item.delivery)).collect(),
            });
        }
    }

    pub(crate) fn record_memory(&self, action: MemoryModelAction) {
        if self.records(ModelKind::MemoryRuntime) {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .snapshot
                .memory_runtime
                .push(action);
        }
    }

    pub(crate) fn record_archive(&self, mut action: ArchiveModelAction) {
        if self.records(ModelKind::ArchiveScopeRuntime) {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Saved-session names are user text. Preserve equality across
            // writes, reads, and deletion without retaining names in output.
            if let ArchiveModelAction::SaveHistory { history_id }
            | ArchiveModelAction::ForgetHistory { history_id }
            | ArchiveModelAction::ApproveHistorySearch { history_id, .. } = &mut action
            {
                let next = state.history_ids.len() + 1;
                *history_id = state
                    .history_ids
                    .entry(history_id.clone())
                    .or_insert_with(|| format!("history-{next}"))
                    .clone();
            }
            state.snapshot.archive_scope_runtime.push(action);
        }
    }

    /// Allocate a trace-local permission identity. Provider tool-use IDs are
    /// payload and need not be globally unique, while the TLA+ permission
    /// protocol requires a fresh request identity for every policy decision.
    pub(crate) fn next_request_id(&self) -> String {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let id = state.next_request_id;
        state.next_request_id = state
            .next_request_id
            .checked_add(1)
            .expect("model-trace permission ID space exhausted");
        format!("request-{id}")
    }

    /// Record the immutable startup scope once even when the history and
    /// memory clients are constructed independently with the same sink.
    pub(crate) fn record_scope_selection(&self, scope: &WorkspaceScope) {
        if !self.records(ModelKind::ArchiveScopeRuntime) {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (scope_id, action) = match scope {
            WorkspaceScope::Global => ("global".to_string(), ArchiveModelAction::SelectGlobalScope),
            WorkspaceScope::Project { .. } => {
                let label = scope.display_name();
                let next = state.scope_ids.len() + 1;
                let scope_id = state
                    .scope_ids
                    .entry(label)
                    .or_insert_with(|| format!("scope-{next}"))
                    .clone();
                (
                    scope_id.clone(),
                    ArchiveModelAction::SelectProjectScope { scope: scope_id },
                )
            }
        };
        if state.selected_archive_scope.as_ref() == Some(&scope_id) {
            return;
        }
        state.selected_archive_scope = Some(scope_id);
        state.snapshot.archive_scope_runtime.push(action);
    }

    /// Replace a potentially sensitive project-root label with a trace-local
    /// opaque identifier while preserving equality within this trace.
    pub(crate) fn scope_id(&self, label: &str) -> String {
        if label == "global" {
            return "global".to_string();
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let next = state.scope_ids.len() + 1;
        state
            .scope_ids
            .entry(label.to_string())
            .or_insert_with(|| format!("scope-{next}"))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traces_are_model_selective_and_project_labels_are_opaque() {
        let trace = ModelTrace::for_models(&[ModelKind::ArchiveScopeRuntime]);
        trace.record_async(AsyncModelAction::ProviderAnswer);
        trace.observe_queue(&[]);
        trace.record_archive(ArchiveModelAction::SelectProjectScope {
            scope: trace.scope_id("/sensitive/project/root"),
        });

        let snapshot = trace.snapshot();
        assert!(snapshot.async_runtime.is_empty());
        assert!(snapshot.async_queue.is_empty());
        assert_eq!(
            snapshot.archive_scope_runtime,
            vec![ArchiveModelAction::SelectProjectScope {
                scope: "scope-1".to_string(),
            }]
        );
        assert_eq!(trace.scope_id("/sensitive/project/root"), "scope-1");
        assert_eq!(trace.scope_id("/another/project"), "scope-2");
        assert_eq!(trace.scope_id("global"), "global");
    }

    #[test]
    fn queue_observations_read_actual_state_without_prompt_payloads() {
        let trace = ModelTrace::for_models(&[ModelKind::AsyncRuntime]);
        let mut items: Vec<_> = [17, 42]
            .into_iter()
            .map(|id| QueuedPrompt {
                id,
                text: "private prompt payload".into(),
                delivery: DeliveryMode::FollowUp,
                source: crate::runtime::PromptSource::User,
            })
            .collect();
        for item in &items {
            trace.record_async(AsyncModelAction::Enqueue {
                prompt_id: item.id,
                delivery: item.delivery,
            });
        }
        trace.observe_queue(&items);
        // Simulate incorrect contents behind an unchanged, legal-looking
        // action stream. Observation must read Rust, not replay those actions.
        items.swap(0, 1);
        items[0].delivery = DeliveryMode::Steer;
        items.pop();
        trace.observe_queue(&items);
        let snapshot = trace.snapshot();
        assert_eq!(snapshot.async_runtime.len(), 2);
        assert_eq!(
            snapshot.async_queue.last(),
            Some(&QueueObservation {
                after_action: 2,
                prompts: vec![(42, DeliveryMode::Steer)],
            })
        );
        assert!(!serde_json::to_string(&snapshot)
            .unwrap()
            .contains("private prompt payload"));
    }

    #[test]
    fn history_and_memory_clients_share_one_scope_selection() {
        let trace = ModelTrace::for_models(&[ModelKind::ArchiveScopeRuntime]);
        let scope = WorkspaceScope::Project {
            root: "/sensitive/project/root".into(),
        };
        trace.record_scope_selection(&scope);
        trace.record_scope_selection(&scope);

        assert_eq!(
            trace.snapshot().archive_scope_runtime,
            vec![ArchiveModelAction::SelectProjectScope {
                scope: "scope-1".to_string(),
            }]
        );
    }

    #[test]
    fn saved_names_are_redacted_consistently_across_archive_actions() {
        let trace = ModelTrace::for_models(&[ModelKind::ArchiveScopeRuntime]);
        let name = "confidential project notes".to_string();
        trace.record_archive(ArchiveModelAction::SaveHistory {
            history_id: name.clone(),
        });
        trace.record_archive(ArchiveModelAction::ApproveHistorySearch {
            scope: "scope-1".into(),
            history_id: name.clone(),
        });
        trace.record_archive(ArchiveModelAction::ForgetHistory {
            history_id: name.clone(),
        });
        let serialized = serde_json::to_string(&trace.snapshot()).unwrap();
        assert!(!serialized.contains(&name));
        assert_eq!(serialized.matches("history-1").count(), 3);
    }
}
