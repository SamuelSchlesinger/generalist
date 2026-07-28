//! Project-scoped durable conversation storage and explicit archive access.

use crate::scope::{ScopeFilter, WorkspaceScope};
use crate::tool::{DisclosureCapability, DisclosureGrant};
use crate::{
    history_tool_protocol_is_valid, ArchiveModelAction, ContentBlock, Error, MessageOrigin,
    ModelTrace, Result, SavedState,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const SCOPES_DIRECTORY: &str = "scopes";
const SCOPE_MANIFEST: &str = ".scope.json";
const PENDING_QUEUE_NAME: &str = "pending-queue";
const ARCHIVE_SEARCH_LIMIT: usize = 20;

#[derive(Debug, Serialize, Deserialize)]
struct ScopeManifest {
    version: u32,
    scope: WorkspaceScope,
}

/// One current runtime's scoped conversation store.
#[derive(Debug, Clone)]
pub struct HistoryStore {
    home: PathBuf,
    scope: WorkspaceScope,
    scope_directory: PathBuf,
    model_trace: Option<ModelTrace>,
}

impl HistoryStore {
    pub fn open(home: PathBuf, scope: WorkspaceScope) -> Result<Self> {
        Self::open_inner(home, scope, None)
    }

    #[doc(hidden)]
    pub fn open_with_model_trace(
        home: PathBuf,
        scope: WorkspaceScope,
        model_trace: ModelTrace,
    ) -> Result<Self> {
        Self::open_inner(home, scope, Some(model_trace))
    }

    fn open_inner(
        home: PathBuf,
        scope: WorkspaceScope,
        model_trace: Option<ModelTrace>,
    ) -> Result<Self> {
        let history_root = home.join(".generalist").join("history");
        let scopes_root = history_root.join(SCOPES_DIRECTORY);
        ensure_private_directory(&history_root)?;
        ensure_private_directory(&scopes_root)?;
        let scope_directory = scopes_root.join(scope.storage_key());
        ensure_private_directory(&scope_directory)?;
        let manifest = ScopeManifest {
            version: 1,
            scope: scope.clone(),
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|error| Error::Other(format!("Failed to serialize scope: {error}")))?;
        write_atomically(&scope_directory.join(SCOPE_MANIFEST), &manifest_bytes)?;

        let store = Self {
            home,
            scope,
            scope_directory,
            model_trace,
        };
        if let Some(trace) = &store.model_trace {
            trace.record_scope_selection(&store.scope);
        }
        Ok(store)
    }

    pub fn scope(&self) -> &WorkspaceScope {
        &self.scope
    }

    pub fn directory(&self) -> &Path {
        &self.scope_directory
    }

    pub(crate) fn model_trace(&self) -> Option<&ModelTrace> {
        self.model_trace.as_ref()
    }

    /// Save only into the active scope. State claiming another scope is
    /// rejected rather than silently moved.
    pub fn save(&self, state: &SavedState, filename: &str) -> Result<PathBuf> {
        validate_filename(filename)?;
        if state.scope != self.scope {
            return Err(Error::Other(format!(
                "Refusing to save {} state into {} scope",
                state.scope.display_name(),
                self.scope.display_name()
            )));
        }
        let path = self.scope_directory.join(format!("{filename}.json"));
        let bytes = serialize_state(state)?;
        write_atomically(&path, &bytes)?;
        if let Some(trace) = &self.model_trace {
            trace.record_archive(ArchiveModelAction::SaveHistory {
                history_id: filename.to_string(),
            });
        }
        Ok(path)
    }

    /// Load only from the active scoped directory.
    pub fn load(&self, filename: &str) -> Result<SavedState> {
        validate_filename(filename)?;
        for candidate in self.current_search_paths(filename) {
            if let Some(state) = read_state(&candidate.path, &candidate.scope)? {
                return Ok(state);
            }
        }
        Err(Error::Other(format!(
            "No saved conversation named '{filename}' in {} scope",
            self.scope.display_name()
        )))
    }

    pub fn list(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut names = Vec::new();
        for directory in self.current_search_directories() {
            for path in state_files_in(&directory.path) {
                let Some(name) = state_name(&path) else {
                    continue;
                };
                if seen.contains(&name) {
                    continue;
                }
                if read_state(&path, &directory.scope).ok().flatten().is_some() {
                    seen.insert(name.clone());
                    names.push(name);
                }
            }
        }
        names.sort();
        names
    }

    /// Permissioned-tool backend: search sanitized conversation text in an
    /// explicitly selected set of scopes.
    pub fn search_archives(
        &self,
        query: &str,
        filter: ScopeFilter,
        grant: &DisclosureGrant,
    ) -> Result<Vec<ConversationSummary>> {
        grant.ensure_search(DisclosureCapability::SearchConversations, query, filter)?;
        self.search_archives_impl(query, filter)
    }

    fn search_archives_impl(
        &self,
        query: &str,
        filter: ScopeFilter,
    ) -> Result<Vec<ConversationSummary>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(Error::Other(
                "Conversation search query cannot be empty".to_string(),
            ));
        }
        let needle = query.to_lowercase();
        let mut matches = Vec::new();
        for entry in self.archive_entries(filter)? {
            let text = searchable_state_text(&entry.state);
            if !text.to_lowercase().contains(&needle) {
                continue;
            }
            matches.push(ConversationSummary {
                id: entry.id,
                scope: entry.scope.display_name(),
                name: entry.name,
                updated_at: entry.updated_at,
                provider: entry.state.provider,
                model: entry.state.model,
                preview: matching_preview(&text, &needle, 220),
            });
        }
        matches.sort_by_key(|entry| std::cmp::Reverse(entry.updated_at));
        matches.truncate(ARCHIVE_SEARCH_LIMIT);
        Ok(matches)
    }

    /// Read one sanitized archived conversation by the opaque ID returned by
    /// [`Self::search_archives`]. `expected_scope` makes the permission prompt
    /// show which namespace the model intends to read and prevents substitution.
    pub fn read_archive(
        &self,
        id: &str,
        filter: ScopeFilter,
        expected_scope: &str,
        grant: &DisclosureGrant,
    ) -> Result<Option<ArchivedConversation>> {
        grant.ensure_read(
            DisclosureCapability::ReadConversation,
            id,
            filter,
            expected_scope,
        )?;
        self.read_archive_impl(id, filter, expected_scope)
    }

    fn read_archive_impl(
        &self,
        id: &str,
        filter: ScopeFilter,
        expected_scope: &str,
    ) -> Result<Option<ArchivedConversation>> {
        let parsed = Uuid::parse_str(id).map_err(|_| {
            Error::Other("Conversation ID must be a UUID from search results".into())
        })?;
        for entry in self.archive_entries(filter)? {
            if entry.id != parsed.to_string() {
                continue;
            }
            let actual_scope = entry.scope.display_name();
            if actual_scope != expected_scope {
                return Err(Error::Other(format!(
                    "Conversation scope mismatch: expected '{expected_scope}', found '{actual_scope}'"
                )));
            }
            let events = retained_conversation_events(&entry.state);
            return Ok(Some(ArchivedConversation {
                id: entry.id,
                scope: actual_scope,
                name: entry.name,
                updated_at: entry.updated_at,
                provider: entry.state.provider,
                model: entry.state.model,
                goal: entry.state.goal,
                events,
            }));
        }
        Ok(None)
    }

    fn current_search_paths(&self, filename: &str) -> Vec<StatePath> {
        self.current_search_directories()
            .into_iter()
            .map(|directory| StatePath {
                path: directory.path.join(format!("{filename}.json")),
                scope: directory.scope,
            })
            .collect()
    }

    fn current_search_directories(&self) -> Vec<StateDirectory> {
        vec![StateDirectory {
            path: self.scope_directory.clone(),
            scope: self.scope.clone(),
        }]
    }

    fn archive_entries(&self, filter: ScopeFilter) -> Result<Vec<ArchiveEntry>> {
        let mut directories = Vec::new();
        let scopes_root = self
            .home
            .join(".generalist")
            .join("history")
            .join(SCOPES_DIRECTORY);
        if let Ok(entries) = fs::read_dir(&scopes_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if is_symlink(&path) || !path.is_dir() {
                    continue;
                }
                let manifest_path = path.join(SCOPE_MANIFEST);
                if is_symlink(&manifest_path) {
                    continue;
                }
                let Ok(json) = fs::read_to_string(&manifest_path) else {
                    continue;
                };
                let Ok(manifest) = serde_json::from_str::<ScopeManifest>(&json) else {
                    continue;
                };
                if manifest.version != 1
                    || entry.file_name().to_str() != Some(manifest.scope.storage_key().as_str())
                {
                    continue;
                }
                if !filter.includes(&manifest.scope, &self.scope) {
                    continue;
                }
                directories.push(StateDirectory {
                    path,
                    scope: manifest.scope,
                });
            }
        }
        let mut seen_paths = HashSet::new();
        let mut archives = Vec::new();
        for directory in directories {
            for path in state_files_in(&directory.path) {
                if !seen_paths.insert(path.clone()) {
                    continue;
                }
                let Some(name) = state_name(&path) else {
                    continue;
                };
                let state = match read_state(&path, &directory.scope) {
                    Ok(Some(state)) => state,
                    Ok(None) | Err(_) => continue,
                };
                let metadata = fs::metadata(&path).map_err(|error| {
                    Error::Other(format!("Failed to inspect {}: {error}", path.display()))
                })?;
                let updated_at = metadata
                    .modified()
                    .map(DateTime::<Utc>::from)
                    .unwrap_or(DateTime::<Utc>::UNIX_EPOCH);
                let id =
                    Uuid::new_v5(&Uuid::NAMESPACE_URL, path.as_os_str().as_bytes()).to_string();
                archives.push(ArchiveEntry {
                    id,
                    scope: directory.scope.clone(),
                    name,
                    updated_at,
                    state,
                });
            }
        }
        Ok(archives)
    }
}

#[derive(Debug)]
struct StateDirectory {
    path: PathBuf,
    scope: WorkspaceScope,
}

#[derive(Debug)]
struct StatePath {
    path: PathBuf,
    scope: WorkspaceScope,
}

#[derive(Debug)]
struct ArchiveEntry {
    id: String,
    scope: WorkspaceScope,
    name: String,
    updated_at: DateTime<Utc>,
    state: SavedState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConversationSummary {
    pub id: String,
    pub scope: String,
    pub name: String,
    pub updated_at: DateTime<Utc>,
    pub provider: String,
    pub model: String,
    pub preview: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArchivedConversationEvent {
    UserText { text: String },
    AssistantText { text: String },
    ToolCall { name: String },
    ToolResult { is_error: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArchivedConversation {
    pub id: String,
    pub scope: String,
    pub name: String,
    pub updated_at: DateTime<Utc>,
    pub provider: String,
    pub model: String,
    /// Prospective instruction state, kept separate from past events.
    pub goal: Option<String>,
    pub events: Vec<ArchivedConversationEvent>,
}

fn retained_conversation_events(state: &SavedState) -> Vec<ArchivedConversationEvent> {
    let mut events = Vec::new();
    for message in &state.conversation_history {
        for block in &message.content {
            match block {
                ContentBlock::Text { text }
                    if !text.is_empty()
                        && message.role == "user"
                        && message.origin == MessageOrigin::Conversation =>
                {
                    events.push(ArchivedConversationEvent::UserText { text: text.clone() });
                }
                ContentBlock::Text { text } if !text.is_empty() && message.role == "assistant" => {
                    events.push(ArchivedConversationEvent::AssistantText { text: text.clone() });
                }
                ContentBlock::ToolUse { name, .. } => {
                    events.push(ArchivedConversationEvent::ToolCall { name: name.clone() });
                }
                ContentBlock::ToolResult { is_error, .. } => {
                    events.push(ArchivedConversationEvent::ToolResult {
                        is_error: is_error.unwrap_or(false),
                    });
                }
                ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. }
                | ContentBlock::Text { .. } => {}
            }
        }
    }
    events
}

fn searchable_state_text(state: &SavedState) -> String {
    let mut parts = vec![state.provider.clone(), state.model.clone()];
    if let Some(goal) = &state.goal {
        parts.push(format!(
            "[prospective goal from archived state; not a past event] {goal}"
        ));
    }
    for event in retained_conversation_events(state) {
        match event {
            ArchivedConversationEvent::UserText { text }
            | ArchivedConversationEvent::AssistantText { text } => parts.push(text),
            ArchivedConversationEvent::ToolCall { name } => parts.push(name),
            ArchivedConversationEvent::ToolResult { .. } => {}
        }
    }
    parts.join("\n")
}

fn matching_preview(text: &str, needle: &str, limit: usize) -> String {
    let source = text
        .lines()
        .find(|line| line.to_lowercase().contains(needle))
        .unwrap_or(text);
    let normalized = source.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let preview: String = chars.by_ref().take(limit).collect();
    let suffix = if chars.next().is_some() { "…" } else { "" };
    format!("{preview}{suffix}")
}

fn state_files_in(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| !is_symlink(path) && path.is_file())
        .filter(|path| state_name(path).is_some())
        .collect()
}

fn state_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    if name == SCOPE_MANIFEST {
        return None;
    }
    let stem = name.strip_suffix(".json")?;
    if stem == PENDING_QUEUE_NAME || stem.is_empty() {
        return None;
    }
    Some(stem.to_string())
}

fn read_state(path: &Path, location_scope: &WorkspaceScope) -> Result<Option<SavedState>> {
    if is_symlink(path) {
        return Err(Error::Other(format!(
            "Refusing to read symlinked conversation {}",
            path.display()
        )));
    }
    let json = match fs::read_to_string(path) {
        Ok(json) => json,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::IsADirectory => return Ok(None),
        Err(error) => {
            return Err(Error::Other(format!(
                "Failed to read {}: {error}",
                path.display()
            )))
        }
    };
    let state = SavedState::from_json(&json)
        .map_err(|error| Error::Other(format!("Failed to parse {}: {error}", path.display())))?;
    if state.scope != *location_scope {
        return Err(Error::Other(format!(
            "Conversation {} claims {} scope but is stored in {} scope",
            path.display(),
            state.scope.display_name(),
            location_scope.display_name()
        )));
    }
    Ok(Some(state))
}

fn serialize_state(state: &SavedState) -> Result<Vec<u8>> {
    if !history_tool_protocol_is_valid(&state.conversation_history) {
        return Err(Error::Other(
            "Refusing to persist history with an unpaired tool use/result".to_string(),
        ));
    }
    serde_json::to_vec_pretty(state)
        .map_err(|error| Error::Other(format!("Failed to serialize state: {error}")))
}

fn validate_filename(filename: &str) -> Result<()> {
    let filename = filename.trim();
    if filename.is_empty()
        || filename == "."
        || filename == ".."
        || filename == SCOPE_MANIFEST.trim_end_matches(".json")
        || filename.contains('/')
        || filename.contains('\0')
    {
        return Err(Error::Other(
            "Conversation names must be non-empty file names without '/'".to_string(),
        ));
    }
    Ok(())
}

fn ensure_private_directory(path: &Path) -> Result<()> {
    reject_symlink(path, "history directory")?;
    fs::create_dir_all(path).map_err(|error| {
        Error::Other(format!(
            "Failed to create history directory {}: {error}",
            path.display()
        ))
    })?;
    reject_symlink(path, "history directory")?;
    if !path.is_dir() {
        return Err(Error::Other(format!(
            "History path {} is not a directory",
            path.display()
        )));
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|error| {
        Error::Other(format!(
            "Failed to restrict history directory {}: {error}",
            path.display()
        ))
    })
}

fn write_atomically(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Other(format!("{} has no parent directory", path.display())))?;
    reject_symlink(path, "history file")?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".generalist-state-")
        .tempfile_in(parent)
        .map_err(|error| Error::Other(format!("Failed to create state file: {error}")))?;
    temporary
        .write_all(contents)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| Error::Other(format!("Failed to flush state file: {error}")))?;
    temporary
        .as_file()
        .set_permissions(fs::Permissions::from_mode(0o600))
        .map_err(|error| Error::Other(format!("Failed to restrict state file: {error}")))?;
    temporary.persist(path).map_err(|error| {
        Error::Other(format!(
            "Failed to replace {}: {}",
            path.display(),
            error.error
        ))
    })?;
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| Error::Other(format!("Failed to flush {}: {error}", parent.display())))
}

fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
}

fn reject_symlink(path: &Path, description: &str) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(Error::Other(format!(
            "Refusing to use symlinked {description} {}",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::Other(format!(
            "Failed to inspect {description} {}: {error}",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentBlock, Message};

    fn project_scope(temp: &tempfile::TempDir, name: &str) -> WorkspaceScope {
        let project = temp.path().join(name);
        fs::create_dir_all(project.join(".git")).unwrap();
        WorkspaceScope::discover(&project).unwrap()
    }

    fn state(scope: WorkspaceScope, text: &str) -> SavedState {
        let mut state = SavedState::new(scope, "openai".into(), "test-model".into());
        state.conversation_history.push(Message::user_text(text));
        state
            .conversation_history
            .push(Message::assistant(vec![ContentBlock::Text {
                text: format!("answer to {text}"),
            }]));
        state
    }

    #[test]
    fn project_autosaves_are_isolated_and_global_does_not_fallback() {
        let temp = tempfile::tempdir().unwrap();
        let first_scope = project_scope(&temp, "first");
        let second_scope = project_scope(&temp, "second");
        let first = HistoryStore::open(temp.path().to_path_buf(), first_scope.clone()).unwrap();
        let second = HistoryStore::open(temp.path().to_path_buf(), second_scope.clone()).unwrap();
        let global = HistoryStore::open(temp.path().to_path_buf(), WorkspaceScope::Global).unwrap();

        first
            .save(&state(first_scope, "first-only"), "autosave")
            .unwrap();
        second
            .save(&state(second_scope, "second-only"), "autosave")
            .unwrap();

        assert_eq!(
            first.load("autosave").unwrap().conversation_history[0].text(),
            "first-only"
        );
        assert_eq!(
            second.load("autosave").unwrap().conversation_history[0].text(),
            "second-only"
        );
        assert!(global.load("autosave").is_err());
        assert_ne!(first.directory(), second.directory());
    }

    #[test]
    fn legacy_flat_history_is_ignored_by_every_new_scope() {
        let temp = tempfile::tempdir().unwrap();
        let legacy_dir = temp.path().join(".generalist/history");
        fs::create_dir_all(&legacy_dir).unwrap();
        let legacy = SavedState::new(
            WorkspaceScope::Global,
            "openai".into(),
            "legacy-model".into(),
        );
        fs::write(
            legacy_dir.join("old.json"),
            serde_json::to_vec(&legacy).unwrap(),
        )
        .unwrap();

        let project =
            HistoryStore::open(temp.path().to_path_buf(), project_scope(&temp, "project")).unwrap();
        let global = HistoryStore::open(temp.path().to_path_buf(), WorkspaceScope::Global).unwrap();
        assert!(project.load("old").is_err());
        assert!(global.load("old").is_err());
        assert_eq!(
            fs::read(legacy_dir.join("old.json")).unwrap(),
            serde_json::to_vec(&legacy).unwrap()
        );
    }

    #[test]
    fn archive_search_requires_an_explicit_scope_filter_and_reads_by_opaque_id() {
        let temp = tempfile::tempdir().unwrap();
        let first_scope = project_scope(&temp, "first");
        let second_scope = project_scope(&temp, "second");
        let first = HistoryStore::open(temp.path().to_path_buf(), first_scope.clone()).unwrap();
        let second = HistoryStore::open(temp.path().to_path_buf(), second_scope.clone()).unwrap();
        let global = HistoryStore::open(temp.path().to_path_buf(), WorkspaceScope::Global).unwrap();
        first
            .save(&state(first_scope.clone(), "current needle"), "current")
            .unwrap();
        second
            .save(&state(second_scope.clone(), "foreign needle"), "foreign")
            .unwrap();
        global
            .save(&state(WorkspaceScope::Global, "global needle"), "global")
            .unwrap();

        let current = first
            .search_archives_impl("needle", ScopeFilter::Current)
            .unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].scope, first_scope.display_name());
        let other = first
            .search_archives_impl("needle", ScopeFilter::OtherProjects)
            .unwrap();
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].scope, second_scope.display_name());
        let global_current = global
            .search_archives_impl("needle", ScopeFilter::Current)
            .unwrap();
        assert_eq!(global_current.len(), 1);
        assert_eq!(global_current[0].scope, "global");
        let global_other = global
            .search_archives_impl("needle", ScopeFilter::OtherProjects)
            .unwrap();
        assert_eq!(global_other.len(), 2);
        assert!(global_other.iter().all(|entry| entry.scope != "global"));

        let archive = first
            .read_archive_impl(&other[0].id, ScopeFilter::OtherProjects, &other[0].scope)
            .unwrap()
            .unwrap();
        assert!(matches!(
            archive.events.first(),
            Some(ArchivedConversationEvent::UserText { text }) if text == "foreign needle"
        ));
        assert!(first
            .read_archive_impl(
                &other[0].id,
                ScopeFilter::OtherProjects,
                &first_scope.display_name(),
            )
            .is_err());
        assert!(first
            .read_archive_impl(&other[0].id, ScopeFilter::Current, &other[0].scope)
            .unwrap()
            .is_none());
    }

    #[test]
    fn archived_conversations_omit_reasoning_and_tool_payloads() {
        let temp = tempfile::tempdir().unwrap();
        let scope = project_scope(&temp, "project");
        let store = HistoryStore::open(temp.path().to_path_buf(), scope.clone()).unwrap();
        let mut saved = state(scope, "safe text");
        saved.conversation_history.push(Message::assistant(vec![
            ContentBlock::Thinking {
                thinking: "private reasoning".into(),
                signature: "signature".into(),
            },
            ContentBlock::ToolUse {
                name: "bash".into(),
                input: serde_json::json!({"secret": "tool input"}),
                id: "tool-id".into(),
            },
        ]));
        saved
            .conversation_history
            .push(Message::user(vec![ContentBlock::ToolResult {
                content: "tool output".into(),
                tool_use_id: "tool-id".into(),
                is_error: Some(false),
            }]));
        store.save(&saved, "safe").unwrap();
        let result = store
            .search_archives_impl("safe text", ScopeFilter::Current)
            .unwrap();
        let archive = store
            .read_archive_impl(&result[0].id, ScopeFilter::Current, &result[0].scope)
            .unwrap()
            .unwrap();
        let json = serde_json::to_string(&archive).unwrap();
        assert!(json.contains("\"name\":\"bash\""));
        for excluded in [
            "private reasoning",
            "signature",
            "tool input",
            "tool output",
            "tool-id",
        ] {
            assert!(!json.contains(excluded), "retained {excluded}");
        }
    }

    #[test]
    fn save_rejects_scope_mismatch_and_path_traversal() {
        let temp = tempfile::tempdir().unwrap();
        let scope = project_scope(&temp, "project");
        let store = HistoryStore::open(temp.path().to_path_buf(), scope.clone()).unwrap();
        assert!(store
            .save(
                &SavedState::new(WorkspaceScope::Global, "openai".into(), "model".into()),
                "wrong",
            )
            .is_err());
        let scoped = SavedState::new(scope, "openai".into(), "model".into());
        assert!(store.save(&scoped, "../escape").is_err());
    }
}
