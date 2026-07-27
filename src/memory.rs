//! Host-owned, explicit episodic memory.
//!
//! Capture is opt-in per scope, records only settled conversation text and tool
//! names/outcomes, and never injects records into provider prompts. Explicit
//! model-facing search/read tools sit above this store and pass through the
//! ordinary permission gate. A dedicated worker thread is the sole SQLite
//! connection owner so database work cannot block the current-thread TUI
//! reactor.

pub use crate::scope::discover_project_root;
use crate::scope::{ScopeFilter, WorkspaceScope};
use crate::{ContentBlock, Error, Message, MessageOrigin, Result, TurnOutcome};
use chrono::{DateTime, SecondsFormat, Utc};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

const SCHEMA_VERSION: i64 = 1;
const MIN_SQLITE_VERSION: (u32, u32, u32) = (3, 51, 3);
const SEARCH_LIMIT: usize = 20;

/// One retained event in an immutable episode.
///
/// Tool inputs, tool-result content, provider reasoning, signatures, and
/// redacted-reasoning payloads are deliberately absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EpisodeEvent {
    UserText { text: String },
    AssistantText { text: String },
    ToolCall { name: String },
    ToolResult { is_error: bool },
}

/// Host-observed terminal outcome for an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeOutcome {
    Completed,
    PausedOnDenial,
    MaxIterationsReached,
    Refused,
    Interrupted,
    Error,
}

impl EpisodeOutcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::PausedOnDenial => "paused_on_denial",
            Self::MaxIterationsReached => "max_iterations_reached",
            Self::Refused => "refused",
            Self::Interrupted => "interrupted",
            Self::Error => "error",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "completed" => Ok(Self::Completed),
            "paused_on_denial" => Ok(Self::PausedOnDenial),
            "max_iterations_reached" => Ok(Self::MaxIterationsReached),
            "refused" => Ok(Self::Refused),
            "interrupted" => Ok(Self::Interrupted),
            "error" => Ok(Self::Error),
            other => Err(Error::Other(format!(
                "Unknown stored episode outcome '{other}'"
            ))),
        }
    }
}

impl From<TurnOutcome> for EpisodeOutcome {
    fn from(value: TurnOutcome) -> Self {
        match value {
            TurnOutcome::Completed => Self::Completed,
            TurnOutcome::PausedOnDenial => Self::PausedOnDenial,
            TurnOutcome::MaxIterationsReached => Self::MaxIterationsReached,
            TurnOutcome::Refused => Self::Refused,
            TurnOutcome::Interrupted => Self::Interrupted,
        }
    }
}

/// One immutable, project-scoped settled-turn record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub project_root: String,
    pub session_id: String,
    pub started_at: DateTime<Utc>,
    pub settled_at: DateTime<Utc>,
    pub outcome: EpisodeOutcome,
    pub provider: String,
    pub model: String,
    /// `text_and_tool_metadata` or `prompt_only`.
    pub capture_quality: String,
    pub events: Vec<EpisodeEvent>,
}

/// Bounded search result for the current project.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EpisodeSummary {
    pub id: String,
    pub project_root: String,
    pub settled_at: DateTime<Utc>,
    pub outcome: EpisodeOutcome,
    pub preview: String,
}

/// Current status of the local prototype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryStatus {
    pub capture_enabled: bool,
    pub episode_count: u64,
    pub project_root: String,
    pub database_path: PathBuf,
    pub sqlite_version: String,
}

/// Background failures surfaced to the TUI without entering model history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryEvent {
    CaptureFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgetResult {
    NotFound,
    Deleted,
    DeletedCheckpointPending(String),
}

type Reply<T> = oneshot::Sender<Result<T>>;

enum Request {
    Status(Reply<WorkerStatus>),
    SetCapture {
        enabled: bool,
        reply: Reply<()>,
    },
    Record {
        episode: Episode,
        reply: Option<Reply<Option<String>>>,
    },
    Search {
        query: String,
        filter: ScopeFilter,
        reply: Reply<Vec<EpisodeSummary>>,
    },
    Show {
        id_prefix: String,
        filter: ScopeFilter,
        reply: Reply<Option<Episode>>,
    },
    Export(Reply<Vec<Episode>>),
    Forget {
        id_prefix: String,
        reply: Reply<ForgetResult>,
    },
    Flush(Reply<()>),
}

#[derive(Debug)]
struct WorkerStatus {
    capture_enabled: bool,
    episode_count: u64,
    sqlite_version: String,
}

/// Cloneable client for the sole SQLite worker.
#[derive(Clone)]
pub struct EpisodicMemory {
    sender: std_mpsc::Sender<Request>,
    scope: WorkspaceScope,
    project_root: String,
    database_path: PathBuf,
    session_id: String,
}

impl EpisodicMemory {
    /// Open the prototype without a background TUI event sink.
    pub fn open(database_path: PathBuf, project_root: PathBuf) -> Result<Self> {
        Self::open_with_events(database_path, project_root, None)
    }

    /// Open the prototype and route asynchronous capture failures to `events`.
    pub fn open_with_events(
        database_path: PathBuf,
        project_root: PathBuf,
        events: Option<mpsc::UnboundedSender<MemoryEvent>>,
    ) -> Result<Self> {
        let scope = WorkspaceScope::project(&project_root)?;
        Self::open_scoped_with_events(database_path, scope, events)
    }

    /// Open memory for either a discovered project or the explicit global
    /// scope.
    pub fn open_scoped(database_path: PathBuf, scope: WorkspaceScope) -> Result<Self> {
        Self::open_scoped_with_events(database_path, scope, None)
    }

    pub fn open_scoped_with_events(
        database_path: PathBuf,
        scope: WorkspaceScope,
        events: Option<mpsc::UnboundedSender<MemoryEvent>>,
    ) -> Result<Self> {
        let project_key = scope.memory_key();
        let project_display = scope.display_name();
        let worker_path = database_path.clone();
        let worker_display = project_display.clone();
        let (request_tx, request_rx) = std_mpsc::channel();
        let (init_tx, init_rx) = std_mpsc::sync_channel(1);

        thread::Builder::new()
            .name("generalist-memory".to_string())
            .spawn(move || {
                let initialized =
                    MemoryWorker::open(&worker_path, project_key, worker_display, events);
                match initialized {
                    Ok(mut worker) => {
                        let database_path = worker.database_path.clone();
                        let _ = init_tx.send(Ok(database_path));
                        worker.run(request_rx);
                    }
                    Err(error) => {
                        let _ = init_tx.send(Err(error));
                    }
                }
            })
            .map_err(|error| Error::Other(format!("Failed to start memory worker: {error}")))?;

        let database_path = init_rx
            .recv()
            .map_err(|_| Error::Other("Memory worker stopped during startup".to_string()))??;

        Ok(Self {
            sender: request_tx,
            scope,
            project_root: project_display,
            database_path,
            session_id: Uuid::new_v4().to_string(),
        })
    }

    pub fn database_path(&self) -> &Path {
        &self.database_path
    }

    pub fn scope(&self) -> &WorkspaceScope {
        &self.scope
    }

    pub async fn status(&self) -> Result<MemoryStatus> {
        let (reply, response) = oneshot::channel();
        self.send(Request::Status(reply))?;
        let worker = receive(response).await?;
        Ok(MemoryStatus {
            capture_enabled: worker.capture_enabled,
            episode_count: worker.episode_count,
            project_root: self.project_root.clone(),
            database_path: self.database_path.clone(),
            sqlite_version: worker.sqlite_version,
        })
    }

    pub async fn set_capture_enabled(&self, enabled: bool) -> Result<()> {
        let (reply, response) = oneshot::channel();
        self.send(Request::SetCapture { enabled, reply })?;
        receive(response).await
    }

    /// Queue a settled turn without waiting for SQLite.
    ///
    /// Channel ordering ensures a later [`Self::flush`] observes this record.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_settled_turn(
        &self,
        prompt: &str,
        history: &[Message],
        outcome: EpisodeOutcome,
        provider: &str,
        model: &str,
        started_at: DateTime<Utc>,
    ) -> Result<()> {
        self.enqueue_settled_turn_with_origin(
            prompt,
            MessageOrigin::Conversation,
            history,
            outcome,
            provider,
            model,
            started_at,
        )
    }

    /// Queue a settled turn whose initiating prompt has explicit host
    /// provenance. This keeps compaction fallback from retaining internal
    /// control text as user-authored memory.
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_settled_turn_with_origin(
        &self,
        prompt: &str,
        prompt_origin: MessageOrigin,
        history: &[Message],
        outcome: EpisodeOutcome,
        provider: &str,
        model: &str,
        started_at: DateTime<Utc>,
    ) -> Result<()> {
        let episode = self.build_episode(
            prompt,
            prompt_origin,
            history,
            outcome,
            provider,
            model,
            started_at,
        );
        self.send(Request::Record {
            episode,
            reply: None,
        })
    }

    /// Record and await one settled turn. Primarily useful to callers that need
    /// a durable acknowledgement and to deterministic tests.
    #[allow(clippy::too_many_arguments)]
    pub async fn record_settled_turn(
        &self,
        prompt: &str,
        history: &[Message],
        outcome: EpisodeOutcome,
        provider: &str,
        model: &str,
        started_at: DateTime<Utc>,
    ) -> Result<Option<String>> {
        let episode = self.build_episode(
            prompt,
            MessageOrigin::Conversation,
            history,
            outcome,
            provider,
            model,
            started_at,
        );
        let (reply, response) = oneshot::channel();
        self.send(Request::Record {
            episode,
            reply: Some(reply),
        })?;
        receive(response).await
    }

    pub async fn search(&self, query: &str) -> Result<Vec<EpisodeSummary>> {
        self.search_scoped(query, ScopeFilter::Current).await
    }

    /// Search an explicitly selected scope set. This is intended for the
    /// permission-gated model tool; local slash commands use [`Self::search`].
    pub async fn search_scoped(
        &self,
        query: &str,
        filter: ScopeFilter,
    ) -> Result<Vec<EpisodeSummary>> {
        let (reply, response) = oneshot::channel();
        self.send(Request::Search {
            query: query.to_string(),
            filter,
            reply,
        })?;
        receive(response).await
    }

    pub async fn show(&self, id_prefix: &str) -> Result<Option<Episode>> {
        self.show_scoped(id_prefix, ScopeFilter::Current).await
    }

    pub async fn show_scoped(
        &self,
        id_prefix: &str,
        filter: ScopeFilter,
    ) -> Result<Option<Episode>> {
        let (reply, response) = oneshot::channel();
        self.send(Request::Show {
            id_prefix: id_prefix.to_string(),
            filter,
            reply,
        })?;
        receive(response).await
    }

    pub async fn export(&self) -> Result<Vec<Episode>> {
        let (reply, response) = oneshot::channel();
        self.send(Request::Export(reply))?;
        receive(response).await
    }

    pub async fn forget(&self, id_prefix: &str) -> Result<ForgetResult> {
        let (reply, response) = oneshot::channel();
        self.send(Request::Forget {
            id_prefix: id_prefix.to_string(),
            reply,
        })?;
        receive(response).await
    }

    /// Wait until every request previously sent by this handle has completed.
    pub async fn flush(&self) -> Result<()> {
        let (reply, response) = oneshot::channel();
        self.send(Request::Flush(reply))?;
        receive(response).await
    }

    fn send(&self, request: Request) -> Result<()> {
        self.sender
            .send(request)
            .map_err(|_| Error::Other("Memory worker is unavailable".to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_episode(
        &self,
        prompt: &str,
        prompt_origin: MessageOrigin,
        history: &[Message],
        outcome: EpisodeOutcome,
        provider: &str,
        model: &str,
        started_at: DateTime<Utc>,
    ) -> Episode {
        let (events, capture_quality) = retained_events(prompt, prompt_origin, history);
        Episode {
            id: Uuid::new_v4().to_string(),
            project_root: self.project_root.clone(),
            session_id: self.session_id.clone(),
            started_at,
            settled_at: Utc::now(),
            outcome,
            provider: provider.to_string(),
            model: model.to_string(),
            capture_quality,
            events,
        }
    }
}

async fn receive<T>(response: oneshot::Receiver<Result<T>>) -> Result<T> {
    response
        .await
        .map_err(|_| Error::Other("Memory worker dropped a response".to_string()))?
}

struct MemoryWorker {
    connection: Connection,
    project_key: Vec<u8>,
    project_root: String,
    database_path: PathBuf,
    sqlite_version: String,
    events: Option<mpsc::UnboundedSender<MemoryEvent>>,
    background_failures: Vec<String>,
}

impl MemoryWorker {
    fn open(
        database_path: &Path,
        project_key: Vec<u8>,
        project_root: String,
        events: Option<mpsc::UnboundedSender<MemoryEvent>>,
    ) -> Result<Self> {
        let parent = database_path.parent().ok_or_else(|| {
            Error::Other(format!(
                "Memory database {} has no parent directory",
                database_path.display()
            ))
        })?;
        reject_symlink(parent, "memory directory")?;
        fs::create_dir_all(parent).map_err(|error| {
            Error::Other(format!(
                "Failed to create memory directory {}: {error}",
                parent.display()
            ))
        })?;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700)).map_err(|error| {
            Error::Other(format!(
                "Failed to restrict memory directory {}: {error}",
                parent.display()
            ))
        })?;
        reject_symlink(parent, "memory directory")?;
        if !parent.is_dir() {
            return Err(Error::Other(format!(
                "Memory directory {} is not a directory",
                parent.display()
            )));
        }
        let database_name = database_path.file_name().ok_or_else(|| {
            Error::Other(format!(
                "Memory database {} has no file name",
                database_path.display()
            ))
        })?;
        let canonical_parent = fs::canonicalize(parent).map_err(|error| {
            Error::Other(format!(
                "Failed to resolve memory directory {}: {error}",
                parent.display()
            ))
        })?;
        let database_path = canonical_parent.join(database_name);
        reject_symlink(&database_path, "memory database")?;

        let connection = Connection::open_with_flags(
            &database_path,
            OpenFlags::default() | OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )
        .map_err(sqlite_error("open database"))?;
        reject_symlink(&database_path, "memory database")?;
        fs::set_permissions(&database_path, fs::Permissions::from_mode(0o600)).map_err(
            |error| {
                Error::Other(format!(
                    "Failed to restrict memory database {}: {error}",
                    database_path.display()
                ))
            },
        )?;
        connection
            .busy_timeout(Duration::from_secs(2))
            .map_err(sqlite_error("configure busy timeout"))?;
        connection
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=FULL;
                 PRAGMA foreign_keys=ON;
                 PRAGMA secure_delete=ON;
                 PRAGMA trusted_schema=OFF;",
            )
            .map_err(sqlite_error("configure database"))?;
        let journal_mode: String = connection
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .map_err(sqlite_error("read journal mode"))?;
        if !journal_mode.eq_ignore_ascii_case("wal") {
            return Err(Error::Other(format!(
                "Memory database did not enter WAL mode (reported '{journal_mode}')"
            )));
        }

        let sqlite_version: String = connection
            .query_row("SELECT sqlite_version()", [], |row| row.get(0))
            .map_err(sqlite_error("read SQLite version"))?;
        if !version_at_least(&sqlite_version, MIN_SQLITE_VERSION) {
            return Err(Error::Other(format!(
                "SQLite {sqlite_version} is too old; episodic memory requires at least 3.51.3"
            )));
        }

        let user_version: i64 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(sqlite_error("read schema version"))?;
        if user_version > SCHEMA_VERSION {
            return Err(Error::Other(format!(
                "Memory schema version {user_version} is newer than supported version {SCHEMA_VERSION}"
            )));
        }
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS memory_settings (
                     project_key BLOB PRIMARY KEY,
                     capture_enabled INTEGER NOT NULL DEFAULT 0
                         CHECK (capture_enabled IN (0, 1))
                 ) STRICT;

                 CREATE TABLE IF NOT EXISTS episodes (
                     id TEXT PRIMARY KEY,
                     project_key BLOB NOT NULL,
                     project_root TEXT NOT NULL,
                     session_id TEXT NOT NULL,
                     started_at TEXT NOT NULL,
                     settled_at TEXT NOT NULL,
                     outcome TEXT NOT NULL,
                     provider TEXT NOT NULL,
                     model TEXT NOT NULL,
                     capture_quality TEXT NOT NULL,
                     events_json TEXT NOT NULL,
                     search_text TEXT NOT NULL
                 ) STRICT;

                 CREATE INDEX IF NOT EXISTS episodes_project_settled
                     ON episodes(project_key, settled_at DESC);

                 CREATE TRIGGER IF NOT EXISTS episodes_are_immutable
                 BEFORE UPDATE ON episodes
                 BEGIN
                     SELECT RAISE(ABORT, 'episodes are immutable');
                 END;

                 PRAGMA user_version = 1;",
            )
            .map_err(sqlite_error("initialize schema"))?;
        connection
            .execute(
                "INSERT OR IGNORE INTO memory_settings(project_key, capture_enabled)
                 VALUES (?1, 0)",
                params![&project_key],
            )
            .map_err(sqlite_error("initialize project settings"))?;

        Ok(Self {
            connection,
            project_key,
            project_root,
            database_path,
            sqlite_version,
            events,
            background_failures: Vec::new(),
        })
    }

    fn run(&mut self, requests: std_mpsc::Receiver<Request>) {
        while let Ok(request) = requests.recv() {
            match request {
                Request::Status(reply) => {
                    let _ = reply.send(self.status());
                }
                Request::SetCapture { enabled, reply } => {
                    let _ = reply.send(self.set_capture_enabled(enabled));
                }
                Request::Record { episode, reply } => {
                    let result = self.record(episode);
                    match reply {
                        Some(reply) => {
                            let _ = reply.send(result);
                        }
                        None => {
                            if let Err(error) = result {
                                let error = error.to_string();
                                self.background_failures.push(error.clone());
                                if let Some(events) = &self.events {
                                    let _ = events.send(MemoryEvent::CaptureFailed(error));
                                }
                            }
                        }
                    }
                }
                Request::Search {
                    query,
                    filter,
                    reply,
                } => {
                    let _ = reply.send(self.search(&query, filter));
                }
                Request::Show {
                    id_prefix,
                    filter,
                    reply,
                } => {
                    let _ = reply.send(self.show(&id_prefix, filter));
                }
                Request::Export(reply) => {
                    let _ = reply.send(self.export());
                }
                Request::Forget { id_prefix, reply } => {
                    let _ = reply.send(self.forget(&id_prefix));
                }
                Request::Flush(reply) => {
                    let failures = std::mem::take(&mut self.background_failures);
                    let result = if failures.is_empty() {
                        Ok(())
                    } else {
                        Err(Error::Other(format!(
                            "{} asynchronous episode capture(s) failed; first failure: {}",
                            failures.len(),
                            failures[0]
                        )))
                    };
                    let _ = reply.send(result);
                }
            }
        }
    }

    fn capture_enabled(&self) -> Result<bool> {
        self.connection
            .query_row(
                "SELECT capture_enabled FROM memory_settings WHERE project_key = ?1",
                params![&self.project_key],
                |row| row.get::<_, i64>(0),
            )
            .map(|enabled| enabled != 0)
            .map_err(sqlite_error("read capture setting"))
    }

    fn status(&self) -> Result<WorkerStatus> {
        let episode_count: i64 = self
            .connection
            .query_row(
                "SELECT count(*) FROM episodes WHERE project_key = ?1",
                params![&self.project_key],
                |row| row.get(0),
            )
            .map_err(sqlite_error("count episodes"))?;
        Ok(WorkerStatus {
            capture_enabled: self.capture_enabled()?,
            episode_count: episode_count.try_into().map_err(|_| {
                Error::Other("SQLite returned a negative episode count".to_string())
            })?,
            sqlite_version: self.sqlite_version.clone(),
        })
    }

    fn set_capture_enabled(&self, enabled: bool) -> Result<()> {
        self.connection
            .execute(
                "UPDATE memory_settings SET capture_enabled = ?2 WHERE project_key = ?1",
                params![&self.project_key, i64::from(enabled)],
            )
            .map_err(sqlite_error("update capture setting"))?;
        Ok(())
    }

    fn record(&self, episode: Episode) -> Result<Option<String>> {
        if !self.capture_enabled()? {
            return Ok(None);
        }
        if episode.project_root != self.project_root {
            return Err(Error::Other(
                "Refusing to record an episode outside this memory handle's project".to_string(),
            ));
        }
        let events_json = serde_json::to_string(&episode.events)?;
        let search_text = searchable_text(&episode);
        self.connection
            .execute(
                "INSERT INTO episodes(
                     id, project_key, project_root, session_id, started_at,
                     settled_at, outcome, provider, model, capture_quality,
                     events_json, search_text
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    &episode.id,
                    &self.project_key,
                    &episode.project_root,
                    &episode.session_id,
                    timestamp(episode.started_at),
                    timestamp(episode.settled_at),
                    episode.outcome.label(),
                    &episode.provider,
                    &episode.model,
                    &episode.capture_quality,
                    events_json,
                    search_text,
                ],
            )
            .map_err(sqlite_error("record episode"))?;
        Ok(Some(episode.id))
    }

    fn search(&self, query: &str, filter: ScopeFilter) -> Result<Vec<EpisodeSummary>> {
        if query.trim().is_empty() {
            return Err(Error::Other("Memory search query cannot be empty".into()));
        }
        let global_key = WorkspaceScope::global().memory_key();
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, project_root, settled_at, outcome, search_text
                 FROM episodes
                 WHERE (
                        (?1 = 'current' AND project_key = ?2)
                     OR (?1 = 'global' AND project_key = ?3)
                     OR (?1 = 'other_projects'
                         AND project_key != ?2 AND project_key != ?3)
                     OR (?1 = 'all')
                 )
                   AND instr(lower(search_text), lower(?4)) > 0
                 ORDER BY settled_at DESC
                 LIMIT ?5",
            )
            .map_err(sqlite_error("prepare episode search"))?;
        let rows = statement
            .query_map(
                params![
                    filter.as_str(),
                    &self.project_key,
                    &global_key,
                    query,
                    SEARCH_LIMIT as i64
                ],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                },
            )
            .map_err(sqlite_error("search episodes"))?;
        rows.map(|row| {
            let (id, project_root, settled_at, outcome, search_text) =
                row.map_err(sqlite_error("read episode search result"))?;
            Ok(EpisodeSummary {
                id,
                project_root,
                settled_at: parse_timestamp(&settled_at)?,
                outcome: EpisodeOutcome::parse(&outcome)?,
                preview: preview(&search_text, 180),
            })
        })
        .collect()
    }

    fn show(&self, id_prefix: &str, filter: ScopeFilter) -> Result<Option<Episode>> {
        let Some(id) = self.resolve_id(id_prefix, filter)? else {
            return Ok(None);
        };
        let global_key = WorkspaceScope::global().memory_key();
        self.connection
            .query_row(
                "SELECT id, project_root, session_id, started_at, settled_at,
                        outcome, provider, model, capture_quality, events_json
                 FROM episodes
                 WHERE id = ?1
                   AND (
                          (?2 = 'current' AND project_key = ?3)
                       OR (?2 = 'global' AND project_key = ?4)
                       OR (?2 = 'other_projects'
                           AND project_key != ?3 AND project_key != ?4)
                       OR (?2 = 'all')
                   )",
                params![id, filter.as_str(), &self.project_key, &global_key],
                stored_episode_row,
            )
            .optional()
            .map_err(sqlite_error("read episode"))?
            .map(StoredEpisode::decode)
            .transpose()
    }

    fn export(&self) -> Result<Vec<Episode>> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT id, project_root, session_id, started_at, settled_at,
                        outcome, provider, model, capture_quality, events_json
                 FROM episodes
                 WHERE project_key = ?1
                 ORDER BY settled_at ASC",
            )
            .map_err(sqlite_error("prepare episode export"))?;
        let rows = statement
            .query_map(params![&self.project_key], stored_episode_row)
            .map_err(sqlite_error("export episodes"))?;
        rows.map(|row| row.map_err(sqlite_error("read exported episode"))?.decode())
            .collect()
    }

    fn forget(&self, id_prefix: &str) -> Result<ForgetResult> {
        let Some(id) = self.resolve_id(id_prefix, ScopeFilter::Current)? else {
            return Ok(ForgetResult::NotFound);
        };
        let deleted = self
            .connection
            .execute(
                "DELETE FROM episodes WHERE project_key = ?1 AND id = ?2",
                params![&self.project_key, id],
            )
            .map_err(sqlite_error("delete episode"))?;
        if deleted != 1 {
            return Ok(ForgetResult::NotFound);
        }
        match self
            .connection
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            }) {
            Ok((0, _, _)) => Ok(ForgetResult::Deleted),
            Ok((busy, log_frames, checkpointed_frames)) => {
                Ok(ForgetResult::DeletedCheckpointPending(format!(
                    "checkpoint busy={busy}, log_frames={log_frames}, \
                     checkpointed_frames={checkpointed_frames}"
                )))
            }
            Err(error) => Ok(ForgetResult::DeletedCheckpointPending(error.to_string())),
        }
    }

    fn resolve_id(&self, id_prefix: &str, filter: ScopeFilter) -> Result<Option<String>> {
        let prefix = id_prefix.trim();
        if prefix.len() < 4
            || !prefix
                .chars()
                .all(|character| character.is_ascii_hexdigit() || character == '-')
        {
            return Err(Error::Other(
                "Episode ID prefixes must contain at least four hexadecimal or '-' characters"
                    .to_string(),
            ));
        }
        let pattern = format!("{prefix}%");
        let global_key = WorkspaceScope::global().memory_key();
        let mut statement = self
            .connection
            .prepare(
                "SELECT id FROM episodes
                 WHERE (
                        (?1 = 'current' AND project_key = ?2)
                     OR (?1 = 'global' AND project_key = ?3)
                     OR (?1 = 'other_projects'
                         AND project_key != ?2 AND project_key != ?3)
                     OR (?1 = 'all')
                 )
                   AND id LIKE ?4
                 ORDER BY id
                 LIMIT 2",
            )
            .map_err(sqlite_error("prepare episode ID lookup"))?;
        let ids = statement
            .query_map(
                params![filter.as_str(), &self.project_key, &global_key, pattern],
                |row| row.get::<_, String>(0),
            )
            .map_err(sqlite_error("look up episode ID"))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sqlite_error("read episode ID"))?;
        match ids.as_slice() {
            [] => Ok(None),
            [id] => Ok(Some(id.clone())),
            _ => Err(Error::Other(format!(
                "Episode ID prefix '{prefix}' is ambiguous"
            ))),
        }
    }
}

impl Drop for MemoryWorker {
    fn drop(&mut self) {
        let _ = self
            .connection
            .execute_batch("PRAGMA wal_checkpoint(PASSIVE);");
        let _ = fs::set_permissions(&self.database_path, fs::Permissions::from_mode(0o600));
    }
}

struct StoredEpisode {
    id: String,
    project_root: String,
    session_id: String,
    started_at: String,
    settled_at: String,
    outcome: String,
    provider: String,
    model: String,
    capture_quality: String,
    events_json: String,
}

impl StoredEpisode {
    fn decode(self) -> Result<Episode> {
        Ok(Episode {
            id: self.id,
            project_root: self.project_root,
            session_id: self.session_id,
            started_at: parse_timestamp(&self.started_at)?,
            settled_at: parse_timestamp(&self.settled_at)?,
            outcome: EpisodeOutcome::parse(&self.outcome)?,
            provider: self.provider,
            model: self.model,
            capture_quality: self.capture_quality,
            events: serde_json::from_str(&self.events_json)?,
        })
    }
}

fn stored_episode_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredEpisode> {
    Ok(StoredEpisode {
        id: row.get(0)?,
        project_root: row.get(1)?,
        session_id: row.get(2)?,
        started_at: row.get(3)?,
        settled_at: row.get(4)?,
        outcome: row.get(5)?,
        provider: row.get(6)?,
        model: row.get(7)?,
        capture_quality: row.get(8)?,
        events_json: row.get(9)?,
    })
}

fn retained_events(
    prompt: &str,
    prompt_origin: MessageOrigin,
    history: &[Message],
) -> (Vec<EpisodeEvent>, String) {
    let has_initial_prompt = history.first().is_some_and(|message| {
        message.role == "user"
            && message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Text { text } if text == prompt))
    });
    let mut events = Vec::new();
    let is_host_goal_prompt = prompt_origin == MessageOrigin::GoalContinuation
        && crate::goal::is_goal_continuation_prompt(prompt);
    if !has_initial_prompt && !is_host_goal_prompt {
        events.push(EpisodeEvent::UserText {
            text: prompt.to_string(),
        });
    }
    for message in if has_initial_prompt { history } else { &[] } {
        for block in &message.content {
            match block {
                ContentBlock::Text { text }
                    if !text.is_empty()
                        && message.role == "user"
                        && !message.is_goal_continuation() =>
                {
                    events.push(EpisodeEvent::UserText { text: text.clone() });
                }
                ContentBlock::Text { text } if !text.is_empty() && message.role == "assistant" => {
                    events.push(EpisodeEvent::AssistantText { text: text.clone() });
                }
                ContentBlock::ToolUse { name, .. } => {
                    events.push(EpisodeEvent::ToolCall { name: name.clone() });
                }
                ContentBlock::ToolResult { is_error, .. } => {
                    events.push(EpisodeEvent::ToolResult {
                        is_error: is_error.unwrap_or(false),
                    });
                }
                ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. }
                | ContentBlock::Text { .. } => {}
            }
        }
    }
    let quality = if has_initial_prompt {
        "text_and_tool_metadata"
    } else {
        "prompt_only"
    };
    (events, quality.to_string())
}

fn searchable_text(episode: &Episode) -> String {
    let mut parts = vec![
        episode.outcome.label().to_string(),
        episode.provider.clone(),
        episode.model.clone(),
    ];
    for event in &episode.events {
        match event {
            EpisodeEvent::UserText { text } | EpisodeEvent::AssistantText { text } => {
                parts.push(text.clone());
            }
            EpisodeEvent::ToolCall { name, .. } => parts.push(name.clone()),
            EpisodeEvent::ToolResult { .. } => {}
        }
    }
    parts.join("\n")
}

fn preview(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut characters = normalized.chars();
    let prefix: String = characters.by_ref().take(limit).collect();
    if characters.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

fn timestamp(value: DateTime<Utc>) -> String {
    value.to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .map_err(|error| Error::Other(format!("Invalid stored episode timestamp: {error}")))
}

fn version_at_least(version: &str, minimum: (u32, u32, u32)) -> bool {
    let mut components = version
        .split('.')
        .map(|part| part.parse::<u32>().unwrap_or(0));
    let actual = (
        components.next().unwrap_or(0),
        components.next().unwrap_or(0),
        components.next().unwrap_or(0),
    );
    actual >= minimum
}

fn sqlite_error(context: &'static str) -> impl FnOnce(rusqlite::Error) -> Error {
    move |error| Error::Other(format!("Failed to {context}: {error}"))
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

pub fn default_memory_path(home: &Path) -> PathBuf {
    home.join(".generalist")
        .join("memory")
        .join("scoped-episodes.sqlite3")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn open_memory(temp: &TempDir, project_name: &str) -> EpisodicMemory {
        let project = temp.path().join(project_name);
        fs::create_dir_all(project.join(".git")).unwrap();
        EpisodicMemory::open(temp.path().join("episodes.sqlite3"), project).unwrap()
    }

    fn history_with_sensitive_tool_data(prompt: &str) -> Vec<Message> {
        vec![
            Message::user_text(prompt),
            Message::assistant(vec![
                ContentBlock::Thinking {
                    thinking: "private chain of thought".to_string(),
                    signature: "provider-signature".to_string(),
                },
                ContentBlock::ToolUse {
                    name: "bash".to_string(),
                    input: json!({"secret": "tool-input-secret"}),
                    id: "provider-tool-id-secret".to_string(),
                },
            ]),
            Message::user(vec![ContentBlock::ToolResult {
                content: "tool-output-secret".to_string(),
                tool_use_id: "provider-tool-id-secret".to_string(),
                is_error: Some(false),
            }]),
            Message::assistant(vec![
                ContentBlock::RedactedThinking {
                    data: "redacted-provider-payload".to_string(),
                },
                ContentBlock::Text {
                    text: "The build passed.".to_string(),
                },
            ]),
        ]
    }

    #[test]
    fn default_database_path_does_not_reuse_the_unscoped_store() {
        let home = Path::new("/profile");
        assert_eq!(
            default_memory_path(home),
            home.join(".generalist/memory/scoped-episodes.sqlite3")
        );
        assert_ne!(
            default_memory_path(home),
            home.join(".generalist/memory/episodes.sqlite3")
        );
    }

    #[tokio::test]
    async fn capture_is_paused_by_default() {
        let temp = TempDir::new().unwrap();
        let memory = open_memory(&temp, "one");
        let id = memory
            .record_settled_turn(
                "hello",
                &[Message::user_text("hello")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap();
        assert_eq!(id, None);
        assert_eq!(memory.status().await.unwrap().episode_count, 0);
    }

    #[tokio::test]
    async fn capture_and_setting_changes_observe_fifo_order() {
        let temp = TempDir::new().unwrap();
        let memory = open_memory(&temp, "one");
        memory
            .enqueue_settled_turn(
                "skipped-before-resume",
                &[Message::user_text("skipped-before-resume")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .unwrap();
        memory.set_capture_enabled(true).await.unwrap();
        assert_eq!(memory.status().await.unwrap().episode_count, 0);

        memory
            .enqueue_settled_turn(
                "captured-after-resume",
                &[Message::user_text("captured-after-resume")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .unwrap();
        memory.flush().await.unwrap();
        assert_eq!(memory.status().await.unwrap().episode_count, 1);
    }

    #[tokio::test]
    async fn episodes_omit_reasoning_and_tool_payloads() {
        let temp = TempDir::new().unwrap();
        let memory = open_memory(&temp, "one");
        memory.set_capture_enabled(true).await.unwrap();
        let prompt = "Run the build";
        let id = memory
            .record_settled_turn(
                prompt,
                &history_with_sensitive_tool_data(prompt),
                EpisodeOutcome::Completed,
                "openrouter",
                "model",
                Utc::now(),
            )
            .await
            .unwrap()
            .unwrap();
        let episode = memory.show(&id[..8]).await.unwrap().unwrap();
        let serialized = serde_json::to_string(&episode).unwrap();
        assert!(serialized.contains(prompt));
        assert!(serialized.contains("The build passed."));
        assert!(serialized.contains("\"name\":\"bash\""));
        for excluded in [
            "private chain of thought",
            "provider-signature",
            "tool-input-secret",
            "tool-output-secret",
            "provider-tool-id-secret",
            "redacted-provider-payload",
        ] {
            assert!(!serialized.contains(excluded), "retained {excluded}");
        }
    }

    #[test]
    fn duplicate_steering_text_does_not_move_the_episode_boundary() {
        let prompt = "repeat";
        let history = vec![
            Message::user_text(prompt),
            Message::assistant(vec![ContentBlock::Text {
                text: "first answer".to_string(),
            }]),
            Message::user_text(prompt),
            Message::assistant(vec![ContentBlock::Text {
                text: "second answer".to_string(),
            }]),
        ];
        let (events, quality) = retained_events(prompt, MessageOrigin::Conversation, &history);
        assert_eq!(quality, "text_and_tool_metadata");
        assert_eq!(
            events,
            vec![
                EpisodeEvent::UserText {
                    text: prompt.to_string()
                },
                EpisodeEvent::AssistantText {
                    text: "first answer".to_string()
                },
                EpisodeEvent::UserText {
                    text: prompt.to_string()
                },
                EpisodeEvent::AssistantText {
                    text: "second answer".to_string()
                },
            ]
        );
    }

    #[test]
    fn host_goal_continuations_are_not_retained_as_user_authored_text() {
        let prompt = crate::goal::GOAL_CONTINUATION_PROMPT;
        let history = vec![
            Message::goal_continuation(),
            Message::assistant(vec![ContentBlock::Text {
                text: "made more progress".to_string(),
            }]),
        ];

        let (events, quality) = retained_events(prompt, MessageOrigin::GoalContinuation, &history);

        assert_eq!(quality, "text_and_tool_metadata");
        assert_eq!(
            events,
            vec![EpisodeEvent::AssistantText {
                text: "made more progress".to_string()
            }]
        );

        let manual = vec![Message::user_text(prompt)];
        let (events, _) = retained_events(prompt, MessageOrigin::Conversation, &manual);
        assert_eq!(
            events,
            vec![EpisodeEvent::UserText {
                text: prompt.to_string()
            }],
            "matching text remains user-authored without host provenance"
        );

        let (events, quality) = retained_events(prompt, MessageOrigin::GoalContinuation, &[]);
        assert_eq!(quality, "prompt_only");
        assert!(
            events.is_empty(),
            "compaction fallback must not retain host control text"
        );

        let (events, _) = retained_events("ordinary prompt", MessageOrigin::GoalContinuation, &[]);
        assert_eq!(
            events,
            vec![EpisodeEvent::UserText {
                text: "ordinary prompt".to_string()
            }],
            "provenance alone cannot suppress arbitrary prompt text"
        );
    }

    #[test]
    fn a_relocated_history_boundary_degrades_to_prompt_only() {
        let (events, quality) =
            retained_events("original prompt", MessageOrigin::Conversation, &[]);
        assert_eq!(quality, "prompt_only");
        assert_eq!(
            events,
            vec![EpisodeEvent::UserText {
                text: "original prompt".to_string()
            }]
        );
    }

    #[tokio::test]
    async fn project_handles_cannot_search_or_delete_each_others_episodes() {
        let temp = TempDir::new().unwrap();
        let first = open_memory(&temp, "one");
        let second = open_memory(&temp, "two");
        first.set_capture_enabled(true).await.unwrap();
        second.set_capture_enabled(true).await.unwrap();
        let first_id = first
            .record_settled_turn(
                "alpha-only",
                &[Message::user_text("alpha-only")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap()
            .unwrap();
        second
            .record_settled_turn(
                "beta-only",
                &[Message::user_text("beta-only")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap();
        assert_eq!(first.search("beta-only").await.unwrap(), Vec::new());
        assert_eq!(second.search("alpha-only").await.unwrap(), Vec::new());
        assert_eq!(
            second.forget(&first_id[..8]).await.unwrap(),
            ForgetResult::NotFound
        );
        assert!(first.show(&first_id[..8]).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn global_scope_is_explicit_and_cross_scope_search_is_bounded_by_filter() {
        let temp = TempDir::new().unwrap();
        let project = open_memory(&temp, "project");
        let global = EpisodicMemory::open_scoped(
            temp.path().join("episodes.sqlite3"),
            WorkspaceScope::Global,
        )
        .unwrap();
        project.set_capture_enabled(true).await.unwrap();
        global.set_capture_enabled(true).await.unwrap();
        let project_id = project
            .record_settled_turn(
                "shared needle project",
                &[Message::user_text("shared needle project")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap()
            .unwrap();
        let global_id = global
            .record_settled_turn(
                "shared needle global",
                &[Message::user_text("shared needle global")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap()
            .unwrap();

        let current = project
            .search_scoped("shared needle", ScopeFilter::Current)
            .await
            .unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(current[0].id, project_id);
        let global_matches = project
            .search_scoped("shared needle", ScopeFilter::Global)
            .await
            .unwrap();
        assert_eq!(global_matches.len(), 1);
        assert_eq!(global_matches[0].id, global_id);
        assert_eq!(global_matches[0].project_root, "global");
        let all = project
            .search_scoped("shared needle", ScopeFilter::All)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
        let global_current = global
            .search_scoped("shared needle", ScopeFilter::Current)
            .await
            .unwrap();
        assert_eq!(global_current.len(), 1);
        assert_eq!(global_current[0].id, global_id);
        let global_other = global
            .search_scoped("shared needle", ScopeFilter::OtherProjects)
            .await
            .unwrap();
        assert_eq!(global_other.len(), 1);
        assert_eq!(global_other[0].id, project_id);
        assert!(project
            .show_scoped(&global_id, ScopeFilter::Current)
            .await
            .unwrap()
            .is_none());
        assert!(project
            .show_scoped(&global_id, ScopeFilter::Global)
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn episodes_are_immutable_but_can_be_forgotten_from_the_live_store() {
        let temp = TempDir::new().unwrap();
        let memory = open_memory(&temp, "one");
        memory.set_capture_enabled(true).await.unwrap();
        let id = memory
            .record_settled_turn(
                "immutable",
                &[Message::user_text("immutable")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap()
            .unwrap();
        let direct = Connection::open(memory.database_path()).unwrap();
        let update = direct.execute(
            "UPDATE episodes SET outcome = 'error' WHERE id = ?1",
            params![&id],
        );
        assert!(update.is_err());
        assert_eq!(
            memory.forget(&id[..8]).await.unwrap(),
            ForgetResult::Deleted
        );
        assert!(memory.show(&id[..8]).await.unwrap().is_none());
        assert_eq!(memory.status().await.unwrap().episode_count, 0);
    }

    #[tokio::test]
    async fn forget_reports_a_busy_wal_checkpoint_after_committing_the_delete() {
        let temp = TempDir::new().unwrap();
        let memory = open_memory(&temp, "one");
        memory.set_capture_enabled(true).await.unwrap();
        let id = memory
            .record_settled_turn(
                "checkpoint",
                &[Message::user_text("checkpoint")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap()
            .unwrap();

        let reader = Connection::open(memory.database_path()).unwrap();
        reader.execute_batch("BEGIN").unwrap();
        let count: i64 = reader
            .query_row("SELECT count(*) FROM episodes", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 1);

        let result = memory.forget(&id[..8]).await.unwrap();
        assert!(
            matches!(result, ForgetResult::DeletedCheckpointPending(_)),
            "unexpected checkpoint result: {result:?}"
        );
        assert!(memory.show(&id[..8]).await.unwrap().is_none());
        reader.execute_batch("ROLLBACK").unwrap();
    }

    #[tokio::test]
    async fn database_and_directory_are_restricted_to_the_unix_owner() {
        let temp = TempDir::new().unwrap();
        let memory = open_memory(&temp, "one");
        memory.set_capture_enabled(true).await.unwrap();
        memory
            .record_settled_turn(
                "permissions",
                &[Message::user_text("permissions")],
                EpisodeOutcome::Completed,
                "test",
                "model",
                Utc::now(),
            )
            .await
            .unwrap();
        let database_mode = fs::metadata(memory.database_path())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let directory_mode = fs::metadata(memory.database_path().parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(database_mode, 0o600);
        assert_eq!(directory_mode, 0o700);
        for suffix in ["-wal", "-shm"] {
            let sidecar = PathBuf::from(format!("{}{suffix}", memory.database_path().display()));
            let sidecar_mode = fs::metadata(sidecar).unwrap().permissions().mode() & 0o777;
            assert_eq!(sidecar_mode, 0o600);
        }
    }

    #[test]
    fn symlinked_database_is_rejected() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        let memory_directory = temp.path().join("memory");
        let outside = temp.path().join("outside.sqlite3");
        fs::create_dir_all(project.join(".git")).unwrap();
        fs::create_dir_all(&memory_directory).unwrap();
        fs::write(&outside, b"not a database").unwrap();
        let database = memory_directory.join("episodes.sqlite3");
        symlink(&outside, &database).unwrap();

        let error = match EpisodicMemory::open(database, project) {
            Ok(_) => panic!("symlinked memory database was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("symlinked memory database"));
    }

    #[test]
    fn symlinked_memory_directory_is_rejected() {
        use std::os::unix::fs::symlink;

        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        let actual_directory = temp.path().join("actual-memory");
        let linked_directory = temp.path().join("linked-memory");
        fs::create_dir_all(project.join(".git")).unwrap();
        fs::create_dir_all(&actual_directory).unwrap();
        symlink(&actual_directory, &linked_directory).unwrap();

        let error = match EpisodicMemory::open(linked_directory.join("episodes.sqlite3"), project) {
            Ok(_) => panic!("symlinked memory directory was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("symlinked memory directory"));
    }

    #[test]
    fn project_discovery_uses_the_nearest_git_root() {
        let temp = TempDir::new().unwrap();
        let project = temp.path().join("project");
        let nested = project.join("a").join("b");
        fs::create_dir_all(project.join(".git")).unwrap();
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(
            discover_project_root(&nested).unwrap(),
            fs::canonicalize(project).unwrap()
        );
    }
}
