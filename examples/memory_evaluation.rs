//! Deterministic evaluation of Generalist's explicit episodic-memory baseline.
//!
//! This is an evaluation harness, not a second memory implementation. It uses
//! the public `EpisodicMemory`, `HistoryStore`, archive-tool, and permission
//! paths, plus subprocess crashes of this same executable.

use chrono::Utc;
use generalist::tools::{SearchConversationsTool, SearchMemoriesTool};
use generalist::{
    ContentBlock, Episode, EpisodeEvent, EpisodeOutcome, EpisodicMemory, ForgetResult,
    HistoryStore, Message, SavedState, ToolCallOutcome, ToolRegistry, WorkspaceScope,
};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use uuid::Uuid;

const CORPUS: &str = include_str!("../benchmarks/episodic_memory/cases.json");
const LATENCY_SAMPLES_PER_QUERY: usize = 25;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Deserialize)]
struct Corpus {
    schema_version: u32,
    records: Vec<Record>,
    queries: Vec<Query>,
    other_project_record: Record,
}

#[derive(Debug, Clone, Deserialize)]
struct Record {
    id: String,
    user: String,
    assistant: String,
}

#[derive(Debug, Deserialize)]
struct Query {
    id: String,
    query: String,
    relevant_record_ids: Vec<String>,
    kind: String,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    schema_version: u32,
    corpus_schema_version: u32,
    experiment: &'static str,
    caveat: &'static str,
    counts: CountReport,
    retrieval: BTreeMap<String, RetrievalReport>,
    latency_ms: BTreeMap<String, LatencyReport>,
    retained_bytes: BTreeMap<String, u64>,
    scope_isolation: ScopeIsolationReport,
    deletion: DeletionReport,
    write_lock: WriteLockReport,
    exit_boundary: ExitBoundaryReport,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct CountReport {
    corpus_records: usize,
    scored_queries: usize,
    b0_episodes: u64,
    b1_episodes_before_delete: u64,
    autosave_archives: usize,
    named_archives: usize,
}

#[derive(Debug, Serialize)]
struct RetrievalReport {
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
    precision: f64,
    recall: f64,
    reciprocal_rank_mean: f64,
    abstention_accuracy: f64,
    queries: Vec<QueryReport>,
}

#[derive(Debug, Serialize)]
struct QueryReport {
    id: String,
    kind: String,
    expected: Vec<String>,
    retrieved: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LatencyReport {
    samples: usize,
    p50: f64,
    p95: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct ScopeIsolationReport {
    current_scope_matches: Vec<String>,
    all_scope_match_count: usize,
    leaked_other_project_record: bool,
}

#[derive(Debug, Serialize)]
struct DeletionReport {
    live_search_absent: bool,
    restart_search_absent: bool,
    exported_snapshot_still_contains_record: bool,
    checkpoint_completed: bool,
}

#[derive(Debug, Serialize)]
struct WriteLockReport {
    elapsed_ms: f64,
    executor_ticks_while_waiting: u64,
    returned_locked_error: bool,
    setting_unchanged: bool,
}

#[derive(Debug, Serialize)]
struct ExitBoundaryReport {
    enqueue_attempts: usize,
    enqueue_absent: usize,
    enqueue_complete: usize,
    acknowledged_attempts: usize,
    acknowledged_complete: usize,
    locked_attempts: usize,
    locked_absent: usize,
    duplicate_or_partial_rows: usize,
    immutable_update_rejected: bool,
    integrity_check: String,
}

fn assistant_text(text: impl Into<String>) -> Message {
    Message::assistant(vec![ContentBlock::Text { text: text.into() }])
}

fn record_history(record: &Record) -> Vec<Message> {
    vec![
        Message::user_text(record.user.clone()),
        assistant_text(record.assistant.clone()),
    ]
}

fn private_project(root: &Path, name: &str) -> Result<(PathBuf, WorkspaceScope)> {
    let project = root.join(name);
    fs::create_dir_all(project.join(".git"))?;
    let scope = WorkspaceScope::discover(&project)?;
    Ok((project, scope))
}

fn directory_bytes(path: &Path) -> io::Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    if path.is_file() {
        return Ok(path.metadata()?.len());
    }
    let mut bytes = 0;
    for entry in fs::read_dir(path)? {
        bytes += directory_bytes(&entry?.path())?;
    }
    Ok(bytes)
}

fn database_bytes(path: &Path) -> io::Result<u64> {
    let mut bytes = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
    for suffix in ["-wal", "-shm"] {
        let sidecar = PathBuf::from(format!("{}{suffix}", path.display()));
        bytes += sidecar
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
    }
    Ok(bytes)
}

fn percentile(mut samples: Vec<f64>, fraction: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(f64::total_cmp);
    let index = ((samples.len() - 1) as f64 * fraction).ceil() as usize;
    samples[index]
}

fn latency_report(samples: Vec<f64>) -> LatencyReport {
    LatencyReport {
        samples: samples.len(),
        p50: percentile(samples.clone(), 0.50),
        p95: percentile(samples.clone(), 0.95),
        max: percentile(samples, 1.0),
    }
}

fn tool_result_text(result: generalist::ToolCallResult) -> Result<String> {
    if result.outcome != ToolCallOutcome::Success {
        return Err(generalist::Error::Other(format!(
            "archive search failed with {:?}",
            result.outcome
        ))
        .into());
    }
    match result.block {
        ContentBlock::ToolResult { content, .. } => Ok(content),
        other => Err(generalist::Error::Other(format!(
            "archive search returned non-result block: {other:?}"
        ))
        .into()),
    }
}

async fn search_history(
    registry: &mut ToolRegistry,
    query: &str,
    call_index: usize,
) -> Result<Vec<String>> {
    let result = registry
        .execute_tool(
            "search_conversations",
            json!({"query": query, "scope": "current"}),
            format!("history-eval-{call_index}"),
        )
        .await;
    let value: Value = serde_json::from_str(&tool_result_text(result)?)?;
    Ok(value["matches"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item["name"].as_str().map(str::to_string))
        .collect())
}

async fn search_memory_tool(
    registry: &mut ToolRegistry,
    query: &str,
    call_index: usize,
) -> Result<Vec<String>> {
    let result = registry
        .execute_tool(
            "search_memories",
            json!({"query": query, "scope": "current"}),
            format!("memory-eval-{call_index}"),
        )
        .await;
    let value: Value = serde_json::from_str(&tool_result_text(result)?)?;
    Ok(value["matches"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item["id"].as_str().map(str::to_string))
        .collect())
}

fn score_queries(
    corpus: &Corpus,
    retrieved: &[Vec<String>],
    aliases: &HashMap<String, String>,
) -> RetrievalReport {
    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    let mut reciprocal_rank = 0.0;
    let mut supported_queries = 0;
    let mut abstentions = 0;
    let mut abstention_correct = 0;
    let mut queries = Vec::new();

    for (query, raw) in corpus.queries.iter().zip(retrieved) {
        let actual = raw
            .iter()
            .filter_map(|id| aliases.get(id).cloned())
            .collect::<Vec<_>>();
        let expected = query
            .relevant_record_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let actual_set = actual.iter().cloned().collect::<BTreeSet<_>>();
        true_positives += actual_set.intersection(&expected).count();
        false_positives += actual_set.difference(&expected).count();
        false_negatives += expected.difference(&actual_set).count();
        if expected.is_empty() {
            abstentions += 1;
            abstention_correct += usize::from(actual.is_empty());
        } else {
            supported_queries += 1;
            if let Some(rank) = actual.iter().position(|id| expected.contains(id)) {
                reciprocal_rank += 1.0 / (rank + 1) as f64;
            }
        }
        queries.push(QueryReport {
            id: query.id.clone(),
            kind: query.kind.clone(),
            expected: expected.into_iter().collect(),
            retrieved: actual,
        });
    }

    let precision_denominator = true_positives + false_positives;
    let recall_denominator = true_positives + false_negatives;
    RetrievalReport {
        true_positives,
        false_positives,
        false_negatives,
        precision: if precision_denominator == 0 {
            1.0
        } else {
            true_positives as f64 / precision_denominator as f64
        },
        recall: if recall_denominator == 0 {
            1.0
        } else {
            true_positives as f64 / recall_denominator as f64
        },
        reciprocal_rank_mean: if supported_queries == 0 {
            0.0
        } else {
            reciprocal_rank / supported_queries as f64
        },
        abstention_accuracy: if abstentions == 0 {
            1.0
        } else {
            abstention_correct as f64 / abstentions as f64
        },
        queries,
    }
}

fn complete_episode(episode: &Episode, marker: &str) -> bool {
    episode.outcome == EpisodeOutcome::Completed
        && episode.capture_quality == "text_and_tool_metadata"
        && episode
            .events
            .iter()
            .any(|event| matches!(event, EpisodeEvent::UserText { text } if text.contains(marker)))
        && episode.events.iter().any(
            |event| matches!(event, EpisodeEvent::AssistantText { text } if text.contains(marker)),
        )
}

fn child_enqueue(args: &[String]) -> Result<()> {
    if args.len() != 6 {
        return Err(generalist::Error::Other(
            "child enqueue expected database, project, marker, delay, and gate mode".into(),
        )
        .into());
    }
    let database = PathBuf::from(&args[1]);
    let project = PathBuf::from(&args[2]);
    let marker = &args[3];
    let delay_ms = args[4].parse::<u64>().map_err(|error| {
        generalist::Error::Other(format!("invalid child delay '{}': {error}", args[4]))
    })?;
    let gated = args[5] == "gated";
    let memory = EpisodicMemory::open(database, project)?;
    if gated {
        println!("READY");
        io::stdout().flush()?;
        let mut trigger = [0_u8; 1];
        io::stdin().read_exact(&mut trigger)?;
    }
    let history = vec![
        Message::user_text(format!("crash boundary {marker}")),
        assistant_text(format!("complete crash boundary response {marker}")),
    ];
    memory.enqueue_settled_turn(
        &format!("crash boundary {marker}"),
        &history,
        EpisodeOutcome::Completed,
        "evaluation",
        "deterministic",
        Utc::now(),
    )?;
    println!("ENQUEUED");
    io::stdout().flush()?;
    thread::sleep(Duration::from_millis(delay_ms));
    std::process::exit(0);
}

fn child_acknowledged(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err(generalist::Error::Other(
            "acknowledged child expected database, project, and marker".into(),
        )
        .into());
    }
    let database = PathBuf::from(&args[1]);
    let project = PathBuf::from(&args[2]);
    let marker = &args[3];
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| generalist::Error::Other(error.to_string()))?;
    runtime.block_on(async {
        let memory = EpisodicMemory::open(database, project)?;
        let history = vec![
            Message::user_text(format!("acknowledged boundary {marker}")),
            assistant_text(format!("complete acknowledged response {marker}")),
        ];
        let id = memory
            .record_settled_turn(
                &format!("acknowledged boundary {marker}"),
                &history,
                EpisodeOutcome::Completed,
                "evaluation",
                "deterministic",
                Utc::now(),
            )
            .await?;
        if id.is_none() {
            return Err(generalist::Error::Other(
                "capture unexpectedly paused in acknowledged child".into(),
            ));
        }
        Ok::<(), generalist::Error>(())
    })?;
    std::process::exit(0);
}

async fn evaluate_exit_boundary(root: &Path) -> Result<ExitBoundaryReport> {
    let (project, _) = private_project(root, "crash-project")?;
    let database = root.join("crash-episodes.sqlite3");
    let setup = EpisodicMemory::open(database.clone(), project.clone())?;
    setup.set_capture_enabled(true).await?;
    drop(setup);
    thread::sleep(Duration::from_millis(20));

    let executable = env::current_exe()?;
    let mut enqueue_absent = 0;
    let mut enqueue_complete = 0;
    let mut duplicate_or_partial_rows = 0;
    let delays = [0_u64, 1, 5, 20, 0, 1, 5, 20];
    for (index, delay) in delays.into_iter().enumerate() {
        let marker = format!("ENQUEUE-{index}-{}", Uuid::new_v4());
        let status = Command::new(&executable)
            .args([
                "__child_enqueue",
                &database.to_string_lossy(),
                &project.to_string_lossy(),
                &marker,
                &delay.to_string(),
                "direct",
            ])
            .stdout(Stdio::null())
            .status()?;
        if !status.success() {
            return Err(generalist::Error::Other(format!(
                "enqueue crash child {index} failed with {status}"
            ))
            .into());
        }
        let memory = EpisodicMemory::open(database.clone(), project.clone())?;
        let matches = memory.search(&marker).await?;
        match matches.as_slice() {
            [] => enqueue_absent += 1,
            [summary] => {
                let episode = memory.show(&summary.id).await?.ok_or_else(|| {
                    generalist::Error::Other("committed crash row disappeared".into())
                })?;
                if complete_episode(&episode, &marker) {
                    enqueue_complete += 1;
                } else {
                    duplicate_or_partial_rows += 1;
                }
            }
            _ => duplicate_or_partial_rows += 1,
        }
    }

    let acknowledged_attempts = 3;
    let mut acknowledged_complete = 0;
    for index in 0..acknowledged_attempts {
        let marker = format!("ACK-{index}-{}", Uuid::new_v4());
        let status = Command::new(&executable)
            .args([
                "__child_acknowledged",
                &database.to_string_lossy(),
                &project.to_string_lossy(),
                &marker,
            ])
            .stdout(Stdio::null())
            .status()?;
        if !status.success() {
            return Err(generalist::Error::Other(format!(
                "acknowledged crash child {index} failed with {status}"
            ))
            .into());
        }
        let memory = EpisodicMemory::open(database.clone(), project.clone())?;
        let matches = memory.search(&marker).await?;
        if let [summary] = matches.as_slice() {
            let episode = memory
                .show(&summary.id)
                .await?
                .ok_or_else(|| generalist::Error::Other("acknowledged row disappeared".into()))?;
            if complete_episode(&episode, &marker) {
                acknowledged_complete += 1;
            } else {
                duplicate_or_partial_rows += 1;
            }
        } else {
            duplicate_or_partial_rows += 1;
        }
    }

    let locked_attempts = 3;
    let mut locked_absent = 0;
    for index in 0..locked_attempts {
        let marker = format!("LOCKED-{index}-{}", Uuid::new_v4());
        let mut child = Command::new(&executable)
            .args([
                "__child_enqueue",
                &database.to_string_lossy(),
                &project.to_string_lossy(),
                &marker,
                "30000",
                "gated",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().ok_or_else(|| {
            generalist::Error::Other("locked child stdout was unavailable".into())
        })?;
        let mut lines = io::BufReader::new(stdout);
        let mut line = String::new();
        lines.read_line(&mut line)?;
        if line.trim() != "READY" {
            return Err(generalist::Error::Other(format!(
                "locked child did not become ready: {line:?}"
            ))
            .into());
        }
        let lock = Connection::open(&database)?;
        lock.execute_batch("BEGIN IMMEDIATE")?;
        child
            .stdin
            .as_mut()
            .ok_or_else(|| generalist::Error::Other("locked child stdin unavailable".into()))?
            .write_all(b"x")?;
        line.clear();
        lines.read_line(&mut line)?;
        if line.trim() != "ENQUEUED" {
            return Err(generalist::Error::Other(format!(
                "locked child did not enqueue: {line:?}"
            ))
            .into());
        }
        child.kill()?;
        let _ = child.wait()?;
        lock.execute_batch("ROLLBACK")?;
        let memory = EpisodicMemory::open(database.clone(), project.clone())?;
        let matches = memory.search(&marker).await?;
        if matches.is_empty() {
            locked_absent += 1;
        } else {
            duplicate_or_partial_rows += 1;
        }
    }

    let direct = Connection::open(&database)?;
    let first_id: String =
        direct.query_row("SELECT id FROM episodes LIMIT 1", [], |row| row.get(0))?;
    let immutable_update_rejected = direct
        .execute(
            "UPDATE episodes SET outcome = 'error' WHERE id = ?1",
            [&first_id],
        )
        .is_err();
    let integrity_check: String =
        direct.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;

    Ok(ExitBoundaryReport {
        enqueue_attempts: delays.len(),
        enqueue_absent,
        enqueue_complete,
        acknowledged_attempts,
        acknowledged_complete,
        locked_attempts,
        locked_absent,
        duplicate_or_partial_rows,
        immutable_update_rejected,
        integrity_check,
    })
}

async fn evaluate() -> Result<EvaluationReport> {
    let corpus: Corpus = serde_json::from_str(CORPUS)?;
    if corpus.schema_version != 1 {
        return Err(generalist::Error::Other(format!(
            "unsupported memory evaluation corpus version {}",
            corpus.schema_version
        ))
        .into());
    }
    let temp = TempDir::new()?;
    let (_project, scope) = private_project(temp.path(), "project-alpha")?;
    let (_other_project, other_scope) = private_project(temp.path(), "project-beta")?;
    let b0_path = temp.path().join("b0.sqlite3");
    let b1_path = temp.path().join("b1.sqlite3");
    let autosave_store = HistoryStore::open(temp.path().join("autosave-home"), scope.clone())?;
    let named_store = HistoryStore::open(temp.path().join("named-home"), scope.clone())?;

    let mut b1_aliases = HashMap::new();
    let b0_memory = EpisodicMemory::open_scoped(b0_path.clone(), scope.clone())?;
    let first_b1 = EpisodicMemory::open_scoped(b1_path.clone(), scope.clone())?;
    first_b1.set_capture_enabled(true).await?;
    drop(first_b1);

    for record in &corpus.records {
        let history = record_history(record);
        let skipped = b0_memory
            .record_settled_turn(
                &record.user,
                &history,
                EpisodeOutcome::Completed,
                "evaluation",
                "deterministic",
                Utc::now(),
            )
            .await?;
        if skipped.is_some() {
            return Err(generalist::Error::Other(
                "B0 retained an episode while capture was paused".into(),
            )
            .into());
        }
        let b1_session = EpisodicMemory::open_scoped(b1_path.clone(), scope.clone())?;
        let episode_id = b1_session
            .record_settled_turn(
                &record.user,
                &history,
                EpisodeOutcome::Completed,
                "evaluation",
                "deterministic",
                Utc::now(),
            )
            .await?
            .ok_or_else(|| {
                generalist::Error::Other("B1 capture unexpectedly became paused".into())
            })?;
        b1_aliases.insert(episode_id, record.id.clone());

        let mut state = SavedState::new(scope.clone(), "evaluation".into(), "deterministic".into());
        state.conversation_history = history;
        autosave_store.save(&state, "autosave")?;
        named_store.save(&state, &record.id)?;
        thread::sleep(Duration::from_millis(2));
    }

    let other_memory = EpisodicMemory::open_scoped(b1_path.clone(), other_scope)?;
    other_memory.set_capture_enabled(true).await?;
    let other_history = record_history(&corpus.other_project_record);
    let other_id = other_memory
        .record_settled_turn(
            &corpus.other_project_record.user,
            &other_history,
            EpisodeOutcome::Completed,
            "evaluation",
            "deterministic",
            Utc::now(),
        )
        .await?
        .ok_or_else(|| generalist::Error::Other("other-scope capture was paused".into()))?;

    let b1_memory = EpisodicMemory::open_scoped(b1_path.clone(), scope.clone())?;
    let b0_episodes = b0_memory.status().await?.episode_count;
    let b1_episodes_before_delete = b1_memory.status().await?.episode_count;

    let mut b0_retrieved = Vec::new();
    let mut b1_retrieved = Vec::new();
    let mut autosave_retrieved = Vec::new();
    let mut named_retrieved = Vec::new();
    let mut memory_registry = ToolRegistry::new();
    memory_registry.register(Arc::new(SearchMemoriesTool::new(b1_memory.clone())))?;
    let mut autosave_registry = ToolRegistry::new();
    autosave_registry.register(Arc::new(SearchConversationsTool::new(
        autosave_store.clone(),
    )))?;
    let mut named_registry = ToolRegistry::new();
    named_registry.register(Arc::new(SearchConversationsTool::new(named_store.clone())))?;
    let mut autosave_aliases = HashMap::new();
    autosave_aliases.insert(
        "autosave".to_string(),
        corpus.records.last().unwrap().id.clone(),
    );
    let named_aliases = corpus
        .records
        .iter()
        .map(|record| (record.id.clone(), record.id.clone()))
        .collect::<HashMap<_, _>>();

    for (index, query) in corpus.queries.iter().enumerate() {
        b0_retrieved.push(
            b0_memory
                .search(&query.query)
                .await?
                .into_iter()
                .map(|match_| match_.id)
                .collect(),
        );
        b1_retrieved.push(search_memory_tool(&mut memory_registry, &query.query, index).await?);
        autosave_retrieved.push(search_history(&mut autosave_registry, &query.query, index).await?);
        named_retrieved.push(search_history(&mut named_registry, &query.query, index).await?);
    }
    let empty_aliases = HashMap::new();
    let mut retrieval = BTreeMap::new();
    retrieval.insert(
        "b0_paused".to_string(),
        score_queries(&corpus, &b0_retrieved, &empty_aliases),
    );
    retrieval.insert(
        "b1_episodic".to_string(),
        score_queries(&corpus, &b1_retrieved, &b1_aliases),
    );
    retrieval.insert(
        "history_autosave".to_string(),
        score_queries(&corpus, &autosave_retrieved, &autosave_aliases),
    );
    retrieval.insert(
        "history_named".to_string(),
        score_queries(&corpus, &named_retrieved, &named_aliases),
    );

    let mut memory_latency = Vec::new();
    let mut autosave_latency = Vec::new();
    let mut named_latency = Vec::new();
    let mut call_index = corpus.queries.len();
    for _ in 0..LATENCY_SAMPLES_PER_QUERY {
        for query in &corpus.queries {
            let started = Instant::now();
            let _ = search_memory_tool(&mut memory_registry, &query.query, call_index).await?;
            memory_latency.push(started.elapsed().as_secs_f64() * 1_000.0);
            call_index += 1;
            let started = Instant::now();
            let _ = search_history(&mut autosave_registry, &query.query, call_index).await?;
            autosave_latency.push(started.elapsed().as_secs_f64() * 1_000.0);
            call_index += 1;
            let started = Instant::now();
            let _ = search_history(&mut named_registry, &query.query, call_index).await?;
            named_latency.push(started.elapsed().as_secs_f64() * 1_000.0);
            call_index += 1;
        }
    }
    let mut latency_ms = BTreeMap::new();
    latency_ms.insert("b1_episodic".into(), latency_report(memory_latency));
    latency_ms.insert("history_autosave".into(), latency_report(autosave_latency));
    latency_ms.insert("history_named".into(), latency_report(named_latency));

    let current_matches = b1_memory.search("CONV-AURORA-17").await?;
    let grant_registry_result = memory_registry
        .execute_tool(
            "search_memories",
            json!({"query": "CONV-AURORA-17", "scope": "all"}),
            "scope-all-eval".into(),
        )
        .await;
    let all_value: Value = serde_json::from_str(&tool_result_text(grant_registry_result)?)?;
    let all_scope_match_count = all_value["matches"].as_array().map_or(0, Vec::len);
    let scope_isolation = ScopeIsolationReport {
        current_scope_matches: current_matches
            .iter()
            .filter_map(|summary| b1_aliases.get(&summary.id).cloned())
            .collect(),
        all_scope_match_count,
        leaked_other_project_record: current_matches.iter().any(|summary| summary.id == other_id),
    };

    let retained_export = b1_memory.export().await?;
    let b0_logical_bytes = serde_json::to_vec(&b0_memory.export().await?)?.len() as u64;
    let logical_memory_bytes = serde_json::to_vec(&retained_export)?.len() as u64;
    let b1_allocated_before_delete = database_bytes(&b1_path)?;
    let deletion_target = b1_aliases
        .iter()
        .find_map(|(id, record)| (record == "rare_safety").then(|| id.clone()))
        .ok_or_else(|| generalist::Error::Other("missing deletion target".into()))?;
    let forget = b1_memory.forget(&deletion_target).await?;
    let live_search_absent = b1_memory.search("SAFE-QUARTZ-73").await?.is_empty();
    let exported_snapshot_still_contains_record = retained_export
        .iter()
        .any(|episode| episode.id == deletion_target);
    drop(b1_memory);
    let reopened = EpisodicMemory::open_scoped(b1_path.clone(), scope.clone())?;
    let restart_search_absent = reopened.search("SAFE-QUARTZ-73").await?.is_empty();
    let b1_allocated_after_delete = database_bytes(&b1_path)?;
    let deletion = DeletionReport {
        live_search_absent,
        restart_search_absent,
        exported_snapshot_still_contains_record,
        checkpoint_completed: matches!(forget, ForgetResult::Deleted),
    };

    let lock_connection = Connection::open(&b1_path)?;
    lock_connection.execute_batch("BEGIN IMMEDIATE")?;
    let capture_before = reopened.status().await?.capture_enabled;
    let started = Instant::now();
    let mut ticks = 0_u64;
    let operation = reopened.set_capture_enabled(false);
    tokio::pin!(operation);
    let mut ticker = tokio::time::interval(Duration::from_millis(10));
    let locked_result = loop {
        tokio::select! {
            result = &mut operation => break result,
            _ = ticker.tick() => ticks += 1,
        }
    };
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    lock_connection.execute_batch("ROLLBACK")?;
    let capture_after = reopened.status().await?.capture_enabled;
    let write_lock = WriteLockReport {
        elapsed_ms,
        executor_ticks_while_waiting: ticks,
        returned_locked_error: locked_result
            .err()
            .is_some_and(|error| error.to_string().contains("database is locked")),
        setting_unchanged: capture_before == capture_after,
    };

    let mut retained_bytes = BTreeMap::new();
    retained_bytes.insert(
        "b0_allocated_sqlite_with_sidecars".into(),
        database_bytes(&b0_path)?,
    );
    retained_bytes.insert("b0_logical_export".into(), b0_logical_bytes);
    retained_bytes.insert(
        "b1_allocated_sqlite_before_delete".into(),
        b1_allocated_before_delete,
    );
    retained_bytes.insert(
        "b1_allocated_sqlite_after_delete".into(),
        b1_allocated_after_delete,
    );
    retained_bytes.insert("b1_logical_export".into(), logical_memory_bytes);
    retained_bytes.insert(
        "history_autosave".into(),
        directory_bytes(autosave_store.directory())?,
    );
    retained_bytes.insert(
        "history_named".into(),
        directory_bytes(named_store.directory())?,
    );

    let exit_boundary = evaluate_exit_boundary(temp.path()).await?;
    let b1 = retrieval.get("b1_episodic").unwrap();
    let named = retrieval.get("history_named").unwrap();
    let passed = b0_episodes == 0
        && b1_episodes_before_delete == corpus.records.len() as u64
        && b1.recall == 1.0
        && named.recall == 1.0
        && !scope_isolation.leaked_other_project_record
        && scope_isolation.all_scope_match_count == 2
        && deletion.live_search_absent
        && deletion.restart_search_absent
        && deletion.exported_snapshot_still_contains_record
        && write_lock.returned_locked_error
        && write_lock.setting_unchanged
        && write_lock.executor_ticks_while_waiting > 1
        && exit_boundary.enqueue_absent + exit_boundary.enqueue_complete
            == exit_boundary.enqueue_attempts
        && exit_boundary.acknowledged_complete == exit_boundary.acknowledged_attempts
        && exit_boundary.locked_absent == exit_boundary.locked_attempts
        && exit_boundary.duplicate_or_partial_rows == 0
        && exit_boundary.immutable_update_rejected
        && exit_boundary.integrity_check == "ok";

    Ok(EvaluationReport {
        schema_version: 1,
        corpus_schema_version: corpus.schema_version,
        experiment: "explicit episodic memory B0/B1 deterministic lifecycle evaluation",
        caveat:
            "Storage/retrieval mechanics only; this run does not establish model answer quality.",
        counts: CountReport {
            corpus_records: corpus.records.len(),
            scored_queries: corpus.queries.len(),
            b0_episodes,
            b1_episodes_before_delete,
            autosave_archives: autosave_store.list().len(),
            named_archives: named_store.list().len(),
        },
        retrieval,
        latency_ms,
        retained_bytes,
        scope_isolation,
        deletion,
        write_lock,
        exit_boundary,
        passed,
    })
}

fn main() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("__child_enqueue") => return child_enqueue(&args),
        Some("__child_acknowledged") => return child_acknowledged(&args),
        Some(argument) => {
            return Err(generalist::Error::Other(format!(
                "unknown memory evaluation argument '{argument}'"
            ))
            .into())
        }
        None => {}
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| generalist::Error::Other(error.to_string()))?;
    let report = runtime.block_on(evaluate())?;
    println!("{}", serde_json::to_string(&report)?);
    if !report.passed {
        std::process::exit(1);
    }
    Ok(())
}
