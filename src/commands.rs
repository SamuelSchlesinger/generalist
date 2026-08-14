//! Pure executors for local slash commands (`/permissions`, `/tools`,
//! `/history`, `/memory`): they take the relevant runtime handles and return
//! display text, with no reactor or terminal coupling.

use crate::write_atomically;
use generalist::{
    truncate_middle, Agent, ArchivedConversation, ArchivedConversationEvent, Episode, EpisodeEvent,
    EpisodicMemory, Error, ForgetResult, HistoryCommand, HistoryStore, MemoryCommand,
    MemoryPermissionHandler, PermissionCommand, Result, ToolDef, ToolsCommand,
};
use std::collections::HashSet;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// Run one blocking history-store operation off the reactor thread, so
/// spawned background tasks (stream collectors, MCP readers) keep making
/// progress while files are read, written, and fsynced.
pub(crate) async fn with_history_store<T: Send + 'static>(
    history_store: &HistoryStore,
    operation: impl FnOnce(HistoryStore) -> T + Send + 'static,
) -> Result<T> {
    let store = history_store.clone();
    tokio::task::spawn_blocking(move || operation(store))
        .await
        .map_err(|error| Error::Other(format!("History worker failed: {error}")))
}

pub(crate) fn write_memory_export(directory: &Path, episodes: &[Episode]) -> Result<PathBuf> {
    fs::create_dir_all(directory).map_err(|error| {
        Error::Other(format!(
            "Failed to create export directory {}: {error}",
            directory.display()
        ))
    })?;
    fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).map_err(|error| {
        Error::Other(format!(
            "Failed to restrict export directory {}: {error}",
            directory.display()
        ))
    })?;
    let export_id = uuid::Uuid::new_v4().to_string();
    let path = directory.join(format!(
        "episodes-{}-{}.json",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        &export_id[..8]
    ));
    let contents = serde_json::to_vec_pretty(episodes)
        .map_err(|error| Error::Other(format!("Failed to serialize episode export: {error}")))?;
    write_atomically(&path, &contents)?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).map_err(|error| {
        Error::Other(format!(
            "Failed to restrict episode export {}: {error}",
            path.display()
        ))
    })?;
    Ok(path)
}

pub(crate) fn run_permission_command(
    command: PermissionCommand<'_>,
    handler: &MemoryPermissionHandler,
) -> String {
    match command {
        PermissionCommand::List => {
            let policy = handler.remembered_policy();
            let exact = handler.session_exact_allow_counts();
            let mut always_allow = policy.always_allow.into_iter().collect::<Vec<_>>();
            let mut always_deny = policy.always_deny.into_iter().collect::<Vec<_>>();
            always_allow.sort_unstable();
            always_deny.sort_unstable();
            if always_allow.is_empty() && always_deny.is_empty() && exact.is_empty() {
                return "No remembered tool permissions; the next permissioned use of each tool will ask."
                    .to_string();
            }
            let mut lines = vec!["Remembered tool permissions:".to_string()];
            lines.push("Always allow:".to_string());
            if always_allow.is_empty() {
                lines.push("  (none)".to_string());
            } else {
                lines.extend(always_allow.into_iter().map(|tool| format!("  {tool}")));
            }
            lines.push("Always deny:".to_string());
            if always_deny.is_empty() {
                lines.push("  (none)".to_string());
            } else {
                lines.extend(always_deny.into_iter().map(|tool| format!("  {tool}")));
            }
            if !exact.is_empty() {
                lines.push("Session-only exact-input allows:".to_string());
                lines.extend(exact.into_iter().map(|(tool, count)| {
                    format!(
                        "  {tool}: {count} input{}",
                        if count == 1 { "" } else { "s" }
                    )
                }));
            }
            lines.join("\n")
        }
        PermissionCommand::Reset(tool) => {
            if handler.reset_remembered_tool(tool) {
                format!(
                    "Reset the remembered permission for '{tool}'; its next permissioned use will ask."
                )
            } else {
                format!("No remembered permission exists for '{tool}'.")
            }
        }
        PermissionCommand::Clear => {
            let count = handler.clear_remembered_policy();
            if count == 0 {
                "No remembered tool permissions to clear.".to_string()
            } else {
                format!(
                    "Cleared {count} remembered tool permission(s); their next permissioned use will ask."
                )
            }
        }
    }
}

pub(crate) const TOOL_LIST_LIMIT: usize = 60;
pub(crate) const TOOL_SUMMARY_CHARS: usize = 180;
pub(crate) const TOOL_DETAIL_CHARS: usize = 30_000;
pub(crate) const HISTORY_LIST_LIMIT: usize = 60;
pub(crate) const HISTORY_DETAIL_CHARS: usize = 30_000;

pub(crate) fn one_line_tool_summary(description: &str) -> String {
    let flattened = description.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_middle(&flattened, TOOL_SUMMARY_CHARS)
}

pub(crate) fn bounded_tool_detail(detail: String) -> String {
    let count = detail.chars().count();
    if count <= TOOL_DETAIL_CHARS {
        return detail;
    }
    let kept = detail.chars().take(TOOL_DETAIL_CHARS).collect::<String>();
    format!(
        "{kept}\n\n[{} characters omitted; this display is bounded]",
        count - TOOL_DETAIL_CHARS
    )
}

pub(crate) fn advertised_interface(agent: &Agent) -> String {
    let names = agent
        .advertised_tool_defs()
        .into_iter()
        .map(|definition| definition.name)
        .collect::<Vec<_>>();
    if names.is_empty() {
        "Model-facing tools: (none)".to_string()
    } else {
        format!("Model-facing tools: {}", names.join(", "))
    }
}

pub(crate) fn sorted_bridge_catalog(agent: &Agent) -> (Vec<ToolDef>, HashSet<String>) {
    let mut definitions = agent.registry.get_bridge_tool_defs();
    definitions.sort_by(|left, right| {
        left.name
            .to_ascii_lowercase()
            .cmp(&right.name.to_ascii_lowercase())
            .then_with(|| left.name.cmp(&right.name))
    });
    let progressive = agent
        .registry
        .code_only_tool_defs()
        .into_iter()
        .map(|definition| definition.name)
        .collect();
    (definitions, progressive)
}

pub(crate) fn find_named_tool<'a>(
    definitions: &'a [ToolDef],
    name: &str,
) -> Result<Option<&'a ToolDef>> {
    if let Some(definition) = definitions
        .iter()
        .find(|definition| definition.name == name)
    {
        return Ok(Some(definition));
    }
    let matches = definitions
        .iter()
        .filter(|definition| definition.name.eq_ignore_ascii_case(name))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [definition] => Ok(Some(*definition)),
        _ => Err(Error::Other(format!(
            "Tool name '{name}' is ambiguous; use exact case: {}",
            matches
                .iter()
                .map(|definition| definition.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

pub(crate) fn render_tool_definition(name: &str, exposure: &str, definition: &ToolDef) -> String {
    let schema = serde_json::to_string_pretty(&definition.input_schema)
        .unwrap_or_else(|_| definition.input_schema.to_string());
    bounded_tool_detail(format!(
        "Tool {name}\nExposure: {exposure}\n\nDescription:\n{}\n\nInput schema:\n{schema}",
        definition.description
    ))
}

pub(crate) fn run_tools_command(command: ToolsCommand<'_>, agent: &Agent) -> String {
    let (bridges, progressive) = sorted_bridge_catalog(agent);
    match command {
        ToolsCommand::List | ToolsCommand::Search(_) => {
            let query = match command {
                ToolsCommand::Search(query) => Some(query),
                ToolsCommand::List => None,
                ToolsCommand::Show(_) => unreachable!(),
            };
            let normalized_query = query.map(|query| query.to_lowercase());
            let matches = bridges
                .iter()
                .filter(|definition| {
                    normalized_query.as_ref().is_none_or(|query| {
                        definition.name.to_lowercase().contains(query)
                            || definition.description.to_lowercase().contains(query)
                    })
                })
                .collect::<Vec<_>>();
            let progressive_count = bridges
                .iter()
                .filter(|definition| progressive.contains(&definition.name))
                .count();
            let mut lines = vec![advertised_interface(agent)];
            lines.push(if let Some(query) = query {
                format!(
                    "Registered bridge tools matching '{query}': {} of {}",
                    matches.len(),
                    bridges.len()
                )
            } else {
                format!(
                    "Registered bridge tools: {} total ({progressive_count} schema-on-demand)",
                    bridges.len()
                )
            });
            if matches.is_empty() {
                lines.push("  (no matches)".to_string());
            } else {
                lines.extend(matches.iter().take(TOOL_LIST_LIMIT).map(|definition| {
                    let qualifier = if progressive.contains(&definition.name) {
                        " [schema on demand]"
                    } else {
                        ""
                    };
                    format!(
                        "  tools.{}{qualifier} — {}",
                        definition.name,
                        one_line_tool_summary(&definition.description)
                    )
                }));
                if matches.len() > TOOL_LIST_LIMIT {
                    lines.push(format!(
                        "  … {} more omitted; narrow the list with /tools search <query>",
                        matches.len() - TOOL_LIST_LIMIT
                    ));
                }
            }
            lines.push("Use /tools show <name> for one description and input schema.".to_string());
            lines.join("\n")
        }
        ToolsCommand::Show(name) => {
            match find_named_tool(&bridges, name) {
                Ok(Some(definition)) => {
                    let exposure = if progressive.contains(&definition.name) {
                        "bridge capability; full schema is available to scripts on demand via __doc__"
                    } else if agent.uses_builtin_code_mode() {
                        "bridge capability; compact signature and description are preloaded in the python runner"
                    } else {
                        "registered model-facing capability"
                    };
                    return render_tool_definition(
                        &format!("tools.{}", definition.name),
                        exposure,
                        definition,
                    );
                }
                Err(error) => return error.to_string(),
                Ok(None) => {}
            }

            let advertised = agent.advertised_tool_defs();
            match find_named_tool(&advertised, name) {
                Ok(Some(definition)) => {
                    let exposure = if definition.name == "python" && agent.uses_builtin_code_mode()
                    {
                        "model-facing built-in runner; registered capabilities are called through tools.<name> inside its script"
                    } else if definition.name == generalist::UPDATE_GOAL_TOOL_NAME {
                        "model-facing host control while an objective is active; permission-free and not a bridge capability"
                    } else {
                        "model-facing capability"
                    };
                    render_tool_definition(&definition.name, exposure, definition)
                }
                Err(error) => error.to_string(),
                Ok(None) if name.eq_ignore_ascii_case(generalist::UPDATE_GOAL_TOOL_NAME) => {
                    "Tool 'update_goal' is advertised only while an active goal exists.".to_string()
                }
                Ok(None) => format!(
                    "No registered or currently advertised tool named '{name}'. Use /tools search <query>."
                ),
            }
        }
    }
}

pub(crate) fn bounded_history_detail(detail: String) -> String {
    let count = detail.chars().count();
    if count <= HISTORY_DETAIL_CHARS {
        return detail;
    }
    let kept = detail
        .chars()
        .take(HISTORY_DETAIL_CHARS)
        .collect::<String>();
    format!(
        "{kept}\n\n[{} characters omitted; inspect the saved session file for the complete sanitized view]",
        count - HISTORY_DETAIL_CHARS
    )
}

pub(crate) fn render_archived_conversation(conversation: &ArchivedConversation) -> String {
    let mut lines = vec![
        format!("Saved session {}", conversation.name),
        format!(
            "{} · {} · {} / {}",
            conversation.updated_at.format("%Y-%m-%d %H:%M:%S UTC"),
            conversation.scope,
            conversation.provider,
            conversation.model
        ),
    ];
    if let Some(goal) = &conversation.goal {
        lines.push(format!(
            "prospective goal (not a past event):\n{}",
            truncate_middle(goal, 1_200)
        ));
    }
    for event in &conversation.events {
        match event {
            ArchivedConversationEvent::UserText { text } => {
                lines.push(format!("user:\n{}", truncate_middle(text, 1_200)));
            }
            ArchivedConversationEvent::AssistantText { text } => {
                lines.push(format!("assistant:\n{}", truncate_middle(text, 1_200)));
            }
            ArchivedConversationEvent::ToolCall { name } => {
                lines.push(format!("tool: {name} (input omitted)"));
            }
            ArchivedConversationEvent::ToolResult { is_error } => {
                lines.push(format!(
                    "tool result: {} (content omitted)",
                    if *is_error { "error" } else { "success" }
                ));
            }
        }
    }
    bounded_history_detail(lines.join("\n\n"))
}

pub(crate) async fn run_history_command(
    command: HistoryCommand<'_>,
    history: &HistoryStore,
) -> Result<String> {
    match command {
        HistoryCommand::List => {
            let names = with_history_store(history, move |store| store.list()).await?;
            let mut lines = vec![format!(
                "Saved sessions in {}: {}",
                history.scope().display_name(),
                names.len()
            )];
            if names.is_empty() {
                lines.push("  (none)".to_string());
            } else {
                lines.extend(
                    names
                        .iter()
                        .take(HISTORY_LIST_LIMIT)
                        .map(|name| format!("  {name}")),
                );
                if names.len() > HISTORY_LIST_LIMIT {
                    lines.push(format!(
                        "  … {} more omitted; narrow the list with /history search <query>",
                        names.len() - HISTORY_LIST_LIMIT
                    ));
                }
            }
            lines.push("Use /history show <name> to inspect without loading.".to_string());
            Ok(lines.join("\n"))
        }
        HistoryCommand::Search(query) => {
            let requested = query.to_string();
            let matches = with_history_store(history, move |store| {
                store.search_current_archives(&requested)
            })
            .await??;
            if matches.is_empty() {
                return Ok(format!(
                    "No current-scope saved sessions matched ‘{query}’."
                ));
            }
            let mut lines = vec![format!(
                "{} current-scope saved session(s) matched ‘{query}’:",
                matches.len()
            )];
            lines.extend(matches.into_iter().map(|conversation| {
                format!(
                    "{} · {} · {} / {} · {}",
                    conversation.name,
                    conversation.updated_at.format("%Y-%m-%d %H:%M"),
                    conversation.provider,
                    conversation.model,
                    conversation.preview
                )
            }));
            lines.push("Use /history show <name> to inspect without loading.".to_string());
            Ok(lines.join("\n"))
        }
        HistoryCommand::Show(name) => {
            let requested = name.to_string();
            match with_history_store(history, move |store| {
                store.inspect_current_archive(&requested)
            })
            .await??
            {
                Some(conversation) => Ok(render_archived_conversation(&conversation)),
                None => Ok(format!(
                    "No current-scope saved session named '{name}'. Use /history list."
                )),
            }
        }
        HistoryCommand::Forget(_) => Err(Error::Other(
            "Forgetting a saved session requires interactive host confirmation".to_string(),
        )),
    }
}

pub(crate) fn render_episode(episode: &Episode) -> String {
    let mut lines = vec![
        format!("Episode {}", episode.id),
        format!(
            "{} · {} · {} / {}",
            episode.settled_at.format("%Y-%m-%d %H:%M:%S UTC"),
            episode.outcome.label(),
            episode.provider,
            episode.model
        ),
        format!(
            "Capture: {} · project: {}",
            episode.capture_quality, episode.project_root
        ),
    ];
    for event in &episode.events {
        match event {
            EpisodeEvent::UserText { text } => {
                lines.push(format!("user:\n{}", truncate_middle(text, 1_200)));
            }
            EpisodeEvent::AssistantText { text } => {
                lines.push(format!("assistant:\n{}", truncate_middle(text, 1_200)));
            }
            EpisodeEvent::ToolCall { name, .. } => {
                lines.push(format!("tool: {name} (input omitted)"));
            }
            EpisodeEvent::ToolResult { is_error, .. } => {
                lines.push(format!(
                    "tool result: {} (content omitted)",
                    if *is_error { "error" } else { "success" }
                ));
            }
        }
    }
    lines.join("\n\n")
}

pub(crate) async fn run_memory_command(
    command: MemoryCommand<'_>,
    memory: &EpisodicMemory,
    exports_directory: &Path,
) -> Result<Vec<String>> {
    match command {
        MemoryCommand::Status => {
            let status = memory.status().await?;
            Ok(vec![format!(
                "Episodic memory: {} · {} episode(s)\nScope: {}\nSQLite {} at {}\n\
                 Explicit search only; no automatic retrieval. User/assistant text is retained; \
                 provider reasoning and tool payloads are omitted.",
                if status.capture_enabled {
                    "recording"
                } else {
                    "paused"
                },
                status.episode_count,
                status.project_root,
                status.sqlite_version,
                status.database_path.display()
            )])
        }
        MemoryCommand::Pause => {
            memory.set_capture_enabled(false).await?;
            Ok(vec![
                "Episodic capture paused for the current scope.".to_string()
            ])
        }
        MemoryCommand::Resume => {
            memory.set_capture_enabled(true).await?;
            Ok(vec![
                "Episodic capture enabled for the current scope. Future settled user/assistant text \
                 is retained; provider reasoning and tool payloads are omitted. Pause capture \
                 before entering sensitive text."
                    .to_string(),
            ])
        }
        MemoryCommand::Search(query) => {
            let matches = memory.search(query).await?;
            let mut lines = Vec::new();
            if matches.summaries.is_empty() {
                lines.push(format!("No current-scope episodes matched “{query}”."));
            } else {
                lines.push(format!(
                    "{} current-scope episode(s) matched “{query}”:",
                    matches.summaries.len()
                ));
                for episode in matches.summaries {
                    let short_id: String = episode.id.chars().take(8).collect();
                    lines.push(format!(
                        "{} · {} · {} · {}",
                        short_id,
                        episode.settled_at.format("%Y-%m-%d %H:%M"),
                        episode.outcome.label(),
                        episode.preview
                    ));
                }
            }
            for corrupt in matches.corrupt {
                lines.push(format!(
                    "Skipped corrupt episode {}: {} (`/memory forget {}` removes it)",
                    corrupt.id, corrupt.error, corrupt.id
                ));
            }
            Ok(lines)
        }
        MemoryCommand::Show(id) => match memory.show(id).await? {
            Some(episode) => Ok(vec![render_episode(&episode)]),
            None => Ok(vec![format!(
                "No current-scope episode matches ID prefix '{id}'."
            )]),
        },
        MemoryCommand::Export => {
            let export = memory.export().await?;
            let count = export.episodes.len();
            let corrupt = export.corrupt;
            let exports_directory = exports_directory.to_path_buf();
            let path = tokio::task::spawn_blocking(move || {
                write_memory_export(&exports_directory, &export.episodes)
            })
            .await
            .map_err(|error| Error::Other(format!("Episode export worker failed: {error}")))??;
            let mut lines = vec![format!(
                "Exported {count} current-scope episode(s) to {}",
                path.display()
            )];
            for corrupt in corrupt {
                lines.push(format!(
                    "Skipped corrupt episode {}: {} (`/memory forget {}` removes it)",
                    corrupt.id, corrupt.error, corrupt.id
                ));
            }
            Ok(lines)
        }
        MemoryCommand::Forget(id) => match memory.forget(id).await? {
            ForgetResult::Deleted => Ok(vec![
                "Episode deleted from the live SQLite store. This does not erase external \
                     exports, backups, or filesystem snapshots."
                    .to_string(),
            ]),
            ForgetResult::DeletedCheckpointPending(error) => Ok(vec![format!(
                "Episode deleted from live queries, but WAL truncation is still pending: \
                     {error}. Prior exports, backups, and filesystem snapshots are outside this \
                     guarantee."
            )]),
            ForgetResult::NotFound => Ok(vec![format!(
                "No current-scope episode matches ID prefix '{id}'."
            )]),
        },
    }
}
