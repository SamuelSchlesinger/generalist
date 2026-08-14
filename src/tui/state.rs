//! The TUI's application state: conversation, reasoning, activity, queue,
//! composer, and modal state, plus the projection of [`AgentEvent`]s onto it.

use crate::command::{complete_local_command, CommandCompletion};
use crate::permissions::ToolExecutionRequest;
use crate::runtime::{PromptId, PromptSource, QueuedPrompt};
use crate::types::{truncate_middle, ContentBlock, Message, Usage};
use crate::{AgentEvent, ToolCallOutcome};
use chrono::Local;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

use super::editor::{apply_editor_key, insert_char, insert_text};
use super::render::chat_kind_label;
use super::sanitize_terminal_text;

const MAX_ACTIVITY_ITEMS: usize = 100;
const DEFAULT_MAX_DISPLAY_CHARS: usize = 2_000;
pub(crate) const MAX_CONVERSATION_DISPLAY_CHARS: usize = 64 * 1024;
const ASSISTANT_DISPLAY_MARKER: &str =
    "\n\n[Middle omitted from the UI; use /copy last for the full committed response.]\n\n";
const USER_DISPLAY_MARKER: &str =
    "\n\n[Middle omitted from the UI; use /copy all for the full committed prompt.]\n\n";
const REASONING_DISPLAY_MARKER: &str =
    "\n\n[Middle omitted from the UI; use /copy reasoning for the full inspectable committed reasoning.]\n\n";
const LIVE_DISPLAY_MARKER: &str =
    "\n\n[Live preview capped in the UI; the committed response will replace it.]\n\n";
const LOCAL_DISPLAY_MARKER: &str = "\n\n[Middle omitted from the UI.]\n\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatKind {
    User,
    Assistant,
    Info,
    Error,
}

#[derive(Debug, Clone)]
pub(crate) struct ChatEntry {
    pub(crate) kind: ChatKind,
    pub(crate) timestamp: String,
    pub(crate) body: String,
    pub(crate) display_capped: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ChatSearch {
    pub(crate) query: String,
    pub(crate) cursor: usize,
    pub(crate) matches: Vec<usize>,
    pub(crate) selected: usize,
}

impl ChatSearch {
    fn refresh(&mut self, entries: &[ChatEntry]) {
        let previously_selected = self.matches.get(self.selected).copied();
        let needle = self.query.trim().to_lowercase();
        self.matches = if needle.is_empty() {
            Vec::new()
        } else {
            entries
                .iter()
                .enumerate()
                .filter_map(|(index, entry)| {
                    let (label, _) = chat_kind_label(entry.kind);
                    (label.to_lowercase().contains(&needle)
                        || entry.body.to_lowercase().contains(&needle))
                    .then_some(index)
                })
                .collect()
        };
        self.selected = previously_selected
            .and_then(|entry| {
                self.matches
                    .iter()
                    .position(|candidate| *candidate == entry)
            })
            .unwrap_or(0)
            .min(self.matches.len().saturating_sub(1));
        self.cursor = self.cursor.min(self.query.chars().count());
    }

    pub(crate) fn selected_entry(&self) -> Option<usize> {
        self.matches.get(self.selected).copied()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReasoningEntry {
    pub(crate) timestamp: String,
    pub(crate) body: String,
    pub(crate) display_capped: bool,
    pub(crate) finished: bool,
    pub(crate) abort_reason: Option<String>,
}

fn project_display(value: &str, marker: &str) -> (String, bool) {
    let value = sanitize_terminal_text(value);
    let count = value.chars().count();
    if count <= MAX_CONVERSATION_DISPLAY_CHARS {
        return (value, false);
    }

    let marker_chars = marker.chars().count();
    if marker_chars >= MAX_CONVERSATION_DISPLAY_CHARS {
        return (
            marker
                .chars()
                .take(MAX_CONVERSATION_DISPLAY_CHARS)
                .collect(),
            true,
        );
    }
    let retained = MAX_CONVERSATION_DISPLAY_CHARS - marker_chars;
    let start_chars = retained / 2;
    let end_chars = retained - start_chars;
    let start = value.chars().take(start_chars).collect::<String>();
    let end = value.chars().skip(count - end_chars).collect::<String>();
    (format!("{start}{marker}{end}"), true)
}

fn project_chat_display(kind: ChatKind, value: &str) -> (String, bool) {
    let marker = match kind {
        ChatKind::Assistant => ASSISTANT_DISPLAY_MARKER,
        ChatKind::User => USER_DISPLAY_MARKER,
        ChatKind::Info | ChatKind::Error => LOCAL_DISPLAY_MARKER,
    };
    project_display(value, marker)
}

fn project_reasoning_display(value: &str) -> (String, bool) {
    project_display(value, REASONING_DISPLAY_MARKER)
}

/// Append already-sanitized streamed text to a live entry body, re-projecting
/// through [`LIVE_DISPLAY_MARKER`] once the accumulated text exceeds the
/// display budget. Returns whether the body is now display-capped.
fn append_live_delta(body: &mut String, delta: &str) -> bool {
    body.push_str(delta);
    if body.chars().count() > MAX_CONVERSATION_DISPLAY_CHARS {
        let (capped, display_capped) = project_display(body, LIVE_DISPLAY_MARKER);
        *body = capped;
        return display_capped;
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivityStatus {
    Running,
    Success,
    Failed,
    Rejected,
    Denied,
    Cancelled,
}

#[derive(Debug, Clone)]
pub(crate) struct ActivityEntry {
    pub(crate) name: String,
    pub(crate) via_code_mode: bool,
    pub(crate) status: ActivityStatus,
    pub(crate) timestamp: String,
    pub(crate) input: String,
    pub(crate) output: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SourceToolInput<'a> {
    pub(crate) field: &'static str,
    pub(crate) title: &'static str,
    pub(crate) source: &'a str,
}

pub(crate) fn source_tool_spec(tool_name: &str) -> Option<(&'static str, &'static str)> {
    match tool_name {
        "python" => Some(("code", "Python source")),
        "bash" => Some(("command", "Shell command")),
        _ => None,
    }
}

pub(crate) fn source_tool_input<'a>(
    tool_name: &str,
    input: &'a Value,
) -> Option<SourceToolInput<'a>> {
    let (field, title) = source_tool_spec(tool_name)?;
    let source = input.get(field)?.as_str()?;
    Some(SourceToolInput {
        field,
        title,
        source,
    })
}

pub(crate) fn tool_input_preview(tool_name: &str, input: &Value, max_chars: usize) -> String {
    let preview = if let Some(source) = source_tool_input(tool_name, input) {
        let source_text = sanitize_terminal_text(source.source);
        if source_text.is_empty() {
            format!("<empty {}>", source.field)
        } else {
            source_text
        }
    } else {
        sanitize_terminal_text(&serde_json::to_string(input).unwrap_or_default())
    };
    truncate_middle(&preview, max_chars)
}

#[derive(Debug, Clone)]
pub(crate) enum Modal {
    Help,
    Select {
        title: String,
        items: Vec<String>,
        selected: usize,
    },
    Prompt {
        title: String,
        value: String,
        cursor: usize,
    },
    Permission {
        id: u64,
        request: ToolExecutionRequest,
        selected: usize,
        scroll: u16,
    },
    Queue {
        selected: usize,
        editing: Option<QueueEditor>,
    },
    Search(ChatSearch),
    Reasoning {
        scroll: usize,
        max_scroll: usize,
        follow_latest: bool,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct QueueEditor {
    pub(crate) id: PromptId,
    pub(crate) value: String,
    pub(crate) cursor: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProviderUsageTotals {
    pub(crate) attempts: u64,
    pub(crate) usage_reports: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
    pub(crate) cache_creation_input_tokens: u64,
    pub(crate) cache_read_reports: u64,
    pub(crate) cache_creation_reports: u64,
}

impl ProviderUsageTotals {
    pub(crate) fn record_attempt(&mut self) {
        self.attempts = self.attempts.saturating_add(1);
    }

    pub(crate) fn record_report(&mut self, usage: &Usage) {
        self.usage_reports = self.usage_reports.saturating_add(1);
        self.input_tokens = self.input_tokens.saturating_add(usage.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(usage.output_tokens);
        if let Some(tokens) = usage.cache_read_input_tokens {
            self.cache_read_reports = self.cache_read_reports.saturating_add(1);
            self.cache_read_input_tokens = self.cache_read_input_tokens.saturating_add(tokens);
        }
        if let Some(tokens) = usage.cache_creation_input_tokens {
            self.cache_creation_reports = self.cache_creation_reports.saturating_add(1);
            self.cache_creation_input_tokens =
                self.cache_creation_input_tokens.saturating_add(tokens);
        }
    }

    fn merge(&mut self, other: &Self) {
        self.attempts = self.attempts.saturating_add(other.attempts);
        self.usage_reports = self.usage_reports.saturating_add(other.usage_reports);
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .saturating_add(other.cache_read_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(other.cache_creation_input_tokens);
        self.cache_read_reports = self
            .cache_read_reports
            .saturating_add(other.cache_read_reports);
        self.cache_creation_reports = self
            .cache_creation_reports
            .saturating_add(other.cache_creation_reports);
    }

    fn unreported_attempts(&self) -> u64 {
        self.attempts.saturating_sub(self.usage_reports)
    }
}

#[derive(Debug)]
pub(crate) struct AppState {
    pub(crate) api: String,
    pub(crate) model: String,
    pub(crate) bridge_count: usize,
    pub(crate) context_tokens: u64,
    pub(crate) provider_usage: BTreeMap<(String, String), ProviderUsageTotals>,
    pub(crate) active_usage_bucket: Option<(String, String)>,
    pub(crate) goal: Option<String>,
    pub(crate) copy_mode: bool,
    pub(crate) chat: Vec<ChatEntry>,
    pub(crate) reasoning: Vec<ReasoningEntry>,
    pub(crate) active_reasoning: Option<usize>,
    pub(crate) committing_reasoning: Option<usize>,
    pub(crate) activity: Vec<ActivityEntry>,
    pub(crate) active_tools: Vec<(String, usize)>,
    pub(crate) streaming_chat: Option<usize>,
    pub(crate) committing_streaming_chat: Option<usize>,
    pub(crate) input: String,
    pub(crate) input_cursor: usize,
    pub(crate) input_history: Vec<String>,
    pub(crate) history_cursor: Option<usize>,
    pub(crate) history_draft: String,
    pub(crate) chat_scroll: usize,
    pub(crate) chat_max_scroll: usize,
    pub(crate) follow_latest: bool,
    pub(crate) pending_chat_jump: Option<usize>,
    pub(crate) busy: bool,
    pub(crate) turn_active: bool,
    pub(crate) spinner_tick: usize,
    pub(crate) status: String,
    pub(crate) modal: Option<Modal>,
    pub(crate) queue: Vec<QueuedPrompt>,
    pub(crate) max_display_chars: usize,
}

impl AppState {
    pub(crate) fn new(api: impl Into<String>, model: impl Into<String>) -> Self {
        let api = sanitize_terminal_text(&api.into());
        let model = sanitize_terminal_text(&model.into());
        Self {
            api,
            model,
            bridge_count: 0,
            context_tokens: 0,
            provider_usage: BTreeMap::new(),
            active_usage_bucket: None,
            goal: None,
            copy_mode: false,
            chat: Vec::new(),
            reasoning: Vec::new(),
            active_reasoning: None,
            committing_reasoning: None,
            activity: Vec::new(),
            active_tools: Vec::new(),
            streaming_chat: None,
            committing_streaming_chat: None,
            input: String::new(),
            input_cursor: 0,
            input_history: Vec::new(),
            history_cursor: None,
            history_draft: String::new(),
            chat_scroll: 0,
            chat_max_scroll: 0,
            follow_latest: true,
            pending_chat_jump: None,
            busy: false,
            turn_active: false,
            spinner_tick: 0,
            status: "Ready".to_string(),
            modal: None,
            queue: Vec::new(),
            max_display_chars: DEFAULT_MAX_DISPLAY_CHARS,
        }
    }

    fn timestamp() -> String {
        Local::now().format("%H:%M:%S").to_string()
    }

    fn current_usage_bucket(&self) -> (String, String) {
        (self.api.clone(), self.model.clone())
    }

    pub(crate) fn provider_usage_report(&self) -> String {
        if self.provider_usage.is_empty() {
            return "Provider usage: no API attempts recorded in this process. Provider reports, context estimates, and monetary cost are separate."
                .to_string();
        }

        let mut total = ProviderUsageTotals::default();
        let mut lines = vec![
            "Provider usage (process-local provider reports; not a cost estimate):".to_string(),
        ];
        for ((api, model), usage) in &self.provider_usage {
            total.merge(usage);
            lines.push(format!(
                "- {api} / {model}: {}; {}; {}; {} input; {} output; cache read {}; cache creation {}",
                counted(usage.attempts, "attempt", "attempts"),
                counted(usage.usage_reports, "usage report", "usage reports"),
                counted(
                    usage.unreported_attempts(),
                    "unreported attempt",
                    "unreported attempts",
                ),
                usage.input_tokens,
                usage.output_tokens,
                optional_usage_total(
                    usage.cache_read_input_tokens,
                    usage.cache_read_reports,
                    usage.usage_reports,
                ),
                optional_usage_total(
                    usage.cache_creation_input_tokens,
                    usage.cache_creation_reports,
                    usage.usage_reports,
                ),
            ));
        }
        lines.push(format!(
            "Total: {}; {}; {}; {} input; {} output; cache read {}; cache creation {}",
            counted(total.attempts, "attempt", "attempts"),
            counted(total.usage_reports, "usage report", "usage reports"),
            counted(
                total.unreported_attempts(),
                "unreported attempt",
                "unreported attempts",
            ),
            total.input_tokens,
            total.output_tokens,
            optional_usage_total(
                total.cache_read_input_tokens,
                total.cache_read_reports,
                total.usage_reports,
            ),
            optional_usage_total(
                total.cache_creation_input_tokens,
                total.cache_creation_reports,
                total.usage_reports,
            ),
        ));
        lines.push(format!(
            "Current context estimate: {} tokens (separate from cumulative provider reports).",
            self.context_tokens
        ));
        lines.join("\n")
    }

    pub(crate) fn reset_provider_usage(&mut self) {
        let active = self.active_usage_bucket.clone();
        self.provider_usage.clear();
        if let Some(bucket) = active {
            self.provider_usage
                .entry(bucket)
                .or_default()
                .record_attempt();
        }
    }

    pub(crate) fn push_chat(&mut self, kind: ChatKind, body: impl Into<String>) {
        let (body, display_capped) = project_chat_display(kind, &body.into());
        self.chat.push(ChatEntry {
            kind,
            timestamp: Self::timestamp(),
            body,
            display_capped,
        });
        self.refresh_chat_search();
    }

    fn push_chat_delta(&mut self, index: usize, delta: &str) {
        let Some(entry) = self.chat.get_mut(index) else {
            return;
        };
        if entry.display_capped {
            return;
        }
        entry.display_capped = append_live_delta(&mut entry.body, &sanitize_terminal_text(delta));
    }

    pub(crate) fn push_user(&mut self, body: impl Into<String>) {
        self.push_chat(ChatKind::User, body);
    }

    pub(crate) fn push_info(&mut self, body: impl Into<String>) {
        self.push_chat(ChatKind::Info, body);
    }

    pub(crate) fn push_error(&mut self, body: impl Into<String>) {
        self.push_chat(ChatKind::Error, body);
    }

    pub(crate) fn clear_conversation(&mut self) {
        self.chat.clear();
        self.reasoning.clear();
        self.active_reasoning = None;
        self.committing_reasoning = None;
        self.activity.clear();
        self.active_tools.clear();
        self.streaming_chat = None;
        self.committing_streaming_chat = None;
        self.chat_scroll = 0;
        self.chat_max_scroll = 0;
        self.follow_latest = true;
        self.pending_chat_jump = None;
        self.refresh_chat_search();
    }

    pub(crate) fn open_chat_search(&mut self) {
        let mut search = ChatSearch::default();
        search.refresh(&self.chat);
        self.modal = Some(Modal::Search(search));
    }

    fn refresh_chat_search(&mut self) {
        let entries = &self.chat;
        if let Some(Modal::Search(search)) = &mut self.modal {
            search.refresh(entries);
        }
    }

    pub(crate) fn paste_chat_search(&mut self, text: &str) -> bool {
        let entries = &self.chat;
        let Some(Modal::Search(search)) = &mut self.modal else {
            return false;
        };
        let text = text.replace("\r\n", " ").replace(['\r', '\n'], " ");
        insert_text(&mut search.query, &mut search.cursor, &text);
        search.refresh(entries);
        true
    }

    pub(crate) fn handle_chat_search_key(&mut self, key: KeyEvent) {
        let Some(Modal::Search(mut search)) = self.modal.take() else {
            return;
        };
        let mut query_changed = false;

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c' | 'f') => return,
                KeyCode::Char('p') => search.selected = search.selected.saturating_sub(1),
                KeyCode::Char('n') => {
                    search.selected = search
                        .selected
                        .saturating_add(1)
                        .min(search.matches.len().saturating_sub(1));
                }
                _ => {
                    let before = search.query.len();
                    apply_editor_key(&mut search.query, &mut search.cursor, &key);
                    query_changed = search.query.len() != before;
                }
            }
        } else {
            match key.code {
                KeyCode::Esc => return,
                KeyCode::Enter => {
                    if let Some(entry) = search.selected_entry() {
                        self.pending_chat_jump = Some(entry);
                        self.status = format!(
                            "Conversation match {} of {}",
                            search.selected + 1,
                            search.matches.len()
                        );
                        return;
                    }
                }
                KeyCode::Up | KeyCode::BackTab => {
                    search.selected = search.selected.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Tab => {
                    search.selected = search
                        .selected
                        .saturating_add(1)
                        .min(search.matches.len().saturating_sub(1))
                }
                KeyCode::PageUp => search.selected = search.selected.saturating_sub(10),
                KeyCode::PageDown => {
                    search.selected = search
                        .selected
                        .saturating_add(10)
                        .min(search.matches.len().saturating_sub(1))
                }
                _ => {
                    let before = search.query.len();
                    apply_editor_key(&mut search.query, &mut search.cursor, &key);
                    query_changed = search.query.len() != before;
                }
            }
        }

        if query_changed {
            search.refresh(&self.chat);
        }
        self.modal = Some(Modal::Search(search));
    }

    pub(crate) fn load_history(&mut self, history: &[Message]) {
        self.clear_conversation();
        let mut tool_entries = HashMap::<String, usize>::new();
        for message in history {
            let text = message.text();
            if !text.is_empty() {
                if message.is_goal_continuation() {
                    self.push_info("Automatic continuation of the active goal");
                } else {
                    let kind = if message.role == "user" {
                        ChatKind::User
                    } else {
                        ChatKind::Assistant
                    };
                    self.push_chat(kind, text);
                }
            }
            if message.role == "assistant" {
                let reasoning = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Thinking { thinking, .. } if !thinking.is_empty() => {
                            Some(thinking.as_str())
                        }
                        ContentBlock::RedactedThinking { .. } => {
                            Some("[Provider redacted this reasoning block.]")
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if !reasoning.is_empty() {
                    let (body, display_capped) = project_reasoning_display(&reasoning);
                    self.reasoning.push(ReasoningEntry {
                        timestamp: "saved".to_string(),
                        body,
                        display_capped,
                        finished: true,
                        abort_reason: None,
                    });
                }
            }
            for block in &message.content {
                match block {
                    ContentBlock::ToolUse { name, input, id } => {
                        let index = self.activity.len();
                        tool_entries.insert(id.clone(), index);
                        self.activity.push(ActivityEntry {
                            name: sanitize_terminal_text(name),
                            via_code_mode: false,
                            status: ActivityStatus::Running,
                            timestamp: "saved".to_string(),
                            input: tool_input_preview(name, input, self.max_display_chars),
                            output: String::new(),
                        });
                    }
                    ContentBlock::ToolResult {
                        content,
                        tool_use_id,
                        is_error,
                    } => {
                        if let Some(index) = tool_entries.remove(tool_use_id) {
                            let Some(item) = self.activity.get_mut(index) else {
                                continue;
                            };
                            item.status = if is_code_mode_protocol_rejection(content) {
                                ActivityStatus::Rejected
                            } else if is_error.unwrap_or(false) {
                                ActivityStatus::Failed
                            } else {
                                ActivityStatus::Success
                            };
                            item.output = truncate_middle(
                                &sanitize_terminal_text(content),
                                self.max_display_chars,
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        self.trim_activity();
    }

    pub(crate) fn update_chat_metrics(&mut self, max_scroll: usize) {
        self.chat_max_scroll = max_scroll;
        clamp_scroll(&mut self.chat_scroll, &mut self.follow_latest, max_scroll);
    }

    pub(crate) fn scroll_chat_up(&mut self, lines: usize) {
        if self.chat.is_empty() {
            return;
        }
        self.follow_latest = false;
        self.chat_scroll = self.chat_scroll.saturating_sub(lines);
    }

    pub(crate) fn scroll_chat_down(&mut self, lines: usize) {
        self.chat_scroll = self
            .chat_scroll
            .saturating_add(lines)
            .min(self.chat_max_scroll);
        self.follow_latest = self.chat_scroll == self.chat_max_scroll;
    }

    pub(crate) fn lines_from_latest(&self) -> usize {
        self.chat_max_scroll.saturating_sub(self.chat_scroll)
    }

    pub(crate) fn handle_mouse_scroll(&mut self, kind: MouseEventKind) {
        let queue_len = self.queue.len();
        match (&mut self.modal, kind) {
            (Some(Modal::Permission { scroll, .. }), MouseEventKind::ScrollUp) => {
                *scroll = scroll.saturating_sub(3)
            }
            (Some(Modal::Permission { scroll, .. }), MouseEventKind::ScrollDown) => {
                *scroll = scroll.saturating_add(3)
            }
            (
                Some(Modal::Reasoning {
                    scroll,
                    follow_latest,
                    ..
                }),
                MouseEventKind::ScrollUp,
            ) => {
                *scroll = scroll.saturating_sub(3);
                *follow_latest = false;
            }
            (
                Some(Modal::Reasoning {
                    scroll,
                    max_scroll,
                    follow_latest,
                }),
                MouseEventKind::ScrollDown,
            ) => {
                *scroll = scroll.saturating_add(3).min(*max_scroll);
                *follow_latest = *scroll == *max_scroll;
            }
            (
                Some(Modal::Queue {
                    selected,
                    editing: None,
                }),
                MouseEventKind::ScrollUp,
            ) => *selected = selected.saturating_sub(1),
            (
                Some(Modal::Queue {
                    selected,
                    editing: None,
                }),
                MouseEventKind::ScrollDown,
            ) => *selected = selected.saturating_add(1).min(queue_len.saturating_sub(1)),
            (Some(Modal::Search(search)), MouseEventKind::ScrollUp) => {
                search.selected = search.selected.saturating_sub(1)
            }
            (Some(Modal::Search(search)), MouseEventKind::ScrollDown) => {
                search.selected = search
                    .selected
                    .saturating_add(1)
                    .min(search.matches.len().saturating_sub(1))
            }
            (Some(_), _) => {}
            (None, MouseEventKind::ScrollUp) => self.scroll_chat_up(3),
            (None, MouseEventKind::ScrollDown) => self.scroll_chat_down(3),
            (None, _) => {}
        }
    }

    fn trim_activity(&mut self) {
        if self.activity.len() > MAX_ACTIVITY_ITEMS {
            let remove = self.activity.len() - MAX_ACTIVITY_ITEMS;
            self.activity.drain(0..remove);
            self.active_tools.retain_mut(|(_, index)| {
                if *index < remove {
                    false
                } else {
                    *index -= remove;
                    true
                }
            });
        }
    }

    pub(crate) fn cancel_running_activity(&mut self) {
        for (_, index) in self.active_tools.drain(..) {
            if let Some(item) = self.activity.get_mut(index) {
                item.status = ActivityStatus::Cancelled;
                if item.output.is_empty() {
                    item.output = "Interrupted before a definitive result.".to_string();
                }
            }
        }
    }

    fn start_reasoning_attempt(&mut self) {
        self.reasoning.push(ReasoningEntry {
            timestamp: Self::timestamp(),
            body: String::new(),
            display_capped: false,
            finished: false,
            abort_reason: None,
        });
        self.active_reasoning = self.reasoning.len().checked_sub(1);
    }

    fn finish_reasoning_attempt(&mut self) {
        if let Some(index) = self.active_reasoning.take() {
            if let Some(entry) = self.reasoning.get_mut(index) {
                entry.finished = true;
            }
        }
    }

    fn push_reasoning_delta(&mut self, reasoning: String) {
        let reasoning = sanitize_terminal_text(&reasoning);
        if reasoning.is_empty() {
            return;
        }
        let index = self
            .active_reasoning
            .or_else(|| self.reasoning.len().checked_sub(1));
        let Some(index) = index else {
            self.start_reasoning_attempt();
            return self.push_reasoning_delta(reasoning);
        };
        if let Some(entry) = self.reasoning.get_mut(index) {
            if entry.display_capped {
                return;
            }
            entry.display_capped = append_live_delta(&mut entry.body, &reasoning);
        }
    }

    fn abort_reasoning_attempt(&mut self, reason: String) {
        let reason = sanitize_terminal_text(&reason);
        let index = self
            .active_reasoning
            .take()
            .or_else(|| self.reasoning.len().checked_sub(1));
        if let Some(entry) = index.and_then(|index| self.reasoning.get_mut(index)) {
            entry.finished = true;
            entry.abort_reason = Some(reason);
        }
    }

    pub(crate) fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::ApiCallStarted => {
                let bucket = self.current_usage_bucket();
                self.provider_usage
                    .entry(bucket.clone())
                    .or_default()
                    .record_attempt();
                self.active_usage_bucket = Some(bucket);
                self.busy = true;
                self.streaming_chat = None;
                self.committing_streaming_chat = None;
                self.committing_reasoning = None;
                self.finish_reasoning_attempt();
                self.start_reasoning_attempt();
                self.status = "Thinking".to_string();
            }
            AgentEvent::ApiCallFinished { usage } => {
                let bucket = self
                    .active_usage_bucket
                    .take()
                    .unwrap_or_else(|| self.current_usage_bucket());
                if let Some(usage) = usage.as_ref() {
                    self.provider_usage
                        .entry(bucket)
                        .or_default()
                        .record_report(usage);
                }
                self.committing_streaming_chat = self.streaming_chat.take();
                self.committing_reasoning = self.active_reasoning;
                self.finish_reasoning_attempt();
                self.status = usage
                    .map(|usage| {
                        format!(
                            "Ready · {} in / {} out",
                            compact_number(usage.input_tokens),
                            compact_number(usage.output_tokens)
                        )
                    })
                    .unwrap_or_else(|| "Ready".to_string());
            }
            AgentEvent::AssistantText(text) => {
                self.streaming_chat = None;
                self.committing_streaming_chat = None;
                self.push_chat(ChatKind::Assistant, text);
            }
            AgentEvent::AssistantTextDelta(text) => {
                if let Some(index) = self.streaming_chat {
                    self.push_chat_delta(index, &text);
                } else {
                    let (body, display_capped) = project_display(&text, LIVE_DISPLAY_MARKER);
                    self.chat.push(ChatEntry {
                        kind: ChatKind::Assistant,
                        timestamp: Self::timestamp(),
                        body,
                        display_capped,
                    });
                    self.streaming_chat = self.chat.len().checked_sub(1);
                }
            }
            AgentEvent::ReasoningDelta(reasoning) => {
                if self.active_reasoning.is_none() {
                    self.committing_reasoning = None;
                }
                self.push_reasoning_delta(reasoning);
            }
            AgentEvent::StreamCommitted { text, reasoning } => {
                let chat_index = self
                    .committing_streaming_chat
                    .take()
                    .or_else(|| self.streaming_chat.take());
                let reasoning_index = self
                    .committing_reasoning
                    .take()
                    .or_else(|| self.active_reasoning.take());

                if let Some(text) = text {
                    let (text, display_capped) = project_chat_display(ChatKind::Assistant, &text);
                    if let Some(entry) = chat_index.and_then(|index| self.chat.get_mut(index)) {
                        entry.body = text;
                        entry.display_capped = display_capped;
                    } else if !text.is_empty() {
                        self.chat.push(ChatEntry {
                            kind: ChatKind::Assistant,
                            timestamp: Self::timestamp(),
                            body: text,
                            display_capped,
                        });
                    }
                }
                if let Some(reasoning) = reasoning {
                    let (reasoning, display_capped) = project_reasoning_display(&reasoning);
                    if let Some(entry) =
                        reasoning_index.and_then(|index| self.reasoning.get_mut(index))
                    {
                        entry.body = reasoning;
                        entry.display_capped = display_capped;
                        entry.finished = true;
                        entry.abort_reason = None;
                    } else if !reasoning.is_empty() {
                        self.reasoning.push(ReasoningEntry {
                            timestamp: Self::timestamp(),
                            body: reasoning,
                            display_capped,
                            finished: true,
                            abort_reason: None,
                        });
                    }
                }
            }
            AgentEvent::StreamDisplayTruncated {
                text_bytes,
                reasoning_bytes,
            } => {
                if text_bytes > 0 {
                    let marker = format!(
                        "\n\n[Live preview omitted {text_bytes} streamed bytes; awaiting committed response.]"
                    );
                    if let Some(index) = self.streaming_chat {
                        self.push_chat_delta(index, &marker);
                    } else {
                        self.push_chat(ChatKind::Assistant, marker);
                        self.streaming_chat = self.chat.len().checked_sub(1);
                    }
                }
                if reasoning_bytes > 0 {
                    self.push_reasoning_delta(format!(
                        "\n\n[Live preview omitted {reasoning_bytes} reasoning bytes; awaiting committed response.]"
                    ));
                }
            }
            AgentEvent::AssistantStreamAborted { reason } => {
                let reason = sanitize_terminal_text(&reason);
                let index = self
                    .streaming_chat
                    .take()
                    .or_else(|| self.committing_streaming_chat.take());
                if let Some(index) = index {
                    if let Some(entry) = self.chat.get_mut(index) {
                        let status = format!(
                            "\n\n[Uncommitted stream: {reason}; preview may be incomplete.]"
                        );
                        entry.display_capped = append_live_delta(&mut entry.body, &status);
                    }
                } else {
                    self.push_info(format!("Uncommitted assistant stream: {reason}"));
                }
            }
            AgentEvent::ReasoningStreamAborted { reason } => {
                self.committing_reasoning = None;
                self.abort_reasoning_attempt(reason);
            }
            AgentEvent::ToolCallStarted { name, input } => {
                let via_code_mode = name != "python"
                    && self
                        .active_tools
                        .iter()
                        .any(|(active_name, _)| active_name == "python");
                let index = self.activity.len();
                self.activity.push(ActivityEntry {
                    name: sanitize_terminal_text(&name),
                    via_code_mode,
                    status: ActivityStatus::Running,
                    timestamp: Self::timestamp(),
                    input: tool_input_preview(&name, &input, self.max_display_chars),
                    output: String::new(),
                });
                self.active_tools.push((name, index));
                self.status = "Running tool".to_string();
                self.trim_activity();
            }
            AgentEvent::ToolCallFinished {
                name,
                outcome,
                content,
            } => {
                if let Some(active_index) = self
                    .active_tools
                    .iter()
                    .rposition(|(active_name, _)| active_name == &name)
                {
                    let (_, item_index) = self.active_tools.remove(active_index);
                    if let Some(item) = self.activity.get_mut(item_index) {
                        item.status = match outcome {
                            ToolCallOutcome::Success => ActivityStatus::Success,
                            ToolCallOutcome::Failed => ActivityStatus::Failed,
                            ToolCallOutcome::Denied => ActivityStatus::Denied,
                            ToolCallOutcome::Cancelled => ActivityStatus::Cancelled,
                        };
                        item.output = truncate_middle(
                            &sanitize_terminal_text(&content),
                            self.max_display_chars,
                        );
                    }
                }
                let name = sanitize_terminal_text(&name);
                self.status = match outcome {
                    ToolCallOutcome::Success => format!("{} completed", name),
                    ToolCallOutcome::Failed => format!("{} failed", name),
                    ToolCallOutcome::Denied => format!("{} denied", name),
                    ToolCallOutcome::Cancelled => format!("{} interrupted", name),
                };
            }
            AgentEvent::Retrying {
                attempt,
                max_retries,
                delay_secs,
                error,
            } => self.push_info(format!(
                "Transient API error ({error}); retry {attempt}/{max_retries} in {delay_secs}s"
            )),
            AgentEvent::Notice(message) => self.push_info(message),
            AgentEvent::SteeringCommitted { prompts } => {
                for prompt in prompts {
                    if prompt.source == PromptSource::GoalContinuation {
                        self.push_info("Automatic continuation of the active goal");
                    } else {
                        self.push_user(prompt.text);
                    }
                }
                self.status = "Steering active turn".to_string();
            }
            AgentEvent::GoalCompleted { goal } => {
                self.goal = None;
                self.push_info(format!(
                    "Goal completed: {}",
                    truncate_middle(&sanitize_terminal_text(&goal), 400)
                ));
                self.status = "Goal completed".to_string();
            }
            AgentEvent::HistoryCheckpoint { context_tokens, .. } => {
                self.context_tokens = context_tokens;
            }
        }
        self.refresh_chat_search();
    }

    pub(crate) fn insert_input_char(&mut self, ch: char) {
        insert_char(&mut self.input, &mut self.input_cursor, ch);
        self.history_cursor = None;
    }

    pub(crate) fn insert_input_text(&mut self, text: &str) {
        insert_text(&mut self.input, &mut self.input_cursor, text);
        self.history_cursor = None;
    }

    /// Complete a catalog-backed command only when the cursor is at the end.
    /// Returning false preserves Tab's ordinary follow-up submission path.
    pub(crate) fn complete_slash_command(&mut self) -> bool {
        if self.input_cursor != self.input.chars().count() {
            return false;
        }
        match complete_local_command(&self.input) {
            Some(CommandCompletion::Replace(replacement)) => {
                self.input = replacement;
                self.input_cursor = self.input.chars().count();
                self.history_cursor = None;
                self.status = "Completed slash-command prefix".to_string();
                true
            }
            Some(CommandCompletion::Candidates(candidates)) => {
                self.status = format!("Command matches: {}", candidates.join(" · "));
                true
            }
            Some(CommandCompletion::Complete) => {
                self.status =
                    "Command prefix complete · add arguments if needed or press Enter".to_string();
                true
            }
            None => false,
        }
    }

    pub(crate) fn history_previous(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let next = match self.history_cursor {
            None => {
                self.history_draft = self.input.clone();
                self.input_history.len() - 1
            }
            Some(index) => index.saturating_sub(1),
        };
        self.history_cursor = Some(next);
        self.input = self.input_history[next].clone();
        self.input_cursor = self.input.chars().count();
    }

    pub(crate) fn history_next(&mut self) {
        let Some(index) = self.history_cursor else {
            return;
        };
        if index + 1 < self.input_history.len() {
            let next = index + 1;
            self.history_cursor = Some(next);
            self.input = self.input_history[next].clone();
        } else {
            self.history_cursor = None;
            self.input = std::mem::take(&mut self.history_draft);
        }
        self.input_cursor = self.input.chars().count();
    }

    pub(crate) fn take_input(&mut self) -> String {
        let input = std::mem::take(&mut self.input);
        self.input_cursor = 0;
        self.history_cursor = None;
        self.history_draft.clear();
        if !input.trim().is_empty() {
            self.input_history.push(input.clone());
        }
        input
    }

    pub(crate) fn restore_to_composer(&mut self, prompt: &QueuedPrompt) -> bool {
        if !self.input.trim().is_empty() {
            self.status =
                "Composer already has a draft; clear it before restoring queued text".to_string();
            return false;
        }
        self.input.clone_from(&prompt.text);
        self.input_cursor = self.input.chars().count();
        self.history_cursor = None;
        self.history_draft.clear();
        self.status = format!("Restored queued {}", prompt.delivery.label());
        true
    }
}

/// Clamp a scroll offset to a freshly measured `max_scroll`, snapping to the
/// bottom while following the latest output and resuming follow mode when a
/// paused viewport lands back on the bottom row.
pub(crate) fn clamp_scroll(scroll: &mut usize, follow_latest: &mut bool, max_scroll: usize) {
    if *follow_latest {
        *scroll = max_scroll;
    } else {
        *scroll = (*scroll).min(max_scroll);
        if *scroll == max_scroll {
            *follow_latest = true;
        }
    }
}

fn is_code_mode_protocol_rejection(content: &str) -> bool {
    content.contains("code mode permits only the model-facing `python` capability tool")
        || content.contains("Direct tool calls are disabled in code mode")
}

pub(crate) fn compact_number(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn counted(value: u64, singular: &str, plural: &str) -> String {
    format!("{} {}", value, if value == 1 { singular } else { plural })
}

fn optional_usage_total(tokens: u64, field_reports: u64, usage_reports: u64) -> String {
    if usage_reports == 0 {
        "unavailable (no usage reports)".to_string()
    } else if field_reports == 0 {
        format!("unavailable (0/{usage_reports} reports)")
    } else {
        format!("{tokens} ({field_reports}/{usage_reports} reports)")
    }
}
