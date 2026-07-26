//! Full-screen Ratatui frontend for the interactive agent CLI.

use crate::permissions::{PermissionChoice, ToolExecutionRequest};
use crate::runtime::{DeliveryMode, PromptId, PromptQueue, QueuedPrompt};
use crate::types::{truncate_middle, ContentBlock, Message};
use crate::{AgentEvent, ToolCallOutcome};
use chrono::Local;
use crossterm::cursor::Show;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use ratatui::{Frame, Terminal};
use serde_json::Value;
use std::collections::HashMap;
use std::io::{self, Stdout};
use std::time::Duration;
use tokio::time::MissedTickBehavior;
use unicode_width::UnicodeWidthChar;

const TICK_RATE: Duration = Duration::from_millis(100);
const MAX_ACTIVITY_ITEMS: usize = 100;
const DEFAULT_MAX_DISPLAY_CHARS: usize = 2_000;

const BG: Color = Color::Rgb(12, 16, 24);
const PANEL: Color = Color::Rgb(20, 26, 38);
const PANEL_ALT: Color = Color::Rgb(26, 34, 48);
const BORDER: Color = Color::Rgb(59, 75, 99);
const TEXT: Color = Color::Rgb(220, 226, 235);
const MUTED: Color = Color::Rgb(124, 139, 161);
const CYAN: Color = Color::Rgb(92, 207, 230);
const GREEN: Color = Color::Rgb(111, 214, 151);
const YELLOW: Color = Color::Rgb(241, 196, 96);
const RED: Color = Color::Rgb(244, 112, 122);
const PURPLE: Color = Color::Rgb(183, 148, 244);

type TuiTerminal = Terminal<CrosstermBackend<Stdout>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatKind {
    User,
    Assistant,
    Info,
    Error,
}

#[derive(Debug, Clone)]
struct ChatEntry {
    kind: ChatKind,
    timestamp: String,
    body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivityStatus {
    Running,
    Success,
    Failed,
    Rejected,
    Denied,
    Cancelled,
}

#[derive(Debug, Clone)]
struct ActivityEntry {
    name: String,
    via_code_mode: bool,
    status: ActivityStatus,
    timestamp: String,
    input: String,
    output: String,
}

#[derive(Debug, Clone)]
enum Modal {
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
}

#[derive(Debug, Clone)]
struct QueueEditor {
    id: PromptId,
    value: String,
    cursor: usize,
}

#[derive(Debug)]
struct AppState {
    api: String,
    model: String,
    bridge_count: usize,
    context_tokens: u64,
    chat: Vec<ChatEntry>,
    activity: Vec<ActivityEntry>,
    active_tools: Vec<(String, usize)>,
    streaming_chat: Option<usize>,
    input: String,
    input_cursor: usize,
    input_history: Vec<String>,
    history_cursor: Option<usize>,
    history_draft: String,
    chat_scroll: usize,
    chat_max_scroll: usize,
    follow_latest: bool,
    busy: bool,
    spinner_tick: usize,
    status: String,
    modal: Option<Modal>,
    queue: Vec<QueuedPrompt>,
    max_display_chars: usize,
}

impl AppState {
    fn new(api: impl Into<String>, model: impl Into<String>) -> Self {
        let api = sanitize_terminal_text(&api.into());
        let model = sanitize_terminal_text(&model.into());
        Self {
            api,
            model,
            bridge_count: 0,
            context_tokens: 0,
            chat: Vec::new(),
            activity: Vec::new(),
            active_tools: Vec::new(),
            streaming_chat: None,
            input: String::new(),
            input_cursor: 0,
            input_history: Vec::new(),
            history_cursor: None,
            history_draft: String::new(),
            chat_scroll: 0,
            chat_max_scroll: 0,
            follow_latest: true,
            busy: false,
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

    fn push_chat(&mut self, kind: ChatKind, body: impl Into<String>) {
        self.chat.push(ChatEntry {
            kind,
            timestamp: Self::timestamp(),
            body: sanitize_terminal_text(&body.into()),
        });
    }

    fn push_user(&mut self, body: impl Into<String>) {
        self.push_chat(ChatKind::User, body);
    }

    fn push_info(&mut self, body: impl Into<String>) {
        self.push_chat(ChatKind::Info, body);
    }

    fn push_error(&mut self, body: impl Into<String>) {
        self.push_chat(ChatKind::Error, body);
    }

    fn clear_conversation(&mut self) {
        self.chat.clear();
        self.activity.clear();
        self.active_tools.clear();
        self.streaming_chat = None;
        self.chat_scroll = 0;
        self.chat_max_scroll = 0;
        self.follow_latest = true;
    }

    fn load_history(&mut self, history: &[Message]) {
        self.clear_conversation();
        let mut tool_entries = HashMap::<String, usize>::new();
        for message in history {
            let text = message.text();
            if !text.is_empty() {
                let kind = if message.role == "user" {
                    ChatKind::User
                } else {
                    ChatKind::Assistant
                };
                self.push_chat(kind, text);
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
                            input: truncate_middle(
                                &serde_json::to_string(input).unwrap_or_default(),
                                self.max_display_chars,
                            ),
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

    fn update_chat_metrics(&mut self, max_scroll: usize) {
        self.chat_max_scroll = max_scroll;
        if self.follow_latest {
            self.chat_scroll = max_scroll;
        } else {
            self.chat_scroll = self.chat_scroll.min(max_scroll);
            if self.chat_scroll == max_scroll {
                self.follow_latest = true;
            }
        }
    }

    fn scroll_chat_up(&mut self, lines: usize) {
        if self.chat.is_empty() {
            return;
        }
        self.follow_latest = false;
        self.chat_scroll = self.chat_scroll.saturating_sub(lines);
    }

    fn scroll_chat_down(&mut self, lines: usize) {
        self.chat_scroll = self
            .chat_scroll
            .saturating_add(lines)
            .min(self.chat_max_scroll);
        self.follow_latest = self.chat_scroll == self.chat_max_scroll;
    }

    fn lines_from_latest(&self) -> usize {
        self.chat_max_scroll.saturating_sub(self.chat_scroll)
    }

    fn handle_mouse_scroll(&mut self, kind: MouseEventKind) {
        let queue_len = self.queue.len();
        match (&mut self.modal, kind) {
            (Some(Modal::Permission { scroll, .. }), MouseEventKind::ScrollUp) => {
                *scroll = scroll.saturating_sub(3)
            }
            (Some(Modal::Permission { scroll, .. }), MouseEventKind::ScrollDown) => {
                *scroll = scroll.saturating_add(3)
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

    fn cancel_running_activity(&mut self) {
        for (_, index) in self.active_tools.drain(..) {
            if let Some(item) = self.activity.get_mut(index) {
                item.status = ActivityStatus::Cancelled;
                if item.output.is_empty() {
                    item.output = "Interrupted before a definitive result.".to_string();
                }
            }
        }
    }

    fn apply_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::ApiCallStarted => {
                self.busy = true;
                self.streaming_chat = None;
                self.status = "Thinking".to_string();
            }
            AgentEvent::ApiCallFinished { usage } => {
                self.streaming_chat = None;
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
                self.push_chat(ChatKind::Assistant, text);
            }
            AgentEvent::AssistantTextDelta(text) => {
                let text = sanitize_terminal_text(&text);
                if let Some(index) = self.streaming_chat {
                    if let Some(entry) = self.chat.get_mut(index) {
                        entry.body.push_str(&text);
                    }
                } else {
                    self.push_chat(ChatKind::Assistant, text);
                    self.streaming_chat = self.chat.len().checked_sub(1);
                }
            }
            AgentEvent::AssistantStreamAborted { reason } => {
                let reason = sanitize_terminal_text(&reason);
                if let Some(index) = self.streaming_chat.take() {
                    if let Some(entry) = self.chat.get_mut(index) {
                        entry
                            .body
                            .push_str(&format!("\n\n[Uncommitted stream: {reason}]"));
                    }
                } else {
                    self.push_info(format!("Uncommitted assistant stream: {reason}"));
                }
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
                    input: truncate_middle(
                        &serde_json::to_string(&input).unwrap_or_default(),
                        self.max_display_chars,
                    ),
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
                    self.push_user(prompt.text);
                }
                self.status = "Steering active turn".to_string();
            }
            AgentEvent::HistoryCheckpoint { context_tokens, .. } => {
                self.context_tokens = context_tokens;
            }
        }
    }

    fn insert_input_char(&mut self, ch: char) {
        insert_char(&mut self.input, &mut self.input_cursor, ch);
        self.history_cursor = None;
    }

    fn insert_input_text(&mut self, text: &str) {
        insert_text(&mut self.input, &mut self.input_cursor, text);
        self.history_cursor = None;
    }

    fn backspace_input(&mut self) {
        backspace(&mut self.input, &mut self.input_cursor);
        self.history_cursor = None;
    }

    fn delete_input(&mut self) {
        delete_at_cursor(&mut self.input, self.input_cursor);
        self.history_cursor = None;
    }

    fn history_previous(&mut self) {
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

    fn history_next(&mut self) {
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

    fn take_input(&mut self) -> String {
        let input = std::mem::take(&mut self.input);
        self.input_cursor = 0;
        self.history_cursor = None;
        self.history_draft.clear();
        if !input.trim().is_empty() {
            self.input_history.push(input.clone());
        }
        input
    }

    fn restore_to_composer(&mut self, prompt: &QueuedPrompt) -> bool {
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

/// A controller action produced by one terminal event.
#[derive(Debug)]
pub enum UiAction {
    None,
    /// A queue-manager operation already changed the authoritative queue.
    QueueChanged,
    Submit {
        text: String,
        delivery: DeliveryMode,
    },
    Interrupt,
    Exit,
    Permission {
        id: u64,
        choice: PermissionChoice,
    },
}

impl UiAction {
    /// Whether handling this action changes queue-bearing durable state.
    ///
    /// `Submit` is included because the controller enqueues it immediately
    /// after this method is called. Display-only events such as scrolling and
    /// composer edits must not force an atomic autosave.
    pub fn requires_queue_persist(&self) -> bool {
        matches!(self, Self::QueueChanged | Self::Submit { .. })
    }
}

/// The only owner of terminal input, raw mode, and Ratatui drawing.
pub struct TerminalUi {
    terminal: TuiTerminal,
    events: EventStream,
    app: AppState,
    dirty: bool,
    active: bool,
}

impl TerminalUi {
    pub fn start(api: impl Into<String>, model: impl Into<String>) -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        if let Err(error) = execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        ) {
            leave_terminal(&mut stdout);
            let _ = disable_raw_mode();
            return Err(error);
        }
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(error) => {
                let mut stdout = io::stdout();
                leave_terminal(&mut stdout);
                let _ = disable_raw_mode();
                return Err(error);
            }
        };
        if let Err(error) = terminal.clear() {
            leave_terminal(terminal.backend_mut());
            let _ = terminal.show_cursor();
            let _ = disable_raw_mode();
            return Err(error);
        }
        let mut ui = Self {
            terminal,
            events: EventStream::new(),
            app: AppState::new(api, model),
            dirty: true,
            active: true,
        };
        ui.draw()?;
        Ok(ui)
    }

    pub fn draw(&mut self) -> io::Result<()> {
        let app = &mut self.app;
        self.terminal.draw(|frame| render(frame, app))?;
        self.dirty = false;
        Ok(())
    }

    pub fn draw_if_dirty(&mut self) -> io::Result<()> {
        if self.dirty {
            self.draw()?;
        }
        Ok(())
    }

    pub async fn next_event(&mut self) -> io::Result<Event> {
        match self.events.next().await {
            Some(Ok(event)) => Ok(event),
            Some(Err(error)) => Err(error),
            None => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "terminal event stream ended",
            )),
        }
    }

    pub fn tick(&mut self) {
        if self.app.busy {
            self.app.spinner_tick = self.app.spinner_tick.wrapping_add(1);
            // The 50 ms reactor tick bounds input/event latency. Animating at
            // half that rate avoids rebuilding a long transcript at 20 FPS
            // when the only visual change is the spinner.
            self.dirty |= self.app.spinner_tick.is_multiple_of(2);
        }
    }

    pub fn set_session(&mut self, api: &str, model: &str, bridge_count: usize) {
        let api = sanitize_terminal_text(api);
        let model = sanitize_terminal_text(model);
        if self.app.api == api && self.app.model == model && self.app.bridge_count == bridge_count {
            return;
        }
        self.app.api = api;
        self.app.model = model;
        self.app.bridge_count = bridge_count;
        self.dirty = true;
    }

    pub fn set_context_tokens(&mut self, tokens: u64) {
        if self.app.context_tokens == tokens {
            return;
        }
        self.app.context_tokens = tokens;
        self.dirty = true;
    }

    pub fn set_busy(&mut self, busy: bool, status: &str) {
        let status = sanitize_terminal_text(status);
        let changed = self.app.busy != busy || self.app.status != status;
        self.app.busy = busy;
        self.app.status = status;
        if !busy {
            self.app.streaming_chat = None;
        }
        self.dirty |= changed;
    }

    pub fn status(&mut self, message: &str) {
        let message = sanitize_terminal_text(message);
        if self.app.status == message {
            return;
        }
        self.app.status = message;
        self.dirty = true;
    }

    pub fn push_user(&mut self, body: &str) {
        self.app.push_user(body);
        self.dirty = true;
    }

    pub fn info(&mut self, message: &str) {
        self.app.push_info(message);
        self.app.status = sanitize_terminal_text(message);
        self.dirty = true;
    }

    pub fn error(&mut self, message: &str) {
        self.app.push_error(message);
        self.app.status = "Error".to_string();
        self.dirty = true;
    }

    pub fn clear_conversation(&mut self) {
        self.app.clear_conversation();
        self.dirty = true;
    }

    pub fn load_history(&mut self, history: &[Message]) {
        self.app.load_history(history);
        self.dirty = true;
    }

    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        self.app.apply_agent_event(event);
        self.dirty = true;
    }

    /// Retire nested code-mode activity whose futures were dropped with the
    /// turn. The outer `python` activity receives its own cancellation event;
    /// bridged calls otherwise have no future left to emit one.
    pub fn cancel_running_activity(&mut self) {
        self.app.cancel_running_activity();
        self.dirty = true;
    }

    pub fn sync_queue(&mut self, queue: &PromptQueue) {
        let snapshot = queue.snapshot();
        let mut changed = self.app.queue != snapshot;
        if changed {
            self.app.queue = snapshot;
        }
        if let Some(Modal::Queue { selected, .. }) = self.app.modal.as_mut() {
            let clamped = (*selected).min(self.app.queue.len().saturating_sub(1));
            changed |= clamped != *selected;
            *selected = clamped;
        }
        self.dirty |= changed;
    }

    pub fn open_permission(&mut self, id: u64, request: ToolExecutionRequest) {
        self.app.modal = Some(Modal::Permission {
            id,
            request,
            selected: 1,
            scroll: 0,
        });
        self.app.status = "Permission required".to_string();
        self.dirty = true;
    }

    pub fn close_permission(&mut self, id: u64) {
        if matches!(
            self.app.modal,
            Some(Modal::Permission {
                id: modal_id,
                ..
            }) if modal_id == id
        ) {
            self.app.modal = None;
            self.dirty = true;
        }
    }

    pub fn open_help(&mut self) {
        self.app.modal = Some(Modal::Help);
        self.dirty = true;
    }

    pub async fn show_help(&mut self) -> io::Result<()> {
        self.open_help();
        let mut ticker = tokio::time::interval(TICK_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        self.draw()?;
        loop {
            tokio::select! {
                _ = ticker.tick() => self.tick(),
                event = self.next_event() => {
                    let event = event?;
                    self.dirty = true;
                    if let Event::Key(key) = event {
                        if is_key_press(key)
                            && (matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::F(1))
                                || (key.code == KeyCode::Char('c')
                                    && key.modifiers.contains(KeyModifiers::CONTROL)))
                        {
                            self.app.modal = None;
                            return Ok(());
                        }
                    }
                }
            }
            self.draw_if_dirty()?;
        }
    }

    pub async fn select(&mut self, title: &str, items: &[String]) -> io::Result<Option<usize>> {
        if items.is_empty() {
            return Ok(None);
        }
        self.app.modal = Some(Modal::Select {
            title: title.to_string(),
            items: items.to_vec(),
            selected: 0,
        });
        let mut ticker = tokio::time::interval(TICK_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        self.dirty = true;
        self.draw()?;
        loop {
            tokio::select! {
                _ = ticker.tick() => self.tick(),
                event = self.next_event() => {
                    let event = event?;
                    self.dirty = true;
                    let Event::Key(key) = event else {
                        self.draw_if_dirty()?;
                        continue;
                    };
                    if !is_key_press(key) {
                        continue;
                    }
                    self.dirty = true;
                    let Modal::Select { items, selected, .. } =
                        self.app.modal.as_mut().expect("select modal")
                    else {
                        unreachable!()
                    };
                    match key.code {
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.app.modal = None;
                            return Ok(None);
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            *selected = selected.saturating_sub(1)
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            *selected = (*selected + 1).min(items.len() - 1)
                        }
                        KeyCode::Home => *selected = 0,
                        KeyCode::End => *selected = items.len() - 1,
                        KeyCode::Enter => {
                            let answer = *selected;
                            self.app.modal = None;
                            return Ok(Some(answer));
                        }
                        KeyCode::Esc => {
                            self.app.modal = None;
                            return Ok(None);
                        }
                        _ => {}
                    }
                }
            }
            self.draw_if_dirty()?;
        }
    }

    pub async fn prompt(&mut self, title: &str, default: &str) -> io::Result<Option<String>> {
        self.app.modal = Some(Modal::Prompt {
            title: title.to_string(),
            value: default.to_string(),
            cursor: default.chars().count(),
        });
        let mut ticker = tokio::time::interval(TICK_RATE);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        self.dirty = true;
        self.draw()?;
        loop {
            tokio::select! {
                _ = ticker.tick() => self.tick(),
                event = self.next_event() => {
                    self.dirty = true;
                    match event? {
                        Event::Paste(text) => {
                            let Modal::Prompt { value, cursor, .. } =
                                self.app.modal.as_mut().expect("prompt modal")
                            else {
                                unreachable!()
                            };
                            let text = text.replace(['\r', '\n'], "");
                            insert_text(value, cursor, &text);
                        }
                        Event::Key(key) if is_key_press(key) => {
                            let Modal::Prompt { value, cursor, .. } =
                                self.app.modal.as_mut().expect("prompt modal")
                            else {
                                unreachable!()
                            };
                            match key.code {
                                KeyCode::Char('c')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.app.modal = None;
                                    return Ok(None);
                                }
                                KeyCode::Enter => {
                                    let value = value.clone();
                                    self.app.modal = None;
                                    return Ok(Some(value));
                                }
                                KeyCode::Esc => {
                                    self.app.modal = None;
                                    return Ok(None);
                                }
                                KeyCode::Char(ch) => insert_char(value, cursor, ch),
                                KeyCode::Backspace => backspace(value, cursor),
                                KeyCode::Delete => delete_at_cursor(value, *cursor),
                                KeyCode::Left => *cursor = cursor.saturating_sub(1),
                                KeyCode::Right => {
                                    *cursor = (*cursor + 1).min(value.chars().count())
                                }
                                KeyCode::Home => *cursor = 0,
                                KeyCode::End => *cursor = value.chars().count(),
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                }
            }
            self.draw_if_dirty()?;
        }
    }

    pub fn handle_event(&mut self, event: Event, queue: &PromptQueue) -> UiAction {
        self.dirty = true;
        match event {
            Event::Key(key) if is_key_press(key) => self.handle_key(key, queue),
            Event::Paste(text) => {
                if let Some(Modal::Queue {
                    editing: Some(editor),
                    ..
                }) = self.app.modal.as_mut()
                {
                    insert_text(&mut editor.value, &mut editor.cursor, &text);
                } else if self.app.modal.is_none() {
                    self.app.insert_input_text(&text);
                }
                UiAction::None
            }
            Event::Mouse(mouse) => {
                self.app.handle_mouse_scroll(mouse.kind);
                UiAction::None
            }
            Event::Resize(_, _) => UiAction::None,
            _ => UiAction::None,
        }
    }

    fn handle_key(&mut self, key: KeyEvent, queue: &PromptQueue) -> UiAction {
        match self.app.modal {
            Some(Modal::Help) => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::F(1))
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    self.app.modal = None;
                }
                return UiAction::None;
            }
            Some(Modal::Permission { .. }) => return self.handle_permission_key(key),
            Some(Modal::Queue { .. }) => return self.handle_queue_key(key, queue),
            Some(Modal::Select { .. } | Modal::Prompt { .. }) => return UiAction::None,
            None => {}
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    return if self.app.busy {
                        UiAction::Interrupt
                    } else {
                        UiAction::Exit
                    }
                }
                KeyCode::Char('d') if self.app.input.is_empty() => return UiAction::Exit,
                KeyCode::Char('a') => self.app.input_cursor = 0,
                KeyCode::Char('e') => self.app.input_cursor = self.app.input.chars().count(),
                KeyCode::Char('u') => {
                    self.app.input.clear();
                    self.app.input_cursor = 0;
                }
                KeyCode::Char('k') => truncate_at_char(&mut self.app.input, self.app.input_cursor),
                KeyCode::Char('w') => {
                    delete_previous_word(&mut self.app.input, &mut self.app.input_cursor)
                }
                KeyCode::Enter | KeyCode::Char('j') => self.app.insert_input_char('\n'),
                _ => {}
            }
            return UiAction::None;
        }

        if key.modifiers.contains(KeyModifiers::ALT) && key.code == KeyCode::Up {
            if !self.app.input.trim().is_empty() {
                self.app.status =
                    "Composer already has a draft; clear it before restoring queued text"
                        .to_string();
                return UiAction::None;
            }
            if let Some(prompt) = queue.restore_latest() {
                let restored = self.app.restore_to_composer(&prompt);
                debug_assert!(restored);
                self.sync_queue(queue);
                return UiAction::QueueChanged;
            }
            return UiAction::None;
        }

        if let Some(delivery) = submission_delivery(key, self.app.busy) {
            return self.submit(delivery);
        }

        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.app.insert_input_char('\n');
                UiAction::None
            }
            KeyCode::Char(ch) => {
                self.app.insert_input_char(ch);
                UiAction::None
            }
            KeyCode::Backspace => {
                self.app.backspace_input();
                UiAction::None
            }
            KeyCode::Delete => {
                self.app.delete_input();
                UiAction::None
            }
            KeyCode::Left => {
                self.app.input_cursor = self.app.input_cursor.saturating_sub(1);
                UiAction::None
            }
            KeyCode::Right => {
                self.app.input_cursor =
                    (self.app.input_cursor + 1).min(self.app.input.chars().count());
                UiAction::None
            }
            KeyCode::Home => {
                self.app.input_cursor = 0;
                UiAction::None
            }
            KeyCode::End => {
                self.app.input_cursor = self.app.input.chars().count();
                UiAction::None
            }
            KeyCode::Up => {
                self.app.history_previous();
                UiAction::None
            }
            KeyCode::Down => {
                self.app.history_next();
                UiAction::None
            }
            KeyCode::PageUp => {
                self.app.scroll_chat_up(10);
                UiAction::None
            }
            KeyCode::PageDown => {
                self.app.scroll_chat_down(10);
                UiAction::None
            }
            KeyCode::Esc if self.app.busy => UiAction::Interrupt,
            KeyCode::Esc => {
                self.app.input.clear();
                self.app.input_cursor = 0;
                UiAction::None
            }
            KeyCode::F(1) => {
                self.open_help();
                UiAction::None
            }
            KeyCode::F(2) => {
                self.sync_queue(queue);
                self.app.modal = Some(Modal::Queue {
                    selected: 0,
                    editing: None,
                });
                UiAction::None
            }
            _ => UiAction::None,
        }
    }

    fn submit(&mut self, delivery: DeliveryMode) -> UiAction {
        let text = self.app.take_input();
        if text.trim().is_empty() {
            return UiAction::None;
        }
        UiAction::Submit { text, delivery }
    }

    fn handle_permission_key(&mut self, key: KeyEvent) -> UiAction {
        let Some(Modal::Permission {
            id,
            request,
            mut selected,
            mut scroll,
        }) = self.app.modal.take()
        else {
            return UiAction::None;
        };
        let choice = match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                Some(PermissionChoice::DenyOnce)
            }
            KeyCode::Char('a') => Some(PermissionChoice::AllowAlways),
            KeyCode::Char('y') => Some(PermissionChoice::AllowOnce),
            KeyCode::Char('d') => Some(PermissionChoice::DenyAlways),
            KeyCode::Char('n') | KeyCode::Esc => Some(PermissionChoice::DenyOnce),
            KeyCode::Enter => Some(permission_choice(selected)),
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
                None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(3);
                None
            }
            KeyCode::PageUp => {
                scroll = scroll.saturating_sub(8);
                None
            }
            KeyCode::PageDown => {
                scroll = scroll.saturating_add(8);
                None
            }
            _ => None,
        };
        if let Some(choice) = choice {
            UiAction::Permission { id, choice }
        } else {
            self.app.modal = Some(Modal::Permission {
                id,
                request,
                selected,
                scroll,
            });
            UiAction::None
        }
    }

    fn handle_queue_key(&mut self, key: KeyEvent, queue: &PromptQueue) -> UiAction {
        let Some(Modal::Queue {
            mut selected,
            mut editing,
        }) = self.app.modal.take()
        else {
            return UiAction::None;
        };

        if key.code == KeyCode::F(2) {
            return UiAction::None;
        }

        if let Some(mut editor) = editing.take() {
            let mut queue_changed = false;
            if key.modifiers.contains(KeyModifiers::CONTROL) {
                match key.code {
                    KeyCode::Char('c') => {}
                    KeyCode::Char('a') => {
                        editor.cursor = 0;
                        editing = Some(editor);
                    }
                    KeyCode::Char('e') => {
                        editor.cursor = editor.value.chars().count();
                        editing = Some(editor);
                    }
                    KeyCode::Char('u') => {
                        editor.value.clear();
                        editor.cursor = 0;
                        editing = Some(editor);
                    }
                    KeyCode::Char('k') => {
                        truncate_at_char(&mut editor.value, editor.cursor);
                        editing = Some(editor);
                    }
                    KeyCode::Char('w') => {
                        delete_previous_word(&mut editor.value, &mut editor.cursor);
                        editing = Some(editor);
                    }
                    KeyCode::Enter | KeyCode::Char('j') => {
                        insert_char(&mut editor.value, &mut editor.cursor, '\n');
                        editing = Some(editor);
                    }
                    _ => editing = Some(editor),
                }
                self.sync_queue(queue);
                self.app.modal = Some(Modal::Queue { selected, editing });
                return UiAction::None;
            }
            match key.code {
                KeyCode::Enter => {
                    if editor.value.trim().is_empty() {
                        self.app.status = "Queued prompt cannot be empty".to_string();
                        editing = Some(editor);
                    } else if queue.edit(editor.id, editor.value.clone()) {
                        queue_changed = true;
                    } else {
                        self.app.status =
                            "Queued prompt was already claimed; edit was not applied".to_string();
                    }
                }
                KeyCode::Esc => {}
                KeyCode::Char(ch) => {
                    insert_char(&mut editor.value, &mut editor.cursor, ch);
                    editing = Some(editor);
                }
                KeyCode::Backspace => {
                    backspace(&mut editor.value, &mut editor.cursor);
                    editing = Some(editor);
                }
                KeyCode::Delete => {
                    delete_at_cursor(&mut editor.value, editor.cursor);
                    editing = Some(editor);
                }
                KeyCode::Left => {
                    editor.cursor = editor.cursor.saturating_sub(1);
                    editing = Some(editor);
                }
                KeyCode::Right => {
                    editor.cursor = (editor.cursor + 1).min(editor.value.chars().count());
                    editing = Some(editor);
                }
                KeyCode::Home => {
                    editor.cursor = 0;
                    editing = Some(editor);
                }
                KeyCode::End => {
                    editor.cursor = editor.value.chars().count();
                    editing = Some(editor);
                }
                _ => editing = Some(editor),
            }
            self.sync_queue(queue);
            self.app.modal = Some(Modal::Queue { selected, editing });
            return if queue_changed {
                UiAction::QueueChanged
            } else {
                UiAction::None
            };
        }

        let items = queue.snapshot();
        let selected_id = items.get(selected).map(|item| item.id);
        let mut queue_changed = false;
        match key.code {
            KeyCode::Esc => return UiAction::None,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return UiAction::None;
            }
            KeyCode::Up | KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(id) = selected_id {
                    if queue.move_by(id, -1) {
                        selected = selected.saturating_sub(1);
                        queue_changed = true;
                    }
                }
            }
            KeyCode::Down | KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(id) = selected_id {
                    if queue.move_by(id, 1) {
                        selected = (selected + 1).min(items.len().saturating_sub(1));
                        queue_changed = true;
                    }
                }
            }
            KeyCode::Up | KeyCode::Char('k') => selected = selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(items.len().saturating_sub(1))
            }
            KeyCode::Char('d') | KeyCode::Delete => {
                if let Some(id) = selected_id {
                    queue_changed = queue.delete(id);
                }
            }
            KeyCode::Char('s') => {
                if let Some(id) = selected_id {
                    queue_changed = queue.toggle_delivery(id, self.app.busy);
                }
            }
            KeyCode::Char('e') | KeyCode::Enter => {
                if let Some(item) = items.get(selected) {
                    editing = Some(QueueEditor {
                        id: item.id,
                        value: item.text.clone(),
                        cursor: item.text.chars().count(),
                    });
                }
            }
            KeyCode::Char('r') => {
                if let Some(item) = items.get(selected) {
                    if self.app.restore_to_composer(item) {
                        queue_changed = queue.delete(item.id);
                        self.sync_queue(queue);
                        return if queue_changed {
                            UiAction::QueueChanged
                        } else {
                            UiAction::None
                        };
                    }
                }
            }
            _ => {}
        }
        self.sync_queue(queue);
        selected = selected.min(self.app.queue.len().saturating_sub(1));
        self.app.modal = Some(Modal::Queue { selected, editing });
        if queue_changed {
            UiAction::QueueChanged
        } else {
            UiAction::None
        }
    }
}

impl Drop for TerminalUi {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let _ = disable_raw_mode();
        leave_terminal(self.terminal.backend_mut());
        let _ = self.terminal.show_cursor();
        self.active = false;
    }
}

fn leave_terminal(writer: &mut impl io::Write) {
    let _ = execute!(
        writer,
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste,
        Show
    );
}

fn render(frame: &mut Frame<'_>, app: &mut AppState) {
    frame.render_widget(Block::new().style(Style::default().bg(BG)), frame.area());
    let [header_area, body_area, input_area, footer_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    render_header(frame, app, header_area);
    if body_area.width >= 100 {
        let [chat_area, activity_area] =
            Layout::horizontal([Constraint::Percentage(72), Constraint::Percentage(28)])
                .spacing(1)
                .areas(body_area);
        render_chat(frame, app, chat_area);
        render_activity(frame, app, activity_area);
    } else if body_area.height >= 20 {
        let [chat_area, activity_area] =
            Layout::vertical([Constraint::Min(9), Constraint::Length(8)])
                .spacing(1)
                .areas(body_area);
        render_chat(frame, app, chat_area);
        render_activity(frame, app, activity_area);
    } else {
        render_chat(frame, app, body_area);
    }
    render_input(frame, app, input_area);
    render_footer(frame, app, footer_area);
    if let Some(modal) = &mut app.modal {
        render_modal(frame, modal, &app.queue);
    }
}

fn render_header(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
    let block = Block::new()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(PANEL));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [identity, state] =
        Layout::horizontal([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(inner);
    let context = if app.context_tokens == 0 {
        "context —".to_string()
    } else {
        format!("context {}", compact_number(app.context_tokens))
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" GENERALIST ", Style::default().fg(BG).bg(CYAN).bold()),
            Span::raw("  "),
            Span::styled(&app.api, Style::default().fg(TEXT).bold()),
            Span::styled("  /  ", Style::default().fg(MUTED)),
            Span::styled(&app.model, Style::default().fg(PURPLE)),
            Span::styled(
                format!(
                    "  ·  code mode / {} bridges  ·  {context}  ·  {} queued",
                    app.bridge_count,
                    app.queue.len()
                ),
                Style::default().fg(MUTED),
            ),
        ]))
        .style(Style::default().bg(PANEL)),
        identity,
    );
    let status = if app.busy {
        format!("{} {}", spinner(app.spinner_tick), app.status)
    } else {
        format!("● {}", app.status)
    };
    frame.render_widget(
        Paragraph::new(status).alignment(Alignment::Right).style(
            Style::default()
                .fg(if app.busy { YELLOW } else { GREEN })
                .bg(PANEL),
        ),
        state,
    );
}

fn render_chat(frame: &mut Frame<'_>, app: &mut AppState, area: Rect) {
    let block = panel_block(" Conversation ");
    let inner = block.inner(area);
    let text = chat_text(&app.chat);
    let width = inner.width.max(1);
    let paragraph = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(TEXT).bg(PANEL));
    // Use the same WordWrapper implementation for scroll bounds and drawing.
    // A width-based estimate undercounts rows whenever wrapping moves a whole
    // word to the next line, which makes the real bottom unreachable.
    let total_lines = paragraph.line_count(width);
    let visible = inner.height as usize;
    let max_scroll = total_lines.saturating_sub(visible);
    app.update_chat_metrics(max_scroll);
    let lines_from_latest = app.lines_from_latest();
    let title = if app.follow_latest {
        " Conversation ".to_string()
    } else {
        format!(" Conversation · {lines_from_latest} lines from latest ")
    };
    frame.render_widget(panel_block(title), area);
    let scroll = app.chat_scroll;
    frame.render_widget(
        paragraph.scroll((scroll.min(u16::MAX as usize) as u16, 0)),
        inner,
    );

    if max_scroll > 0 && inner.width > 2 {
        let mut scrollbar_state = ScrollbarState::new(total_lines).position(scroll);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .thumb_symbol("▐")
                .track_symbol(Some("│")),
            inner,
            &mut scrollbar_state,
        );
    }
}

fn render_activity(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
    let block = panel_block(format!(" Queue {} · Tool activity ", app.queue.len()));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if app.activity.is_empty() && app.queue.is_empty() {
        frame.render_widget(
            Paragraph::new("No queued work or tool calls")
                .style(Style::default().fg(MUTED).bg(PANEL))
                .alignment(Alignment::Center),
            inner,
        );
        return;
    }
    let mut items = app
        .queue
        .iter()
        .take(6)
        .map(|prompt| {
            let (label, color) = match prompt.delivery {
                DeliveryMode::Steer => ("↪ steer", PURPLE),
                DeliveryMode::FollowUp => ("＋ next", CYAN),
            };
            ListItem::new(vec![
                Line::from(Span::styled(label, Style::default().fg(color).bold())),
                Line::from(Span::styled(
                    format!(
                        "  {}",
                        truncate_middle(&sanitize_terminal_text(prompt.text.trim()), 100)
                    ),
                    Style::default().fg(TEXT),
                )),
            ])
            .style(Style::default().bg(PANEL))
        })
        .collect::<Vec<_>>();
    if !app.queue.is_empty() && !app.activity.is_empty() {
        items.push(ListItem::new(Line::from(Span::styled(
            "─ recent tools ─",
            Style::default().fg(BORDER),
        ))));
    }
    items.extend(app.activity.iter().rev().map(|item| {
        let (icon, color) = activity_icon(item.status);
        let name = if item.via_code_mode {
            format!("↳ tools.{}", item.name)
        } else if item.status == ActivityStatus::Rejected {
            format!("rejected native · {}", item.name)
        } else {
            item.name.clone()
        };
        let mut lines = vec![Line::from(vec![
            Span::styled(format!("{icon} "), Style::default().fg(color).bold()),
            Span::styled(name, Style::default().fg(YELLOW).bold()),
            Span::styled(format!("  {}", item.timestamp), Style::default().fg(MUTED)),
        ])];
        let detail = if item.output.is_empty() {
            &item.input
        } else {
            &item.output
        };
        if let Some(first) = detail.lines().find(|line| !line.trim().is_empty()) {
            lines.push(Line::from(Span::styled(
                format!("  {}", truncate_middle(first, 120)),
                Style::default().fg(MUTED),
            )));
        }
        ListItem::new(lines).style(Style::default().bg(PANEL))
    }));
    frame.render_widget(List::new(items).style(Style::default().bg(PANEL)), inner);
}

fn render_input(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
    let hint = if app.busy {
        "Enter steer · Tab/Alt+Enter follow-up · Ctrl+J newline"
    } else {
        "Enter send · Ctrl+J newline"
    };
    let block = Block::new()
        .title(Line::from(vec![
            Span::styled(" Message ", Style::default().fg(CYAN).bold()),
            Span::styled(hint, Style::default().fg(MUTED)),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(CYAN))
        .style(Style::default().bg(PANEL_ALT));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let width = inner.width.max(1);
    let (shown, cursor_x) = visible_editor(&app.input, app.input_cursor, width as usize);
    let input = if shown.is_empty() {
        Line::from(Span::styled(
            "Ask anything, or type /help…",
            Style::default().fg(MUTED),
        ))
    } else {
        Line::from(Span::styled(shown, Style::default().fg(TEXT)))
    };
    frame.render_widget(
        Paragraph::new(input).style(Style::default().bg(PANEL_ALT)),
        inner,
    );
    if app.modal.is_none() {
        frame.set_cursor_position(Position::new(
            inner.x + cursor_x.min(inner.width.saturating_sub(1)),
            inner.y,
        ));
    }
}

fn render_footer(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
    let left = if app.busy {
        " F1 help  F2 queue  Alt+↑ restore  Esc/Ctrl+C interrupt"
    } else {
        " F1 help  F2 queue  Alt+↑ restore  Ctrl+C quit"
    };
    let right = if app.follow_latest {
        "following latest "
    } else {
        "scroll paused "
    };
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Percentage(72), Constraint::Percentage(28)]).areas(area);
    frame.render_widget(
        Paragraph::new(left).style(Style::default().fg(MUTED).bg(BG)),
        left_area,
    );
    frame.render_widget(
        Paragraph::new(right)
            .alignment(Alignment::Right)
            .style(Style::default().fg(MUTED).bg(BG)),
        right_area,
    );
}

fn render_modal(frame: &mut Frame<'_>, modal: &mut Modal, queue: &[QueuedPrompt]) {
    match modal {
        Modal::Help => render_help(frame),
        Modal::Select {
            title,
            items,
            selected,
        } => render_select(frame, title, items, *selected),
        Modal::Prompt {
            title,
            value,
            cursor,
        } => render_prompt(frame, title, value, *cursor),
        Modal::Permission {
            request,
            selected,
            scroll,
            ..
        } => render_permission(frame, request, *selected, scroll),
        Modal::Queue { selected, editing } => {
            render_queue(frame, queue, *selected, editing.as_ref())
        }
    }
}

fn render_help(frame: &mut Frame<'_>) {
    let area = centered(frame.area(), 78, 80, 62, 20);
    frame.render_widget(Clear, area);
    let block = modal_block(" Help & shortcuts ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let lines = vec![
        Line::from(Span::styled("Commands", Style::default().fg(CYAN).bold())),
        Line::from("  /save       save this conversation"),
        Line::from("  /load       load a saved conversation"),
        Line::from("  /model      switch API or model"),
        Line::from("  /compact    summarize older context"),
        Line::from("  /clear      clear conversation history"),
        Line::from("  exit        close generalist"),
        Line::from(""),
        Line::from(Span::styled("While idle", Style::default().fg(CYAN).bold())),
        Line::from("  Enter send · Ctrl+J newline · ↑/↓ input history"),
        Line::from(""),
        Line::from(Span::styled(
            "While running",
            Style::default().fg(CYAN).bold(),
        )),
        Line::from("  Enter steer at next safe boundary"),
        Line::from("  Tab or Alt+Enter queue a separate follow-up"),
        Line::from("  Esc or Ctrl+C interrupt safely"),
        Line::from(""),
        Line::from(Span::styled("Queue", Style::default().fg(CYAN).bold())),
        Line::from("  F2 manage · Alt+↑ restore latest"),
        Line::from("  e edit · d delete · s steer/follow-up"),
        Line::from("  Ctrl+↑/↓ reorder · r restore selected"),
        Line::from(""),
        Line::from(Span::styled("Editor", Style::default().fg(CYAN).bold())),
        Line::from("  Ctrl+A/E home/end · Ctrl+U clear · Ctrl+W delete word"),
        Line::from("  PgUp/PgDn scroll conversation · mouse wheel also works"),
        Line::from(""),
        Line::from(Span::styled(
            "Esc, Enter, or F1 closes this window",
            Style::default().fg(MUTED),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL_ALT)),
        inner,
    );
}

fn render_queue(
    frame: &mut Frame<'_>,
    queue: &[QueuedPrompt],
    selected: usize,
    editing: Option<&QueueEditor>,
) {
    let area = centered(frame.area(), 84, 82, 66, 16);
    frame.render_widget(Clear, area);
    let block = modal_block(format!(" Queued messages ({}) ", queue.len()));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [list_area, editor_area, hint_area] = Layout::vertical([
        Constraint::Min(6),
        Constraint::Length(if editing.is_some() { 3 } else { 0 }),
        Constraint::Length(2),
    ])
    .areas(inner);

    if queue.is_empty() {
        frame.render_widget(
            Paragraph::new("The queue is empty. Keep typing while the agent works.")
                .alignment(Alignment::Center)
                .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
            list_area,
        );
    } else {
        let visible_items = (list_area.height as usize / 2).max(1);
        let start = selected
            .saturating_add(1)
            .saturating_sub(visible_items)
            .min(queue.len().saturating_sub(visible_items));
        let items = queue
            .iter()
            .enumerate()
            .skip(start)
            .take(visible_items)
            .map(|(index, prompt)| {
                let active = index == selected;
                let (label, color) = match prompt.delivery {
                    DeliveryMode::Steer => ("STEER", PURPLE),
                    DeliveryMode::FollowUp => ("NEXT ", CYAN),
                };
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            if active { "› " } else { "  " },
                            Style::default().fg(CYAN).bold(),
                        ),
                        Span::styled(label, Style::default().fg(color).bold()),
                        Span::styled(format!("  #{}", prompt.id), Style::default().fg(MUTED)),
                    ]),
                    Line::from(Span::styled(
                        format!(
                            "    {}",
                            truncate_middle(&sanitize_terminal_text(prompt.text.trim()), 180)
                        ),
                        Style::default().fg(if active { TEXT } else { MUTED }),
                    )),
                ])
                .style(Style::default().bg(if active {
                    PANEL
                } else {
                    PANEL_ALT
                }))
            })
            .collect::<Vec<_>>();
        frame.render_widget(List::new(items), list_area);
    }

    if let Some(editor) = editing {
        let block = Block::new()
            .title(" Edit queued prompt ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(CYAN));
        let editor_inner = block.inner(editor_area);
        frame.render_widget(block, editor_area);
        let (shown, cursor_x) =
            visible_editor(&editor.value, editor.cursor, editor_inner.width as usize);
        frame.render_widget(
            Paragraph::new(shown).style(Style::default().fg(TEXT).bg(PANEL_ALT)),
            editor_inner,
        );
        frame.set_cursor_position(Position::new(
            editor_inner.x + cursor_x.min(editor_inner.width.saturating_sub(1)),
            editor_inner.y,
        ));
    }

    let hint = if editing.is_some() {
        "Enter save · Esc cancel edit"
    } else {
        "e edit · d delete · s mode · Ctrl+↑/↓ reorder · r restore · Esc close"
    };
    frame.render_widget(
        Paragraph::new(hint)
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
        hint_area,
    );
}

fn render_select(frame: &mut Frame<'_>, title: &str, items: &[String], selected: usize) {
    let height = (items.len() as u16 + 4).clamp(7, 20);
    let area = centered(frame.area(), 60, 60, 44, height);
    frame.render_widget(Clear, area);
    let block = modal_block(format!(" {} ", sanitize_terminal_text(title)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let list = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let selected = index == selected;
            ListItem::new(Line::from(vec![
                Span::styled(
                    if selected { "› " } else { "  " },
                    Style::default().fg(CYAN).bold(),
                ),
                Span::styled(
                    sanitize_terminal_text(item),
                    Style::default()
                        .fg(if selected { TEXT } else { MUTED })
                        .add_modifier(if selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
            ]))
            .style(Style::default().bg(if selected { PANEL } else { PANEL_ALT }))
        })
        .collect::<Vec<_>>();
    frame.render_widget(List::new(list), inner);
}

fn render_prompt(frame: &mut Frame<'_>, title: &str, value: &str, cursor: usize) {
    let area = centered(frame.area(), 66, 66, 48, 5);
    frame.render_widget(Clear, area);
    let block = modal_block(format!(" {} ", sanitize_terminal_text(title)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let (shown, cursor_x) = visible_editor(value, cursor, inner.width as usize);
    frame.render_widget(
        Paragraph::new(shown).style(Style::default().fg(TEXT).bg(PANEL_ALT)),
        inner,
    );
    frame.set_cursor_position(Position::new(
        inner.x + cursor_x.min(inner.width.saturating_sub(1)),
        inner.y,
    ));
}

fn render_permission(
    frame: &mut Frame<'_>,
    request: &ToolExecutionRequest,
    selected: usize,
    scroll: &mut u16,
) {
    let area = centered(frame.area(), 88, 92, 70, 18);
    frame.render_widget(Clear, area);
    let block = Block::new()
        .title(Line::from(vec![
            Span::styled(" Permission required ", Style::default().fg(YELLOW).bold()),
            Span::styled(
                sanitize_terminal_text(&request.tool_name),
                Style::default().fg(CYAN).bold(),
            ),
            Span::raw(" "),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(YELLOW))
        .style(Style::default().bg(PANEL_ALT));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [description_area, detail_area, choices_area, hint_area] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(5),
        Constraint::Length(4),
        Constraint::Length(1),
    ])
    .areas(inner);
    frame.render_widget(
        Paragraph::new(sanitize_terminal_text(&request.tool_description))
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
        description_area,
    );
    let detail = permission_detail(request);
    let detail_inner = Block::new().borders(Borders::ALL).inner(detail_area);
    let detail = Paragraph::new(detail).wrap(Wrap { trim: false });
    let detail_lines = detail.line_count(detail_inner.width.max(1));
    let max_scroll = detail_lines.saturating_sub(detail_inner.height as usize);
    *scroll = (*scroll as usize).min(max_scroll).min(u16::MAX as usize) as u16;
    frame.render_widget(
        detail
            .scroll((*scroll, 0))
            .block(
                Block::new()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(BORDER))
                    .title(" Input "),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
        detail_area,
    );
    let choices = [
        "[a] Always allow this tool",
        "[y] Allow once",
        "[d] Always deny this tool",
        "[n] Deny once",
    ];
    let items = choices
        .iter()
        .enumerate()
        .map(|(index, choice)| {
            let active = index == selected;
            ListItem::new(Line::from(vec![
                Span::styled(
                    if active { "› " } else { "  " },
                    Style::default()
                        .fg(if index < 2 { GREEN } else { RED })
                        .bold(),
                ),
                Span::styled(
                    *choice,
                    Style::default()
                        .fg(if active { TEXT } else { MUTED })
                        .add_modifier(if active {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
            ]))
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(items).style(Style::default().bg(PANEL_ALT)),
        choices_area,
    );
    frame.render_widget(
        Paragraph::new("↑/↓ choose · Enter confirm · PgUp/PgDn inspect · Esc deny once")
            .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
        hint_area,
    );
}

fn chat_text(entries: &[ChatEntry]) -> Text<'static> {
    let mut lines = Vec::<Line<'static>>::new();
    for entry in entries {
        let (label, color) = match entry.kind {
            ChatKind::User => ("YOU", GREEN),
            ChatKind::Assistant => ("ASSISTANT", CYAN),
            ChatKind::Info => ("INFO", YELLOW),
            ChatKind::Error => ("ERROR", RED),
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{}  ", entry.timestamp), Style::default().fg(MUTED)),
            Span::styled(label, Style::default().fg(color).bold()),
        ]));
        let mut in_code = false;
        for body_line in entry.body.lines() {
            if body_line.trim_start().starts_with("```") {
                in_code = !in_code;
                lines.push(Line::from(Span::styled(
                    format!("  {body_line}"),
                    Style::default().fg(PURPLE),
                )));
            } else if in_code {
                lines.push(Line::from(Span::styled(
                    format!("  {body_line}"),
                    Style::default().fg(YELLOW).bg(PANEL_ALT),
                )));
            } else if body_line.starts_with('#') {
                lines.push(Line::from(Span::styled(
                    format!("  {body_line}"),
                    Style::default().fg(TEXT).bold(),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    format!("  {body_line}"),
                    Style::default().fg(TEXT),
                )));
            }
        }
        lines.push(Line::from(""));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "Start a conversation below. Tool calls will appear in the activity panel.",
            Style::default().fg(MUTED),
        )));
    }
    Text::from(lines)
}

fn permission_detail(request: &ToolExecutionRequest) -> Text<'static> {
    let diff = (request.tool_name == "patch_file")
        .then(|| request.input.get("diff").and_then(Value::as_str))
        .flatten();
    if let Some(diff) = diff {
        let mut lines = Vec::new();
        if let Some(path) = request.input.get("path").and_then(Value::as_str) {
            lines.push(Line::from(vec![
                Span::styled("Target: ", Style::default().fg(MUTED)),
                Span::styled(sanitize_terminal_text(path), Style::default().fg(YELLOW)),
            ]));
        }
        for line in diff.lines() {
            let color = if line.starts_with("+++") || line.starts_with("---") {
                CYAN
            } else if line.starts_with("@@") {
                PURPLE
            } else if line.starts_with('+') {
                GREEN
            } else if line.starts_with('-') {
                RED
            } else {
                TEXT
            };
            lines.push(Line::from(Span::styled(
                sanitize_terminal_text(line),
                Style::default().fg(color),
            )));
        }
        Text::from(lines)
    } else {
        Text::from(sanitize_terminal_text(
            &serde_json::to_string_pretty(&request.input)
                .unwrap_or_else(|_| request.input.to_string()),
        ))
    }
}

fn panel_block<'a>(title: impl Into<Line<'a>>) -> Block<'a> {
    Block::new()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(PANEL))
}

fn modal_block<'a>(title: impl Into<Line<'a>>) -> Block<'a> {
    Block::new()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(Style::default().fg(CYAN))
        .style(Style::default().bg(PANEL_ALT))
}

fn centered(parent: Rect, percent_x: u16, percent_y: u16, min_width: u16, min_height: u16) -> Rect {
    let width = ((parent.width as u32 * percent_x as u32) / 100) as u16;
    let height = ((parent.height as u32 * percent_y as u32) / 100) as u16;
    let width = width.max(min_width.min(parent.width)).min(parent.width);
    let height = height.max(min_height.min(parent.height)).min(parent.height);
    Rect {
        x: parent.x + parent.width.saturating_sub(width) / 2,
        y: parent.y + parent.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn visible_editor(value: &str, cursor: usize, width: usize) -> (String, u16) {
    if width == 0 {
        return (String::new(), 0);
    }
    let chars = value.chars().map(editor_display_char).collect::<Vec<_>>();
    let cursor = cursor.min(chars.len());
    let mut start = cursor;
    let mut before_width = 0usize;
    while start > 0 {
        let char_width = chars[start - 1].width().unwrap_or(0).max(1);
        if before_width + char_width >= width {
            break;
        }
        before_width += char_width;
        start -= 1;
    }
    let mut shown = String::new();
    let mut shown_width = 0usize;
    for ch in chars.iter().skip(start) {
        let char_width = ch.width().unwrap_or(0).max(1);
        if shown_width + char_width > width {
            break;
        }
        shown.push(*ch);
        shown_width += char_width;
    }
    (shown, before_width.min(u16::MAX as usize) as u16)
}

fn editor_display_char(ch: char) -> char {
    if ch == '\n' {
        '↵'
    } else {
        terminal_display_char(ch)
    }
}

fn sanitize_terminal_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch == '\n' {
                '\n'
            } else {
                terminal_display_char(ch)
            }
        })
        .collect()
}

fn terminal_display_char(ch: char) -> char {
    match ch {
        '\t' => '⇥',
        '\r' => '↵',
        '\u{1b}' => '␛',
        '\u{7f}' => '␡',
        ch if ch <= '\u{1f}' => char::from_u32(0x2400 + ch as u32).unwrap_or('�'),
        ch if ch.is_control() => '�',
        ch => ch,
    }
}

fn insert_text(value: &mut String, cursor: &mut usize, text: &str) {
    let text = text.chars().filter(|ch| *ch != '\r').collect::<String>();
    if text.is_empty() {
        return;
    }
    let inserted_chars = text.chars().count();
    let byte = byte_index(value, *cursor);
    value.insert_str(byte, &text);
    *cursor += inserted_chars;
}

fn insert_char(value: &mut String, cursor: &mut usize, ch: char) {
    let byte = byte_index(value, *cursor);
    value.insert(byte, ch);
    *cursor += 1;
}

fn backspace(value: &mut String, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }
    let start = byte_index(value, *cursor - 1);
    let end = byte_index(value, *cursor);
    value.replace_range(start..end, "");
    *cursor -= 1;
}

fn delete_at_cursor(value: &mut String, cursor: usize) {
    if cursor >= value.chars().count() {
        return;
    }
    let start = byte_index(value, cursor);
    let end = byte_index(value, cursor + 1);
    value.replace_range(start..end, "");
}

fn delete_previous_word(value: &mut String, cursor: &mut usize) {
    while *cursor > 0
        && value
            .chars()
            .nth(*cursor - 1)
            .is_some_and(char::is_whitespace)
    {
        backspace(value, cursor);
    }
    while *cursor > 0
        && value
            .chars()
            .nth(*cursor - 1)
            .is_some_and(|ch| !ch.is_whitespace())
    {
        backspace(value, cursor);
    }
}

fn truncate_at_char(value: &mut String, cursor: usize) {
    value.truncate(byte_index(value, cursor));
}

fn byte_index(value: &str, character_index: usize) -> usize {
    value
        .char_indices()
        .map(|(index, _)| index)
        .nth(character_index)
        .unwrap_or(value.len())
}

fn is_key_press(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

fn submission_delivery(key: KeyEvent, busy: bool) -> Option<DeliveryMode> {
    if key
        .modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::SHIFT)
    {
        return None;
    }
    match key.code {
        KeyCode::Tab => Some(DeliveryMode::FollowUp),
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => Some(DeliveryMode::FollowUp),
        KeyCode::Enter => Some(if busy {
            DeliveryMode::Steer
        } else {
            DeliveryMode::FollowUp
        }),
        _ => None,
    }
}

fn permission_choice(index: usize) -> PermissionChoice {
    match index {
        0 => PermissionChoice::AllowAlways,
        1 => PermissionChoice::AllowOnce,
        2 => PermissionChoice::DenyAlways,
        _ => PermissionChoice::DenyOnce,
    }
}

fn is_code_mode_protocol_rejection(content: &str) -> bool {
    content.contains("code mode permits only the model-facing `python` tool")
        || content.contains("Direct tool calls are disabled in code mode")
}

fn activity_icon(status: ActivityStatus) -> (&'static str, Color) {
    match status {
        ActivityStatus::Running => ("●", YELLOW),
        ActivityStatus::Success => ("✓", GREEN),
        ActivityStatus::Failed => ("✗", RED),
        ActivityStatus::Rejected => ("⊘", PURPLE),
        ActivityStatus::Denied => ("⊘", RED),
        ActivityStatus::Cancelled => ("■", MUTED),
    }
}

fn spinner(tick: usize) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[tick % FRAMES.len()]
}

fn compact_number(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}m", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use serde_json::json;

    #[test]
    fn editor_operations_are_unicode_safe() {
        let mut value = "a🦀c".to_string();
        let mut cursor = 2;
        insert_char(&mut value, &mut cursor, 'ß');
        assert_eq!(value, "a🦀ßc");
        backspace(&mut value, &mut cursor);
        assert_eq!(value, "a🦀c");
        backspace(&mut value, &mut cursor);
        assert_eq!(value, "ac");
        assert_eq!(cursor, 1);
    }

    #[test]
    fn bulk_paste_is_unicode_safe_and_filters_carriage_returns() {
        let mut value = "ab".to_string();
        let mut cursor = 1;
        insert_text(&mut value, &mut cursor, "\r🦀x");
        assert_eq!(value, "a🦀xb");
        assert_eq!(cursor, 3);
    }

    #[test]
    fn untrusted_control_bytes_are_rendered_inert() {
        let sanitized = sanitize_terminal_text("red\x1b[31m\tbell\u{7}\nok");
        assert_eq!(sanitized, "red␛[31m⇥bell␇\nok");
        assert!(!sanitized.contains('\u{1b}'));
        assert!(!sanitized.chars().any(|ch| ch.is_control() && ch != '\n'));

        let (editor, _) = visible_editor("a\x1bb\n", 4, 20);
        assert_eq!(editor, "a␛b↵");

        let mut app = AppState::new("api\x1b", "model");
        app.push_chat(ChatKind::Assistant, "answer\x1b[2J");
        assert_eq!(app.api, "api␛");
        assert_eq!(app.chat[0].body, "answer␛[2J");
    }

    #[test]
    fn terminal_cleanup_disables_every_enabled_mode() {
        let mut output = Vec::new();
        leave_terminal(&mut output);
        assert!(output.windows(8).any(|window| window == b"\x1b[?1049l"));
        assert!(output.windows(8).any(|window| window == b"\x1b[?2004l"));
        assert!(
            output.windows(8).any(|window| window == b"\x1b[?1000l")
                || output.windows(8).any(|window| window == b"\x1b[?1006l")
        );
    }

    #[test]
    fn paused_scroll_is_anchored_and_cannot_overshoot() {
        let mut app = AppState::new("api", "model");
        app.push_chat(ChatKind::Assistant, "content");
        app.update_chat_metrics(100);
        assert_eq!(app.chat_scroll, 100);
        assert!(app.follow_latest);

        app.scroll_chat_up(10);
        assert_eq!(app.chat_scroll, 90);
        assert!(!app.follow_latest);

        // New output extends the bottom, but a paused viewport keeps the same
        // absolute top line instead of drifting with the stream.
        app.update_chat_metrics(130);
        assert_eq!(app.chat_scroll, 90);
        assert_eq!(app.lines_from_latest(), 40);

        app.scroll_chat_up(usize::MAX);
        assert_eq!(app.chat_scroll, 0);
        app.scroll_chat_down(usize::MAX);
        assert_eq!(app.chat_scroll, 130);
        assert!(app.follow_latest);
    }

    #[test]
    fn mouse_wheel_respects_modal_focus() {
        let mut app = AppState::new("api", "model");
        app.update_chat_metrics(100);
        app.modal = Some(Modal::Help);
        app.handle_mouse_scroll(MouseEventKind::ScrollUp);
        assert_eq!(app.chat_scroll, 100, "help leaked scroll to transcript");

        app.modal = Some(Modal::Permission {
            id: 1,
            request: ToolExecutionRequest {
                tool_use_id: "tool".into(),
                tool_name: "bash".into(),
                input: json!({"command": "true"}),
                tool_description: "run".into(),
            },
            selected: 1,
            scroll: 0,
        });
        app.handle_mouse_scroll(MouseEventKind::ScrollDown);
        assert!(matches!(
            app.modal,
            Some(Modal::Permission { scroll: 3, .. })
        ));
        assert_eq!(app.chat_scroll, 100);

        app.queue = (0..2)
            .map(|id| QueuedPrompt {
                id,
                text: id.to_string(),
                delivery: DeliveryMode::FollowUp,
            })
            .collect();
        app.modal = Some(Modal::Queue {
            selected: 0,
            editing: None,
        });
        app.handle_mouse_scroll(MouseEventKind::ScrollDown);
        assert!(matches!(app.modal, Some(Modal::Queue { selected: 1, .. })));
        assert_eq!(app.chat_scroll, 100);
    }

    #[test]
    fn restoring_queue_text_never_overwrites_a_draft() {
        let prompt = QueuedPrompt {
            id: 7,
            text: "queued".into(),
            delivery: DeliveryMode::FollowUp,
        };
        let mut app = AppState::new("api", "model");
        app.input = "unsent draft".into();
        app.input_cursor = app.input.chars().count();

        assert!(!app.restore_to_composer(&prompt));
        assert_eq!(app.input, "unsent draft");

        app.input.clear();
        app.input_cursor = 0;
        assert!(app.restore_to_composer(&prompt));
        assert_eq!(app.input, "queued");
    }

    #[test]
    fn saved_same_named_tool_results_stay_with_their_ids() {
        let history = vec![
            Message::assistant(vec![
                ContentBlock::ToolUse {
                    name: "read_file".into(),
                    input: json!({"path": "one"}),
                    id: "one".into(),
                },
                ContentBlock::ToolUse {
                    name: "read_file".into(),
                    input: json!({"path": "two"}),
                    id: "two".into(),
                },
            ]),
            Message::user(vec![
                ContentBlock::ToolResult {
                    content: "first output".into(),
                    tool_use_id: "one".into(),
                    is_error: None,
                },
                ContentBlock::ToolResult {
                    content: "second output".into(),
                    tool_use_id: "two".into(),
                    is_error: Some(true),
                },
            ]),
        ];
        let mut app = AppState::new("api", "model");
        app.load_history(&history);

        assert_eq!(app.activity[0].output, "first output");
        assert_eq!(app.activity[0].status, ActivityStatus::Success);
        assert_eq!(app.activity[1].output, "second output");
        assert_eq!(app.activity[1].status, ActivityStatus::Failed);
    }

    #[test]
    fn saved_protocol_violation_is_labeled_rejected_not_executed() {
        let history = vec![
            Message::assistant(vec![ContentBlock::ToolUse {
                name: "tools.firecrawl_search".into(),
                input: json!({"query": "x"}),
                id: "bad-native".into(),
            }]),
            Message::user(vec![ContentBlock::ToolResult {
                content: "The provider emitted undeclared native tool call \
                          `tools.firecrawl_search`. It was not executed: code mode permits only \
                          the model-facing `python` tool."
                    .into(),
                tool_use_id: "bad-native".into(),
                is_error: Some(true),
            }]),
        ];
        let mut app = AppState::new("api", "model");
        app.load_history(&history);

        assert_eq!(app.activity[0].status, ActivityStatus::Rejected);
    }

    #[test]
    fn aborted_stream_is_visibly_marked_uncommitted() {
        let mut app = AppState::new("api", "model");
        app.apply_agent_event(AgentEvent::AssistantTextDelta("partial".into()));
        app.apply_agent_event(AgentEvent::AssistantStreamAborted {
            reason: "interrupted before commit".into(),
        });
        app.apply_agent_event(AgentEvent::ApiCallFinished { usage: None });

        assert!(app.chat[0].body.starts_with("partial"));
        assert!(app.chat[0].body.contains("[Uncommitted stream:"));
        assert!(app.streaming_chat.is_none());
    }

    #[test]
    fn streaming_and_nested_tools_update_existing_entries() {
        let mut app = AppState::new("OpenAI-compatible", "qwen");
        app.apply_agent_event(AgentEvent::ApiCallStarted);
        app.apply_agent_event(AgentEvent::AssistantTextDelta("hel".into()));
        app.apply_agent_event(AgentEvent::AssistantTextDelta("lo".into()));
        assert_eq!(app.chat.len(), 1);
        assert_eq!(app.chat[0].body, "hello");

        app.apply_agent_event(AgentEvent::ToolCallStarted {
            name: "python".into(),
            input: json!({"code": "print(1)"}),
        });
        app.apply_agent_event(AgentEvent::ToolCallStarted {
            name: "read_file".into(),
            input: json!({"path": "README.md"}),
        });
        app.apply_agent_event(AgentEvent::ToolCallFinished {
            name: "read_file".into(),
            outcome: ToolCallOutcome::Success,
            content: "contents".into(),
        });
        app.apply_agent_event(AgentEvent::ToolCallFinished {
            name: "python".into(),
            outcome: ToolCallOutcome::Success,
            content: "done".into(),
        });
        assert_eq!(app.activity.len(), 2);
        assert!(app
            .activity
            .iter()
            .all(|item| item.status == ActivityStatus::Success));
        assert!(!app.activity[0].via_code_mode);
        assert!(app.activity[1].via_code_mode);
    }

    #[test]
    fn dashboard_renders_with_test_backend() {
        let mut app = AppState::new("OpenAI-compatible", "qwen3.6:35b-a3b");
        app.bridge_count = 14;
        app.context_tokens = 12_400;
        app.push_user("Build a better interface");
        app.push_chat(ChatKind::Assistant, "Working on it.");
        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &mut app)).unwrap();
        let rendered = terminal.backend().to_string();
        assert!(rendered.contains("GENERALIST"));
        assert!(rendered.contains("OpenAI-compatible"));
        assert!(rendered.contains("code mode"));
        assert!(rendered.contains("Conversation"));
        assert!(rendered.contains("Tool activity"));
        assert!(rendered.contains("Build a better interface"));
    }

    #[test]
    fn word_wrapped_chat_reaches_the_real_bottom() {
        let mut app = AppState::new("api", "model");
        let long_words = (0..80)
            .map(|_| "abcdefghijklmnop")
            .collect::<Vec<_>>()
            .join(" ");
        app.push_user(long_words);
        app.push_chat(ChatKind::Assistant, "BOTTOM-MARKER");

        let backend = TestBackend::new(32, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render_chat(frame, &mut app, frame.area()))
            .unwrap();

        assert!(app.follow_latest);
        assert!(
            terminal.backend().to_string().contains("BOTTOM-MARKER"),
            "follow-latest stopped above Ratatui's word-wrapped bottom"
        );

        app.scroll_chat_up(usize::MAX);
        terminal
            .draw(|frame| render_chat(frame, &mut app, frame.area()))
            .unwrap();
        assert!(!app.follow_latest);
        assert!(!terminal.backend().to_string().contains("BOTTOM-MARKER"));

        app.scroll_chat_down(usize::MAX);
        terminal
            .draw(|frame| render_chat(frame, &mut app, frame.area()))
            .unwrap();
        assert!(app.follow_latest);
        assert!(
            terminal.backend().to_string().contains("BOTTOM-MARKER"),
            "scroll-down stopped above Ratatui's word-wrapped bottom"
        );
    }

    #[test]
    fn queue_modal_keeps_a_long_selection_visible() {
        let queue = (0..14)
            .map(|id| QueuedPrompt {
                id,
                text: format!("queue-item-{id}"),
                delivery: DeliveryMode::FollowUp,
            })
            .collect::<Vec<_>>();
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| render_queue(frame, &queue, 13, None))
            .unwrap();
        let rendered = terminal.backend().to_string();
        assert!(rendered.contains("queue-item-13"));
        assert!(!rendered.contains("queue-item-0"));
    }

    #[test]
    fn dashboard_and_modals_render_on_tiny_terminals() {
        let request = ToolExecutionRequest {
            tool_use_id: "tool".into(),
            tool_name: "bash".into(),
            input: json!({"command": "printf x"}),
            tool_description: "run a command".into(),
        };
        let modals = vec![
            None,
            Some(Modal::Help),
            Some(Modal::Select {
                title: "Select".into(),
                items: vec!["one".into()],
                selected: 0,
            }),
            Some(Modal::Prompt {
                title: "Prompt".into(),
                value: "value".into(),
                cursor: 5,
            }),
            Some(Modal::Permission {
                id: 1,
                request,
                selected: 1,
                scroll: u16::MAX,
            }),
            Some(Modal::Queue {
                selected: 0,
                editing: None,
            }),
        ];
        for (width, height) in [(1, 1), (8, 3), (24, 8), (60, 16)] {
            for modal in &modals {
                let mut app = AppState::new("api", "model");
                app.modal = modal.clone();
                let backend = TestBackend::new(width, height);
                let mut terminal = Terminal::new(backend).unwrap();
                terminal
                    .draw(|frame| render(frame, &mut app))
                    .unwrap_or_else(|error| {
                        panic!("{width}x{height} failed for {modal:?}: {error}")
                    });
            }
        }
    }

    #[test]
    fn composer_keys_distinguish_busy_steering_from_followups() {
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
        let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        let shift_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT);

        assert_eq!(
            submission_delivery(enter, false),
            Some(DeliveryMode::FollowUp)
        );
        assert_eq!(submission_delivery(enter, true), Some(DeliveryMode::Steer));
        assert_eq!(
            submission_delivery(alt_enter, true),
            Some(DeliveryMode::FollowUp)
        );
        assert_eq!(submission_delivery(tab, true), Some(DeliveryMode::FollowUp));
        assert_eq!(submission_delivery(shift_enter, true), None);
    }

    #[test]
    fn only_queue_mutations_request_terminal_event_persistence() {
        assert!(!UiAction::None.requires_queue_persist());
        assert!(!UiAction::Interrupt.requires_queue_persist());
        assert!(!UiAction::Exit.requires_queue_persist());
        assert!(UiAction::QueueChanged.requires_queue_persist());
        assert!(UiAction::Submit {
            text: "queued".into(),
            delivery: DeliveryMode::FollowUp,
        }
        .requires_queue_persist());
    }

    #[test]
    fn interrupted_turn_retires_all_nested_activity() {
        let mut app = AppState::new("OpenAI-compatible", "model");
        app.apply_agent_event(AgentEvent::ToolCallStarted {
            name: "python".into(),
            input: json!({}),
        });
        app.apply_agent_event(AgentEvent::ToolCallStarted {
            name: "bash".into(),
            input: json!({}),
        });

        app.cancel_running_activity();

        assert!(app.active_tools.is_empty());
        assert!(app
            .activity
            .iter()
            .all(|item| item.status == ActivityStatus::Cancelled));
    }
}
