//! Terminal lifecycle and input routing: [`TerminalUi`] owns raw mode, the
//! crossterm event stream, and drawing, and translates events into
//! [`UiAction`]s for the controller.

use crate::clipboard::write_osc52;
use crate::permissions::{PermissionChoice, ToolExecutionRequest};
use crate::runtime::{DeliveryMode, PromptQueue};
use crate::types::Message;
use crate::AgentEvent;
use crossterm::cursor::Show;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io::{self, Stdout};
use std::time::Duration;
use tokio::time::MissedTickBehavior;

use super::editor::{
    apply_editor_key, handle_prompt_editor_key, insert_char, insert_text, PromptEditorAction,
};
use super::render::render;
use super::sanitize_terminal_text;
use super::state::{AppState, Modal, QueueEditor};

const TICK_RATE: Duration = Duration::from_millis(100);

type TuiTerminal = Terminal<CrosstermBackend<Stdout>>;

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
        if self.app.copy_mode {
            return Ok(());
        }
        self.draw_now()
    }

    fn draw_now(&mut self) -> io::Result<()> {
        let app = &mut self.app;
        self.terminal.draw(|frame| render(frame, app))?;
        self.dirty = false;
        Ok(())
    }

    pub fn draw_if_dirty(&mut self) -> io::Result<()> {
        if self.dirty && !self.app.copy_mode {
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

    fn toggle_copy_mode(&mut self) -> io::Result<()> {
        if self.app.copy_mode {
            execute!(self.terminal.backend_mut(), EnableMouseCapture)?;
            self.app.copy_mode = false;
            self.dirty = true;
            self.draw_now()
        } else {
            execute!(self.terminal.backend_mut(), DisableMouseCapture)?;
            self.app.copy_mode = true;
            self.dirty = true;
            if let Err(error) = self.draw_now() {
                self.app.copy_mode = false;
                let _ = execute!(self.terminal.backend_mut(), EnableMouseCapture);
                return Err(error);
            }
            Ok(())
        }
    }

    /// Enter native terminal selection mode from an explicit local command.
    pub fn enter_copy_mode(&mut self) -> io::Result<()> {
        if !self.app.copy_mode {
            self.toggle_copy_mode()?;
        }
        Ok(())
    }

    /// Send an explicit, write-only clipboard request to the host terminal.
    pub fn request_clipboard_copy(&mut self, text: &str) -> io::Result<usize> {
        write_osc52(self.terminal.backend_mut(), text)
    }

    /// Toggle copy mode on F3, or resume it with F3/Esc, and suppress all
    /// other application input while it is active. The terminal keeps its
    /// native copy shortcuts; only the TUI's mouse capture and redraws are
    /// suspended.
    fn copy_mode_owns_event(&mut self, event: &Event) -> io::Result<bool> {
        if matches!(
            event,
            Event::Key(key) if copy_mode_toggle_key(self.app.copy_mode, *key)
        ) {
            self.toggle_copy_mode()?;
            return Ok(true);
        }
        Ok(self.app.copy_mode)
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

    /// Render cumulative provider usage observed since this UI process began.
    pub fn provider_usage_report(&self) -> String {
        self.app.provider_usage_report()
    }

    /// Clear cumulative provider usage. A currently active attempt, if any,
    /// remains represented so its eventual report cannot exceed its attempts.
    pub fn reset_provider_usage(&mut self) {
        self.app.reset_provider_usage();
        self.dirty = true;
    }

    pub fn set_goal(&mut self, goal: Option<&str>) {
        let goal = goal.map(sanitize_terminal_text);
        if self.app.goal == goal {
            return;
        }
        self.app.goal = goal;
        self.dirty = true;
    }

    pub fn set_busy(&mut self, busy: bool, status: &str) {
        let status = sanitize_terminal_text(status);
        let clear_turn = !busy && self.app.turn_active;
        let changed = self.app.busy != busy || self.app.status != status || clear_turn;
        self.app.busy = busy;
        self.app.status = status;
        if !busy {
            self.app.turn_active = false;
            self.app.streaming_chat = None;
            self.app.committing_streaming_chat = None;
            self.app.committing_reasoning = None;
        }
        self.dirty |= changed;
    }

    /// Record whether an agent turn currently owns mutable conversation state.
    /// Background operations may be busy without accepting steering.
    pub fn set_turn_active(&mut self, active: bool) {
        if self.app.turn_active == active {
            return;
        }
        self.app.turn_active = active;
        self.dirty = true;
    }

    pub fn set_bridge_count(&mut self, bridge_count: usize) {
        if self.app.bridge_count == bridge_count {
            return;
        }
        self.app.bridge_count = bridge_count;
        self.dirty = true;
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
                    if self.copy_mode_owns_event(&event)? {
                        continue;
                    }
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
                    if self.copy_mode_owns_event(&event)? {
                        continue;
                    }
                    self.dirty = true;
                    let Event::Key(key) = event else {
                        self.draw_if_dirty()?;
                        continue;
                    };
                    if !is_key_press(key) {
                        continue;
                    }
                    self.dirty = true;
                    // The modal lives in `app.modal` so the ordinary render
                    // path can draw it. If a future code path replaces it
                    // while this loop runs, resolve as cancelled instead of
                    // panicking.
                    let Some(Modal::Select { items, selected, .. }) = self.app.modal.as_mut()
                    else {
                        return Ok(None);
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
                    let event = event?;
                    if self.copy_mode_owns_event(&event)? {
                        continue;
                    }
                    self.dirty = true;
                    match event {
                        Event::Paste(text) => {
                            // See `select`: resolve as cancelled if the
                            // prompt modal was replaced out from under us.
                            let Some(Modal::Prompt { value, cursor, .. }) =
                                self.app.modal.as_mut()
                            else {
                                return Ok(None);
                            };
                            let text = text.replace(['\r', '\n'], "");
                            insert_text(value, cursor, &text);
                        }
                        Event::Key(key) if is_key_press(key) => {
                            let Some(Modal::Prompt { value, cursor, .. }) =
                                self.app.modal.as_mut()
                            else {
                                return Ok(None);
                            };
                            match handle_prompt_editor_key(value, cursor, key) {
                                PromptEditorAction::Cancel => {
                                    self.app.modal = None;
                                    return Ok(None);
                                }
                                PromptEditorAction::Submit => {
                                    let value = value.clone();
                                    self.app.modal = None;
                                    return Ok(Some(value));
                                }
                                PromptEditorAction::Continue => {}
                            }
                        }
                        _ => {}
                    }
                }
            }
            self.draw_if_dirty()?;
        }
    }

    pub fn handle_event(&mut self, event: Event, queue: &PromptQueue) -> io::Result<UiAction> {
        if self.copy_mode_owns_event(&event)? {
            return Ok(UiAction::None);
        }
        self.dirty = true;
        Ok(match event {
            Event::Key(key) if is_key_press(key) => self.handle_key(key, queue),
            Event::Paste(text) => {
                if !self.app.paste_chat_search(&text) {
                    if let Some(Modal::Queue {
                        editing: Some(editor),
                        ..
                    }) = self.app.modal.as_mut()
                    {
                        insert_text(&mut editor.value, &mut editor.cursor, &text);
                    } else if self.app.modal.is_none() {
                        self.app.insert_input_text(&text);
                    }
                }
                UiAction::None
            }
            Event::Mouse(mouse) => {
                self.app.handle_mouse_scroll(mouse.kind);
                UiAction::None
            }
            Event::Resize(_, _) => UiAction::None,
            _ => UiAction::None,
        })
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
            Some(Modal::Search(_)) => {
                self.app.handle_chat_search_key(key);
                return UiAction::None;
            }
            Some(Modal::Reasoning { .. }) => return self.handle_reasoning_key(key),
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
                KeyCode::Char('f') => self.app.open_chat_search(),
                KeyCode::Enter | KeyCode::Char('j') => self.app.insert_input_char('\n'),
                _ => {
                    apply_editor_key(&mut self.app.input, &mut self.app.input_cursor, &key);
                }
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

        if key.code == KeyCode::Tab && key.modifiers.is_empty() && self.app.complete_slash_command()
        {
            return UiAction::None;
        }

        if let Some(delivery) = submission_delivery(key, self.app.turn_active) {
            return self.submit(delivery);
        }

        match key.code {
            KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.app.insert_input_char('\n');
                UiAction::None
            }
            // Edit keys reset history browsing; pure cursor motion does not.
            KeyCode::Char(_) | KeyCode::Backspace | KeyCode::Delete => {
                apply_editor_key(&mut self.app.input, &mut self.app.input_cursor, &key);
                self.app.history_cursor = None;
                UiAction::None
            }
            KeyCode::Left | KeyCode::Right | KeyCode::Home | KeyCode::End => {
                apply_editor_key(&mut self.app.input, &mut self.app.input_cursor, &key);
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
            KeyCode::F(4) => {
                self.app.modal = Some(Modal::Reasoning {
                    scroll: 0,
                    max_scroll: 0,
                    follow_latest: true,
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

    fn handle_reasoning_key(&mut self, key: KeyEvent) -> UiAction {
        let Some(Modal::Reasoning {
            mut scroll,
            max_scroll,
            mut follow_latest,
        }) = self.app.modal.take()
        else {
            return UiAction::None;
        };

        match key.code {
            KeyCode::F(4) | KeyCode::Esc | KeyCode::Enter => return UiAction::None,
            KeyCode::PageUp => {
                scroll = scroll.saturating_sub(10);
                follow_latest = false;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                scroll = scroll.saturating_sub(1);
                follow_latest = false;
            }
            KeyCode::PageDown => {
                scroll = scroll.saturating_add(10).min(max_scroll);
                follow_latest = scroll == max_scroll;
            }
            KeyCode::Down | KeyCode::Char('j') => {
                scroll = scroll.saturating_add(1).min(max_scroll);
                follow_latest = scroll == max_scroll;
            }
            KeyCode::Home => {
                scroll = 0;
                follow_latest = false;
            }
            KeyCode::End => {
                scroll = max_scroll;
                follow_latest = true;
            }
            _ => {}
        }

        self.app.modal = Some(Modal::Reasoning {
            scroll,
            max_scroll,
            follow_latest,
        });
        UiAction::None
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

        if let Some(editor) = editing.take() {
            return self.handle_queue_editor_key(key, queue, selected, editor);
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
                    queue_changed = queue.toggle_delivery(id, self.app.turn_active);
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

    /// Handle one key while the queue modal is editing a prompt. `editing`
    /// stays `None` on Ctrl+C, Esc, and a resolved Enter, which returns the
    /// modal to browse mode.
    fn handle_queue_editor_key(
        &mut self,
        key: KeyEvent,
        queue: &PromptQueue,
        selected: usize,
        mut editor: QueueEditor,
    ) -> UiAction {
        let mut editing = None;
        let mut queue_changed = false;
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {}
                KeyCode::Enter | KeyCode::Char('j') => {
                    insert_char(&mut editor.value, &mut editor.cursor, '\n');
                    editing = Some(editor);
                }
                _ => {
                    apply_editor_key(&mut editor.value, &mut editor.cursor, &key);
                    editing = Some(editor);
                }
            }
        } else {
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
                _ => {
                    apply_editor_key(&mut editor.value, &mut editor.cursor, &key);
                    editing = Some(editor);
                }
            }
        }
        self.sync_queue(queue);
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

pub(crate) fn leave_terminal(writer: &mut impl io::Write) {
    let _ = execute!(
        writer,
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste,
        Show
    );
}

fn is_key_press(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

pub(crate) fn copy_mode_toggle_key(copy_mode: bool, key: KeyEvent) -> bool {
    is_key_press(key) && (key.code == KeyCode::F(3) || (copy_mode && key.code == KeyCode::Esc))
}

pub(crate) fn submission_delivery(key: KeyEvent, turn_active: bool) -> Option<DeliveryMode> {
    if key
        .modifiers
        .intersects(KeyModifiers::CONTROL | KeyModifiers::SHIFT)
    {
        return None;
    }
    match key.code {
        KeyCode::Tab => Some(DeliveryMode::FollowUp),
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::ALT) => Some(DeliveryMode::FollowUp),
        KeyCode::Enter => Some(if turn_active {
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
