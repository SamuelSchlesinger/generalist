//! Full-screen Ratatui frontend for the interactive agent CLI.
//!
//! The module is split into focused layers:
//!
//! - `state` — `AppState` and the projection of
//!   [`AgentEvent`](crate::AgentEvent)s onto it
//! - `terminal` — [`TerminalUi`], the only owner of raw mode, terminal
//!   input, and drawing, plus its key routing
//! - `render` — the pure render layer
//! - `editor` — single-line editor primitives shared by every text field
//! - `python_highlight` — the small Python lexer used for source previews

mod editor;
mod python_highlight;
mod render;
mod state;
mod terminal;
#[cfg(test)]
mod tests;

pub use terminal::{TerminalUi, UiAction};

#[cfg(test)]
use self::{editor::*, python_highlight::*, render::*, state::*, terminal::*};
#[cfg(test)]
use crate::permissions::ToolExecutionRequest;
#[cfg(test)]
use crate::runtime::{DeliveryMode, PromptQueue, PromptSource, QueuedPrompt};
#[cfg(test)]
use crate::types::{ContentBlock, Message, Usage};
#[cfg(test)]
use crate::{AgentEvent, ToolCallOutcome};
#[cfg(test)]
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEventKind};
#[cfg(test)]
use ratatui::style::Modifier;
#[cfg(test)]
use ratatui::text::{Span, Text};
#[cfg(test)]
use ratatui::Terminal;

pub(crate) fn sanitize_terminal_text(value: &str) -> String {
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

pub(crate) fn terminal_display_char(ch: char) -> char {
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
