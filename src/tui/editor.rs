//! Single-line editor primitives shared by every editable text field in the
//! TUI: the composer, the prompt modal, the chat-search query, and the queue
//! editor.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use unicode_width::UnicodeWidthChar;

use super::terminal_display_char;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptEditorAction {
    Continue,
    Submit,
    Cancel,
}

pub(crate) fn visible_editor(value: &str, cursor: usize, width: usize) -> (String, u16) {
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

pub(crate) fn insert_text(value: &mut String, cursor: &mut usize, text: &str) {
    let text = text.chars().filter(|ch| *ch != '\r').collect::<String>();
    if text.is_empty() {
        return;
    }
    let inserted_chars = text.chars().count();
    let byte = byte_index(value, *cursor);
    value.insert_str(byte, &text);
    *cursor += inserted_chars;
}

/// Apply one key event to a `(text, cursor)` pair using the editing bindings
/// shared by every line editor in the TUI: Ctrl+A/E/U/K/W plus plain
/// Char/Backspace/Delete/Left/Right/Home/End.
///
/// Returns whether the key was consumed. Caller-specific bindings (submit,
/// cancel, history, search navigation, newline insertion, ...) must be
/// handled by the caller before or instead of routing a key here.
pub(crate) fn apply_editor_key(value: &mut String, cursor: &mut usize, key: &KeyEvent) -> bool {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('a') => *cursor = 0,
            KeyCode::Char('e') => *cursor = value.chars().count(),
            KeyCode::Char('u') => {
                value.clear();
                *cursor = 0;
            }
            KeyCode::Char('k') => truncate_at_char(value, *cursor),
            KeyCode::Char('w') => delete_previous_word(value, cursor),
            _ => return false,
        }
        return true;
    }

    match key.code {
        KeyCode::Char(ch) => insert_char(value, cursor, ch),
        KeyCode::Backspace => backspace(value, cursor),
        KeyCode::Delete => delete_at_cursor(value, *cursor),
        KeyCode::Left => *cursor = cursor.saturating_sub(1),
        KeyCode::Right => *cursor = (*cursor + 1).min(value.chars().count()),
        KeyCode::Home => *cursor = 0,
        KeyCode::End => *cursor = value.chars().count(),
        _ => return false,
    }
    true
}

pub(crate) fn handle_prompt_editor_key(
    value: &mut String,
    cursor: &mut usize,
    key: KeyEvent,
) -> PromptEditorAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        if key.code == KeyCode::Char('c') {
            return PromptEditorAction::Cancel;
        }
        apply_editor_key(value, cursor, &key);
        return PromptEditorAction::Continue;
    }

    match key.code {
        KeyCode::Enter => PromptEditorAction::Submit,
        KeyCode::Esc => PromptEditorAction::Cancel,
        _ => {
            apply_editor_key(value, cursor, &key);
            PromptEditorAction::Continue
        }
    }
}

pub(crate) fn insert_char(value: &mut String, cursor: &mut usize, ch: char) {
    let byte = byte_index(value, *cursor);
    value.insert(byte, ch);
    *cursor += 1;
}

pub(crate) fn backspace(value: &mut String, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }
    let start = byte_index(value, *cursor - 1);
    let end = byte_index(value, *cursor);
    value.replace_range(start..end, "");
    *cursor -= 1;
}

pub(crate) fn delete_at_cursor(value: &mut String, cursor: usize) {
    if cursor >= value.chars().count() {
        return;
    }
    let start = byte_index(value, cursor);
    let end = byte_index(value, cursor + 1);
    value.replace_range(start..end, "");
}

pub(crate) fn delete_previous_word(value: &mut String, cursor: &mut usize) {
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

pub(crate) fn truncate_at_char(value: &mut String, cursor: usize) {
    value.truncate(byte_index(value, cursor));
}

fn byte_index(value: &str, character_index: usize) -> usize {
    value
        .char_indices()
        .map(|(index, _)| index)
        .nth(character_index)
        .unwrap_or(value.len())
}
