//! The pure render layer: everything that turns [`AppState`] into Ratatui
//! widgets, plus the palette shared across the TUI.

use crate::command::{complete_local_command, CommandCompletion, COMMAND_SPECS};
use crate::permissions::ToolExecutionRequest;
use crate::runtime::{DeliveryMode, PromptSource, QueuedPrompt};
use crate::types::truncate_middle;
use ratatui::layout::{Alignment, Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use ratatui::Frame;
use serde_json::Value;

use super::editor::visible_editor;
use super::python_highlight::{highlight_python_source, python_plain_style};
use super::sanitize_terminal_text;
use super::state::{
    clamp_scroll, compact_number, source_tool_input, source_tool_spec, ActivityStatus, AppState,
    ChatEntry, ChatKind, ChatSearch, Modal, QueueEditor, ReasoningEntry, SourceToolInput,
};

pub(crate) const BG: Color = Color::Rgb(12, 16, 24);
pub(crate) const PANEL: Color = Color::Rgb(20, 26, 38);
pub(crate) const PANEL_ALT: Color = Color::Rgb(26, 34, 48);
pub(crate) const BORDER: Color = Color::Rgb(59, 75, 99);
pub(crate) const TEXT: Color = Color::Rgb(220, 226, 235);
pub(crate) const MUTED: Color = Color::Rgb(124, 139, 161);
pub(crate) const CYAN: Color = Color::Rgb(92, 207, 230);
pub(crate) const GREEN: Color = Color::Rgb(111, 214, 151);
pub(crate) const YELLOW: Color = Color::Rgb(241, 196, 96);
pub(crate) const RED: Color = Color::Rgb(244, 112, 122);
pub(crate) const PURPLE: Color = Color::Rgb(183, 148, 244);

pub(crate) fn render(frame: &mut Frame<'_>, app: &mut AppState) {
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
        render_modal(frame, modal, &app.chat, &app.queue, &app.reasoning);
    }
}

fn render_header(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
    let block = Block::new()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(PANEL));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [top, goal_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(inner);
    let [identity, state] =
        Layout::horizontal([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(top);
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
    let goal = app.goal.as_deref().map_or_else(
        || {
            Line::from(vec![
                Span::styled(" Goal ", Style::default().fg(MUTED).bold()),
                Span::styled("none · use /goal to set one", Style::default().fg(MUTED)),
            ])
        },
        |goal| {
            Line::from(vec![
                Span::styled(" Goal ", Style::default().fg(BG).bg(GREEN).bold()),
                Span::raw("  "),
                Span::styled(goal.replace('\n', " ↵ "), Style::default().fg(TEXT)),
            ])
        },
    );
    frame.render_widget(
        Paragraph::new(goal).style(Style::default().bg(PANEL)),
        goal_area,
    );
}

pub(crate) fn render_chat(frame: &mut Frame<'_>, app: &mut AppState, area: Rect) {
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
    if let Some(entry) = app.pending_chat_jump.take() {
        app.chat_scroll = chat_entry_start(&app.chat, entry, width).min(max_scroll);
        app.follow_latest = app.chat_scroll == max_scroll;
    }
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
        // ScrollbarState treats `content_length - 1` as the greatest legal
        // position. Our position is a bounded top-row offset whose greatest
        // value is `max_scroll`, not `total_lines - 1`. Passing total_lines
        // leaves track below the thumb at the true bottom and falsely suggests
        // that more conversation is hidden.
        let mut scrollbar_state = ScrollbarState::new(max_scroll.saturating_add(1))
            .position(scroll)
            .viewport_content_length(visible);
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

pub(crate) fn render_activity(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
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
            let (label, color) = match (prompt.source, prompt.delivery) {
                (PromptSource::GoalContinuation, _) => ("◎ goal", GREEN),
                (_, DeliveryMode::Steer) => ("↪ steer", PURPLE),
                (_, DeliveryMode::FollowUp) => ("＋ next", CYAN),
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
        if source_tool_spec(&item.name).is_some() {
            let source_lines = if item.name == "python" {
                item.input
                    .split('\n')
                    .zip(highlight_python_source(&item.input))
                    .filter(|(line, _)| !line.trim().is_empty())
                    .map(|(_, spans)| spans)
                    .collect::<Vec<_>>()
            } else {
                item.input
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| vec![Span::styled(line.to_string(), python_plain_style())])
                    .collect::<Vec<_>>()
            };
            for (index, source_spans) in source_lines.iter().take(2).enumerate() {
                let prefix = if item.name == "bash" && index == 0 {
                    "  $ "
                } else {
                    "  │ "
                };
                let mut highlighted = vec![Span::styled(prefix, Style::default().fg(MUTED))];
                highlighted.extend(source_spans.iter().cloned());
                lines.push(Line::from(highlighted));
            }
            if source_lines.len() > 2 {
                lines.push(Line::from(Span::styled(
                    "  │ …",
                    Style::default().fg(MUTED),
                )));
            }
            if let Some(first) = item.output.lines().find(|line| !line.trim().is_empty()) {
                lines.push(Line::from(Span::styled(
                    format!("  ↳ {}", truncate_middle(first, 116)),
                    Style::default().fg(MUTED),
                )));
            }
        } else {
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
        }
        ListItem::new(lines).style(Style::default().bg(PANEL))
    }));
    frame.render_widget(List::new(items).style(Style::default().bg(PANEL)), inner);
}

fn render_input(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
    let command_mode = app.input.trim_start().starts_with('/');
    let (label, hint, accent) = if command_mode {
        (
            " Command ",
            if app.busy {
                "Tab complete · Enter queue until idle"
            } else {
                "Tab complete · Enter run · /help lists commands"
            },
            PURPLE,
        )
    } else if app.turn_active {
        (
            " Message ",
            "Enter steer · Tab/Alt+Enter follow-up · Ctrl+J newline",
            CYAN,
        )
    } else if app.busy {
        (
            " Message ",
            "Enter queue · background work continues · Ctrl+J newline",
            CYAN,
        )
    } else {
        (" Message ", "Enter send · Ctrl+J newline", CYAN)
    };
    let block = Block::new()
        .title(Line::from(vec![
            Span::styled(label, Style::default().fg(accent).bold()),
            Span::styled(hint, Style::default().fg(MUTED)),
        ]))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
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
    if app.modal.is_none() && !app.copy_mode {
        frame.set_cursor_position(Position::new(
            inner.x + cursor_x.min(inner.width.saturating_sub(1)),
            inner.y,
        ));
    }
}

fn render_footer(frame: &mut Frame<'_>, app: &AppState, area: Rect) {
    let command_hint;
    let left = if app.copy_mode {
        " F3/Esc resume · select text · use your terminal's copy shortcut"
    } else if app.input.trim_start().starts_with('/') {
        command_hint = command_footer_hint(&app.input);
        command_hint.as_str()
    } else if app.busy {
        " F1 help  F2 queue  F3 copy  F4 reasoning  Ctrl+F find  Esc/Ctrl+C interrupt"
    } else {
        " F1 help  F2 queue  F3 copy  F4 reasoning  Ctrl+F find  Ctrl+C quit"
    };
    let right = if app.copy_mode {
        "display paused "
    } else if app.follow_latest {
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

pub(crate) fn command_footer_hint(input: &str) -> String {
    match complete_local_command(input) {
        Some(CommandCompletion::Replace(replacement)) => {
            format!(" Tab → {}", replacement.trim())
        }
        Some(CommandCompletion::Candidates(candidates)) => {
            format!(" {}", candidates.join("  "))
        }
        Some(CommandCompletion::Complete) => {
            " command prefix complete · add arguments if needed or press Enter".to_string()
        }
        None => format!(
            " {}",
            COMMAND_SPECS
                .iter()
                .map(|command| command.name)
                .collect::<Vec<_>>()
                .join("  ")
        ),
    }
}

fn render_modal(
    frame: &mut Frame<'_>,
    modal: &mut Modal,
    chat: &[ChatEntry],
    queue: &[QueuedPrompt],
    reasoning: &[ReasoningEntry],
) {
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
        Modal::Search(search) => render_chat_search(frame, search, chat),
        Modal::Reasoning {
            scroll,
            max_scroll,
            follow_latest,
        } => render_reasoning(frame, reasoning, scroll, max_scroll, follow_latest),
    }
}

fn render_chat_search(frame: &mut Frame<'_>, search: &ChatSearch, entries: &[ChatEntry]) {
    let area = centered(frame.area(), 86, 78, 58, 12);
    frame.render_widget(Clear, area);
    let match_label = if search.matches.len() == 1 {
        "1 match".to_string()
    } else {
        format!("{} matches", search.matches.len())
    };
    let block = modal_block(format!(" Find in conversation · {match_label} "));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [query_area, list_area, hint_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(2),
    ])
    .areas(inner);

    let query_block = Block::new()
        .title(" Query ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(PURPLE))
        .style(Style::default().bg(PANEL_ALT));
    let query_inner = query_block.inner(query_area);
    frame.render_widget(query_block, query_area);
    let (shown, cursor_x) =
        visible_editor(&search.query, search.cursor, query_inner.width as usize);
    let shown = if shown.is_empty() {
        "Type to search visible conversation entries…".to_string()
    } else {
        shown
    };
    frame.render_widget(
        Paragraph::new(shown).style(
            Style::default()
                .fg(if search.query.is_empty() { MUTED } else { TEXT })
                .bg(PANEL_ALT),
        ),
        query_inner,
    );

    if search.query.trim().is_empty() {
        frame.render_widget(
            Paragraph::new(format!("{} conversation entries", entries.len()))
                .alignment(Alignment::Center)
                .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
            list_area,
        );
    } else if search.matches.is_empty() {
        frame.render_widget(
            Paragraph::new("No matching conversation entries")
                .alignment(Alignment::Center)
                .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
            list_area,
        );
    } else {
        let visible = (list_area.height as usize).max(1);
        let start = search
            .selected
            .saturating_add(1)
            .saturating_sub(visible)
            .min(search.matches.len().saturating_sub(visible));
        let items = search
            .matches
            .iter()
            .enumerate()
            .skip(start)
            .take(visible)
            .filter_map(|(match_index, entry_index)| {
                let entry = entries.get(*entry_index)?;
                let active = match_index == search.selected;
                let (label, color) = chat_kind_label(entry.kind);
                Some(
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            if active { "› " } else { "  " },
                            Style::default().fg(CYAN).bold(),
                        ),
                        Span::styled(
                            format!("{}  {label:<9} ", entry.timestamp),
                            Style::default().fg(color).bold(),
                        ),
                        Span::styled(
                            chat_search_preview(entry, &search.query),
                            Style::default().fg(if active { TEXT } else { MUTED }),
                        ),
                    ]))
                    .style(Style::default().bg(if active {
                        PANEL
                    } else {
                        PANEL_ALT
                    })),
                )
            })
            .collect::<Vec<_>>();
        frame.render_widget(List::new(items), list_area);
    }

    let selection = search.matches.get(search.selected).map_or_else(
        || "no selection".to_string(),
        |_| format!("{} / {}", search.selected + 1, search.matches.len()),
    );
    frame.render_widget(
        Paragraph::new(format!(
            "↑/↓ or Tab choose · Enter jump · Esc/Ctrl+F close · {selection}"
        ))
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
        hint_area,
    );
    frame.set_cursor_position(Position::new(
        query_inner.x + cursor_x.min(query_inner.width.saturating_sub(1)),
        query_inner.y,
    ));
}

fn chat_search_preview(entry: &ChatEntry, query: &str) -> String {
    let needle = query.trim().to_lowercase();
    let line = entry
        .body
        .lines()
        .find(|line| line.to_lowercase().contains(&needle))
        .or_else(|| entry.body.lines().find(|line| !line.trim().is_empty()))
        .unwrap_or_default();
    truncate_middle(line.trim(), 140)
}

pub(crate) fn render_help(frame: &mut Frame<'_>) {
    let area = centered(frame.area(), 94, 84, 70, 20);
    frame.render_widget(Clear, area);
    let block = modal_block(" Help & shortcuts ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [content_area, hint_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
    let [commands_area, shortcuts_area] =
        Layout::horizontal([Constraint::Percentage(52), Constraint::Percentage(48)])
            .spacing(1)
            .areas(content_area);

    let mut command_lines = vec![Line::from(Span::styled(
        "Slash commands",
        Style::default().fg(CYAN).bold(),
    ))];
    command_lines.extend(COMMAND_SPECS.iter().map(|command| {
        Line::from(vec![
            Span::styled(
                format!("  {:<14}", command.name),
                Style::default().fg(PURPLE),
            ),
            Span::styled(command.description, Style::default().fg(TEXT)),
        ])
    }));
    frame.render_widget(
        Paragraph::new(command_lines)
            .wrap(Wrap { trim: false })
            .style(Style::default().bg(PANEL_ALT)),
        commands_area,
    );

    let shortcut_lines = vec![
        Line::from(Span::styled("While idle", Style::default().fg(CYAN).bold())),
        Line::from("  Enter send · Ctrl+J newline · ↑/↓ input history"),
        Line::from(""),
        Line::from(Span::styled(
            "While running",
            Style::default().fg(CYAN).bold(),
        )),
        Line::from("  Enter steer at next safe boundary"),
        Line::from("  Tab completes slash prefixes; otherwise queues follow-up"),
        Line::from("  Alt+Enter always queues a separate follow-up"),
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
        Line::from("  Ctrl+F find conversation text and jump to a match"),
        Line::from("  F3 native selection · /copy last|all|select · F4 reasoning"),
    ];
    frame.render_widget(
        Paragraph::new(shortcut_lines)
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(TEXT).bg(PANEL_ALT)),
        shortcuts_area,
    );
    frame.render_widget(
        Paragraph::new("Esc, Enter, or F1 closes this window")
            .alignment(Alignment::Center)
            .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
        hint_area,
    );
}

fn render_reasoning(
    frame: &mut Frame<'_>,
    entries: &[ReasoningEntry],
    scroll: &mut usize,
    max_scroll: &mut usize,
    follow_latest: &mut bool,
) {
    let area = centered(frame.area(), 92, 88, 70, 20);
    frame.render_widget(Clear, area);
    let block = modal_block(" Model reasoning · provider supplied ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [content_area, hint_area] =
        Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);

    let paragraph = Paragraph::new(reasoning_text(entries))
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(TEXT).bg(PANEL_ALT));
    let width = content_area.width.max(1);
    let visible = content_area.height as usize;
    let total_lines = paragraph.line_count(width);
    *max_scroll = total_lines.saturating_sub(visible);
    clamp_scroll(scroll, follow_latest, *max_scroll);
    frame.render_widget(
        paragraph.scroll(((*scroll).min(u16::MAX as usize) as u16, 0)),
        content_area,
    );

    if *max_scroll > 0 && content_area.width > 2 {
        let mut scrollbar_state = ScrollbarState::new(max_scroll.saturating_add(1))
            .position(*scroll)
            .viewport_content_length(visible);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .thumb_symbol("▐")
                .track_symbol(Some("│")),
            content_area,
            &mut scrollbar_state,
        );
    }

    let status = if *follow_latest {
        "following latest"
    } else {
        "scroll paused"
    };
    frame.render_widget(
        Paragraph::new(format!(
            "F4/Esc close · ↑/↓/PgUp/PgDn scroll · F3 copy · {status}"
        ))
        .alignment(Alignment::Center)
        .style(Style::default().fg(MUTED).bg(PANEL_ALT)),
        hint_area,
    );
}

pub(crate) fn render_queue(
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
                let (label, color) = match (prompt.source, prompt.delivery) {
                    (PromptSource::GoalContinuation, _) => ("GOAL ", GREEN),
                    (_, DeliveryMode::Steer) => ("STEER", PURPLE),
                    (_, DeliveryMode::FollowUp) => ("NEXT ", CYAN),
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
                    .title(format!(" {} ", permission_detail_title(request))),
            )
            .style(Style::default().fg(TEXT).bg(PANEL)),
        detail_area,
    );
    let choices = [
        if crate::permissions::remembers_exact_input(&request.tool_name) {
            "[a] Always allow this exact input (this session)"
        } else {
            "[a] Always allow this tool"
        },
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

pub(crate) fn chat_kind_label(kind: ChatKind) -> (&'static str, Color) {
    match kind {
        ChatKind::User => ("YOU", GREEN),
        ChatKind::Assistant => ("ASSISTANT", CYAN),
        ChatKind::Info => ("INFO", YELLOW),
        ChatKind::Error => ("ERROR", RED),
    }
}

fn chat_lines(entries: &[ChatEntry]) -> Vec<Line<'static>> {
    let mut lines = Vec::<Line<'static>>::new();
    for entry in entries {
        let (label, color) = chat_kind_label(entry.kind);
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
    lines
}

pub(crate) fn chat_text(entries: &[ChatEntry]) -> Text<'static> {
    let mut lines = chat_lines(entries);
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "Start a conversation below. Tool calls will appear in the activity panel.",
            Style::default().fg(MUTED),
        )));
    }
    Text::from(lines)
}

pub(crate) fn chat_entry_start(entries: &[ChatEntry], entry: usize, width: u16) -> usize {
    let entry = entry.min(entries.len());
    if entry == 0 {
        return 0;
    }
    Paragraph::new(Text::from(chat_lines(&entries[..entry])))
        .wrap(Wrap { trim: false })
        .line_count(width.max(1))
}

fn reasoning_text(entries: &[ReasoningEntry]) -> Text<'static> {
    if entries.is_empty() {
        return Text::from(vec![
            Line::from(Span::styled(
                "No model requests have exposed inspectable reasoning in this session.",
                Style::default().fg(MUTED),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "This view shows only reasoning fields supplied by the configured provider.",
                Style::default().fg(MUTED),
            )),
        ]);
    }

    let mut lines = Vec::<Line<'static>>::new();
    for (index, entry) in entries.iter().enumerate() {
        let (status, color) = if entry.finished {
            ("complete", GREEN)
        } else {
            ("live", YELLOW)
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("REQUEST {}  ", index + 1),
                Style::default().fg(PURPLE).bold(),
            ),
            Span::styled(entry.timestamp.clone(), Style::default().fg(MUTED)),
            Span::styled(format!("  {status}"), Style::default().fg(color).bold()),
        ]));

        if entry.body.is_empty() {
            let message = if entry.finished {
                "  Provider supplied no inspectable reasoning for this request."
            } else {
                "  Waiting for provider-supplied reasoning…"
            };
            lines.push(Line::from(Span::styled(
                message,
                Style::default().fg(MUTED),
            )));
        } else {
            lines.extend(entry.body.lines().map(|line| {
                Line::from(Span::styled(format!("  {line}"), Style::default().fg(TEXT)))
            }));
        }

        if let Some(reason) = &entry.abort_reason {
            lines.push(Line::from(Span::styled(
                format!("  Stream ended before commit: {reason}"),
                Style::default().fg(RED),
            )));
        }
        lines.push(Line::from(""));
    }
    Text::from(lines)
}

pub(crate) fn permission_detail(request: &ToolExecutionRequest) -> Text<'static> {
    if let Some(source) = source_tool_input(&request.tool_name, &request.input) {
        return source_permission_detail(source, &request.input);
    }
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

pub(crate) fn permission_detail_title(request: &ToolExecutionRequest) -> &'static str {
    source_tool_input(&request.tool_name, &request.input)
        .map(|source| source.title)
        .unwrap_or(if request.tool_name == "patch_file" {
            "Proposed changes"
        } else {
            "Input"
        })
}

fn source_permission_detail(source: SourceToolInput<'_>, input: &Value) -> Text<'static> {
    let source_text = sanitize_terminal_text(source.source);
    let source_lines = if source.field == "code" {
        highlight_python_source(&source_text)
    } else {
        source_text
            .split('\n')
            .map(|line| vec![Span::styled(line.to_string(), python_plain_style())])
            .collect::<Vec<_>>()
    };
    let number_width = source_lines.len().max(1).to_string().len();
    let mut lines = Vec::new();
    if source.source.is_empty() {
        lines.push(Line::from(Span::styled(
            format!("(empty {})", source.field),
            Style::default().fg(RED),
        )));
    } else {
        for (index, source_spans) in source_lines.into_iter().enumerate() {
            let mut highlighted = vec![Span::styled(
                format!("{:>number_width$} │ ", index + 1),
                Style::default().fg(MUTED),
            )];
            highlighted.extend(source_spans);
            lines.push(Line::from(highlighted));
        }
    }

    let option_lines = source_option_lines(input, source.field);
    if !option_lines.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Options",
            Style::default().fg(PURPLE).bold(),
        )));
        lines.extend(option_lines.into_iter().map(|line| {
            Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(MUTED),
            ))
        }));
    }
    Text::from(lines)
}

fn source_option_lines(input: &Value, source_field: &str) -> Vec<String> {
    let Some(input) = input.as_object() else {
        return Vec::new();
    };
    let mut lines = Vec::new();
    for (key, value) in input.iter().filter(|(key, _)| key.as_str() != source_field) {
        let key = sanitize_terminal_text(key);
        let rendered = match value {
            Value::String(value) => sanitize_terminal_text(value),
            _ => sanitize_terminal_text(
                &serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()),
            ),
        };
        let mut rendered_lines = rendered.split('\n');
        let first = rendered_lines.next().unwrap_or_default();
        lines.push(format!("{key}: {first}"));
        lines.extend(rendered_lines.map(|line| format!("  {line}")));
    }
    lines
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
