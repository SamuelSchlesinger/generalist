//! A small, self-contained Python lexer used to highlight tool source
//! previews in the activity panel and permission modal.

use ratatui::style::Style;
use ratatui::text::Span;

use super::render::{CYAN, GREEN, MUTED, PURPLE, TEXT, YELLOW};
use super::sanitize_terminal_text;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PythonStringState {
    TripleSingle,
    TripleDouble,
}

pub(crate) fn highlight_python_source(source: &str) -> Vec<Vec<Span<'static>>> {
    let source = sanitize_terminal_text(source);
    let mut string_state = None;
    source
        .split('\n')
        .map(|line| highlight_python_line(line, &mut string_state))
        .collect()
}

fn highlight_python_line(
    line: &str,
    string_state: &mut Option<PythonStringState>,
) -> Vec<Span<'static>> {
    let chars = line.chars().collect::<Vec<_>>();
    let mut spans = Vec::new();
    let mut index = 0;
    let mut expects_definition_name = false;

    if let Some(state) = *string_state {
        let quote = match state {
            PythonStringState::TripleSingle => '\'',
            PythonStringState::TripleDouble => '"',
        };
        if let Some(end) = find_triple_quote_end(&chars, 0, quote) {
            push_python_span(&mut spans, &chars[0..end], python_string_style());
            index = end;
            *string_state = None;
        } else {
            push_python_span(&mut spans, &chars, python_string_style());
            return spans;
        }
    }

    while index < chars.len() {
        let ch = chars[index];
        if ch.is_whitespace() {
            let end = take_while(&chars, index, char::is_whitespace);
            push_python_span(&mut spans, &chars[index..end], python_plain_style());
            index = end;
            continue;
        }
        if ch == '#' {
            push_python_span(&mut spans, &chars[index..], python_comment_style());
            break;
        }
        if ch == '@' && chars[..index].iter().all(|ch| ch.is_whitespace()) {
            let mut end = index + 1;
            while end < chars.len()
                && (is_python_identifier_continue(chars[end]) || chars[end] == '.')
            {
                end += 1;
            }
            push_python_span(&mut spans, &chars[index..end], python_decorator_style());
            index = end;
            continue;
        }
        if ch == '\'' || ch == '"' {
            let (end, state) = scan_python_string(&chars, index);
            push_python_span(&mut spans, &chars[index..end], python_string_style());
            *string_state = state;
            index = end;
            continue;
        }
        if is_python_identifier_start(ch) {
            let end = take_while(&chars, index + 1, is_python_identifier_continue);
            let token = chars[index..end].iter().collect::<String>();
            if is_python_string_prefix(&token)
                && end < chars.len()
                && matches!(chars[end], '\'' | '"')
            {
                let (string_end, state) = scan_python_string(&chars, end);
                push_python_span(&mut spans, &chars[index..string_end], python_string_style());
                *string_state = state;
                index = string_end;
                continue;
            }

            let style = if expects_definition_name {
                expects_definition_name = false;
                python_definition_style()
            } else if is_python_keyword(&token) {
                if matches!(token.as_str(), "def" | "class") {
                    expects_definition_name = true;
                }
                python_keyword_style()
            } else if is_python_builtin(&token) || next_non_space_is_call(&chars, end) {
                python_builtin_style()
            } else {
                python_plain_style()
            };
            push_python_span(&mut spans, &chars[index..end], style);
            index = end;
            continue;
        }
        if ch.is_ascii_digit()
            || (ch == '.'
                && chars
                    .get(index + 1)
                    .is_some_and(|next| next.is_ascii_digit()))
        {
            let end = scan_python_number(&chars, index);
            push_python_span(&mut spans, &chars[index..end], python_number_style());
            index = end;
            continue;
        }
        if is_python_operator(ch) {
            let end = take_while(&chars, index + 1, is_python_operator);
            push_python_span(&mut spans, &chars[index..end], python_operator_style());
            index = end;
            continue;
        }

        push_python_span(&mut spans, std::slice::from_ref(&ch), python_plain_style());
        index += 1;
    }
    spans
}

fn take_while(chars: &[char], mut index: usize, predicate: fn(char) -> bool) -> usize {
    while index < chars.len() && predicate(chars[index]) {
        index += 1;
    }
    index
}

fn scan_python_string(chars: &[char], quote_index: usize) -> (usize, Option<PythonStringState>) {
    let quote = chars[quote_index];
    let triple =
        chars.get(quote_index + 1) == Some(&quote) && chars.get(quote_index + 2) == Some(&quote);
    if triple {
        if let Some(end) = find_triple_quote_end(chars, quote_index + 3, quote) {
            return (end, None);
        }
        let state = if quote == '\'' {
            PythonStringState::TripleSingle
        } else {
            PythonStringState::TripleDouble
        };
        return (chars.len(), Some(state));
    }

    let mut index = quote_index + 1;
    while index < chars.len() {
        if chars[index] == quote && !python_quote_is_escaped(chars, index) {
            return (index + 1, None);
        }
        index += 1;
    }
    (chars.len(), None)
}

fn find_triple_quote_end(chars: &[char], mut index: usize, quote: char) -> Option<usize> {
    while index + 2 < chars.len() {
        if chars[index] == quote
            && chars[index + 1] == quote
            && chars[index + 2] == quote
            && !python_quote_is_escaped(chars, index)
        {
            return Some(index + 3);
        }
        index += 1;
    }
    None
}

fn python_quote_is_escaped(chars: &[char], index: usize) -> bool {
    let mut backslashes = 0;
    let mut cursor = index;
    while cursor > 0 && chars[cursor - 1] == '\\' {
        backslashes += 1;
        cursor -= 1;
    }
    backslashes % 2 == 1
}

fn scan_python_number(chars: &[char], mut index: usize) -> usize {
    index += 1;
    while index < chars.len() {
        let ch = chars[index];
        if ch.is_ascii_alphanumeric()
            || matches!(ch, '_' | '.')
            || (matches!(ch, '+' | '-') && index > 0 && matches!(chars[index - 1], 'e' | 'E'))
        {
            index += 1;
        } else {
            break;
        }
    }
    index
}

fn next_non_space_is_call(chars: &[char], mut index: usize) -> bool {
    while chars.get(index).is_some_and(|ch| ch.is_whitespace()) {
        index += 1;
    }
    chars.get(index) == Some(&'(')
}

fn is_python_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_alphabetic()
}

fn is_python_identifier_continue(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

fn is_python_string_prefix(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "r" | "u" | "b" | "f" | "br" | "rb" | "fr" | "rf"
    )
}

fn is_python_keyword(token: &str) -> bool {
    matches!(
        token,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "case"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "match"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "type"
            | "while"
            | "with"
            | "yield"
    )
}

fn is_python_builtin(token: &str) -> bool {
    matches!(
        token,
        "bool"
            | "bytes"
            | "dict"
            | "enumerate"
            | "filter"
            | "float"
            | "int"
            | "len"
            | "list"
            | "map"
            | "open"
            | "print"
            | "range"
            | "set"
            | "str"
            | "super"
            | "tools"
            | "tuple"
            | "zip"
    )
}

fn is_python_operator(ch: char) -> bool {
    matches!(
        ch,
        '+' | '-' | '*' | '/' | '%' | '@' | '&' | '|' | '^' | '~' | '<' | '>' | '=' | '!' | ':'
    )
}

fn push_python_span(spans: &mut Vec<Span<'static>>, chars: &[char], style: Style) {
    if !chars.is_empty() {
        spans.push(Span::styled(chars.iter().collect::<String>(), style));
    }
}

pub(crate) fn python_plain_style() -> Style {
    Style::default().fg(TEXT)
}

fn python_keyword_style() -> Style {
    Style::default().fg(PURPLE).bold()
}

fn python_definition_style() -> Style {
    Style::default().fg(CYAN).bold()
}

fn python_builtin_style() -> Style {
    Style::default().fg(CYAN)
}

fn python_string_style() -> Style {
    Style::default().fg(GREEN)
}

fn python_comment_style() -> Style {
    Style::default().fg(MUTED).italic()
}

fn python_number_style() -> Style {
    Style::default().fg(CYAN)
}

fn python_decorator_style() -> Style {
    Style::default().fg(YELLOW).bold()
}

fn python_operator_style() -> Style {
    Style::default().fg(YELLOW)
}
