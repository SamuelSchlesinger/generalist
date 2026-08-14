//! Shared foundations for the on-disk archive stores (episodic memory and
//! conversation history): private-directory hardening and the single
//! retention policy deciding what conversation content is ever persisted.

use crate::error::{Error, Result};
use crate::types::{ContentBlock, Message, MessageOrigin};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

/// One retained conversation event, before a store maps it to its own
/// serialized event type.
///
/// Tool inputs, tool-result content, provider reasoning, signatures, and
/// redacted-reasoning payloads are deliberately unrepresentable here: what
/// this enum cannot carry, no archive can leak.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RetainedEvent {
    UserText { text: String },
    AssistantText { text: String },
    ToolCall { name: String },
    ToolResult { is_error: bool },
}

/// The one retention policy shared by every archive store.
///
/// Retained: user text authored in conversation, assistant text, tool
/// names, and tool-result error flags. Dropped: host-control text (any
/// non-`Conversation` origin, even when its exact text was forged),
/// reasoning of both kinds, tool inputs, and tool outputs.
pub(crate) fn retained_events(history: &[Message]) -> Vec<RetainedEvent> {
    let mut events = Vec::new();
    for message in history {
        for block in &message.content {
            match block {
                ContentBlock::Text { text }
                    if !text.is_empty()
                        && message.role == "user"
                        && message.origin == MessageOrigin::Conversation =>
                {
                    events.push(RetainedEvent::UserText { text: text.clone() });
                }
                ContentBlock::Text { text } if !text.is_empty() && message.role == "assistant" => {
                    events.push(RetainedEvent::AssistantText { text: text.clone() });
                }
                ContentBlock::ToolUse { name, .. } => {
                    events.push(RetainedEvent::ToolCall { name: name.clone() });
                }
                ContentBlock::ToolResult { is_error, .. } => {
                    events.push(RetainedEvent::ToolResult {
                        is_error: is_error.unwrap_or(false),
                    });
                }
                ContentBlock::Thinking { .. }
                | ContentBlock::RedactedThinking { .. }
                | ContentBlock::Text { .. } => {}
            }
        }
    }
    events
}

/// Whitespace-normalize `text` and keep at most `limit` characters, marking
/// elision with an ellipsis.
pub(crate) fn normalized_preview(text: &str, limit: usize) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut characters = normalized.chars();
    let prefix: String = characters.by_ref().take(limit).collect();
    if characters.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}

/// Preview the first line containing `needle` (already lowercased), falling
/// back to the whole text.
pub(crate) fn matching_preview(text: &str, needle: &str, limit: usize) -> String {
    let source = text
        .lines()
        .find(|line| line.to_lowercase().contains(needle))
        .unwrap_or(text);
    normalized_preview(source, limit)
}

/// Refuse to operate through a symlinked path.
pub(crate) fn reject_symlink(path: &Path, description: &str) -> Result<()> {
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

/// Create `path` as a private (0700) directory, refusing symlinks before
/// and after creation.
pub(crate) fn ensure_private_directory(path: &Path, description: &str) -> Result<()> {
    reject_symlink(path, description)?;
    fs::create_dir_all(path).map_err(|error| {
        Error::Other(format!(
            "Failed to create {description} {}: {error}",
            path.display()
        ))
    })?;
    // Re-check before chmod: set_permissions follows symlinks, so applying it
    // before this check could modify an attacker-substituted target.
    reject_symlink(path, description)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|error| {
        Error::Other(format!(
            "Failed to restrict {description} {}: {error}",
            path.display()
        ))
    })?;
    reject_symlink(path, description)?;
    if !path.is_dir() {
        return Err(Error::Other(format!(
            "{description} {} is not a directory",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn retention_policy_drops_sensitive_payloads_and_host_text() {
        let mut goal_shaped = Message::user_text("do the next step");
        goal_shaped.origin = MessageOrigin::GoalContinuation;
        let history = vec![
            Message::user_text("real question"),
            goal_shaped,
            Message::assistant(vec![
                ContentBlock::Thinking {
                    thinking: "private reasoning".into(),
                    signature: "sig".into(),
                },
                ContentBlock::RedactedThinking {
                    data: "opaque".into(),
                },
                ContentBlock::ToolUse {
                    name: "bash".into(),
                    input: json!({"secret": "input"}),
                    id: "id-1".into(),
                },
            ]),
            Message::user(vec![ContentBlock::ToolResult {
                content: "secret output".into(),
                tool_use_id: "id-1".into(),
                is_error: Some(true),
            }]),
            Message::assistant(vec![ContentBlock::Text {
                text: "the answer".into(),
            }]),
        ];

        let events = retained_events(&history);
        assert_eq!(
            events,
            vec![
                RetainedEvent::UserText {
                    text: "real question".into()
                },
                RetainedEvent::ToolCall {
                    name: "bash".into()
                },
                RetainedEvent::ToolResult { is_error: true },
                RetainedEvent::AssistantText {
                    text: "the answer".into()
                },
            ]
        );
        // Nothing retained mentions the dropped payloads.
        let joined = format!("{events:?}");
        assert!(!joined.contains("secret"));
        assert!(!joined.contains("private reasoning"));
        assert!(!joined.contains("do the next step"));
    }

    #[test]
    fn previews_normalize_and_bound() {
        assert_eq!(normalized_preview("a  b\n\nc", 10), "a b c");
        assert_eq!(normalized_preview("abcdef", 3), "abc…");
        let text = "first line\nneedle line here\nlast";
        assert_eq!(matching_preview(text, "needle", 40), "needle line here");
        assert_eq!(matching_preview(text, "absent", 5), "first…");
    }
}
