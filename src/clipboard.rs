//! Explicit terminal clipboard integration.
//!
//! Clipboard writes use OSC 52, so the terminal emulator remains the authority
//! for the user's local clipboard even when Generalist runs through SSH. The
//! application never reads ambient clipboard contents, and callers must invoke
//! this path from an explicit UI action.

use crate::types::Message;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::io::{self, Write};

/// Keep one clipboard request bounded even if conversation history is large.
///
/// Terminal OSC 52 limits vary. One MiB is a generous application-side safety
/// bound; a terminal may enforce a smaller policy and ignore the request, in
/// which case native selection remains available.
pub const MAX_OSC52_SOURCE_BYTES: usize = 1024 * 1024;

/// Return the most recent committed assistant text in conversation history.
///
/// Tool-only messages and provider reasoning are structurally absent because
/// [`Message::text`] exposes only ordinary text blocks.
pub fn latest_assistant_text(history: &[Message]) -> Option<String> {
    history
        .iter()
        .rev()
        .filter(|message| message.role == "assistant")
        .map(Message::text)
        .find(|text| !text.is_empty())
}

/// Return the most recent inspectable provider reasoning in committed history.
///
/// Only the human-readable `thinking` field is returned. Provider signatures
/// and redacted reasoning payloads are deliberately excluded.
pub fn latest_assistant_reasoning(history: &[Message]) -> Option<String> {
    history
        .iter()
        .rev()
        .filter(|message| message.role == "assistant")
        .map(|message| {
            message
                .content
                .iter()
                .filter_map(|block| match block {
                    crate::types::ContentBlock::Thinking { thinking, .. }
                        if !thinking.is_empty() =>
                    {
                        Some(thinking.as_str())
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .find(|reasoning| !reasoning.is_empty())
}

/// Render committed user/assistant text as a plain, model-visible transcript.
///
/// Host-authored goal continuations and tool-only messages are omitted. A
/// manually authored message with the same text as a goal continuation remains
/// ordinary conversation content because provenance, not text matching,
/// controls the omission.
pub fn conversation_transcript(history: &[Message]) -> Option<String> {
    let entries = history
        .iter()
        .filter(|message| !message.is_goal_continuation())
        .filter_map(|message| {
            let text = message.text();
            if text.is_empty() {
                return None;
            }
            let label = if message.role == "assistant" {
                "Assistant"
            } else {
                "User"
            };
            Some(format!("{label}:\n{text}"))
        })
        .collect::<Vec<_>>();
    (!entries.is_empty()).then(|| entries.join("\n\n"))
}

/// Ask the host terminal to set its clipboard using OSC 52.
///
/// The payload is base64 encoded before it enters the control sequence, so
/// untrusted conversation text cannot inject terminal commands or terminate
/// the OSC sequence early. Successful I/O means the request reached the
/// terminal; terminal clipboard policy may still reject it silently.
pub fn write_osc52(writer: &mut impl Write, text: &str) -> io::Result<usize> {
    let source_bytes = text.len();
    if source_bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot copy empty text",
        ));
    }
    if source_bytes > MAX_OSC52_SOURCE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "clipboard text is {source_bytes} bytes; OSC 52 requests are limited to {MAX_OSC52_SOURCE_BYTES} bytes"
            ),
        ));
    }

    let encoded = STANDARD.encode(text.as_bytes());
    let mut sequence = Vec::with_capacity(encoded.len() + 10);
    sequence.extend_from_slice(b"\x1b]52;c;");
    sequence.extend_from_slice(encoded.as_bytes());
    sequence.extend_from_slice(b"\x1b\\");
    writer.write_all(&sequence)?;
    writer.flush()?;
    Ok(source_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goal::GOAL_CONTINUATION_PROMPT;
    use crate::types::{ContentBlock, MessageOrigin};

    #[test]
    fn latest_assistant_text_ignores_tool_only_messages_and_reasoning() {
        let history = vec![
            Message::assistant(vec![ContentBlock::Text {
                text: "first answer".into(),
            }]),
            Message::assistant(vec![
                ContentBlock::Thinking {
                    thinking: "private reasoning".into(),
                    signature: "signature".into(),
                },
                ContentBlock::ToolUse {
                    name: "python".into(),
                    input: serde_json::json!({"code": "secret"}),
                    id: "tool-1".into(),
                },
            ]),
            Message::user(vec![ContentBlock::ToolResult {
                content: "secret output".into(),
                tool_use_id: "tool-1".into(),
                is_error: None,
            }]),
        ];

        assert_eq!(
            latest_assistant_text(&history).as_deref(),
            Some("first answer")
        );
    }

    #[test]
    fn latest_reasoning_excludes_signatures_and_redacted_payloads() {
        let history = vec![Message::assistant(vec![
            ContentBlock::Thinking {
                thinking: "inspectable first".into(),
                signature: "provider-signature-secret".into(),
            },
            ContentBlock::RedactedThinking {
                data: "opaque-provider-payload".into(),
            },
            ContentBlock::Thinking {
                thinking: "inspectable second".into(),
                signature: "another-signature".into(),
            },
        ])];

        let reasoning = latest_assistant_reasoning(&history).unwrap();
        assert_eq!(reasoning, "inspectable first\n\ninspectable second");
        assert!(!reasoning.contains("signature"));
        assert!(!reasoning.contains("opaque"));
    }

    #[test]
    fn transcript_uses_provenance_and_omits_non_text_payloads() {
        let mut manual_match = Message::user_text(GOAL_CONTINUATION_PROMPT);
        manual_match.origin = MessageOrigin::Conversation;
        let history = vec![
            Message::user_text("question"),
            Message::assistant(vec![
                ContentBlock::Thinking {
                    thinking: "reasoning".into(),
                    signature: String::new(),
                },
                ContentBlock::RedactedThinking {
                    data: "redacted secret".into(),
                },
                ContentBlock::Text {
                    text: "answer".into(),
                },
                ContentBlock::ToolUse {
                    name: "python".into(),
                    input: serde_json::json!({"code": "tool input secret"}),
                    id: "tool-1".into(),
                },
            ]),
            Message::user(vec![ContentBlock::ToolResult {
                content: "tool result secret".into(),
                tool_use_id: "tool-1".into(),
                is_error: None,
            }]),
            Message::goal_continuation(),
            manual_match,
        ];

        let transcript = conversation_transcript(&history).unwrap();
        assert_eq!(
            transcript,
            format!("User:\nquestion\n\nAssistant:\nanswer\n\nUser:\n{GOAL_CONTINUATION_PROMPT}")
        );
        assert!(!transcript.contains("reasoning"));
        assert!(!transcript.contains("secret"));
    }

    #[test]
    fn osc52_base64_encodes_untrusted_text_and_flushes_a_complete_sequence() {
        let mut output = Vec::new();
        let text = "line 1\n\x1b]52;c;injected\x07🦀";
        assert_eq!(write_osc52(&mut output, text).unwrap(), text.len());
        assert_eq!(
            output,
            format!("\x1b]52;c;{}\x1b\\", STANDARD.encode(text)).into_bytes()
        );
        assert_eq!(output.iter().filter(|byte| **byte == 0x1b).count(), 2);
    }

    #[test]
    fn osc52_rejects_empty_and_oversized_payloads_before_writing() {
        let mut output = Vec::new();
        assert_eq!(
            write_osc52(&mut output, "").unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            write_osc52(&mut output, &"x".repeat(MAX_OSC52_SOURCE_BYTES + 1))
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(output.is_empty());
    }
}
