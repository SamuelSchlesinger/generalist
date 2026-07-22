//! Provider-neutral conversation types.
//!
//! These types define the internal representation of a conversation. Each
//! [`crate::provider::Provider`] implementation translates them to and from
//! its own wire format, so nothing outside the provider modules depends on a
//! specific vendor API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single message in a conversation.
///
/// `role` is `"user"` or `"assistant"`. Tool results are carried in user
/// messages, matching the convention used by every major provider.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::Text { text: text.into() }])
    }

    /// All tool-use requests contained in this message.
    pub fn tool_uses(&self) -> Vec<ToolUse> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { name, input, id } => Some(ToolUse {
                    name: name.clone(),
                    input: input.clone(),
                    id: id.clone(),
                }),
                _ => None,
            })
            .collect()
    }

    /// Concatenated text content of this message.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// One block of message content.
///
/// The serialized form matches the Anthropic wire format (`type` tag with
/// snake_case variants); the OpenAI provider translates as needed. The
/// `Thinking`/`RedactedThinking` variants exist so reasoning blocks returned
/// by a model can be replayed unchanged on subsequent requests, which some
/// providers require.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        name: String,
        input: Value,
        id: String,
    },
    ToolResult {
        content: String,
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
}

/// A tool-use request extracted from a message.
#[derive(Debug, Clone)]
pub struct ToolUse {
    pub name: String,
    pub input: Value,
    pub id: String,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence,
    Refusal,
    Other(String),
}

impl StopReason {
    pub fn parse(s: &str) -> Self {
        match s {
            "end_turn" | "stop" => StopReason::EndTurn,
            "max_tokens" | "length" | "model_context_window_exceeded" => StopReason::MaxTokens,
            "tool_use" | "tool_calls" => StopReason::ToolUse,
            "stop_sequence" => StopReason::StopSequence,
            "refusal" | "content_filter" => StopReason::Refusal,
            other => StopReason::Other(other.to_string()),
        }
    }
}

/// Tool definition advertised to the model.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Token accounting for a single completion.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}

/// A request for one model completion.
#[derive(Debug)]
pub struct CompletionRequest<'a> {
    pub system: Option<&'a str>,
    pub messages: &'a [Message],
    pub tools: &'a [ToolDef],
    pub max_tokens: u32,
}

/// The model's response to a [`CompletionRequest`].
#[derive(Debug)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Option<Usage>,
}

/// Rough token estimate for messages: serialized length / 4.
///
/// Used for compaction triggers between provider-reported measurements;
/// deliberately cheap rather than exact.
pub fn estimate_tokens(messages: &[Message]) -> u64 {
    messages
        .iter()
        .map(|m| {
            serde_json::to_string(m)
                .map(|s| s.len() as u64)
                .unwrap_or(0)
        })
        .sum::<u64>()
        / 4
}

/// Truncate `s` to at most `max_chars` characters, keeping the beginning and
/// end and noting how many characters were dropped from the middle.
///
/// Operates on character boundaries, so it is safe for any UTF-8 input.
pub fn truncate_middle(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars || max_chars < 40 {
        return s.to_string();
    }
    let keep = (max_chars - 30) / 2;
    let start: String = s.chars().take(keep).collect();
    let end: String = s.chars().skip(count - keep).collect();
    let dropped = count - 2 * keep;
    format!(
        "{}\n[... {} characters truncated ...]\n{}",
        start, dropped, end
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn content_block_wire_format() {
        let block = ContentBlock::ToolResult {
            content: "4".to_string(),
            tool_use_id: "toolu_1".to_string(),
            is_error: None,
        };
        let v = serde_json::to_value(&block).unwrap();
        assert_eq!(v["type"], "tool_result");
        assert_eq!(v["tool_use_id"], "toolu_1");
        // is_error must be omitted, not null
        assert!(v.get("is_error").is_none());

        let thinking: ContentBlock =
            serde_json::from_value(json!({"type": "thinking", "thinking": "", "signature": "sig"}))
                .unwrap();
        assert_eq!(
            thinking,
            ContentBlock::Thinking {
                thinking: String::new(),
                signature: "sig".to_string()
            }
        );
    }

    #[test]
    fn tool_use_round_trip() {
        let msg = Message::assistant(vec![
            ContentBlock::Text {
                text: "checking".into(),
            },
            ContentBlock::ToolUse {
                name: "calculator".into(),
                input: json!({"expression": "2+2"}),
                id: "toolu_1".into(),
            },
        ]);
        let uses = msg.tool_uses();
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0].name, "calculator");
        assert_eq!(msg.text(), "checking");

        let round: Message = serde_json::from_str(&serde_json::to_string(&msg).unwrap()).unwrap();
        assert_eq!(round, msg);
    }

    #[test]
    fn stop_reason_parsing() {
        assert_eq!(StopReason::parse("end_turn"), StopReason::EndTurn);
        assert_eq!(StopReason::parse("stop"), StopReason::EndTurn);
        assert_eq!(StopReason::parse("length"), StopReason::MaxTokens);
        assert_eq!(StopReason::parse("tool_calls"), StopReason::ToolUse);
        assert_eq!(StopReason::parse("refusal"), StopReason::Refusal);
        assert_eq!(
            StopReason::parse("pause_turn"),
            StopReason::Other("pause_turn".into())
        );
    }

    #[test]
    fn truncate_middle_behavior() {
        let short = "hello";
        assert_eq!(truncate_middle(short, 100), short);

        let long: String = "x".repeat(1000);
        let truncated = truncate_middle(&long, 100);
        assert!(truncated.chars().count() < 1000);
        assert!(truncated.contains("truncated"));

        // Multi-byte characters must not panic.
        let emoji: String = "🦀".repeat(500);
        let t = truncate_middle(&emoji, 100);
        assert!(t.contains("truncated"));
    }
}
