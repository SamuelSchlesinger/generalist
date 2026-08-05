//! Provider-neutral conversation types.
//!
//! These types define the internal representation of a conversation. Each
//! [`crate::provider::Provider`] implementation translates them to and from
//! its own wire format, so nothing outside the provider modules depends on a
//! specific vendor API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Host provenance for a conversation message.
///
/// Ordinary user and assistant messages default to `Conversation`; the
/// separate goal-continuation value prevents host control text from being
/// rendered or retained as if the user authored it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageOrigin {
    #[default]
    Conversation,
    GoalContinuation,
}

/// A single message in a conversation.
///
/// `role` is `"user"` or `"assistant"`. Tool results are carried in user
/// messages, matching the convention used by every major provider.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "MessageOrigin::is_conversation")]
    pub origin: MessageOrigin,
}

impl Message {
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content,
            origin: MessageOrigin::Conversation,
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            origin: MessageOrigin::Conversation,
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self::user(vec![ContentBlock::Text { text: text.into() }])
    }

    /// The exact host-authored prompt that continues an active goal.
    pub fn goal_continuation() -> Self {
        Self {
            role: "user".to_string(),
            content: vec![ContentBlock::Text {
                text: crate::goal::GOAL_CONTINUATION_PROMPT.to_string(),
            }],
            origin: MessageOrigin::GoalContinuation,
        }
    }

    pub fn is_goal_continuation(&self) -> bool {
        self.origin == MessageOrigin::GoalContinuation
            && self.role == "user"
            && matches!(
                self.content.as_slice(),
                [ContentBlock::Text { text }]
                    if crate::goal::is_goal_continuation_prompt(text)
            )
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

impl MessageOrigin {
    fn is_conversation(&self) -> bool {
        *self == Self::Conversation
    }
}

/// One block of message content.
///
/// The serialized form matches the Anthropic wire format (`type` tag with
/// snake_case variants); the OpenAI provider translates as needed. The
/// `Thinking`/`RedactedThinking` variants exist so reasoning blocks returned
/// by a model can be replayed unchanged on subsequent requests, which some
/// providers require. A `Thinking` block with an empty signature is inspectable
/// reasoning from a compatible provider, not a replayable Anthropic block.
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

/// Host-enforced bounds for one provider completion.
///
/// `max_tokens` is only a request sent to a provider. These limits remain
/// authoritative when a compatible or custom provider ignores that hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionLimits {
    /// Combined payload bytes retained across text, reasoning, signatures,
    /// tool identifiers, and tool inputs.
    pub max_response_bytes: usize,
    /// Maximum number of provider content blocks.
    pub max_content_blocks: usize,
    /// Maximum tool calls accepted from one completion.
    pub max_tool_uses: usize,
}

impl Default for CompletionLimits {
    fn default() -> Self {
        Self {
            max_response_bytes: 1024 * 1024,
            max_content_blocks: 1024,
            max_tool_uses: 256,
        }
    }
}

impl CompletionLimits {
    pub(crate) fn checked_response_bytes(self, current: usize, additional: usize) -> Result<usize> {
        let observed = current.saturating_add(additional);
        if observed > self.max_response_bytes {
            return Err(Error::Other(format!(
                "provider completion exceeded the host payload limit of {} bytes",
                self.max_response_bytes
            )));
        }
        Ok(observed)
    }

    /// Bound for transport framing around the logical completion payload.
    /// Tiny SSE fragments have substantial JSON/frame overhead, so this is
    /// deliberately looser than `max_response_bytes` while remaining finite.
    pub(crate) fn max_wire_bytes(self) -> usize {
        self.max_response_bytes.saturating_mul(16).max(64 * 1024)
    }

    /// Validate a complete response before it becomes conversation history.
    pub fn validate_response(self, response: &CompletionResponse) -> Result<()> {
        if response.content.len() > self.max_content_blocks {
            return Err(Error::Other(format!(
                "provider completion contained {} blocks; host limit is {}",
                response.content.len(),
                self.max_content_blocks
            )));
        }
        let tool_uses = response
            .content
            .iter()
            .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
            .count();
        if tool_uses > self.max_tool_uses {
            return Err(Error::Other(format!(
                "provider completion contained {tool_uses} tool calls; host limit is {}",
                self.max_tool_uses
            )));
        }
        let payload_bytes = response.content.iter().fold(0usize, |total, block| {
            total.saturating_add(content_block_payload_bytes(block))
        });
        self.checked_response_bytes(0, payload_bytes)?;
        Ok(())
    }
}

pub(crate) fn json_payload_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 0,
        Value::Bool(_) => 1,
        Value::Number(number) => number.to_string().len(),
        Value::String(text) => text.len(),
        Value::Array(values) => values.iter().fold(0usize, |total, value| {
            total.saturating_add(json_payload_bytes(value))
        }),
        Value::Object(values) => values.iter().fold(0usize, |total, (key, value)| {
            total
                .saturating_add(key.len())
                .saturating_add(json_payload_bytes(value))
        }),
    }
}

fn content_block_payload_bytes(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text { text } => text.len(),
        ContentBlock::ToolUse { name, input, id } => name
            .len()
            .saturating_add(id.len())
            .saturating_add(json_payload_bytes(input)),
        ContentBlock::ToolResult {
            content,
            tool_use_id,
            ..
        } => content.len().saturating_add(tool_use_id.len()),
        ContentBlock::Thinking {
            thinking,
            signature,
        } => thinking.len().saturating_add(signature.len()),
        ContentBlock::RedactedThinking { data } => data.len(),
    }
}

/// A request for one model completion.
#[derive(Debug)]
pub struct CompletionRequest<'a> {
    pub system: Option<&'a str>,
    pub messages: &'a [Message],
    pub tools: &'a [ToolDef],
    /// Optional provider output-token request. `None` lets adapters omit it
    /// when their protocol permits; adapters whose protocol requires a value
    /// resolve the selected model's advertised maximum.
    pub max_tokens: Option<u32>,
    pub limits: CompletionLimits,
}

/// The model's response to a [`CompletionRequest`].
#[derive(Debug)]
pub struct CompletionResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Option<Usage>,
}

/// One inspectable fragment emitted while a provider response streams.
///
/// Reasoning is provider-supplied model output, not host inference. Providers
/// that do not expose reasoning simply emit no [`CompletionDelta::Reasoning`]
/// values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionDelta {
    Text(String),
    Reasoning(String),
}

impl CompletionDelta {
    pub fn len_bytes(&self) -> usize {
        match self {
            Self::Text(text) | Self::Reasoning(text) => text.len(),
        }
    }
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
    if count <= max_chars {
        return s.to_string();
    }
    if max_chars < 40 {
        return s.chars().take(max_chars).collect();
    }

    // The marker length depends on the number of omitted characters. Find the
    // small fixed point so the documented `max_chars` bound remains exact,
    // including around decimal digit boundaries.
    let mut retained = max_chars;
    let marker = loop {
        let dropped = count.saturating_sub(retained);
        let marker = format!("\n[... {dropped} characters truncated ...]\n");
        let next = max_chars.saturating_sub(marker.chars().count());
        if next == retained {
            break marker;
        }
        retained = next;
    };
    if marker.chars().count() > max_chars {
        return marker.chars().take(max_chars).collect();
    }
    let start_chars = retained / 2;
    let end_chars = retained - start_chars;
    let start: String = s.chars().take(start_chars).collect();
    let end: String = s.chars().skip(count - end_chars).collect();
    format!("{start}{marker}{end}")
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
    fn goal_continuation_requires_exact_host_provenance_and_shape() {
        let host = Message::goal_continuation();
        assert!(host.is_goal_continuation());

        let mut forged = host.clone();
        forged.content.push(ContentBlock::ToolResult {
            content: "extra".into(),
            tool_use_id: "forged".into(),
            is_error: None,
        });
        assert!(!forged.is_goal_continuation());

        assert!(!Message::user_text(crate::goal::GOAL_CONTINUATION_PROMPT).is_goal_continuation());
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
        assert!(truncated.chars().count() <= 100);
        assert!(truncated.contains("truncated"));

        // Multi-byte characters must not panic.
        let emoji: String = "🦀".repeat(500);
        let t = truncate_middle(&emoji, 100);
        assert!(t.chars().count() <= 100);
        assert!(t.contains("truncated"));

        assert_eq!(truncate_middle("abcdef", 3), "abc");
    }

    #[test]
    fn completion_limits_are_independent_of_provider_token_accounting() {
        let response = CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "1234567".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: Some(Usage {
                output_tokens: 1,
                ..Usage::default()
            }),
        };
        let limits = CompletionLimits {
            max_response_bytes: 6,
            ..CompletionLimits::default()
        };

        let error = limits.validate_response(&response).unwrap_err().to_string();
        assert!(error.contains("payload limit of 6 bytes"));
    }

    #[test]
    fn completion_limits_reject_excess_blocks_and_tool_calls() {
        let blocks = CompletionResponse {
            content: vec![
                ContentBlock::Text { text: "a".into() },
                ContentBlock::Text { text: "b".into() },
            ],
            stop_reason: StopReason::EndTurn,
            usage: None,
        };
        let block_error = CompletionLimits {
            max_content_blocks: 1,
            ..CompletionLimits::default()
        }
        .validate_response(&blocks)
        .unwrap_err()
        .to_string();
        assert!(block_error.contains("2 blocks"));

        let tools = CompletionResponse {
            content: (0..2)
                .map(|index| ContentBlock::ToolUse {
                    name: "echo".into(),
                    input: serde_json::json!({"index": index}),
                    id: format!("tool-{index}"),
                })
                .collect(),
            stop_reason: StopReason::ToolUse,
            usage: None,
        };
        let tool_error = CompletionLimits {
            max_tool_uses: 1,
            ..CompletionLimits::default()
        }
        .validate_response(&tools)
        .unwrap_err()
        .to_string();
        assert!(tool_error.contains("2 tool calls"));
    }
}
