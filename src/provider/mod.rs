//! LLM provider abstraction.
//!
//! The agent core speaks only [`Provider`]; concrete implementations translate
//! the neutral types in [`crate::types`] to a vendor wire format. Two are
//! included:
//!
//! - [`AnthropicProvider`] — the Anthropic Messages API
//! - [`OpenAiProvider`] — any OpenAI-compatible chat-completions API
//!   (OpenAI itself, Ollama, Groq, Mistral, vLLM, ...)
//! - [`OpenRouterProvider`] — OpenRouter's OpenAI-compatible endpoint, with a
//!   distinct persistence identity

pub mod anthropic;
pub mod openai;
pub mod openrouter;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAiProvider;
pub use openrouter::OpenRouterProvider;

use crate::error::{Error, Result};
use crate::types::{CompletionDelta, CompletionLimits, CompletionRequest, CompletionResponse};
use async_trait::async_trait;

/// A backend capable of producing one model completion per call.
///
/// Implementations should be stateless with respect to the conversation:
/// the full history arrives in every [`CompletionRequest`].
///
/// Futures are `?Send`: the agent is a single-task loop (UI callbacks hold
/// non-`Sync` state), so provider calls never cross threads. Run the agent
/// inside a `LocalSet` if you need to spawn it.
#[async_trait(?Send)]
pub trait Provider: Send + Sync {
    /// Short stable identifier used for persistence and provider selection,
    /// e.g. `"anthropic"` or `"openai"`. This identifies the adapter, not
    /// necessarily the service behind a compatible endpoint.
    fn id(&self) -> &'static str;

    /// Human-facing API/backend label. Custom providers default to their
    /// stable ID; protocol adapters should override this when the ID could
    /// misleadingly imply a particular service.
    fn display_name(&self) -> &str {
        self.id()
    }

    /// The model this provider instance targets.
    fn model(&self) -> &str;

    /// Request a single completion.
    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse>;

    /// Request a completion, streaming assistant text and any provider-supplied
    /// reasoning through `on_delta` as they arrive. Returns the complete
    /// response as if [`Provider::complete`] had been called.
    ///
    /// Providers must stop producing the completion when `on_delta` returns
    /// an error. The host uses that path to enforce completion limits before
    /// an untrusted stream can grow further.
    ///
    /// The default implementation falls back to non-streaming, so custom
    /// providers work without implementing it.
    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        let _ = on_delta;
        self.complete(request).await
    }
}

/// Extract the provider's error message and machine-readable error type
/// from a parsed error payload, when it carries an `error` member.
///
/// Handles both the object form (`{"error": {"message": ..., "type": ...}}`,
/// used by Anthropic and OpenAI) and the bare-string form (`{"error": "..."}`,
/// used by Ollama).
pub(crate) fn parse_error_value(value: &serde_json::Value) -> Option<(String, Option<String>)> {
    let error = value.get("error")?;
    if let serde_json::Value::String(text) = error {
        return Some((text.clone(), None));
    }
    let message = error
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown provider error")
        .to_string();
    let error_type = error
        .get("type")
        .or_else(|| error.get("code"))
        .and_then(|t| t.as_str())
        .map(str::to_string);
    Some((message, error_type))
}

/// Extract message and error type from a raw error body, falling back to
/// the body text when it is not structured JSON.
pub(crate) fn parse_error_body(body: &[u8]) -> (String, Option<String>) {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| parse_error_value(&value))
        .unwrap_or_else(|| (String::from_utf8_lossy(body).into_owned(), None))
}

/// Guarantee every tool_use id in one response is non-empty and unique.
///
/// Small OpenAI-compatible models sometimes omit or repeat tool-call ids, and
/// the downstream history invariant (exactly one result per unique id) must
/// hold for whatever the adapter emits. Chat-completion providers are
/// stateless, so a synthesized id replays consistently: the paired tool
/// result carries the same id back on the next request.
pub(crate) fn ensure_unique_tool_call_ids(content: &mut [crate::types::ContentBlock]) {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut counter = 0usize;
    for block in content {
        if let crate::types::ContentBlock::ToolUse { id, .. } = block {
            while id.is_empty() || seen.contains(id.as_str()) {
                counter += 1;
                let base = if id.is_empty() { "call" } else { id.as_str() };
                let candidate = format!("{base}_{counter}");
                if !seen.contains(&candidate) {
                    *id = candidate;
                }
            }
            seen.insert(id.clone());
        }
    }
}

pub(crate) fn account_wire_bytes(
    limits: CompletionLimits,
    received: &mut usize,
    additional: usize,
) -> Result<()> {
    let observed = received.saturating_add(additional);
    let limit = limits.max_wire_bytes();
    if observed > limit {
        return Err(Error::Limit(format!(
            "provider response exceeded the host wire limit of {limit} bytes"
        )));
    }
    *received = observed;
    Ok(())
}

pub(crate) async fn read_response_body_bounded(
    mut response: reqwest::Response,
    limits: CompletionLimits,
) -> Result<Vec<u8>> {
    let limit = limits.max_wire_bytes();
    if response
        .content_length()
        .is_some_and(|content_length| content_length > limit as u64)
    {
        return Err(Error::Limit(format!(
            "provider response exceeded the host wire limit of {limit} bytes"
        )));
    }
    let mut body = Vec::new();
    let mut received = 0usize;
    while let Some(chunk) = response.chunk().await? {
        account_wire_bytes(limits, &mut received, chunk.len())?;
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Build the shared HTTP client used by providers.
///
/// No whole-request deadline: a hard agent call on a slow local model can
/// legitimately stream for longer than any fixed ceiling, and cutting it
/// mid-stream only to retry the same slow request makes things worse. A
/// stall is instead detected by inactivity — `read_timeout` bounds the gap
/// between reads (generous, because local prefill can be silent for
/// minutes) — and the user can always cancel sooner.
pub(crate) fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .read_timeout(std::time::Duration::from_secs(600))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(crate::error::Error::from)
}

/// Reassemble SSE `data:` payloads from a chunked byte stream.
///
/// Feed raw chunks with [`SseAssembler::push`]; complete `data` payloads come
/// back in order. Handles multi-line `data:` fields, ignores comments and
/// other fields, and treats `[DONE]` as an ordinary payload for the caller
/// to interpret.
#[derive(Default)]
pub(crate) struct SseAssembler {
    // HTTP chunks may split a multi-byte UTF-8 code point. Keep raw bytes
    // until a complete SSE line arrives so lossless text is not corrupted at
    // an arbitrary transport boundary.
    buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseAssembler {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.buffer.drain(..=newline).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                // Blank line terminates one event.
                if !self.data_lines.is_empty() {
                    events.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data_lines.push(data.trim_start().to_string());
            }
            // `event:`/`id:`/comments are irrelevant here: both providers
            // discriminate on the JSON payload's own `type` field.
        }
        events
    }

    /// Flush a trailing event that wasn't newline-terminated.
    pub fn finish(&mut self) -> Option<String> {
        let trailing = String::from_utf8_lossy(&self.buffer);
        if let Some(data) = trailing
            .trim_end_matches(['\n', '\r'])
            .strip_prefix("data:")
        {
            self.data_lines.push(data.trim_start().to_string());
        }
        self.buffer.clear();
        if self.data_lines.is_empty() {
            None
        } else {
            let event = self.data_lines.join("\n");
            self.data_lines.clear();
            Some(event)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_assembler_handles_split_chunks_and_multiline_data() {
        let mut assembler = SseAssembler::default();
        assert!(assembler.push(b"event: message\nda").is_empty());
        let events = assembler.push(b"ta: {\"a\":1}\n\ndata: part1\ndata: part2\n\n");
        assert_eq!(
            events,
            vec!["{\"a\":1}".to_string(), "part1\npart2".to_string()]
        );
        let events = assembler.push(b"data: [DONE]\n");
        assert!(events.is_empty());
        assert_eq!(assembler.finish(), Some("[DONE]".to_string()));
    }

    #[test]
    fn sse_assembler_handles_crlf_line_endings() {
        // Real HTTP servers commonly send CRLF; the blank line that
        // terminates an event is then "\r\n", not "\n".
        let mut assembler = SseAssembler::default();
        let events = assembler.push(b"data: {\"a\":1}\r\n\r\ndata: two\r\n\r\n");
        assert_eq!(events, vec!["{\"a\":1}".to_string(), "two".to_string()]);
        assert_eq!(assembler.finish(), None);
    }

    #[test]
    fn sse_assembler_preserves_utf8_split_across_transport_chunks() {
        let bytes = "data: {\"text\":\"café\"}\n\n".as_bytes();
        let split = bytes
            .windows(2)
            .position(|pair| pair == "é".as_bytes())
            .expect("UTF-8 marker")
            + 1;
        let mut assembler = SseAssembler::default();

        assert!(assembler.push(&bytes[..split]).is_empty());
        assert_eq!(
            assembler.push(&bytes[split..]),
            vec!["{\"text\":\"café\"}".to_string()]
        );
    }

    #[test]
    fn missing_and_duplicate_tool_call_ids_are_synthesized_unique() {
        use crate::types::ContentBlock;
        use serde_json::json;

        let tool_use = |id: &str| ContentBlock::ToolUse {
            name: "echo".to_string(),
            input: json!({}),
            id: id.to_string(),
        };
        let mut content = vec![
            tool_use(""),
            tool_use(""),
            tool_use("dup"),
            tool_use("dup"),
            ContentBlock::Text { text: "t".into() },
            tool_use("unique"),
        ];
        ensure_unique_tool_call_ids(&mut content);
        let ids: Vec<&str> = content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids.len(), 5);
        assert!(ids.iter().all(|id| !id.is_empty()));
        let unique: std::collections::HashSet<&&str> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
        // Provider-supplied ids that were already unique are untouched.
        assert_eq!(ids[2], "dup");
        assert_eq!(ids[4], "unique");
    }

    #[test]
    fn wire_accounting_is_finite_even_for_tiny_payload_limits() {
        let limits = CompletionLimits {
            max_response_bytes: 1,
            ..CompletionLimits::default()
        };
        let mut received = 0;
        account_wire_bytes(limits, &mut received, 64 * 1024).unwrap();
        let error = account_wire_bytes(limits, &mut received, 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("wire limit of 65536 bytes"));
        assert_eq!(received, 64 * 1024);
    }
}
