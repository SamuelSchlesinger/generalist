//! LLM provider abstraction.
//!
//! The agent core speaks only [`Provider`]; concrete implementations translate
//! the neutral types in [`crate::types`] to a vendor wire format. Two are
//! included:
//!
//! - [`AnthropicProvider`] — the Anthropic Messages API
//! - [`OpenAiProvider`] — any OpenAI-compatible chat-completions API
//!   (OpenAI itself, Ollama, Groq, Mistral, vLLM, ...)

pub mod anthropic;
pub mod openai;

pub use anthropic::AnthropicProvider;
pub use openai::OpenAiProvider;

use crate::error::Result;
use crate::types::{CompletionDelta, CompletionRequest, CompletionResponse};
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
    /// The default implementation falls back to non-streaming, so custom
    /// providers work without implementing it.
    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta),
    ) -> Result<CompletionResponse> {
        let _ = on_delta;
        self.complete(request).await
    }
}

/// Build the shared HTTP client used by providers.
///
/// reqwest has no default timeout; agent calls can legitimately run for
/// minutes on hard tasks, so use a generous ceiling rather than none at all.
pub(crate) fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
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
    buffer: String,
    data_lines: Vec<String>,
}

impl SseAssembler {
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut events = Vec::new();
        while let Some(newline) = self.buffer.find('\n') {
            let line: String = self.buffer.drain(..=newline).collect();
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
        if let Some(data) = self
            .buffer
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
}
