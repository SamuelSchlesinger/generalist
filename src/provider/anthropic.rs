//! Anthropic Messages API provider.

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{CompletionRequest, CompletionResponse, ContentBlock, StopReason, Usage};
use async_trait::async_trait;
use serde_json::{json, Value};

const MESSAGES_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

/// Models offered in the CLI picker. The first entry is the default.
pub const SUGGESTED_MODELS: &[&str] = &["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"];

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        Ok(Self {
            api_key,
            model,
            client: super::http_client()?,
        })
    }

    /// Whether the model accepts `thinking: {type: "adaptive"}`.
    ///
    /// Adaptive thinking exists on the 4.6+ Opus/Sonnet generations and the
    /// Claude 5 family; older models (and Haiku 4.5) reject it.
    fn supports_adaptive_thinking(model: &str) -> bool {
        const PREFIXES: &[&str] = &[
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable",
            "claude-mythos",
        ];
        PREFIXES.iter().any(|p| model.starts_with(p))
    }

    /// Build the request body, adding prompt-cache breakpoints.
    ///
    /// Caching is a prefix match, so we mark two stable boundaries: the system
    /// prompt (which also covers the tool definitions rendered before it) and
    /// the last block of the last message, so each turn extends the cached
    /// prefix instead of re-reading the whole conversation at full price.
    fn build_body(&self, req: &CompletionRequest<'_>) -> Result<Value> {
        let last_msg = req.messages.len().saturating_sub(1);
        let mut messages = Vec::with_capacity(req.messages.len());
        for (i, message) in req.messages.iter().enumerate() {
            let mut content: Vec<Value> = message
                .content
                .iter()
                .map(serde_json::to_value)
                .collect::<std::result::Result<_, _>>()?;
            if i == last_msg {
                if let Some(block) = content.last_mut() {
                    let cacheable = matches!(
                        block.get("type").and_then(|t| t.as_str()),
                        Some("text") | Some("tool_use") | Some("tool_result")
                    );
                    if cacheable {
                        block["cache_control"] = json!({"type": "ephemeral"});
                    }
                }
            }
            messages.push(json!({"role": message.role, "content": content}));
        }

        let mut body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        });

        if let Some(system) = req.system {
            body["system"] = json!([{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }]);
        }
        if !req.tools.is_empty() {
            body["tools"] = serde_json::to_value(req.tools)?;
        }
        if Self::supports_adaptive_thinking(&self.model) {
            body["thinking"] = json!({"type": "adaptive"});
        }
        Ok(body)
    }

    fn parse_response(value: Value) -> Result<CompletionResponse> {
        let content = value
            .get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| Error::Other("response missing content array".to_string()))?
            .iter()
            // Skip block types we don't model (server tool results etc.)
            // rather than failing the whole response.
            .filter_map(|block| serde_json::from_value::<ContentBlock>(block.clone()).ok())
            .collect();

        let stop_reason = value
            .get("stop_reason")
            .and_then(|s| s.as_str())
            .map(StopReason::parse)
            .unwrap_or(StopReason::EndTurn);

        let usage = value.get("usage").map(|u| Usage {
            input_tokens: u.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            output_tokens: u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            cache_read_input_tokens: u.get("cache_read_input_tokens").and_then(|v| v.as_u64()),
            cache_creation_input_tokens: u
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_u64()),
        });

        Ok(CompletionResponse {
            content,
            stop_reason,
            usage,
        })
    }
}

/// Accumulates Anthropic SSE events into a complete response.
///
/// Pure state (no I/O) so the event grammar is unit-testable.
#[derive(Default)]
struct StreamState {
    blocks: Vec<Value>,
    /// Partial `input_json_delta` text per block index (tool_use inputs
    /// stream as JSON fragments).
    partial_json: Vec<String>,
    stop_reason: Option<StopReason>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read: Option<u64>,
    cache_creation: Option<u64>,
}

impl StreamState {
    /// Apply one SSE event; returns any user-visible text delta.
    fn apply(&mut self, event: &Value) -> Result<Option<String>> {
        match event.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "message_start" => {
                if let Some(usage) = event.pointer("/message/usage") {
                    self.input_tokens = usage
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    self.cache_read = usage
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_u64());
                    self.cache_creation = usage
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_u64());
                }
            }
            "content_block_start" => {
                let block = event.get("content_block").cloned().unwrap_or(Value::Null);
                self.blocks.push(block);
                self.partial_json.push(String::new());
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let Some(delta) = event.get("delta") else {
                    return Ok(None);
                };
                if index >= self.blocks.len() {
                    return Ok(None);
                }
                match delta.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "text_delta" => {
                        let text = delta.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        if let Some(existing) = self.blocks[index]
                            .get_mut("text")
                            .and_then(|t| t.as_str().map(String::from))
                        {
                            self.blocks[index]["text"] = json!(existing + text);
                        }
                        return Ok(Some(text.to_string()));
                    }
                    "input_json_delta" => {
                        let fragment = delta
                            .get("partial_json")
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        self.partial_json[index].push_str(fragment);
                    }
                    "thinking_delta" => {
                        let fragment = delta.get("thinking").and_then(|t| t.as_str()).unwrap_or("");
                        if let Some(existing) = self.blocks[index]
                            .get_mut("thinking")
                            .and_then(|t| t.as_str().map(String::from))
                        {
                            self.blocks[index]["thinking"] = json!(existing + fragment);
                        }
                    }
                    "signature_delta" => {
                        let fragment = delta
                            .get("signature")
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        if let Some(existing) = self.blocks[index]
                            .get_mut("signature")
                            .and_then(|t| t.as_str().map(String::from))
                        {
                            self.blocks[index]["signature"] = json!(existing + fragment);
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(stop) = event.pointer("/delta/stop_reason").and_then(|s| s.as_str()) {
                    self.stop_reason = Some(StopReason::parse(stop));
                }
                if let Some(out) = event
                    .pointer("/usage/output_tokens")
                    .and_then(|v| v.as_u64())
                {
                    self.output_tokens = out;
                }
            }
            "error" => {
                let message = event
                    .pointer("/error/message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("stream error");
                return Err(Error::Api {
                    status: 0,
                    message: message.to_string(),
                });
            }
            // ping / content_block_stop / message_stop carry no state we need.
            _ => {}
        }
        Ok(None)
    }

    fn into_response(mut self) -> CompletionResponse {
        // Finalize tool_use inputs from their streamed JSON fragments.
        for (block, partial) in self.blocks.iter_mut().zip(&self.partial_json) {
            if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") && !partial.is_empty()
            {
                block["input"] =
                    serde_json::from_str(partial).unwrap_or(json!({"_unparsed_input": partial}));
            }
        }
        let content = self
            .blocks
            .into_iter()
            .filter_map(|block| serde_json::from_value::<ContentBlock>(block).ok())
            .collect();
        CompletionResponse {
            content,
            stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
            usage: Some(Usage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                cache_read_input_tokens: self.cache_read,
                cache_creation_input_tokens: self.cache_creation,
            }),
        }
    }
}

#[async_trait(?Send)]
impl Provider for AnthropicProvider {
    fn id(&self) -> &'static str {
        "anthropic"
    }

    fn display_name(&self) -> &str {
        "Anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        let body = self.build_body(&request)?;
        let response = self
            .client
            .post(MESSAGES_ENDPOINT)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(str::to_string)
                })
                .unwrap_or(text);
            return Err(Error::Api {
                status: status.as_u16(),
                message,
            });
        }

        Self::parse_response(serde_json::from_str(&text)?)
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(String),
    ) -> Result<CompletionResponse> {
        let mut body = self.build_body(&request)?;
        body["stream"] = json!(true);

        let response = self
            .client
            .post(MESSAGES_ENDPOINT)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await?;
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .map(str::to_string)
                })
                .unwrap_or(text);
            return Err(Error::Api {
                status: status.as_u16(),
                message,
            });
        }

        let mut response = response;
        let mut assembler = crate::provider::SseAssembler::default();
        let mut state = StreamState::default();
        let mut saw_message_stop = false;
        while let Some(chunk) = response.chunk().await? {
            for payload in assembler.push(&chunk) {
                if let Ok(event) = serde_json::from_str::<Value>(&payload) {
                    if event.get("type").and_then(|t| t.as_str()) == Some("message_stop") {
                        saw_message_stop = true;
                    }
                    if let Some(text) = state.apply(&event)? {
                        on_delta(text);
                    }
                }
            }
        }
        if let Some(payload) = assembler.finish() {
            if let Ok(event) = serde_json::from_str::<Value>(&payload) {
                if event.get("type").and_then(|t| t.as_str()) == Some("message_stop") {
                    saw_message_stop = true;
                }
                if let Some(text) = state.apply(&event)? {
                    on_delta(text);
                }
            }
        }
        // A connection cut mid-stream yields a syntactically fine but
        // incomplete message; surface it as a retryable-shaped error rather
        // than acting on a half response.
        if !saw_message_stop {
            return Err(Error::Api {
                status: 0,
                message: "stream ended before message_stop".to_string(),
            });
        }
        Ok(state.into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, ToolDef};

    fn provider() -> AnthropicProvider {
        AnthropicProvider::new("test-key".into(), "claude-opus-4-8".into()).unwrap()
    }

    #[test]
    fn body_omits_system_when_absent() {
        let messages = vec![Message::user_text("hi")];
        let req = CompletionRequest {
            system: None,
            messages: &messages,
            tools: &[],
            max_tokens: 100,
        };
        let body = provider().build_body(&req).unwrap();
        assert!(body.get("system").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn body_adds_cache_breakpoints_and_thinking() {
        let messages = vec![Message::user_text("hi")];
        let tools = vec![ToolDef {
            name: "t".into(),
            description: "d".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let req = CompletionRequest {
            system: Some("be helpful"),
            messages: &messages,
            tools: &tools,
            max_tokens: 100,
        };
        let body = provider().build_body(&req).unwrap();
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["tools"][0]["name"], "t");
    }

    #[test]
    fn haiku_gets_no_thinking_param() {
        let p = AnthropicProvider::new("k".into(), "claude-haiku-4-5".into()).unwrap();
        let messages = vec![Message::user_text("hi")];
        let req = CompletionRequest {
            system: None,
            messages: &messages,
            tools: &[],
            max_tokens: 10,
        };
        let body = p.build_body(&req).unwrap();
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn stream_state_accumulates_text_tools_and_thinking() {
        let events = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 12, "cache_read_input_tokens": 4}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"comm"}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "and\": \"ls\"}"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
            json!({"type": "message_stop"}),
        ];
        let mut state = StreamState::default();
        let mut streamed = String::new();
        for event in &events {
            if let Some(text) = state.apply(event).unwrap() {
                streamed.push_str(&text);
            }
        }
        assert_eq!(streamed, "Hello");
        let response = state.into_response();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.content.len(), 3);
        match &response.content[0] {
            ContentBlock::Thinking {
                thinking,
                signature,
            } => {
                assert_eq!(thinking, "hmm");
                assert_eq!(signature, "sig");
            }
            other => panic!("expected thinking, got {:?}", other),
        }
        match &response.content[2] {
            ContentBlock::ToolUse { name, input, id } => {
                assert_eq!(name, "bash");
                assert_eq!(id, "t1");
                assert_eq!(input["command"], "ls");
            }
            other => panic!("expected tool use, got {:?}", other),
        }
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 9);
        assert_eq!(usage.cache_read_input_tokens, Some(4));
    }

    #[test]
    fn stream_error_events_surface_as_errors() {
        let mut state = StreamState::default();
        let result = state.apply(&json!({"type": "error", "error": {"message": "overloaded"}}));
        assert!(result.is_err());
    }

    #[test]
    fn parse_response_extracts_blocks_and_skips_unknown() {
        let value = serde_json::json!({
            "content": [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": "hello"},
                {"type": "some_future_block", "data": 1},
                {"type": "tool_use", "name": "bash", "input": {"command": "ls"}, "id": "toolu_1"}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 3}
        });
        let resp = AnthropicProvider::parse_response(value).unwrap();
        assert_eq!(resp.content.len(), 3);
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.cache_read_input_tokens, Some(3));
    }
}
