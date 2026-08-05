//! Anthropic Messages API provider.

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    json_payload_bytes, CompletionDelta, CompletionLimits, CompletionRequest, CompletionResponse,
    ContentBlock, StopReason, Usage,
};
use async_trait::async_trait;
use serde_json::{json, Value};

const MESSAGES_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";
const MODELS_ENDPOINT: &str = "https://api.anthropic.com/v1/models/";
const API_VERSION: &str = "2023-06-01";

/// Models offered in the CLI picker. The first entry is the default.
pub const SUGGESTED_MODELS: &[&str] = &["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"];

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    client: reqwest::Client,
    model_max_tokens: tokio::sync::OnceCell<u32>,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        Ok(Self {
            api_key,
            model,
            client: super::http_client()?,
            model_max_tokens: tokio::sync::OnceCell::new(),
        })
    }

    async fn discover_model_max_tokens(&self) -> Result<u32> {
        let mut url = reqwest::Url::parse(MODELS_ENDPOINT)
            .map_err(|error| Error::Other(format!("invalid Anthropic Models URL: {error}")))?;
        url.path_segments_mut()
            .map_err(|_| Error::Other("Anthropic Models URL cannot contain a model ID".into()))?
            .push(&self.model);
        let response = self
            .client
            .get(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .send()
            .await?;
        let status = response.status();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(crate::error::Error::parse_retry_after);
        let body = crate::provider::read_response_body_bounded(
            response,
            CompletionLimits {
                max_response_bytes: 256 * 1024,
                max_content_blocks: 1,
                max_tool_uses: 0,
            },
        )
        .await?;
        if !status.is_success() {
            let message = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|value| {
                    value
                        .pointer("/error/message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
            return Err(Error::Api {
                status: status.as_u16(),
                message: format!(
                    "could not discover the model output limit ({message}); pass --max-tokens to bypass discovery"
                ),
                retry_after,
            });
        }
        Self::parse_model_max_tokens(&serde_json::from_slice(&body)?)
    }

    fn parse_model_max_tokens(model: &Value) -> Result<u32> {
        let value = model
            .get("max_tokens")
            .and_then(Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                Error::Other(
                    "Anthropic model metadata omitted a positive max_tokens value; pass --max-tokens explicitly"
                        .into(),
                )
            })?;
        u32::try_from(value).map_err(|_| {
            Error::Other(format!(
                "Anthropic model max_tokens value {value} does not fit this client"
            ))
        })
    }

    async fn resolve_max_tokens(&self, requested: Option<u32>) -> Result<u32> {
        if let Some(requested) = requested {
            return Ok(requested);
        }
        Ok(*self
            .model_max_tokens
            .get_or_try_init(|| self.discover_model_max_tokens())
            .await?)
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
    fn build_body(&self, req: &CompletionRequest<'_>, max_tokens: u32) -> Result<Value> {
        let mut messages = Vec::with_capacity(req.messages.len());
        for message in req.messages {
            let content: Vec<Value> = message
                .content
                .iter()
                // Signed Anthropic thinking blocks must be replayed unchanged.
                // OpenAI-compatible reasoning extensions are stored in the
                // same neutral variant with an empty signature so the TUI can
                // inspect and persist them, but they are not valid Anthropic
                // input after a provider switch.
                .filter(|block| {
                    !matches!(
                        block,
                        ContentBlock::Thinking { signature, .. } if signature.is_empty()
                    )
                })
                .map(serde_json::to_value)
                .collect::<std::result::Result<_, _>>()?;
            if !content.is_empty() {
                messages.push(json!({"role": message.role, "content": content}));
            }
        }
        if let Some(block) = messages
            .last_mut()
            .and_then(|message| message["content"].as_array_mut())
            .and_then(|content| content.last_mut())
        {
            let cacheable = matches!(
                block.get("type").and_then(|t| t.as_str()),
                Some("text") | Some("tool_use") | Some("tool_result")
            );
            if cacheable {
                block["cache_control"] = json!({"type": "ephemeral"});
            }
        }

        let mut body = json!({
            "model": self.model,
            "max_tokens": max_tokens,
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
    retained_bytes: usize,
    tool_uses: usize,
}

impl StreamState {
    fn initial_block_bytes(block: &Value) -> usize {
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => block
                .get("text")
                .and_then(Value::as_str)
                .map_or(0, str::len),
            "thinking" => block
                .get("thinking")
                .and_then(Value::as_str)
                .map_or(0, str::len)
                .saturating_add(
                    block
                        .get("signature")
                        .and_then(Value::as_str)
                        .map_or(0, str::len),
                ),
            "redacted_thinking" => block
                .get("data")
                .and_then(Value::as_str)
                .map_or(0, str::len),
            "tool_use" => block
                .get("id")
                .and_then(Value::as_str)
                .map_or(0, str::len)
                .saturating_add(
                    block
                        .get("name")
                        .and_then(Value::as_str)
                        .map_or(0, str::len),
                )
                .saturating_add(block.get("input").map_or(0, json_payload_bytes)),
            _ => 0,
        }
    }

    /// Apply one SSE event; returns every inspectable delta it contains.
    fn apply(&mut self, event: &Value, limits: CompletionLimits) -> Result<Vec<CompletionDelta>> {
        let mut emitted = Vec::new();
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
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let index = usize::try_from(index).map_err(|_| {
                    Error::Other("provider content-block index does not fit this host".to_string())
                })?;
                if index != self.blocks.len() {
                    return Err(Error::Other(format!(
                        "provider content-block index {index} was out of sequence; expected {}",
                        self.blocks.len()
                    )));
                }
                if self.blocks.len() >= limits.max_content_blocks {
                    return Err(Error::Other(format!(
                        "provider completion exceeded the host limit of {} blocks",
                        limits.max_content_blocks
                    )));
                }
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    if self.tool_uses >= limits.max_tool_uses {
                        return Err(Error::Other(format!(
                            "provider completion exceeded the host limit of {} tool calls",
                            limits.max_tool_uses
                        )));
                    }
                    self.tool_uses += 1;
                }
                self.retained_bytes = limits.checked_response_bytes(
                    self.retained_bytes,
                    Self::initial_block_bytes(&block),
                )?;
                self.blocks.push(block);
                self.partial_json.push(String::new());
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let index = usize::try_from(index).map_err(|_| {
                    Error::Other("provider content-block index does not fit this host".to_string())
                })?;
                let Some(delta) = event.get("delta") else {
                    return Ok(emitted);
                };
                if index >= self.blocks.len() {
                    return Err(Error::Other(format!(
                        "provider delta referenced unknown content-block index {index}"
                    )));
                }
                match delta.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "text_delta" => {
                        let text = delta.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        self.retained_bytes =
                            limits.checked_response_bytes(self.retained_bytes, text.len())?;
                        if let Some(Value::String(existing)) = self.blocks[index].get_mut("text") {
                            existing.push_str(text);
                        }
                        if !text.is_empty() {
                            emitted.push(CompletionDelta::Text(text.to_string()));
                        }
                    }
                    "input_json_delta" => {
                        let fragment = delta
                            .get("partial_json")
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        self.retained_bytes =
                            limits.checked_response_bytes(self.retained_bytes, fragment.len())?;
                        self.partial_json[index].push_str(fragment);
                    }
                    "thinking_delta" => {
                        let fragment = delta.get("thinking").and_then(|t| t.as_str()).unwrap_or("");
                        self.retained_bytes =
                            limits.checked_response_bytes(self.retained_bytes, fragment.len())?;
                        if let Some(Value::String(existing)) =
                            self.blocks[index].get_mut("thinking")
                        {
                            existing.push_str(fragment);
                        }
                        if !fragment.is_empty() {
                            emitted.push(CompletionDelta::Reasoning(fragment.to_string()));
                        }
                    }
                    "signature_delta" => {
                        let fragment = delta
                            .get("signature")
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        self.retained_bytes =
                            limits.checked_response_bytes(self.retained_bytes, fragment.len())?;
                        if let Some(Value::String(existing)) =
                            self.blocks[index].get_mut("signature")
                        {
                            existing.push_str(fragment);
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
                    retry_after: None,
                });
            }
            // ping / content_block_stop / message_stop carry no state we need.
            _ => {}
        }
        Ok(emitted)
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
        let limits = request.limits;
        let max_tokens = self.resolve_max_tokens(request.max_tokens).await?;
        let body = self.build_body(&request, max_tokens)?;
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
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(crate::error::Error::parse_retry_after);
        let response_body = crate::provider::read_response_body_bounded(response, limits).await?;
        if !status.is_success() {
            let message = serde_json::from_slice::<Value>(&response_body)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| String::from_utf8_lossy(&response_body).into_owned());
            return Err(Error::Api {
                status: status.as_u16(),
                message,
                retry_after,
            });
        }

        let response = Self::parse_response(serde_json::from_slice(&response_body)?)?;
        limits.validate_response(&response)?;
        Ok(response)
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        let limits = request.limits;
        let max_tokens = self.resolve_max_tokens(request.max_tokens).await?;
        let mut body = self.build_body(&request, max_tokens)?;
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
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(crate::error::Error::parse_retry_after);
        if !status.is_success() {
            let response_body =
                crate::provider::read_response_body_bounded(response, limits).await?;
            let message = serde_json::from_slice::<Value>(&response_body)
                .ok()
                .and_then(|v| {
                    v.pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .map(str::to_string)
                })
                .unwrap_or_else(|| String::from_utf8_lossy(&response_body).into_owned());
            return Err(Error::Api {
                status: status.as_u16(),
                message,
                retry_after,
            });
        }

        let mut response = response;
        let mut assembler = crate::provider::SseAssembler::default();
        let mut state = StreamState::default();
        let mut saw_message_stop = false;
        let mut received = 0usize;
        while let Some(chunk) = response.chunk().await? {
            crate::provider::account_wire_bytes(limits, &mut received, chunk.len())?;
            for payload in assembler.push(&chunk) {
                if let Ok(event) = serde_json::from_str::<Value>(&payload) {
                    if event.get("type").and_then(|t| t.as_str()) == Some("message_stop") {
                        saw_message_stop = true;
                    }
                    for delta in state.apply(&event, limits)? {
                        on_delta(delta)?;
                    }
                }
            }
        }
        if let Some(payload) = assembler.finish() {
            if let Ok(event) = serde_json::from_str::<Value>(&payload) {
                if event.get("type").and_then(|t| t.as_str()) == Some("message_stop") {
                    saw_message_stop = true;
                }
                for delta in state.apply(&event, limits)? {
                    on_delta(delta)?;
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
                retry_after: None,
            });
        }
        let response = state.into_response();
        limits.validate_response(&response)?;
        Ok(response)
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
            max_tokens: Some(100),
            limits: CompletionLimits::default(),
        };
        let body = provider().build_body(&req, 100).unwrap();
        assert!(body.get("system").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn body_uses_resolved_model_max_or_explicit_token_override() {
        let messages = vec![Message::user_text("hi")];
        let delegated = CompletionRequest {
            system: None,
            messages: &messages,
            tools: &[],
            max_tokens: None,
            limits: CompletionLimits::default(),
        };
        assert_eq!(
            provider().build_body(&delegated, 128_000).unwrap()["max_tokens"],
            128_000
        );

        let explicit = CompletionRequest {
            max_tokens: Some(32_000),
            ..delegated
        };
        assert_eq!(
            provider().build_body(&explicit, 32_000).unwrap()["max_tokens"],
            32_000
        );
    }

    #[test]
    fn model_metadata_requires_a_positive_client_sized_maximum() {
        assert_eq!(
            AnthropicProvider::parse_model_max_tokens(&json!({"max_tokens": 128000})).unwrap(),
            128_000
        );
        assert!(AnthropicProvider::parse_model_max_tokens(&json!({"max_tokens": 0})).is_err());
        assert!(AnthropicProvider::parse_model_max_tokens(&json!({})).is_err());
        assert!(AnthropicProvider::parse_model_max_tokens(
            &json!({"max_tokens": u64::from(u32::MAX) + 1})
        )
        .is_err());
    }

    #[tokio::test]
    async fn explicit_token_override_bypasses_model_discovery() {
        assert_eq!(
            provider().resolve_max_tokens(Some(32_000)).await.unwrap(),
            32_000
        );
    }

    #[test]
    fn body_adds_cache_breakpoints_and_thinking() {
        let messages = vec![
            Message::assistant(vec![ContentBlock::Thinking {
                thinking: "unsigned-only compatible reasoning".into(),
                signature: String::new(),
            }]),
            Message::assistant(vec![
                ContentBlock::Thinking {
                    thinking: "unsigned compatible reasoning".into(),
                    signature: String::new(),
                },
                ContentBlock::Thinking {
                    thinking: "signed Anthropic reasoning".into(),
                    signature: "signature".into(),
                },
                ContentBlock::Text { text: "hi".into() },
            ]),
            Message::user_text("continue"),
        ];
        let tools = vec![ToolDef {
            name: "t".into(),
            description: "d".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let req = CompletionRequest {
            system: Some("be helpful"),
            messages: &messages,
            tools: &tools,
            max_tokens: Some(100),
            limits: CompletionLimits::default(),
        };
        let body = provider().build_body(&req, 100).unwrap();
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
        assert_eq!(body["messages"][0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(
            body["messages"][0]["content"][0]["thinking"],
            "signed Anthropic reasoning"
        );
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"]["type"],
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
            max_tokens: Some(10),
            limits: CompletionLimits::default(),
        };
        let body = p.build_body(&req, 10).unwrap();
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
        let mut streamed_text = String::new();
        let mut streamed_reasoning = String::new();
        for event in &events {
            for delta in state.apply(event, CompletionLimits::default()).unwrap() {
                match delta {
                    CompletionDelta::Text(text) => streamed_text.push_str(&text),
                    CompletionDelta::Reasoning(reasoning) => {
                        streamed_reasoning.push_str(&reasoning)
                    }
                }
            }
        }
        assert_eq!(streamed_text, "Hello");
        assert_eq!(streamed_reasoning, "hmm");
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
        let result = state.apply(
            &json!({"type": "error", "error": {"message": "overloaded"}}),
            CompletionLimits::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn stream_state_rejects_unknown_indexes_and_payload_overflow_before_append() {
        let mut state = StreamState::default();
        let unknown = state
            .apply(
                &json!({"type": "content_block_delta", "index": 500, "delta": {"type": "text_delta", "text": "x"}}),
                CompletionLimits::default(),
            )
            .unwrap_err()
            .to_string();
        assert!(unknown.contains("unknown content-block index 500"));
        assert!(state.blocks.is_empty());

        state
            .apply(
                &json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                CompletionLimits::default(),
            )
            .unwrap();
        let overflow = state
            .apply(
                &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "four"}}),
                CompletionLimits {
                    max_response_bytes: 3,
                    ..CompletionLimits::default()
                },
            )
            .unwrap_err()
            .to_string();
        assert!(overflow.contains("payload limit of 3 bytes"));
        assert_eq!(state.blocks[0]["text"], "");
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
