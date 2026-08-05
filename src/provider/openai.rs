//! OpenAI-compatible chat-completions provider.
//!
//! Speaks the `/chat/completions` dialect implemented by OpenAI, Ollama,
//! Groq, Mistral, vLLM, LM Studio, and many others. Point `base_url` at any
//! compatible server (e.g. `http://localhost:11434/v1` for Ollama).

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionDelta, CompletionLimits, CompletionRequest, CompletionResponse, ContentBlock,
    Message, StopReason, ToolDef, Usage,
};
use async_trait::async_trait;
use serde_json::{json, Value};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAiProvider {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAiProvider {
    pub fn new(api_key: String, base_url: String, model: String) -> Result<Self> {
        let base_url = base_url.trim_end_matches('/').to_string();
        Ok(Self {
            api_key,
            base_url,
            model,
            client: super::http_client()?,
        })
    }

    /// Convert neutral messages to the chat-completions message list.
    ///
    /// Tool results become `role: "tool"` messages; assistant tool-use blocks
    /// become `tool_calls`; thinking blocks are dropped (they are meaningless
    /// to other providers).
    fn to_wire_messages(system: Option<&str>, messages: &[Message]) -> Vec<Value> {
        let mut out = Vec::new();
        if let Some(system) = system {
            out.push(json!({"role": "system", "content": system}));
        }
        for message in messages {
            match message.role.as_str() {
                "assistant" => {
                    let text = message.text();
                    let tool_calls: Vec<Value> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::ToolUse { name, input, id } => Some(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": input.to_string(),
                                },
                            })),
                            _ => None,
                        })
                        .collect();
                    let mut msg = json!({"role": "assistant"});
                    msg["content"] = if text.is_empty() {
                        Value::Null
                    } else {
                        Value::String(text)
                    };
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = Value::Array(tool_calls);
                    }
                    out.push(msg);
                }
                _ => {
                    // Tool results must directly follow the assistant message
                    // that requested them, so emit them before any user text.
                    for block in &message.content {
                        if let ContentBlock::ToolResult {
                            content,
                            tool_use_id,
                            ..
                        } = block
                        {
                            out.push(json!({
                                "role": "tool",
                                "tool_call_id": tool_use_id,
                                "content": content,
                            }));
                        }
                    }
                    let text = message.text();
                    if !text.is_empty() {
                        out.push(json!({"role": "user", "content": text}));
                    }
                }
            }
        }
        out
    }

    fn to_wire_tools(tools: &[ToolDef]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                })
            })
            .collect()
    }

    fn build_body(&self, request: &CompletionRequest<'_>, streaming: bool) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": Self::to_wire_messages(request.system, request.messages),
        });
        if streaming {
            body["stream"] = json!(true);
            body["stream_options"] = json!({"include_usage": true});
        }
        if let Some(max_tokens) = request.max_tokens {
            let field = if self.base_url == DEFAULT_BASE_URL {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            body[field] = json!(max_tokens);
        }
        if !request.tools.is_empty() {
            body["tools"] = Value::Array(Self::to_wire_tools(request.tools));
        }
        body
    }

    /// Reasoning extensions used by common OpenAI-compatible servers.
    ///
    /// Qwen/vLLM/SGLang commonly use `reasoning_content`; Ollama's model
    /// message format uses `thinking`, and some compatible gateways use
    /// `reasoning`. Official OpenAI responses need not expose any of them.
    fn reasoning_text(value: &Value) -> Option<&str> {
        ["reasoning_content", "reasoning", "thinking"]
            .into_iter()
            .find_map(|field| value.get(field).and_then(Value::as_str))
            .filter(|text| !text.is_empty())
    }

    fn from_wire_response(value: &Value) -> Result<CompletionResponse> {
        let choice = value
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .ok_or_else(|| Error::Other("response missing choices".to_string()))?;
        let message = choice
            .get("message")
            .ok_or_else(|| Error::Other("choice missing message".to_string()))?;

        let mut content = Vec::new();
        if let Some(reasoning) = Self::reasoning_text(message) {
            content.push(ContentBlock::Thinking {
                thinking: reasoning.to_string(),
                signature: String::new(),
            });
        }
        if let Some(text) = message.get("content").and_then(|c| c.as_str()) {
            if !text.is_empty() {
                content.push(ContentBlock::Text {
                    text: text.to_string(),
                });
            }
        }
        if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for call in tool_calls {
                let id = call
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let function = call.get("function").cloned().unwrap_or(Value::Null);
                let name = function
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let arguments = function
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let input = serde_json::from_str(arguments)
                    .unwrap_or_else(|_| json!({"_unparsed_arguments": arguments}));
                content.push(ContentBlock::ToolUse { name, input, id });
            }
        }

        let stop_reason = choice
            .get("finish_reason")
            .and_then(|s| s.as_str())
            .map(StopReason::parse)
            .unwrap_or(StopReason::EndTurn);

        let usage = value.get("usage").map(|u| Usage {
            input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
            output_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        });

        Ok(CompletionResponse {
            content,
            stop_reason,
            usage,
        })
    }
}

/// Accumulates chat-completion stream chunks into a complete response.
#[derive(Default)]
struct ChunkState {
    text: String,
    reasoning: String,
    /// (id, name, arguments-fragments) per tool_call index.
    tool_calls: Vec<(String, String, String)>,
    finish_reason: Option<StopReason>,
    usage: Option<Usage>,
    retained_bytes: usize,
}

impl ChunkState {
    /// Apply one stream chunk; returns every inspectable delta it contains.
    fn apply(&mut self, chunk: &Value, limits: CompletionLimits) -> Result<Vec<CompletionDelta>> {
        let mut emitted = Vec::new();
        // The final usage chunk has an empty `choices` array.
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(Usage {
                input_tokens: usage
                    .get("prompt_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                output_tokens: usage
                    .get("completion_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            });
        }
        let choice = chunk
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first());
        let Some(choice) = choice else {
            return Ok(emitted);
        };
        if let Some(finish) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            self.finish_reason = Some(StopReason::parse(finish));
        }
        let Some(delta) = choice.get("delta") else {
            return Ok(emitted);
        };
        if let Some(calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for call in calls {
                let index = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                let index = usize::try_from(index).map_err(|_| {
                    Error::Other("provider tool-call index does not fit this host".to_string())
                })?;
                if index >= limits.max_tool_uses {
                    return Err(Error::Other(format!(
                        "provider tool-call index {index} exceeds the host limit of {} tool calls",
                        limits.max_tool_uses
                    )));
                }
                if index > self.tool_calls.len() {
                    return Err(Error::Other(format!(
                        "provider tool-call index {index} was out of sequence; expected at most {}",
                        self.tool_calls.len()
                    )));
                }
                if index == self.tool_calls.len() {
                    let projected_blocks = self
                        .tool_calls
                        .len()
                        .saturating_add(1)
                        .saturating_add(usize::from(!self.text.is_empty()))
                        .saturating_add(usize::from(!self.reasoning.is_empty()));
                    if projected_blocks > limits.max_content_blocks {
                        return Err(Error::Other(format!(
                            "provider completion exceeded the host limit of {} blocks",
                            limits.max_content_blocks
                        )));
                    }
                    self.tool_calls
                        .push((String::new(), String::new(), String::new()));
                }
                let id = call.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = call
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let arguments = call
                    .pointer("/function/arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                self.retained_bytes = limits.checked_response_bytes(
                    self.retained_bytes,
                    id.len()
                        .saturating_add(name.len())
                        .saturating_add(arguments.len()),
                )?;
                let entry = &mut self.tool_calls[index];
                if !id.is_empty() {
                    entry.0.push_str(id);
                }
                if !name.is_empty() {
                    entry.1.push_str(name);
                }
                if !arguments.is_empty() {
                    entry.2.push_str(arguments);
                }
            }
        }
        if let Some(reasoning) = OpenAiProvider::reasoning_text(delta) {
            if self.reasoning.is_empty()
                && self
                    .tool_calls
                    .len()
                    .saturating_add(usize::from(!self.text.is_empty()))
                    .saturating_add(1)
                    > limits.max_content_blocks
            {
                return Err(Error::Other(format!(
                    "provider completion exceeded the host limit of {} blocks",
                    limits.max_content_blocks
                )));
            }
            self.retained_bytes =
                limits.checked_response_bytes(self.retained_bytes, reasoning.len())?;
            self.reasoning.push_str(reasoning);
            emitted.push(CompletionDelta::Reasoning(reasoning.to_string()));
        }
        if let Some(text) = delta
            .get("content")
            .and_then(|c| c.as_str())
            .filter(|text| !text.is_empty())
        {
            if self.text.is_empty()
                && self
                    .tool_calls
                    .len()
                    .saturating_add(usize::from(!self.reasoning.is_empty()))
                    .saturating_add(1)
                    > limits.max_content_blocks
            {
                return Err(Error::Other(format!(
                    "provider completion exceeded the host limit of {} blocks",
                    limits.max_content_blocks
                )));
            }
            self.retained_bytes = limits.checked_response_bytes(self.retained_bytes, text.len())?;
            self.text.push_str(text);
            emitted.push(CompletionDelta::Text(text.to_string()));
        }
        Ok(emitted)
    }

    fn into_response(self) -> CompletionResponse {
        let mut content = Vec::new();
        if !self.reasoning.is_empty() {
            content.push(ContentBlock::Thinking {
                thinking: self.reasoning,
                signature: String::new(),
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text { text: self.text });
        }
        for (id, name, arguments) in self.tool_calls {
            let input = serde_json::from_str(&arguments)
                .unwrap_or_else(|_| json!({"_unparsed_arguments": arguments}));
            content.push(ContentBlock::ToolUse { name, input, id });
        }
        let has_tools = content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }));
        CompletionResponse {
            content,
            stop_reason: self.finish_reason.unwrap_or(if has_tools {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            }),
            usage: self.usage,
        }
    }
}

#[async_trait(?Send)]
impl Provider for OpenAiProvider {
    fn id(&self) -> &'static str {
        // Persistence key for this adapter. Do not use as a UI label: a
        // compatible endpoint may be Ollama, LM Studio, vLLM, etc.
        "openai"
    }

    fn display_name(&self) -> &str {
        if self.base_url == DEFAULT_BASE_URL {
            "OpenAI"
        } else {
            "OpenAI-compatible"
        }
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        let limits = request.limits;
        let body = self.build_body(&request, false);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
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

        let response = Self::from_wire_response(&serde_json::from_slice(&response_body)?)?;
        limits.validate_response(&response)?;
        Ok(response)
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        let limits = request.limits;
        let body = self.build_body(&request, true);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
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
        let mut state = ChunkState::default();
        let mut done = false;
        let mut received = 0usize;
        while let Some(chunk) = response.chunk().await? {
            crate::provider::account_wire_bytes(limits, &mut received, chunk.len())?;
            for payload in assembler.push(&chunk) {
                if payload.trim() == "[DONE]" {
                    done = true;
                    continue;
                }
                if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                    for delta in state.apply(&value, limits)? {
                        on_delta(delta)?;
                    }
                }
            }
        }
        if let Some(payload) = assembler.finish() {
            if payload.trim() == "[DONE]" {
                done = true;
            } else if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                for delta in state.apply(&value, limits)? {
                    on_delta(delta)?;
                }
            }
        }
        // Some compatible servers omit [DONE]; accept a stream that at least
        // reported a finish_reason, otherwise treat it as cut short.
        if !done && state.finish_reason.is_none() {
            return Err(Error::Api {
                status: 0,
                message: "stream ended before completion".to_string(),
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

    #[test]
    fn display_name_describes_service_or_protocol_without_changing_stable_id() {
        let official =
            OpenAiProvider::new("key".into(), DEFAULT_BASE_URL.into(), "gpt-4o".into()).unwrap();
        let compatible = OpenAiProvider::new(
            "unused".into(),
            "http://localhost:11434/v1".into(),
            "qwen3.6:35b-a3b".into(),
        )
        .unwrap();

        assert_eq!(official.id(), "openai");
        assert_eq!(official.display_name(), "OpenAI");
        assert_eq!(compatible.id(), "openai");
        assert_eq!(compatible.display_name(), "OpenAI-compatible");
    }

    #[test]
    fn request_body_uses_the_service_appropriate_explicit_token_field() {
        let provider =
            OpenAiProvider::new("key".into(), DEFAULT_BASE_URL.into(), "model".into()).unwrap();
        let messages = vec![Message::user_text("hi")];
        let request = CompletionRequest {
            system: None,
            messages: &messages,
            tools: &[],
            max_tokens: None,
            limits: CompletionLimits::default(),
        };
        let body = provider.build_body(&request, false);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("stream").is_none());

        let request = CompletionRequest {
            max_tokens: Some(32_000),
            ..request
        };
        let body = provider.build_body(&request, true);
        assert_eq!(body["max_completion_tokens"], 32_000);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);

        let compatible = OpenAiProvider::new(
            "key".into(),
            "http://localhost:11434/v1".into(),
            "model".into(),
        )
        .unwrap();
        let body = compatible.build_body(&request, false);
        assert_eq!(body["max_tokens"], 32_000);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn wire_messages_map_tool_results_to_tool_role() {
        let messages = vec![
            Message::user_text("run ls"),
            Message::assistant(vec![
                ContentBlock::Text {
                    text: "sure".into(),
                },
                ContentBlock::ToolUse {
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                    id: "call_1".into(),
                },
            ]),
            Message::user(vec![ContentBlock::ToolResult {
                content: "a.txt".into(),
                tool_use_id: "call_1".into(),
                is_error: None,
            }]),
        ];
        let wire = OpenAiProvider::to_wire_messages(Some("sys"), &messages);
        assert_eq!(wire[0]["role"], "system");
        assert_eq!(wire[1]["role"], "user");
        assert_eq!(wire[2]["role"], "assistant");
        assert_eq!(wire[2]["tool_calls"][0]["function"]["name"], "bash");
        assert_eq!(wire[3]["role"], "tool");
        assert_eq!(wire[3]["tool_call_id"], "call_1");
    }

    #[test]
    fn thinking_blocks_are_dropped() {
        let messages = vec![Message::assistant(vec![
            ContentBlock::Thinking {
                thinking: "hmm".into(),
                signature: "s".into(),
            },
            ContentBlock::Text {
                text: "answer".into(),
            },
        ])];
        let wire = OpenAiProvider::to_wire_messages(None, &messages);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0]["content"], "answer");
        assert!(wire[0].get("tool_calls").is_none());
    }

    #[test]
    fn parses_tool_call_response() {
        let value = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{\"city\": \"Paris\"}"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}
        });
        let resp = OpenAiProvider::from_wire_response(&value).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        match &resp.content[0] {
            ContentBlock::ToolUse { name, input, id } => {
                assert_eq!(name, "weather");
                assert_eq!(input["city"], "Paris");
                assert_eq!(id, "call_9");
            }
            other => panic!("expected tool use, got {:?}", other),
        }
        assert_eq!(resp.usage.unwrap().input_tokens, 7);
    }

    #[test]
    fn parses_openai_compatible_reasoning_without_mixing_it_into_text() {
        let value = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "reasoning_content": "inspect evidence",
                    "content": "final answer"
                }
            }]
        });
        let response = OpenAiProvider::from_wire_response(&value).unwrap();
        assert_eq!(
            response.content[0],
            ContentBlock::Thinking {
                thinking: "inspect evidence".to_string(),
                signature: String::new(),
            }
        );
        assert_eq!(
            response.content[1],
            ContentBlock::Text {
                text: "final answer".to_string(),
            }
        );
    }

    #[test]
    fn chunk_state_accumulates_reasoning_text_and_tool_calls() {
        let chunks = [
            json!({"choices": [{"delta": {"reasoning_content": "inspect "}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"reasoning": "the "}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"thinking": "evidence"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"content": "Hel"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"content": "lo"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "weather", "arguments": "{\"ci"}}]}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ty\": \"Paris\"}"}}]}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 8}}),
        ];
        let mut state = ChunkState::default();
        let mut streamed_text = String::new();
        let mut streamed_reasoning = String::new();
        for chunk in &chunks {
            for delta in state.apply(chunk, CompletionLimits::default()).unwrap() {
                match delta {
                    CompletionDelta::Text(text) => streamed_text.push_str(&text),
                    CompletionDelta::Reasoning(reasoning) => {
                        streamed_reasoning.push_str(&reasoning)
                    }
                }
            }
        }
        assert_eq!(streamed_text, "Hello");
        assert_eq!(streamed_reasoning, "inspect the evidence");
        let response = state.into_response();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.content.len(), 3);
        assert_eq!(
            response.content[0],
            ContentBlock::Thinking {
                thinking: "inspect the evidence".to_string(),
                signature: String::new(),
            }
        );
        match &response.content[2] {
            ContentBlock::ToolUse { name, input, id } => {
                assert_eq!(name, "weather");
                assert_eq!(id, "call_1");
                assert_eq!(input["city"], "Paris");
            }
            other => panic!("expected tool use, got {:?}", other),
        }
        assert_eq!(response.usage.unwrap().output_tokens, 8);
    }

    #[test]
    fn chunk_state_rejects_sparse_tool_indexes_without_allocating_placeholders() {
        let mut state = ChunkState::default();
        let chunk = json!({
            "choices": [{
                "delta": {"tool_calls": [{"index": 200, "function": {"name": "x"}}]},
                "finish_reason": null
            }]
        });
        let error = state
            .apply(&chunk, CompletionLimits::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("out of sequence"));
        assert!(state.tool_calls.is_empty());
    }

    #[test]
    fn chunk_state_rejects_payload_and_block_overflow_before_append() {
        let mut state = ChunkState::default();
        let text = json!({
            "choices": [{"delta": {"content": "four"}, "finish_reason": null}]
        });
        let error = state
            .apply(
                &text,
                CompletionLimits {
                    max_response_bytes: 3,
                    ..CompletionLimits::default()
                },
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("payload limit of 3 bytes"));
        assert!(state.text.is_empty());

        let error = state
            .apply(
                &text,
                CompletionLimits {
                    max_content_blocks: 0,
                    ..CompletionLimits::default()
                },
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("limit of 0 blocks"));
        assert!(state.text.is_empty());
    }

    #[test]
    fn malformed_arguments_do_not_panic() {
        let value = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "t", "arguments": "not json"}
                    }]
                }
            }]
        });
        let resp = OpenAiProvider::from_wire_response(&value).unwrap();
        match &resp.content[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert_eq!(input["_unparsed_arguments"], "not json");
            }
            other => panic!("expected tool use, got {:?}", other),
        }
    }
}
