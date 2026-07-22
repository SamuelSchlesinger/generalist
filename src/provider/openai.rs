//! OpenAI-compatible chat-completions provider.
//!
//! Speaks the `/chat/completions` dialect implemented by OpenAI, Ollama,
//! Groq, Mistral, vLLM, LM Studio, and many others. Point `base_url` at any
//! compatible server (e.g. `http://localhost:11434/v1` for Ollama).

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, Message, StopReason, ToolDef, Usage,
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
    /// (id, name, arguments-fragments) per tool_call index.
    tool_calls: Vec<(String, String, String)>,
    finish_reason: Option<StopReason>,
    usage: Option<Usage>,
}

impl ChunkState {
    /// Apply one stream chunk; returns any user-visible text delta.
    fn apply(&mut self, chunk: &Value) -> Option<String> {
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
            .and_then(|c| c.first())?;
        if let Some(finish) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            self.finish_reason = Some(StopReason::parse(finish));
        }
        let delta = choice.get("delta")?;
        if let Some(calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for call in calls {
                let index = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                while self.tool_calls.len() <= index {
                    self.tool_calls
                        .push((String::new(), String::new(), String::new()));
                }
                let entry = &mut self.tool_calls[index];
                if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                    entry.0.push_str(id);
                }
                if let Some(name) = call.pointer("/function/name").and_then(|v| v.as_str()) {
                    entry.1.push_str(name);
                }
                if let Some(args) = call.pointer("/function/arguments").and_then(|v| v.as_str()) {
                    entry.2.push_str(args);
                }
            }
        }
        let text = delta
            .get("content")
            .and_then(|c| c.as_str())
            .filter(|t| !t.is_empty())?;
        self.text.push_str(text);
        Some(text.to_string())
    }

    fn into_response(self) -> CompletionResponse {
        let mut content = Vec::new();
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
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": Self::to_wire_messages(request.system, request.messages),
        });
        if !request.tools.is_empty() {
            body["tools"] = Value::Array(Self::to_wire_tools(request.tools));
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
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

        Self::from_wire_response(&serde_json::from_str(&text)?)
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(String),
    ) -> Result<CompletionResponse> {
        let mut body = json!({
            "model": self.model,
            "messages": Self::to_wire_messages(request.system, request.messages),
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        if !request.tools.is_empty() {
            body["tools"] = Value::Array(Self::to_wire_tools(request.tools));
        }

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
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
        let mut state = ChunkState::default();
        let mut done = false;
        while let Some(chunk) = response.chunk().await? {
            for payload in assembler.push(&chunk) {
                if payload.trim() == "[DONE]" {
                    done = true;
                    continue;
                }
                if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                    if let Some(text) = state.apply(&value) {
                        on_delta(text);
                    }
                }
            }
        }
        if let Some(payload) = assembler.finish() {
            if payload.trim() == "[DONE]" {
                done = true;
            } else if let Ok(value) = serde_json::from_str::<Value>(&payload) {
                if let Some(text) = state.apply(&value) {
                    on_delta(text);
                }
            }
        }
        // Some compatible servers omit [DONE]; accept a stream that at least
        // reported a finish_reason, otherwise treat it as cut short.
        if !done && state.finish_reason.is_none() {
            return Err(Error::Api {
                status: 0,
                message: "stream ended before completion".to_string(),
            });
        }
        Ok(state.into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn chunk_state_accumulates_streamed_text_and_tool_calls() {
        let chunks = [
            json!({"choices": [{"delta": {"content": "Hel"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"content": "lo"}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "weather", "arguments": "{\"ci"}}]}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ty\": \"Paris\"}"}}]}, "finish_reason": null}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 8}}),
        ];
        let mut state = ChunkState::default();
        let mut streamed = String::new();
        for chunk in &chunks {
            if let Some(text) = state.apply(chunk) {
                streamed.push_str(&text);
            }
        }
        assert_eq!(streamed, "Hello");
        let response = state.into_response();
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.content.len(), 2);
        match &response.content[1] {
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
