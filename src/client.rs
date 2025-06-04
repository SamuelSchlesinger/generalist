use crate::error::{Error, Result};
use crate::message::{ContentBlock, Message};
use crate::request::{MessageRequest, MessageResponse};
use crate::tool::ToolRegistry;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::Value;
use std::collections::HashMap;

/// API endpoint for the Claude Messages API
pub const MESSAGES_ENDPOINT: &str = "https://api.anthropic.com/v1/messages";

/// Claude API client for interacting with Anthropic's AI models
///
/// The main entry point for using the Claude API. This struct handles authentication,
/// request formatting, and response parsing for interactions with Claude models.
///
/// # Example
///
/// ```rust
/// use claude::Claude;
///
/// let client = Claude::new(
///     "your-api-key".to_string(),
///     "claude-3-haiku-20240307".to_string()
/// );
/// ```
#[derive(Clone)]
pub struct Claude {
    /// Anthropic API key
    api_key: String,
    /// HTTP client for making API requests
    client: reqwest::Client,
    /// Default Claude model to use for requests
    model: String,
}

impl Claude {
    /// Create a new Claude API client
    ///
    /// # Arguments
    ///
    /// * `api_key` - Your Anthropic API key (starting with "sk-ant-api")
    /// * `model` - The Claude model to use
    ///
    /// # Available Models
    ///
    /// - `claude-3-opus-20240229` - Most capable model, best for complex tasks
    /// - `claude-3-sonnet-20240229` - Balanced performance and cost
    /// - `claude-3-haiku-20240307` - Fastest and most affordable
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::Claude;
    ///
    /// let client = Claude::new(
    ///     "your-api-key".to_string(),
    ///     "claude-3-haiku-20240307".to_string()
    /// );
    /// ```
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            model,
        }
    }

    /// Get the model name for this client
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a message to the Claude API
    ///
    /// Makes a direct request to the Anthropic Messages API using the provided MessageRequest.
    /// This is the low-level method for API interaction - most users should prefer
    /// [`run_conversation_turn`](Self::run_conversation_turn) for tool-enabled conversations.
    ///
    /// # Arguments
    ///
    /// * `request` - A complete MessageRequest containing messages, tools, and generation settings
    ///
    /// # Returns
    ///
    /// Returns a Result containing either the MessageResponse from Claude or an Error.
    ///
    /// # Errors
    ///
    /// - [`Error::Header`] - If the API key header can't be created
    /// - [`Error::Request`] - If the HTTP request fails
    /// - [`Error::Response`] - If the API returns a non-success status code
    /// - [`Error::Parse`] - If the API response can't be parsed
    ///
    /// # Example
    ///
    /// ```rust
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use claude::{Claude, MessageRequest, Message, ContentBlock};
    ///
    /// let client = Claude::new(
    ///     "your-api-key".to_string(),
    ///     "claude-3-haiku-20240307".to_string()
    /// );
    ///
    /// let request = MessageRequest {
    ///     model: client.model().to_string(),
    ///     messages: vec![
    ///         Message::user(vec![
    ///             ContentBlock::Text { text: "Hello!".to_string() }
    ///         ])
    ///     ],
    ///     tools: vec![],
    ///     max_tokens: 1024,
    ///     system: None,
    ///     temperature: None,
    /// };
    ///
    /// let response = client.next_message(request).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn next_message(&self, request: MessageRequest) -> Result<MessageResponse> {
        // According to Anthropic docs, we need three headers:
        let mut headers = HeaderMap::new();

        // 1. x-api-key
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key)
                .map_err(|_| Error::Header("Failed to create x-api-key header".to_string()))?,
        );

        // 2. content-type
        headers.insert("content-type", HeaderValue::from_static("application/json"));

        // 3. anthropic-version
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));

        let response = self
            .client
            .post(MESSAGES_ENDPOINT)
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            // Try to parse error details from response
            if let Ok(error_json) = serde_json::from_str::<Value>(&text) {
                if let Some(error_msg) = error_json
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                {
                    return Err(Error::Response(
                        error_msg.to_string(),
                        Some(status.as_u16()),
                    ));
                }
            }

            return Err(Error::Response(text, Some(status.as_u16())));
        }

        let response_text = response.text().await?;
        let message_response: MessageResponse = serde_json::from_str(&response_text)?;

        Ok(message_response)
    }

    /// Run a complete conversation turn with automatic tool handling
    ///
    /// This is the high-level method for having a tool-enabled conversation with Claude.
    /// It handles:
    /// - Sending your message to Claude
    /// - Processing any tool use requests from Claude
    /// - Executing tools (with permission checking)
    /// - Continuing the conversation until Claude produces a final response
    ///
    /// # Arguments
    ///
    /// * `user_message` - The user's message text
    /// * `tool_registry` - Registry containing available tools
    /// * `system_prompt` - Optional system prompt to guide Claude's behavior
    /// * `conversation_history` - Optional previous messages in the conversation
    /// * `max_iterations` - Maximum tool execution rounds (default: 10)
    ///
    /// # Returns
    ///
    /// Returns the final response from Claude after all tool executions are complete.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// use claude::{Claude, ToolRegistry};
    /// use std::sync::Arc;
    /// # use claude::tools::weather::WeatherTool;
    ///
    /// let client = Claude::new(
    ///     "your-api-key".to_string(),
    ///     "claude-3-haiku-20240307".to_string()
    /// );
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry.register(Arc::new(WeatherTool))?;
    ///
    /// let response = client.run_conversation_turn(
    ///     "What's the weather in London?",
    ///     &mut registry,
    ///     Some("You are a helpful weather assistant."),
    ///     None,
    ///     None
    /// ).await?;
    ///
    /// println!("Claude: {}", response);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run_conversation_turn(
        &self,
        user_message: &str,
        tool_registry: &mut ToolRegistry,
        system_prompt: Option<&str>,
        conversation_history: Option<Vec<Message>>,
        max_iterations: Option<usize>,
    ) -> Result<String> {
        let max_iterations = max_iterations.unwrap_or(10);
        let mut messages = conversation_history.unwrap_or_default();

        // Add the user's message
        messages.push(Message::user(vec![ContentBlock::Text {
            text: user_message.to_string(),
        }]));

        let mut iteration = 0;

        loop {
            if iteration >= max_iterations {
                return Err(Error::Other(format!(
                    "Maximum iterations ({}) reached without completion",
                    max_iterations
                )));
            }

            // Create request with current conversation state
            let request = MessageRequest {
                model: self.model.to_string(),
                messages: messages.clone(),
                tools: tool_registry.get_tool_defs(),
                max_tokens: 4096,
                system: system_prompt.map(|s| s.to_string()),
                temperature: None,
            };

            // Get Claude's response
            let response = self.next_message(request).await?;

            // Add Claude's response to conversation history
            messages.push((&response).into());

            // Check if Claude wants to use any tools
            let tool_uses = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { name, input, id } => {
                        Some((name.clone(), input.clone(), id.clone()))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();

            // If no tool uses, return the response
            if tool_uses.is_empty() {
                // Extract text content from response
                let text_content = response
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                return Ok(text_content);
            }

            // Execute tools and collect results
            let mut tool_results = Vec::new();
            for (tool_name, input, tool_use_id) in tool_uses {
                let result = tool_registry
                    .execute_tool(&tool_name, input, tool_use_id)
                    .await?;
                tool_results.push(result);
            }

            // Add tool results to conversation
            messages.push(Message::user(tool_results));

            iteration += 1;
        }
    }

    /// Get conversation summary statistics
    ///
    /// Analyzes a conversation history and returns statistics about messages,
    /// tool usage, and token counts.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use claude::{Claude, Message};
    /// # let client = Claude::new("api-key".to_string(), "model".to_string());
    /// # let messages = vec![];
    /// let stats = client.conversation_stats(&messages);
    /// println!("Total messages: {}", stats.get("total_messages").unwrap());
    /// println!("Tool uses: {}", stats.get("tool_uses").unwrap());
    /// ```
    pub fn conversation_stats(&self, messages: &[Message]) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        stats.insert("total_messages".to_string(), messages.len());

        let user_messages = messages.iter().filter(|m| m.role == "user").count();
        let assistant_messages = messages.iter().filter(|m| m.role == "assistant").count();

        stats.insert("user_messages".to_string(), user_messages);
        stats.insert("assistant_messages".to_string(), assistant_messages);

        let tool_uses = messages
            .iter()
            .flat_map(|m| &m.content)
            .filter(|block| matches!(block, ContentBlock::ToolUse { .. }))
            .count();

        let tool_results = messages
            .iter()
            .flat_map(|m| &m.content)
            .filter(|block| matches!(block, ContentBlock::ToolResult { .. }))
            .count();

        stats.insert("tool_uses".to_string(), tool_uses);
        stats.insert("tool_results".to_string(), tool_results);

        stats
    }
}
