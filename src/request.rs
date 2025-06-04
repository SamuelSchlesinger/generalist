use crate::message::{ContentBlock, Message};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool definition for Claude to understand how to use a tool
///
/// Describes a tool that Claude can invoke during conversations,
/// including its name, purpose, and expected input format.
///
/// # Example
///
/// ```rust
/// use claude::ToolDef;
/// use serde_json::json;
///
/// let tool = ToolDef {
///     name: "calculator".to_string(),
///     description: "Performs basic arithmetic operations".to_string(),
///     input_schema: json!({
///         "type": "object",
///         "properties": {
///             "expression": {
///                 "type": "string",
///                 "description": "The mathematical expression to evaluate"
///             }
///         },
///         "required": ["expression"]
///     }),
/// };
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolDef {
    /// Tool name that Claude will use to reference it
    pub name: String,
    /// Human-readable description of what the tool does
    pub description: String,
    /// JSON Schema describing the required input format for the tool
    pub input_schema: Value,
}

/// Request structure for the Claude Messages API
///
/// `MessageRequest` contains all parameters needed to send a message to Claude.
/// This struct is serialized to JSON and sent to the Anthropic API.
///
/// # Required Fields
///
/// - `model`: The Claude model to use
/// - `messages`: The conversation history
/// - `tools`: Available tools (can be empty)
/// - `max_tokens`: Maximum tokens in response
///
/// # Optional Fields
///
/// - `system`: System prompt to guide behavior
/// - `temperature`: Controls randomness (0.0-1.0)
///
/// # Example
///
/// ```rust
/// use claude::{MessageRequest, Message, ContentBlock};
///
/// let request = MessageRequest {
///     model: "claude-3-haiku-20240307".to_string(),
///     messages: vec![
///         Message::user(vec![
///             ContentBlock::Text { text: "Hello!".to_string() }
///         ])
///     ],
///     tools: vec![],
///     max_tokens: 1024,
///     system: Some("You are a helpful assistant.".to_string()),
///     temperature: Some(0.7),
/// };
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct MessageRequest {
    /// The Claude model to use (e.g., "claude-3-haiku-20240307")
    pub model: String,
    /// Complete conversation history including the latest message
    pub messages: Vec<Message>,
    /// Tools that Claude can use during this conversation
    pub tools: Vec<ToolDef>,
    /// Maximum number of tokens Claude should generate in its response
    pub max_tokens: u32,
    /// Optional system prompt to guide Claude's behavior
    pub system: Option<String>,
    /// Optional temperature setting (0.0-1.0) to control randomness
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// Response from the Claude Messages API
///
/// `MessageResponse` contains Claude's response and metadata about the generation.
/// This struct is deserialized from the API's JSON response.
///
/// # Fields
///
/// - `id`: Unique message identifier
/// - `model`: The model that generated the response
/// - `role`: Always "assistant" for Claude's responses
/// - `content`: Array of content blocks (text, tool uses, etc.)
/// - `stop_reason`: Why generation stopped ("end_turn", "max_tokens", "stop_sequence")
/// - `stop_sequence`: The stop sequence that triggered stopping (if any)
/// - `usage`: Token usage statistics
///
/// # Example
///
/// ```rust
/// # use claude::{MessageResponse, ContentBlock};
/// # fn process_response(response: MessageResponse) {
/// // Extract text content from response
/// for block in &response.content {
///     match block {
///         ContentBlock::Text { text } => {
///             println!("Claude said: {}", text);
///         },
///         ContentBlock::ToolUse { name, input, id } => {
///             println!("Claude wants to use tool: {}", name);
///         },
///         _ => {}
///     }
/// }
///
/// // Check token usage
/// if let Some(usage) = response.usage {
///     println!("Tokens used: {} input, {} output",
///         usage.input_tokens, usage.output_tokens);
/// }
/// # }
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct MessageResponse {
    /// Unique identifier for the message
    pub id: String,
    /// The model that generated the response
    pub model: String,
    /// Role of the responder (always "assistant")
    pub role: String,
    /// Content blocks in the response
    pub content: Vec<ContentBlock>,
    /// Reason why generation stopped
    pub stop_reason: String,
    /// Stop sequence that triggered stopping (if any)
    pub stop_sequence: Option<String>,
    /// Token usage statistics for this request
    pub usage: Option<Usage>,
}

impl Into<Message> for &MessageResponse {
    /// Convert a MessageResponse into a Message for conversation history
    ///
    /// # Example
    ///
    /// ```rust
    /// # use claude::{MessageResponse, Message, ContentBlock};
    /// # let response = MessageResponse {
    /// #     id: "msg_123".to_string(),
    /// #     model: "claude-3-haiku-20240307".to_string(),
    /// #     role: "assistant".to_string(),
    /// #     content: vec![ContentBlock::Text { text: "Hello!".to_string() }],
    /// #     stop_reason: "end_turn".to_string(),
    /// #     stop_sequence: None,
    /// #     usage: None,
    /// # };
    /// let message: Message = (&response).into();
    /// assert_eq!(message.role, "assistant");
    /// assert_eq!(message.content.len(), response.content.len());
    /// ```
    fn into(self) -> Message {
        Message {
            role: self.role.clone(),
            content: self.content.clone(),
        }
    }
}

/// Token usage statistics for a Claude API request
///
/// Provides detailed token counts for billing and usage tracking.
///
/// # Fields
///
/// - `input_tokens`: Tokens in the input messages and system prompt
/// - `output_tokens`: Tokens generated by Claude in the response
/// - `cache_creation_input_tokens`: Tokens used for cache creation (if applicable)
/// - `cache_read_input_tokens`: Tokens read from cache (if applicable)
///
/// # Example
///
/// ```rust
/// # use claude::Usage;
/// # let usage = Usage {
/// #     input_tokens: 50,
/// #     output_tokens: 100,
/// #     cache_creation_input_tokens: None,
/// #     cache_read_input_tokens: None,
/// # };
/// println!("Total tokens used: {}", usage.input_tokens + usage.output_tokens);
///
/// if let Some(cached) = usage.cache_read_input_tokens {
///     println!("Tokens read from cache: {}", cached);
/// }
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens processed
    pub input_tokens: u32,
    /// Number of output tokens generated
    pub output_tokens: u32,
    /// Tokens used for cache creation (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Tokens read from cache (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}
