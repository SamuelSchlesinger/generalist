use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Message representation for Claude API interactions
///
/// Messages form the core of conversations with Claude. Each message has a role
/// (either "user" or "assistant") and contains one or more content blocks.
///
/// # Content Types
///
/// Messages can contain:
/// - **Text**: Simple text content
/// - **Tool Use**: Requests from Claude to use a specific tool
/// - **Tool Results**: Results from tool executions sent back to Claude
///
/// # Example
///
/// ```rust
/// use claude::{Message, ContentBlock};
///
/// // Create a simple user message
/// let user_msg = Message::user(vec![
///     ContentBlock::Text { text: "Hello, Claude!".to_string() }
/// ]);
///
/// // Create an assistant message with mixed content
/// let assistant_msg = Message::assistant(vec![
///     ContentBlock::Text { text: "I'll calculate that for you.".to_string() },
///     ContentBlock::ToolUse {
///         name: "calculator".to_string(),
///         input: serde_json::json!({"expression": "2+2"}),
///         id: "tool_123".to_string(),
///     }
/// ]);
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    /// The role of the message sender: "user" or "assistant"
    pub role: String,
    /// Content blocks that make up the message (text, tool usage, etc.)
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a new user message
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::{Message, ContentBlock};
    ///
    /// let msg = Message::user(vec![
    ///     ContentBlock::Text { text: "What's the weather?".to_string() }
    /// ]);
    /// assert_eq!(msg.role, "user");
    /// ```
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    /// Create a new assistant message
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::{Message, ContentBlock};
    ///
    /// let msg = Message::assistant(vec![
    ///     ContentBlock::Text { text: "I can help with that.".to_string() }
    /// ]);
    /// assert_eq!(msg.role, "assistant");
    /// ```
    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }

    /// Check if this message contains any tool use requests
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::{Message, ContentBlock};
    ///
    /// let msg = Message::assistant(vec![
    ///     ContentBlock::Text { text: "Let me calculate that.".to_string() },
    ///     ContentBlock::ToolUse {
    ///         name: "calculator".to_string(),
    ///         input: serde_json::json!({"x": 5}),
    ///         id: "tool_123".to_string(),
    ///     }
    /// ]);
    /// assert!(msg.has_tool_use());
    /// ```
    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
    }

    /// Extract all tool use requests from this message
    ///
    /// Returns a vector of tuples containing (tool_name, input, tool_id)
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::{Message, ContentBlock};
    ///
    /// let msg = Message::assistant(vec![
    ///     ContentBlock::ToolUse {
    ///         name: "weather".to_string(),
    ///         input: serde_json::json!({"city": "London"}),
    ///         id: "tool_123".to_string(),
    ///     }
    /// ]);
    ///
    /// let tools = msg.get_tool_uses();
    /// assert_eq!(tools.len(), 1);
    /// assert_eq!(tools[0].0, "weather");
    /// ```
    pub fn get_tool_uses(&self) -> Vec<(String, Value, String)> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { name, input, id } => {
                    Some((name.clone(), input.clone(), id.clone()))
                }
                _ => None,
            })
            .collect()
    }
}

/// Content block types used in messages
///
/// Represents different types of content that can appear in a message:
/// - Text content
/// - Tool usage requests from Claude
/// - Tool execution results
///
/// # Example
///
/// ```rust
/// use claude::ContentBlock;
///
/// // Text content
/// let text = ContentBlock::Text {
///     text: "Hello!".to_string()
/// };
///
/// // Tool use request
/// let tool_use = ContentBlock::ToolUse {
///     name: "calculator".to_string(),
///     input: serde_json::json!({"expression": "2+2"}),
///     id: "tool_123".to_string(),
/// };
///
/// // Tool result
/// let tool_result = ContentBlock::ToolResult {
///     content: "4".to_string(),
///     tool_use_id: "tool_123".to_string(),
///     is_error: None,
/// };
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content in a message
    Text {
        /// The text content
        text: String,
    },
    /// Tool usage request from Claude
    ToolUse {
        /// Name of the tool to use
        name: String,
        /// Input parameters for the tool
        input: Value,
        /// Unique identifier for this tool use
        id: String,
    },
    /// Result from executing a tool
    ToolResult {
        /// Content from the tool execution (must be a string)
        content: String,
        /// ID of the corresponding tool use request
        tool_use_id: String,
        /// Optional error flag if the tool execution failed
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

impl Into<ContentBlock> for String {
    /// Convert a string into a text content block
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::ContentBlock;
    ///
    /// let block: ContentBlock = "Hello, world!".to_string().into();
    /// match block {
    ///     ContentBlock::Text { text } => assert_eq!(text, "Hello, world!"),
    ///     _ => panic!("Expected text block"),
    /// }
    /// ```
    fn into(self) -> ContentBlock {
        ContentBlock::Text { text: self }
    }
}

impl Into<ContentBlock> for &str {
    /// Convert a string slice into a text content block
    ///
    /// # Example
    ///
    /// ```rust
    /// use claude::ContentBlock;
    ///
    /// let block: ContentBlock = "Hello!".into();
    /// match block {
    ///     ContentBlock::Text { text } => assert_eq!(text, "Hello!"),
    ///     _ => panic!("Expected text block"),
    /// }
    /// ```
    fn into(self) -> ContentBlock {
        ContentBlock::Text {
            text: self.to_string(),
        }
    }
}

/// Tool use information extracted from a content block
#[derive(Debug, Clone)]
pub struct ToolUse {
    /// Name of the tool
    pub name: String,
    /// Input parameters
    pub input: Value,
    /// Tool use identifier
    pub id: String,
}

impl TryInto<ToolUse> for &ContentBlock {
    type Error = Error;

    /// Try to convert a ContentBlock into a ToolUse
    ///
    /// # Errors
    ///
    /// Returns an error if the content block is not a ToolUse variant
    fn try_into(self) -> Result<ToolUse> {
        match self {
            ContentBlock::ToolUse { name, input, id } => Ok(ToolUse {
                name: name.clone(),
                input: input.clone(),
                id: id.clone(),
            }),
            _ => Err(Error::Other("Content block is not a ToolUse".to_string())),
        }
    }
}
