use crate::Message;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatbotState {
    pub conversation_history: Vec<Message>,
    pub model: String,
    pub always_allow_tools: HashSet<String>,
    pub always_deny_tools: HashSet<String>,
    pub system_prompt: Option<String>,
    pub max_result_length: usize,
}

impl ChatbotState {
    pub fn new(model: String) -> Self {
        Self {
            conversation_history: Vec::new(),
            model,
            always_allow_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
            system_prompt: None,
            max_result_length: 200,
        }
    }

    pub fn from_conversation(conversation: Vec<Message>, model: String) -> Self {
        Self {
            conversation_history: conversation,
            model,
            always_allow_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
            system_prompt: None,
            max_result_length: 200,
        }
    }
}
