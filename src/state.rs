//! Persistent session state for the CLI.

use crate::types::Message;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

fn default_provider() -> String {
    "anthropic".to_string()
}

/// Everything needed to resume a conversation: history, provider/model, and
/// remembered permission decisions.
///
/// Older save files (which lacked `provider` and carried extra fields) load
/// via serde defaults; files that are just a bare `Vec<Message>` are handled
/// by [`SavedState::from_legacy_json`].
#[derive(Debug, Serialize, Deserialize)]
pub struct SavedState {
    #[serde(default = "default_provider")]
    pub provider: String,
    pub model: String,
    pub conversation_history: Vec<Message>,
    #[serde(default)]
    pub always_allow_tools: HashSet<String>,
    #[serde(default)]
    pub always_deny_tools: HashSet<String>,
}

impl SavedState {
    pub fn new(provider: String, model: String) -> Self {
        Self {
            provider,
            model,
            conversation_history: Vec::new(),
            always_allow_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
        }
    }

    /// Parse a save file, accepting both the current format and the original
    /// format that stored only a conversation array.
    pub fn from_legacy_json(json: &str, fallback_model: &str) -> Option<Self> {
        if let Ok(state) = serde_json::from_str::<SavedState>(json) {
            return Some(state);
        }
        let messages: Vec<Message> = serde_json::from_str(json).ok()?;
        Some(Self {
            provider: default_provider(),
            model: fallback_model.to_string(),
            conversation_history: messages,
            always_allow_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_current_format() {
        let json = r#"{
            "provider": "openai",
            "model": "gpt-4o",
            "conversation_history": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "always_allow_tools": ["calculator"],
            "always_deny_tools": []
        }"#;
        let state = SavedState::from_legacy_json(json, "fallback").unwrap();
        assert_eq!(state.provider, "openai");
        assert_eq!(state.conversation_history.len(), 1);
        assert!(state.always_allow_tools.contains("calculator"));
    }

    #[test]
    fn loads_pre_provider_format_with_extra_fields() {
        // The shape written by earlier versions of this program.
        let json = r#"{
            "model": "claude-3-7-sonnet-latest",
            "conversation_history": [],
            "always_allow_tools": [],
            "always_deny_tools": [],
            "system_prompt": "old",
            "max_result_length": 200
        }"#;
        let state = SavedState::from_legacy_json(json, "fallback").unwrap();
        assert_eq!(state.provider, "anthropic");
        assert_eq!(state.model, "claude-3-7-sonnet-latest");
    }

    #[test]
    fn loads_bare_conversation_array() {
        let json = r#"[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]"#;
        let state = SavedState::from_legacy_json(json, "some-model").unwrap();
        assert_eq!(state.model, "some-model");
        assert_eq!(state.conversation_history.len(), 1);
    }
}
