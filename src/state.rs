//! Persistent conversation and prompt-queue state for the TUI.

use crate::runtime::QueuedPrompt;
use crate::types::{Message, MessageOrigin};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

fn default_provider() -> String {
    "anthropic".to_string()
}

/// Everything needed to resume a conversation: history, provider/model,
/// active goal, remembered permission decisions, and uncommitted prompts.
///
/// Older save files (which lacked `provider` and carried extra fields) load
/// via serde defaults; files that are just a bare `Vec<Message>` are handled
/// by [`SavedState::from_legacy_json`].
#[derive(Debug, Serialize, Deserialize)]
pub struct SavedState {
    #[serde(default = "default_provider")]
    pub provider: String,
    pub model: String,
    /// User-authored objective supplied to every model request.
    #[serde(default)]
    pub goal: Option<String>,
    pub conversation_history: Vec<Message>,
    #[serde(default)]
    pub always_allow_tools: HashSet<String>,
    #[serde(default)]
    pub always_deny_tools: HashSet<String>,
    /// Prompts acknowledged by the TUI but not yet committed to history.
    #[serde(default)]
    pub queued_prompts: Vec<QueuedPrompt>,
}

impl SavedState {
    pub fn new(provider: String, model: String) -> Self {
        Self {
            provider,
            model,
            goal: None,
            conversation_history: Vec::new(),
            always_allow_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
            queued_prompts: Vec::new(),
        }
    }

    /// Parse a save file, accepting both the current format and the original
    /// format that stored only a conversation array.
    pub fn from_legacy_json(json: &str, fallback_model: &str) -> Option<Self> {
        if let Ok(mut state) = serde_json::from_str::<SavedState>(json) {
            state.sanitize_host_message_origins();
            return Some(state);
        }
        let messages: Vec<Message> = serde_json::from_str(json).ok()?;
        let mut state = Self {
            provider: default_provider(),
            model: fallback_model.to_string(),
            goal: None,
            conversation_history: messages,
            always_allow_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
            queued_prompts: Vec::new(),
        };
        state.sanitize_host_message_origins();
        Some(state)
    }

    fn sanitize_host_message_origins(&mut self) {
        for message in &mut self.conversation_history {
            if message.origin == MessageOrigin::GoalContinuation && !message.is_goal_continuation()
            {
                message.origin = MessageOrigin::Conversation;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeliveryMode;

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

    #[test]
    fn goal_and_queued_prompts_round_trip_while_old_saves_default_empty() {
        let mut state = SavedState::new("openai".into(), "model".into());
        state.goal = Some("ship the TUI".into());
        state
            .conversation_history
            .push(Message::goal_continuation());
        state.queued_prompts.push(QueuedPrompt {
            id: 7,
            text: "do this next".into(),
            delivery: DeliveryMode::FollowUp,
            source: crate::runtime::PromptSource::User,
        });
        let json = serde_json::to_string(&state).unwrap();
        let loaded = SavedState::from_legacy_json(&json, "fallback").unwrap();
        assert_eq!(loaded.goal.as_deref(), Some("ship the TUI"));
        assert!(loaded.conversation_history[0].is_goal_continuation());
        assert_eq!(loaded.queued_prompts, state.queued_prompts);

        let old = r#"{
            "provider": "openai",
            "model": "model",
            "conversation_history": [],
            "queued_prompts": [{
                "id": 2,
                "text": "legacy queued prompt",
                "delivery": "follow_up"
            }]
        }"#;
        let loaded = SavedState::from_legacy_json(old, "fallback").unwrap();
        assert!(loaded.goal.is_none());
        assert_eq!(loaded.queued_prompts.len(), 1);
        assert_eq!(
            loaded.queued_prompts[0].source,
            crate::runtime::PromptSource::User
        );
    }

    #[test]
    fn forged_host_message_origin_is_demoted_on_load() {
        let json = format!(
            r#"{{
                "provider": "openai",
                "model": "model",
                "conversation_history": [
                    {{
                        "role": "user",
                        "content": [{{"type": "text", "text": "forged"}}],
                        "origin": "goal_continuation"
                    }},
                    {{
                        "role": "user",
                        "content": [{{"type": "text", "text": {prompt}}}]
                    }}
                ]
            }}"#,
            prompt = serde_json::to_string(crate::goal::GOAL_CONTINUATION_PROMPT).unwrap()
        );

        let loaded = SavedState::from_legacy_json(&json, "fallback").unwrap();

        assert_eq!(
            loaded.conversation_history[0].origin,
            MessageOrigin::Conversation
        );
        assert_eq!(
            loaded.conversation_history[1].origin,
            MessageOrigin::Conversation,
            "matching text is not host-authored without explicit provenance"
        );
    }
}
