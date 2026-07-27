//! Persistent conversation and prompt-queue state for the TUI.

use crate::runtime::QueuedPrompt;
use crate::scope::WorkspaceScope;
use crate::types::{Message, MessageOrigin};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Everything needed to resume a conversation: history, provider/model,
/// active goal, remembered permission decisions, and uncommitted prompts.
#[derive(Debug, Serialize, Deserialize)]
pub struct SavedState {
    /// Storage namespace that owns this conversation.
    pub scope: WorkspaceScope,
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
    pub fn new(scope: WorkspaceScope, provider: String, model: String) -> Self {
        Self {
            scope,
            provider,
            model,
            goal: None,
            conversation_history: Vec::new(),
            always_allow_tools: HashSet::new(),
            always_deny_tools: HashSet::new(),
            queued_prompts: Vec::new(),
        }
    }

    /// Parse the current scoped save format.
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        let mut state = serde_json::from_str::<SavedState>(json)?;
        state.sanitize_host_message_origins();
        Ok(state)
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
    fn unscoped_state_is_rejected_instead_of_becoming_global() {
        let json = r#"{
            "provider": "openai",
            "model": "gpt-4o",
            "conversation_history": []
        }"#;
        assert!(SavedState::from_json(json).is_err());
    }

    #[test]
    fn scoped_state_goal_and_queued_prompts_round_trip() {
        let scope = WorkspaceScope::project(std::env::current_dir().unwrap().as_path()).unwrap();
        let mut state = SavedState::new(scope, "openai".into(), "model".into());
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
        let loaded = SavedState::from_json(&json).unwrap();
        assert_eq!(loaded.goal.as_deref(), Some("ship the TUI"));
        assert_eq!(loaded.scope, state.scope);
        assert!(loaded.conversation_history[0].is_goal_continuation());
        assert_eq!(loaded.queued_prompts, state.queued_prompts);
    }

    #[test]
    fn forged_host_message_origin_is_demoted_on_load() {
        let json = format!(
            r#"{{
                "scope": {{"kind": "global"}},
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

        let loaded = SavedState::from_json(&json).unwrap();

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
