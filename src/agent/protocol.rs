//! The executable tool-use/tool-result history invariant.

use crate::types::{ContentBlock, Message};
use std::collections::HashSet;

/// Whether every assistant tool use has exactly one result in the immediately
/// following user message, with no orphan results or role inversions.
///
/// This is the executable counterpart of `ToolHistoryIsValid` in
/// `spec/AsyncRuntime.tla`.
pub fn history_tool_protocol_is_valid(history: &[Message]) -> bool {
    let mut expected_results: Option<HashSet<&str>> = None;

    for message in history {
        let tool_uses = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let tool_results = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();

        if !tool_uses.is_empty() && message.role != "assistant" {
            return false;
        }
        if !tool_results.is_empty() && message.role != "user" {
            return false;
        }

        if let Some(expected) = expected_results.take() {
            let actual = tool_results.iter().copied().collect::<HashSet<_>>();
            if message.role != "user" || actual.len() != tool_results.len() || actual != expected {
                return false;
            }
        } else if !tool_results.is_empty() {
            return false;
        }

        if !tool_uses.is_empty() {
            let expected = tool_uses.iter().copied().collect::<HashSet<_>>();
            if expected.len() != tool_uses.len() {
                return false;
            }
            expected_results = Some(expected);
        }
    }

    expected_results.is_none()
}
