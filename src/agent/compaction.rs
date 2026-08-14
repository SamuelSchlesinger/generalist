//! Context compaction: summarize older history, keep recent turns verbatim.

use super::{Agent, AgentEvent};
use crate::error::Result;
use crate::types::{estimate_tokens, truncate_middle, CompletionRequest, ContentBlock, Message};

impl Agent {
    /// Summarize older history into a single message, keeping recent turns
    /// verbatim. Returns `Ok(false)` when there is nothing safe to compact.
    pub async fn compact(&mut self, on_event: &mut dyn FnMut(AgentEvent)) -> Result<bool> {
        const COMPACTION_INSTRUCTION: &str =
            "Summarize the conversation above for continuation in a fresh context. Preserve: \
             the user's goals and constraints; key findings and decisions with their \
             rationale; exact file paths, function names, commands, URLs, and error messages \
             that may be needed again; the current state of in-progress work and what \
             remains. Dense plain prose and lists; no preamble.";

        let Some(cut) = self.compaction_cut() else {
            return Ok(false);
        };
        let mut to_summarize: Vec<Message> = self.history[..cut]
            .iter()
            .map(summarizable_message)
            .collect();
        to_summarize.push(Message::user_text(COMPACTION_INSTRUCTION));
        let request = CompletionRequest {
            system: Some("You produce faithful, dense summaries of agent conversations."),
            messages: &to_summarize,
            tools: &[],
            max_tokens: Some(2_000),
            limits: self.completion_limits,
        };
        let response = self.provider.complete(request).await?;
        self.completion_limits.validate_response(&response)?;
        let summary = response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if summary.is_empty() {
            return Err(crate::error::Error::Other(
                "empty compaction summary".to_string(),
            ));
        }
        // The boundary may carry tool results whose tool_use partner is being
        // summarized away; render them as plain text so the remaining history
        // stays protocol-valid.
        for block in &mut self.history[cut].content {
            if let ContentBlock::ToolResult {
                content, is_error, ..
            } = block
            {
                let label = if *is_error == Some(true) {
                    "tool error"
                } else {
                    "tool result"
                };
                let text = format!("[{label} from a compacted call]\n{content}");
                *block = ContentBlock::Text { text };
            }
        }
        let replaced = cut;
        self.history.splice(
            0..cut,
            [Message::user_text(format!(
                "[Context summary — {} earlier messages were compacted]\n{}",
                replaced, summary
            ))],
        );
        self.history_revision = self.history_revision.wrapping_add(1);
        self.last_context_tokens = None;
        self.invalidate_estimated_tokens_cache();
        on_event(AgentEvent::Notice(format!(
            "Compacted {} messages into a summary (context ~{}k tokens).",
            replaced,
            estimate_tokens(&self.history) / 1000,
        )));
        self.emit_checkpoint(on_event);
        Ok(true)
    }

    /// Index of the first message to keep verbatim. Everything before it is
    /// summarized. A plain user turn (no tool results) is preferred so
    /// tool_use/tool_result pairs are never split; inside one long tool-use
    /// turn no such boundary exists, so any user message is accepted and
    /// [`Agent::compact`] converts its orphaned tool results to text.
    fn compaction_cut(&self) -> Option<usize> {
        let mut acc: u64 = 0;
        let mut budget_cut = None;
        for (i, message) in self.history.iter().enumerate().rev() {
            acc += estimate_tokens(std::slice::from_ref(message));
            if acc >= self.compaction_keep_recent_tokens {
                budget_cut = Some(i);
                break;
            }
        }
        let budget_cut = budget_cut?;
        let mut cut = budget_cut;
        while cut > 0 {
            let message = &self.history[cut];
            let plain_user = message.role == "user"
                && !message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolResult { .. }));
            if plain_user {
                break;
            }
            cut -= 1;
        }
        if cut < 2 {
            // Mid-turn fallback: cut at a tool-result user message rather
            // than letting one long turn outgrow the context window.
            cut = budget_cut;
            while cut > 0 && self.history[cut].role != "user" {
                cut -= 1;
            }
        }
        // Need at least two messages ahead of the boundary for a summary to
        // be worth a model call.
        (cut >= 2).then_some(cut)
    }
}

/// Provider-safe rendering of a message for the compaction request.
///
/// Tool blocks become plain text so the request is valid on providers that
/// reject tool_use/tool_result traffic when no tools are defined, and
/// reasoning blocks are dropped rather than replayed. A message that becomes
/// empty keeps a placeholder block so role alternation is preserved.
fn summarizable_message(message: &Message) -> Message {
    const MAX_TOOL_INPUT_CHARS: usize = 2_000;
    let mut content: Vec<ContentBlock> = message
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(ContentBlock::Text { text: text.clone() }),
            ContentBlock::ToolUse { name, input, .. } => Some(ContentBlock::Text {
                text: format!(
                    "[called tool {name} with {}]",
                    truncate_middle(&input.to_string(), MAX_TOOL_INPUT_CHARS)
                ),
            }),
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                let label = if *is_error == Some(true) {
                    "tool error"
                } else {
                    "tool result"
                };
                Some(ContentBlock::Text {
                    text: format!("[{label}]\n{content}"),
                })
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => None,
        })
        .collect();
    if content.is_empty() {
        content.push(ContentBlock::Text {
            text: "[reasoning elided]".to_string(),
        });
    }
    Message {
        role: message.role.clone(),
        content,
        origin: message.origin,
    }
}
