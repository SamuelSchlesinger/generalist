//! The turn loop: provider rounds, tool batches, steering, and checkpoints.

use super::events::cancelled_tool_result;
use super::{history_tool_protocol_is_valid, Agent, AgentEvent, TurnOutcome};
use crate::error::Result;
use crate::goal::UPDATE_GOAL_TOOL_NAME;
use crate::model_trace::AsyncModelAction;
use crate::runtime::TurnControl;
use crate::tool::ToolCallOutcome;
use crate::types::{truncate_middle, ContentBlock, Message, StopReason};
use serde_json::Value;

/// How one tool batch ended: the paired results plus whether cancellation or
/// a denial interrupted it.
struct BatchOutcome {
    results: Vec<ContentBlock>,
    cancelled: bool,
    denied: bool,
}

impl Agent {
    /// Run one user turn to completion (or pause/refusal/cap).
    ///
    /// On `Err`, the history still contains everything that happened up to
    /// the failure — including tool calls that already executed — so the
    /// conversation can continue after a retry or a model switch.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> Result<TurnOutcome> {
        self.begin_turn(user_input);
        let mut control = TurnControl::detached();
        self.run_started_turn(on_event, &mut control).await
    }

    /// Continue a turn whose initial user message has already been recorded.
    ///
    /// The asynchronous TUI supplies a real [`TurnControl`] so terminal input,
    /// steering, and cooperative cancellation keep progressing while this
    /// future is pending. Library callers normally use [`Agent::run_turn`].
    pub async fn run_started_turn(
        &mut self,
        on_event: &mut dyn FnMut(AgentEvent),
        control: &mut TurnControl,
    ) -> Result<TurnOutcome> {
        self.model_cancel_recorded.set(false);
        for iteration in 0..self.max_iterations {
            if control.is_cancelled() {
                return Ok(self.traced_outcome(TurnOutcome::Interrupted));
            }

            if self.context_tokens() > self.compaction_threshold_tokens {
                let compacted = {
                    let compact = self.compact(on_event);
                    tokio::pin!(compact);
                    tokio::select! {
                        result = &mut compact => Some(result),
                        _ = control.cancelled() => None,
                    }
                };
                match compacted {
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        on_event(AgentEvent::Notice(format!("Compaction failed: {error}")));
                    }
                    None => return Ok(self.traced_outcome(TurnOutcome::Interrupted)),
                }
            }

            let completion = self.complete_with_retry(on_event, control).await;
            let Some((response, streamed)) = (match completion {
                Ok(completion) => completion,
                Err(error) => {
                    self.trace_async(AsyncModelAction::ProviderFailure);
                    self.trace_async(AsyncModelAction::SettleTurn);
                    return Err(error);
                }
            }) else {
                return Ok(self.traced_outcome(TurnOutcome::Interrupted));
            };
            if let Some(usage) = &response.usage {
                self.last_context_tokens = Some(
                    usage.input_tokens
                        + usage.output_tokens
                        + usage.cache_read_input_tokens.unwrap_or(0)
                        + usage.cache_creation_input_tokens.unwrap_or(0),
                );
            }

            self.history
                .push(Message::assistant(response.content.clone()));

            if streamed.text || streamed.reasoning {
                let text = streamed.text.then(|| {
                    response
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                });
                let reasoning = streamed.reasoning.then(|| {
                    response
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n")
                });
                on_event(AgentEvent::StreamCommitted { text, reasoning });
            }

            if !streamed.reasoning {
                for block in &response.content {
                    if let ContentBlock::Thinking { thinking, .. } = block {
                        if !thinking.is_empty() {
                            on_event(AgentEvent::ReasoningDelta(thinking.clone()));
                        }
                    }
                }
            }

            if !streamed.text {
                for block in &response.content {
                    if let ContentBlock::Text { text } = block {
                        if !text.is_empty() {
                            on_event(AgentEvent::AssistantText(text.clone()));
                        }
                    }
                }
            }

            let tool_uses: Vec<_> = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { name, input, id } => {
                        Some((name.clone(), input.clone(), id.clone()))
                    }
                    _ => None,
                })
                .collect();

            if response.stop_reason == StopReason::Refusal {
                self.trace_async(AsyncModelAction::ProviderRefusal);
                if !tool_uses.is_empty() {
                    let results = tool_uses
                        .iter()
                        .map(|(_, _, id)| ContentBlock::ToolResult {
                            content: "This tool call was not executed because the model refused \
                                      the request."
                                .to_string(),
                            tool_use_id: id.clone(),
                            is_error: Some(true),
                        })
                        .collect();
                    self.history.push(Message::user(results));
                }
                on_event(AgentEvent::Notice(
                    "The model declined to continue with this request.".to_string(),
                ));
                self.emit_checkpoint(on_event);
                return Ok(self.traced_outcome(TurnOutcome::Refused));
            }

            if tool_uses.is_empty() {
                self.trace_async(AsyncModelAction::ProviderAnswer);
                if control.is_cancelled() {
                    self.emit_checkpoint(on_event);
                    return Ok(self.traced_outcome(TurnOutcome::Interrupted));
                }
                if iteration + 1 < self.max_iterations && self.commit_steering(control, on_event) {
                    continue;
                }
                if response.stop_reason == StopReason::MaxTokens {
                    on_event(AgentEvent::Notice(
                        "Response was cut off at the output-token limit and may be incomplete."
                            .to_string(),
                    ));
                }
                self.emit_checkpoint(on_event);
                return Ok(self.traced_outcome(TurnOutcome::Completed));
            }

            self.trace_async(AsyncModelAction::ProviderToolBatch {
                count: tool_uses.len(),
            });

            // A truncated response can contain tool calls whose JSON arguments
            // parsed but are silently incomplete. Executing them risks acting
            // on corrupt input (e.g. a half-written file edit), so fail every
            // call and let the model re-issue them with complete arguments.
            if response.stop_reason == StopReason::MaxTokens {
                on_event(AgentEvent::Notice(
                    "Response hit the output-token limit mid-tool-call; asking the model to re-issue."
                        .to_string(),
                ));
                let results = tool_uses
                    .into_iter()
                    .map(|(_, _, id)| ContentBlock::ToolResult {
                        content: "This tool call was not executed: the response hit the output \
                                  token limit, so its arguments may be truncated. Re-issue the \
                                  tool call with complete arguments."
                            .to_string(),
                        tool_use_id: id,
                        is_error: Some(true),
                    })
                    .collect::<Vec<_>>();
                for _ in &results {
                    self.trace_async(AsyncModelAction::CompleteTool);
                }
                self.history.push(Message::user(results));
                let steered =
                    iteration + 1 < self.max_iterations && self.commit_steering(control, on_event);
                if !steered {
                    if iteration + 1 < self.max_iterations {
                        self.trace_async(AsyncModelAction::ContinueAfterTools);
                    }
                    self.emit_checkpoint(on_event);
                }
                continue;
            }

            let BatchOutcome {
                results,
                mut cancelled,
                denied,
            } = self.execute_tool_batch(tool_uses, on_event, control).await;

            // Every tool_use gets a result — required by the APIs even when
            // a call was denied or cancellation interrupts the batch.
            self.history.push(Message::user(results));

            // The controller may have processed an interrupt while the final
            // tool or permission future became ready. Do not steer merely
            // because that future won the inner select race.
            cancelled |= control.is_cancelled();
            if cancelled {
                on_event(AgentEvent::Notice(
                    "Turn interrupted; unfinished tool calls were recorded as cancelled."
                        .to_string(),
                ));
                self.emit_checkpoint(on_event);
                return Ok(self.traced_outcome(TurnOutcome::Interrupted));
            }

            if iteration + 1 < self.max_iterations && self.commit_steering(control, on_event) {
                continue;
            }

            if denied {
                on_event(AgentEvent::Notice(
                    "Tool call denied — pausing so you can redirect.".to_string(),
                ));
                self.emit_checkpoint(on_event);
                return Ok(self.traced_outcome(TurnOutcome::PausedOnDenial));
            }
            if iteration + 1 < self.max_iterations {
                self.trace_async(AsyncModelAction::ContinueAfterTools);
            }
            self.emit_checkpoint(on_event);
        }

        on_event(AgentEvent::Notice(format!(
            "Stopped after {} tool-execution rounds without completing.",
            self.max_iterations
        )));
        Ok(self.traced_outcome(TurnOutcome::MaxIterationsReached))
    }

    /// Execute one batch of tool calls, pairing every `tool_use` with a
    /// result even when cancellation or a denial interrupts the batch.
    async fn execute_tool_batch(
        &mut self,
        tool_uses: Vec<(String, Value, String)>,
        on_event: &mut dyn FnMut(AgentEvent),
        control: &mut TurnControl,
    ) -> BatchOutcome {
        let mut results = Vec::with_capacity(tool_uses.len());
        let mut denied = false;
        let mut cancelled = false;
        let mut index = 0;
        while index < tool_uses.len() {
            if control.is_cancelled() {
                self.trace_cancel_request();
                for (_, _, id) in &tool_uses[index..] {
                    results.push(cancelled_tool_result(id.clone()));
                    self.trace_async(AsyncModelAction::RepairCancelledTool);
                }
                cancelled = true;
                break;
            }

            let (name, input, id) = tool_uses[index].clone();
            if self.builtin_code_mode_enabled()
                && !self.is_code_mode_call(&name)
                && name != UPDATE_GOAL_TOOL_NAME
            {
                // A compatible server/model may still emit a function name
                // that was never advertised (some models copy a
                // `tools.foo(...)` expression out of the prompt). Treat
                // that as a provider-protocol violation, never as tool
                // activity: pair it for history validity, explain the
                // required boundary, and let the next model round retry.
                let bridge_name = name.strip_prefix("tools.").unwrap_or(&name);
                let content = format!(
                    "The provider emitted undeclared native tool call `{name}`. It was not \
                         executed: code mode permits only the model-facing `python` capability \
                         tool (plus the host-owned `update_goal` control while a goal is active). \
                         Retry with a Python script using `import tools; \
                         tools.{bridge_name}(...)`."
                );
                on_event(AgentEvent::Notice(format!(
                    "Rejected undeclared native tool call `{name}` before execution; asking \
                         the model to retry through code mode."
                )));
                results.push(ContentBlock::ToolResult {
                    content,
                    tool_use_id: id,
                    is_error: Some(true),
                });
                index += 1;
                continue;
            }

            on_event(AgentEvent::ToolCallStarted {
                name: name.clone(),
                input: input.clone(),
            });

            let cancellation_id = id.clone();
            let maybe_result = if name == UPDATE_GOAL_TOOL_NAME {
                Some(self.execute_goal_update(input, id, on_event))
            } else {
                let execution = async {
                    if self.is_code_mode_call(&name) {
                        self.execute_code_mode(input, id, on_event).await
                    } else {
                        self.registry.execute_tool(&name, input, id).await
                    }
                };
                tokio::pin!(execution);
                tokio::select! {
                    result = &mut execution => Some(result),
                    _ = control.cancelled() => None,
                }
            };

            let Some(mut result) = maybe_result else {
                self.trace_cancel_request();
                let content =
                    "Tool execution was interrupted; its completion is unknown.".to_string();
                on_event(AgentEvent::ToolCallFinished {
                    name,
                    outcome: ToolCallOutcome::Cancelled,
                    content,
                });
                results.push(cancelled_tool_result(cancellation_id));
                self.trace_async(AsyncModelAction::RepairCancelledTool);
                for (_, _, id) in &tool_uses[index + 1..] {
                    results.push(cancelled_tool_result(id.clone()));
                    self.trace_async(AsyncModelAction::RepairCancelledTool);
                }
                cancelled = true;
                break;
            };
            if let ContentBlock::ToolResult { content, .. } = &mut result.block {
                *content = truncate_middle(content, self.max_tool_result_chars);
            }
            let content = match &result.block {
                ContentBlock::ToolResult { content, .. } => content.clone(),
                _ => String::new(),
            };
            on_event(AgentEvent::ToolCallFinished {
                name,
                outcome: result.outcome,
                content,
            });

            let was_denied = result.outcome == ToolCallOutcome::Denied;
            if was_denied {
                denied = true;
            } else {
                self.trace_async(AsyncModelAction::CompleteTool);
            }
            results.push(result.block);
            index += 1;

            if was_denied {
                // A denial stops the rest of the batch: the user is
                // redirecting, so later calls queued alongside the denied
                // one must not fire behind their back. Each still gets a
                // paired result, mirroring the cancellation path.
                for (_, _, remaining_id) in &tool_uses[index..] {
                    results.push(ContentBlock::ToolResult {
                        content: "This tool call was not executed because an earlier call in \
                                      the same batch was denied."
                            .to_string(),
                        tool_use_id: remaining_id.clone(),
                        is_error: Some(true),
                    });
                }
                break;
            }
        }

        BatchOutcome {
            results,
            cancelled,
            denied,
        }
    }

    /// Commit every queued steer at a history-valid boundary. There is no
    /// await between claiming the stable IDs, updating history, and committing
    /// the claim, so cancellation cannot strand a prompt mid-transition.
    pub(super) fn commit_steering(
        &mut self,
        control: &TurnControl,
        on_event: &mut dyn FnMut(AgentEvent),
    ) -> bool {
        let Some(claim) = control.claim_steering() else {
            return false;
        };
        let prompts = claim.prompts().to_vec();
        let text_blocks = prompts
            .iter()
            .map(|prompt| ContentBlock::Text {
                text: prompt.text.clone(),
            })
            .collect::<Vec<_>>();
        if let Some(last) = self
            .history
            .last_mut()
            .filter(|message| message.role == "user")
        {
            last.content.extend(text_blocks);
        } else {
            self.history.push(Message::user(text_blocks));
        }
        self.invalidate_estimated_tokens_cache();
        let prompts = claim.commit();
        on_event(AgentEvent::SteeringCommitted { prompts });
        self.emit_checkpoint(on_event);
        true
    }

    pub(super) fn emit_checkpoint(&self, on_event: &mut dyn FnMut(AgentEvent)) {
        debug_assert!(
            history_tool_protocol_is_valid(&self.history),
            "attempted to checkpoint history with an unpaired tool use/result"
        );
        on_event(AgentEvent::HistoryCheckpoint {
            history: self.history.clone(),
            goal: self.goal.clone(),
            context_tokens: self.context_tokens(),
        });
    }
}
