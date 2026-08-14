//! Provider completion with streaming, host limits, and retry backoff.

use super::events::{emit_stream_aborted, StreamedKinds};
use super::{Agent, AgentEvent};
use crate::error::Result;
use crate::runtime::TurnControl;
use crate::types::{CompletionDelta, CompletionRequest};
use std::time::Duration;

impl Agent {
    pub(super) async fn complete_with_retry(
        &self,
        on_event: &mut dyn FnMut(AgentEvent),
        control: &mut TurnControl,
    ) -> Result<Option<(crate::types::CompletionResponse, StreamedKinds)>> {
        let tools = self.model_tool_defs();
        let system_prompt = self.effective_system_prompt();
        let mut attempt: u32 = 0;
        loop {
            on_event(AgentEvent::ApiCallStarted);
            let request = CompletionRequest {
                system: Some(system_prompt.as_ref()),
                messages: &self.history,
                tools: &tools,
                max_tokens: self.max_tokens,
                limits: self.completion_limits,
            };
            let mut streamed = StreamedKinds::default();
            let mut streamed_bytes = 0usize;
            let mut callback_limit_error = None::<String>;
            let maybe_result = {
                let mut forward = |delta: CompletionDelta| -> Result<()> {
                    if let Some(error) = &callback_limit_error {
                        return Err(crate::error::Error::Limit(error.clone()));
                    }
                    streamed_bytes = match self
                        .completion_limits
                        .checked_response_bytes(streamed_bytes, delta.len_bytes())
                    {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            // Keep the structured kind: store the bare
                            // message and re-wrap as `Limit` at each exit.
                            let message = match error {
                                crate::error::Error::Limit(message) => message,
                                other => other.to_string(),
                            };
                            callback_limit_error = Some(message.clone());
                            return Err(crate::error::Error::Limit(message));
                        }
                    };
                    match delta {
                        CompletionDelta::Text(text) => {
                            streamed.text = true;
                            on_event(AgentEvent::AssistantTextDelta(text));
                        }
                        CompletionDelta::Reasoning(reasoning) => {
                            streamed.reasoning = true;
                            on_event(AgentEvent::ReasoningDelta(reasoning));
                        }
                    }
                    Ok(())
                };
                let completion = self.provider.complete_streaming(request, &mut forward);
                tokio::pin!(completion);
                tokio::select! {
                    result = &mut completion => Some(result),
                    _ = control.cancelled() => None,
                }
            };
            // A custom provider may violate the callback contract and ignore
            // its error. The host still rejects the attempt before commit.
            let maybe_result = if let Some(error) = callback_limit_error {
                maybe_result.map(|_| Err(crate::error::Error::Limit(error)))
            } else {
                maybe_result
            };
            let Some(result) = maybe_result else {
                emit_stream_aborted(
                    on_event,
                    streamed,
                    "interrupted before the response was committed".to_string(),
                );
                on_event(AgentEvent::ApiCallFinished { usage: None });
                return Ok(None);
            };
            match result {
                Ok(response) => {
                    if let Err(error) = self.completion_limits.validate_response(&response) {
                        emit_stream_aborted(
                            on_event,
                            streamed,
                            format!("response was rejected before it was committed: {error}"),
                        );
                        on_event(AgentEvent::ApiCallFinished { usage: None });
                        return Err(error);
                    }
                    on_event(AgentEvent::ApiCallFinished {
                        usage: response.usage.clone(),
                    });
                    return Ok(Some((response, streamed)));
                }
                Err(e) if e.is_retryable() && attempt < self.max_retries => {
                    emit_stream_aborted(
                        on_event,
                        streamed,
                        format!("stream failed before it was committed: {e}"),
                    );
                    on_event(AgentEvent::ApiCallFinished { usage: None });
                    let delay_secs = retry_delay_secs(attempt, e.retry_after());
                    on_event(AgentEvent::Retrying {
                        attempt: attempt + 1,
                        max_retries: self.max_retries,
                        delay_secs,
                        error: e.to_string(),
                    });
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(delay_secs)) => {}
                        _ = control.cancelled() => return Ok(None),
                    }
                    attempt += 1;
                }
                Err(e) => {
                    emit_stream_aborted(
                        on_event,
                        streamed,
                        format!("stream failed before it was committed: {e}"),
                    );
                    on_event(AgentEvent::ApiCallFinished { usage: None });
                    return Err(e);
                }
            }
        }
    }
}

/// Delay before the next retry: exponential backoff (1, 2, 4, ...) with the
/// server's `Retry-After` as a lower bound.
///
/// The header is a floor, not an override: rate limits often clear sooner
/// than its worst-case value, so we never wait *less* than it asks, but also
/// never longer than our own schedule would. The 60s cap keeps a turn
/// interruptible and stops a hostile or buggy header from parking the loop.
pub(crate) fn retry_delay_secs(attempt: u32, retry_after: Option<u64>) -> u64 {
    (1u64 << attempt.min(20))
        .max(retry_after.unwrap_or(0))
        .min(60)
}
