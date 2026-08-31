//! In-process model execution on one long-lived, scheduler-aware worker.
//!
//! MLX objects and caches are thread-affine in the embedded engine design. The
//! worker therefore owns the backend for its entire lifetime and communicates
//! with the async agent through bounded channels. This module deliberately has
//! no mandatory MLX dependency; the native backend plugs in behind
//! [`LocalInferenceBackend`] while [`model_pack`] owns the validated weight
//! discovery boundary.

#[cfg(feature = "embedded-mlx-native")]
pub mod mlx;
pub mod model_pack;
mod platform;
#[cfg(feature = "embedded-mlx-native")]
pub mod qwen;

use crate::error::{Error, Result};
use crate::types::{
    CompletionDelta, CompletionLimits, CompletionRequest, CompletionResponse, Message, ToolDef,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use tokio::sync::mpsc as async_mpsc;

const EVENT_BUFFER_CAPACITY: usize = 16;

/// Owned form of the provider request that can cross onto the inference
/// thread. It remains semantically identical to [`CompletionRequest`].
#[derive(Debug, Clone)]
pub struct LocalCompletionRequest {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub max_tokens: Option<u32>,
    pub limits: CompletionLimits,
}

impl From<CompletionRequest<'_>> for LocalCompletionRequest {
    fn from(request: CompletionRequest<'_>) -> Self {
        Self {
            system: request.system.map(str::to_owned),
            messages: request.messages.to_vec(),
            tools: request.tools.to_vec(),
            max_tokens: request.max_tokens,
            limits: request.limits,
        }
    }
}

/// Backend contract for a model whose weights, graph cache, and token cache
/// stay resident on the dedicated inference thread.
pub trait LocalInferenceBackend: Send + 'static {
    fn complete(
        &mut self,
        request: LocalCompletionRequest,
        output: &mut GenerationOutput<'_>,
    ) -> Result<CompletionResponse>;
}

/// Streaming and cancellation handle presented to the synchronous backend.
///
/// Backends must check [`GenerationOutput::is_cancelled`] between expensive
/// phases and token steps. [`GenerationOutput::emit`] also checks it so a
/// backend cannot continue streaming after the host rejects a delta.
pub struct GenerationOutput<'a> {
    cancelled: &'a AtomicBool,
    events: &'a async_mpsc::Sender<WorkerEvent>,
}

impl GenerationOutput<'_> {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn emit(&mut self, delta: CompletionDelta) -> Result<()> {
        if self.is_cancelled() {
            return Err(cancelled_error());
        }
        let (acknowledgement_tx, acknowledgement_rx) = mpsc::sync_channel(0);
        self.events
            .blocking_send(WorkerEvent::Delta {
                delta,
                acknowledgement: acknowledgement_tx,
            })
            .map_err(|_| cancelled_error())?;
        match acknowledgement_rx.recv() {
            Ok(true) => Ok(()),
            Ok(false) | Err(_) => Err(cancelled_error()),
        }
    }
}

enum WorkerCommand {
    Complete {
        request: LocalCompletionRequest,
        cancelled: Arc<AtomicBool>,
        events: async_mpsc::Sender<WorkerEvent>,
    },
    Shutdown,
}

enum WorkerEvent {
    Delta {
        delta: CompletionDelta,
        acknowledgement: mpsc::SyncSender<bool>,
    },
    Finished(Result<CompletionResponse>),
}

/// Serial executor that permanently owns one initialized local model.
pub(crate) struct LocalInferenceWorker {
    commands: mpsc::Sender<WorkerCommand>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl LocalInferenceWorker {
    /// Start the inference thread, apply the platform scheduling policy, and
    /// initialize the backend on that same thread.
    pub(crate) fn spawn<B, F>(thread_name: &str, initialize: F) -> Result<Self>
    where
        B: LocalInferenceBackend,
        F: FnOnce() -> Result<B> + Send + 'static,
    {
        let (commands_tx, commands_rx) = mpsc::channel();
        let (initialized_tx, initialized_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name(thread_name.to_owned())
            .spawn(move || {
                if let Err(error) = platform::configure_inference_thread() {
                    let _ = initialized_tx.send(Err(error));
                    return;
                }
                let mut backend = match initialize() {
                    Ok(backend) => backend,
                    Err(error) => {
                        let _ = initialized_tx.send(Err(error));
                        return;
                    }
                };
                if initialized_tx.send(Ok(())).is_err() {
                    return;
                }

                while let Ok(command) = commands_rx.recv() {
                    match command {
                        WorkerCommand::Complete {
                            request,
                            cancelled,
                            events,
                        } => {
                            let result = backend.complete(
                                request,
                                &mut GenerationOutput {
                                    cancelled: &cancelled,
                                    events: &events,
                                },
                            );
                            let _ = events.blocking_send(WorkerEvent::Finished(result));
                        }
                        WorkerCommand::Shutdown => break,
                    }
                }
            })
            .map_err(|error| Error::Other(format!("failed to start inference thread: {error}")))?;

        match initialized_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                commands: commands_tx,
                thread: Mutex::new(Some(worker)),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err(Error::Other(
                    "inference thread exited during initialization".to_string(),
                ))
            }
        }
    }

    /// Execute one completion without blocking the async agent task.
    pub(crate) async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cancel_on_drop = CancelOnDrop(Some(Arc::clone(&cancelled)));
        let (events_tx, mut events_rx) = async_mpsc::channel(EVENT_BUFFER_CAPACITY);
        self.commands
            .send(WorkerCommand::Complete {
                request: request.into(),
                cancelled,
                events: events_tx,
            })
            .map_err(|_| Error::Other("inference thread is not running".to_string()))?;

        while let Some(event) = events_rx.recv().await {
            match event {
                WorkerEvent::Delta {
                    delta,
                    acknowledgement,
                } => {
                    let callback_result = on_delta(delta);
                    let _ = acknowledgement.send(callback_result.is_ok());
                    callback_result?;
                }
                WorkerEvent::Finished(result) => {
                    cancel_on_drop.disarm();
                    return result;
                }
            }
        }
        Err(Error::Stream {
            message: "embedded inference stream ended before completion".to_string(),
            retryable: false,
        })
    }
}

impl Drop for LocalInferenceWorker {
    fn drop(&mut self) {
        let _ = self.commands.send(WorkerCommand::Shutdown);
        if let Ok(thread) = self.thread.get_mut() {
            if let Some(thread) = thread.take() {
                let _ = thread.join();
            }
        }
    }
}

struct CancelOnDrop(Option<Arc<AtomicBool>>);

impl CancelOnDrop {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(cancelled) = &self.0 {
            cancelled.store(true, Ordering::Release);
        }
    }
}

fn cancelled_error() -> Error {
    Error::Other("embedded inference was cancelled".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, StopReason, Usage};
    use std::sync::atomic::AtomicUsize;

    struct ScriptedBackend {
        calls: Arc<AtomicUsize>,
    }

    impl LocalInferenceBackend for ScriptedBackend {
        fn complete(
            &mut self,
            request: LocalCompletionRequest,
            output: &mut GenerationOutput<'_>,
        ) -> Result<CompletionResponse> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(request.system.as_deref(), Some("system"));
            assert_eq!(request.messages.len(), 1);
            assert_eq!(request.tools.len(), 1);
            assert_eq!(request.max_tokens, Some(32));
            assert_eq!(request.limits, CompletionLimits::default());
            output.emit(CompletionDelta::Reasoning("reason".to_string()))?;
            output.emit(CompletionDelta::Text("answer".to_string()))?;
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "answer".to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage {
                    input_tokens: 3,
                    output_tokens: 2,
                    ..Usage::default()
                }),
            })
        }
    }

    fn request<'a>(messages: &'a [Message], tools: &'a [ToolDef]) -> CompletionRequest<'a> {
        CompletionRequest {
            system: Some("system"),
            messages,
            tools,
            max_tokens: Some(32),
            limits: CompletionLimits::default(),
        }
    }

    #[tokio::test]
    async fn initializes_and_streams_on_the_worker() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend_calls = Arc::clone(&calls);
        let worker = LocalInferenceWorker::spawn("test-inference", move || {
            assert_eq!(thread::current().name(), Some("test-inference"));
            Ok(ScriptedBackend {
                calls: backend_calls,
            })
        })
        .unwrap();
        let messages = [Message::user_text("hello")];
        let tools = [ToolDef {
            name: "tool".to_string(),
            description: "description".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let mut deltas = Vec::new();
        let response = worker
            .complete_streaming(request(&messages, &tools), &mut |delta| {
                deltas.push(delta);
                Ok(())
            })
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            deltas,
            vec![
                CompletionDelta::Reasoning("reason".to_string()),
                CompletionDelta::Text("answer".to_string())
            ]
        );
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn callback_error_cancels_generation() {
        let worker = LocalInferenceWorker::spawn("test-cancellation", || {
            Ok(ScriptedBackend {
                calls: Arc::new(AtomicUsize::new(0)),
            })
        })
        .unwrap();
        let messages = [Message::user_text("hello")];
        let tools = [ToolDef {
            name: "tool".to_string(),
            description: "description".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let mut callback_count = 0;
        let error = worker
            .complete_streaming(request(&messages, &tools), &mut |_| {
                callback_count += 1;
                Err(Error::Limit("stop".to_string()))
            })
            .await
            .unwrap_err();

        assert_eq!(callback_count, 1);
        assert!(matches!(error, Error::Limit(message) if message == "stop"));
    }

    #[tokio::test]
    async fn dropping_completion_notifies_a_cooperative_backend() {
        struct CooperativeBackend {
            started: Arc<AtomicBool>,
            stopped: Arc<AtomicBool>,
        }

        impl LocalInferenceBackend for CooperativeBackend {
            fn complete(
                &mut self,
                _request: LocalCompletionRequest,
                output: &mut GenerationOutput<'_>,
            ) -> Result<CompletionResponse> {
                self.started.store(true, Ordering::Release);
                while !output.is_cancelled() {
                    thread::sleep(std::time::Duration::from_millis(1));
                }
                self.stopped.store(true, Ordering::Release);
                Err(cancelled_error())
            }
        }

        let started = Arc::new(AtomicBool::new(false));
        let stopped = Arc::new(AtomicBool::new(false));
        let backend_started = Arc::clone(&started);
        let backend_stopped = Arc::clone(&stopped);
        let worker = LocalInferenceWorker::spawn("test-drop-cancellation", move || {
            Ok(CooperativeBackend {
                started: backend_started,
                stopped: backend_stopped,
            })
        })
        .unwrap();
        let messages = [Message::user_text("hello")];
        let tools = [ToolDef {
            name: "tool".to_string(),
            description: "description".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let mut callback = |_| Ok(());
        let mut completion =
            Box::pin(worker.complete_streaming(request(&messages, &tools), &mut callback));

        tokio::select! {
            result = &mut completion => panic!("completion ended before cancellation: {result:?}"),
            _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
        }
        assert!(started.load(Ordering::Acquire));
        drop(completion);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !stopped.load(Ordering::Acquire) {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("backend did not observe cancellation after the future was dropped");
    }

    #[test]
    fn initialization_error_is_returned_to_the_caller() {
        struct NeverBackend;
        impl LocalInferenceBackend for NeverBackend {
            fn complete(
                &mut self,
                _request: LocalCompletionRequest,
                _output: &mut GenerationOutput<'_>,
            ) -> Result<CompletionResponse> {
                unreachable!()
            }
        }

        let result = LocalInferenceWorker::spawn::<NeverBackend, _>("failed-inference", || {
            Err(Error::Other("load failed".to_string()))
        });
        assert!(matches!(result, Err(Error::Other(message)) if message == "load failed"));
    }
}
