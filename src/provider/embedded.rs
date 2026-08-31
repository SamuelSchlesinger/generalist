//! Adapter for a model running inside the Generalist process.

use super::Provider;
use crate::error::Result;
#[cfg(feature = "embedded-mlx-native")]
pub use crate::local_inference::mlx::{
    ArrayDrift, ArrayFingerprint, EmbeddingCheckpoint, Layer0Checkpoint, Layer0CheckpointSpec,
    Layer0GdnCheckpoint, Layer0ProjectionCheckpoint, MlxBatchDecodeCheckpoint, MlxBatchDecodeRow,
    MlxCachedDecodeCheckpoint, MlxDecodeCheckpoint, MlxDecodeComparison, MlxDecodeProfile,
    MlxModelLoadCheckpoint, MlxMtpCheckpoint, MlxQwenModel, MlxQwenSession,
    MlxQwenSpeculativeSession, MlxRuntime, MlxSpeculativeStep, MlxTokenStep, NativeTensorProbe,
    NativeTensorSource, Nvfp4ProjectionCheckpoint, RmsNormCheckpoint,
};
pub use crate::local_inference::model_pack::{
    BlobDescriptor, ModelComponent, ModelPack, SafetensorInspection, Sha256Digest,
    TensorDescriptor, TensorMetadata,
};
#[cfg(feature = "embedded-mlx-native")]
pub use crate::local_inference::qwen::{
    render_qwen_prompt, QwenEmbeddedBackend, DEFAULT_EMBEDDED_MODEL,
};
use crate::local_inference::LocalInferenceWorker;
pub use crate::local_inference::{GenerationOutput, LocalCompletionRequest, LocalInferenceBackend};
use crate::types::{CompletionDelta, CompletionRequest, CompletionResponse};
use async_trait::async_trait;

/// Provider adapter for an in-process model backend.
///
/// The model initializer runs on a dedicated long-lived inference thread. The
/// same thread owns all backend state for the provider's lifetime, allowing an
/// implementation to retain weights, compiled graphs, recurrent state, and
/// validated prefix caches without exposing those details to the agent loop.
pub struct EmbeddedProvider {
    model: String,
    worker: LocalInferenceWorker,
}

impl EmbeddedProvider {
    /// Initialize an embedded backend on its permanent inference thread.
    ///
    /// On macOS the worker requests `QOS_CLASS_USER_INITIATED`; other Unix
    /// platforms currently retain their default scheduling policy.
    pub fn new<B, F>(model: impl Into<String>, initialize: F) -> Result<Self>
    where
        B: LocalInferenceBackend,
        F: FnOnce() -> Result<B> + Send + 'static,
    {
        Ok(Self {
            model: model.into(),
            worker: LocalInferenceWorker::spawn("generalist-inference", initialize)?,
        })
    }

    /// Load the specialized dense Qwen3.8 MLX backend on its permanent
    /// inference thread.
    #[cfg(feature = "embedded-mlx-native")]
    pub fn qwen(model: impl Into<String>) -> Result<Self> {
        let model = model.into();
        let initialized_model = model.clone();
        Self::new(model, move || QwenEmbeddedBackend::load(&initialized_model))
    }
}

#[async_trait(?Send)]
impl Provider for EmbeddedProvider {
    fn id(&self) -> &'static str {
        "embedded"
    }

    fn display_name(&self) -> &str {
        "Embedded"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        self.worker
            .complete_streaming(request, &mut |_| Ok(()))
            .await
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        self.worker.complete_streaming(request, on_delta).await
    }
}
