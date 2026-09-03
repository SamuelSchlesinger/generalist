//! Cerebras provider using its OpenAI-compatible chat-completions endpoint.

use crate::error::Result;
use crate::provider::{OpenAiProvider, Provider};
use crate::types::{CompletionDelta, CompletionRequest, CompletionResponse};
use async_trait::async_trait;

pub const DEFAULT_BASE_URL: &str = "https://api.cerebras.ai/v1";
pub const PUBLIC_MODELS_URL: &str = "https://api.cerebras.ai/public/v1/models";
pub const DEFAULT_MODEL: &str = "qwen-3.8-27b";
pub const GPT_OSS_120B_MODEL: &str = "gpt-oss-120b";
pub const GEMMA_4_31B_MODEL: &str = "gemma-4-31b";

/// Shared Cerebras models offered in the CLI picker. The first entry is the
/// default. Arbitrary model IDs remain available for dedicated endpoints.
pub const SUGGESTED_MODELS: &[&str] = &[DEFAULT_MODEL, GPT_OSS_120B_MODEL, GEMMA_4_31B_MODEL];

/// Cerebras has a distinct identity and credential even though its transport
/// is OpenAI-compatible. The inner profile also preserves Cerebras reasoning
/// across tool rounds and uses its documented completion-token field.
pub struct CerebrasProvider {
    inner: OpenAiProvider,
}

impl CerebrasProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        Ok(Self {
            inner: OpenAiProvider::for_cerebras(api_key, DEFAULT_BASE_URL.to_string(), model)?,
        })
    }
}

#[async_trait(?Send)]
impl Provider for CerebrasProvider {
    fn id(&self) -> &'static str {
        "cerebras"
    }

    fn display_name(&self) -> &str {
        "Cerebras"
    }

    fn model(&self) -> &str {
        self.inner.model()
    }

    async fn complete(&self, request: CompletionRequest<'_>) -> Result<CompletionResponse> {
        self.inner.complete(request).await
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        self.inner.complete_streaming(request, on_delta).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_distinct_persistence_identity_and_qwen_default() {
        let provider = CerebrasProvider::new("key".into(), DEFAULT_MODEL.into()).unwrap();

        assert_eq!(provider.id(), "cerebras");
        assert_eq!(provider.display_name(), "Cerebras");
        assert_eq!(provider.model(), "qwen-3.8-27b");
    }

    #[test]
    fn suggests_every_current_shared_tool_model_with_qwen_first() {
        assert_eq!(
            SUGGESTED_MODELS,
            [DEFAULT_MODEL, GPT_OSS_120B_MODEL, GEMMA_4_31B_MODEL]
        );
    }

    #[tokio::test]
    #[ignore = "live Cerebras catalog check; exercised by the scheduled workflow"]
    async fn suggested_models_match_live_shared_tool_catalog() {
        let response = reqwest::get(PUBLIC_MODELS_URL)
            .await
            .expect("Cerebras public models request");
        let status = response.status();
        let body = response.bytes().await.expect("Cerebras public models body");
        assert!(
            status.is_success(),
            "Cerebras public models request failed with {status}: {}",
            String::from_utf8_lossy(&body)
        );
        let catalog: serde_json::Value =
            serde_json::from_slice(&body).expect("Cerebras public models JSON");
        let models = catalog
            .get("data")
            .and_then(serde_json::Value::as_array)
            .expect("Cerebras public models data array");

        let is_supported_agent_model = |model: &&serde_json::Value| {
            !model
                .get("deprecated")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
                && model
                    .pointer("/capabilities/streaming")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
                && model
                    .pointer("/capabilities/tools")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
        };
        let mut live = models
            .iter()
            .filter(is_supported_agent_model)
            .filter_map(|model| model.get("id").and_then(serde_json::Value::as_str))
            .collect::<Vec<_>>();
        let mut suggested = SUGGESTED_MODELS.to_vec();
        live.sort_unstable();
        suggested.sort_unstable();

        assert_eq!(
            suggested, live,
            "update the curated Cerebras picker when its shared tool catalog changes"
        );
        for suggested in SUGGESTED_MODELS {
            let model = models
                .iter()
                .find(|model| {
                    model.get("id").and_then(serde_json::Value::as_str) == Some(suggested)
                })
                .unwrap_or_else(|| panic!("suggested Cerebras model '{suggested}' is missing"));
            assert_eq!(
                model
                    .pointer("/supported_parameters/max_completion_tokens")
                    .and_then(serde_json::Value::as_bool),
                Some(true),
                "suggested Cerebras model '{suggested}' lacks max_completion_tokens"
            );
        }
    }
}
