//! OpenRouter provider using its OpenAI-compatible chat-completions endpoint.

use crate::error::Result;
use crate::provider::{OpenAiProvider, Provider};
use crate::types::{CompletionDelta, CompletionRequest, CompletionResponse};
use async_trait::async_trait;

pub const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_MODEL: &str = "moonshotai/kimi-k3";
pub const GLM_5_3_FLASH_MODEL: &str = "z-ai/glm-5.3-flash";
pub const GEMINI_3_7_FLASH_MODEL: &str = "google/gemini-3.7-flash";
pub const SUGGESTED_MODELS: &[&str] = &[
    DEFAULT_MODEL,
    GLM_5_3_FLASH_MODEL,
    GEMINI_3_7_FLASH_MODEL,
    "qwen/qwen3.8-max",
];

/// OpenRouter gets its own stable provider ID so saved sessions reload with
/// `OPENROUTER_API_KEY`, rather than whichever key backs the generic OpenAI
/// adapter.
pub struct OpenRouterProvider {
    inner: OpenAiProvider,
}

impl OpenRouterProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        Self::with_base_url(api_key, DEFAULT_BASE_URL.to_string(), model)
    }

    fn with_base_url(api_key: String, base_url: String, model: String) -> Result<Self> {
        Ok(Self {
            inner: OpenAiProvider::new(api_key, base_url, model)?,
        })
    }
}

#[async_trait(?Send)]
impl Provider for OpenRouterProvider {
    fn id(&self) -> &'static str {
        "openrouter"
    }

    fn display_name(&self) -> &str {
        "OpenRouter"
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
    fn has_distinct_persistence_and_display_identity() {
        let provider =
            OpenRouterProvider::new("key".into(), DEFAULT_MODEL.into()).expect("provider");

        assert_eq!(provider.id(), "openrouter");
        assert_eq!(provider.display_name(), "OpenRouter");
        assert_eq!(provider.model(), "moonshotai/kimi-k3");
    }

    #[test]
    fn suggests_glm_gemini_and_the_benchmarked_qwen_model_without_changing_the_default() {
        assert_eq!(
            SUGGESTED_MODELS,
            [
                DEFAULT_MODEL,
                GLM_5_3_FLASH_MODEL,
                GEMINI_3_7_FLASH_MODEL,
                "qwen/qwen3.8-max"
            ]
        );
    }

    #[tokio::test]
    #[ignore = "live OpenRouter catalog check; exercised by the scheduled workflow"]
    async fn suggested_models_exist_and_support_tools_in_live_catalog() {
        let response = reqwest::get(format!("{DEFAULT_BASE_URL}/models"))
            .await
            .expect("OpenRouter models request");
        let status = response.status();
        let body = response.bytes().await.expect("OpenRouter models body");
        assert!(
            status.is_success(),
            "OpenRouter models request failed with {status}: {}",
            String::from_utf8_lossy(&body)
        );
        let catalog: serde_json::Value =
            serde_json::from_slice(&body).expect("OpenRouter models JSON");
        let models = catalog
            .get("data")
            .and_then(serde_json::Value::as_array)
            .expect("OpenRouter models data array");

        for suggested in SUGGESTED_MODELS {
            let model = models
                .iter()
                .find(|model| {
                    model.get("id").and_then(serde_json::Value::as_str) == Some(suggested)
                })
                .unwrap_or_else(|| panic!("suggested OpenRouter model '{suggested}' is missing"));
            let supports_tools = model
                .get("supported_parameters")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|parameters| {
                    parameters
                        .iter()
                        .any(|parameter| parameter.as_str() == Some("tools"))
                });
            assert!(
                supports_tools,
                "suggested OpenRouter model '{suggested}' does not advertise tool support"
            );
        }
    }
}
