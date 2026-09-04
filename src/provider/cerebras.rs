//! Cerebras provider using its OpenAI-compatible chat-completions endpoint.

use crate::error::{Error, Result};
use crate::provider::openai::ResponseHeadersObserver;
use crate::provider::{OpenAiProvider, Provider};
use crate::types::{CompletionDelta, CompletionRequest, CompletionResponse};
use async_trait::async_trait;
use reqwest::header::HeaderMap;
use reqwest::StatusCode;
use serde_json::json;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub const DEFAULT_BASE_URL: &str = "https://api.cerebras.ai/v1";
pub const PUBLIC_MODELS_URL: &str = "https://api.cerebras.ai/public/v1/models";
pub const DEFAULT_MODEL: &str = "qwen-3.8-27b";
pub const GPT_OSS_120B_MODEL: &str = "gpt-oss-120b";
pub const GEMMA_4_31B_MODEL: &str = "gemma-4-31b";
/// Keep one agent round focused enough to reach a tool call. Callers can
/// override this with `Agent::max_tokens` / `--max-tokens`.
pub const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 8_192;

const LIMIT_TOKENS_MINUTE: &str = "x-ratelimit-limit-tokens-minute";
const REMAINING_TOKENS_MINUTE: &str = "x-ratelimit-remaining-tokens-minute";
const RESET_TOKENS_MINUTE: &str = "x-ratelimit-reset-tokens-minute";

/// Shared Cerebras models offered in the CLI picker. The first entry is the
/// default. Arbitrary model IDs remain available for dedicated endpoints.
pub const SUGGESTED_MODELS: &[&str] = &[DEFAULT_MODEL, GPT_OSS_120B_MODEL, GEMMA_4_31B_MODEL];

/// Cerebras has a distinct identity and credential even though its transport
/// is OpenAI-compatible. The inner profile also preserves Cerebras reasoning
/// across tool rounds and uses its documented completion-token field.
pub struct CerebrasProvider {
    inner: OpenAiProvider,
    token_meter: Arc<CerebrasTokenMeter>,
}

impl CerebrasProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        Self::with_base_url(api_key, DEFAULT_BASE_URL.to_string(), model)
    }

    fn with_base_url(api_key: String, base_url: String, model: String) -> Result<Self> {
        let token_meter = Arc::new(CerebrasTokenMeter::new());
        let observer_meter = Arc::clone(&token_meter);
        let observer: ResponseHeadersObserver = Arc::new(move |status, headers| {
            observer_meter.observe(status, headers);
        });
        Ok(Self {
            inner: OpenAiProvider::for_cerebras_with_observer(api_key, base_url, model, observer)?,
            token_meter,
        })
    }

    /// Seed the local token bucket before Cerebras has returned live quota
    /// headers. Every later response header remains authoritative, so this is
    /// useful for first-request pacing without freezing a tier-specific limit
    /// into the binary.
    pub fn with_tokens_per_minute(self, tokens_per_minute: u64) -> Result<Self> {
        if tokens_per_minute == 0 {
            return Err(Error::Other(
                "Cerebras tokens per minute must be greater than zero".to_string(),
            ));
        }
        self.token_meter.configure(tokens_per_minute);
        Ok(self)
    }

    fn with_request_defaults<'a>(&self, request: CompletionRequest<'a>) -> CompletionRequest<'a> {
        CompletionRequest {
            // Cerebras admission charges either this explicit budget or the
            // model's remaining maximum sequence length. Always send a focused
            // budget so custom/dedicated model IDs are metered predictably too.
            max_tokens: request.max_tokens.or(Some(DEFAULT_MAX_COMPLETION_TOKENS)),
            ..request
        }
    }

    fn estimated_admission_tokens(request: &CompletionRequest<'_>) -> u64 {
        // Cerebras admits against input tokens plus max_completion_tokens.
        // We do not ship each model's tokenizer, so use a deliberately
        // conservative byte estimate that also includes message/tool framing.
        let payload = json!({
            "system": request.system,
            "messages": request.messages,
            "tools": request.tools,
        });
        let bytes = serde_json::to_vec(&payload)
            .map(|payload| payload.len() as u64)
            .unwrap_or(0);
        let estimated_input = bytes.div_ceil(3).saturating_add(128);
        estimated_input.saturating_add(u64::from(request.max_tokens.unwrap_or(0)))
    }
}

#[derive(Debug)]
struct CerebrasTokenMeter {
    state: Mutex<TokenBucket>,
}

impl CerebrasTokenMeter {
    fn new() -> Self {
        Self {
            state: Mutex::new(TokenBucket::new()),
        }
    }

    fn state(&self) -> MutexGuard<'_, TokenBucket> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn configure(&self, tokens_per_minute: u64) {
        self.state()
            .set_limit(tokens_per_minute, Instant::now(), true);
    }

    fn observe(&self, status: StatusCode, headers: &HeaderMap) {
        self.state()
            .observe_headers(status, headers, Instant::now());
    }

    async fn acquire(&self, tokens: u64) -> Result<()> {
        loop {
            let reservation = self.state().reserve(tokens, Instant::now());
            match reservation {
                Reservation::Ready => return Ok(()),
                Reservation::Wait(duration) => tokio::time::sleep(duration).await,
                Reservation::TooLarge { requested, limit } => {
                    return Err(Error::Limit(format!(
                        "Cerebras estimates this request at {requested} tokens, above the live \
                         {limit} tokens/minute limit; reduce context or --max-tokens"
                    )));
                }
            }
        }
    }
}

#[derive(Debug)]
struct TokenBucket {
    limit: Option<u64>,
    available: f64,
    updated_at: Instant,
    blocked_until: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reservation {
    Ready,
    Wait(Duration),
    TooLarge { requested: u64, limit: u64 },
}

impl TokenBucket {
    fn new() -> Self {
        Self {
            limit: None,
            available: 0.0,
            updated_at: Instant::now(),
            blocked_until: None,
        }
    }

    fn set_limit(&mut self, limit: u64, now: Instant, fill: bool) {
        self.refill(now);
        let previous = self.limit;
        self.limit = Some(limit);
        self.available = if fill {
            limit as f64
        } else if previous.is_none() {
            0.0
        } else {
            self.available.min(limit as f64)
        };
        self.updated_at = now;
    }

    fn refill(&mut self, now: Instant) {
        let Some(limit) = self.limit else {
            self.updated_at = now;
            return;
        };
        let elapsed = now.saturating_duration_since(self.updated_at).as_secs_f64();
        self.available = (self.available + elapsed * limit as f64 / 60.0).min(limit as f64);
        self.updated_at = now;
    }

    fn reserve(&mut self, requested: u64, now: Instant) -> Reservation {
        self.refill(now);
        if let Some(blocked_until) = self.blocked_until {
            if blocked_until > now {
                return Reservation::Wait(blocked_until.duration_since(now));
            }
            self.blocked_until = None;
        }
        let Some(limit) = self.limit else {
            return Reservation::Ready;
        };
        if requested > limit {
            return Reservation::TooLarge { requested, limit };
        }
        if self.available >= requested as f64 {
            self.available -= requested as f64;
            return Reservation::Ready;
        }
        let missing = requested as f64 - self.available;
        let seconds = missing * 60.0 / limit as f64;
        Reservation::Wait(Duration::from_secs_f64(seconds.max(0.001)))
    }

    fn observe_headers(&mut self, status: StatusCode, headers: &HeaderMap, now: Instant) {
        let limit = integer_header(headers, LIMIT_TOKENS_MINUTE);
        if let Some(limit) = limit.filter(|limit| *limit > 0) {
            self.set_limit(limit, now, false);
        } else {
            self.refill(now);
        }

        let remaining = integer_header(headers, REMAINING_TOKENS_MINUTE);
        if let Some(remaining) = remaining {
            self.available = self
                .limit
                .map_or(remaining as f64, |limit| remaining.min(limit) as f64);
        }

        if status == StatusCode::TOO_MANY_REQUESTS {
            if remaining.is_none() {
                self.available = 0.0;
            }
            let reset = seconds_header(headers, RESET_TOKENS_MINUTE)
                .or_else(|| seconds_header(headers, "retry-after"));
            if let Some(reset) = reset {
                self.blocked_until = Some(now + reset);
            }
        } else {
            self.blocked_until = None;
        }
        self.updated_at = now;
    }
}

fn integer_header(headers: &HeaderMap, name: &str) -> Option<u64> {
    headers.get(name)?.to_str().ok()?.trim().parse().ok()
}

fn seconds_header(headers: &HeaderMap, name: &str) -> Option<Duration> {
    let raw = headers.get(name)?.to_str().ok()?.trim();
    let seconds = raw
        .strip_suffix('s')
        .unwrap_or(raw)
        .trim()
        .parse::<f64>()
        .ok()?;
    seconds
        .is_finite()
        .then(|| Duration::from_secs_f64(seconds.max(0.0)))
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
        let request = self.with_request_defaults(request);
        self.token_meter
            .acquire(Self::estimated_admission_tokens(&request))
            .await?;
        self.inner.complete(request).await
    }

    async fn complete_streaming(
        &self,
        request: CompletionRequest<'_>,
        on_delta: &mut dyn FnMut(CompletionDelta) -> Result<()>,
    ) -> Result<CompletionResponse> {
        let request = self.with_request_defaults(request);
        self.token_meter
            .acquire(Self::estimated_admission_tokens(&request))
            .await?;
        self.inner.complete_streaming(request, on_delta).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompletionLimits, Message};
    use reqwest::header::HeaderValue;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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

    #[test]
    fn applies_focused_default_token_budget_to_every_model_but_preserves_an_override() {
        let provider = CerebrasProvider::new("key".into(), DEFAULT_MODEL.into()).unwrap();
        let messages = vec![Message::user_text("hi")];
        let request = CompletionRequest {
            system: None,
            messages: &messages,
            tools: &[],
            max_tokens: None,
            limits: CompletionLimits::default(),
        };
        assert_eq!(
            provider.with_request_defaults(request).max_tokens,
            Some(DEFAULT_MAX_COMPLETION_TOKENS)
        );

        let request = CompletionRequest {
            system: None,
            messages: &messages,
            tools: &[],
            max_tokens: Some(12_345),
            limits: CompletionLimits::default(),
        };
        assert_eq!(
            provider.with_request_defaults(request).max_tokens,
            Some(12_345)
        );

        let dedicated = CerebrasProvider::new("key".into(), "team/custom-model".into()).unwrap();
        let request = CompletionRequest {
            system: None,
            messages: &messages,
            tools: &[],
            max_tokens: None,
            limits: CompletionLimits::default(),
        };
        assert_eq!(
            dedicated.with_request_defaults(request).max_tokens,
            Some(DEFAULT_MAX_COMPLETION_TOKENS)
        );
    }

    #[test]
    fn admission_estimate_includes_prompt_tools_and_completion_budget() {
        let provider = CerebrasProvider::new("key".into(), DEFAULT_MODEL.into()).unwrap();
        let messages = vec![Message::user_text("hello")];
        let tools = vec![crate::types::ToolDef {
            name: "example".into(),
            description: "An example tool".into(),
            input_schema: json!({"type": "object"}),
        }];
        let request = provider.with_request_defaults(CompletionRequest {
            system: Some("system"),
            messages: &messages,
            tools: &tools,
            max_tokens: None,
            limits: CompletionLimits::default(),
        });

        assert!(
            CerebrasProvider::estimated_admission_tokens(&request)
                > u64::from(DEFAULT_MAX_COMPLETION_TOKENS)
        );
    }

    #[test]
    fn token_bucket_paces_reservations_at_the_minute_rate() {
        let start = Instant::now();
        let mut bucket = TokenBucket::new();
        bucket.set_limit(60_000, start, true);

        assert_eq!(bucket.reserve(40_000, start), Reservation::Ready);
        assert_eq!(
            bucket.reserve(30_000, start),
            Reservation::Wait(Duration::from_secs(10))
        );
        assert_eq!(
            bucket.reserve(60_001, start),
            Reservation::TooLarge {
                requested: 60_001,
                limit: 60_000,
            }
        );
        assert_eq!(
            bucket.reserve(30_000, start + Duration::from_secs(10)),
            Reservation::Ready
        );
    }

    #[test]
    fn live_headers_replace_the_seed_and_429_reset_blocks_retries() {
        let start = Instant::now();
        let mut bucket = TokenBucket::new();
        bucket.set_limit(1_000_000, start, true);

        let mut headers = HeaderMap::new();
        headers.insert(LIMIT_TOKENS_MINUTE, HeaderValue::from_static("60000"));
        headers.insert(REMAINING_TOKENS_MINUTE, HeaderValue::from_static("10000"));
        bucket.observe_headers(StatusCode::OK, &headers, start);
        assert_eq!(bucket.limit, Some(60_000));
        assert_eq!(
            bucket.reserve(20_000, start),
            Reservation::Wait(Duration::from_secs(10))
        );

        headers.insert(RESET_TOKENS_MINUTE, HeaderValue::from_static("2.5s"));
        bucket.observe_headers(StatusCode::TOO_MANY_REQUESTS, &headers, start);
        assert_eq!(
            bucket.reserve(1, start),
            Reservation::Wait(Duration::from_millis(2_500))
        );
        assert_eq!(
            bucket.reserve(1, start + Duration::from_millis(2_500)),
            Reservation::Ready
        );

        let mut unknown_limit = TokenBucket::new();
        let mut retry_headers = HeaderMap::new();
        retry_headers.insert("retry-after", HeaderValue::from_static("3"));
        unknown_limit.observe_headers(StatusCode::TOO_MANY_REQUESTS, &retry_headers, start);
        assert_eq!(
            unknown_limit.reserve(1, start),
            Reservation::Wait(Duration::from_secs(3))
        );
    }

    #[test]
    fn configured_limit_rejects_zero() {
        let error = CerebrasProvider::new("key".into(), DEFAULT_MODEL.into())
            .unwrap()
            .with_tokens_per_minute(0)
            .err()
            .unwrap();
        assert!(error.to_string().contains("greater than zero"));
    }

    #[tokio::test]
    async fn transport_headers_update_the_live_meter() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = async {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = vec![0; 16 * 1024];
            let _ = stream.read(&mut request).await.unwrap();
            let body = r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nx-ratelimit-limit-tokens-minute: 12345\r\nx-ratelimit-remaining-tokens-minute: 2345\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        };
        let client = async {
            let provider =
                CerebrasProvider::with_base_url("key".into(), base_url, DEFAULT_MODEL.into())
                    .unwrap();
            let messages = [Message::user_text("hello")];
            provider
                .complete(CompletionRequest {
                    system: None,
                    messages: &messages,
                    tools: &[],
                    max_tokens: Some(100),
                    limits: CompletionLimits::default(),
                })
                .await
                .unwrap();
            let state = provider.token_meter.state();
            assert_eq!(state.limit, Some(12_345));
            assert!((state.available - 2_345.0).abs() < 1.0);
        };

        tokio::join!(server, client);
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
