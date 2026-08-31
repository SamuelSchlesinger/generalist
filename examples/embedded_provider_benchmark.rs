//! End-to-end latency and cancellation benchmark for the same adaptive
//! provider path used by `generalist --embedded`.

use generalist::provider::embedded::{EmbeddedProvider, DEFAULT_EMBEDDED_MODEL};
use generalist::{
    CompletionDelta, CompletionLimits, CompletionRequest, CompletionResponse, Message, Provider,
    ToolDef,
};
use ring::digest::{digest, SHA256};
use serde_json::{json, Value};
use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

const DEFAULT_OUTPUT_TOKENS: u32 = 192;
const DEFAULT_CANCEL_PROMPT_REPETITIONS: usize = 1024;
const DEFAULT_CANCEL_AFTER_MILLISECONDS: u64 = 50;
const DEFAULT_COMPLETION_ROUNDS: usize = 1;

struct CompletionMeasurement {
    response: CompletionResponse,
    duration: Duration,
    first_delta_duration: Duration,
    delta_count: usize,
    reasoning_bytes: usize,
    text_bytes: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let output_tokens = optional_u32(arguments.next(), DEFAULT_OUTPUT_TOKENS, "OUTPUT_TOKENS")?;
    let cancel_prompt_repetitions = optional_usize(
        arguments.next(),
        DEFAULT_CANCEL_PROMPT_REPETITIONS,
        "CANCEL_PROMPT_REPETITIONS",
    )?;
    let cancel_after_milliseconds = optional_u64(
        arguments.next(),
        DEFAULT_CANCEL_AFTER_MILLISECONDS,
        "CANCEL_AFTER_MILLISECONDS",
    )?;
    let completion_rounds = optional_usize(
        arguments.next(),
        DEFAULT_COMPLETION_ROUNDS,
        "COMPLETION_ROUNDS",
    )?;
    if arguments.next().is_some()
        || output_tokens < 2
        || cancel_prompt_repetitions == 0
        || cancel_after_milliseconds == 0
        || !(1..=5).contains(&completion_rounds)
    {
        return Err("usage: embedded_provider_benchmark [OUTPUT_TOKENS=192] [CANCEL_PROMPT_REPETITIONS=1024] [CANCEL_AFTER_MILLISECONDS=50] [COMPLETION_ROUNDS=1]".into());
    }

    let load_started = Instant::now();
    let provider = EmbeddedProvider::qwen(DEFAULT_EMBEDDED_MODEL)?;
    let load_duration = load_started.elapsed();

    let mut messages = vec![Message::user_text(
        "Implement a bounded concurrent Rust cache with single-flight loading, prompt cancellation, and deterministic tests. Think through the ownership and concurrency invariants carefully before giving the implementation.",
    )];
    let tools: [ToolDef; 0] = [];
    let system = "You are a systems programmer. Produce compilable Rust and scrutinize cancellation, ownership, and concurrency invariants.";
    let mut completion_results = Vec::with_capacity(completion_rounds);
    for round in 1..=completion_rounds {
        if round > 1 {
            messages.push(Message::user_text(format!(
                "Continue the implementation from the prior response. Re-check its invariants and fill in the next missing details. Continuation {round}."
            )));
        }
        let measurement =
            measure_completion(&provider, system, &messages, &tools, output_tokens).await?;
        if round > 1
            && measurement
                .response
                .usage
                .as_ref()
                .and_then(|usage| usage.cache_read_input_tokens)
                .unwrap_or(0)
                == 0
        {
            return Err("continuation completion did not reuse a prompt checkpoint".into());
        }
        let response_content = measurement.response.content.clone();
        completion_results.push(completion_json(round, &measurement)?);
        if round < completion_rounds {
            messages.push(Message::assistant(response_content));
        }
    }
    let first_completion = completion_results.remove(0);

    let cancel_prompt = format!(
        "Analyze this deliberately long sequence and do not answer until every item has been checked:\n{}",
        "ownership cancellation invariant deterministic cache verification; "
            .repeat(cancel_prompt_repetitions)
    );
    let cancel_messages = [Message::user_text(cancel_prompt)];
    let cancel_request = CompletionRequest {
        system: Some("Think carefully and inspect every item before answering."),
        messages: &cancel_messages,
        tools: &tools,
        max_tokens: Some(4096),
        limits: CompletionLimits::default(),
    };
    let mut cancellation_delta_count = 0usize;
    {
        let mut on_cancellation_delta = |_| {
            cancellation_delta_count += 1;
            Ok(())
        };
        let completion = provider.complete_streaming(cancel_request, &mut on_cancellation_delta);
        tokio::pin!(completion);
        tokio::select! {
            result = &mut completion => {
                return Err(format!(
                    "cancellation probe completed before its {} ms deadline: {result:?}",
                    cancel_after_milliseconds,
                ).into());
            }
            _ = tokio::time::sleep(Duration::from_millis(cancel_after_milliseconds)) => {}
        }
    }
    let shutdown_started = Instant::now();
    drop(provider);
    let cancellation_shutdown_duration = shutdown_started.elapsed();

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "protocol": {
                "model": DEFAULT_EMBEDDED_MODEL,
                "provider_path": "EmbeddedProvider::qwen using the production adaptive decoder",
                "requested_output_tokens": output_tokens,
                "completion_rounds": completion_rounds,
                "cancellation_prompt_repetitions": cancel_prompt_repetitions,
                "cancellation_requested_after_milliseconds": cancel_after_milliseconds,
                "cancellation_contract": "dropping the provider completion sets its cancellation flag; dropping the provider then joins the real inference worker",
                "steady_stream_rate_scope": "approximate: output tokens after the first emitted text/reasoning delta divided by remaining wall time",
            },
            "model_load_milliseconds": milliseconds(load_duration),
            "completion": first_completion,
            "continuations": completion_results,
            "cancellation": {
                "deltas_before_cancellation": cancellation_delta_count,
                "worker_shutdown_milliseconds": milliseconds(cancellation_shutdown_duration),
            },
        }))?
    );
    Ok(())
}

async fn measure_completion(
    provider: &EmbeddedProvider,
    system: &str,
    messages: &[Message],
    tools: &[ToolDef],
    output_tokens: u32,
) -> Result<CompletionMeasurement, Box<dyn Error>> {
    let request = CompletionRequest {
        system: Some(system),
        messages,
        tools,
        max_tokens: Some(output_tokens),
        limits: CompletionLimits::default(),
    };
    let started = Instant::now();
    let mut first_delta_duration = None;
    let mut delta_count = 0usize;
    let mut reasoning_bytes = 0usize;
    let mut text_bytes = 0usize;
    let response = provider
        .complete_streaming(request, &mut |delta| {
            first_delta_duration.get_or_insert_with(|| started.elapsed());
            delta_count += 1;
            match delta {
                CompletionDelta::Reasoning(text) => reasoning_bytes += text.len(),
                CompletionDelta::Text(text) => text_bytes += text.len(),
            }
            Ok(())
        })
        .await?;
    let duration = started.elapsed();
    let first_delta_duration = first_delta_duration
        .ok_or("embedded provider completed without emitting a reasoning or text delta")?;
    Ok(CompletionMeasurement {
        response,
        duration,
        first_delta_duration,
        delta_count,
        reasoning_bytes,
        text_bytes,
    })
}

fn completion_json(
    round: usize,
    measurement: &CompletionMeasurement,
) -> Result<Value, Box<dyn Error>> {
    let usage = measurement
        .response
        .usage
        .as_ref()
        .ok_or("embedded provider completion omitted token usage")?;
    if usage.output_tokens == 0 {
        return Err("embedded provider completion reported zero output tokens".into());
    }
    let post_first_delta = measurement
        .duration
        .saturating_sub(measurement.first_delta_duration);
    if post_first_delta.is_zero() {
        return Err("embedded provider emitted its first delta only after completion".into());
    }
    let serialized_content = serde_json::to_vec(&measurement.response.content)?;
    let content_sha256 = hex(digest(&SHA256, &serialized_content).as_ref());
    Ok(json!({
        "round": round,
        "total_milliseconds": milliseconds(measurement.duration),
        "first_delta_milliseconds": milliseconds(measurement.first_delta_duration),
        "end_to_end_output_tokens_per_second": usage.output_tokens as f64 / measurement.duration.as_secs_f64(),
        "approximate_post_first_delta_tokens_per_second": (usage.output_tokens.saturating_sub(1)) as f64 / post_first_delta.as_secs_f64(),
        "input_tokens": usage.input_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "output_tokens": usage.output_tokens,
        "delta_count": measurement.delta_count,
        "reasoning_bytes": measurement.reasoning_bytes,
        "text_bytes": measurement.text_bytes,
        "stop_reason": format!("{:?}", measurement.response.stop_reason),
        "content_sha256": content_sha256,
    }))
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn optional_u32(
    value: Option<std::ffi::OsString>,
    default: u32,
    name: &str,
) -> Result<u32, Box<dyn Error>> {
    optional(value, default, name)
}

fn optional_u64(
    value: Option<std::ffi::OsString>,
    default: u64,
    name: &str,
) -> Result<u64, Box<dyn Error>> {
    optional(value, default, name)
}

fn optional_usize(
    value: Option<std::ffi::OsString>,
    default: usize,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    optional(value, default, name)
}

fn optional<T>(
    value: Option<std::ffi::OsString>,
    default: T,
    name: &str,
) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .map(|value| {
            value
                .into_string()
                .map_err(|_| format!("{name} must be valid UTF-8"))?
                .parse::<T>()
                .map_err(|error| format!("invalid {name}: {error}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
        .map_err(Into::into)
}
