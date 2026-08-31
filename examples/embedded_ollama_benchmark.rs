//! Paired, reasoning-preserving comparison of Generalist's embedded Qwen
//! provider and Ollama using the exact same raw prompt bytes.

use generalist::provider::embedded::{
    render_qwen_prompt, EmbeddedProvider, LocalCompletionRequest, DEFAULT_EMBEDDED_MODEL,
};
use generalist::{
    CompletionDelta, CompletionLimits, CompletionRequest, ContentBlock, Message, Provider, ToolDef,
};
use ring::digest::{digest, SHA256};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::env;
use std::error::Error;
use std::ffi::OsString;
#[cfg(target_os = "macos")]
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

const DEFAULT_OUTPUT_TOKENS: u32 = 128;
const DEFAULT_REPLICAS: usize = 1;
const DEFAULT_COOLDOWN_MILLISECONDS: u64 = 2_000;
const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const CONTEXT_TOKENS: u32 = 32_768;
const SEED: u64 = 1;
const MAX_IDLE_GPU_UTILIZATION_PERCENT: u64 = 5;
const GPU_OBSERVATION_SAMPLES: usize = 3;
const GPU_SAMPLE_INTERVAL_MILLISECONDS: u64 = 250;
const MIN_IDLE_MEMORY_FREE_PERCENT: f64 = 50.0;
const MIN_IDLE_CPU_PERCENT: f64 = 80.0;
const MIN_WHOLE_SAMPLE_CPU_IDLE_PERCENT: f64 = 80.0;
const MAX_BACKGROUND_PROCESS_CPU_PERCENT: f64 = 50.0;
const MAX_RESIDUAL_GPU_MEMORY_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const SYSTEM: &str = "You are a systems programmer. Produce compilable Rust and scrutinize cancellation, ownership, and concurrency invariants.";
const USER: &str = "Implement a bounded concurrent Rust cache with single-flight loading, prompt cancellation, and deterministic tests. Think through the ownership and concurrency invariants carefully before giving the implementation.";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Engine {
    Embedded,
    Ollama,
}

impl Engine {
    fn label(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::Ollama => "ollama",
        }
    }

    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "embedded" => Ok(Self::Embedded),
            "ollama" => Ok(Self::Ollama),
            _ => Err(format!("unknown benchmark engine `{value}`").into()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Measurement {
    total_milliseconds: f64,
    first_delta_milliseconds: f64,
    end_to_end_output_tokens_per_second: f64,
    approximate_post_first_delta_tokens_per_second: f64,
    input_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    output_tokens: u64,
    reasoning_bytes: usize,
    text_bytes: usize,
    stop_reason: String,
    content_sha256: String,
    ollama_server: Option<OllamaServerTiming>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OllamaServerTiming {
    total_milliseconds: f64,
    load_milliseconds: f64,
    prompt_eval_count: u64,
    prompt_eval_milliseconds: f64,
    eval_count: u64,
    eval_milliseconds: f64,
    decode_tokens_per_second: f64,
}

struct MeasurementData {
    input_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    output_tokens: u64,
    reasoning: String,
    text: String,
    stop_reason: String,
    ollama_server: Option<OllamaServerTiming>,
}

#[derive(Debug, Deserialize, Serialize)]
struct EngineSample {
    model_load_milliseconds: Option<f64>,
    cold: Measurement,
    warm: Measurement,
    shutdown_milliseconds: Option<f64>,
    cold_warm_content_match: bool,
}

#[derive(Debug, Serialize)]
struct ProcessLoad {
    pid: u32,
    cpu_percent: f64,
    resident_mib: f64,
    command: String,
}

#[derive(Debug, Serialize)]
struct SystemSnapshot {
    load_average: Option<[f64; 3]>,
    cpu_idle_percent: Option<f64>,
    memory_free_percent: Option<f64>,
    gpu_device_utilization_percent: Option<u64>,
    gpu_renderer_utilization_percent: Option<u64>,
    gpu_tiler_utilization_percent: Option<u64>,
    gpu_in_use_system_memory_bytes: Option<u64>,
    thermal_state: Option<String>,
    top_processes: Vec<ProcessLoad>,
    observation_errors: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuStatistics {
    device_utilization_percent: u64,
    renderer_utilization_percent: u64,
    tiler_utilization_percent: u64,
    in_use_system_memory_bytes: u64,
}

#[derive(Debug, Serialize)]
struct ObservedSample {
    replica: usize,
    order_position: usize,
    engine: Engine,
    environment_before: SystemSnapshot,
    whole_sample_cpu_idle_percent: Option<f64>,
    result: EngineSample,
    environment_after: SystemSnapshot,
}

#[derive(Clone, Copy)]
struct CpuTicks {
    user: u32,
    system: u32,
    idle: u32,
    nice: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let all_arguments = env::args_os().skip(1).collect::<Vec<_>>();
    if all_arguments.first().and_then(|value| value.to_str()) == Some("--child") {
        return child_main(&all_arguments[1..]).await;
    }
    let mut arguments = all_arguments.into_iter();
    let output_tokens = optional_u32(arguments.next(), DEFAULT_OUTPUT_TOKENS, "OUTPUT_TOKENS")?;
    let replicas = optional_usize(arguments.next(), DEFAULT_REPLICAS, "REPLICAS")?;
    let cooldown_milliseconds = optional_u64(
        arguments.next(),
        DEFAULT_COOLDOWN_MILLISECONDS,
        "COOLDOWN_MILLISECONDS",
    )?;
    let output_path = arguments.next().map(PathBuf::from);
    if arguments.next().is_some()
        || output_tokens < 2
        || !(1..=7).contains(&replicas)
        || cooldown_milliseconds > 60_000
    {
        return Err("usage: embedded_ollama_benchmark [OUTPUT_TOKENS=128] [REPLICAS=1..7] [COOLDOWN_MILLISECONDS=2000] [OUTPUT_PATH]".into());
    }

    let ollama_base_url = env::var("GENERALIST_OLLAMA_BASE_URL")
        .unwrap_or_else(|_| DEFAULT_OLLAMA_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(300))
        .build()?;
    let messages = vec![Message::user_text(USER)];
    let tools: Vec<ToolDef> = Vec::new();
    let local_request = LocalCompletionRequest {
        system: Some(SYSTEM.to_string()),
        messages: messages.clone(),
        tools: tools.clone(),
        max_tokens: Some(output_tokens),
        limits: CompletionLimits::default(),
    };
    let raw_prompt = render_qwen_prompt(&local_request)?;
    let prompt_sha256 = sha256(raw_prompt.as_bytes());
    let ollama_model = inspect_ollama_model(&client, &ollama_base_url).await?;
    unload_ollama(&client, &ollama_base_url).await?;

    let mut observations = Vec::with_capacity(replicas * 2);
    let mut embedded_rates = Vec::with_capacity(replicas);
    let mut ollama_rates = Vec::with_capacity(replicas);
    let mut paired_speedups = Vec::with_capacity(replicas);
    let mut embedded_end_to_end_rates = Vec::with_capacity(replicas);
    let mut ollama_end_to_end_rates = Vec::with_capacity(replicas);
    let mut paired_end_to_end_speedups = Vec::with_capacity(replicas);
    let mut embedded_first_delta_milliseconds = Vec::with_capacity(replicas);
    let mut ollama_first_delta_milliseconds = Vec::with_capacity(replicas);
    let mut cross_engine_content_matches = Vec::with_capacity(replicas);
    let mut first_sample = true;

    for replica_index in 0..replicas {
        let order = if replica_index.is_multiple_of(2) {
            [Engine::Embedded, Engine::Ollama]
        } else {
            [Engine::Ollama, Engine::Embedded]
        };
        let mut embedded_result = None;
        let mut ollama_result = None;
        for (order_index, engine) in order.into_iter().enumerate() {
            if !first_sample && cooldown_milliseconds != 0 {
                std::thread::sleep(Duration::from_millis(cooldown_milliseconds));
            }
            first_sample = false;
            if engine == Engine::Embedded {
                unload_ollama(&client, &ollama_base_url).await?;
            }
            let environment_before = system_snapshot();
            if let Some(reason) = pre_sample_environment_rejection(&environment_before) {
                return Err(format!(
                    "refusing to run {} replica {} from a contaminated environment: {reason}",
                    engine.label(),
                    replica_index + 1,
                )
                .into());
            }
            let (result, whole_sample_cpu_idle_percent) =
                run_isolated_sample(engine, output_tokens, &ollama_base_url)?;
            let environment_after = system_snapshot();
            match engine {
                Engine::Embedded => {
                    embedded_rates.push(result.warm.approximate_post_first_delta_tokens_per_second);
                    embedded_end_to_end_rates.push(result.warm.end_to_end_output_tokens_per_second);
                    embedded_first_delta_milliseconds.push(result.warm.first_delta_milliseconds);
                    embedded_result = Some(result.cold.content_sha256.clone());
                }
                Engine::Ollama => {
                    ollama_rates.push(result.warm.approximate_post_first_delta_tokens_per_second);
                    ollama_end_to_end_rates.push(result.warm.end_to_end_output_tokens_per_second);
                    ollama_first_delta_milliseconds.push(result.warm.first_delta_milliseconds);
                    ollama_result = Some(result.cold.content_sha256.clone());
                }
            }
            observations.push(ObservedSample {
                replica: replica_index + 1,
                order_position: order_index + 1,
                engine,
                environment_before,
                whole_sample_cpu_idle_percent,
                result,
                environment_after,
            });
        }
        let embedded_rate = *embedded_rates
            .last()
            .ok_or("replica omitted embedded result")?;
        let ollama_rate = *ollama_rates.last().ok_or("replica omitted Ollama result")?;
        paired_speedups.push(embedded_rate / ollama_rate);
        paired_end_to_end_speedups.push(
            embedded_end_to_end_rates
                .last()
                .ok_or("replica omitted embedded end-to-end result")?
                / ollama_end_to_end_rates
                    .last()
                    .ok_or("replica omitted Ollama end-to-end result")?,
        );
        cross_engine_content_matches.push(embedded_result == ollama_result);
    }
    unload_ollama(&client, &ollama_base_url).await?;

    let embedded_spread = spread(&embedded_rates);
    let ollama_spread = spread(&ollama_rates);
    let environment_preconditions_verified = observations
        .iter()
        .all(|sample| pre_sample_environment_rejection(&sample.environment_before).is_none());
    let residency_isolation_verified = observations.iter().all(|sample| {
        sample
            .environment_after
            .gpu_in_use_system_memory_bytes
            .is_some_and(|value| value <= MAX_RESIDUAL_GPU_MEMORY_BYTES)
    });
    let post_sample_environment_verified = observations.iter().all(|sample| {
        sample.environment_after.observation_errors.is_empty()
            && sample.environment_after.thermal_state.as_deref() == Some("nominal")
    });
    let whole_sample_environment_verified = observations.iter().all(|sample| {
        sample
            .whole_sample_cpu_idle_percent
            .is_some_and(|value| value >= MIN_WHOLE_SAMPLE_CPU_IDLE_PERCENT)
    });
    let embedded_cold_warm_content_stable = observations
        .iter()
        .filter(|sample| sample.engine == Engine::Embedded)
        .all(|sample| sample.result.cold_warm_content_match);
    let ollama_cold_warm_content_stable = observations
        .iter()
        .filter(|sample| sample.engine == Engine::Ollama)
        .all(|sample| sample.result.cold_warm_content_match);
    let evidence_grade = if replicas < 3 {
        "screen-only"
    } else if !environment_preconditions_verified
        || !whole_sample_environment_verified
        || !post_sample_environment_verified
        || !residency_isolation_verified
    {
        "replicated-but-environment-unverified"
    } else if embedded_spread > 1.05 || ollama_spread > 1.05 {
        "replicated-but-unstable"
    } else {
        "replicated-stable"
    };
    let output = json!({
        "schema_version": 1,
        "protocol": {
            "model": DEFAULT_EMBEDDED_MODEL,
            "ollama_model": ollama_model,
            "prompt_source": "Generalist production Qwen renderer; exact bytes sent to Ollama /api/generate with raw=true",
            "prompt_sha256": prompt_sha256,
            "prompt_bytes": raw_prompt.len(),
            "reasoning_effort": "xhigh",
            "reasoning_required": true,
            "ollama_think": true,
            "ollama_raw_prompt": true,
            "ollama_reasoning_parse": "raw generation is split at Qwen's </think> delimiter with the same initial-thinking semantics as the embedded provider",
            "sampling": {
                "mode": "greedy",
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
                "min_p": 0,
            },
            "seed": SEED,
            "context_tokens": CONTEXT_TOKENS,
            "requested_output_tokens": output_tokens,
            "replicas": replicas,
            "order": "AB/BA rotating by replica",
            "warm_measurement": "second identical request with the model resident and prompt prefix reusable",
            "cooldown_milliseconds": cooldown_milliseconds,
            "residency_isolation": "every engine sample runs in a fresh child process; process exit releases embedded MLX allocations; Ollama is explicitly unloaded before and after use",
            "environment_contract": {
                "maximum_pre_sample_gpu_utilization_percent": MAX_IDLE_GPU_UTILIZATION_PERCENT,
                "gpu_utilization_sampling": {
                    "priming_reads_discarded": 1,
                    "post_prime_samples": GPU_OBSERVATION_SAMPLES,
                    "sample_interval_milliseconds": GPU_SAMPLE_INTERVAL_MILLISECONDS,
                    "aggregation": "maximum",
                },
                "minimum_pre_sample_memory_free_percent": MIN_IDLE_MEMORY_FREE_PERCENT,
                "minimum_pre_sample_cpu_idle_percent": MIN_IDLE_CPU_PERCENT,
                "minimum_whole_sample_cpu_idle_percent": MIN_WHOLE_SAMPLE_CPU_IDLE_PERCENT,
                "maximum_pre_sample_background_process_cpu_percent": MAX_BACKGROUND_PROCESS_CPU_PERCENT,
                "required_pre_sample_thermal_state": "nominal",
                "maximum_post_process_gpu_memory_bytes": MAX_RESIDUAL_GPU_MEMORY_BYTES,
            },
            "rate_scope": "approximate output tokens after first emitted reasoning/text delta divided by remaining client wall time",
            "content_contract": "both engines must emit reasoning and consume the fixed token budget; hashes are recorded as semantic-stability evidence but bit identity is not required",
        },
        "summary": {
            "evidence_grade": evidence_grade,
            "embedded_warm_median_tokens_per_second": median(&embedded_rates),
            "ollama_warm_median_tokens_per_second": median(&ollama_rates),
            "embedded_warm_median_end_to_end_tokens_per_second": median(&embedded_end_to_end_rates),
            "ollama_warm_median_end_to_end_tokens_per_second": median(&ollama_end_to_end_rates),
            "embedded_warm_median_first_delta_milliseconds": median(&embedded_first_delta_milliseconds),
            "ollama_warm_median_first_delta_milliseconds": median(&ollama_first_delta_milliseconds),
            "embedded_warm_spread_ratio": embedded_spread,
            "ollama_warm_spread_ratio": ollama_spread,
            "environment_preconditions_verified": environment_preconditions_verified,
            "whole_sample_environment_verified": whole_sample_environment_verified,
            "post_sample_environment_verified": post_sample_environment_verified,
            "residency_isolation_verified": residency_isolation_verified,
            "embedded_cold_warm_content_stable": embedded_cold_warm_content_stable,
            "ollama_cold_warm_content_stable": ollama_cold_warm_content_stable,
            "paired_embedded_over_ollama_speedups": paired_speedups,
            "median_paired_embedded_over_ollama_speedup": median(&paired_speedups),
            "paired_embedded_over_ollama_end_to_end_speedups": paired_end_to_end_speedups,
            "median_paired_embedded_over_ollama_end_to_end_speedup": median(&paired_end_to_end_speedups),
            "cross_engine_cold_content_matches": cross_engine_content_matches,
        },
        "samples": observations,
    });
    let serialized = serde_json::to_string_pretty(&output)?;
    if let Some(path) = output_path {
        std::fs::write(path, serialized.as_bytes())?;
    }
    println!("{serialized}");
    Ok(())
}

async fn child_main(arguments: &[OsString]) -> Result<(), Box<dyn Error>> {
    if arguments.len() != 2 {
        return Err(
            "internal usage: embedded_ollama_benchmark --child ENGINE OUTPUT_TOKENS".into(),
        );
    }
    let engine = Engine::parse(
        arguments[0]
            .to_str()
            .ok_or("child ENGINE must be valid UTF-8")?,
    )?;
    let output_tokens = arguments[1]
        .to_str()
        .ok_or("child OUTPUT_TOKENS must be valid UTF-8")?
        .parse::<u32>()?;
    let messages = vec![Message::user_text(USER)];
    let tools: Vec<ToolDef> = Vec::new();
    let result = match engine {
        Engine::Embedded => run_embedded_sample(&messages, &tools, output_tokens).await?,
        Engine::Ollama => {
            let base_url = env::var("GENERALIST_OLLAMA_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_OLLAMA_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string();
            let client = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(300))
                .build()?;
            let raw_prompt = render_qwen_prompt(&LocalCompletionRequest {
                system: Some(SYSTEM.to_string()),
                messages,
                tools,
                max_tokens: Some(output_tokens),
                limits: CompletionLimits::default(),
            })?;
            let result = run_ollama_sample(&client, &base_url, &raw_prompt, output_tokens).await;
            let unload_result = unload_ollama(&client, &base_url).await;
            let result = result?;
            unload_result?;
            result
        }
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn run_isolated_sample(
    engine: Engine,
    output_tokens: u32,
    ollama_base_url: &str,
) -> Result<(EngineSample, Option<f64>), Box<dyn Error>> {
    let executable = env::current_exe()?;
    let output_tokens = output_tokens.to_string();
    let cpu_before = cpu_ticks();
    let output = Command::new(executable)
        .args(["--child", engine.label(), output_tokens.as_str()])
        .env("GENERALIST_OLLAMA_BASE_URL", ollama_base_url)
        .output()?;
    let whole_sample_cpu_idle_percent = cpu_before
        .zip(cpu_ticks())
        .and_then(|(before, after)| cpu_idle_percent_between(before, after));
    if !output.status.success() {
        return Err(format!(
            "isolated {} sample exited {}: {}{}",
            engine.label(),
            output.status,
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout),
        )
        .into());
    }
    let result: EngineSample =
        serde_json::from_slice(&output.stdout).map_err(|error| -> Box<dyn Error> {
            format!(
                "isolated {} sample emitted invalid JSON ({error}): {}",
                engine.label(),
                String::from_utf8_lossy(&output.stdout),
            )
            .into()
        })?;
    Ok((result, whole_sample_cpu_idle_percent))
}

async fn run_embedded_sample(
    messages: &[Message],
    tools: &[ToolDef],
    output_tokens: u32,
) -> Result<EngineSample, Box<dyn Error>> {
    let load_started = Instant::now();
    let provider = EmbeddedProvider::qwen(DEFAULT_EMBEDDED_MODEL)?;
    let model_load_milliseconds = milliseconds(load_started.elapsed());
    let cold = measure_embedded(&provider, messages, tools, output_tokens).await?;
    let warm = measure_embedded(&provider, messages, tools, output_tokens).await?;
    validate_measurement_pair(Engine::Embedded, output_tokens, &cold, &warm)?;
    let shutdown_started = Instant::now();
    drop(provider);
    let shutdown_milliseconds = milliseconds(shutdown_started.elapsed());
    Ok(EngineSample {
        model_load_milliseconds: Some(model_load_milliseconds),
        cold_warm_content_match: cold.content_sha256 == warm.content_sha256,
        cold,
        warm,
        shutdown_milliseconds: Some(shutdown_milliseconds),
    })
}

async fn measure_embedded(
    provider: &EmbeddedProvider,
    messages: &[Message],
    tools: &[ToolDef],
    output_tokens: u32,
) -> Result<Measurement, Box<dyn Error>> {
    let request = CompletionRequest {
        system: Some(SYSTEM),
        messages,
        tools,
        max_tokens: Some(output_tokens),
        limits: CompletionLimits::default(),
    };
    let started = Instant::now();
    let mut first_delta_duration = None;
    let response = provider
        .complete_streaming(request, &mut |delta| {
            match &delta {
                CompletionDelta::Reasoning(text) | CompletionDelta::Text(text)
                    if !text.is_empty() =>
                {
                    first_delta_duration.get_or_insert_with(|| started.elapsed());
                }
                _ => {}
            }
            Ok(())
        })
        .await?;
    let duration = started.elapsed();
    let first_delta_duration = first_delta_duration.ok_or("embedded emitted no content delta")?;
    let usage = response.usage.ok_or("embedded omitted usage")?;
    let (reasoning, text) = content_strings(&response.content)?;
    measurement(
        duration,
        first_delta_duration,
        MeasurementData {
            input_tokens: usage.input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            output_tokens: usage.output_tokens,
            reasoning,
            text,
            stop_reason: format!("{:?}", response.stop_reason),
            ollama_server: None,
        },
    )
}

async fn run_ollama_sample(
    client: &reqwest::Client,
    base_url: &str,
    raw_prompt: &str,
    output_tokens: u32,
) -> Result<EngineSample, Box<dyn Error>> {
    unload_ollama(client, base_url).await?;
    let cold = measure_ollama(client, base_url, raw_prompt, output_tokens, "10m").await?;
    let warm = measure_ollama(client, base_url, raw_prompt, output_tokens, "0").await?;
    validate_measurement_pair(Engine::Ollama, output_tokens, &cold, &warm)?;
    Ok(EngineSample {
        model_load_milliseconds: cold
            .ollama_server
            .as_ref()
            .map(|timing| timing.load_milliseconds),
        cold_warm_content_match: cold.content_sha256 == warm.content_sha256,
        cold,
        warm,
        shutdown_milliseconds: None,
    })
}

async fn measure_ollama(
    client: &reqwest::Client,
    base_url: &str,
    raw_prompt: &str,
    output_tokens: u32,
    keep_alive: &str,
) -> Result<Measurement, Box<dyn Error>> {
    let body = json!({
        "model": DEFAULT_EMBEDDED_MODEL,
        "prompt": raw_prompt,
        "raw": true,
        "think": true,
        "stream": true,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": CONTEXT_TOKENS,
            "num_predict": output_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "min_p": 0,
            "seed": SEED,
        },
    });
    let started = Instant::now();
    let mut response = client
        .post(format!("{base_url}/api/generate"))
        .json(&body)
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await?;
        return Err(format!("Ollama generate returned {status}: {body}").into());
    }
    let mut pending = Vec::new();
    let mut first_delta_duration = None;
    let mut ollama_thinking = String::new();
    let mut raw_response = String::new();
    let mut final_event = None;
    while let Some(chunk) = response.chunk().await? {
        pending.extend_from_slice(&chunk);
        while let Some(newline) = pending.iter().position(|byte| *byte == b'\n') {
            let line = pending.drain(..=newline).collect::<Vec<_>>();
            consume_ollama_event(
                &line,
                started,
                &mut first_delta_duration,
                &mut ollama_thinking,
                &mut raw_response,
                &mut final_event,
            )?;
        }
    }
    if !pending.iter().all(u8::is_ascii_whitespace) {
        consume_ollama_event(
            &pending,
            started,
            &mut first_delta_duration,
            &mut ollama_thinking,
            &mut raw_response,
            &mut final_event,
        )?;
    }
    let duration = started.elapsed();
    let first_delta_duration = first_delta_duration.ok_or("Ollama emitted no content delta")?;
    let final_event = final_event.ok_or("Ollama stream ended without a done event")?;
    let timing = ollama_timing(&final_event)?;
    let (reasoning, text) = if ollama_thinking.is_empty() {
        split_raw_qwen_response(&raw_response)
    } else {
        (ollama_thinking, raw_response)
    };
    measurement(
        duration,
        first_delta_duration,
        MeasurementData {
            input_tokens: timing.prompt_eval_count,
            cache_read_input_tokens: None,
            output_tokens: timing.eval_count,
            reasoning,
            text,
            stop_reason: final_event
                .get("done_reason")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            ollama_server: Some(timing),
        },
    )
}

fn split_raw_qwen_response(response: &str) -> (String, String) {
    let leading = response.len() - response.trim_start_matches(char::is_whitespace).len();
    let response = if response[leading..].starts_with("<think>") {
        response[leading + "<think>".len()..].trim_start_matches(char::is_whitespace)
    } else {
        response
    };
    if let Some(index) = response.find("</think>") {
        (
            response[..index].to_string(),
            response[index + "</think>".len()..]
                .trim_start_matches(char::is_whitespace)
                .trim_end()
                .to_string(),
        )
    } else {
        (response.to_string(), String::new())
    }
}

fn consume_ollama_event(
    line: &[u8],
    started: Instant,
    first_delta_duration: &mut Option<Duration>,
    reasoning: &mut String,
    text: &mut String,
    final_event: &mut Option<Value>,
) -> Result<(), Box<dyn Error>> {
    let line = line
        .iter()
        .copied()
        .take_while(|byte| *byte != b'\n')
        .collect::<Vec<_>>();
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(());
    }
    let event: Value = serde_json::from_slice(&line)?;
    if let Some(error) = event.get("error").and_then(Value::as_str) {
        return Err(format!("Ollama stream error: {error}").into());
    }
    for (field, output) in [("thinking", reasoning), ("response", text)] {
        if let Some(fragment) = event.get(field).and_then(Value::as_str) {
            if !fragment.is_empty() {
                first_delta_duration.get_or_insert_with(|| started.elapsed());
                output.push_str(fragment);
            }
        }
    }
    if event.get("done").and_then(Value::as_bool) == Some(true) {
        *final_event = Some(event);
    }
    Ok(())
}

fn ollama_timing(event: &Value) -> Result<OllamaServerTiming, Box<dyn Error>> {
    let required = |field: &str| {
        event
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("Ollama final event omitted {field}"))
    };
    let total = required("total_duration")?;
    let load = required("load_duration")?;
    let prompt_eval_count = required("prompt_eval_count")?;
    let prompt_eval = required("prompt_eval_duration")?;
    let eval_count = required("eval_count")?;
    let eval_duration = required("eval_duration")?;
    if eval_duration == 0 {
        return Err("Ollama reported zero eval duration".into());
    }
    Ok(OllamaServerTiming {
        total_milliseconds: nanoseconds_to_milliseconds(total),
        load_milliseconds: nanoseconds_to_milliseconds(load),
        prompt_eval_count,
        prompt_eval_milliseconds: nanoseconds_to_milliseconds(prompt_eval),
        eval_count,
        eval_milliseconds: nanoseconds_to_milliseconds(eval_duration),
        decode_tokens_per_second: eval_count as f64 * 1_000_000_000.0 / eval_duration as f64,
    })
}

fn measurement(
    duration: Duration,
    first_delta_duration: Duration,
    data: MeasurementData,
) -> Result<Measurement, Box<dyn Error>> {
    if data.output_tokens == 0 {
        return Err("completion reported zero output tokens".into());
    }
    if data.reasoning.trim().is_empty() {
        return Err(format!(
            "completion omitted reasoning: text_bytes={}, stop_reason={}",
            data.text.len(),
            data.stop_reason,
        )
        .into());
    }
    let post_first_delta = duration.saturating_sub(first_delta_duration);
    if post_first_delta.is_zero() {
        return Err("first content delta arrived only after completion".into());
    }
    let content_sha256 = canonical_content_sha256(&data.reasoning, &data.text)?;
    Ok(Measurement {
        total_milliseconds: milliseconds(duration),
        first_delta_milliseconds: milliseconds(first_delta_duration),
        end_to_end_output_tokens_per_second: data.output_tokens as f64 / duration.as_secs_f64(),
        approximate_post_first_delta_tokens_per_second: data.output_tokens.saturating_sub(1) as f64
            / post_first_delta.as_secs_f64(),
        input_tokens: data.input_tokens,
        cache_read_input_tokens: data.cache_read_input_tokens,
        output_tokens: data.output_tokens,
        reasoning_bytes: data.reasoning.len(),
        text_bytes: data.text.len(),
        stop_reason: data.stop_reason,
        content_sha256,
        ollama_server: data.ollama_server,
    })
}

fn validate_measurement_pair(
    engine: Engine,
    requested_output_tokens: u32,
    cold: &Measurement,
    warm: &Measurement,
) -> Result<(), Box<dyn Error>> {
    if cold.output_tokens != u64::from(requested_output_tokens)
        || warm.output_tokens != u64::from(requested_output_tokens)
    {
        return Err(format!(
            "{} did not consume the fixed output-token budget: cold={}, warm={}, requested={requested_output_tokens}",
            engine.label(),
            cold.output_tokens,
            warm.output_tokens,
        )
        .into());
    }
    Ok(())
}

async fn unload_ollama(client: &reqwest::Client, base_url: &str) -> Result<(), Box<dyn Error>> {
    let response = client
        .post(format!("{base_url}/api/generate"))
        .json(&json!({"model": DEFAULT_EMBEDDED_MODEL, "keep_alive": 0}))
        .send()
        .await?;
    if response.status().is_success() {
        Ok(())
    } else {
        let status = response.status();
        let body = response.text().await?;
        Err(format!("failed to unload Ollama model ({status}): {body}").into())
    }
}

async fn inspect_ollama_model(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<Value, Box<dyn Error>> {
    let response = client.get(format!("{base_url}/api/tags")).send().await?;
    if !response.status().is_success() {
        return Err(format!("Ollama tags returned {}", response.status()).into());
    }
    let body: Value = response.json().await?;
    body.get("models")
        .and_then(Value::as_array)
        .and_then(|models| {
            models.iter().find(|model| {
                model.get("name").and_then(Value::as_str) == Some(DEFAULT_EMBEDDED_MODEL)
            })
        })
        .cloned()
        .ok_or_else(|| format!("Ollama model {DEFAULT_EMBEDDED_MODEL} is not installed").into())
}

fn content_strings(content: &[ContentBlock]) -> Result<(String, String), Box<dyn Error>> {
    let mut reasoning = String::new();
    let mut text = String::new();
    for block in content {
        match block {
            ContentBlock::Thinking { thinking, .. } => reasoning.push_str(thinking),
            ContentBlock::Text { text: fragment } => text.push_str(fragment),
            ContentBlock::ToolUse { .. }
            | ContentBlock::ToolResult { .. }
            | ContentBlock::RedactedThinking { .. } => {
                return Err("unexpected non-text block in no-tool benchmark".into());
            }
        }
    }
    Ok((reasoning, text))
}

fn canonical_content_sha256(reasoning: &str, text: &str) -> Result<String, Box<dyn Error>> {
    Ok(sha256(&serde_json::to_vec(&json!({
        "reasoning": reasoning,
        "text": text,
    }))?))
}

fn system_snapshot() -> SystemSnapshot {
    let mut observation_errors = Vec::new();
    let load_average = load_average();
    let cpu_idle_percent = match command_output("top", &["-l", "2", "-s", "1", "-n", "0"]) {
        Ok(output) => parse_cpu_idle_percent(&output),
        Err(error) => {
            observation_errors.push(error);
            None
        }
    };
    let memory_free_percent = match command_output("memory_pressure", &[]) {
        Ok(output) => parse_number_after(&output, "System-wide memory free percentage:"),
        Err(error) => {
            observation_errors.push(error);
            None
        }
    };
    let (gpu_device, gpu_renderer, gpu_tiler, gpu_memory) = match sampled_gpu_statistics() {
        Ok(statistics) => (
            Some(statistics.device_utilization_percent),
            Some(statistics.renderer_utilization_percent),
            Some(statistics.tiler_utilization_percent),
            Some(statistics.in_use_system_memory_bytes),
        ),
        Err(error) => {
            observation_errors.push(error);
            (None, None, None, None)
        }
    };
    let top_processes = match command_output("ps", &["-Ao", "pid=,%cpu=,rss=,comm="]) {
        Ok(output) => parse_processes(&output),
        Err(error) => {
            observation_errors.push(error);
            Vec::new()
        }
    };
    SystemSnapshot {
        load_average,
        cpu_idle_percent,
        memory_free_percent,
        gpu_device_utilization_percent: gpu_device,
        gpu_renderer_utilization_percent: gpu_renderer,
        gpu_tiler_utilization_percent: gpu_tiler,
        gpu_in_use_system_memory_bytes: gpu_memory,
        thermal_state: thermal_state().map(str::to_string),
        top_processes,
        observation_errors,
    }
}

fn pre_sample_environment_rejection(snapshot: &SystemSnapshot) -> Option<String> {
    if !snapshot.observation_errors.is_empty() {
        return Some(format!(
            "environment observation failed: {}",
            snapshot.observation_errors.join("; ")
        ));
    }
    let Some(cpu_idle) = snapshot.cpu_idle_percent else {
        return Some("CPU idle observation is unavailable".to_string());
    };
    if cpu_idle < MIN_IDLE_CPU_PERCENT {
        return Some(format!(
            "CPU idle {cpu_idle:.2}% is below {MIN_IDLE_CPU_PERCENT:.2}%"
        ));
    }
    let Some(gpu_utilization) = snapshot.gpu_device_utilization_percent else {
        return Some("GPU utilization observation is unavailable".to_string());
    };
    if gpu_utilization > MAX_IDLE_GPU_UTILIZATION_PERCENT {
        return Some(format!(
            "GPU utilization {gpu_utilization}% exceeds {MAX_IDLE_GPU_UTILIZATION_PERCENT}%"
        ));
    }
    let Some(memory_free) = snapshot.memory_free_percent else {
        return Some("free-memory observation is unavailable".to_string());
    };
    if memory_free < MIN_IDLE_MEMORY_FREE_PERCENT {
        return Some(format!(
            "free memory {memory_free:.2}% is below {MIN_IDLE_MEMORY_FREE_PERCENT:.2}%"
        ));
    }
    let Some(gpu_memory) = snapshot.gpu_in_use_system_memory_bytes else {
        return Some("GPU-memory observation is unavailable".to_string());
    };
    if gpu_memory > MAX_RESIDUAL_GPU_MEMORY_BYTES {
        return Some(format!(
            "GPU memory {gpu_memory} bytes exceeds {MAX_RESIDUAL_GPU_MEMORY_BYTES} bytes"
        ));
    }
    if snapshot.thermal_state.as_deref() != Some("nominal") {
        return Some(format!(
            "thermal state is {}, not nominal",
            snapshot.thermal_state.as_deref().unwrap_or("unavailable")
        ));
    }
    let Some(process) = snapshot.top_processes.first() else {
        return Some("background-process observation is empty".to_string());
    };
    if process.cpu_percent > MAX_BACKGROUND_PROCESS_CPU_PERCENT {
        return Some(format!(
            "background process {} (PID {}) uses {:.1}% CPU, above {:.1}%",
            process.command, process.pid, process.cpu_percent, MAX_BACKGROUND_PROCESS_CPU_PERCENT,
        ));
    }
    None
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn cpu_ticks() -> Option<CpuTicks> {
    let mut info = MaybeUninit::<libc::host_cpu_load_info>::zeroed();
    let mut count = libc::HOST_CPU_LOAD_INFO_COUNT;
    // SAFETY: `info` has the exact layout and size requested by
    // HOST_CPU_LOAD_INFO, and `count` reports that initialized size.
    let status = unsafe {
        libc::host_statistics(
            libc::mach_host_self(),
            libc::HOST_CPU_LOAD_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if status != libc::KERN_SUCCESS || count < libc::HOST_CPU_LOAD_INFO_COUNT {
        return None;
    }
    // SAFETY: a successful host_statistics call initialized all four fields.
    let ticks = unsafe { info.assume_init() }.cpu_ticks;
    Some(CpuTicks {
        user: ticks[libc::CPU_STATE_USER as usize],
        system: ticks[libc::CPU_STATE_SYSTEM as usize],
        idle: ticks[libc::CPU_STATE_IDLE as usize],
        nice: ticks[libc::CPU_STATE_NICE as usize],
    })
}

#[cfg(not(target_os = "macos"))]
fn cpu_ticks() -> Option<CpuTicks> {
    None
}

fn cpu_idle_percent_between(before: CpuTicks, after: CpuTicks) -> Option<f64> {
    let elapsed = |start: u32, end: u32| u64::from(end.wrapping_sub(start));
    let idle = elapsed(before.idle, after.idle);
    let total = idle
        + elapsed(before.user, after.user)
        + elapsed(before.system, after.system)
        + elapsed(before.nice, after.nice);
    (total != 0).then_some(idle as f64 * 100.0 / total as f64)
}

#[cfg(target_os = "macos")]
fn thermal_state() -> Option<&'static str> {
    use std::ffi::{c_char, c_void};

    type Object = *mut c_void;
    type Selector = *mut c_void;

    #[link(name = "Foundation", kind = "framework")]
    extern "C" {}
    #[link(name = "objc")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> Object;
        fn sel_registerName(name: *const c_char) -> Selector;
        fn objc_msgSend();
    }

    // SAFETY: NSProcessInfo and these zero-argument selectors are stable
    // Foundation APIs. The objc_msgSend casts match their declared return
    // types on 64-bit macOS.
    let value = unsafe {
        let class = objc_getClass(c"NSProcessInfo".as_ptr());
        if class.is_null() {
            return None;
        }
        let send_object: unsafe extern "C" fn(Object, Selector) -> Object =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let process_info = send_object(class, sel_registerName(c"processInfo".as_ptr()));
        if process_info.is_null() {
            return None;
        }
        let send_integer: unsafe extern "C" fn(Object, Selector) -> isize =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        send_integer(process_info, sel_registerName(c"thermalState".as_ptr()))
    };
    match value {
        0 => Some("nominal"),
        1 => Some("fair"),
        2 => Some("serious"),
        3 => Some("critical"),
        _ => None,
    }
}

#[cfg(not(target_os = "macos"))]
fn thermal_state() -> Option<&'static str> {
    None
}

fn parse_cpu_idle_percent(output: &str) -> Option<f64> {
    let line = output
        .lines()
        .rfind(|line| line.starts_with("CPU usage:"))?;
    let idle = line.split(',').find(|field| field.contains("idle"))?;
    idle.split_whitespace()
        .find(|field| field.ends_with('%'))?
        .trim_end_matches('%')
        .parse()
        .ok()
}

fn load_average() -> Option<[f64; 3]> {
    let mut values = [0.0; 3];
    // SAFETY: `values` points to three writable doubles for the duration of
    // the call, exactly matching the element count passed to getloadavg.
    let count = unsafe { libc::getloadavg(values.as_mut_ptr(), values.len() as i32) };
    (count == values.len() as i32).then_some(values)
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn sampled_gpu_statistics() -> Result<GpuStatistics, String> {
    const IOREG_ARGUMENTS: &[&str] = &["-r", "-d", "1", "-c", "AGXAccelerator", "-l"];

    // AGX PerformanceStatistics can retain a completed observation window
    // until it is read. Prime that state, then measure a fresh quiet window.
    // Taking the maximum of the fresh samples keeps admission fail-closed for
    // real intermittent GPU work instead of accepting a median-only lull.
    let _ = command_output("ioreg", IOREG_ARGUMENTS)?;
    let mut samples = Vec::with_capacity(GPU_OBSERVATION_SAMPLES);
    for _ in 0..GPU_OBSERVATION_SAMPLES {
        std::thread::sleep(Duration::from_millis(GPU_SAMPLE_INTERVAL_MILLISECONDS));
        let output = command_output("ioreg", IOREG_ARGUMENTS)?;
        samples.push(parse_gpu_statistics(&output).ok_or_else(|| {
            "ioreg omitted one or more required AGX performance statistics".to_string()
        })?);
    }
    maximum_gpu_statistics(&samples)
        .ok_or_else(|| "GPU observation window contained no samples".to_string())
}

fn parse_gpu_statistics(output: &str) -> Option<GpuStatistics> {
    Some(GpuStatistics {
        device_utilization_percent: parse_ioreg_integer(output, "Device Utilization %")?,
        renderer_utilization_percent: parse_ioreg_integer(output, "Renderer Utilization %")?,
        tiler_utilization_percent: parse_ioreg_integer(output, "Tiler Utilization %")?,
        in_use_system_memory_bytes: parse_ioreg_integer(output, "In use system memory")?,
    })
}

fn maximum_gpu_statistics(samples: &[GpuStatistics]) -> Option<GpuStatistics> {
    let first = *samples.first()?;
    Some(
        samples
            .iter()
            .copied()
            .skip(1)
            .fold(first, |maximum, sample| GpuStatistics {
                device_utilization_percent: maximum
                    .device_utilization_percent
                    .max(sample.device_utilization_percent),
                renderer_utilization_percent: maximum
                    .renderer_utilization_percent
                    .max(sample.renderer_utilization_percent),
                tiler_utilization_percent: maximum
                    .tiler_utilization_percent
                    .max(sample.tiler_utilization_percent),
                in_use_system_memory_bytes: maximum
                    .in_use_system_memory_bytes
                    .max(sample.in_use_system_memory_bytes),
            }),
    )
}

fn parse_processes(output: &str) -> Vec<ProcessLoad> {
    let own_pid = std::process::id();
    let mut processes = output
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields.next()?.parse::<u32>().ok()?;
            let cpu_percent = fields.next()?.parse::<f64>().ok()?;
            let resident_kib = fields.next()?.parse::<f64>().ok()?;
            let command = fields.collect::<Vec<_>>().join(" ");
            (pid != own_pid && command != "ps").then_some(ProcessLoad {
                pid,
                cpu_percent,
                resident_mib: resident_kib / 1024.0,
                command,
            })
        })
        .collect::<Vec<_>>();
    processes.sort_by(|left, right| {
        right
            .cpu_percent
            .partial_cmp(&left.cpu_percent)
            .unwrap_or(Ordering::Equal)
    });
    processes.truncate(3);
    processes
}

fn parse_number_after(output: &str, prefix: &str) -> Option<f64> {
    let suffix = output.split(prefix).nth(1)?;
    suffix
        .split_whitespace()
        .next()?
        .trim_end_matches('%')
        .parse()
        .ok()
}

fn parse_ioreg_integer(output: &str, key: &str) -> Option<u64> {
    let suffix = output.split(&format!("\"{key}\"=")).nth(1)?;
    let digits = suffix
        .chars()
        .skip_while(|character| !character.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse().ok()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    }
}

fn spread(values: &[f64]) -> f64 {
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum / minimum
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn nanoseconds_to_milliseconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / 1_000_000.0
}

fn sha256(bytes: &[u8]) -> String {
    let value = digest(&SHA256, bytes);
    let mut encoded = String::with_capacity(value.as_ref().len() * 2);
    for byte in value.as_ref() {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
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

#[cfg(test)]
mod tests {
    use super::{
        cpu_idle_percent_between, maximum_gpu_statistics, pre_sample_environment_rejection,
        CpuTicks, GpuStatistics, ProcessLoad, SystemSnapshot,
    };

    fn quiet_snapshot() -> SystemSnapshot {
        SystemSnapshot {
            load_average: Some([1.0, 1.0, 1.0]),
            cpu_idle_percent: Some(90.0),
            memory_free_percent: Some(90.0),
            gpu_device_utilization_percent: Some(0),
            gpu_renderer_utilization_percent: Some(0),
            gpu_tiler_utilization_percent: Some(0),
            gpu_in_use_system_memory_bytes: Some(512 * 1024 * 1024),
            thermal_state: Some("nominal".to_string()),
            top_processes: vec![ProcessLoad {
                pid: 42,
                cpu_percent: 10.0,
                resident_mib: 64.0,
                command: "background".to_string(),
            }],
            observation_errors: Vec::new(),
        }
    }

    #[test]
    fn whole_sample_cpu_idle_uses_all_tick_classes() {
        let idle = cpu_idle_percent_between(
            CpuTicks {
                user: 100,
                system: 200,
                idle: 300,
                nice: 400,
            },
            CpuTicks {
                user: 110,
                system: 220,
                idle: 370,
                nice: 400,
            },
        )
        .expect("nonzero tick delta");
        assert!((idle - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn whole_sample_cpu_idle_handles_mach_tick_wraparound() {
        let idle = cpu_idle_percent_between(
            CpuTicks {
                user: u32::MAX - 5,
                system: 50,
                idle: u32::MAX - 5,
                nice: 70,
            },
            CpuTicks {
                user: 4,
                system: 50,
                idle: 4,
                nice: 70,
            },
        )
        .expect("wrapped tick delta");
        assert!((idle - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn quiet_pre_sample_environment_is_admitted() {
        assert_eq!(pre_sample_environment_rejection(&quiet_snapshot()), None);
    }

    #[test]
    fn busy_gpu_is_rejected_before_model_load() {
        let mut snapshot = quiet_snapshot();
        snapshot.gpu_device_utilization_percent = Some(6);
        assert_eq!(
            pre_sample_environment_rejection(&snapshot).as_deref(),
            Some("GPU utilization 6% exceeds 5%")
        );
    }

    #[test]
    fn missing_cpu_observation_cannot_pass_open() {
        let mut snapshot = quiet_snapshot();
        snapshot.cpu_idle_percent = None;
        assert_eq!(
            pre_sample_environment_rejection(&snapshot).as_deref(),
            Some("CPU idle observation is unavailable")
        );
    }

    #[test]
    fn gpu_observation_uses_each_fresh_sample_maximally() {
        let maximum = maximum_gpu_statistics(&[
            GpuStatistics {
                device_utilization_percent: 1,
                renderer_utilization_percent: 4,
                tiler_utilization_percent: 2,
                in_use_system_memory_bytes: 100,
            },
            GpuStatistics {
                device_utilization_percent: 7,
                renderer_utilization_percent: 3,
                tiler_utilization_percent: 5,
                in_use_system_memory_bytes: 80,
            },
        ])
        .expect("nonempty GPU sample window");
        assert_eq!(
            maximum,
            GpuStatistics {
                device_utilization_percent: 7,
                renderer_utilization_percent: 4,
                tiler_utilization_percent: 5,
                in_use_system_memory_bytes: 100,
            }
        );
    }
}
