//! Exactness, latency, and cancellation-bound probe for incremental Qwen
//! prefill. The bounded chunk duration is the worst interval during which a
//! synchronous Metal transaction cannot observe Ctrl+C.

use generalist::provider::embedded::{
    MlxQwenModel, MlxQwenSpeculativeSession, MlxRuntime, MlxSpeculativeStep, ModelPack,
};
use serde_json::json;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_PROMPT_TOKENS: usize = 256;
const DEFAULT_EXTENSION_TOKENS: usize = 64;
const DEFAULT_CHUNK_TOKENS: usize = 64;
const OUTPUT_TOKENS: usize = 8;
const MAX_NATIVE_COMMIT_TOKENS: usize = 8;
const MTP_DRAFT_DEPTH: usize = 2;
const VERIFICATION_QMV_RESULTS: i32 = 4;
const TOKEN_FIXTURE: [i32; 101] = [
    248045, 8678, 198, 2523, 513, 4162, 264, 31626, 27502, 13, 5374, 1472, 279, 3377, 303, 279,
    11886, 7047, 5323, 13, 2732, 12995, 321, 27517, 61446, 13, 4486, 31626, 369, 4434, 11, 2113,
    1132, 1510, 95429, 25, 361, 8944, 42181, 303, 279, 1965, 5323, 11, 440, 874, 15673, 1518, 466,
    1238, 424, 13, 3054, 524, 70884, 466, 10390, 279, 7047, 944, 13, 248046, 198, 248045, 846, 198,
    9764, 264, 62, 15, 283, 220, 18, 321, 264, 8102, 77, 10, 16, 8, 283, 220, 17, 32624, 1054, 478,
    307, 13, 21902, 264, 62, 16, 15, 13, 248046, 198, 248045, 74455, 198, 248068, 198,
];

struct PrefillMeasurement {
    duration: Duration,
    longest_transaction: Duration,
    output_token_ids: Vec<i32>,
    checkpoint: Option<MlxQwenSpeculativeSession>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    let prompt_tokens = optional_usize(arguments.next(), DEFAULT_PROMPT_TOKENS, "PROMPT_TOKENS")?;
    let extension_tokens = optional_usize(
        arguments.next(),
        DEFAULT_EXTENSION_TOKENS,
        "EXTENSION_TOKENS",
    )?;
    let chunk_tokens = optional_usize(arguments.next(), DEFAULT_CHUNK_TOKENS, "CHUNK_TOKENS")?;
    if arguments.next().is_some()
        || prompt_tokens < 2
        || extension_tokens == 0
        || !(1..=512).contains(&chunk_tokens)
    {
        return Err("usage: embedded_prefix_cache_benchmark METALLIB MANIFEST BLOB_ROOT [PROMPT_TOKENS=256] [EXTENSION_TOKENS=64] [CHUNK_TOKENS=64]".into());
    }

    let prompt = fixture_tokens(prompt_tokens);
    let mut extended = prompt.clone();
    extended.extend(fixture_tokens(extension_tokens));
    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let runtime = MlxRuntime::initialize(metallib)?;
    let (mut model, _) = MlxQwenModel::load_with_mtp(&runtime, &pack)?;

    let full = full_prefill(&mut model, &prompt)?;
    let chunked = chunked_prefill(&mut model, &prompt, chunk_tokens, None)?;
    ensure_equal("chunked base prefill", &full, &chunked)?;

    let full_extended = full_prefill(&mut model, &extended)?;
    let cached_extended = chunked_prefill(
        &mut model,
        &extended,
        chunk_tokens,
        Some((
            chunked
                .checkpoint
                .as_ref()
                .ok_or("chunked base prefill omitted its checkpoint")?
                .try_clone()?,
            prompt.len() - 1,
        )),
    )?;
    ensure_equal("cached extended prefill", &full_extended, &cached_extended)?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "protocol": {
                "prompt_tokens": prompt.len(),
                "extension_tokens": extension_tokens,
                "extended_prompt_tokens": extended.len(),
                "chunk_tokens": chunk_tokens,
                "cache_read_tokens": prompt.len() - 1,
                "generated_tokens_checked": OUTPUT_TOKENS,
                "semantic_contract": "chunked and cached token streams equal a fresh full prefill",
                "cancellation_contract": "Ctrl+C is checked between bounded native prefill transactions",
            },
            "full_base": measurement_json(&full),
            "chunked_base": measurement_json(&chunked),
            "full_extended": measurement_json(&full_extended),
            "cached_extended": measurement_json(&cached_extended),
            "cache_speedup": full_extended.duration.as_secs_f64()
                / cached_extended.duration.as_secs_f64(),
            "exact": true,
        }))?
    );
    Ok(())
}

fn full_prefill(
    model: &mut MlxQwenModel,
    prompt: &[i32],
) -> Result<PrefillMeasurement, Box<dyn Error>> {
    let mut session = configured_session(model)?;
    let started = Instant::now();
    let seed = model.speculative_prefill(&mut session, prompt)?;
    let duration = started.elapsed();
    let output_token_ids = decode(model, &mut session, seed)?;
    Ok(PrefillMeasurement {
        duration,
        longest_transaction: duration,
        output_token_ids,
        checkpoint: None,
    })
}

fn chunked_prefill(
    model: &mut MlxQwenModel,
    prompt: &[i32],
    chunk_tokens: usize,
    restored: Option<(MlxQwenSpeculativeSession, usize)>,
) -> Result<PrefillMeasurement, Box<dyn Error>> {
    let (mut session, mut cursor) = match restored {
        Some(restored) => restored,
        None => (configured_session(model)?, 0),
    };
    let started = Instant::now();
    let mut longest_transaction = Duration::ZERO;
    let checkpoint_position = prompt.len() - 1;
    while cursor < checkpoint_position {
        let end = (cursor + chunk_tokens).min(checkpoint_position);
        let transaction_started = Instant::now();
        let step = model.speculative_prefill_chunk(
            &mut session,
            &prompt[cursor..end],
            Some(prompt[end]),
        )?;
        longest_transaction = longest_transaction.max(transaction_started.elapsed());
        if step.position != end - 1
            || step.input_token_id != prompt[end - 1]
            || step.output_token_ids.as_slice() != [prompt[end]]
        {
            return Err("incremental prefill returned an inconsistent held token".into());
        }
        cursor = end;
    }
    let checkpoint = session.try_clone()?;
    let transaction_started = Instant::now();
    let seed =
        model.speculative_prefill_chunk(&mut session, &prompt[checkpoint_position..], None)?;
    longest_transaction = longest_transaction.max(transaction_started.elapsed());
    let duration = started.elapsed();
    let output_token_ids = decode(model, &mut session, seed)?;
    Ok(PrefillMeasurement {
        duration,
        longest_transaction,
        output_token_ids,
        checkpoint: Some(checkpoint),
    })
}

fn configured_session(
    model: &mut MlxQwenModel,
) -> Result<MlxQwenSpeculativeSession, Box<dyn Error>> {
    let mut session = model.start_speculative_session()?;
    session.set_verification_qmv_results_per_simdgroup(VERIFICATION_QMV_RESULTS)?;
    Ok(session)
}

fn decode(
    model: &mut MlxQwenModel,
    session: &mut MlxQwenSpeculativeSession,
    seed: MlxSpeculativeStep,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let mut token_ids = seed.output_token_ids;
    while token_ids.len() < OUTPUT_TOKENS {
        let remaining = OUTPUT_TOKENS - token_ids.len();
        let step = model.speculative_step(
            session,
            MTP_DRAFT_DEPTH,
            remaining.min(MAX_NATIVE_COMMIT_TOKENS),
        )?;
        token_ids.extend(step.output_token_ids);
    }
    Ok(token_ids)
}

fn ensure_equal(
    description: &str,
    expected: &PrefillMeasurement,
    observed: &PrefillMeasurement,
) -> Result<(), Box<dyn Error>> {
    if expected.output_token_ids == observed.output_token_ids {
        Ok(())
    } else {
        Err(format!("{description} changed the greedy token stream").into())
    }
}

fn fixture_tokens(count: usize) -> Vec<i32> {
    TOKEN_FIXTURE.iter().copied().cycle().take(count).collect()
}

fn measurement_json(measurement: &PrefillMeasurement) -> serde_json::Value {
    json!({
        "milliseconds": measurement.duration.as_secs_f64() * 1000.0,
        "longest_native_transaction_milliseconds":
            measurement.longest_transaction.as_secs_f64() * 1000.0,
        "output_token_ids": measurement.output_token_ids,
    })
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing required {name}").into())
}

fn optional_usize(
    value: Option<std::ffi::OsString>,
    default: usize,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    value
        .map(|value| {
            value
                .into_string()
                .map_err(|_| format!("{name} must be valid UTF-8"))?
                .parse::<usize>()
                .map_err(|error| format!("invalid {name}: {error}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
        .map_err(Into::into)
}
