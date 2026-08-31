use generalist::provider::embedded::{MlxQwenModel, MlxRuntime, ModelPack};
use ring::digest::{Context, SHA256};
use serde_json::{json, Value};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const DEFAULT_OUTPUT_TOKENS: usize = 192;
const DEFAULT_ROUNDS: usize = 3;
const WARMUP_TOKENS: usize = 64;
const MAX_OUTPUT_TOKENS: usize = 4096;
const MAX_ROUNDS: usize = 20;
const MAX_STABLE_DECODE_SPREAD_RATIO: f64 = 1.10;
const FROZEN_RECURRENCE_PROMPT_SHA256: &str =
    "231763bebfa8d71fb89820343d150cefddab3554783946289c1e0f54a9be3bf4";
const PROMPT_TOKEN_IDS: [i32; 101] = [
    248045, 8678, 198, 2523, 513, 4162, 264, 31626, 27502, 13, 5374, 1472, 279, 3377, 303, 279,
    11886, 7047, 5323, 13, 2732, 12995, 321, 27517, 61446, 13, 4486, 31626, 369, 4434, 11, 2113,
    1132, 1510, 95429, 25, 361, 8944, 42181, 303, 279, 1965, 5323, 11, 440, 874, 15673, 1518, 466,
    1238, 424, 13, 3054, 524, 70884, 466, 10390, 279, 7047, 944, 13, 248046, 198, 248045, 846, 198,
    9764, 264, 62, 15, 283, 220, 18, 321, 264, 8102, 77, 10, 16, 8, 283, 220, 17, 32624, 1054, 478,
    307, 13, 21902, 264, 62, 16, 15, 13, 248046, 198, 248045, 74455, 198, 248068, 198,
];

const XHIGH_REASONING_INSTRUCTION: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

struct Prompt {
    name: String,
    description: String,
    token_ids: Vec<i32>,
}

#[derive(Clone, Copy)]
struct Mode {
    name: &'static str,
    depth: usize,
    verification_qmv_results: usize,
    verification_shape_override: Option<VerificationShapeOverride>,
    verification_wide_qmv: bool,
    verification_unfused_wide_qmv: bool,
}

impl Mode {
    fn mtp_draft_vocab_limit(self) -> usize {
        if self.name.contains("draft65k") {
            65_536
        } else {
            0
        }
    }

    fn direct_cache_rollback(self) -> bool {
        !self.name.contains("replay")
    }

    fn mtp_draft_prompt_extras(self) -> bool {
        self.name.contains("prompt")
    }
}

#[derive(Clone, Copy)]
struct VerificationShapeOverride {
    output_width: i32,
    input_width: i32,
    results_per_simdgroup: i32,
}

struct Measurement {
    draft_configuration_duration: Duration,
    prefill_duration: Duration,
    duration: Duration,
    tokens: Vec<i32>,
    iterations: usize,
    drafted: usize,
    accepted: usize,
    proposal_milliseconds: f64,
    verification_milliseconds: f64,
    replay_milliseconds: f64,
    correction_milliseconds: f64,
    commits: Vec<Commit>,
}

struct Commit {
    position: usize,
    input: i32,
    outputs: Vec<i32>,
    drafted: usize,
    accepted: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    let output_tokens = optional_usize(arguments.next(), DEFAULT_OUTPUT_TOKENS, "TOKENS")?;
    let rounds = optional_usize(arguments.next(), DEFAULT_ROUNDS, "ROUNDS")?;
    let verification_qmv_results = optional_usize(arguments.next(), 4, "VERIFY_R")?;
    let profile = arguments
        .next()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "PROFILE must be valid UTF-8")
        })
        .transpose()?
        .unwrap_or_else(|| "standard".to_owned());
    let prompt_profile = arguments
        .next()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "PROMPT must be valid UTF-8")
        })
        .transpose()?
        .unwrap_or_else(|| "recurrence".to_owned());
    if arguments.next().is_some()
        || !(2..=MAX_OUTPUT_TOKENS).contains(&output_tokens)
        || !(1..=MAX_ROUNDS).contains(&rounds)
        || !matches!(verification_qmv_results, 1 | 2 | 4 | 8)
        || !matches!(
            profile.as_str(),
            "standard"
                | "depth-focused"
                | "geometry-sweep"
                | "shape-sweep"
                | "shape-focused"
                | "wide-focused"
                | "shortlist-focused"
                | "rollback-focused"
                | "optimized-depth-focused"
        )
        || !matches!(
            prompt_profile.as_str(),
            "recurrence" | "coding" | "tool" | "lean" | "multilingual"
        )
    {
        return Err("usage: embedded_speculative_benchmark METALLIB MANIFEST BLOB_ROOT [TOKENS=192] [ROUNDS=3] [VERIFY_R=4] [PROFILE=standard|depth-focused|geometry-sweep|shape-sweep|shape-focused|wide-focused|shortlist-focused|rollback-focused|optimized-depth-focused] [PROMPT=recurrence|coding|tool|lean|multilingual]".into());
    }

    let modes = if profile == "optimized-depth-focused" {
        vec![
            Mode {
                name: "fixed2-wide-draft65k-prompt-rollback",
                depth: 2,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed3-wide-draft65k-prompt-rollback",
                depth: 3,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed4-wide-draft65k-prompt-rollback",
                depth: 4,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
        ]
    } else if profile == "rollback-focused" {
        vec![
            Mode {
                name: "fixed3-wide-draft65k-prompt-replay",
                depth: 3,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed3-wide-draft65k-prompt-rollback",
                depth: 3,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
        ]
    } else if profile == "shortlist-focused" {
        vec![
            Mode {
                name: "fixed4-wide-draft-full-rollback",
                depth: 4,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed4-wide-draft65k-prompt-rollback",
                depth: 4,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
        ]
    } else if profile == "wide-focused" {
        vec![
            Mode {
                name: "fixed2-r4",
                depth: 2,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed2-wide-unfused",
                depth: 2,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: true,
            },
            Mode {
                name: "fixed2-wide",
                depth: 2,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed3-wide",
                depth: 3,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: true,
                verification_unfused_wide_qmv: false,
            },
        ]
    } else if profile == "depth-focused" {
        vec![
            Mode {
                name: "fixed2",
                depth: 2,
                verification_qmv_results,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed3",
                depth: 3,
                verification_qmv_results,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
        ]
    } else if profile == "geometry-sweep" {
        vec![
            Mode {
                name: "fixed0",
                depth: 0,
                verification_qmv_results: 0,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed2-r2",
                depth: 2,
                verification_qmv_results: 2,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed2-r4",
                depth: 2,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
        ]
    } else if matches!(profile.as_str(), "shape-sweep" | "shape-focused") {
        let mut modes = vec![
            Mode {
                name: "fixed0",
                depth: 0,
                verification_qmv_results: 0,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed2-r4",
                depth: 2,
                verification_qmv_results: 4,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
        ];
        modes.extend([
            shape_override_mode("fixed2-r4-except-16384x5120-r2", 16384, 5120),
            shape_override_mode("fixed2-r4-except-5120x6144-r2", 5120, 6144),
        ]);
        if profile == "shape-sweep" {
            modes.extend([
                shape_override_mode("fixed2-r4-except-34816x5120-r2", 34816, 5120),
                shape_override_mode("fixed2-r4-except-5120x17408-r2", 5120, 17408),
                shape_override_mode("fixed2-r4-except-12288x5120-r2", 12288, 5120),
                shape_override_mode("fixed2-r4-except-248320x5120-r2", 248320, 5120),
            ]);
        }
        modes
    } else {
        vec![
            Mode {
                name: "fixed0",
                depth: 0,
                verification_qmv_results: 0,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed1",
                depth: 1,
                verification_qmv_results,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed2",
                depth: 2,
                verification_qmv_results,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
            Mode {
                name: "fixed3",
                depth: 3,
                verification_qmv_results,
                verification_shape_override: None,
                verification_wide_qmv: false,
                verification_unfused_wide_qmv: false,
            },
        ]
    };

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let prompt = load_prompt(&pack, &prompt_profile)?;
    let runtime = MlxRuntime::initialize(metallib)?;
    let load_started = Instant::now();
    let (mut model, load) = MlxQwenModel::load_with_mtp(&runtime, &pack)?;
    let load_duration = load_started.elapsed();
    if load.mtp_layer_count != 1 {
        return Err("benchmark model did not load exactly one MTP layer".into());
    }

    eprintln!("oracle {} tokens={output_tokens}", modes[0].name);
    let oracle = run(&mut model, modes[0], output_tokens, &prompt.token_ids)?;
    let output_sha256 = token_sha256(&oracle.tokens);
    let timed_tokens = output_tokens - 1;
    let warmup_tokens = output_tokens.min(WARMUP_TOKENS);
    for mode in modes.iter().copied().skip(1) {
        eprintln!("warmup {} tokens={warmup_tokens}", mode.name);
        let warmup = run(&mut model, mode, warmup_tokens, &prompt.token_ids)?;
        if warmup.tokens != oracle.tokens[..warmup_tokens] {
            return Err(format!(
                "{} warmup diverged from plain greedy output: {}",
                mode.name,
                mismatch_description(&oracle.tokens[..warmup_tokens], &warmup)
            )
            .into());
        }
    }

    let mut samples: Vec<Vec<Measurement>> = (0..modes.len())
        .map(|_| Vec::with_capacity(rounds))
        .collect();
    for round in 0..rounds {
        for offset in 0..modes.len() {
            let mode_index = (round + offset) % modes.len();
            let mode = modes[mode_index];
            let measurement = run(&mut model, mode, output_tokens, &prompt.token_ids)?;
            if measurement.tokens != oracle.tokens {
                return Err(format!(
                    "{} round {} diverged from plain greedy output: {}",
                    mode.name,
                    round + 1,
                    mismatch_description(&oracle.tokens, &measurement)
                )
                .into());
            }
            eprintln!(
                "measured round={} mode={} milliseconds={:.3} tokens_per_second={:.3} drafted={} accepted={}",
                round + 1,
                mode.name,
                milliseconds(measurement.duration),
                timed_tokens as f64 / measurement.duration.as_secs_f64(),
                measurement.drafted,
                measurement.accepted,
            );
            samples[mode_index].push(measurement);
        }
    }

    let baseline_median = median_duration(&samples[0]);
    let prompt_extra_token_ids =
        prompt_draft_extra_token_ids(&prompt.token_ids, modes[0].mtp_draft_vocab_limit());
    let uncovered_output_high_tokens = oracle
        .tokens
        .iter()
        .copied()
        .filter(|token| {
            (65_536..248_000).contains(token)
                && prompt_extra_token_ids.binary_search(token).is_err()
        })
        .collect::<Vec<_>>();
    let maximum_decode_spread_ratio = samples
        .iter()
        .map(|mode_samples| duration_spread_ratio(mode_samples))
        .fold(1.0_f64, f64::max);
    let timing_status = if rounds < 3 {
        "screen-only"
    } else if maximum_decode_spread_ratio <= MAX_STABLE_DECODE_SPREAD_RATIO {
        "stable"
    } else {
        "contaminated"
    };
    let mode_results: Vec<Value> = modes
        .iter()
        .enumerate()
        .map(|(index, mode)| {
            let median = median_duration(&samples[index]);
            let prefill_median = median_prefill_duration(&samples[index]);
            let paired_speedup = median_f64(
                samples[0]
                    .iter()
                    .zip(&samples[index])
                    .map(|(baseline, candidate)| {
                        baseline.duration.as_secs_f64() / candidate.duration.as_secs_f64()
                    })
                    .collect(),
            );
            json!({
                "name": mode.name,
                "fixed_depth": mode.depth,
                "verification_qmv_results_per_simdgroup": (mode.depth != 0).then_some(mode.verification_qmv_results),
                "verification_wide_qmv": mode.verification_wide_qmv,
                "verification_unfused_wide_qmv": mode.verification_unfused_wide_qmv,
                "mtp_draft_vocab_limit": mode.mtp_draft_vocab_limit(),
                "mtp_draft_prompt_extras": mode.mtp_draft_prompt_extras(),
                "direct_cache_rollback": mode.direct_cache_rollback(),
                "verification_shape_override": mode.verification_shape_override.map(|shape| json!({
                    "output_width": shape.output_width,
                    "input_width": shape.input_width,
                    "results_per_simdgroup": shape.results_per_simdgroup,
                })),
                "median_tokens_per_second": format!("{:.12}", timed_tokens as f64 / median.as_secs_f64()),
                "median_prefill_tokens_per_second": format!("{:.12}", prompt.token_ids.len() as f64 / prefill_median.as_secs_f64()),
                "median_speedup_over_baseline": format!("{:.6}", baseline_median.as_secs_f64() / median.as_secs_f64()),
                "median_paired_speedup_over_baseline": format!("{paired_speedup:.6}"),
                "median_speedup_over_fixed0": (modes[0].depth == 0).then(|| format!("{:.6}", baseline_median.as_secs_f64() / median.as_secs_f64())),
                "decode_duration_spread_ratio": format!("{:.6}", duration_spread_ratio(&samples[index])),
                "runs": samples[index].iter().map(|sample| {
                    json!({
                        "draft_configuration_milliseconds": format!("{:.3}", milliseconds(sample.draft_configuration_duration)),
                        "prompt_eval_count": prompt.token_ids.len(),
                        "prompt_eval_duration_ns": duration_nanoseconds(sample.prefill_duration),
                        "prompt_tokens_per_second": format!("{:.12}", prompt.token_ids.len() as f64 / sample.prefill_duration.as_secs_f64()),
                        "generated_count": output_tokens,
                        "eval_count": timed_tokens,
                        "eval_duration_ns": duration_nanoseconds(sample.duration),
                        "tokens_per_second": format!("{:.12}", timed_tokens as f64 / sample.duration.as_secs_f64()),
                        "iterations": sample.iterations,
                        "drafted": sample.drafted,
                        "accepted": sample.accepted,
                        "acceptance": format!("{:.6}", acceptance(sample)),
                        "average_draft": format!("{:.6}", average(sample.drafted, sample.iterations)),
                        "average_accepted": format!("{:.6}", average(sample.accepted, sample.iterations)),
                        "proposal_milliseconds": format!("{:.3}", sample.proposal_milliseconds),
                        "verification_milliseconds": format!("{:.3}", sample.verification_milliseconds),
                        "replay_milliseconds": format!("{:.3}", sample.replay_milliseconds),
                        "correction_milliseconds": format!("{:.3}", sample.correction_milliseconds),
                        "output_token_sha256": output_sha256,
                    })
                }).collect::<Vec<_>>(),
            })
        })
        .collect();

    let output = json!({
        "load": {
            "milliseconds": format!("{:.3}", milliseconds(load_duration)),
            "blob_count": load.blob_count,
            "array_count": load.array_count,
            "layer_count": load.layer_count,
            "linear_layer_count": load.linear_layer_count,
            "full_attention_layer_count": load.full_attention_layer_count,
            "mtp_layer_count": load.mtp_layer_count,
        },
        "protocol": {
            "prompt": prompt.description,
            "prompt_profile": prompt.name,
            "prompt_token_sha256": token_sha256(&prompt.token_ids),
            "frozen_recurrence_prompt_sha256": (prompt_profile == "recurrence").then_some(FROZEN_RECURRENCE_PROMPT_SHA256),
            "prompt_tokens": prompt.token_ids.len(),
            "prompt_token_id_bands": {
                "below_65k": prompt.token_ids.iter().filter(|&&token| token < 65_536).count(),
                "from_65k_to_special": prompt.token_ids.iter().filter(|&&token| (65_536..248_000).contains(&token)).count(),
                "special": prompt.token_ids.iter().filter(|&&token| token >= 248_000).count(),
            },
            "prompt_high_token_ids": high_ordinary_token_ids(&prompt.token_ids),
            "generated_tokens": output_tokens,
            "timed_decode_tokens": timed_tokens,
            "replicas_per_mode": rounds,
            "warmup_tokens": warmup_tokens,
            "mode_order": "rotated one position per replica",
            "timing_status": timing_status,
            "timing_status_rule": "fewer than three replicas is screen-only; otherwise every mode's maximum/minimum decode duration must be at most 1.10",
            "maximum_decode_spread_ratio": format!("{maximum_decode_spread_ratio:.6}"),
            "baseline_mode": modes[0].name,
            "profile": profile,
            "standard_verification_qmv_results_per_simdgroup": verification_qmv_results,
            "prefill_timer_scope": "complete prompt state transition; target-only for fixed0 and target-plus-MTP for speculative modes",
            "decode_timer_scope": "starts after prefill has produced the first greedy completion token; generated_tokens minus one model-input evaluations are timed",
            "semantic_contract": "every complete token-id stream must equal ordinary greedy target decoding",
            "output_token_hash": "sha256 of signed i32 token ids encoded little-endian",
            "output_token_sha256": output_sha256,
            "output_token_id_bands": {
                "below_65k": oracle.tokens.iter().filter(|&&token| token < 65_536).count(),
                "from_65k_to_special": oracle.tokens.iter().filter(|&&token| (65_536..248_000).contains(&token)).count(),
                "special": oracle.tokens.iter().filter(|&&token| token >= 248_000).count(),
            },
            "output_high_token_ids": high_ordinary_token_ids(&oracle.tokens),
            "output_uncovered_high_token_count": uncovered_output_high_tokens.len(),
            "output_uncovered_high_token_ids": high_ordinary_token_ids(&uncovered_output_high_tokens),
        },
        "modes": mode_results,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn run(
    model: &mut MlxQwenModel,
    mode: Mode,
    output_tokens: usize,
    prompt_token_ids: &[i32],
) -> Result<Measurement, Box<dyn Error>> {
    if mode.depth == 0 {
        run_plain(model, output_tokens, prompt_token_ids)
    } else {
        run_speculative(model, mode, output_tokens, prompt_token_ids)
    }
}

fn run_plain(
    model: &mut MlxQwenModel,
    output_tokens: usize,
    prompt_token_ids: &[i32],
) -> Result<Measurement, Box<dyn Error>> {
    let mut session = model.start_session()?;
    let prefill_started = Instant::now();
    let seed = model.prefill(&mut session, prompt_token_ids)?;
    let prefill_duration = prefill_started.elapsed();
    if seed.position != prompt_token_ids.len() - 1
        || seed.input_token_id != prompt_token_ids[prompt_token_ids.len() - 1]
    {
        return Err("plain target prefill returned an inconsistent position".into());
    }
    let mut input = seed.output_token_id;
    let mut tokens = Vec::with_capacity(output_tokens);
    tokens.push(input);
    let started = Instant::now();
    for generated_index in 1..output_tokens {
        let step = model.cached_decode_argmax(&mut session, input)?;
        let position = prompt_token_ids.len() + generated_index - 1;
        if step.position != position || step.input_token_id != input {
            return Err("plain target session returned an inconsistent position".into());
        }
        tokens.push(step.output_token_id);
        input = step.output_token_id;
    }
    Ok(Measurement {
        draft_configuration_duration: Duration::ZERO,
        prefill_duration,
        duration: started.elapsed(),
        tokens,
        iterations: output_tokens - 1,
        drafted: 0,
        accepted: 0,
        proposal_milliseconds: 0.0,
        verification_milliseconds: 0.0,
        replay_milliseconds: 0.0,
        correction_milliseconds: 0.0,
        commits: Vec::new(),
    })
}

fn run_speculative(
    model: &mut MlxQwenModel,
    mode: Mode,
    output_tokens: usize,
    prompt_token_ids: &[i32],
) -> Result<Measurement, Box<dyn Error>> {
    let mut session = model.start_speculative_session()?;
    session.set_verification_qmv_results_per_simdgroup(mode.verification_qmv_results as i32)?;
    session.set_verification_wide_qmv(mode.verification_wide_qmv)?;
    if mode.verification_unfused_wide_qmv {
        session.set_verification_unfused_wide_qmv(true)?;
    }
    let draft_configuration_started = Instant::now();
    session.set_mtp_draft_vocab_limit(mode.mtp_draft_vocab_limit())?;
    let prompt_extras = if mode.mtp_draft_prompt_extras() {
        prompt_draft_extra_token_ids(prompt_token_ids, mode.mtp_draft_vocab_limit())
    } else {
        Vec::new()
    };
    model.set_mtp_draft_extra_tokens(&mut session, &prompt_extras)?;
    let draft_configuration_duration = draft_configuration_started.elapsed();
    session.set_direct_cache_rollback(mode.direct_cache_rollback())?;
    if let Some(shape) = mode.verification_shape_override {
        session.set_verification_qmv_shape_override(
            mode.verification_qmv_results as i32,
            shape.output_width,
            shape.input_width,
            shape.results_per_simdgroup,
        )?;
    }
    let prefill_started = Instant::now();
    let seed = model.speculative_prefill(&mut session, prompt_token_ids)?;
    let prefill_duration = prefill_started.elapsed();
    if seed.position != prompt_token_ids.len() - 1
        || seed.input_token_id != prompt_token_ids[prompt_token_ids.len() - 1]
        || seed.output_token_ids.len() != 1
    {
        return Err("speculative prefill returned an inconsistent commit".into());
    }
    let mut commits = vec![Commit {
        position: seed.position,
        input: seed.input_token_id,
        outputs: seed.output_token_ids.clone(),
        drafted: seed.drafted,
        accepted: seed.accepted,
    }];
    let mut tokens = seed.output_token_ids;
    let started = Instant::now();
    let mut iterations = 0;
    let mut drafted = 0;
    let mut accepted = 0;
    let mut proposal_milliseconds = 0.0;
    let mut verification_milliseconds = 0.0;
    let mut replay_milliseconds = 0.0;
    let mut correction_milliseconds = 0.0;
    while tokens.len() < output_tokens {
        let generated_position = tokens.len();
        let position = prompt_token_ids.len() + generated_position - 1;
        let input = *tokens.last().ok_or("speculative token stream is empty")?;
        let remaining = output_tokens - generated_position;
        let step = model.speculative_step(&mut session, mode.depth, remaining.min(8))?;
        if step.position != position
            || step.input_token_id != input
            || step.output_token_ids.len() > remaining
        {
            return Err("speculative session returned an inconsistent commit".into());
        }
        iterations += 1;
        drafted += step.drafted;
        accepted += step.accepted;
        proposal_milliseconds += step.proposal_milliseconds;
        verification_milliseconds += step.verification_milliseconds;
        replay_milliseconds += step.replay_milliseconds;
        correction_milliseconds += step.correction_milliseconds;
        commits.push(Commit {
            position: step.position,
            input: step.input_token_id,
            outputs: step.output_token_ids.clone(),
            drafted: step.drafted,
            accepted: step.accepted,
        });
        tokens.extend_from_slice(&step.output_token_ids);
    }
    Ok(Measurement {
        draft_configuration_duration,
        prefill_duration,
        duration: started.elapsed(),
        tokens,
        iterations,
        drafted,
        accepted,
        proposal_milliseconds,
        verification_milliseconds,
        replay_milliseconds,
        correction_milliseconds,
        commits,
    })
}

fn shape_override_mode(name: &'static str, output_width: i32, input_width: i32) -> Mode {
    Mode {
        name,
        depth: 2,
        verification_qmv_results: 4,
        verification_shape_override: Some(VerificationShapeOverride {
            output_width,
            input_width,
            results_per_simdgroup: 2,
        }),
        verification_wide_qmv: false,
        verification_unfused_wide_qmv: false,
    }
}

fn load_prompt(pack: &ModelPack, profile: &str) -> Result<Prompt, Box<dyn Error>> {
    if profile == "recurrence" {
        return Ok(Prompt {
            name: profile.to_owned(),
            description: "the exact medium-effort, think-enabled recurrence_ten_steps chat prompt used by the frozen Ollama depth-control run".to_owned(),
            token_ids: PROMPT_TOKEN_IDS.to_vec(),
        });
    }

    let (system_suffix, user, description) = match profile {
        "coding" => (
            "You are a systems programmer. Produce compilable Rust and scrutinize cancellation, ownership, and concurrency invariants.",
            "Implement a bounded concurrent Rust cache with single-flight loading, prompt cancellation, and deterministic tests. Explain the invariants, then give the complete implementation.",
            "a Rust systems-programming task representative of Generalist coding turns",
        ),
        "tool" => (
            "You may call the read_source function. If a call is useful, emit exactly one XML tool call with path and line_start parameters.",
            "Inspect src/local_inference/qwen.rs around the adaptive MTP controller before recommending a performance change. Use the available read_source tool first.",
            "a tool-selection task requesting Generalist-style structured call markup",
        ),
        "lean" => (
            "You are an expert Lean 4 programmer. Check theorem statements and proof terms carefully.",
            "Prove in Lean 4, without Mathlib, that reversing a list twice returns the original list. Explain the induction and then provide a complete theorem proof.",
            "a formal-reasoning and Lean code task",
        ),
        "multilingual" => (
            "Answer in Chinese while preserving code identifiers and mathematical notation exactly.",
            "请分析一个异步任务取消时可能出现的竞态条件，并给出一个安全的 Rust 状态机设计。先逐步推理，再给出简洁结论。",
            "a Chinese reasoning task with Rust terminology",
        ),
        _ => return Err(format!("unknown prompt profile {profile}").into()),
    };
    let rendered = format!(
        "<|im_start|>system\n{XHIGH_REASONING_INSTRUCTION}\n\n{system_suffix}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    let tokenizer_descriptor = pack.document("tokenizer.json")?;
    let tokenizer_path = pack.blob_path(tokenizer_descriptor);
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
        format!(
            "failed to load benchmark tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    let encoding = tokenizer
        .encode(rendered, false)
        .map_err(|error| format!("failed to tokenize {profile} benchmark prompt: {error}"))?;
    let token_ids = encoding
        .get_ids()
        .iter()
        .map(|&token| i32::try_from(token).map_err(|error| error.into()))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    if token_ids.is_empty() {
        return Err(format!("{profile} benchmark prompt tokenized to no tokens").into());
    }
    Ok(Prompt {
        name: profile.to_owned(),
        description: description.to_owned(),
        token_ids,
    })
}

fn median_duration(samples: &[Measurement]) -> Duration {
    let mut durations: Vec<Duration> = samples.iter().map(|sample| sample.duration).collect();
    durations.sort_unstable();
    durations[durations.len() / 2]
}

fn median_prefill_duration(samples: &[Measurement]) -> Duration {
    let mut durations: Vec<Duration> = samples
        .iter()
        .map(|sample| sample.prefill_duration)
        .collect();
    durations.sort_unstable();
    durations[durations.len() / 2]
}

fn duration_spread_ratio(samples: &[Measurement]) -> f64 {
    let minimum = samples
        .iter()
        .map(|sample| sample.duration.as_secs_f64())
        .fold(f64::INFINITY, f64::min);
    let maximum = samples
        .iter()
        .map(|sample| sample.duration.as_secs_f64())
        .fold(0.0_f64, f64::max);
    maximum / minimum
}

fn median_f64(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn high_ordinary_token_ids(tokens: &[i32]) -> Vec<i32> {
    let mut high = tokens
        .iter()
        .copied()
        .filter(|token| (65_536..248_000).contains(token))
        .collect::<Vec<_>>();
    high.sort_unstable();
    high.dedup();
    high
}

fn prompt_draft_extra_token_ids(tokens: &[i32], vocab_limit: usize) -> Vec<i32> {
    let mut selected = std::collections::BTreeSet::new();
    for &token in tokens.iter().rev() {
        if ((vocab_limit as i32)..248_000).contains(&token) {
            selected.insert(token);
            if selected.len() == 256 {
                break;
            }
        }
    }
    selected.into_iter().collect()
}

fn mismatch_description(expected: &[i32], actual: &Measurement) -> String {
    let shared = expected.len().min(actual.tokens.len());
    if let Some(index) = (0..shared).find(|index| expected[*index] != actual.tokens[*index]) {
        let commit = actual
            .commits
            .iter()
            .find(|commit| {
                index >= commit.position && index < commit.position + commit.outputs.len()
            })
            .map(|commit| {
                format!(
                    "; commit position {} input {} outputs {:?} drafted {} accepted {}",
                    commit.position, commit.input, commit.outputs, commit.drafted, commit.accepted
                )
            })
            .unwrap_or_default();
        let history = actual
            .commits
            .iter()
            .map(|commit| {
                format!(
                    "{}:{:?}/{}/{}",
                    commit.position, commit.outputs, commit.drafted, commit.accepted
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        return format!(
            "token {} expected {} actual {}{}; history {}",
            index, expected[index], actual.tokens[index], commit, history
        );
    }
    format!(
        "length expected {} actual {}",
        expected.len(),
        actual.tokens.len()
    )
}

fn acceptance(sample: &Measurement) -> f64 {
    if sample.drafted == 0 {
        0.0
    } else {
        sample.accepted as f64 / sample.drafted as f64
    }
}

fn average(value: usize, iterations: usize) -> f64 {
    if iterations == 0 {
        0.0
    } else {
        value as f64 / iterations as f64
    }
}

fn token_sha256(tokens: &[i32]) -> String {
    let mut context = Context::new(&SHA256);
    for token in tokens {
        context.update(&token.to_le_bytes());
    }
    context
        .finish()
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn duration_nanoseconds(duration: Duration) -> u64 {
    duration.as_nanos().try_into().unwrap_or(u64::MAX)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn optional_usize(
    value: Option<std::ffi::OsString>,
    default: usize,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    match value {
        Some(value) => Ok(value
            .into_string()
            .map_err(|_| format!("{name} must be valid UTF-8"))?
            .parse()?),
        None => Ok(default),
    }
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
