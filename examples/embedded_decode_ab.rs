use generalist::provider::embedded::{
    MlxQwenModel, MlxQwenSession, MlxRuntime, MlxTokenStep, ModelPack,
};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const CLEAN_EXACT_CONTROL_MILLISECONDS: f64 = 72.549;
const MAX_ACCEPTED_CONTROL_SLOWDOWN: f64 = 1.65;

#[derive(Clone, Copy)]
enum Experiment {
    Norm,
    Mlp,
    Scale,
    Qmv,
}

impl Experiment {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "norm" => Ok(Self::Norm),
            "mlp" => Ok(Self::Mlp),
            "scale" => Ok(Self::Scale),
            "qmv" => Ok(Self::Qmv),
            _ => Err("EXPERIMENT must be norm, mlp, scale, or qmv".into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Norm => "norm",
            Self::Mlp => "mlp",
            Self::Scale => "scale",
            Self::Qmv => "qmv",
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    let experiment = Experiment::parse(
        &arguments
            .next()
            .ok_or("missing EXPERIMENT")?
            .into_string()
            .map_err(|_| "EXPERIMENT must be valid UTF-8")?,
    )?;
    let rounds = match arguments.next() {
        Some(value) => value
            .into_string()
            .map_err(|_| "ROUNDS must be valid UTF-8")?
            .parse::<usize>()?,
        None => 127,
    };
    if arguments.next().is_some() || !(3..=255).contains(&rounds) {
        return Err(
            "usage: embedded_decode_ab METALLIB MANIFEST BLOB_ROOT EXPERIMENT [ROUNDS=127]".into(),
        );
    }

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let runtime = MlxRuntime::initialize(metallib)?;
    let (mut model, load) = MlxQwenModel::load(&runtime, &pack)?;

    let expected = model.zero_state_decode_checkpoint(42)?;
    if expected.output_token_id != 1219
        || expected.hidden.fingerprint != 0x6dfc_18bc_2b1c_2044
        || expected.logits.fingerprint != 0x4efe_fa79_3523_b51f
    {
        return Err("A/B warmup diverged from the exact oracle".into());
    }
    let initial_control_started = Instant::now();
    let initial_control_checkpoint = model.zero_state_decode_checkpoint(42)?;
    let initial_control = initial_control_started.elapsed();
    if initial_control_checkpoint != expected {
        return Err("A/B initial control diverged from the exact warmup".into());
    }

    let mut candidate_session = model.start_session()?;
    let candidate_seed = model.cached_decode_checkpoint(&mut candidate_session, 42)?;
    let mut baseline_session = model.start_session()?;
    let baseline_seed = match experiment {
        Experiment::Scale => {
            model.cached_decode_unfused_scale_checkpoint(&mut baseline_session, 42)?
        }
        Experiment::Norm | Experiment::Mlp => {
            model.cached_decode_checkpoint(&mut baseline_session, 42)?
        }
        Experiment::Qmv => model.cached_decode_wide_qmv_checkpoint(&mut baseline_session, 42)?,
    };
    if candidate_seed.position != 0
        || candidate_seed.decode != expected
        || baseline_seed != candidate_seed
    {
        return Err("A/B session seed diverged from the exact oracle".into());
    }

    let mut candidate_input = candidate_seed.decode.output_token_id;
    let mut baseline_input = baseline_seed.decode.output_token_id;
    let mut candidate_samples = Vec::with_capacity(rounds);
    let mut baseline_samples = Vec::with_capacity(rounds);
    let mut candidate_tokens = Vec::with_capacity(rounds);
    for step_index in 0..rounds {
        let candidate_first = step_index % 2 == 0;
        let (candidate_step, candidate_elapsed, baseline_step, baseline_elapsed) =
            if candidate_first {
                let candidate = measured_step(
                    &mut model,
                    &mut candidate_session,
                    candidate_input,
                    experiment,
                    true,
                )?;
                let baseline = measured_step(
                    &mut model,
                    &mut baseline_session,
                    baseline_input,
                    experiment,
                    false,
                )?;
                (candidate.0, candidate.1, baseline.0, baseline.1)
            } else {
                let baseline = measured_step(
                    &mut model,
                    &mut baseline_session,
                    baseline_input,
                    experiment,
                    false,
                )?;
                let candidate = measured_step(
                    &mut model,
                    &mut candidate_session,
                    candidate_input,
                    experiment,
                    true,
                )?;
                (candidate.0, candidate.1, baseline.0, baseline.1)
            };
        let expected_position = step_index + 1;
        if candidate_step.position != expected_position
            || baseline_step.position != expected_position
            || candidate_step.input_token_id != candidate_input
            || baseline_step.input_token_id != baseline_input
            || candidate_step.output_token_id != baseline_step.output_token_id
        {
            return Err("A/B cached paths diverged".into());
        }
        if step_index == 0 && candidate_step.output_token_id != 73 {
            return Err("A/B position-one token diverged from the exact oracle".into());
        }
        candidate_input = candidate_step.output_token_id;
        baseline_input = baseline_step.output_token_id;
        candidate_samples.push(candidate_elapsed);
        baseline_samples.push(baseline_elapsed);
        candidate_tokens.push(candidate_step.output_token_id);
    }

    let candidate_final =
        model.cached_decode_checkpoint(&mut candidate_session, candidate_input)?;
    let baseline_final = match experiment {
        Experiment::Qmv => {
            model.cached_decode_wide_qmv_checkpoint(&mut baseline_session, baseline_input)?
        }
        Experiment::Norm | Experiment::Mlp | Experiment::Scale => {
            model.cached_decode_checkpoint(&mut baseline_session, baseline_input)?
        }
    };
    if candidate_final != baseline_final {
        return Err("A/B final full-tensor checkpoints diverged".into());
    }

    let post_control_started = Instant::now();
    let post_control_checkpoint = model.zero_state_decode_checkpoint(42)?;
    let post_control = post_control_started.elapsed();
    if post_control_checkpoint != expected {
        return Err("A/B post-run control diverged from the exact oracle".into());
    }

    let candidate_median = median_duration(&candidate_samples);
    let baseline_median = median_duration(&baseline_samples);
    let paired_ratios: Vec<f64> = baseline_samples
        .iter()
        .zip(&candidate_samples)
        .map(|(baseline, candidate)| baseline.as_secs_f64() / candidate.as_secs_f64())
        .collect();
    let paired_speedup = median_f64(&paired_ratios);
    let paired_differences: Vec<f64> = baseline_samples
        .iter()
        .zip(&candidate_samples)
        .map(|(baseline, candidate)| {
            (baseline.as_secs_f64() - candidate.as_secs_f64()) * 1_000_000.0
        })
        .collect();
    let median_difference_microseconds = median_f64(&paired_differences);
    let candidate_wins = baseline_samples
        .iter()
        .zip(&candidate_samples)
        .filter(|(baseline, candidate)| candidate < baseline)
        .count();
    let control_limit = CLEAN_EXACT_CONTROL_MILLISECONDS * MAX_ACCEPTED_CONTROL_SLOWDOWN;
    let timing_status = if milliseconds(initial_control) <= control_limit
        && milliseconds(post_control) <= control_limit
    {
        "accepted"
    } else {
        "contaminated"
    };

    println!(
        "experiment={} rounds={} model_blobs={} arrays={} layers={}/{}/{}",
        experiment.name(),
        rounds,
        load.blob_count,
        load.array_count,
        load.layer_count,
        load.linear_layer_count,
        load.full_attention_layer_count
    );
    println!(
        "candidate_median_milliseconds={:.3} baseline_median_milliseconds={:.3} median_speedup={:.6}",
        milliseconds(candidate_median),
        milliseconds(baseline_median),
        baseline_median.as_secs_f64() / candidate_median.as_secs_f64()
    );
    println!(
        "paired_median_speedup={:.6} paired_median_difference_microseconds={:.3} candidate_wins={}/{}",
        paired_speedup, median_difference_microseconds, candidate_wins, rounds
    );
    println!(
        "final_position={} final_tokens={}/{} final_hidden={:016x} final_logits={:016x}",
        candidate_final.position,
        candidate_final.decode.input_token_id,
        candidate_final.decode.output_token_id,
        candidate_final.decode.hidden.fingerprint,
        candidate_final.decode.logits.fingerprint
    );
    println!(
        "continuation_fingerprint={:016x}",
        token_fingerprint(&candidate_tokens)
    );
    println!(
        "timing_status={} initial_control_milliseconds={:.3} post_control_milliseconds={:.3} control_limit_milliseconds={:.3}",
        timing_status,
        milliseconds(initial_control),
        milliseconds(post_control),
        control_limit
    );
    if timing_status == "contaminated" {
        eprintln!(
            "warning: benchmark timings are contaminated by system load; semantic checks remain valid"
        );
    }
    Ok(())
}

fn measured_step(
    model: &mut MlxQwenModel,
    session: &mut MlxQwenSession,
    input_token_id: i32,
    experiment: Experiment,
    candidate: bool,
) -> Result<(MlxTokenStep, Duration), Box<dyn Error>> {
    let started = Instant::now();
    let step = match (experiment, candidate) {
        (Experiment::Scale, true) | (Experiment::Norm, true) | (Experiment::Mlp, true) => {
            model.cached_decode_argmax(session, input_token_id)?
        }
        (Experiment::Qmv, true) => model.cached_decode_argmax(session, input_token_id)?,
        (Experiment::Qmv, false) => model.cached_decode_argmax_wide_qmv(session, input_token_id)?,
        (Experiment::Scale, false) => {
            model.cached_decode_argmax_unfused_scale(session, input_token_id)?
        }
        (Experiment::Norm, false) => {
            model.cached_decode_argmax_dynamic_norm(session, input_token_id)?
        }
        (Experiment::Mlp, false) => {
            model.cached_decode_argmax_split_mlp(session, input_token_id)?
        }
    };
    Ok((step, started.elapsed()))
}

fn median_duration(values: &[Duration]) -> Duration {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn median_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn token_fingerprint(tokens: &[i32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
