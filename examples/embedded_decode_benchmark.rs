use generalist::provider::embedded::{
    MlxQwenModel, MlxQwenSession, MlxRuntime, MlxTokenStep, ModelPack,
};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// This is the median exact zero-state checkpoint latency from the accepted
// seven-round run on the target M4 Pro. A large movement in this unchanged
// control means hot-path timings from the same run are not comparable.
const CLEAN_EXACT_CONTROL_MILLISECONDS: f64 = 72.549;
const MAX_ACCEPTED_CONTROL_SLOWDOWN: f64 = 1.65;

#[derive(Clone, Copy)]
enum StateSchedule {
    Eager,
    Deferred,
    Asynchronous,
}

#[derive(Clone, Copy)]
enum NormSchedule {
    Precomputed,
    Dynamic,
}

#[derive(Clone, Copy)]
enum MlpSchedule {
    Packed,
    Split,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    let measured_rounds = match arguments.next() {
        Some(value) => value
            .into_string()
            .map_err(|_| "ROUNDS must be valid UTF-8")?
            .parse::<usize>()?,
        None => 9,
    };
    if arguments.next().is_some() || !(1..=100).contains(&measured_rounds) {
        return Err(
            "usage: embedded_decode_benchmark METALLIB MANIFEST BLOB_ROOT [ROUNDS=9]".into(),
        );
    }

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let runtime = MlxRuntime::initialize(metallib)?;
    let load_started = Instant::now();
    let (mut model, load) = MlxQwenModel::load(&runtime, &pack)?;
    let load_elapsed = load_started.elapsed();

    let cold_started = Instant::now();
    let expected = model.zero_state_decode_checkpoint(42)?;
    let cold_elapsed = cold_started.elapsed();
    let fused_cold_started = Instant::now();
    let fused_expected = model.zero_state_decode_experimental_fused_checkpoint(42)?;
    let fused_cold_elapsed = fused_cold_started.elapsed();
    let mut exact_samples = Vec::with_capacity(measured_rounds);
    let mut fused_samples = Vec::with_capacity(measured_rounds);
    for round in 0..measured_rounds {
        for fused in [round % 2 != 0, round % 2 == 0] {
            let started = Instant::now();
            let actual = if fused {
                model.zero_state_decode_experimental_fused_checkpoint(42)?
            } else {
                model.zero_state_decode_checkpoint(42)?
            };
            let elapsed = started.elapsed();
            if fused {
                fused_samples.push(elapsed);
                if actual != fused_expected {
                    return Err("experimental fused decode was not deterministic".into());
                }
            } else {
                exact_samples.push(elapsed);
                if actual != expected {
                    return Err("measured decode changed the exact reference checkpoint".into());
                }
            }
        }
    }
    let comparison = model.compare_zero_state_fused(42)?;
    if comparison.exact != expected || comparison.experimental_fused != fused_expected {
        return Err("drift comparison did not reproduce the benchmark checkpoints".into());
    }
    let profile = model.zero_state_decode_profile(42)?;
    if profile.decode != expected {
        return Err("synchronized profile changed the exact reference checkpoint".into());
    }

    let mut exact_sorted = exact_samples.clone();
    exact_sorted.sort_unstable();
    let exact_minimum = exact_sorted[0];
    let exact_median = exact_sorted[exact_sorted.len() / 2];
    let exact_maximum = exact_sorted[exact_sorted.len() - 1];
    let exact_control_ratio = milliseconds(exact_median) / CLEAN_EXACT_CONTROL_MILLISECONDS;
    let initial_timing_status = if exact_control_ratio <= MAX_ACCEPTED_CONTROL_SLOWDOWN {
        "accepted"
    } else {
        "contaminated"
    };
    let mut fused_sorted = fused_samples.clone();
    fused_sorted.sort_unstable();
    let fused_minimum = fused_sorted[0];
    let fused_median = fused_sorted[fused_sorted.len() / 2];
    let fused_maximum = fused_sorted[fused_sorted.len() - 1];
    println!(
        "model_blobs={} arrays={} layers={}/{}/{}",
        load.blob_count,
        load.array_count,
        load.layer_count,
        load.linear_layer_count,
        load.full_attention_layer_count
    );
    println!(
        "load_milliseconds={:.3} exact_cold_milliseconds={:.3} fused_cold_milliseconds={:.3}",
        milliseconds(load_elapsed),
        milliseconds(cold_elapsed),
        milliseconds(fused_cold_elapsed)
    );
    println!(
        "rounds={} exact_min_milliseconds={:.3} exact_median_milliseconds={:.3} exact_max_milliseconds={:.3}",
        measured_rounds,
        milliseconds(exact_minimum),
        milliseconds(exact_median),
        milliseconds(exact_maximum)
    );
    println!(
        "fused_min_milliseconds={:.3} fused_median_milliseconds={:.3} fused_max_milliseconds={:.3}",
        milliseconds(fused_minimum),
        milliseconds(fused_median),
        milliseconds(fused_maximum)
    );
    println!(
        "exact_median_tokens_per_second={:.3} fused_median_tokens_per_second={:.3} fused_speedup={:.4}",
        1.0 / exact_median.as_secs_f64(),
        1.0 / fused_median.as_secs_f64(),
        exact_median.as_secs_f64() / fused_median.as_secs_f64()
    );
    println!(
        "initial_timing_status={} exact_control_reference_milliseconds={:.3} exact_control_ratio={:.3} maximum_accepted_control_slowdown={:.3}",
        initial_timing_status,
        CLEAN_EXACT_CONTROL_MILLISECONDS,
        exact_control_ratio,
        MAX_ACCEPTED_CONTROL_SLOWDOWN
    );
    println!(
        "tokens={}/{} hidden={:016x} logits={:016x}",
        expected.input_token_id,
        expected.output_token_id,
        expected.hidden.fingerprint,
        expected.logits.fingerprint
    );
    println!(
        "fused_tokens={}/{} hidden={:016x} logits={:016x}",
        fused_expected.input_token_id,
        fused_expected.output_token_id,
        fused_expected.hidden.fingerprint,
        fused_expected.logits.fingerprint
    );
    println!(
        "fused_hidden_sum_delta={:.12} fused_logits_sum_delta={:.12}",
        fused_expected.hidden.sum - expected.hidden.sum,
        fused_expected.logits.sum - expected.logits.sum
    );
    println!(
        "hidden_drift_max_abs={:.9} mean_abs={:.9} rms={:.9} relative_l2={:.9} cosine={:.12}",
        comparison.hidden_drift.max_abs,
        comparison.hidden_drift.mean_abs,
        comparison.hidden_drift.root_mean_square,
        comparison.hidden_drift.relative_l2,
        comparison.hidden_drift.cosine_similarity
    );
    println!(
        "logits_drift_max_abs={:.9} mean_abs={:.9} rms={:.9} relative_l2={:.9} cosine={:.12}",
        comparison.logits_drift.max_abs,
        comparison.logits_drift.mean_abs,
        comparison.logits_drift.root_mean_square,
        comparison.logits_drift.relative_l2,
        comparison.logits_drift.cosine_similarity
    );
    let linear_layers: Vec<f64> = profile
        .layer_milliseconds
        .iter()
        .enumerate()
        .filter_map(|(layer, &milliseconds)| (layer % 4 != 3).then_some(milliseconds))
        .collect();
    let full_attention_layers: Vec<f64> = profile
        .layer_milliseconds
        .iter()
        .enumerate()
        .filter_map(|(layer, &milliseconds)| (layer % 4 == 3).then_some(milliseconds))
        .collect();
    println!(
        "profile_graph_milliseconds={:.3} checkpoint_milliseconds={:.3} layer_sum_milliseconds={:.3}",
        profile.graph_milliseconds,
        profile.checkpoint_milliseconds,
        profile.layer_milliseconds.iter().sum::<f64>()
    );
    println!(
        "profile_linear_mean_milliseconds={:.3} full_attention_mean_milliseconds={:.3} final_norm_milliseconds={:.3} lm_head_argmax_milliseconds={:.3}",
        mean(&linear_layers),
        mean(&full_attention_layers),
        profile.final_norm_milliseconds,
        profile.lm_head_argmax_milliseconds
    );
    println!(
        "profile_layers={}",
        profile
            .layer_milliseconds
            .iter()
            .enumerate()
            .map(|(layer, milliseconds)| format!("{layer}:{milliseconds:.3}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    let mut session = model.start_session()?;
    let cached0_started = Instant::now();
    let cached0 = model.cached_decode_checkpoint(&mut session, 42)?;
    let cached0_elapsed = cached0_started.elapsed();
    if cached0.position != 0 || cached0.decode != expected {
        return Err("cached position-zero decode diverged from the exact reference".into());
    }
    let cached1_started = Instant::now();
    let cached1 = model.cached_decode_checkpoint(&mut session, cached0.decode.output_token_id)?;
    let cached1_elapsed = cached1_started.elapsed();
    println!(
        "cached_step1_actual_tokens={}/{} hidden={:016x} logits={:016x} hidden_sum={:.12} logits_sum={:.12}",
        cached1.decode.input_token_id,
        cached1.decode.output_token_id,
        cached1.decode.hidden.fingerprint,
        cached1.decode.logits.fingerprint,
        cached1.decode.hidden.sum,
        cached1.decode.logits.sum
    );
    if cached1.position != 1
        || cached1.decode.input_token_id != 1219
        || cached1.decode.output_token_id != 73
        || cached1.decode.hidden.fingerprint != 0x6624_5d3b_40e3_4351
        || cached1.decode.logits.fingerprint != 0xf1c9_3f5e_efb9_64c8
    {
        return Err("cached position-one decode diverged from the Ollama oracle".into());
    }
    session.reset()?;
    let reset0 = model.cached_decode_checkpoint(&mut session, 42)?;
    if reset0.position != 0 || reset0.decode != expected {
        return Err("reset cached decode did not reproduce position zero".into());
    }
    let mut deferred_session = model.start_session()?;
    let deferred_seed = model.cached_decode_checkpoint(&mut deferred_session, 42)?;
    if deferred_seed.position != 0 || deferred_seed.decode != expected {
        return Err("deferred-state benchmark seed diverged from position zero".into());
    }
    let mut async_session = model.start_session()?;
    let async_seed = model.cached_decode_checkpoint(&mut async_session, 42)?;
    if async_seed.position != 0 || async_seed.decode != expected {
        return Err("async-state benchmark seed diverged from position zero".into());
    }

    let mut hot_input = reset0.decode.output_token_id;
    let mut deferred_input = deferred_seed.decode.output_token_id;
    let mut async_input = async_seed.decode.output_token_id;
    let mut hot_samples = Vec::with_capacity(measured_rounds);
    let mut deferred_samples = Vec::with_capacity(measured_rounds);
    let mut async_samples = Vec::with_capacity(measured_rounds);
    let mut hot_tokens = Vec::with_capacity(measured_rounds);
    let mut deferred_tokens = Vec::with_capacity(measured_rounds);
    let mut async_tokens = Vec::with_capacity(measured_rounds);
    for step_index in 0..measured_rounds {
        let order = match step_index % 3 {
            0 => [
                StateSchedule::Eager,
                StateSchedule::Deferred,
                StateSchedule::Asynchronous,
            ],
            1 => [
                StateSchedule::Deferred,
                StateSchedule::Asynchronous,
                StateSchedule::Eager,
            ],
            _ => [
                StateSchedule::Asynchronous,
                StateSchedule::Eager,
                StateSchedule::Deferred,
            ],
        };
        let mut hot_step = None;
        let mut deferred_step = None;
        let mut async_step = None;
        for schedule in order {
            match schedule {
                StateSchedule::Eager => {
                    let measured = measured_cached_step(
                        &mut model,
                        &mut session,
                        hot_input,
                        StateSchedule::Eager,
                    )?;
                    hot_step = Some(measured.0);
                    hot_samples.push(measured.1);
                }
                StateSchedule::Deferred => {
                    let measured = measured_cached_step(
                        &mut model,
                        &mut deferred_session,
                        deferred_input,
                        StateSchedule::Deferred,
                    )?;
                    deferred_step = Some(measured.0);
                    deferred_samples.push(measured.1);
                }
                StateSchedule::Asynchronous => {
                    let measured = measured_cached_step(
                        &mut model,
                        &mut async_session,
                        async_input,
                        StateSchedule::Asynchronous,
                    )?;
                    async_step = Some(measured.0);
                    async_samples.push(measured.1);
                }
            }
        }
        let hot_step = hot_step.ok_or("eager lane was not measured")?;
        let deferred_step = deferred_step.ok_or("deferred lane was not measured")?;
        let async_step = async_step.ok_or("async lane was not measured")?;
        let expected_position = step_index + 1;
        if hot_step.position != expected_position
            || deferred_step.position != expected_position
            || async_step.position != expected_position
            || hot_step.input_token_id != hot_input
            || deferred_step.input_token_id != deferred_input
            || async_step.input_token_id != async_input
            || hot_step.output_token_id != deferred_step.output_token_id
            || hot_step.output_token_id != async_step.output_token_id
        {
            return Err("paired state-scheduling hot paths diverged".into());
        }
        if step_index == 0 && hot_step.output_token_id != 73 {
            return Err("paired hot paths diverged from the position-one oracle".into());
        }
        hot_input = hot_step.output_token_id;
        deferred_input = deferred_step.output_token_id;
        async_input = async_step.output_token_id;
        hot_tokens.push(hot_step.output_token_id);
        deferred_tokens.push(deferred_step.output_token_id);
        async_tokens.push(async_step.output_token_id);
    }
    if deferred_tokens != hot_tokens || async_tokens != hot_tokens {
        return Err("state scheduling changed the greedy token sequence".into());
    }
    let mut hot_sorted = hot_samples.clone();
    hot_sorted.sort_unstable();
    let hot_minimum = hot_sorted[0];
    let hot_median = hot_sorted[hot_sorted.len() / 2];
    let hot_maximum = hot_sorted[hot_sorted.len() - 1];
    let mut deferred_sorted = deferred_samples.clone();
    deferred_sorted.sort_unstable();
    let deferred_minimum = deferred_sorted[0];
    let deferred_median = deferred_sorted[deferred_sorted.len() / 2];
    let deferred_maximum = deferred_sorted[deferred_sorted.len() - 1];
    let mut async_sorted = async_samples.clone();
    async_sorted.sort_unstable();
    let async_minimum = async_sorted[0];
    let async_median = async_sorted[async_sorted.len() / 2];
    let async_maximum = async_sorted[async_sorted.len() - 1];
    let deferred_paired_speedup = paired_median_speedup(&hot_samples, &deferred_samples);
    let async_paired_speedup = paired_median_speedup(&hot_samples, &async_samples);

    let mut precomputed_norm_session = model.start_session()?;
    let precomputed_norm_seed =
        model.cached_decode_checkpoint(&mut precomputed_norm_session, 42)?;
    let mut dynamic_norm_session = model.start_session()?;
    let dynamic_norm_seed = model.cached_decode_checkpoint(&mut dynamic_norm_session, 42)?;
    if precomputed_norm_seed.position != 0
        || precomputed_norm_seed.decode != expected
        || dynamic_norm_seed != precomputed_norm_seed
    {
        return Err("norm A/B benchmark seed diverged from position zero".into());
    }
    let mut precomputed_norm_input = precomputed_norm_seed.decode.output_token_id;
    let mut dynamic_norm_input = dynamic_norm_seed.decode.output_token_id;
    let mut precomputed_norm_samples = Vec::with_capacity(measured_rounds);
    let mut dynamic_norm_samples = Vec::with_capacity(measured_rounds);
    let mut precomputed_norm_tokens = Vec::with_capacity(measured_rounds);
    let mut dynamic_norm_tokens = Vec::with_capacity(measured_rounds);
    for step_index in 0..measured_rounds {
        let order = if step_index % 2 == 0 {
            [NormSchedule::Precomputed, NormSchedule::Dynamic]
        } else {
            [NormSchedule::Dynamic, NormSchedule::Precomputed]
        };
        let mut precomputed_step = None;
        let mut dynamic_step = None;
        for schedule in order {
            match schedule {
                NormSchedule::Precomputed => {
                    let measured = measured_norm_step(
                        &mut model,
                        &mut precomputed_norm_session,
                        precomputed_norm_input,
                        NormSchedule::Precomputed,
                    )?;
                    precomputed_step = Some(measured.0);
                    precomputed_norm_samples.push(measured.1);
                }
                NormSchedule::Dynamic => {
                    let measured = measured_norm_step(
                        &mut model,
                        &mut dynamic_norm_session,
                        dynamic_norm_input,
                        NormSchedule::Dynamic,
                    )?;
                    dynamic_step = Some(measured.0);
                    dynamic_norm_samples.push(measured.1);
                }
            }
        }
        let precomputed_step = precomputed_step.ok_or("precomputed norm lane was not measured")?;
        let dynamic_step = dynamic_step.ok_or("dynamic norm lane was not measured")?;
        let expected_position = step_index + 1;
        if precomputed_step.position != expected_position
            || dynamic_step.position != expected_position
            || precomputed_step.input_token_id != precomputed_norm_input
            || dynamic_step.input_token_id != dynamic_norm_input
            || precomputed_step.output_token_id != dynamic_step.output_token_id
        {
            return Err("paired norm hot paths diverged".into());
        }
        if step_index == 0 && precomputed_step.output_token_id != 73 {
            return Err("paired norm hot paths diverged from the position-one oracle".into());
        }
        precomputed_norm_input = precomputed_step.output_token_id;
        dynamic_norm_input = dynamic_step.output_token_id;
        precomputed_norm_tokens.push(precomputed_step.output_token_id);
        dynamic_norm_tokens.push(dynamic_step.output_token_id);
    }
    if dynamic_norm_tokens != precomputed_norm_tokens {
        return Err("norm scheduling changed the greedy token sequence".into());
    }
    let mut precomputed_norm_sorted = precomputed_norm_samples.clone();
    precomputed_norm_sorted.sort_unstable();
    let precomputed_norm_median = precomputed_norm_sorted[precomputed_norm_sorted.len() / 2];
    let mut dynamic_norm_sorted = dynamic_norm_samples.clone();
    dynamic_norm_sorted.sort_unstable();
    let dynamic_norm_median = dynamic_norm_sorted[dynamic_norm_sorted.len() / 2];
    let precomputed_norm_paired_speedup =
        paired_median_speedup(&dynamic_norm_samples, &precomputed_norm_samples);

    let mut packed_mlp_session = model.start_session()?;
    let packed_mlp_seed = model.cached_decode_checkpoint(&mut packed_mlp_session, 42)?;
    let mut split_mlp_session = model.start_session()?;
    let split_mlp_seed = model.cached_decode_checkpoint(&mut split_mlp_session, 42)?;
    if packed_mlp_seed.position != 0
        || packed_mlp_seed.decode != expected
        || split_mlp_seed != packed_mlp_seed
    {
        return Err("MLP A/B benchmark seed diverged from position zero".into());
    }
    let mut packed_mlp_input = packed_mlp_seed.decode.output_token_id;
    let mut split_mlp_input = split_mlp_seed.decode.output_token_id;
    let mut packed_mlp_samples = Vec::with_capacity(measured_rounds);
    let mut split_mlp_samples = Vec::with_capacity(measured_rounds);
    let mut packed_mlp_tokens = Vec::with_capacity(measured_rounds);
    let mut split_mlp_tokens = Vec::with_capacity(measured_rounds);
    for step_index in 0..measured_rounds {
        let order = if step_index % 2 == 0 {
            [MlpSchedule::Packed, MlpSchedule::Split]
        } else {
            [MlpSchedule::Split, MlpSchedule::Packed]
        };
        let mut packed_step = None;
        let mut split_step = None;
        for schedule in order {
            match schedule {
                MlpSchedule::Packed => {
                    let measured = measured_mlp_step(
                        &mut model,
                        &mut packed_mlp_session,
                        packed_mlp_input,
                        MlpSchedule::Packed,
                    )?;
                    packed_step = Some(measured.0);
                    packed_mlp_samples.push(measured.1);
                }
                MlpSchedule::Split => {
                    let measured = measured_mlp_step(
                        &mut model,
                        &mut split_mlp_session,
                        split_mlp_input,
                        MlpSchedule::Split,
                    )?;
                    split_step = Some(measured.0);
                    split_mlp_samples.push(measured.1);
                }
            }
        }
        let packed_step = packed_step.ok_or("packed MLP lane was not measured")?;
        let split_step = split_step.ok_or("split MLP lane was not measured")?;
        let expected_position = step_index + 1;
        if packed_step.position != expected_position
            || split_step.position != expected_position
            || packed_step.input_token_id != packed_mlp_input
            || split_step.input_token_id != split_mlp_input
            || packed_step.output_token_id != split_step.output_token_id
        {
            return Err("paired MLP hot paths diverged".into());
        }
        if step_index == 0 && packed_step.output_token_id != 73 {
            return Err("paired MLP hot paths diverged from the position-one oracle".into());
        }
        packed_mlp_input = packed_step.output_token_id;
        split_mlp_input = split_step.output_token_id;
        packed_mlp_tokens.push(packed_step.output_token_id);
        split_mlp_tokens.push(split_step.output_token_id);
    }
    if split_mlp_tokens != packed_mlp_tokens {
        return Err("MLP projection packing changed the greedy token sequence".into());
    }
    let mut packed_mlp_sorted = packed_mlp_samples.clone();
    packed_mlp_sorted.sort_unstable();
    let packed_mlp_median = packed_mlp_sorted[packed_mlp_sorted.len() / 2];
    let mut split_mlp_sorted = split_mlp_samples.clone();
    split_mlp_sorted.sort_unstable();
    let split_mlp_median = split_mlp_sorted[split_mlp_sorted.len() / 2];
    let packed_mlp_paired_speedup = paired_median_speedup(&split_mlp_samples, &packed_mlp_samples);

    let mut deferred_validation_session = model.start_session()?;
    let deferred0 =
        model.cached_decode_argmax_deferred_state(&mut deferred_validation_session, 42)?;
    if deferred0.position != 0
        || deferred0.input_token_id != 42
        || deferred0.output_token_id != 1219
    {
        return Err("deferred-state position-zero token diverged from the oracle".into());
    }
    let deferred1 = model
        .cached_decode_checkpoint(&mut deferred_validation_session, deferred0.output_token_id)?;
    if deferred1 != cached1 {
        return Err("deferred-state cache diverged at the next exact checkpoint".into());
    }
    let mut async_validation_session = model.start_session()?;
    let async0 = model.cached_decode_argmax_async_state(&mut async_validation_session, 42)?;
    if async0.position != 0 || async0.input_token_id != 42 || async0.output_token_id != 1219 {
        return Err("async-state position-zero token diverged from the oracle".into());
    }
    let async1 =
        model.cached_decode_checkpoint(&mut async_validation_session, async0.output_token_id)?;
    if async1 != cached1 {
        return Err("async-state cache diverged at the next exact checkpoint".into());
    }
    println!(
        "cached_step0_milliseconds={:.3} cached_step1_milliseconds={:.3}",
        milliseconds(cached0_elapsed),
        milliseconds(cached1_elapsed)
    );
    println!(
        "cached_step1_tokens={}/{} hidden={:016x} logits={:016x}",
        cached1.decode.input_token_id,
        cached1.decode.output_token_id,
        cached1.decode.hidden.fingerprint,
        cached1.decode.logits.fingerprint
    );
    println!(
        "argmax_steps={} argmax_min_milliseconds={:.3} argmax_median_milliseconds={:.3} argmax_max_milliseconds={:.3} argmax_median_tokens_per_second={:.3}",
        measured_rounds,
        milliseconds(hot_minimum),
        milliseconds(hot_median),
        milliseconds(hot_maximum),
        1.0 / hot_median.as_secs_f64()
    );
    println!(
        "argmax_output_tokens={}",
        hot_tokens
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "deferred_argmax_steps={} deferred_argmax_min_milliseconds={:.3} deferred_argmax_median_milliseconds={:.3} deferred_argmax_max_milliseconds={:.3} deferred_argmax_median_tokens_per_second={:.3} deferred_argmax_speedup={:.4} deferred_paired_median_speedup={:.4}",
        measured_rounds,
        milliseconds(deferred_minimum),
        milliseconds(deferred_median),
        milliseconds(deferred_maximum),
        1.0 / deferred_median.as_secs_f64(),
        hot_median.as_secs_f64() / deferred_median.as_secs_f64(),
        deferred_paired_speedup
    );
    println!(
        "deferred_argmax_output_tokens={}",
        deferred_tokens
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "async_argmax_steps={} async_argmax_min_milliseconds={:.3} async_argmax_median_milliseconds={:.3} async_argmax_max_milliseconds={:.3} async_argmax_median_tokens_per_second={:.3} async_argmax_speedup={:.4} async_paired_median_speedup={:.4}",
        measured_rounds,
        milliseconds(async_minimum),
        milliseconds(async_median),
        milliseconds(async_maximum),
        1.0 / async_median.as_secs_f64(),
        hot_median.as_secs_f64() / async_median.as_secs_f64(),
        async_paired_speedup
    );
    println!(
        "async_argmax_output_tokens={}",
        async_tokens
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "norm_ab_steps={} precomputed_norm_median_milliseconds={:.3} dynamic_norm_median_milliseconds={:.3} precomputed_norm_speedup={:.4} precomputed_norm_paired_median_speedup={:.4}",
        measured_rounds,
        milliseconds(precomputed_norm_median),
        milliseconds(dynamic_norm_median),
        dynamic_norm_median.as_secs_f64() / precomputed_norm_median.as_secs_f64(),
        precomputed_norm_paired_speedup
    );
    println!(
        "norm_ab_output_tokens={}",
        precomputed_norm_tokens
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "mlp_ab_steps={} packed_mlp_median_milliseconds={:.3} split_mlp_median_milliseconds={:.3} packed_mlp_speedup={:.4} packed_mlp_paired_median_speedup={:.4}",
        measured_rounds,
        milliseconds(packed_mlp_median),
        milliseconds(split_mlp_median),
        split_mlp_median.as_secs_f64() / packed_mlp_median.as_secs_f64(),
        packed_mlp_paired_speedup
    );
    println!(
        "mlp_ab_output_tokens={}",
        packed_mlp_tokens
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let post_control_started = Instant::now();
    let post_control = model.zero_state_decode_checkpoint(42)?;
    let post_control_elapsed = post_control_started.elapsed();
    if post_control != expected {
        return Err("post-run control changed the exact reference checkpoint".into());
    }
    let post_control_ratio = milliseconds(post_control_elapsed) / CLEAN_EXACT_CONTROL_MILLISECONDS;
    let timing_status = if initial_timing_status == "accepted"
        && post_control_ratio <= MAX_ACCEPTED_CONTROL_SLOWDOWN
    {
        "accepted"
    } else {
        "contaminated"
    };
    println!(
        "timing_status={} post_control_milliseconds={:.3} post_control_ratio={:.3}",
        timing_status,
        milliseconds(post_control_elapsed),
        post_control_ratio
    );
    if timing_status == "contaminated" {
        eprintln!(
            "warning: benchmark timings are contaminated by system load; semantic checks remain valid"
        );
    }
    Ok(())
}

fn measured_cached_step(
    model: &mut MlxQwenModel,
    session: &mut MlxQwenSession,
    input_token_id: i32,
    schedule: StateSchedule,
) -> Result<(MlxTokenStep, Duration), Box<dyn Error>> {
    let started = Instant::now();
    let step = match schedule {
        StateSchedule::Eager => model.cached_decode_argmax(session, input_token_id)?,
        StateSchedule::Deferred => {
            model.cached_decode_argmax_deferred_state(session, input_token_id)?
        }
        StateSchedule::Asynchronous => {
            model.cached_decode_argmax_async_state(session, input_token_id)?
        }
    };
    Ok((step, started.elapsed()))
}

fn measured_norm_step(
    model: &mut MlxQwenModel,
    session: &mut MlxQwenSession,
    input_token_id: i32,
    schedule: NormSchedule,
) -> Result<(MlxTokenStep, Duration), Box<dyn Error>> {
    let started = Instant::now();
    let step = match schedule {
        NormSchedule::Precomputed => model.cached_decode_argmax(session, input_token_id)?,
        NormSchedule::Dynamic => {
            model.cached_decode_argmax_dynamic_norm(session, input_token_id)?
        }
    };
    Ok((step, started.elapsed()))
}

fn measured_mlp_step(
    model: &mut MlxQwenModel,
    session: &mut MlxQwenSession,
    input_token_id: i32,
    schedule: MlpSchedule,
) -> Result<(MlxTokenStep, Duration), Box<dyn Error>> {
    let started = Instant::now();
    let step = match schedule {
        MlpSchedule::Packed => model.cached_decode_argmax(session, input_token_id)?,
        MlpSchedule::Split => model.cached_decode_argmax_split_mlp(session, input_token_id)?,
    };
    Ok((step, started.elapsed()))
}

fn paired_median_speedup(reference: &[Duration], candidate: &[Duration]) -> f64 {
    let mut ratios: Vec<f64> = reference
        .iter()
        .zip(candidate)
        .map(|(reference, candidate)| reference.as_secs_f64() / candidate.as_secs_f64())
        .collect();
    ratios.sort_by(f64::total_cmp);
    ratios[ratios.len() / 2]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
