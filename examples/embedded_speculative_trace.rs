use generalist::provider::embedded::{MlxQwenModel, MlxRuntime, ModelPack};
use std::env;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    if arguments.next().is_some() {
        return Err("usage: embedded_speculative_trace METALLIB MANIFEST BLOB_ROOT".into());
    }

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let runtime = MlxRuntime::initialize(metallib)?;
    let (mut model, load) = MlxQwenModel::load_with_mtp(&runtime, &pack)?;
    let mut session = model.start_speculative_session()?;
    // Match the embedded provider's production decode profile. Leaving these
    // at the native cache defaults traces the retained differential oracles,
    // not the path users exercise through `generalist --embedded`.
    session.set_verification_qmv_results_per_simdgroup(4)?;
    session.set_verification_wide_qmv(true)?;
    session.set_mtp_draft_vocab_limit(65_536)?;
    session.set_direct_cache_rollback(true)?;
    let seed = model.speculative_seed(&mut session, 42)?;
    let step = model.speculative_step(&mut session, 2, 8)?;
    if seed.output_token_ids != [1219]
        || step.position != 1
        || step.input_token_id != 1219
        || step.output_token_ids != [73, 3706]
        || step.drafted != 2
        || step.accepted != 1
    {
        return Err("speculative trace diverged from the frozen target oracle".into());
    }

    println!(
        "layers={}/{}/{} seed={} outputs={} proposal_ms={:.3} verification_ms={:.3} replay_ms={:.3} correction_ms={:.3}",
        load.layer_count,
        load.linear_layer_count,
        load.full_attention_layer_count,
        seed.output_token_ids[0],
        step
            .output_token_ids
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        step.proposal_milliseconds,
        step.verification_milliseconds,
        step.replay_milliseconds,
        step.correction_milliseconds,
    );
    Ok(())
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
