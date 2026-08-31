use generalist::provider::embedded::{ArrayFingerprint, MlxQwenModel, MlxRuntime, ModelPack};
use serde_json::{json, Value};
use std::env;
use std::error::Error;
use std::path::PathBuf;

const EXPECTED_GREEDY_TOKENS: [i32; 16] = [
    1219, 73, 3706, 23102, 288, 37105, 163283, 627, 288, 728, 13373, 84, 1110, 469, 39111, 35306,
];

const EXPECTED_BATCH_OUTPUTS: [i32; 3] = [1219, 73, 3706];
const EXPECTED_BATCH_HIDDEN: [u64; 3] = [
    0x6dfc_18bc_2b1c_2044,
    0x6624_5d3b_40e3_4351,
    0xf67d_623f_dfbf_9893,
];
const EXPECTED_BATCH_LOGITS: [u64; 3] = [
    0x4efe_fa79_3523_b51f,
    0xf1c9_3f5e_efb9_64c8,
    0xa34e_d4ed_be19_8caf,
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    if arguments.next().is_some() {
        return Err("usage: embedded_mtp_checkpoint METALLIB MANIFEST BLOB_ROOT".into());
    }

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let runtime = MlxRuntime::initialize(metallib)?;
    let (mut model, load) = MlxQwenModel::load_with_mtp(&runtime, &pack)?;
    let checkpoint = model.mtp_checkpoint(42)?;
    let batch = model.batch_decode_checkpoint(&[42, 1219, 73])?;
    if load.blob_count != 866
        || load.array_count != 1684
        || load.mtp_layer_count != 1
        || checkpoint.input_token_id != 42
        || checkpoint.lookahead_token_id != 1219
        || checkpoint.draft_token_id != 73
        || checkpoint.target_token_id != 73
        || checkpoint.target_hidden.fingerprint != 0x6dfc_18bc_2b1c_2044
        || checkpoint.draft_hidden.fingerprint != 0x63c2_b418_5edb_3fa9
        || checkpoint.draft_logits.fingerprint != 0x4a60_ca77_b307_63ad
        || checkpoint.next_target_hidden.fingerprint != 0x6624_5d3b_40e3_4351
        || checkpoint.next_target_logits.fingerprint != 0xf1c9_3f5e_efb9_64c8
    {
        return Err("embedded MTP checkpoint diverged from the pinned Ollama oracle".into());
    }
    if batch.rows.len() != EXPECTED_BATCH_OUTPUTS.len()
        || batch.rows.iter().enumerate().any(|(index, row)| {
            row.input_token_id != [42, 1219, 73][index]
                || row.batch_output_token_id != EXPECTED_BATCH_OUTPUTS[index]
                || row.sequential_output_token_id != EXPECTED_BATCH_OUTPUTS[index]
                || row.batch_hidden != row.sequential_hidden
                || row.batch_logits != row.sequential_logits
                || row.batch_hidden.fingerprint != EXPECTED_BATCH_HIDDEN[index]
                || row.batch_logits.fingerprint != EXPECTED_BATCH_LOGITS[index]
        })
    {
        return Err("fused target batch diverged from sequential or pinned numerics".into());
    }

    let mut target_session = model.start_session()?;
    let mut target_tokens = Vec::with_capacity(EXPECTED_GREEDY_TOKENS.len());
    let mut target_input = 42;
    for expected_position in 0..EXPECTED_GREEDY_TOKENS.len() {
        let step = model.cached_decode_argmax(&mut target_session, target_input)?;
        if step.position != expected_position || step.input_token_id != target_input {
            return Err("ordinary target session returned an inconsistent position".into());
        }
        target_tokens.push(step.output_token_id);
        target_input = step.output_token_id;
    }

    let mut speculative_session = model.start_speculative_session()?;
    let seed = model.speculative_seed(&mut speculative_session, 42)?;
    if seed.position != 0
        || seed.input_token_id != 42
        || seed.output_token_ids != [EXPECTED_GREEDY_TOKENS[0]]
        || seed.drafted != 0
        || seed.accepted != 0
    {
        return Err("speculative seed diverged from the target oracle".into());
    }
    let mut speculative_tokens = seed.output_token_ids.clone();
    let mut speculative_steps = vec![json!({
        "position": seed.position,
        "input": seed.input_token_id,
        "outputs": seed.output_token_ids,
        "drafted": seed.drafted,
        "accepted": seed.accepted,
    })];
    while speculative_tokens.len() < EXPECTED_GREEDY_TOKENS.len() {
        let position = speculative_tokens.len();
        let input = *speculative_tokens
            .last()
            .ok_or("speculative stream unexpectedly empty")?;
        let remaining = EXPECTED_GREEDY_TOKENS.len() - position;
        let step = model.speculative_step(&mut speculative_session, 2, remaining.min(8))?;
        if step.position != position
            || step.input_token_id != input
            || step.output_token_ids.len() > remaining
        {
            return Err("speculative session returned an inconsistent commit".into());
        }
        speculative_tokens.extend_from_slice(&step.output_token_ids);
        speculative_steps.push(json!({
            "position": step.position,
            "input": step.input_token_id,
            "outputs": step.output_token_ids,
            "drafted": step.drafted,
            "accepted": step.accepted,
        }));
    }
    if target_tokens != EXPECTED_GREEDY_TOKENS
        || speculative_tokens != EXPECTED_GREEDY_TOKENS
        || speculative_tokens != target_tokens
    {
        return Err("speculative continuation diverged from exact greedy target output".into());
    }
    let batch_rows: Vec<Value> = batch
        .rows
        .iter()
        .map(|row| {
            json!({
                "input": row.input_token_id,
                "batch_output": row.batch_output_token_id,
                "sequential_output": row.sequential_output_token_id,
                "hidden_exact": row.batch_hidden == row.sequential_hidden,
                "logits_exact": row.batch_logits == row.sequential_logits,
                "batch_hidden": fingerprint(&row.batch_hidden),
                "sequential_hidden": fingerprint(&row.sequential_hidden),
                "batch_logits": fingerprint(&row.batch_logits),
                "sequential_logits": fingerprint(&row.sequential_logits),
            })
        })
        .collect();
    let output = json!({
        "load": {
            "blob_count": load.blob_count,
            "array_count": load.array_count,
            "layer_count": load.layer_count,
            "linear_layer_count": load.linear_layer_count,
            "full_attention_layer_count": load.full_attention_layer_count,
            "mtp_layer_count": load.mtp_layer_count,
        },
        "tokens": {
            "input": checkpoint.input_token_id,
            "lookahead": checkpoint.lookahead_token_id,
            "draft": checkpoint.draft_token_id,
            "target": checkpoint.target_token_id,
            "accepted": checkpoint.draft_token_id == checkpoint.target_token_id,
        },
        "target_hidden": fingerprint(&checkpoint.target_hidden),
        "draft_hidden": fingerprint(&checkpoint.draft_hidden),
        "draft_logits": fingerprint(&checkpoint.draft_logits),
        "next_target_hidden": fingerprint(&checkpoint.next_target_hidden),
        "next_target_logits": fingerprint(&checkpoint.next_target_logits),
        "batch_rows": batch_rows,
        "greedy_target_tokens": target_tokens,
        "speculative_tokens": speculative_tokens,
        "speculative_steps": speculative_steps,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn fingerprint(value: &ArrayFingerprint) -> Value {
    json!({
        "dtype": value.dtype,
        "elements": value.elements,
        "sum": format!("{:.12}", value.sum),
        "squared_sum": format!("{:.12}", value.squared_sum),
        "first": format!("{:.9}", value.first),
        "middle": format!("{:.9}", value.middle),
        "last": format!("{:.9}", value.last),
        "fingerprint": format!("{:016x}", value.fingerprint),
    })
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
