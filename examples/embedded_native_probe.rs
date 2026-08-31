use generalist::provider::embedded::{
    ArrayFingerprint, Layer0CheckpointSpec, MlxQwenModel, MlxRuntime, ModelPack, NativeTensorSource,
};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const EMBEDDING_NAME: &str = "model.language_model.embed_tokens.weight";
const PROJECTION_NAME: &str = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
const PROJECTION_SCALE_NAME: &str =
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight.scale";
const PROJECTION_GLOBAL_SCALE_NAME: &str =
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight.global_scale";
const Z_PROJECTION_NAME: &str = "model.language_model.layers.0.linear_attn.in_proj_z.weight";
const BETA_PROJECTION_NAME: &str = "model.language_model.layers.0.linear_attn.in_proj_b.weight";
const ALPHA_PROJECTION_NAME: &str = "model.language_model.layers.0.linear_attn.in_proj_a.weight";
const CONV_WEIGHT_NAME: &str = "model.language_model.layers.0.linear_attn.conv1d.weight";
const DT_BIAS_NAME: &str = "model.language_model.layers.0.linear_attn.dt_bias";
const A_LOG_NAME: &str = "model.language_model.layers.0.linear_attn.A_log";
const GDN_NORM_NAME: &str = "model.language_model.layers.0.linear_attn.norm.weight";
const OUT_PROJECTION_NAME: &str = "model.language_model.layers.0.linear_attn.out_proj.weight";
const POST_NORM_NAME: &str = "model.language_model.layers.0.post_attention_layernorm.weight";
const MLP_GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const MLP_UP_NAME: &str = "model.language_model.layers.0.mlp.up_proj.weight";
const MLP_DOWN_NAME: &str = "model.language_model.layers.0.mlp.down_proj.weight";

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let metallib = required_path(arguments.next(), "METALLIB")?;
    let manifest = required_path(arguments.next(), "MANIFEST")?;
    let blob_root = required_path(arguments.next(), "BLOB_ROOT")?;
    let tensor_name = arguments
        .next()
        .ok_or("missing TENSOR_NAME argument")?
        .into_string()
        .map_err(|_| "TENSOR_NAME must be valid UTF-8")?;
    if arguments.next().is_some() {
        return Err("usage: embedded_native_probe METALLIB MANIFEST BLOB_ROOT TENSOR_NAME".into());
    }

    let pack = ModelPack::open_ollama(manifest, blob_root)?;
    let descriptor = pack.tensor(&tensor_name)?;
    let embedding_descriptor = pack.tensor(EMBEDDING_NAME)?;
    let projection_descriptor = pack.tensor(PROJECTION_NAME)?;
    let z_descriptor = pack.tensor(Z_PROJECTION_NAME)?;
    let beta_descriptor = pack.tensor(BETA_PROJECTION_NAME)?;
    let alpha_descriptor = pack.tensor(ALPHA_PROJECTION_NAME)?;
    let conv_descriptor = pack.tensor(CONV_WEIGHT_NAME)?;
    let dt_bias_descriptor = pack.tensor(DT_BIAS_NAME)?;
    let a_log_descriptor = pack.tensor(A_LOG_NAME)?;
    let gdn_norm_descriptor = pack.tensor(GDN_NORM_NAME)?;
    let out_projection_descriptor = pack.tensor(OUT_PROJECTION_NAME)?;
    let post_norm_descriptor = pack.tensor(POST_NORM_NAME)?;
    let mlp_gate_descriptor = pack.tensor(MLP_GATE_NAME)?;
    let mlp_up_descriptor = pack.tensor(MLP_UP_NAME)?;
    let mlp_down_descriptor = pack.tensor(MLP_DOWN_NAME)?;
    for blob in [
        &descriptor.blob,
        &embedding_descriptor.blob,
        &projection_descriptor.blob,
        &z_descriptor.blob,
        &beta_descriptor.blob,
        &alpha_descriptor.blob,
        &conv_descriptor.blob,
        &dt_bias_descriptor.blob,
        &a_log_descriptor.blob,
        &gdn_norm_descriptor.blob,
        &out_projection_descriptor.blob,
        &post_norm_descriptor.blob,
        &mlp_gate_descriptor.blob,
        &mlp_up_descriptor.blob,
        &mlp_down_descriptor.blob,
    ] {
        pack.verify_blob(blob)?;
    }
    let runtime = MlxRuntime::initialize(metallib)?;
    let result = runtime.probe_tensor(pack.blob_path(&descriptor.blob), &tensor_name)?;
    let norm = runtime.rms_norm_checkpoint(
        pack.blob_path(&descriptor.blob),
        &tensor_name,
        5120,
        1.0e-6,
        true,
    )?;
    let embedding = runtime.embedding_checkpoint(
        pack.blob_path(&embedding_descriptor.blob),
        EMBEDDING_NAME,
        pack.blob_path(&descriptor.blob),
        &tensor_name,
        42,
        5120,
        1.0e-6,
        true,
    )?;
    let projection = runtime.nvfp4_projection_checkpoint(
        pack.blob_path(&embedding_descriptor.blob),
        EMBEDDING_NAME,
        pack.blob_path(&descriptor.blob),
        &tensor_name,
        pack.blob_path(&projection_descriptor.blob),
        PROJECTION_NAME,
        PROJECTION_SCALE_NAME,
        PROJECTION_GLOBAL_SCALE_NAME,
        42,
        5120,
        10240,
        1.0e-6,
        true,
    )?;
    let embedding_path = pack.blob_path(&embedding_descriptor.blob);
    let input_norm_path = pack.blob_path(&descriptor.blob);
    let qkv_path = pack.blob_path(&projection_descriptor.blob);
    let z_path = pack.blob_path(&z_descriptor.blob);
    let beta_path = pack.blob_path(&beta_descriptor.blob);
    let alpha_path = pack.blob_path(&alpha_descriptor.blob);
    let conv_path = pack.blob_path(&conv_descriptor.blob);
    let dt_bias_path = pack.blob_path(&dt_bias_descriptor.blob);
    let a_log_path = pack.blob_path(&a_log_descriptor.blob);
    let gdn_norm_path = pack.blob_path(&gdn_norm_descriptor.blob);
    let out_projection_path = pack.blob_path(&out_projection_descriptor.blob);
    let post_norm_path = pack.blob_path(&post_norm_descriptor.blob);
    let mlp_gate_path = pack.blob_path(&mlp_gate_descriptor.blob);
    let mlp_up_path = pack.blob_path(&mlp_up_descriptor.blob);
    let mlp_down_path = pack.blob_path(&mlp_down_descriptor.blob);
    let layer0 = runtime.layer0_checkpoint(&Layer0CheckpointSpec {
        embedding: NativeTensorSource {
            path: &embedding_path,
            name: EMBEDDING_NAME,
        },
        input_norm: NativeTensorSource {
            path: &input_norm_path,
            name: &tensor_name,
        },
        qkv: NativeTensorSource {
            path: &qkv_path,
            name: PROJECTION_NAME,
        },
        z: NativeTensorSource {
            path: &z_path,
            name: Z_PROJECTION_NAME,
        },
        beta: NativeTensorSource {
            path: &beta_path,
            name: BETA_PROJECTION_NAME,
        },
        alpha: NativeTensorSource {
            path: &alpha_path,
            name: ALPHA_PROJECTION_NAME,
        },
        conv: NativeTensorSource {
            path: &conv_path,
            name: CONV_WEIGHT_NAME,
        },
        dt_bias: NativeTensorSource {
            path: &dt_bias_path,
            name: DT_BIAS_NAME,
        },
        a_log: NativeTensorSource {
            path: &a_log_path,
            name: A_LOG_NAME,
        },
        gdn_norm: NativeTensorSource {
            path: &gdn_norm_path,
            name: GDN_NORM_NAME,
        },
        out_proj: NativeTensorSource {
            path: &out_projection_path,
            name: OUT_PROJECTION_NAME,
        },
        post_norm: NativeTensorSource {
            path: &post_norm_path,
            name: POST_NORM_NAME,
        },
        mlp_gate: NativeTensorSource {
            path: &mlp_gate_path,
            name: MLP_GATE_NAME,
        },
        mlp_up: NativeTensorSource {
            path: &mlp_up_path,
            name: MLP_UP_NAME,
        },
        mlp_down: NativeTensorSource {
            path: &mlp_down_path,
            name: MLP_DOWN_NAME,
        },
        token_id: 42,
        input_width: 5120,
        qkv_width: 10240,
        z_width: 6144,
        ba_width: 96,
        mlp_width: 17408,
        epsilon: 1.0e-6,
        shift_norm_weight: true,
    })?;
    let load_started = Instant::now();
    let (mut model, model_load) = MlxQwenModel::load(&runtime, &pack)?;
    let load_elapsed = load_started.elapsed();
    let cold_decode_started = Instant::now();
    let decode = model.zero_state_decode_checkpoint(42)?;
    let cold_decode_elapsed = cold_decode_started.elapsed();
    let warm_decode_started = Instant::now();
    let warm_decode = model.zero_state_decode_checkpoint(42)?;
    let warm_decode_elapsed = warm_decode_started.elapsed();
    if warm_decode != decode {
        return Err("warm full-model decode did not reproduce the cold checkpoint".into());
    }
    let layer0_gdn = &layer0.gdn;
    let layer0_projections = &layer0_gdn.projections;

    println!("metallib={}", runtime.metallib_path().display());
    println!("tensor={tensor_name}");
    println!("dtype={} shape={:?}", result.dtype, result.shape);
    println!("elements={} bytes={}", result.elements, result.bytes);
    println!("gpu_sum={:.9}", result.gpu_sum);
    println!(
        "rms_norm_dtypes={}/{}/{} width={}",
        norm.input_dtype, norm.weight_dtype, norm.output_dtype, norm.width
    );
    println!(
        "rms_norm_sum={:.12} squared_sum={:.12}",
        norm.output_sum, norm.output_squared_sum
    );
    println!(
        "rms_norm_samples={:.9}/{:.9}/{:.9}",
        norm.output_first, norm.output_middle, norm.output_last
    );
    println!(
        "rms_norm_reference_error={:.12} fingerprint={:016x}",
        norm.max_abs_reference_error, norm.output_fingerprint
    );
    println!(
        "embedding_token={} dtype={} fingerprint={:016x} sum={:.12}",
        embedding.token_id,
        embedding.embedding.dtype,
        embedding.embedding.fingerprint,
        embedding.embedding.sum
    );
    println!(
        "embedding_norm_dtype={} fingerprint={:016x} sum={:.12}",
        embedding.normalized.dtype, embedding.normalized.fingerprint, embedding.normalized.sum
    );
    println!(
        "nvfp4_layout={}/{} dtypes={}/{}/{} global_scale={:.12} encoded_global_scale={:.12}",
        projection.group_size,
        projection.bits,
        projection.weight_dtype,
        projection.scale_dtype,
        projection.global_scale_dtype,
        projection.global_scale,
        projection.encoded_global_scale
    );
    print_fingerprint("nvfp4_normalized_input", &projection.normalized_input);
    print_fingerprint("nvfp4_raw_projection", &projection.raw_projection);
    print_fingerprint("nvfp4_output", &projection.output);
    print_fingerprint("nvfp4_fused_output", &projection.fused_output);
    println!(
        "layer0_projection_widths={}/{}/{}/{} global_scales={:.12}/{:.12}",
        layer0_projections.input_width,
        layer0_projections.qkv_width,
        layer0_projections.z_width,
        layer0_projections.ba_width,
        layer0_projections.qkv_global_scale,
        layer0_projections.z_global_scale
    );
    print_fingerprint(
        "layer0_normalized_input",
        &layer0_projections.normalized_input,
    );
    print_fingerprint("layer0_raw_qkvz", &layer0_projections.raw_qkvz);
    print_fingerprint("layer0_mixed_qkvz", &layer0_projections.mixed_qkvz);
    print_fingerprint("layer0_mixed_ba", &layer0_projections.mixed_ba);
    println!(
        "layer0_gdn_geometry=kernel{}/heads{}x{}/{}x{}",
        layer0_gdn.conv_kernel_width,
        layer0_gdn.key_heads,
        layer0_gdn.key_dim,
        layer0_gdn.value_heads,
        layer0_gdn.value_dim
    );
    print_fingerprint("layer0_conv_out", &layer0_gdn.conv_out);
    print_fingerprint("layer0_conv_state", &layer0_gdn.conv_state);
    print_fingerprint("layer0_a_exp", &layer0_gdn.a_exp);
    print_fingerprint("layer0_gdn_out", &layer0_gdn.gdn_out);
    print_fingerprint("layer0_delta_state", &layer0_gdn.delta_state);
    println!("layer0_mlp_width={}", layer0.mlp_width);
    print_fingerprint("layer0_gdn_norm", &layer0.gdn_norm);
    print_fingerprint("layer0_gated", &layer0.gated);
    print_fingerprint("layer0_attention_out", &layer0.attention_out);
    print_fingerprint("layer0_after_attention", &layer0.after_attention);
    print_fingerprint("layer0_post_norm", &layer0.post_norm);
    print_fingerprint("layer0_mlp_gate", &layer0.mlp_gate);
    print_fingerprint("layer0_mlp_up", &layer0.mlp_up);
    print_fingerprint("layer0_swiglu", &layer0.swiglu);
    print_fingerprint("layer0_mlp_out", &layer0.mlp_out);
    print_fingerprint("layer0_out", &layer0.layer_out);
    println!(
        "qwen_model_blobs={} arrays={} layers={}/{}/{} load_milliseconds={:.3}",
        model_load.blob_count,
        model_load.array_count,
        model_load.layer_count,
        model_load.linear_layer_count,
        model_load.full_attention_layer_count,
        load_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "qwen_decode_tokens={}/{} cold_milliseconds={:.3} warm_milliseconds={:.3}",
        decode.input_token_id,
        decode.output_token_id,
        cold_decode_elapsed.as_secs_f64() * 1000.0,
        warm_decode_elapsed.as_secs_f64() * 1000.0
    );
    print_fingerprint("qwen_decode_hidden", &decode.hidden);
    print_fingerprint("qwen_decode_logits", &decode.logits);
    Ok(())
}

fn print_fingerprint(name: &str, value: &ArrayFingerprint) {
    println!(
        "{name} dtype={} elements={} sum={:.12} squared_sum={:.12}",
        value.dtype, value.elements, value.sum, value.squared_sum
    );
    println!(
        "{name} samples={:.9}/{:.9}/{:.9} fingerprint={:016x}",
        value.first, value.middle, value.last, value.fingerprint
    );
}

fn required_path(value: Option<std::ffi::OsString>, name: &str) -> Result<PathBuf, Box<dyn Error>> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing {name} argument").into())
}
