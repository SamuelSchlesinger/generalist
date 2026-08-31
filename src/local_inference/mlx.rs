//! Narrow Rust ownership boundary over the statically linked MLX runtime.
//!
//! This checkpoint API deliberately exposes no raw MLX handles. The next model
//! port can extend the native shim while Rust retains lifetime, thread, model
//! pack, cancellation, and provider ownership.

// The feature-disabled branches use explicit early returns so the native
// implementation can remain in the same function behind its own cfg block.
#![allow(clippy::needless_return)]

#[cfg(generalist_mlx_native_linked)]
use super::model_pack::ModelComponent;
use super::model_pack::ModelPack;
use crate::error::{Error, Result};
#[cfg(generalist_mlx_native_linked)]
use std::collections::HashSet;
#[cfg(generalist_mlx_native_linked)]
use std::ffi::{c_char, c_int, CStr, CString};
use std::marker::PhantomData;
use std::path::Path;
#[cfg(generalist_mlx_native_linked)]
use std::path::PathBuf;
#[cfg(generalist_mlx_native_linked)]
use std::ptr::NonNull;
#[cfg(generalist_mlx_native_linked)]
use std::sync::OnceLock;
use std::{cell::Cell, fmt};

#[cfg(generalist_mlx_native_linked)]
const ERROR_CAPACITY: usize = 1024;
#[cfg(generalist_mlx_native_linked)]
const MAX_TENSOR_DIMS: usize = 8;
#[cfg(generalist_mlx_native_linked)]
const QWEN_LAYER_COUNT: usize = 64;
#[cfg(generalist_mlx_native_linked)]
const QWEN_MAX_BATCH_CHECKPOINT_TOKENS: usize = 8;
#[cfg(generalist_mlx_native_linked)]
const QWEN_MAX_PREFILL_CHUNK_TOKENS: usize = 512;
#[cfg(generalist_mlx_native_linked)]
const QWEN_MAX_MTP_DRAFT_EXTRA_TOKENS: usize = 256;
#[cfg(generalist_mlx_native_linked)]
static METALLIB_PATH: OnceLock<PathBuf> = OnceLock::new();

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ProbeResult {
    dtype: c_int,
    ndim: usize,
    shape: [c_int; MAX_TENSOR_DIMS],
    elements: usize,
    bytes: usize,
    gpu_sum: f32,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RmsNormResult {
    input_dtype: c_int,
    weight_dtype: c_int,
    output_dtype: c_int,
    width: usize,
    output_sum: f64,
    output_squared_sum: f64,
    output_first: f32,
    output_middle: f32,
    output_last: f32,
    max_abs_reference_error: f64,
    output_fingerprint: u64,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ArraySummaryResult {
    dtype: c_int,
    elements: usize,
    sum: f64,
    squared_sum: f64,
    first: f32,
    middle: f32,
    last: f32,
    fingerprint: u64,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct EmbeddingResult {
    token_id: c_int,
    width: usize,
    embedding: ArraySummaryResult,
    normalized: ArraySummaryResult,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Nvfp4ProjectionResult {
    token_id: c_int,
    input_width: usize,
    output_width: usize,
    group_size: c_int,
    bits: c_int,
    weight_dtype: c_int,
    scale_dtype: c_int,
    global_scale_dtype: c_int,
    global_scale: f32,
    encoded_global_scale: f32,
    normalized_input: ArraySummaryResult,
    raw_projection: ArraySummaryResult,
    output: ArraySummaryResult,
    fused_output: ArraySummaryResult,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Layer0ProjectionResult {
    token_id: c_int,
    input_width: usize,
    qkv_width: usize,
    z_width: usize,
    ba_width: usize,
    qkv_global_scale: f32,
    z_global_scale: f32,
    normalized_input: ArraySummaryResult,
    raw_qkvz: ArraySummaryResult,
    mixed_qkvz: ArraySummaryResult,
    mixed_ba: ArraySummaryResult,
    gdn_executed: c_int,
    conv_kernel_width: usize,
    key_heads: usize,
    key_dim: usize,
    value_heads: usize,
    value_dim: usize,
    conv_out: ArraySummaryResult,
    conv_state: ArraySummaryResult,
    a_exp: ArraySummaryResult,
    gdn_out: ArraySummaryResult,
    delta_state: ArraySummaryResult,
    full_layer_executed: c_int,
    mlp_width: usize,
    gdn_norm: ArraySummaryResult,
    gated: ArraySummaryResult,
    attention_out: ArraySummaryResult,
    after_attention: ArraySummaryResult,
    post_norm: ArraySummaryResult,
    mlp_gate: ArraySummaryResult,
    mlp_up: ArraySummaryResult,
    swiglu: ArraySummaryResult,
    mlp_out: ArraySummaryResult,
    layer_out: ArraySummaryResult,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
struct ModelBlobSource {
    path: *const c_char,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ModelLoadResult {
    blob_count: usize,
    array_count: usize,
    layer_count: usize,
    linear_layer_count: usize,
    full_attention_layer_count: usize,
    mtp_layer_count: usize,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct DecodeResult {
    input_token_id: c_int,
    output_token_id: c_int,
    hidden: ArraySummaryResult,
    logits: ArraySummaryResult,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CachedDecodeResult {
    position: usize,
    decode: DecodeResult,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct TokenStepResult {
    position: usize,
    input_token_id: c_int,
    output_token_id: c_int,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct MtpCheckpointResult {
    input_token_id: c_int,
    lookahead_token_id: c_int,
    draft_token_id: c_int,
    target_token_id: c_int,
    target_hidden: ArraySummaryResult,
    draft_hidden: ArraySummaryResult,
    draft_logits: ArraySummaryResult,
    next_target_hidden: ArraySummaryResult,
    next_target_logits: ArraySummaryResult,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct BatchDecodeCheckpointResult {
    token_count: usize,
    input_token_ids: [c_int; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
    batch_output_token_ids: [c_int; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
    sequential_output_token_ids: [c_int; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
    batch_hidden: [ArraySummaryResult; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
    batch_logits: [ArraySummaryResult; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
    sequential_hidden: [ArraySummaryResult; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
    sequential_logits: [ArraySummaryResult; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SpeculativeStepResult {
    position: usize,
    input_token_id: c_int,
    output_count: usize,
    output_token_ids: [c_int; QWEN_MAX_BATCH_CHECKPOINT_TOKENS],
    drafted: usize,
    accepted: usize,
    proposal_milliseconds: f64,
    verification_milliseconds: f64,
    replay_milliseconds: f64,
    correction_milliseconds: f64,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DecodeProfileResult {
    decode: DecodeResult,
    layer_milliseconds: [f64; QWEN_LAYER_COUNT],
    final_norm_milliseconds: f64,
    lm_head_argmax_milliseconds: f64,
    graph_milliseconds: f64,
    checkpoint_milliseconds: f64,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct ArrayDriftResult {
    max_abs: f64,
    mean_abs: f64,
    root_mean_square: f64,
    relative_l2: f64,
    cosine_similarity: f64,
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct DecodeComparisonResult {
    exact: DecodeResult,
    experimental_fused: DecodeResult,
    hidden_drift: ArrayDriftResult,
    logits_drift: ArrayDriftResult,
}

#[cfg(generalist_mlx_native_linked)]
impl Default for DecodeProfileResult {
    fn default() -> Self {
        Self {
            decode: DecodeResult::default(),
            layer_milliseconds: [0.0; QWEN_LAYER_COUNT],
            final_norm_milliseconds: 0.0,
            lm_head_argmax_milliseconds: 0.0,
            graph_milliseconds: 0.0,
            checkpoint_milliseconds: 0.0,
        }
    }
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
struct NativeQwenModel {
    _private: [u8; 0],
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
struct NativeQwenCache {
    _private: [u8; 0],
}

#[cfg(generalist_mlx_native_linked)]
#[repr(C)]
struct NativeQwenSpeculativeCache {
    _private: [u8; 0],
}

#[cfg(generalist_mlx_native_linked)]
impl Default for ProbeResult {
    fn default() -> Self {
        Self {
            dtype: 0,
            ndim: 0,
            shape: [0; MAX_TENSOR_DIMS],
            elements: 0,
            bytes: 0,
            gpu_sum: 0.0,
        }
    }
}

#[cfg(generalist_mlx_native_linked)]
unsafe extern "C" {
    fn generalist_mlx_initialize(
        metallib_path: *const c_char,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_probe_tensor(
        safetensors_path: *const c_char,
        tensor_name: *const c_char,
        result: *mut ProbeResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_rms_norm_checkpoint(
        safetensors_path: *const c_char,
        weight_name: *const c_char,
        width: usize,
        epsilon: f32,
        shift_weight: c_int,
        result: *mut RmsNormResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_embedding_checkpoint(
        embedding_path: *const c_char,
        embedding_name: *const c_char,
        norm_path: *const c_char,
        norm_name: *const c_char,
        token_id: c_int,
        width: usize,
        epsilon: f32,
        shift_norm_weight: c_int,
        result: *mut EmbeddingResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_nvfp4_projection_checkpoint(
        embedding_path: *const c_char,
        embedding_name: *const c_char,
        norm_path: *const c_char,
        norm_name: *const c_char,
        projection_path: *const c_char,
        weight_name: *const c_char,
        scale_name: *const c_char,
        global_scale_name: *const c_char,
        token_id: c_int,
        input_width: usize,
        output_width: usize,
        epsilon: f32,
        shift_norm_weight: c_int,
        result: *mut Nvfp4ProjectionResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_layer0_projection_checkpoint(
        embedding_path: *const c_char,
        embedding_name: *const c_char,
        norm_path: *const c_char,
        norm_name: *const c_char,
        qkv_path: *const c_char,
        qkv_name: *const c_char,
        z_path: *const c_char,
        z_name: *const c_char,
        beta_path: *const c_char,
        beta_name: *const c_char,
        alpha_path: *const c_char,
        alpha_name: *const c_char,
        conv_path: *const c_char,
        conv_name: *const c_char,
        dt_bias_path: *const c_char,
        dt_bias_name: *const c_char,
        a_log_path: *const c_char,
        a_log_name: *const c_char,
        gdn_norm_path: *const c_char,
        gdn_norm_name: *const c_char,
        out_proj_path: *const c_char,
        out_proj_name: *const c_char,
        post_norm_path: *const c_char,
        post_norm_name: *const c_char,
        mlp_gate_path: *const c_char,
        mlp_gate_name: *const c_char,
        mlp_up_path: *const c_char,
        mlp_up_name: *const c_char,
        mlp_down_path: *const c_char,
        mlp_down_name: *const c_char,
        token_id: c_int,
        input_width: usize,
        qkv_width: usize,
        z_width: usize,
        ba_width: usize,
        mlp_width: usize,
        epsilon: f32,
        shift_norm_weight: c_int,
        result: *mut Layer0ProjectionResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_model_new(
        sources: *const ModelBlobSource,
        source_count: usize,
        model: *mut *mut NativeQwenModel,
        result: *mut ModelLoadResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_model_free(model: *mut NativeQwenModel);
    fn generalist_mlx_qwen_cache_new(
        cache: *mut *mut NativeQwenCache,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cache_free(cache: *mut NativeQwenCache);
    fn generalist_mlx_qwen_cache_reset(
        cache: *mut NativeQwenCache,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_new(
        cache: *mut *mut NativeQwenSpeculativeCache,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_clone(
        source: *const NativeQwenSpeculativeCache,
        cache: *mut *mut NativeQwenSpeculativeCache,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_free(cache: *mut NativeQwenSpeculativeCache);
    fn generalist_mlx_qwen_speculative_cache_reset(
        cache: *mut NativeQwenSpeculativeCache,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_set_verification_geometry(
        cache: *mut NativeQwenSpeculativeCache,
        results_per_simdgroup: c_int,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_set_verification_shape_override(
        cache: *mut NativeQwenSpeculativeCache,
        default_results_per_simdgroup: c_int,
        output_width: c_int,
        input_width: c_int,
        override_results_per_simdgroup: c_int,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_set_verification_qmv_path(
        cache: *mut NativeQwenSpeculativeCache,
        path: c_int,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_set_mtp_draft_vocab_limit(
        cache: *mut NativeQwenSpeculativeCache,
        vocab_limit: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_set_mtp_draft_extra_tokens(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenSpeculativeCache,
        token_ids: *const c_int,
        token_count: usize,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_cache_set_direct_cache_rollback(
        cache: *mut NativeQwenSpeculativeCache,
        enabled: c_int,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_prefill(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_ids: *const c_int,
        token_count: usize,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_prefill(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenSpeculativeCache,
        token_ids: *const c_int,
        token_count: usize,
        result: *mut SpeculativeStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_prefill_chunk(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenSpeculativeCache,
        token_ids: *const c_int,
        token_count: usize,
        next_token_id: c_int,
        result: *mut SpeculativeStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_seed(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenSpeculativeCache,
        token_id: c_int,
        result: *mut SpeculativeStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_speculative_step(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenSpeculativeCache,
        draft_depth: usize,
        max_output_tokens: usize,
        result: *mut SpeculativeStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_checkpoint(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut CachedDecodeResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_unfused_scale_checkpoint(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut CachedDecodeResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_wide_qmv_checkpoint(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut CachedDecodeResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_argmax(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_argmax_unfused_scale(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_argmax_wide_qmv(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_argmax_dynamic_norm(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_argmax_split_mlp(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_argmax_deferred_state(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_cached_decode_argmax_async_state(
        model: *mut NativeQwenModel,
        cache: *mut NativeQwenCache,
        token_id: c_int,
        result: *mut TokenStepResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_mtp_checkpoint(
        model: *mut NativeQwenModel,
        token_id: c_int,
        result: *mut MtpCheckpointResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_batch_decode_checkpoint(
        model: *mut NativeQwenModel,
        token_ids: *const c_int,
        token_count: usize,
        result: *mut BatchDecodeCheckpointResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_zero_state_decode_checkpoint(
        model: *mut NativeQwenModel,
        token_id: c_int,
        result: *mut DecodeResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_zero_state_decode_experimental_fused_checkpoint(
        model: *mut NativeQwenModel,
        token_id: c_int,
        result: *mut DecodeResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_zero_state_decode_profile(
        model: *mut NativeQwenModel,
        token_id: c_int,
        result: *mut DecodeProfileResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
    fn generalist_mlx_qwen_compare_zero_state_fused(
        model: *mut NativeQwenModel,
        token_id: c_int,
        result: *mut DecodeComparisonResult,
        error: *mut c_char,
        error_capacity: usize,
    ) -> c_int;
}

/// Process-global MLX runtime initialized with one immutable Metal library.
/// Construct it on Generalist's dedicated inference thread before any MLX op.
#[derive(Debug)]
pub struct MlxRuntime {
    metallib_path: &'static Path,
}

/// Persistent owner for the exact Qwen3.8 27B reference graph and its
/// prepacked linear-attention tensors. It is movable onto Generalist's
/// dedicated inference thread, but intentionally not shareable between
/// threads; decode operations require exclusive access.
pub struct MlxQwenModel {
    #[cfg(generalist_mlx_native_linked)]
    raw: NonNull<NativeQwenModel>,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: ownership may move to the permanent inference thread before use.
// The lack of Sync plus `&mut self` decode methods prevents concurrent access.
unsafe impl Send for MlxQwenModel {}

/// Persistent, sequence-local recurrent and KV state for Qwen3.8. A session
/// may move onto the dedicated inference thread but cannot be shared or used
/// concurrently. Its state advances only through a successful cached decode.
pub struct MlxQwenSession {
    #[cfg(generalist_mlx_native_linked)]
    raw: NonNull<NativeQwenCache>,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: the opaque native cache has unique ownership and all mutation
// requires `&mut MlxQwenSession` on the destination inference thread.
unsafe impl Send for MlxQwenSession {}

/// Sequence-local target plus MTP state for greedy speculative decoding.
/// The cache owns both histories so accepted drafts and target corrections are
/// committed as one native state transition.
pub struct MlxQwenSpeculativeSession {
    #[cfg(generalist_mlx_native_linked)]
    raw: NonNull<NativeQwenSpeculativeCache>,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: the opaque native cache has unique ownership and all mutation
// requires `&mut MlxQwenSpeculativeSession` on the inference thread.
unsafe impl Send for MlxQwenSpeculativeSession {}

impl fmt::Debug for MlxQwenSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MlxQwenSession")
            .field("thread_exclusive", &true)
            .finish_non_exhaustive()
    }
}

impl Drop for MlxQwenSession {
    fn drop(&mut self) {
        #[cfg(generalist_mlx_native_linked)]
        // SAFETY: `raw` was returned by the matching constructor and is owned
        // by this session until Drop, which runs exactly once.
        unsafe {
            generalist_mlx_qwen_cache_free(self.raw.as_ptr());
        }
    }
}

impl fmt::Debug for MlxQwenSpeculativeSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MlxQwenSpeculativeSession")
            .field("thread_exclusive", &true)
            .finish_non_exhaustive()
    }
}

impl Drop for MlxQwenSpeculativeSession {
    fn drop(&mut self) {
        #[cfg(generalist_mlx_native_linked)]
        // SAFETY: `raw` was returned by the matching constructor and is owned
        // by this session until Drop, which runs exactly once.
        unsafe {
            generalist_mlx_qwen_speculative_cache_free(self.raw.as_ptr());
        }
    }
}

impl fmt::Debug for MlxQwenModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MlxQwenModel")
            .field("thread_exclusive", &true)
            .finish_non_exhaustive()
    }
}

impl Drop for MlxQwenModel {
    fn drop(&mut self) {
        #[cfg(generalist_mlx_native_linked)]
        // SAFETY: `raw` is uniquely owned and was returned by the matching
        // native constructor. Drop runs exactly once.
        unsafe {
            generalist_mlx_qwen_model_free(self.raw.as_ptr());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxModelLoadCheckpoint {
    pub blob_count: usize,
    pub array_count: usize,
    pub layer_count: usize,
    pub linear_layer_count: usize,
    pub full_attention_layer_count: usize,
    pub mtp_layer_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlxCachedDecodeCheckpoint {
    pub position: usize,
    pub decode: MlxDecodeCheckpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxTokenStep {
    pub position: usize,
    pub input_token_id: i32,
    pub output_token_id: i32,
}

/// One speculative target-model commit. `output_token_ids` contains zero or
/// more accepted MTP drafts followed by the target model's correction or bonus
/// token. Optimized verification may use a mathematically equivalent parallel
/// reduction tree rather than bit-identical token-at-a-time arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct MlxSpeculativeStep {
    pub position: usize,
    pub input_token_id: i32,
    pub output_token_ids: Vec<i32>,
    pub drafted: usize,
    pub accepted: usize,
    pub proposal_milliseconds: f64,
    pub verification_milliseconds: f64,
    pub replay_milliseconds: f64,
    pub correction_milliseconds: f64,
}

/// Exact one-pair MTP checkpoint. The target first predicts `lookahead_token_id`;
/// the MTP head consumes that token with the target hidden and predicts
/// `draft_token_id`; an ordinary second target step supplies the acceptance
/// oracle in `target_token_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct MlxMtpCheckpoint {
    pub input_token_id: i32,
    pub lookahead_token_id: i32,
    pub draft_token_id: i32,
    pub target_token_id: i32,
    pub target_hidden: ArrayFingerprint,
    pub draft_hidden: ArrayFingerprint,
    pub draft_logits: ArrayFingerprint,
    pub next_target_hidden: ArrayFingerprint,
    pub next_target_logits: ArrayFingerprint,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlxBatchDecodeRow {
    pub input_token_id: i32,
    pub batch_output_token_id: i32,
    pub sequential_output_token_id: i32,
    pub batch_hidden: ArrayFingerprint,
    pub batch_logits: ArrayFingerprint,
    pub sequential_hidden: ArrayFingerprint,
    pub sequential_logits: ArrayFingerprint,
}

/// Exact comparison of one fused multi-token target forward with ordinary
/// one-token cache advancement over the same input sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct MlxBatchDecodeCheckpoint {
    pub rows: Vec<MlxBatchDecodeRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeTensorProbe {
    pub dtype: i32,
    pub shape: Vec<i32>,
    pub elements: usize,
    pub bytes: usize,
    pub gpu_sum: f32,
}

/// Numerical fingerprint of Qwen's first real model operation. Dtypes are MLX
/// C enum values; for the current checkpoint `12` denotes BF16.
#[derive(Debug, Clone, PartialEq)]
pub struct RmsNormCheckpoint {
    pub input_dtype: i32,
    pub weight_dtype: i32,
    pub output_dtype: i32,
    pub width: usize,
    pub output_sum: f64,
    pub output_squared_sum: f64,
    pub output_first: f32,
    pub output_middle: f32,
    pub output_last: f32,
    pub max_abs_reference_error: f64,
    pub output_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayFingerprint {
    pub dtype: i32,
    pub elements: usize,
    pub sum: f64,
    pub squared_sum: f64,
    pub first: f32,
    pub middle: f32,
    pub last: f32,
    pub fingerprint: u64,
}

/// End-to-end exact reference checkpoint for one token. The zero-state API
/// uses empty recurrent/KV state; cached decode returns the same payload while
/// advancing a persistent [`MlxQwenSession`].
#[derive(Debug, Clone, PartialEq)]
pub struct MlxDecodeCheckpoint {
    pub input_token_id: i32,
    pub output_token_id: i32,
    pub hidden: ArrayFingerprint,
    pub logits: ArrayFingerprint,
}

/// Synchronized diagnostic profile of the exact first-token graph. Each layer
/// is explicitly evaluated before the next one, so these timings locate costs
/// but are not a steady-state throughput measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct MlxDecodeProfile {
    pub decode: MlxDecodeCheckpoint,
    pub layer_milliseconds: [f64; 64],
    pub final_norm_milliseconds: f64,
    pub lm_head_argmax_milliseconds: f64,
    pub graph_milliseconds: f64,
    pub checkpoint_milliseconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayDrift {
    pub max_abs: f64,
    pub mean_abs: f64,
    pub root_mean_square: f64,
    pub relative_l2: f64,
    pub cosine_similarity: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlxDecodeComparison {
    pub exact: MlxDecodeCheckpoint,
    pub experimental_fused: MlxDecodeCheckpoint,
    pub hidden_drift: ArrayDrift,
    pub logits_drift: ArrayDrift,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingCheckpoint {
    pub token_id: i32,
    pub width: usize,
    pub embedding: ArrayFingerprint,
    pub normalized: ArrayFingerprint,
}

/// Exact numerical boundary around Qwen's first compressed linear operation:
/// BF16 input, packed NVFP4 weights, E4M3 group scales, and the checkpoint's
/// separate F32 double scale. `raw_projection` is before that final scale;
/// `output` is the value consumed by the model.
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4ProjectionCheckpoint {
    pub token_id: i32,
    pub input_width: usize,
    pub output_width: usize,
    pub group_size: i32,
    pub bits: i32,
    pub weight_dtype: i32,
    pub scale_dtype: i32,
    pub global_scale_dtype: i32,
    pub global_scale: f32,
    pub encoded_global_scale: f32,
    pub normalized_input: ArrayFingerprint,
    pub raw_projection: ArrayFingerprint,
    pub output: ArrayFingerprint,
    pub fused_output: ArrayFingerprint,
}

/// Exact boundary around Qwen's packed layer-zero input projections. QKV and
/// gate use one concatenated NVFP4 matmul followed by the checkpoint's separate
/// global-scale operation; beta and alpha use one concatenated BF16 matmul.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0ProjectionCheckpoint {
    pub token_id: i32,
    pub input_width: usize,
    pub qkv_width: usize,
    pub z_width: usize,
    pub ba_width: usize,
    pub qkv_global_scale: f32,
    pub z_global_scale: f32,
    pub normalized_input: ArrayFingerprint,
    pub raw_qkvz: ArrayFingerprint,
    pub mixed_qkvz: ArrayFingerprint,
    pub mixed_ba: ArrayFingerprint,
}

/// Exact single-token, zero-state checkpoint for Qwen's layer-zero causal
/// depthwise convolution and gated-delta recurrence.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0GdnCheckpoint {
    pub projections: Layer0ProjectionCheckpoint,
    pub conv_kernel_width: usize,
    pub key_heads: usize,
    pub key_dim: usize,
    pub value_heads: usize,
    pub value_dim: usize,
    pub conv_out: ArrayFingerprint,
    pub conv_state: ArrayFingerprint,
    pub a_exp: ArrayFingerprint,
    pub gdn_out: ArrayFingerprint,
    pub delta_state: ArrayFingerprint,
}

/// Complete single-token layer-zero checkpoint, including the recurrent
/// branch, gated output projection, both residuals, and dense SwiGLU MLP.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0Checkpoint {
    pub gdn: Layer0GdnCheckpoint,
    pub mlp_width: usize,
    pub gdn_norm: ArrayFingerprint,
    pub gated: ArrayFingerprint,
    pub attention_out: ArrayFingerprint,
    pub after_attention: ArrayFingerprint,
    pub post_norm: ArrayFingerprint,
    pub mlp_gate: ArrayFingerprint,
    pub mlp_up: ArrayFingerprint,
    pub swiglu: ArrayFingerprint,
    pub mlp_out: ArrayFingerprint,
    pub layer_out: ArrayFingerprint,
}

#[derive(Debug, Clone, Copy)]
pub struct NativeTensorSource<'a> {
    pub path: &'a Path,
    pub name: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct Layer0CheckpointSpec<'a> {
    pub embedding: NativeTensorSource<'a>,
    pub input_norm: NativeTensorSource<'a>,
    pub qkv: NativeTensorSource<'a>,
    pub z: NativeTensorSource<'a>,
    pub beta: NativeTensorSource<'a>,
    pub alpha: NativeTensorSource<'a>,
    pub conv: NativeTensorSource<'a>,
    pub dt_bias: NativeTensorSource<'a>,
    pub a_log: NativeTensorSource<'a>,
    pub gdn_norm: NativeTensorSource<'a>,
    pub out_proj: NativeTensorSource<'a>,
    pub post_norm: NativeTensorSource<'a>,
    pub mlp_gate: NativeTensorSource<'a>,
    pub mlp_up: NativeTensorSource<'a>,
    pub mlp_down: NativeTensorSource<'a>,
    pub token_id: i32,
    pub input_width: usize,
    pub qkv_width: usize,
    pub z_width: usize,
    pub ba_width: usize,
    pub mlp_width: usize,
    pub epsilon: f32,
    pub shift_norm_weight: bool,
}

impl MlxRuntime {
    pub fn initialize(metallib_path: impl AsRef<Path>) -> Result<Self> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = metallib_path;
            return Err(native_error(
                "native MLX was not linked; run `make embedded` from the Generalist repository on Apple Silicon macOS",
            ));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let canonical = std::fs::canonicalize(metallib_path.as_ref()).map_err(|error| {
                native_error(format!(
                    "failed to resolve Metal library {}: {error}",
                    metallib_path.as_ref().display()
                ))
            })?;
            if !canonical.is_file() {
                return Err(native_error(format!(
                    "Metal library is not a regular file: {}",
                    canonical.display()
                )));
            }
            if let Some(existing) = METALLIB_PATH.get() {
                if existing != &canonical {
                    return Err(native_error(format!(
                        "MLX is already initialized with {}, cannot replace it with {}",
                        existing.display(),
                        canonical.display()
                    )));
                }
                return Ok(Self {
                    metallib_path: existing,
                });
            }

            let encoded = path_to_cstring(&canonical, "Metal library")?;
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: pointers remain valid for the duration of the call; the shim
            // writes at most ERROR_CAPACITY bytes and retains only a C++ copy of
            // the path string.
            let status = unsafe {
                generalist_mlx_initialize(encoded.as_ptr(), error.as_mut_ptr(), error.len())
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            let stored = METALLIB_PATH.get_or_init(|| canonical);
            Ok(Self {
                metallib_path: stored,
            })
        }
    }

    pub fn metallib_path(&self) -> &Path {
        self.metallib_path
    }

    pub fn probe_tensor(
        &self,
        safetensors_path: impl AsRef<Path>,
        tensor_name: &str,
    ) -> Result<NativeTensorProbe> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (safetensors_path, tensor_name);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let tensor_path = path_to_cstring(safetensors_path.as_ref(), "safetensors blob")?;
            let tensor_name = CString::new(tensor_name)
                .map_err(|_| native_error("tensor name contains a NUL byte"))?;
            let mut result = ProbeResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: all pointers refer to initialized, correctly sized values
            // for the synchronous duration of the C++ shim call.
            let status = unsafe {
                generalist_mlx_probe_tensor(
                    tensor_path.as_ptr(),
                    tensor_name.as_ptr(),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            if result.ndim > MAX_TENSOR_DIMS {
                return Err(native_error(
                    "native tensor rank exceeded the Rust ABI limit",
                ));
            }
            Ok(NativeTensorProbe {
                dtype: result.dtype,
                shape: result.shape[..result.ndim].to_vec(),
                elements: result.elements,
                bytes: result.bytes,
                gpu_sum: result.gpu_sum,
            })
        }
    }

    pub fn rms_norm_checkpoint(
        &self,
        safetensors_path: impl AsRef<Path>,
        weight_name: &str,
        width: usize,
        epsilon: f32,
        shift_weight: bool,
    ) -> Result<RmsNormCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (safetensors_path, weight_name, width, epsilon, shift_weight);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if width == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(native_error("invalid RMSNorm checkpoint dimensions"));
            }
            let tensor_path = path_to_cstring(safetensors_path.as_ref(), "safetensors blob")?;
            let weight_name = CString::new(weight_name)
                .map_err(|_| native_error("RMSNorm weight name contains a NUL byte"))?;
            let mut result = RmsNormResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: every pointer is valid for this synchronous call and result
            // has exactly the layout declared in native/mlx/shim.h.
            let status = unsafe {
                generalist_mlx_rms_norm_checkpoint(
                    tensor_path.as_ptr(),
                    weight_name.as_ptr(),
                    width,
                    epsilon,
                    i32::from(shift_weight),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(RmsNormCheckpoint {
                input_dtype: result.input_dtype,
                weight_dtype: result.weight_dtype,
                output_dtype: result.output_dtype,
                width: result.width,
                output_sum: result.output_sum,
                output_squared_sum: result.output_squared_sum,
                output_first: result.output_first,
                output_middle: result.output_middle,
                output_last: result.output_last,
                max_abs_reference_error: result.max_abs_reference_error,
                output_fingerprint: result.output_fingerprint,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn embedding_checkpoint(
        &self,
        embedding_path: impl AsRef<Path>,
        embedding_name: &str,
        norm_path: impl AsRef<Path>,
        norm_name: &str,
        token_id: i32,
        width: usize,
        epsilon: f32,
        shift_norm_weight: bool,
    ) -> Result<EmbeddingCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (
                embedding_path,
                embedding_name,
                norm_path,
                norm_name,
                token_id,
                width,
                epsilon,
                shift_norm_weight,
            );
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 || width == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(native_error("invalid embedding checkpoint arguments"));
            }
            let embedding_path =
                path_to_cstring(embedding_path.as_ref(), "embedding safetensors blob")?;
            let embedding_name = CString::new(embedding_name)
                .map_err(|_| native_error("embedding name contains a NUL byte"))?;
            let norm_path = path_to_cstring(norm_path.as_ref(), "RMSNorm safetensors blob")?;
            let norm_name = CString::new(norm_name)
                .map_err(|_| native_error("RMSNorm name contains a NUL byte"))?;
            let mut result = EmbeddingResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the C strings and ABI result remain valid for this
            // synchronous call and exactly mirror native/mlx/shim.h.
            let status = unsafe {
                generalist_mlx_embedding_checkpoint(
                    embedding_path.as_ptr(),
                    embedding_name.as_ptr(),
                    norm_path.as_ptr(),
                    norm_name.as_ptr(),
                    token_id,
                    width,
                    epsilon,
                    i32::from(shift_norm_weight),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(EmbeddingCheckpoint {
                token_id: result.token_id,
                width: result.width,
                embedding: array_fingerprint(result.embedding),
                normalized: array_fingerprint(result.normalized),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_projection_checkpoint(
        &self,
        embedding_path: impl AsRef<Path>,
        embedding_name: &str,
        norm_path: impl AsRef<Path>,
        norm_name: &str,
        projection_path: impl AsRef<Path>,
        weight_name: &str,
        scale_name: &str,
        global_scale_name: &str,
        token_id: i32,
        input_width: usize,
        output_width: usize,
        epsilon: f32,
        shift_norm_weight: bool,
    ) -> Result<Nvfp4ProjectionCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (
                embedding_path,
                embedding_name,
                norm_path,
                norm_name,
                projection_path,
                weight_name,
                scale_name,
                global_scale_name,
                token_id,
                input_width,
                output_width,
                epsilon,
                shift_norm_weight,
            );
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0
                || input_width == 0
                || output_width == 0
                || !epsilon.is_finite()
                || epsilon <= 0.0
            {
                return Err(native_error(
                    "invalid NVFP4 projection checkpoint arguments",
                ));
            }
            let embedding_path =
                path_to_cstring(embedding_path.as_ref(), "embedding safetensors blob")?;
            let embedding_name = CString::new(embedding_name)
                .map_err(|_| native_error("embedding name contains a NUL byte"))?;
            let norm_path = path_to_cstring(norm_path.as_ref(), "RMSNorm safetensors blob")?;
            let norm_name = CString::new(norm_name)
                .map_err(|_| native_error("RMSNorm name contains a NUL byte"))?;
            let projection_path =
                path_to_cstring(projection_path.as_ref(), "NVFP4 safetensors blob")?;
            let weight_name = CString::new(weight_name)
                .map_err(|_| native_error("NVFP4 weight name contains a NUL byte"))?;
            let scale_name = CString::new(scale_name)
                .map_err(|_| native_error("NVFP4 scale name contains a NUL byte"))?;
            let global_scale_name = CString::new(global_scale_name)
                .map_err(|_| native_error("NVFP4 global scale name contains a NUL byte"))?;
            let mut result = Nvfp4ProjectionResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: all paths and names remain alive for the synchronous call,
            // and the result layout exactly mirrors native/mlx/shim.h.
            let status = unsafe {
                generalist_mlx_nvfp4_projection_checkpoint(
                    embedding_path.as_ptr(),
                    embedding_name.as_ptr(),
                    norm_path.as_ptr(),
                    norm_name.as_ptr(),
                    projection_path.as_ptr(),
                    weight_name.as_ptr(),
                    scale_name.as_ptr(),
                    global_scale_name.as_ptr(),
                    token_id,
                    input_width,
                    output_width,
                    epsilon,
                    i32::from(shift_norm_weight),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(Nvfp4ProjectionCheckpoint {
                token_id: result.token_id,
                input_width: result.input_width,
                output_width: result.output_width,
                group_size: result.group_size,
                bits: result.bits,
                weight_dtype: result.weight_dtype,
                scale_dtype: result.scale_dtype,
                global_scale_dtype: result.global_scale_dtype,
                global_scale: result.global_scale,
                encoded_global_scale: result.encoded_global_scale,
                normalized_input: array_fingerprint(result.normalized_input),
                raw_projection: array_fingerprint(result.raw_projection),
                output: array_fingerprint(result.output),
                fused_output: array_fingerprint(result.fused_output),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer0_projection_checkpoint(
        &self,
        embedding_path: impl AsRef<Path>,
        embedding_name: &str,
        norm_path: impl AsRef<Path>,
        norm_name: &str,
        qkv_path: impl AsRef<Path>,
        qkv_name: &str,
        z_path: impl AsRef<Path>,
        z_name: &str,
        beta_path: impl AsRef<Path>,
        beta_name: &str,
        alpha_path: impl AsRef<Path>,
        alpha_name: &str,
        token_id: i32,
        input_width: usize,
        qkv_width: usize,
        z_width: usize,
        ba_width: usize,
        epsilon: f32,
        shift_norm_weight: bool,
    ) -> Result<Layer0ProjectionCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (
                embedding_path,
                embedding_name,
                norm_path,
                norm_name,
                qkv_path,
                qkv_name,
                z_path,
                z_name,
                beta_path,
                beta_name,
                alpha_path,
                alpha_name,
                token_id,
                input_width,
                qkv_width,
                z_width,
                ba_width,
                epsilon,
                shift_norm_weight,
            );
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0
                || input_width == 0
                || qkv_width == 0
                || z_width == 0
                || ba_width == 0
                || !ba_width.is_multiple_of(2)
                || !epsilon.is_finite()
                || epsilon <= 0.0
            {
                return Err(native_error(
                    "invalid layer-zero projection checkpoint arguments",
                ));
            }
            let embedding_path =
                path_to_cstring(embedding_path.as_ref(), "embedding safetensors blob")?;
            let embedding_name = CString::new(embedding_name)
                .map_err(|_| native_error("embedding name contains a NUL byte"))?;
            let norm_path = path_to_cstring(norm_path.as_ref(), "RMSNorm safetensors blob")?;
            let norm_name = CString::new(norm_name)
                .map_err(|_| native_error("RMSNorm name contains a NUL byte"))?;
            let qkv_path = path_to_cstring(qkv_path.as_ref(), "QKV safetensors blob")?;
            let qkv_name =
                CString::new(qkv_name).map_err(|_| native_error("QKV name contains a NUL byte"))?;
            let z_path = path_to_cstring(z_path.as_ref(), "Z safetensors blob")?;
            let z_name =
                CString::new(z_name).map_err(|_| native_error("Z name contains a NUL byte"))?;
            let beta_path = path_to_cstring(beta_path.as_ref(), "beta safetensors blob")?;
            let beta_name = CString::new(beta_name)
                .map_err(|_| native_error("beta name contains a NUL byte"))?;
            let alpha_path = path_to_cstring(alpha_path.as_ref(), "alpha safetensors blob")?;
            let alpha_name = CString::new(alpha_name)
                .map_err(|_| native_error("alpha name contains a NUL byte"))?;
            let mut result = Layer0ProjectionResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: every path and tensor name remains alive for this
            // synchronous call, and the ABI result mirrors native/mlx/shim.h.
            let status = unsafe {
                generalist_mlx_layer0_projection_checkpoint(
                    embedding_path.as_ptr(),
                    embedding_name.as_ptr(),
                    norm_path.as_ptr(),
                    norm_name.as_ptr(),
                    qkv_path.as_ptr(),
                    qkv_name.as_ptr(),
                    z_path.as_ptr(),
                    z_name.as_ptr(),
                    beta_path.as_ptr(),
                    beta_name.as_ptr(),
                    alpha_path.as_ptr(),
                    alpha_name.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    token_id,
                    input_width,
                    qkv_width,
                    z_width,
                    ba_width,
                    0,
                    epsilon,
                    i32::from(shift_norm_weight),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(layer0_projection_fingerprint(result))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer0_gdn_checkpoint(
        &self,
        embedding_path: impl AsRef<Path>,
        embedding_name: &str,
        norm_path: impl AsRef<Path>,
        norm_name: &str,
        qkv_path: impl AsRef<Path>,
        qkv_name: &str,
        z_path: impl AsRef<Path>,
        z_name: &str,
        beta_path: impl AsRef<Path>,
        beta_name: &str,
        alpha_path: impl AsRef<Path>,
        alpha_name: &str,
        conv_path: impl AsRef<Path>,
        conv_name: &str,
        dt_bias_path: impl AsRef<Path>,
        dt_bias_name: &str,
        a_log_path: impl AsRef<Path>,
        a_log_name: &str,
        token_id: i32,
        input_width: usize,
        qkv_width: usize,
        z_width: usize,
        ba_width: usize,
        epsilon: f32,
        shift_norm_weight: bool,
    ) -> Result<Layer0GdnCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (
                embedding_path,
                embedding_name,
                norm_path,
                norm_name,
                qkv_path,
                qkv_name,
                z_path,
                z_name,
                beta_path,
                beta_name,
                alpha_path,
                alpha_name,
                conv_path,
                conv_name,
                dt_bias_path,
                dt_bias_name,
                a_log_path,
                a_log_name,
                token_id,
                input_width,
                qkv_width,
                z_width,
                ba_width,
                epsilon,
                shift_norm_weight,
            );
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0
                || input_width == 0
                || qkv_width == 0
                || z_width == 0
                || ba_width == 0
                || !ba_width.is_multiple_of(2)
                || !epsilon.is_finite()
                || epsilon <= 0.0
            {
                return Err(native_error("invalid layer-zero GDN checkpoint arguments"));
            }
            let embedding_path =
                path_to_cstring(embedding_path.as_ref(), "embedding safetensors blob")?;
            let embedding_name = CString::new(embedding_name)
                .map_err(|_| native_error("embedding name contains a NUL byte"))?;
            let norm_path = path_to_cstring(norm_path.as_ref(), "RMSNorm safetensors blob")?;
            let norm_name = CString::new(norm_name)
                .map_err(|_| native_error("RMSNorm name contains a NUL byte"))?;
            let qkv_path = path_to_cstring(qkv_path.as_ref(), "QKV safetensors blob")?;
            let qkv_name =
                CString::new(qkv_name).map_err(|_| native_error("QKV name contains a NUL byte"))?;
            let z_path = path_to_cstring(z_path.as_ref(), "Z safetensors blob")?;
            let z_name =
                CString::new(z_name).map_err(|_| native_error("Z name contains a NUL byte"))?;
            let beta_path = path_to_cstring(beta_path.as_ref(), "beta safetensors blob")?;
            let beta_name = CString::new(beta_name)
                .map_err(|_| native_error("beta name contains a NUL byte"))?;
            let alpha_path = path_to_cstring(alpha_path.as_ref(), "alpha safetensors blob")?;
            let alpha_name = CString::new(alpha_name)
                .map_err(|_| native_error("alpha name contains a NUL byte"))?;
            let conv_path = path_to_cstring(conv_path.as_ref(), "conv safetensors blob")?;
            let conv_name = CString::new(conv_name)
                .map_err(|_| native_error("conv name contains a NUL byte"))?;
            let dt_bias_path = path_to_cstring(dt_bias_path.as_ref(), "dt-bias safetensors blob")?;
            let dt_bias_name = CString::new(dt_bias_name)
                .map_err(|_| native_error("dt-bias name contains a NUL byte"))?;
            let a_log_path = path_to_cstring(a_log_path.as_ref(), "A-log safetensors blob")?;
            let a_log_name = CString::new(a_log_name)
                .map_err(|_| native_error("A-log name contains a NUL byte"))?;
            let mut result = Layer0ProjectionResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: all paths, names, and ABI outputs remain valid for the
            // complete synchronous checkpoint call.
            let status = unsafe {
                generalist_mlx_layer0_projection_checkpoint(
                    embedding_path.as_ptr(),
                    embedding_name.as_ptr(),
                    norm_path.as_ptr(),
                    norm_name.as_ptr(),
                    qkv_path.as_ptr(),
                    qkv_name.as_ptr(),
                    z_path.as_ptr(),
                    z_name.as_ptr(),
                    beta_path.as_ptr(),
                    beta_name.as_ptr(),
                    alpha_path.as_ptr(),
                    alpha_name.as_ptr(),
                    conv_path.as_ptr(),
                    conv_name.as_ptr(),
                    dt_bias_path.as_ptr(),
                    dt_bias_name.as_ptr(),
                    a_log_path.as_ptr(),
                    a_log_name.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    token_id,
                    input_width,
                    qkv_width,
                    z_width,
                    ba_width,
                    0,
                    epsilon,
                    i32::from(shift_norm_weight),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            if result.gdn_executed != 1 {
                return Err(native_error(
                    "native layer-zero checkpoint omitted the requested GDN path",
                ));
            }
            Ok(layer0_gdn_fingerprint(result))
        }
    }

    pub fn layer0_checkpoint(&self, spec: &Layer0CheckpointSpec<'_>) -> Result<Layer0Checkpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = spec;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if spec.token_id < 0
                || spec.input_width == 0
                || spec.qkv_width == 0
                || spec.z_width == 0
                || spec.ba_width == 0
                || !spec.ba_width.is_multiple_of(2)
                || spec.mlp_width == 0
                || !spec.epsilon.is_finite()
                || spec.epsilon <= 0.0
            {
                return Err(native_error("invalid complete layer-zero checkpoint spec"));
            }
            let (embedding_path, embedding_name) =
                tensor_source_to_cstrings(spec.embedding, "embedding")?;
            let (norm_path, norm_name) = tensor_source_to_cstrings(spec.input_norm, "input norm")?;
            let (qkv_path, qkv_name) = tensor_source_to_cstrings(spec.qkv, "QKV")?;
            let (z_path, z_name) = tensor_source_to_cstrings(spec.z, "Z")?;
            let (beta_path, beta_name) = tensor_source_to_cstrings(spec.beta, "beta")?;
            let (alpha_path, alpha_name) = tensor_source_to_cstrings(spec.alpha, "alpha")?;
            let (conv_path, conv_name) = tensor_source_to_cstrings(spec.conv, "conv")?;
            let (dt_bias_path, dt_bias_name) = tensor_source_to_cstrings(spec.dt_bias, "dt bias")?;
            let (a_log_path, a_log_name) = tensor_source_to_cstrings(spec.a_log, "A log")?;
            let (gdn_norm_path, gdn_norm_name) =
                tensor_source_to_cstrings(spec.gdn_norm, "GDN norm")?;
            let (out_proj_path, out_proj_name) =
                tensor_source_to_cstrings(spec.out_proj, "attention output projection")?;
            let (post_norm_path, post_norm_name) =
                tensor_source_to_cstrings(spec.post_norm, "post-attention norm")?;
            let (mlp_gate_path, mlp_gate_name) =
                tensor_source_to_cstrings(spec.mlp_gate, "MLP gate projection")?;
            let (mlp_up_path, mlp_up_name) =
                tensor_source_to_cstrings(spec.mlp_up, "MLP up projection")?;
            let (mlp_down_path, mlp_down_name) =
                tensor_source_to_cstrings(spec.mlp_down, "MLP down projection")?;
            let mut result = Layer0ProjectionResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the complete tensor source set and result ABI remain
            // alive for this synchronous native checkpoint.
            let status = unsafe {
                generalist_mlx_layer0_projection_checkpoint(
                    embedding_path.as_ptr(),
                    embedding_name.as_ptr(),
                    norm_path.as_ptr(),
                    norm_name.as_ptr(),
                    qkv_path.as_ptr(),
                    qkv_name.as_ptr(),
                    z_path.as_ptr(),
                    z_name.as_ptr(),
                    beta_path.as_ptr(),
                    beta_name.as_ptr(),
                    alpha_path.as_ptr(),
                    alpha_name.as_ptr(),
                    conv_path.as_ptr(),
                    conv_name.as_ptr(),
                    dt_bias_path.as_ptr(),
                    dt_bias_name.as_ptr(),
                    a_log_path.as_ptr(),
                    a_log_name.as_ptr(),
                    gdn_norm_path.as_ptr(),
                    gdn_norm_name.as_ptr(),
                    out_proj_path.as_ptr(),
                    out_proj_name.as_ptr(),
                    post_norm_path.as_ptr(),
                    post_norm_name.as_ptr(),
                    mlp_gate_path.as_ptr(),
                    mlp_gate_name.as_ptr(),
                    mlp_up_path.as_ptr(),
                    mlp_up_name.as_ptr(),
                    mlp_down_path.as_ptr(),
                    mlp_down_name.as_ptr(),
                    spec.token_id,
                    spec.input_width,
                    spec.qkv_width,
                    spec.z_width,
                    spec.ba_width,
                    spec.mlp_width,
                    spec.epsilon,
                    i32::from(spec.shift_norm_weight),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            if result.gdn_executed != 1 || result.full_layer_executed != 1 {
                return Err(native_error(
                    "native checkpoint omitted part of the requested complete layer",
                ));
            }
            Ok(layer0_checkpoint_fingerprint(result))
        }
    }
}

impl MlxQwenSession {
    /// Discard every recurrent and KV tensor and return this session to
    /// position zero without reloading the resident model weights.
    pub fn reset(&mut self) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: this method holds the unique session borrow throughout
            // the synchronous native reset.
            let status = unsafe {
                generalist_mlx_qwen_cache_reset(self.raw.as_ptr(), error.as_mut_ptr(), error.len())
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }
}

impl MlxQwenSpeculativeSession {
    /// Create a cheap exact checkpoint of both target and MTP state. Native
    /// MLX arrays are immutable reference-counted values, so subsequent
    /// forwards replace cache roots without mutating the checkpoint.
    pub fn try_clone(&self) -> Result<Self> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let mut raw = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: `self` stays alive for the synchronous clone and native
            // code returns one independently owned cache object.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_clone(
                    self.raw.as_ptr(),
                    &mut raw,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            let raw = NonNull::new(raw)
                .ok_or_else(|| native_error("native Qwen speculative cache clone returned null"))?;
            Ok(Self {
                raw,
                _not_sync: PhantomData,
            })
        }
    }

    /// Discard both target and MTP histories and return this session to its
    /// unseeded state without reloading resident weights.
    pub fn reset(&mut self) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: this method holds the unique speculative-session borrow
            // throughout the synchronous native reset.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_reset(
                    self.raw.as_ptr(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }

    /// Select the number of output results assigned to each SIMD-group for
    /// speculative target verification. Values 1, 2, 4, and 8 expose an exact
    /// Metal tiling sweep; production multi-row verification defaults to four.
    pub fn set_verification_qmv_results_per_simdgroup(
        &mut self,
        results_per_simdgroup: i32,
    ) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = results_per_simdgroup;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if !matches!(results_per_simdgroup, 1 | 2 | 4 | 8) {
                return Err(native_error(
                    "verification QMV results per SIMD-group must be 1, 2, 4, or 8",
                ));
            }
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: this method holds the unique speculative-session borrow
            // throughout the synchronous native configuration transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_set_verification_geometry(
                    self.raw.as_ptr(),
                    results_per_simdgroup,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }

    /// Use one exact QMV geometry for verification generally and a second one
    /// for a single `(output_width, input_width)` projection shape. This is a
    /// benchmark control for model-specific Metal tuning; it does not alter
    /// weights, arithmetic, proposal, correction, or ordinary target decode.
    pub fn set_verification_qmv_shape_override(
        &mut self,
        default_results_per_simdgroup: i32,
        output_width: i32,
        input_width: i32,
        override_results_per_simdgroup: i32,
    ) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (
                default_results_per_simdgroup,
                output_width,
                input_width,
                override_results_per_simdgroup,
            );
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if !matches!(default_results_per_simdgroup, 1 | 2 | 4 | 8)
                || output_width <= 0
                || input_width <= 0
                || !matches!(override_results_per_simdgroup, 1 | 2 | 4 | 8)
            {
                return Err(native_error(
                    "verification QMV shape override requires positive widths and 1, 2, 4, or 8 results per SIMD-group",
                ));
            }
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: this method holds the unique speculative-session borrow
            // throughout the synchronous native configuration transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_set_verification_shape_override(
                    self.raw.as_ptr(),
                    default_results_per_simdgroup,
                    output_width,
                    input_width,
                    override_results_per_simdgroup,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }

    /// Use the production exact-scale small-batch NVFP4 path that reuses weight
    /// groups across verification rows. It preserves the reference
    /// BF16/global-scale/BF16 rounding boundaries, but its parallel reduction
    /// tree is not bit-identical to independent one-row QMV calls.
    pub fn set_verification_wide_qmv(&mut self, enabled: bool) -> Result<()> {
        self.set_verification_qmv_path(if enabled { 2 } else { 0 })
    }

    /// Benchmark the stock MLX wide-QMV reduction followed by separate
    /// global-scale and BF16-conversion dispatches. This isolates wide weight
    /// reuse from Generalist's fused exact-scale kernel.
    pub fn set_verification_unfused_wide_qmv(&mut self, enabled: bool) -> Result<()> {
        self.set_verification_qmv_path(if enabled { 1 } else { 0 })
    }

    /// Restrict only the MTP proposal head to a low-ID vocabulary prefix plus
    /// Qwen control tokens. Target verification still uses all 248,320 rows,
    /// so this can change acceptance and speed but not which target token a
    /// rejected proposal commits. Zero restores the full draft head.
    pub fn set_mtp_draft_vocab_limit(&mut self, vocab_limit: usize) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = vocab_limit;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if !matches!(vocab_limit, 0 | 65_536) {
                return Err(native_error(
                    "MTP draft vocabulary limit must be zero or 65536",
                ));
            }
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: this method holds the unique speculative-session borrow
            // throughout the synchronous native configuration transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_set_mtp_draft_vocab_limit(
                    self.raw.as_ptr(),
                    vocab_limit,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }

    /// Commit a rejected speculative prefix directly from the verification
    /// pass's retained recurrent states. Disabling this restores the
    /// weight-rereading replay path as a differential and performance oracle.
    pub fn set_direct_cache_rollback(&mut self, enabled: bool) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = enabled;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: this method holds the unique speculative-session borrow
            // throughout the synchronous native configuration transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_set_direct_cache_rollback(
                    self.raw.as_ptr(),
                    i32::from(enabled),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }

    fn set_verification_qmv_path(&mut self, path: i32) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = path;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: this method holds the unique speculative-session borrow
            // throughout the synchronous native configuration transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_set_verification_qmv_path(
                    self.raw.as_ptr(),
                    path,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }
}

impl MlxQwenModel {
    /// Load every unique language-model safetensors blob and retain the arrays
    /// plus model-wide prepacked tensors in one native owner. ModelPack has
    /// already validated each content-addressed file's type and declared size;
    /// callers can use `ModelPack::verify_blob` separately when a full 18 GB
    /// startup hash is desired.
    pub fn load(runtime: &MlxRuntime, pack: &ModelPack) -> Result<(Self, MlxModelLoadCheckpoint)> {
        Self::load_components(runtime, pack, false)
    }

    /// Load the target stack together with the checkpoint's inline MTP head.
    /// This is the resident model used by speculative decoding; [`Self::load`]
    /// remains target-only so base-decode differential benchmarks do not pay
    /// for or accidentally exercise the draft component.
    pub fn load_with_mtp(
        runtime: &MlxRuntime,
        pack: &ModelPack,
    ) -> Result<(Self, MlxModelLoadCheckpoint)> {
        Self::load_components(runtime, pack, true)
    }

    fn load_components(
        runtime: &MlxRuntime,
        pack: &ModelPack,
        include_mtp: bool,
    ) -> Result<(Self, MlxModelLoadCheckpoint)> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (runtime, pack, include_mtp);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let _ = runtime.metallib_path();
            if !pack
                .tensors()
                .any(|tensor| tensor.component == ModelComponent::Language)
            {
                return Err(native_error("model pack contains no language-model blobs"));
            }
            if include_mtp
                && !pack
                    .tensors()
                    .any(|tensor| tensor.component == ModelComponent::Mtp)
            {
                return Err(native_error("model pack contains no MTP blobs"));
            }
            let mut seen = HashSet::new();
            let paths: Vec<PathBuf> = pack
                .tensors()
                .filter(|tensor| {
                    tensor.component == ModelComponent::Language
                        || (include_mtp && tensor.component == ModelComponent::Mtp)
                })
                .filter(|tensor| seen.insert(tensor.blob.digest))
                .map(|tensor| pack.blob_path(&tensor.blob))
                .collect();
            let encoded_paths: Vec<CString> = paths
                .iter()
                .map(|path| path_to_cstring(path, "Qwen safetensors blob"))
                .collect::<Result<_>>()?;
            let sources: Vec<ModelBlobSource> = encoded_paths
                .iter()
                .map(|path| ModelBlobSource {
                    path: path.as_ptr(),
                })
                .collect();
            let mut raw = std::ptr::null_mut();
            let mut result = ModelLoadResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: source path storage outlives this synchronous call. On
            // success the native side returns one uniquely owned opaque model;
            // it retains MLX arrays rather than any Rust path pointers.
            let status = unsafe {
                generalist_mlx_qwen_model_new(
                    sources.as_ptr(),
                    sources.len(),
                    &mut raw,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            let raw = NonNull::new(raw)
                .ok_or_else(|| native_error("native Qwen loader returned a null model"))?;
            Ok((
                Self {
                    raw,
                    _not_sync: PhantomData,
                },
                MlxModelLoadCheckpoint {
                    blob_count: result.blob_count,
                    array_count: result.array_count,
                    layer_count: result.layer_count,
                    linear_layer_count: result.linear_layer_count,
                    full_attention_layer_count: result.full_attention_layer_count,
                    mtp_layer_count: result.mtp_layer_count,
                },
            ))
        }
    }

    /// Allocate empty sequence-local state. The 48 recurrent and 16 KV caches
    /// are populated lazily by the first decode, avoiding any Metal work at
    /// session construction time.
    pub fn start_session(&mut self) -> Result<MlxQwenSession> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let mut raw = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: native code writes at most one uniquely owned opaque
            // cache pointer and retains no Rust references.
            let status =
                unsafe { generalist_mlx_qwen_cache_new(&mut raw, error.as_mut_ptr(), error.len()) };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            let raw = NonNull::new(raw)
                .ok_or_else(|| native_error("native Qwen cache constructor returned null"))?;
            Ok(MlxQwenSession {
                raw,
                _not_sync: PhantomData,
            })
        }
    }

    /// Allocate sequence-local target and MTP state. The loaded model must
    /// include its MTP component before the session can be seeded.
    pub fn start_speculative_session(&mut self) -> Result<MlxQwenSpeculativeSession> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            let mut raw = std::ptr::null_mut();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: native code writes at most one uniquely owned opaque
            // speculative cache pointer and retains no Rust references.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_new(&mut raw, error.as_mut_ptr(), error.len())
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            let raw = NonNull::new(raw).ok_or_else(|| {
                native_error("native Qwen speculative cache constructor returned null")
            })?;
            Ok(MlxQwenSpeculativeSession {
                raw,
                _not_sync: PhantomData,
            })
        }
    }

    /// Add a small sorted set of prompt-local rows outside the active MTP
    /// proposal core. The target head remains complete; these rows only let
    /// the cheap draft reuse uncommon terms already present in the prompt.
    pub fn set_mtp_draft_extra_tokens(
        &mut self,
        session: &mut MlxQwenSpeculativeSession,
        token_ids: &[i32],
    ) -> Result<()> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_ids);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_ids.len() > QWEN_MAX_MTP_DRAFT_EXTRA_TOKENS
                || token_ids
                    .iter()
                    .any(|&token| !(65_536..248_000).contains(&token))
                || token_ids.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(native_error(
                    "MTP draft extra tokens must contain at most 256 sorted unique IDs from 65536 through 247999",
                ));
            }
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: both unique owners remain alive for the synchronous
            // gather, and native code copies the IDs before returning.
            let status = unsafe {
                generalist_mlx_qwen_speculative_cache_set_mtp_draft_extra_tokens(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_ids.as_ptr(),
                    token_ids.len(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(())
        }
    }

    /// Append a prompt to an ordinary target session in exact batches and
    /// return the greedy token predicted after its final input token.
    pub fn prefill(
        &mut self,
        session: &mut MlxQwenSession,
        token_ids: &[i32],
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_ids);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_ids.is_empty() || token_ids.iter().any(|token_id| *token_id < 0) {
                return Err(native_error(
                    "prefill requires at least one non-negative token id",
                ));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the token slice remains valid and model/session remain
            // exclusively borrowed throughout the synchronous native prefill.
            let status = unsafe {
                generalist_mlx_qwen_prefill(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_ids.as_ptr(),
                    token_ids.len(),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Prime target and MTP caches from a complete prompt and return the first
    /// exact greedy completion token. The speculative session must be reset.
    pub fn speculative_prefill(
        &mut self,
        session: &mut MlxQwenSpeculativeSession,
        token_ids: &[i32],
    ) -> Result<MlxSpeculativeStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_ids);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_ids.is_empty() || token_ids.iter().any(|token_id| *token_id < 0) {
                return Err(native_error(
                    "speculative prefill requires at least one non-negative token id",
                ));
            }
            let mut result = SpeculativeStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the token slice remains valid and model/session remain
            // exclusively borrowed throughout the synchronous native prefill.
            let status = unsafe {
                generalist_mlx_qwen_speculative_prefill(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_ids.as_ptr(),
                    token_ids.len(),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            speculative_step(result)
        }
    }

    /// Append one bounded prompt chunk to target and MTP state. When
    /// `known_next_token_id` is present, it is the exact next prompt token and
    /// becomes the held input for the following chunk. `None` closes prefill
    /// and returns the first greedy completion token. A seeded session accepts
    /// an append only when its held token equals the chunk's first token.
    pub fn speculative_prefill_chunk(
        &mut self,
        session: &mut MlxQwenSpeculativeSession,
        token_ids: &[i32],
        known_next_token_id: Option<i32>,
    ) -> Result<MlxSpeculativeStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_ids, known_next_token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_ids.is_empty()
                || token_ids.len() > QWEN_MAX_PREFILL_CHUNK_TOKENS
                || token_ids.iter().any(|token_id| *token_id < 0)
                || known_next_token_id.is_some_and(|token_id| token_id < 0)
            {
                return Err(native_error(
                    "speculative prefill chunk requires 1 to 512 non-negative token ids",
                ));
            }
            let mut result = SpeculativeStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: token storage and both exclusive native owners remain
            // valid for the complete synchronous chunk transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_prefill_chunk(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_ids.as_ptr(),
                    token_ids.len(),
                    known_next_token_id.unwrap_or(-1),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            speculative_step(result)
        }
    }

    /// Prime target position zero and the corresponding first MTP proposal.
    /// The returned target token is the first token in the exact greedy stream.
    pub fn speculative_seed(
        &mut self,
        session: &mut MlxQwenSpeculativeSession,
        token_id: i32,
    ) -> Result<MlxSpeculativeStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("speculative seed token id cannot be negative"));
            }
            let mut result = SpeculativeStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and speculative session remain exclusively
            // borrowed for the complete synchronous state transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_seed(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            speculative_step(result)
        }
    }

    /// Propose up to `draft_depth` MTP tokens, verify them in one target batch,
    /// and commit no more than `max_output_tokens`. Rejections always commit
    /// the verifying target row; the configured numerical path may use a
    /// mathematically equivalent reduction tree.
    pub fn speculative_step(
        &mut self,
        session: &mut MlxQwenSpeculativeSession,
        draft_depth: usize,
        max_output_tokens: usize,
    ) -> Result<MlxSpeculativeStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, draft_depth, max_output_tokens);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if max_output_tokens == 0 || max_output_tokens > QWEN_MAX_BATCH_CHECKPOINT_TOKENS {
                return Err(native_error(
                    "speculative step max output must be between 1 and 8 tokens",
                ));
            }
            let mut result = SpeculativeStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and speculative session remain exclusively
            // borrowed for the complete synchronous state transition.
            let status = unsafe {
                generalist_mlx_qwen_speculative_step(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    draft_depth,
                    max_output_tokens,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            speculative_step(result)
        }
    }

    /// Execute the first target/MTP/target pair and summarize every numerical
    /// boundary needed to compare this implementation with the pinned Ollama
    /// oracle. This diagnostic requires a model loaded by
    /// [`Self::load_with_mtp`].
    pub fn mtp_checkpoint(&mut self, token_id: i32) -> Result<MlxMtpCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = token_id;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("MTP checkpoint token id cannot be negative"));
            }
            let mut result = MtpCheckpointResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the uniquely borrowed model remains valid throughout
            // this synchronous diagnostic and the shim writes one fixed-size
            // result without retaining Rust pointers.
            let status = unsafe {
                generalist_mlx_qwen_mtp_checkpoint(
                    self.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxMtpCheckpoint {
                input_token_id: result.input_token_id,
                lookahead_token_id: result.lookahead_token_id,
                draft_token_id: result.draft_token_id,
                target_token_id: result.target_token_id,
                target_hidden: array_fingerprint(result.target_hidden),
                draft_hidden: array_fingerprint(result.draft_hidden),
                draft_logits: array_fingerprint(result.draft_logits),
                next_target_hidden: array_fingerprint(result.next_target_hidden),
                next_target_logits: array_fingerprint(result.next_target_logits),
            })
        }
    }

    /// Compare a fused target forward with sequential cached decoding for up
    /// to eight tokens. The fused cache is private to the diagnostic, so this
    /// method cannot alter a live generation session.
    pub fn batch_decode_checkpoint(
        &mut self,
        token_ids: &[i32],
    ) -> Result<MlxBatchDecodeCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = token_ids;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_ids.is_empty() || token_ids.len() > QWEN_MAX_BATCH_CHECKPOINT_TOKENS {
                return Err(native_error(
                    "batch decode checkpoint requires 1 to 8 tokens",
                ));
            }
            if token_ids.iter().any(|token_id| *token_id < 0) {
                return Err(native_error(
                    "batch decode checkpoint token ids cannot be negative",
                ));
            }
            let mut result = BatchDecodeCheckpointResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the token slice and fixed result storage outlive this
            // synchronous call; the shim retains neither pointer.
            let status = unsafe {
                generalist_mlx_qwen_batch_decode_checkpoint(
                    self.raw.as_ptr(),
                    token_ids.as_ptr(),
                    token_ids.len(),
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            if result.token_count != token_ids.len() {
                return Err(native_error(
                    "native batch decode checkpoint returned the wrong row count",
                ));
            }
            let rows = (0..result.token_count)
                .map(|index| MlxBatchDecodeRow {
                    input_token_id: result.input_token_ids[index],
                    batch_output_token_id: result.batch_output_token_ids[index],
                    sequential_output_token_id: result.sequential_output_token_ids[index],
                    batch_hidden: array_fingerprint(result.batch_hidden[index]),
                    batch_logits: array_fingerprint(result.batch_logits[index]),
                    sequential_hidden: array_fingerprint(result.sequential_hidden[index]),
                    sequential_logits: array_fingerprint(result.sequential_logits[index]),
                })
                .collect();
            Ok(MlxBatchDecodeCheckpoint { rows })
        }
    }

    /// Decode one token at the session's current position, atomically
    /// materialize all recurrent/KV outputs, and advance the session by one.
    pub fn cached_decode_checkpoint(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxCachedDecodeCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = CachedDecodeResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and session are separately, exclusively borrowed
            // for the complete synchronous state transition.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_checkpoint(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxCachedDecodeCheckpoint {
                position: result.position,
                decode: decode_checkpoint(result.decode),
            })
        }
    }

    /// Benchmark oracle for exact fused NVFP4 scaling. This recreates the
    /// former three-dispatch BF16-to-F32, scale, and F32-to-BF16 graph.
    pub fn cached_decode_unfused_scale_checkpoint(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxCachedDecodeCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = CachedDecodeResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and session are separately, exclusively borrowed
            // for the complete synchronous state transition.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_unfused_scale_checkpoint(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxCachedDecodeCheckpoint {
                position: result.position,
                decode: decode_checkpoint(result.decode),
            })
        }
    }

    /// Benchmark oracle for the former four-results-per-SIMD QMV geometry.
    /// Production uses two results for large NVFP4 decode projections.
    pub fn cached_decode_wide_qmv_checkpoint(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxCachedDecodeCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = CachedDecodeResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and session are separately, exclusively borrowed
            // for the complete synchronous state transition.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_wide_qmv_checkpoint(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxCachedDecodeCheckpoint {
                position: result.position,
                decode: decode_checkpoint(result.decode),
            })
        }
    }

    /// Execute the real greedy decode hot path without copying hidden states
    /// or the vocabulary logits back to the CPU for checkpoint summaries.
    pub fn cached_decode_argmax(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and sequence cache remain exclusively borrowed
            // until the native state transition has completed.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_argmax(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Benchmark oracle for the unfused NVFP4 global-scale graph. Production
    /// retains the same rounding boundary inside each Metal matvec kernel.
    pub fn cached_decode_argmax_unfused_scale(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and sequence cache remain exclusively borrowed
            // until the native state transition has completed.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_argmax_unfused_scale(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Benchmark oracle for the former wide QMV output tile. The regular
    /// method is the exact two-results-per-SIMD production path.
    pub fn cached_decode_argmax_wide_qmv(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and sequence cache remain exclusively borrowed
            // until the native state transition has completed.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_argmax_wide_qmv(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Benchmark oracle for the precomputed-norm optimization. This executes
    /// the former exact graph, rebuilding every shifted RMSNorm weight during
    /// each token. It is intentionally separate from the production hot path.
    pub fn cached_decode_argmax_dynamic_norm(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and sequence cache remain exclusively borrowed
            // until the native state transition has completed.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_argmax_dynamic_norm(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Benchmark oracle for packed gate/up projection. This keeps the exact
    /// precomputed norm path but executes separate gate and up NVFP4 matvecs,
    /// matching the graph before model-specific MLP prepacking.
    pub fn cached_decode_argmax_split_mlp(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and sequence cache remain exclusively borrowed
            // until the native state transition has completed.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_argmax_split_mlp(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Experimental Ollama-style decode schedule. The sampled token is
    /// synchronized, but recurrent/KV outputs remain lazy until a subsequent
    /// token consumes them. This can shorten the current call but weakens the
    /// eager method's atomic materialization guarantee, so it remains a
    /// separately benchmarked lane.
    pub fn cached_decode_argmax_deferred_state(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and session remain exclusively borrowed. The
            // deferred arrays themselves stay owned by the native session.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_argmax_deferred_state(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Experimental asynchronous state schedule. All recurrent/KV roots are
    /// submitted to MLX, while the host synchronizes only the sampled token.
    /// This tests whether scheduling the branches early improves over leaving
    /// them lazy without changing the eager reference API.
    pub fn cached_decode_argmax_async_state(
        &mut self,
        session: &mut MlxQwenSession,
        token_id: i32,
    ) -> Result<MlxTokenStep> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = (session, token_id);
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("cached decode token id cannot be negative"));
            }
            let mut result = TokenStepResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: model and session remain exclusively borrowed, and MLX
            // retains scheduled arrays through the native session owner.
            let status = unsafe {
                generalist_mlx_qwen_cached_decode_argmax_async_state(
                    self.raw.as_ptr(),
                    session.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxTokenStep {
                position: result.position,
                input_token_id: result.input_token_id,
                output_token_id: result.output_token_id,
            })
        }
    }

    /// Execute the exact full-model first-token reference path. The full
    /// attention layers specialize the one-key softmax to one, which is
    /// mathematically exact for an empty KV cache.
    pub fn zero_state_decode_checkpoint(&mut self, token_id: i32) -> Result<MlxDecodeCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = token_id;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("decode token id cannot be negative"));
            }
            let mut result = DecodeResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the opaque model is uniquely borrowed for the complete
            // synchronous decode and the output ABI is correctly sized.
            let status = unsafe {
                generalist_mlx_qwen_zero_state_decode_checkpoint(
                    self.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(decode_checkpoint(result))
        }
    }

    /// Run the explicitly non-reference lane that applies each scalar NVFP4
    /// global scale inside the Metal matvec. It removes a rounding boundary
    /// and may therefore differ numerically; callers must compare it against
    /// `zero_state_decode_checkpoint` and the reasoning-quality gates.
    pub fn zero_state_decode_experimental_fused_checkpoint(
        &mut self,
        token_id: i32,
    ) -> Result<MlxDecodeCheckpoint> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = token_id;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error(
                    "experimental decode token id cannot be negative",
                ));
            }
            let mut result = DecodeResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the opaque model is exclusively borrowed and the
            // experimental result uses the same fixed decode ABI.
            let status = unsafe {
                generalist_mlx_qwen_zero_state_decode_experimental_fused_checkpoint(
                    self.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(decode_checkpoint(result))
        }
    }

    pub fn compare_zero_state_fused(&mut self, token_id: i32) -> Result<MlxDecodeComparison> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = token_id;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("comparison token id cannot be negative"));
            }
            let mut result = DecodeComparisonResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: native code exclusively owns the model during both
            // forwards and writes a fixed comparison ABI.
            let status = unsafe {
                generalist_mlx_qwen_compare_zero_state_fused(
                    self.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxDecodeComparison {
                exact: decode_checkpoint(result.exact),
                experimental_fused: decode_checkpoint(result.experimental_fused),
                hidden_drift: array_drift(result.hidden_drift),
                logits_drift: array_drift(result.logits_drift),
            })
        }
    }

    pub fn zero_state_decode_profile(&mut self, token_id: i32) -> Result<MlxDecodeProfile> {
        #[cfg(not(generalist_mlx_native_linked))]
        {
            let _ = token_id;
            return Err(native_error("native MLX was not linked into this build"));
        }
        #[cfg(generalist_mlx_native_linked)]
        {
            if token_id < 0 {
                return Err(native_error("profile token id cannot be negative"));
            }
            let mut result = DecodeProfileResult::default();
            let mut error = [0 as c_char; ERROR_CAPACITY];
            // SAFETY: the native profiler holds the exclusive model borrow and
            // writes exactly one fixed-size 64-layer result.
            let status = unsafe {
                generalist_mlx_qwen_zero_state_decode_profile(
                    self.raw.as_ptr(),
                    token_id,
                    &mut result,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                return Err(native_error(error_message(&error)));
            }
            Ok(MlxDecodeProfile {
                decode: decode_checkpoint(result.decode),
                layer_milliseconds: result.layer_milliseconds,
                final_norm_milliseconds: result.final_norm_milliseconds,
                lm_head_argmax_milliseconds: result.lm_head_argmax_milliseconds,
                graph_milliseconds: result.graph_milliseconds,
                checkpoint_milliseconds: result.checkpoint_milliseconds,
            })
        }
    }
}

#[cfg(generalist_mlx_native_linked)]
fn decode_checkpoint(result: DecodeResult) -> MlxDecodeCheckpoint {
    MlxDecodeCheckpoint {
        input_token_id: result.input_token_id,
        output_token_id: result.output_token_id,
        hidden: array_fingerprint(result.hidden),
        logits: array_fingerprint(result.logits),
    }
}

#[cfg(generalist_mlx_native_linked)]
fn speculative_step(result: SpeculativeStepResult) -> Result<MlxSpeculativeStep> {
    if result.output_count == 0 || result.output_count > QWEN_MAX_BATCH_CHECKPOINT_TOKENS {
        return Err(native_error(
            "native speculative decode returned an invalid output count",
        ));
    }
    if result.accepted > result.drafted || result.accepted + 1 != result.output_count {
        return Err(native_error(
            "native speculative decode returned inconsistent acceptance counts",
        ));
    }
    Ok(MlxSpeculativeStep {
        position: result.position,
        input_token_id: result.input_token_id,
        output_token_ids: result.output_token_ids[..result.output_count].to_vec(),
        drafted: result.drafted,
        accepted: result.accepted,
        proposal_milliseconds: result.proposal_milliseconds,
        verification_milliseconds: result.verification_milliseconds,
        replay_milliseconds: result.replay_milliseconds,
        correction_milliseconds: result.correction_milliseconds,
    })
}

#[cfg(generalist_mlx_native_linked)]
fn array_drift(result: ArrayDriftResult) -> ArrayDrift {
    ArrayDrift {
        max_abs: result.max_abs,
        mean_abs: result.mean_abs,
        root_mean_square: result.root_mean_square,
        relative_l2: result.relative_l2,
        cosine_similarity: result.cosine_similarity,
    }
}

#[cfg(generalist_mlx_native_linked)]
fn layer0_checkpoint_fingerprint(result: Layer0ProjectionResult) -> Layer0Checkpoint {
    Layer0Checkpoint {
        gdn: layer0_gdn_fingerprint(result),
        mlp_width: result.mlp_width,
        gdn_norm: array_fingerprint(result.gdn_norm),
        gated: array_fingerprint(result.gated),
        attention_out: array_fingerprint(result.attention_out),
        after_attention: array_fingerprint(result.after_attention),
        post_norm: array_fingerprint(result.post_norm),
        mlp_gate: array_fingerprint(result.mlp_gate),
        mlp_up: array_fingerprint(result.mlp_up),
        swiglu: array_fingerprint(result.swiglu),
        mlp_out: array_fingerprint(result.mlp_out),
        layer_out: array_fingerprint(result.layer_out),
    }
}

#[cfg(generalist_mlx_native_linked)]
fn layer0_gdn_fingerprint(result: Layer0ProjectionResult) -> Layer0GdnCheckpoint {
    Layer0GdnCheckpoint {
        projections: layer0_projection_fingerprint(result),
        conv_kernel_width: result.conv_kernel_width,
        key_heads: result.key_heads,
        key_dim: result.key_dim,
        value_heads: result.value_heads,
        value_dim: result.value_dim,
        conv_out: array_fingerprint(result.conv_out),
        conv_state: array_fingerprint(result.conv_state),
        a_exp: array_fingerprint(result.a_exp),
        gdn_out: array_fingerprint(result.gdn_out),
        delta_state: array_fingerprint(result.delta_state),
    }
}

#[cfg(generalist_mlx_native_linked)]
fn layer0_projection_fingerprint(result: Layer0ProjectionResult) -> Layer0ProjectionCheckpoint {
    Layer0ProjectionCheckpoint {
        token_id: result.token_id,
        input_width: result.input_width,
        qkv_width: result.qkv_width,
        z_width: result.z_width,
        ba_width: result.ba_width,
        qkv_global_scale: result.qkv_global_scale,
        z_global_scale: result.z_global_scale,
        normalized_input: array_fingerprint(result.normalized_input),
        raw_qkvz: array_fingerprint(result.raw_qkvz),
        mixed_qkvz: array_fingerprint(result.mixed_qkvz),
        mixed_ba: array_fingerprint(result.mixed_ba),
    }
}

#[cfg(generalist_mlx_native_linked)]
fn array_fingerprint(result: ArraySummaryResult) -> ArrayFingerprint {
    ArrayFingerprint {
        dtype: result.dtype,
        elements: result.elements,
        sum: result.sum,
        squared_sum: result.squared_sum,
        first: result.first,
        middle: result.middle,
        last: result.last,
        fingerprint: result.fingerprint,
    }
}

#[cfg(generalist_mlx_native_linked)]
fn path_to_cstring(path: &Path, description: &str) -> Result<CString> {
    let encoded = path
        .to_str()
        .ok_or_else(|| native_error(format!("{description} path is not valid UTF-8")))?;
    CString::new(encoded).map_err(|_| native_error(format!("{description} path contains NUL")))
}

#[cfg(generalist_mlx_native_linked)]
fn tensor_source_to_cstrings(
    source: NativeTensorSource<'_>,
    description: &str,
) -> Result<(CString, CString)> {
    let path = path_to_cstring(source.path, &format!("{description} safetensors blob"))?;
    let name = CString::new(source.name)
        .map_err(|_| native_error(format!("{description} name contains a NUL byte")))?;
    Ok((path, name))
}

#[cfg(generalist_mlx_native_linked)]
fn error_message(buffer: &[c_char]) -> String {
    // SAFETY: the zero-initialized buffer remains NUL terminated even if the
    // shim returns without writing a message.
    let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
    if message.is_empty() {
        "native MLX operation failed without an error message".to_string()
    } else {
        message.into_owned()
    }
}

fn native_error(message: impl Into<String>) -> Error {
    Error::Other(format!("embedded MLX: {}", message.into()))
}
