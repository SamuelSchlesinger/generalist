#ifndef GENERALIST_MLX_SHIM_H
#define GENERALIST_MLX_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GENERALIST_MLX_MAX_TENSOR_DIMS 8
#define GENERALIST_MLX_QWEN_LAYER_COUNT 64
#define GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS 8

typedef struct generalist_mlx_tensor_probe_result {
  int dtype;
  size_t ndim;
  int shape[GENERALIST_MLX_MAX_TENSOR_DIMS];
  size_t elements;
  size_t bytes;
  float gpu_sum;
} generalist_mlx_tensor_probe_result;

typedef struct generalist_mlx_rms_norm_result {
  int input_dtype;
  int weight_dtype;
  int output_dtype;
  size_t width;
  double output_sum;
  double output_squared_sum;
  float output_first;
  float output_middle;
  float output_last;
  double max_abs_reference_error;
  unsigned long long output_fingerprint;
} generalist_mlx_rms_norm_result;

typedef struct generalist_mlx_array_summary {
  int dtype;
  size_t elements;
  double sum;
  double squared_sum;
  float first;
  float middle;
  float last;
  unsigned long long fingerprint;
} generalist_mlx_array_summary;

typedef struct generalist_mlx_embedding_result {
  int token_id;
  size_t width;
  generalist_mlx_array_summary embedding;
  generalist_mlx_array_summary normalized;
} generalist_mlx_embedding_result;

typedef struct generalist_mlx_nvfp4_projection_result {
  int token_id;
  size_t input_width;
  size_t output_width;
  int group_size;
  int bits;
  int weight_dtype;
  int scale_dtype;
  int global_scale_dtype;
  float global_scale;
  float encoded_global_scale;
  generalist_mlx_array_summary normalized_input;
  generalist_mlx_array_summary raw_projection;
  generalist_mlx_array_summary output;
  generalist_mlx_array_summary fused_output;
} generalist_mlx_nvfp4_projection_result;

typedef struct generalist_mlx_layer0_projection_result {
  int token_id;
  size_t input_width;
  size_t qkv_width;
  size_t z_width;
  size_t ba_width;
  float qkv_global_scale;
  float z_global_scale;
  generalist_mlx_array_summary normalized_input;
  generalist_mlx_array_summary raw_qkvz;
  generalist_mlx_array_summary mixed_qkvz;
  generalist_mlx_array_summary mixed_ba;
  int gdn_executed;
  size_t conv_kernel_width;
  size_t key_heads;
  size_t key_dim;
  size_t value_heads;
  size_t value_dim;
  generalist_mlx_array_summary conv_out;
  generalist_mlx_array_summary conv_state;
  generalist_mlx_array_summary a_exp;
  generalist_mlx_array_summary gdn_out;
  generalist_mlx_array_summary delta_state;
  int full_layer_executed;
  size_t mlp_width;
  generalist_mlx_array_summary gdn_norm;
  generalist_mlx_array_summary gated;
  generalist_mlx_array_summary attention_out;
  generalist_mlx_array_summary after_attention;
  generalist_mlx_array_summary post_norm;
  generalist_mlx_array_summary mlp_gate;
  generalist_mlx_array_summary mlp_up;
  generalist_mlx_array_summary swiglu;
  generalist_mlx_array_summary mlp_out;
  generalist_mlx_array_summary layer_out;
} generalist_mlx_layer0_projection_result;

typedef struct generalist_mlx_model_blob_source {
  const char *path;
} generalist_mlx_model_blob_source;

typedef struct generalist_mlx_model_load_result {
  size_t blob_count;
  size_t array_count;
  size_t layer_count;
  size_t linear_layer_count;
  size_t full_attention_layer_count;
  size_t mtp_layer_count;
} generalist_mlx_model_load_result;

typedef struct generalist_mlx_decode_result {
  int input_token_id;
  int output_token_id;
  generalist_mlx_array_summary hidden;
  generalist_mlx_array_summary logits;
} generalist_mlx_decode_result;

typedef struct generalist_mlx_cached_decode_result {
  size_t position;
  generalist_mlx_decode_result decode;
} generalist_mlx_cached_decode_result;

typedef struct generalist_mlx_token_step_result {
  size_t position;
  int input_token_id;
  int output_token_id;
} generalist_mlx_token_step_result;

typedef struct generalist_mlx_mtp_checkpoint_result {
  int input_token_id;
  int lookahead_token_id;
  int draft_token_id;
  int target_token_id;
  generalist_mlx_array_summary target_hidden;
  generalist_mlx_array_summary draft_hidden;
  generalist_mlx_array_summary draft_logits;
  generalist_mlx_array_summary next_target_hidden;
  generalist_mlx_array_summary next_target_logits;
} generalist_mlx_mtp_checkpoint_result;

typedef struct generalist_mlx_batch_decode_checkpoint_result {
  size_t token_count;
  int input_token_ids[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
  int batch_output_token_ids[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
  int sequential_output_token_ids[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
  generalist_mlx_array_summary
      batch_hidden[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
  generalist_mlx_array_summary
      batch_logits[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
  generalist_mlx_array_summary
      sequential_hidden[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
  generalist_mlx_array_summary
      sequential_logits[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
} generalist_mlx_batch_decode_checkpoint_result;

typedef struct generalist_mlx_speculative_step_result {
  size_t position;
  int input_token_id;
  size_t output_count;
  int output_token_ids[GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS];
  size_t drafted;
  size_t accepted;
  double proposal_milliseconds;
  double verification_milliseconds;
  double replay_milliseconds;
  double correction_milliseconds;
} generalist_mlx_speculative_step_result;

typedef struct generalist_mlx_decode_profile_result {
  generalist_mlx_decode_result decode;
  double layer_milliseconds[GENERALIST_MLX_QWEN_LAYER_COUNT];
  double final_norm_milliseconds;
  double lm_head_argmax_milliseconds;
  double graph_milliseconds;
  double checkpoint_milliseconds;
} generalist_mlx_decode_profile_result;

typedef struct generalist_mlx_array_drift {
  double max_abs;
  double mean_abs;
  double root_mean_square;
  double relative_l2;
  double cosine_similarity;
} generalist_mlx_array_drift;

typedef struct generalist_mlx_decode_comparison_result {
  generalist_mlx_decode_result exact;
  generalist_mlx_decode_result experimental_fused;
  generalist_mlx_array_drift hidden_drift;
  generalist_mlx_array_drift logits_drift;
} generalist_mlx_decode_comparison_result;

typedef struct generalist_mlx_qwen_model generalist_mlx_qwen_model;
typedef struct generalist_mlx_qwen_cache generalist_mlx_qwen_cache;
typedef struct generalist_mlx_qwen_speculative_cache
    generalist_mlx_qwen_speculative_cache;

int generalist_mlx_initialize(const char *metallib_path, char *error,
                              size_t error_capacity);

int generalist_mlx_probe_tensor(const char *safetensors_path,
                                const char *tensor_name,
                                generalist_mlx_tensor_probe_result *result,
                                char *error, size_t error_capacity);

int generalist_mlx_rms_norm_checkpoint(
    const char *safetensors_path, const char *weight_name, size_t width,
    float epsilon, int shift_weight, generalist_mlx_rms_norm_result *result,
    char *error, size_t error_capacity);

int generalist_mlx_embedding_checkpoint(
    const char *embedding_path, const char *embedding_name,
    const char *norm_path, const char *norm_name, int token_id, size_t width,
    float epsilon, int shift_norm_weight,
    generalist_mlx_embedding_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_nvfp4_projection_checkpoint(
    const char *embedding_path, const char *embedding_name,
    const char *norm_path, const char *norm_name,
    const char *projection_path, const char *weight_name,
    const char *scale_name, const char *global_scale_name, int token_id,
    size_t input_width, size_t output_width, float epsilon,
    int shift_norm_weight, generalist_mlx_nvfp4_projection_result *result,
    char *error, size_t error_capacity);

int generalist_mlx_layer0_projection_checkpoint(
    const char *embedding_path, const char *embedding_name,
    const char *norm_path, const char *norm_name, const char *qkv_path,
    const char *qkv_name, const char *z_path, const char *z_name,
    const char *beta_path, const char *beta_name, const char *alpha_path,
    const char *alpha_name, const char *conv_path, const char *conv_name,
    const char *dt_bias_path, const char *dt_bias_name,
    const char *a_log_path, const char *a_log_name,
    const char *gdn_norm_path, const char *gdn_norm_name,
    const char *out_proj_path, const char *out_proj_name,
    const char *post_norm_path, const char *post_norm_name,
    const char *mlp_gate_path, const char *mlp_gate_name,
    const char *mlp_up_path, const char *mlp_up_name,
    const char *mlp_down_path, const char *mlp_down_name, int token_id,
    size_t input_width, size_t qkv_width, size_t z_width, size_t ba_width,
    size_t mlp_width, float epsilon, int shift_norm_weight,
    generalist_mlx_layer0_projection_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_model_new(
    const generalist_mlx_model_blob_source *sources, size_t source_count,
    generalist_mlx_qwen_model **model,
    generalist_mlx_model_load_result *result, char *error,
    size_t error_capacity);

void generalist_mlx_qwen_model_free(generalist_mlx_qwen_model *model);

int generalist_mlx_qwen_cache_new(generalist_mlx_qwen_cache **cache,
                                  char *error, size_t error_capacity);

void generalist_mlx_qwen_cache_free(generalist_mlx_qwen_cache *cache);

int generalist_mlx_qwen_cache_reset(generalist_mlx_qwen_cache *cache,
                                    char *error, size_t error_capacity);

int generalist_mlx_qwen_speculative_cache_new(
    generalist_mlx_qwen_speculative_cache **cache, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_speculative_cache_clone(
    const generalist_mlx_qwen_speculative_cache *source,
    generalist_mlx_qwen_speculative_cache **cache, char *error,
    size_t error_capacity);

void generalist_mlx_qwen_speculative_cache_free(
    generalist_mlx_qwen_speculative_cache *cache);

int generalist_mlx_qwen_speculative_cache_reset(
    generalist_mlx_qwen_speculative_cache *cache, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_speculative_cache_set_verification_geometry(
    generalist_mlx_qwen_speculative_cache *cache,
    int results_per_simdgroup, char *error, size_t error_capacity);

int generalist_mlx_qwen_speculative_cache_set_verification_shape_override(
    generalist_mlx_qwen_speculative_cache *cache,
    int default_results_per_simdgroup, int output_width, int input_width,
    int override_results_per_simdgroup, char *error, size_t error_capacity);

int generalist_mlx_qwen_speculative_cache_set_verification_qmv_path(
    generalist_mlx_qwen_speculative_cache *cache, int path, char *error,
    size_t error_capacity);
int generalist_mlx_qwen_speculative_cache_set_mtp_draft_vocab_limit(
    generalist_mlx_qwen_speculative_cache *cache, size_t vocab_limit,
    char *error, size_t error_capacity);
int generalist_mlx_qwen_speculative_cache_set_mtp_draft_extra_tokens(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, const int *token_ids,
    size_t token_count, char *error, size_t error_capacity);
int generalist_mlx_qwen_speculative_cache_set_direct_cache_rollback(
    generalist_mlx_qwen_speculative_cache *cache, int enabled,
    char *error, size_t error_capacity);

int generalist_mlx_qwen_prefill(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    const int *token_ids, size_t token_count,
    generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_speculative_prefill(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, const int *token_ids,
    size_t token_count, generalist_mlx_speculative_step_result *result,
    char *error, size_t error_capacity);

int generalist_mlx_qwen_speculative_prefill_chunk(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, const int *token_ids,
    size_t token_count, int next_token_id,
    generalist_mlx_speculative_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_speculative_seed(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, int token_id,
    generalist_mlx_speculative_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_speculative_step(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, size_t draft_depth,
    size_t max_output_tokens, generalist_mlx_speculative_step_result *result,
    char *error, size_t error_capacity);

int generalist_mlx_qwen_cached_decode_checkpoint(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_cached_decode_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_unfused_scale_checkpoint(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_cached_decode_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_wide_qmv_checkpoint(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_cached_decode_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_argmax(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_argmax_unfused_scale(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_argmax_wide_qmv(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_argmax_dynamic_norm(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_argmax_split_mlp(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_argmax_deferred_state(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_cached_decode_argmax_async_state(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_mtp_checkpoint(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_mtp_checkpoint_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_batch_decode_checkpoint(
    generalist_mlx_qwen_model *model, const int *token_ids,
    size_t token_count, generalist_mlx_batch_decode_checkpoint_result *result,
    char *error, size_t error_capacity);

int generalist_mlx_qwen_zero_state_decode_checkpoint(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_zero_state_decode_experimental_fused_checkpoint(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_zero_state_decode_profile(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_profile_result *result, char *error,
    size_t error_capacity);

int generalist_mlx_qwen_compare_zero_state_fused(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_comparison_result *result, char *error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
