#include "shim.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/backend/common/quantized.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/c/error.h"
#include "mlx/c/mlx.h"
#include "mlx/c/private/mlx.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace {

constexpr size_t maximum_probe_bytes = 16U * 1024U * 1024U;

// MLX-C's process-global default handler calls exit(-1). The Rust boundary
// requires native failures to return through the status/error-buffer ABI, so
// install a non-terminating handler before any MLX-C operation. MLX-C wrappers
// already translate their C++ exceptions into failure return values.
void ignore_mlx_c_error(const char *, void *) {}

void install_nonterminating_mlx_error_handler() {
  static std::once_flag installed;
  std::call_once(installed,
                 [] { mlx_set_error_handler(ignore_mlx_c_error, nullptr, nullptr); });
}

void set_error(char *destination, size_t capacity, const char *message) {
  if (destination == nullptr || capacity == 0) {
    return;
  }
  std::snprintf(destination, capacity, "%s", message);
}

int fail(char *error, size_t capacity, const char *message) {
  set_error(error, capacity, message);
  return 1;
}

float checkpoint_input(size_t index) {
  const int centered = static_cast<int>(index % 257U) - 128;
  const int fine = static_cast<int>(index % 7U) - 3;
  return static_cast<float>(centered) / 64.0F +
         static_cast<float>(fine) / 1024.0F;
}

uint64_t fingerprint_float(uint64_t fingerprint, float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (size_t index = 0; index < sizeof(bits); ++index) {
    fingerprint ^= static_cast<uint8_t>(bits >> (index * 8U));
    fingerprint *= 1099511628211ULL;
  }
  return fingerprint;
}

bool summarize_array(mlx_array value, mlx_stream stream,
                     generalist_mlx_array_summary *summary, char *error,
                     size_t error_capacity) {
  mlx_array as_float = mlx_array_new();
  if (mlx_astype(&as_float, value, MLX_FLOAT32, stream) != 0 ||
      mlx_array_eval(as_float) != 0) {
    mlx_array_free(as_float);
    set_error(error, error_capacity, "MLX failed to evaluate checkpoint output");
    return false;
  }
  const float *data = mlx_array_data_float32(as_float);
  if (data == nullptr) {
    mlx_array_free(as_float);
    set_error(error, error_capacity,
              "MLX did not expose evaluated checkpoint output");
    return false;
  }

  *summary = {};
  summary->dtype = static_cast<int>(mlx_array_dtype(value));
  summary->elements = mlx_array_size(value);
  uint64_t fingerprint = 1469598103934665603ULL;
  for (size_t index = 0; index < summary->elements; ++index) {
    const double converted = data[index];
    summary->sum += converted;
    summary->squared_sum += converted * converted;
    fingerprint = fingerprint_float(fingerprint, data[index]);
  }
  if (summary->elements != 0) {
    summary->first = data[0];
    summary->middle = data[summary->elements / 2U];
    summary->last = data[summary->elements - 1U];
  }
  summary->fingerprint = fingerprint;
  mlx_array_free(as_float);
  return true;
}

bool make_fused_nvfp4_qmv(mlx_array *result, mlx_array input,
                          mlx_array weight, mlx_array scales,
                          mlx_array encoded_global_scale, mlx_stream stream,
                          char *error, size_t error_capacity) {
  try {
    const auto &input_array = mlx_array_get_(input);
    const auto &weight_array = mlx_array_get_(weight);
    auto output_shape = input_array.shape();
    output_shape.back() = weight_array.shape(-2);
    std::vector<mlx::core::array> inputs{
        input_array,
        weight_array,
        mlx_array_get_(scales),
        mlx_array_get_(encoded_global_scale),
    };
    mlx_array_set_(
        *result,
        mlx::core::array(
            std::move(output_shape), input_array.dtype(),
            std::make_shared<mlx::core::QuantizedMatmul>(
                mlx_stream_get_(stream), 16, 4,
                mlx::core::QuantizationMode::Nvfp4, true, true),
            std::move(inputs)));
    return true;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    return false;
  }
}

bool load_nvfp4_projection(
    const char *path, const char *name, size_t input_width,
    size_t output_width, mlx_stream cpu_stream,
    mlx_map_string_to_array *arrays, mlx_map_string_to_string *metadata,
    mlx_array *weight, mlx_array *scale, mlx_array *global_scale, char *error,
    size_t error_capacity) {
  const std::string scale_name = std::string(name) + ".scale";
  const std::string global_scale_name = std::string(name) + ".global_scale";
  constexpr size_t group_size = 16U;
  constexpr size_t bits = 4U;
  const size_t packed_width = input_width * bits / 32U;
  const size_t scale_width = input_width / group_size;
  if (mlx_load_safetensors(arrays, metadata, path, cpu_stream) != 0 ||
      mlx_map_string_to_array_get(weight, *arrays, name) != 0 ||
      mlx_map_string_to_array_get(scale, *arrays, scale_name.c_str()) != 0 ||
      mlx_map_string_to_array_get(global_scale, *arrays,
                                  global_scale_name.c_str()) != 0 ||
      weight->ctx == nullptr || scale->ctx == nullptr ||
      global_scale->ctx == nullptr || mlx_array_ndim(*weight) != 2 ||
      mlx_array_dim(*weight, 0) != static_cast<int>(output_width) ||
      mlx_array_dim(*weight, 1) != static_cast<int>(packed_width) ||
      mlx_array_ndim(*scale) != 2 ||
      mlx_array_dim(*scale, 0) != static_cast<int>(output_width) ||
      mlx_array_dim(*scale, 1) != static_cast<int>(scale_width) ||
      mlx_array_size(*global_scale) != 1) {
    set_error(error, error_capacity,
              "NVFP4 projection arrays are missing or have the wrong shape");
    return false;
  }
  return true;
}

bool make_nvfp4_linear_reference(mlx_array *result, mlx_array input,
                                 mlx_array weight, mlx_array scales,
                                 mlx_array global_scale, mlx_stream stream,
                                 char *error, size_t error_capacity) {
  constexpr mlx_optional_int group_size = {16, true};
  constexpr mlx_optional_int bits = {4, true};
  mlx_array raw = mlx_array_new();
  mlx_array scaled = mlx_array_new();
  mlx_array no_bias = mlx_array_new();
  const bool success =
      mlx_quantized_matmul(&raw, input, weight, scales, no_bias, true,
                           group_size, bits, "nvfp4", stream) == 0 &&
      mlx_multiply(&scaled, raw, global_scale, stream) == 0 &&
      mlx_astype(result, scaled, mlx_array_dtype(raw), stream) == 0;
  if (!success) {
    set_error(error, error_capacity,
              "MLX failed to execute the reference NVFP4 linear operation");
  }
  mlx_array_free(no_bias);
  mlx_array_free(scaled);
  mlx_array_free(raw);
  return success;
}

// These two kernels adapt the decode geometry exercised by Ollama v0.32.13's
// MIT-licensed MLX runner. Keeping their source at this boundary lets
// Generalist compile and cache the Metal pipelines in its own process while
// the differential fixtures freeze the mathematical result.
constexpr char depthwise_conv_silu_source[] = R"metal(
auto elem = thread_position_in_grid.x;
int B = dims[0];
int T = dims[1];
uint total = uint(B) * uint(T) * uint(C);
if (elem >= total) {
  return;
}

int c = int(elem % uint(C));
int t = int(elem / uint(C)) % T;
int b = int(elem) / (C * T);
auto in_base = (b * (T + K - 1) + t) * C + c;

float acc = 0.0f;
for (int i = 0; i < K; ++i) {
  acc += static_cast<float>(x[in_base + i * C]) *
         static_cast<float>(w[c * K + i]);
}
InT conv_out = static_cast<InT>(acc);
InT sigmoid = generalist_stable_sigmoid(conv_out);
out[elem] = static_cast<InT>(conv_out * sigmoid);
)metal";

constexpr char depthwise_conv_silu_header[] = R"metal(
template <typename T>
T generalist_stable_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
)metal";

bool make_depthwise_conv_silu(mlx_array *result, mlx_array input,
                              mlx_array weight, int batch, int tokens,
                              int channels, int kernel_width, mlx_stream stream,
                              char *error, size_t error_capacity) {
  const char *input_name_data[] = {"x", "w", "dims"};
  const char *output_name_data[] = {"out"};
  const int dims_values[2] = {batch, tokens};
  const int dims_shape[1] = {2};
  const int output_shape[3] = {batch, tokens, channels};
  mlx_vector_string input_names =
      mlx_vector_string_new_data(input_name_data, 3);
  mlx_vector_string output_names =
      mlx_vector_string_new_data(output_name_data, 1);
  mlx_fast_metal_kernel kernel = mlx_fast_metal_kernel_new(
      "generalist_depthwise_conv_silu", input_names, output_names,
      depthwise_conv_silu_source, depthwise_conv_silu_header, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_vector_array inputs = mlx_vector_array_new();
  mlx_vector_array outputs = mlx_vector_array_new();
  mlx_array dims =
      mlx_array_new_data(dims_values, dims_shape, 1, MLX_INT32);
  bool success = false;

  const int elements = batch * tokens * channels;
  if (input_names.ctx != nullptr && output_names.ctx != nullptr &&
      kernel.ctx != nullptr && config.ctx != nullptr && inputs.ctx != nullptr &&
      outputs.ctx != nullptr && dims.ctx != nullptr &&
      mlx_vector_array_append_value(inputs, input) == 0 &&
      mlx_vector_array_append_value(inputs, weight) == 0 &&
      mlx_vector_array_append_value(inputs, dims) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_dtype(
          config, "InT", mlx_array_dtype(input)) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "C",
                                                        channels) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(
          config, "K", kernel_width) == 0 &&
      mlx_fast_metal_kernel_config_set_grid(config,
                                            ((elements + 255) / 256) * 256, 1,
                                            1) == 0 &&
      mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1) == 0 &&
      mlx_fast_metal_kernel_config_add_output_arg(
          config, output_shape, 3, mlx_array_dtype(input)) == 0 &&
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream) ==
          0 &&
      mlx_vector_array_get(result, outputs, 0) == 0) {
    success = true;
  } else {
    set_error(error, error_capacity,
              "MLX failed to construct the depthwise conv-SiLU kernel");
  }

  mlx_array_free(dims);
  mlx_vector_array_free(outputs);
  mlx_vector_array_free(inputs);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  mlx_vector_string_free(output_names);
  mlx_vector_string_free(input_names);
  return success;
}

// Exact multi-row form of MLX's M=1 BF16 GEMV selected for Qwen's packed
// beta/alpha projection (96 outputs by 5120 inputs). The generic T>1 matmul
// takes gemv_wide and changes its reduction tree; this preserves the standard
// bm1/bn8/sm1/sn32/tm4/tn4 GEMV tree while putting token rows on grid z.
constexpr char exact_ba_projection_source[] = R"metal(
auto lane = thread_position_in_threadgroup.x;
auto simd_idx = thread_position_in_threadgroup.y;
auto token = threadgroup_position_in_grid.z;
auto output_block = threadgroup_position_in_grid.x;

threadgroup float partials[64];
float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
int k = (int(simd_idx) * 32 + int(lane)) * 4;
auto vector = x + int(token) * 5120;
auto matrix = w + int(output_block) * 4 * 5120;

#pragma clang loop unroll(full)
for (int iteration = 0; iteration < 5; ++iteration) {
  float values[4] = {
      static_cast<float>(vector[k]),
      static_cast<float>(vector[k + 1]),
      static_cast<float>(vector[k + 2]),
      static_cast<float>(vector[k + 3]),
  };
#pragma clang loop unroll(full)
  for (int row = 0; row < 4; ++row) {
    auto weights = matrix + row * 5120 + k;
    result[row] += static_cast<float>(weights[0]) * values[0];
    result[row] += static_cast<float>(weights[1]) * values[1];
    result[row] += static_cast<float>(weights[2]) * values[2];
    result[row] += static_cast<float>(weights[3]) * values[3];
  }
  k += 1024;
}

#pragma clang loop unroll(full)
for (int row = 0; row < 4; ++row) {
#pragma clang loop unroll(full)
  for (ushort offset = 16; offset >= 1; offset >>= 1) {
    result[row] += simd_shuffle_down(result[row], offset);
  }
}

if (lane == 0) {
#pragma clang loop unroll(full)
  for (int row = 0; row < 4; ++row) {
    partials[int(simd_idx) * 8 + row] = result[row];
  }
}
threadgroup_barrier(mem_flags::mem_none);

if (simd_idx == 0 && lane == 0) {
#pragma clang loop unroll(full)
  for (int other = 1; other < 8; ++other) {
#pragma clang loop unroll(full)
    for (int row = 0; row < 4; ++row) {
      result[row] += partials[other * 8 + row];
    }
  }
  auto output = out + int(token) * 96 + int(output_block) * 4;
#pragma clang loop unroll(full)
  for (int row = 0; row < 4; ++row) {
    output[row] = static_cast<InT>(result[row]);
  }
}
)metal";

bool make_exact_ba_projection(mlx_array *result, mlx_array input,
                              mlx_array weight, int tokens,
                              mlx_stream stream, char *error,
                              size_t error_capacity) {
  const char *input_name_data[] = {"x", "w"};
  const char *output_name_data[] = {"out"};
  const int output_shape[3] = {1, tokens, 96};
  mlx_vector_string input_names =
      mlx_vector_string_new_data(input_name_data, 2);
  mlx_vector_string output_names =
      mlx_vector_string_new_data(output_name_data, 1);
  mlx_fast_metal_kernel kernel = mlx_fast_metal_kernel_new(
      "generalist_exact_ba_projection", input_names, output_names,
      exact_ba_projection_source, "", true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_vector_array inputs = mlx_vector_array_new();
  mlx_vector_array outputs = mlx_vector_array_new();
  bool success = false;

  if (input_names.ctx != nullptr && output_names.ctx != nullptr &&
      kernel.ctx != nullptr && config.ctx != nullptr && inputs.ctx != nullptr &&
      outputs.ctx != nullptr &&
      mlx_vector_array_append_value(inputs, input) == 0 &&
      mlx_vector_array_append_value(inputs, weight) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_dtype(
          config, "InT", mlx_array_dtype(input)) == 0 &&
      mlx_fast_metal_kernel_config_set_grid(config, 24 * 32, 8, tokens) == 0 &&
      mlx_fast_metal_kernel_config_set_thread_group(config, 32, 8, 1) == 0 &&
      mlx_fast_metal_kernel_config_add_output_arg(
          config, output_shape, 3, mlx_array_dtype(input)) == 0 &&
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream) ==
          0 &&
      mlx_vector_array_get(result, outputs, 0) == 0) {
    success = true;
  } else {
    set_error(error, error_capacity,
              "MLX failed to construct exact Qwen BA projection");
  }

  mlx_vector_array_free(outputs);
  mlx_vector_array_free(inputs);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  mlx_vector_string_free(output_names);
  mlx_vector_string_free(input_names);
  return success;
}

constexpr char gated_delta_decode_source[] = R"metal(
constexpr int SIMDGroups = 4;
constexpr int ValuesPerSIMD = DvTile / SIMDGroups;
constexpr int StatePerLane = Dk / 32;
constexpr int QDim = Hk * Dk;

auto lane = thread_position_in_threadgroup.x;
auto simd_idx = thread_position_in_threadgroup.y;
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
auto value_tile = threadgroup_position_in_grid.y;

threadgroup InT q_cache[Dk];
threadgroup InT k_cache[Dk];
threadgroup InT decay_cache[1];
threadgroup InT beta_cache[1];

float state[ValuesPerSIMD][StatePerLane];
for (int value_iter = 0; value_iter < ValuesPerSIMD; ++value_iter) {
  auto dv_idx = value_tile * DvTile + simd_idx + value_iter * SIMDGroups;
  auto state_offset = (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < StatePerLane; ++i) {
    auto d = StatePerLane * lane + i;
    state[value_iter][i] = state_in[state_offset + d];
  }
}

for (int t = 0; t < SeqT; ++t) {
  if (simd_idx == 0) {
    auto token = packed + (b_idx * SeqT + t) * PackedDim;
    float q_values[StatePerLane];
    float k_values[StatePerLane];
    float q_squares = 0.0f;
    float k_squares = 0.0f;

    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      float q_value = static_cast<float>(token[hk_idx * Dk + d]);
      float k_value = static_cast<float>(token[QDim + hk_idx * Dk + d]);
      q_values[i] = q_value;
      k_values[i] = k_value;
      q_squares += q_value * q_value;
      k_squares += k_value * k_value;
    }

    q_squares = simd_sum(q_squares);
    k_squares = simd_sum(k_squares);
    float q_inv_rms =
        metal::precise::rsqrt(q_squares / float(Dk) + 1.0e-6f);
    float k_inv_rms =
        metal::precise::rsqrt(k_squares / float(Dk) + 1.0e-6f);
    InT q_scale = static_cast<InT>(qk_scale[0]);
    InT k_scale = static_cast<InT>(qk_scale[1]);
    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      InT q_norm = static_cast<InT>(q_values[i] * q_inv_rms);
      InT k_norm = static_cast<InT>(k_values[i] * k_inv_rms);
      q_cache[d] = static_cast<InT>(q_norm * q_scale);
      k_cache[d] = static_cast<InT>(k_norm * k_scale);
    }

    if (lane == 0) {
      auto ba_row = (b_idx * SeqT + t) * 2 * Hv;
      InT gate_input =
          static_cast<InT>(ba[ba_row + Hv + hv_idx] + dt_bias[hv_idx]);
      InT softplus = generalist_gdn_logaddexp(gate_input, static_cast<InT>(0));
      float decay = metal::precise::exp(
          -static_cast<float>(softplus) * a_exp[hv_idx]);
      decay_cache[0] = static_cast<InT>(decay);
      beta_cache[0] = generalist_gdn_sigmoid(ba[ba_row + hv_idx]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int value_iter = 0; value_iter < ValuesPerSIMD; ++value_iter) {
    auto dv_idx = value_tile * DvTile + simd_idx + value_iter * SIMDGroups;
    float projection = 0.0f;
    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      state[value_iter][i] *= static_cast<float>(decay_cache[0]);
      projection +=
          state[value_iter][i] * static_cast<float>(k_cache[d]);
    }
    projection = simd_sum(projection);

    auto token = packed + (b_idx * SeqT + t) * PackedDim;
    float v_value =
        static_cast<float>(token[2 * QDim + hv_idx * Dv + dv_idx]);
    float delta =
        (v_value - projection) * static_cast<float>(beta_cache[0]);

    float out_value = 0.0f;
    for (int i = 0; i < StatePerLane; ++i) {
      auto d = StatePerLane * lane + i;
      state[value_iter][i] += static_cast<float>(k_cache[d]) * delta;
      out_value += state[value_iter][i] * static_cast<float>(q_cache[d]);
    }
    out_value = simd_sum(out_value);
    if (lane == 0) {
      y[((b_idx * SeqT + t) * Hv + hv_idx) * Dv + dv_idx] =
          static_cast<InT>(out_value);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (int value_iter = 0; value_iter < ValuesPerSIMD; ++value_iter) {
  auto dv_idx = value_tile * DvTile + simd_idx + value_iter * SIMDGroups;
  auto state_offset = (n * Dv + dv_idx) * Dk;
  for (int i = 0; i < StatePerLane; ++i) {
    auto d = StatePerLane * lane + i;
    state_out[state_offset + d] = state[value_iter][i];
  }
}
)metal";

constexpr char gated_delta_decode_header[] = R"metal(
template <typename T>
T generalist_gdn_sigmoid(T x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

template <typename T>
T generalist_gdn_logaddexp(T x, T y) {
  if (metal::isnan(x) || metal::isnan(y)) {
    return metal::numeric_limits<T>::quiet_NaN();
  }
  constexpr T inf = metal::numeric_limits<T>::infinity();
  T maxval = metal::max(x, y);
  T minval = metal::min(x, y);
  return (minval == -inf || maxval == inf)
             ? maxval
             : (maxval + log1p(metal::exp(minval - maxval)));
}
)metal";

bool make_gated_delta_decode(mlx_array *output, mlx_array *next_state,
                             mlx_array packed, mlx_array ba,
                             mlx_array dt_bias, mlx_array a_exp,
                             mlx_array prior_state, int batch, int tokens,
                             int key_heads, int key_dim, int value_heads,
                             int value_dim, mlx_stream stream, char *error,
                             size_t error_capacity) {
  const char *input_name_data[] = {"packed", "ba",       "dt_bias",
                                   "a_exp",  "qk_scale", "state_in"};
  const char *output_name_data[] = {"y", "state_out"};
  const float inverse_scale =
      1.0F / std::sqrt(static_cast<float>(key_dim));
  const float qk_scale_values[2] = {inverse_scale * inverse_scale,
                                    inverse_scale};
  const int qk_scale_shape[1] = {2};
  const int output_shape[4] = {batch, tokens, value_heads, value_dim};
  const int state_shape[4] = {batch, value_heads, value_dim, key_dim};
  const int packed_dim = 2 * key_heads * key_dim + value_heads * value_dim;
  constexpr int value_tile = 32;
  mlx_vector_string input_names =
      mlx_vector_string_new_data(input_name_data, 6);
  mlx_vector_string output_names =
      mlx_vector_string_new_data(output_name_data, 2);
  mlx_fast_metal_kernel kernel = mlx_fast_metal_kernel_new(
      "generalist_gated_delta_decode", input_names, output_names,
      gated_delta_decode_source, gated_delta_decode_header, true, false);
  mlx_fast_metal_kernel_config config = mlx_fast_metal_kernel_config_new();
  mlx_vector_array inputs = mlx_vector_array_new();
  mlx_vector_array outputs = mlx_vector_array_new();
  mlx_array qk_scale = mlx_array_new_data(qk_scale_values, qk_scale_shape, 1,
                                          MLX_FLOAT32);
  bool success = false;

  if (input_names.ctx != nullptr && output_names.ctx != nullptr &&
      kernel.ctx != nullptr && config.ctx != nullptr && inputs.ctx != nullptr &&
      outputs.ctx != nullptr && qk_scale.ctx != nullptr &&
      key_dim % 32 == 0 && value_dim % value_tile == 0 &&
      mlx_vector_array_append_value(inputs, packed) == 0 &&
      mlx_vector_array_append_value(inputs, ba) == 0 &&
      mlx_vector_array_append_value(inputs, dt_bias) == 0 &&
      mlx_vector_array_append_value(inputs, a_exp) == 0 &&
      mlx_vector_array_append_value(inputs, qk_scale) == 0 &&
      mlx_vector_array_append_value(inputs, prior_state) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_dtype(
          config, "InT", mlx_array_dtype(packed)) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk",
                                                        key_heads) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk",
                                                        key_dim) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv",
                                                        value_heads) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv",
                                                        value_dim) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "PackedDim",
                                                        packed_dim) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "SeqT",
                                                        tokens) == 0 &&
      mlx_fast_metal_kernel_config_add_template_arg_int(config, "DvTile",
                                                        value_tile) == 0 &&
      mlx_fast_metal_kernel_config_add_output_arg(
          config, output_shape, 4, mlx_array_dtype(packed)) == 0 &&
      mlx_fast_metal_kernel_config_add_output_arg(config, state_shape, 4,
                                                  MLX_FLOAT32) == 0 &&
      mlx_fast_metal_kernel_config_set_grid(
          config, 32, (value_dim / value_tile) * 4,
          batch * value_heads) == 0 &&
      mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1) == 0 &&
      mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, stream) ==
          0 &&
      mlx_vector_array_get(output, outputs, 0) == 0 &&
      mlx_vector_array_get(next_state, outputs, 1) == 0) {
    success = true;
  } else {
    set_error(error, error_capacity,
              "MLX failed to construct the gated-delta decode kernel");
  }

  mlx_array_free(qk_scale);
  mlx_vector_array_free(outputs);
  mlx_vector_array_free(inputs);
  mlx_fast_metal_kernel_config_free(config);
  mlx_fast_metal_kernel_free(kernel);
  mlx_vector_string_free(output_names);
  mlx_vector_string_free(input_names);
  return success;
}

} // namespace

extern "C" int generalist_mlx_initialize(const char *metallib_path,
                                           char *error,
                                           size_t error_capacity) {
  if (metallib_path == nullptr || metallib_path[0] == '\0') {
    return fail(error, error_capacity, "metallib path cannot be empty");
  }
  try {
    install_nonterminating_mlx_error_handler();
    mlx::core::metal::set_metallib_path(std::string(metallib_path));
    bool available = false;
    if (mlx_metal_is_available(&available) != 0 || !available) {
      return fail(error, error_capacity, "MLX did not find a Metal device");
    }
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    return 1;
  }
}

extern "C" int generalist_mlx_probe_tensor(
    const char *safetensors_path, const char *tensor_name,
    generalist_mlx_tensor_probe_result *result, char *error,
    size_t error_capacity) {
  if (safetensors_path == nullptr || tensor_name == nullptr || result == nullptr) {
    return fail(error, error_capacity, "tensor probe received a null argument");
  }

  mlx_stream cpu_stream = mlx_default_cpu_stream_new();
  mlx_map_string_to_array arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
  mlx_array value = mlx_array_new();
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_array as_float = mlx_array_new();
  mlx_array sum = mlx_array_new();
  int outcome = 1;

  if (mlx_load_safetensors(&arrays, &metadata, safetensors_path, cpu_stream) != 0) {
    set_error(error, error_capacity, "MLX failed to load the safetensors blob");
    goto cleanup;
  }
  if (mlx_map_string_to_array_get(&value, arrays, tensor_name) != 0 ||
      value.ctx == nullptr) {
    set_error(error, error_capacity, "requested tensor was not present in the blob");
    goto cleanup;
  }

  result->dtype = static_cast<int>(mlx_array_dtype(value));
  result->ndim = mlx_array_ndim(value);
  if (result->ndim > GENERALIST_MLX_MAX_TENSOR_DIMS) {
    set_error(error, error_capacity, "tensor rank exceeds the probe ABI limit");
    goto cleanup;
  }
  for (size_t index = 0; index < result->ndim; ++index) {
    result->shape[index] = mlx_array_dim(value, static_cast<int>(index));
  }
  result->elements = mlx_array_size(value);
  result->bytes = mlx_array_nbytes(value);
  if (result->bytes > maximum_probe_bytes) {
    set_error(error, error_capacity, "tensor exceeds the 16 MiB checkpoint limit");
    goto cleanup;
  }
  if (mlx_astype(&as_float, value, MLX_FLOAT32, gpu_stream) != 0 ||
      mlx_sum(&sum, as_float, false, gpu_stream) != 0 ||
      mlx_array_eval(sum) != 0 ||
      mlx_array_item_float32(&result->gpu_sum, sum) != 0) {
    set_error(error, error_capacity, "MLX failed to execute the Metal checkpoint");
    goto cleanup;
  }
  outcome = 0;

cleanup:
  mlx_array_free(sum);
  mlx_array_free(as_float);
  mlx_stream_free(gpu_stream);
  mlx_array_free(value);
  mlx_map_string_to_string_free(metadata);
  mlx_map_string_to_array_free(arrays);
  mlx_stream_free(cpu_stream);
  return outcome;
}

extern "C" int generalist_mlx_rms_norm_checkpoint(
    const char *safetensors_path, const char *weight_name, size_t width,
    float epsilon, int shift_weight, generalist_mlx_rms_norm_result *result,
    char *error, size_t error_capacity) {
  if (safetensors_path == nullptr || weight_name == nullptr || result == nullptr ||
      width == 0 || !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return fail(error, error_capacity,
                "RMSNorm checkpoint received an invalid argument");
  }

  *result = {};
  std::vector<float> input_values(width);
  const int input_shape[3] = {1, 1, static_cast<int>(width)};
  mlx_stream cpu_stream = mlx_default_cpu_stream_new();
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_map_string_to_array arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
  mlx_array weight = mlx_array_new();
  mlx_array one_f32 = mlx_array_new_float32(1.0F);
  mlx_array one = mlx_array_new();
  mlx_array effective_weight = mlx_array_new();
  mlx_array input_f32 = mlx_array_new();
  mlx_array input_bf16 = mlx_array_new();
  mlx_array output = mlx_array_new();
  mlx_array input_roundtrip = mlx_array_new();
  mlx_array weight_f32 = mlx_array_new();
  mlx_array output_f32 = mlx_array_new();
  const float *input_data = nullptr;
  const float *weight_data = nullptr;
  const float *output_data = nullptr;
  double mean_square = 0.0;
  double output_sum = 0.0;
  double output_squared_sum = 0.0;
  double max_abs_reference_error = 0.0;
  uint64_t fingerprint = 1469598103934665603ULL;
  int outcome = 1;

  if (width > static_cast<size_t>(INT32_MAX)) {
    set_error(error, error_capacity, "RMSNorm width exceeds the MLX shape limit");
    goto cleanup;
  }
  for (size_t index = 0; index < width; ++index) {
    input_values[index] = checkpoint_input(index);
  }
  if (mlx_load_safetensors(&arrays, &metadata, safetensors_path, cpu_stream) != 0) {
    set_error(error, error_capacity, "MLX failed to load the RMSNorm weight blob");
    goto cleanup;
  }
  if (mlx_map_string_to_array_get(&weight, arrays, weight_name) != 0 ||
      weight.ctx == nullptr || mlx_array_ndim(weight) != 1 ||
      mlx_array_size(weight) != width) {
    set_error(error, error_capacity,
              "RMSNorm weight is missing or has the wrong shape");
    goto cleanup;
  }

  if (shift_weight != 0) {
    if (mlx_astype(&one, one_f32, mlx_array_dtype(weight), gpu_stream) != 0 ||
        mlx_add(&effective_weight, weight, one, gpu_stream) != 0) {
      set_error(error, error_capacity, "MLX failed to shift the RMSNorm weight");
      goto cleanup;
    }
  } else if (mlx_array_set(&effective_weight, weight) != 0) {
    set_error(error, error_capacity, "MLX failed to retain the RMSNorm weight");
    goto cleanup;
  }

  input_f32 = mlx_array_new_data(input_values.data(), input_shape, 3, MLX_FLOAT32);
  if (input_f32.ctx == nullptr ||
      mlx_astype(&input_bf16, input_f32, MLX_BFLOAT16, gpu_stream) != 0 ||
      mlx_fast_rms_norm(&output, input_bf16, effective_weight, epsilon,
                        gpu_stream) != 0 ||
      mlx_astype(&input_roundtrip, input_bf16, MLX_FLOAT32, gpu_stream) != 0 ||
      mlx_astype(&weight_f32, effective_weight, MLX_FLOAT32, gpu_stream) != 0 ||
      mlx_astype(&output_f32, output, MLX_FLOAT32, gpu_stream) != 0 ||
      mlx_array_eval(input_roundtrip) != 0 || mlx_array_eval(weight_f32) != 0 ||
      mlx_array_eval(output_f32) != 0) {
    set_error(error, error_capacity, "MLX failed to execute the RMSNorm checkpoint");
    goto cleanup;
  }

  input_data = mlx_array_data_float32(input_roundtrip);
  weight_data = mlx_array_data_float32(weight_f32);
  output_data = mlx_array_data_float32(output_f32);
  if (input_data == nullptr || weight_data == nullptr || output_data == nullptr) {
    set_error(error, error_capacity,
              "MLX did not expose evaluated RMSNorm checkpoint data");
    goto cleanup;
  }

  for (size_t index = 0; index < width; ++index) {
    const double input = input_data[index];
    mean_square += input * input;
  }
  mean_square /= static_cast<double>(width);
  for (size_t index = 0; index < width; ++index) {
    const double output_value = output_data[index];
    const double reference =
        static_cast<double>(input_data[index]) /
        std::sqrt(mean_square + static_cast<double>(epsilon)) *
        static_cast<double>(weight_data[index]);
    output_sum += output_value;
    output_squared_sum += output_value * output_value;
    max_abs_reference_error =
        std::max(max_abs_reference_error, std::abs(output_value - reference));
    fingerprint = fingerprint_float(fingerprint, output_data[index]);
  }

  result->input_dtype = static_cast<int>(mlx_array_dtype(input_bf16));
  result->weight_dtype = static_cast<int>(mlx_array_dtype(effective_weight));
  result->output_dtype = static_cast<int>(mlx_array_dtype(output));
  result->width = width;
  result->output_sum = output_sum;
  result->output_squared_sum = output_squared_sum;
  result->output_first = output_data[0];
  result->output_middle = output_data[width / 2U];
  result->output_last = output_data[width - 1U];
  result->max_abs_reference_error = max_abs_reference_error;
  result->output_fingerprint = fingerprint;
  outcome = 0;

cleanup:
  mlx_array_free(output_f32);
  mlx_array_free(weight_f32);
  mlx_array_free(input_roundtrip);
  mlx_array_free(output);
  mlx_array_free(input_bf16);
  mlx_array_free(input_f32);
  mlx_array_free(effective_weight);
  mlx_array_free(one);
  mlx_array_free(one_f32);
  mlx_array_free(weight);
  mlx_map_string_to_string_free(metadata);
  mlx_map_string_to_array_free(arrays);
  mlx_stream_free(gpu_stream);
  mlx_stream_free(cpu_stream);
  return outcome;
}

extern "C" int generalist_mlx_embedding_checkpoint(
    const char *embedding_path, const char *embedding_name,
    const char *norm_path, const char *norm_name, int token_id, size_t width,
    float epsilon, int shift_norm_weight,
    generalist_mlx_embedding_result *result, char *error,
    size_t error_capacity) {
  if (embedding_path == nullptr || embedding_name == nullptr ||
      norm_path == nullptr || norm_name == nullptr || result == nullptr ||
      token_id < 0 || width == 0 || width > static_cast<size_t>(INT32_MAX) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return fail(error, error_capacity,
                "embedding checkpoint received an invalid argument");
  }

  *result = {};
  set_error(error, error_capacity, "");
  const int index_shape[2] = {1, 1};
  mlx_stream cpu_stream = mlx_default_cpu_stream_new();
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_map_string_to_array embedding_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string embedding_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array norm_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string norm_metadata = mlx_map_string_to_string_new();
  mlx_array embedding_weight = mlx_array_new();
  mlx_array norm_weight = mlx_array_new();
  mlx_array one_f32 = mlx_array_new_float32(1.0F);
  mlx_array one = mlx_array_new();
  mlx_array effective_norm_weight = mlx_array_new();
  mlx_array indices = mlx_array_new();
  mlx_array embedding = mlx_array_new();
  mlx_array normalized = mlx_array_new();
  int outcome = 1;

  if (mlx_load_safetensors(&embedding_arrays, &embedding_metadata,
                           embedding_path, cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&embedding_weight, embedding_arrays,
                                  embedding_name) != 0 ||
      embedding_weight.ctx == nullptr || mlx_array_ndim(embedding_weight) != 2 ||
      mlx_array_dim(embedding_weight, 1) != static_cast<int>(width) ||
      token_id >= mlx_array_dim(embedding_weight, 0)) {
    set_error(error, error_capacity,
              "embedding weight is missing or has the wrong shape");
    goto cleanup;
  }
  if (mlx_load_safetensors(&norm_arrays, &norm_metadata, norm_path,
                           cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&norm_weight, norm_arrays, norm_name) != 0 ||
      norm_weight.ctx == nullptr || mlx_array_ndim(norm_weight) != 1 ||
      mlx_array_size(norm_weight) != width) {
    set_error(error, error_capacity,
              "normalization weight is missing or has the wrong shape");
    goto cleanup;
  }

  if (shift_norm_weight != 0) {
    if (mlx_astype(&one, one_f32, mlx_array_dtype(norm_weight), gpu_stream) != 0 ||
        mlx_add(&effective_norm_weight, norm_weight, one, gpu_stream) != 0) {
      set_error(error, error_capacity,
                "MLX failed to shift the normalization weight");
      goto cleanup;
    }
  } else if (mlx_array_set(&effective_norm_weight, norm_weight) != 0) {
    set_error(error, error_capacity,
              "MLX failed to retain the normalization weight");
    goto cleanup;
  }

  indices = mlx_array_new_data(&token_id, index_shape, 2, MLX_INT32);
  if (indices.ctx == nullptr ||
      mlx_take_axis(&embedding, embedding_weight, indices, 0, gpu_stream) != 0 ||
      mlx_fast_rms_norm(&normalized, embedding, effective_norm_weight, epsilon,
                        gpu_stream) != 0 ||
      !summarize_array(embedding, gpu_stream, &result->embedding, error,
                       error_capacity) ||
      !summarize_array(normalized, gpu_stream, &result->normalized, error,
                       error_capacity)) {
    if (error != nullptr && error[0] == '\0') {
      set_error(error, error_capacity,
                "MLX failed to execute the embedding checkpoint");
    }
    goto cleanup;
  }

  result->token_id = token_id;
  result->width = width;
  outcome = 0;

cleanup:
  mlx_array_free(normalized);
  mlx_array_free(embedding);
  mlx_array_free(indices);
  mlx_array_free(effective_norm_weight);
  mlx_array_free(one);
  mlx_array_free(one_f32);
  mlx_array_free(norm_weight);
  mlx_array_free(embedding_weight);
  mlx_map_string_to_string_free(norm_metadata);
  mlx_map_string_to_array_free(norm_arrays);
  mlx_map_string_to_string_free(embedding_metadata);
  mlx_map_string_to_array_free(embedding_arrays);
  mlx_stream_free(gpu_stream);
  mlx_stream_free(cpu_stream);
  return outcome;
}

extern "C" int generalist_mlx_nvfp4_projection_checkpoint(
    const char *embedding_path, const char *embedding_name,
    const char *norm_path, const char *norm_name,
    const char *projection_path, const char *weight_name,
    const char *scale_name, const char *global_scale_name, int token_id,
    size_t input_width, size_t output_width, float epsilon,
    int shift_norm_weight, generalist_mlx_nvfp4_projection_result *result,
    char *error, size_t error_capacity) {
  constexpr int group_size_value = 16;
  constexpr int bits_value = 4;
  const mlx_optional_int group_size = {group_size_value, true};
  const mlx_optional_int bits = {bits_value, true};
  if (embedding_path == nullptr || embedding_name == nullptr ||
      norm_path == nullptr || norm_name == nullptr ||
      projection_path == nullptr || weight_name == nullptr ||
      scale_name == nullptr || global_scale_name == nullptr ||
      result == nullptr || token_id < 0 || input_width == 0 ||
      output_width == 0 || input_width > static_cast<size_t>(INT32_MAX) ||
      output_width > static_cast<size_t>(INT32_MAX) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return fail(error, error_capacity,
                "NVFP4 projection checkpoint received an invalid argument");
  }

  *result = {};
  set_error(error, error_capacity, "");
  const int index_shape[2] = {1, 1};
  const size_t packed_input_width = input_width * bits_value / 32U;
  const size_t scale_width = input_width / group_size_value;
  mlx_stream cpu_stream = mlx_default_cpu_stream_new();
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_map_string_to_array embedding_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string embedding_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array norm_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string norm_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array projection_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string projection_metadata = mlx_map_string_to_string_new();
  mlx_array embedding_weight = mlx_array_new();
  mlx_array norm_weight = mlx_array_new();
  mlx_array projection_weight = mlx_array_new();
  mlx_array projection_scale = mlx_array_new();
  mlx_array projection_global_scale = mlx_array_new();
  mlx_array no_bias = mlx_array_new();
  mlx_array one_f32 = mlx_array_new_float32(1.0F);
  mlx_array one = mlx_array_new();
  mlx_array effective_norm_weight = mlx_array_new();
  mlx_array indices = mlx_array_new();
  mlx_array embedding = mlx_array_new();
  mlx_array normalized = mlx_array_new();
  mlx_array raw_projection = mlx_array_new();
  mlx_array scaled_f32 = mlx_array_new();
  mlx_array scaled_projection = mlx_array_new();
  mlx_array encoded_global_scale = mlx_array_new();
  mlx_array fused_projection = mlx_array_new();
  int outcome = 1;

  if (input_width % group_size_value != 0 ||
      input_width % (32U / bits_value) != 0) {
    set_error(error, error_capacity,
              "NVFP4 projection input width is not pack aligned");
    goto cleanup;
  }
  if (mlx_load_safetensors(&embedding_arrays, &embedding_metadata,
                           embedding_path, cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&embedding_weight, embedding_arrays,
                                  embedding_name) != 0 ||
      embedding_weight.ctx == nullptr || mlx_array_ndim(embedding_weight) != 2 ||
      mlx_array_dim(embedding_weight, 1) != static_cast<int>(input_width) ||
      token_id >= mlx_array_dim(embedding_weight, 0)) {
    set_error(error, error_capacity,
              "embedding weight is missing or has the wrong shape");
    goto cleanup;
  }
  if (mlx_load_safetensors(&norm_arrays, &norm_metadata, norm_path,
                           cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&norm_weight, norm_arrays, norm_name) != 0 ||
      norm_weight.ctx == nullptr || mlx_array_ndim(norm_weight) != 1 ||
      mlx_array_size(norm_weight) != input_width) {
    set_error(error, error_capacity,
              "normalization weight is missing or has the wrong shape");
    goto cleanup;
  }
  if (mlx_load_safetensors(&projection_arrays, &projection_metadata,
                           projection_path, cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&projection_weight, projection_arrays,
                                  weight_name) != 0 ||
      mlx_map_string_to_array_get(&projection_scale, projection_arrays,
                                  scale_name) != 0 ||
      mlx_map_string_to_array_get(&projection_global_scale, projection_arrays,
                                  global_scale_name) != 0 ||
      projection_weight.ctx == nullptr || projection_scale.ctx == nullptr ||
      projection_global_scale.ctx == nullptr ||
      mlx_array_ndim(projection_weight) != 2 ||
      mlx_array_dim(projection_weight, 0) != static_cast<int>(output_width) ||
      mlx_array_dim(projection_weight, 1) !=
          static_cast<int>(packed_input_width) ||
      mlx_array_ndim(projection_scale) != 2 ||
      mlx_array_dim(projection_scale, 0) != static_cast<int>(output_width) ||
      mlx_array_dim(projection_scale, 1) != static_cast<int>(scale_width) ||
      mlx_array_size(projection_global_scale) != 1 ||
      mlx_array_dtype(projection_weight) != MLX_UINT32 ||
      mlx_array_dtype(projection_scale) != MLX_UINT8 ||
      mlx_array_dtype(projection_global_scale) != MLX_FLOAT32) {
    set_error(error, error_capacity,
              "NVFP4 projection arrays are missing or violate the packed layout");
    goto cleanup;
  }

  if (shift_norm_weight != 0) {
    if (mlx_astype(&one, one_f32, mlx_array_dtype(norm_weight), gpu_stream) != 0 ||
        mlx_add(&effective_norm_weight, norm_weight, one, gpu_stream) != 0) {
      set_error(error, error_capacity,
                "MLX failed to shift the normalization weight");
      goto cleanup;
    }
  } else if (mlx_array_set(&effective_norm_weight, norm_weight) != 0) {
    set_error(error, error_capacity,
              "MLX failed to retain the normalization weight");
    goto cleanup;
  }

  indices = mlx_array_new_data(&token_id, index_shape, 2, MLX_INT32);
  if (indices.ctx == nullptr ||
      mlx_take_axis(&embedding, embedding_weight, indices, 0, gpu_stream) != 0 ||
      mlx_fast_rms_norm(&normalized, embedding, effective_norm_weight, epsilon,
                        gpu_stream) != 0 ||
      mlx_quantized_matmul(&raw_projection, normalized, projection_weight,
                           projection_scale, no_bias, true, group_size, bits,
                           "nvfp4", gpu_stream) != 0 ||
      mlx_multiply(&scaled_f32, raw_projection, projection_global_scale,
                   gpu_stream) != 0 ||
      mlx_astype(&scaled_projection, scaled_f32,
                 mlx_array_dtype(raw_projection), gpu_stream) != 0 ||
      mlx_array_eval(projection_global_scale) != 0 ||
      mlx_array_item_float32(&result->global_scale,
                             projection_global_scale) != 0 ||
      mlx_array_set_float32(
          &encoded_global_scale,
          result->global_scale * (448.0F * 6.0F)) != 0 ||
      !make_fused_nvfp4_qmv(&fused_projection, normalized,
                            projection_weight, projection_scale,
                            encoded_global_scale, gpu_stream, error,
                            error_capacity) ||
      !summarize_array(normalized, gpu_stream, &result->normalized_input,
                       error, error_capacity) ||
      !summarize_array(raw_projection, gpu_stream, &result->raw_projection,
                       error, error_capacity) ||
      !summarize_array(scaled_projection, gpu_stream, &result->output, error,
                       error_capacity) ||
      !summarize_array(fused_projection, gpu_stream, &result->fused_output,
                       error, error_capacity)) {
    if (error != nullptr && error[0] == '\0') {
      set_error(error, error_capacity,
                "MLX failed to execute the NVFP4 projection checkpoint");
    }
    goto cleanup;
  }

  result->token_id = token_id;
  result->input_width = input_width;
  result->output_width = output_width;
  result->group_size = group_size_value;
  result->bits = bits_value;
  result->weight_dtype = static_cast<int>(mlx_array_dtype(projection_weight));
  result->scale_dtype = static_cast<int>(mlx_array_dtype(projection_scale));
  result->global_scale_dtype =
      static_cast<int>(mlx_array_dtype(projection_global_scale));
  result->encoded_global_scale = result->global_scale * (448.0F * 6.0F);
  outcome = 0;

cleanup:
  mlx_array_free(fused_projection);
  mlx_array_free(encoded_global_scale);
  mlx_array_free(scaled_projection);
  mlx_array_free(scaled_f32);
  mlx_array_free(raw_projection);
  mlx_array_free(normalized);
  mlx_array_free(embedding);
  mlx_array_free(indices);
  mlx_array_free(effective_norm_weight);
  mlx_array_free(one);
  mlx_array_free(one_f32);
  mlx_array_free(no_bias);
  mlx_array_free(projection_global_scale);
  mlx_array_free(projection_scale);
  mlx_array_free(projection_weight);
  mlx_array_free(norm_weight);
  mlx_array_free(embedding_weight);
  mlx_map_string_to_string_free(projection_metadata);
  mlx_map_string_to_array_free(projection_arrays);
  mlx_map_string_to_string_free(norm_metadata);
  mlx_map_string_to_array_free(norm_arrays);
  mlx_map_string_to_string_free(embedding_metadata);
  mlx_map_string_to_array_free(embedding_arrays);
  mlx_stream_free(gpu_stream);
  mlx_stream_free(cpu_stream);
  return outcome;
}

extern "C" int generalist_mlx_layer0_projection_checkpoint(
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
    size_t error_capacity) {
  constexpr int group_size_value = 16;
  constexpr int bits_value = 4;
  const mlx_optional_int group_size = {group_size_value, true};
  const mlx_optional_int bits = {bits_value, true};
  const bool any_gdn_argument =
      conv_path != nullptr || conv_name != nullptr || dt_bias_path != nullptr ||
      dt_bias_name != nullptr || a_log_path != nullptr || a_log_name != nullptr;
  const bool all_gdn_arguments =
      conv_path != nullptr && conv_name != nullptr && dt_bias_path != nullptr &&
      dt_bias_name != nullptr && a_log_path != nullptr && a_log_name != nullptr;
  const bool any_full_layer_argument =
      gdn_norm_path != nullptr || gdn_norm_name != nullptr ||
      out_proj_path != nullptr || out_proj_name != nullptr ||
      post_norm_path != nullptr || post_norm_name != nullptr ||
      mlp_gate_path != nullptr || mlp_gate_name != nullptr ||
      mlp_up_path != nullptr || mlp_up_name != nullptr ||
      mlp_down_path != nullptr || mlp_down_name != nullptr;
  const bool all_full_layer_arguments =
      gdn_norm_path != nullptr && gdn_norm_name != nullptr &&
      out_proj_path != nullptr && out_proj_name != nullptr &&
      post_norm_path != nullptr && post_norm_name != nullptr &&
      mlp_gate_path != nullptr && mlp_gate_name != nullptr &&
      mlp_up_path != nullptr && mlp_up_name != nullptr &&
      mlp_down_path != nullptr && mlp_down_name != nullptr;
  if (embedding_path == nullptr || embedding_name == nullptr ||
      norm_path == nullptr || norm_name == nullptr || qkv_path == nullptr ||
      qkv_name == nullptr || z_path == nullptr || z_name == nullptr ||
      beta_path == nullptr || beta_name == nullptr || alpha_path == nullptr ||
      alpha_name == nullptr || result == nullptr || token_id < 0 ||
      input_width == 0 || qkv_width == 0 || z_width == 0 || ba_width == 0 ||
      ba_width > static_cast<size_t>(INT32_MAX) ||
      input_width > static_cast<size_t>(INT32_MAX) ||
      qkv_width > static_cast<size_t>(INT32_MAX) ||
      z_width > static_cast<size_t>(INT32_MAX) ||
      input_width % group_size_value != 0 ||
      input_width % (32U / bits_value) != 0 ||
      (any_gdn_argument && !all_gdn_arguments) ||
      (any_full_layer_argument && !all_full_layer_arguments) ||
      (all_full_layer_arguments && (!all_gdn_arguments || mlp_width == 0)) ||
      !std::isfinite(epsilon) || epsilon <= 0.0F) {
    return fail(error, error_capacity,
                "layer-zero projection checkpoint received an invalid argument");
  }

  *result = {};
  set_error(error, error_capacity, "");
  const size_t packed_input_width = input_width * bits_value / 32U;
  const size_t scale_width = input_width / group_size_value;
  const size_t half_ba_width = ba_width / 2U;
  size_t conv_kernel_width = 0;
  size_t key_heads = 0;
  size_t key_dim = 0;
  size_t value_heads = 0;
  size_t value_dim = 0;
  const int index_shape[2] = {1, 1};
  const int qkv_global_shape[1] = {static_cast<int>(qkv_width)};
  const int z_global_shape[1] = {static_cast<int>(z_width)};
  mlx_stream cpu_stream = mlx_default_cpu_stream_new();
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_map_string_to_array embedding_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string embedding_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array norm_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string norm_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array qkv_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string qkv_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array z_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string z_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array beta_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string beta_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array alpha_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string alpha_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array conv_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string conv_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array dt_bias_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string dt_bias_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array a_log_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string a_log_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array gdn_norm_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string gdn_norm_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array out_proj_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string out_proj_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array post_norm_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string post_norm_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array mlp_gate_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string mlp_gate_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array mlp_up_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string mlp_up_metadata = mlx_map_string_to_string_new();
  mlx_map_string_to_array mlp_down_arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string mlp_down_metadata = mlx_map_string_to_string_new();
  mlx_array embedding_weight = mlx_array_new();
  mlx_array norm_weight = mlx_array_new();
  mlx_array qkv_weight = mlx_array_new();
  mlx_array qkv_scale = mlx_array_new();
  mlx_array qkv_global_scale = mlx_array_new();
  mlx_array z_weight = mlx_array_new();
  mlx_array z_scale = mlx_array_new();
  mlx_array z_global_scale = mlx_array_new();
  mlx_array beta_weight = mlx_array_new();
  mlx_array alpha_weight = mlx_array_new();
  mlx_array conv_weight = mlx_array_new();
  mlx_array dt_bias = mlx_array_new();
  mlx_array a_log = mlx_array_new();
  mlx_array gdn_norm_weight = mlx_array_new();
  mlx_array out_proj_weight = mlx_array_new();
  mlx_array out_proj_scale = mlx_array_new();
  mlx_array out_proj_global_scale = mlx_array_new();
  mlx_array post_norm_weight = mlx_array_new();
  mlx_array mlp_gate_weight = mlx_array_new();
  mlx_array mlp_gate_scale = mlx_array_new();
  mlx_array mlp_gate_global_scale = mlx_array_new();
  mlx_array mlp_up_weight = mlx_array_new();
  mlx_array mlp_up_scale = mlx_array_new();
  mlx_array mlp_up_global_scale = mlx_array_new();
  mlx_array mlp_down_weight = mlx_array_new();
  mlx_array mlp_down_scale = mlx_array_new();
  mlx_array mlp_down_global_scale = mlx_array_new();
  mlx_array one_f32 = mlx_array_new_float32(1.0F);
  mlx_array one = mlx_array_new();
  mlx_array effective_norm_weight = mlx_array_new();
  mlx_array indices = mlx_array_new();
  mlx_array embedding = mlx_array_new();
  mlx_array normalized = mlx_array_new();
  mlx_array packed_weight = mlx_array_new();
  mlx_array packed_scale = mlx_array_new();
  mlx_array expanded_qkv_global = mlx_array_new();
  mlx_array expanded_z_global = mlx_array_new();
  mlx_array packed_global = mlx_array_new();
  mlx_array raw_qkvz = mlx_array_new();
  mlx_array scaled_qkvz_f32 = mlx_array_new();
  mlx_array mixed_qkvz = mlx_array_new();
  mlx_array packed_ba_weight = mlx_array_new();
  mlx_array packed_ba_transposed = mlx_array_new();
  mlx_array mixed_ba = mlx_array_new();
  mlx_array qkv_input = mlx_array_new();
  mlx_array conv_weight_2d = mlx_array_new();
  mlx_array prior_conv_state = mlx_array_new();
  mlx_array conv_window = mlx_array_new();
  mlx_array conv_out = mlx_array_new();
  mlx_array conv_state = mlx_array_new();
  mlx_array a_log_f32 = mlx_array_new();
  mlx_array a_exp = mlx_array_new();
  mlx_array prior_delta_state = mlx_array_new();
  mlx_array gdn_out = mlx_array_new();
  mlx_array delta_state = mlx_array_new();
  mlx_array z_gate = mlx_array_new();
  mlx_array z_gate_shaped = mlx_array_new();
  mlx_array gdn_normalized = mlx_array_new();
  mlx_array gdn_normalized_f32 = mlx_array_new();
  mlx_array z_gate_f32 = mlx_array_new();
  mlx_array z_gate_sigmoid = mlx_array_new();
  mlx_array z_gate_silu = mlx_array_new();
  mlx_array gated_f32 = mlx_array_new();
  mlx_array gated = mlx_array_new();
  mlx_array gated_flat = mlx_array_new();
  mlx_array attention_out = mlx_array_new();
  mlx_array after_attention = mlx_array_new();
  mlx_array post_norm_one = mlx_array_new();
  mlx_array effective_post_norm_weight = mlx_array_new();
  mlx_array post_normalized = mlx_array_new();
  mlx_array mlp_gate_out = mlx_array_new();
  mlx_array mlp_up_out = mlx_array_new();
  mlx_array mlp_gate_sigmoid = mlx_array_new();
  mlx_array mlp_gate_silu = mlx_array_new();
  mlx_array swiglu = mlx_array_new();
  mlx_array mlp_out = mlx_array_new();
  mlx_array layer_out = mlx_array_new();
  mlx_array no_bias = mlx_array_new();
  mlx_vector_array qkvz_weight_values = mlx_vector_array_new();
  mlx_vector_array qkvz_scale_values = mlx_vector_array_new();
  mlx_vector_array global_values = mlx_vector_array_new();
  mlx_vector_array ba_values = mlx_vector_array_new();
  mlx_vector_array conv_window_values = mlx_vector_array_new();
  int outcome = 1;

  const std::string qkv_scale_name = std::string(qkv_name) + ".scale";
  const std::string qkv_global_name = std::string(qkv_name) + ".global_scale";
  const std::string z_scale_name = std::string(z_name) + ".scale";
  const std::string z_global_name = std::string(z_name) + ".global_scale";
  if (ba_width % 2U != 0) {
    set_error(error, error_capacity, "layer-zero BA width must be even");
    goto cleanup;
  }

  if (mlx_load_safetensors(&embedding_arrays, &embedding_metadata,
                           embedding_path, cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&embedding_weight, embedding_arrays,
                                  embedding_name) != 0 ||
      embedding_weight.ctx == nullptr || mlx_array_ndim(embedding_weight) != 2 ||
      mlx_array_dim(embedding_weight, 1) != static_cast<int>(input_width) ||
      token_id >= mlx_array_dim(embedding_weight, 0)) {
    set_error(error, error_capacity,
              "embedding weight is missing or has the wrong shape");
    goto cleanup;
  }
  if (mlx_load_safetensors(&norm_arrays, &norm_metadata, norm_path,
                           cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&norm_weight, norm_arrays, norm_name) != 0 ||
      norm_weight.ctx == nullptr || mlx_array_ndim(norm_weight) != 1 ||
      mlx_array_size(norm_weight) != input_width) {
    set_error(error, error_capacity,
              "normalization weight is missing or has the wrong shape");
    goto cleanup;
  }
  if (mlx_load_safetensors(&qkv_arrays, &qkv_metadata, qkv_path,
                           cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&qkv_weight, qkv_arrays, qkv_name) != 0 ||
      mlx_map_string_to_array_get(&qkv_scale, qkv_arrays,
                                  qkv_scale_name.c_str()) != 0 ||
      mlx_map_string_to_array_get(&qkv_global_scale, qkv_arrays,
                                  qkv_global_name.c_str()) != 0 ||
      qkv_weight.ctx == nullptr || qkv_scale.ctx == nullptr ||
      qkv_global_scale.ctx == nullptr || mlx_array_ndim(qkv_weight) != 2 ||
      mlx_array_dim(qkv_weight, 0) != static_cast<int>(qkv_width) ||
      mlx_array_dim(qkv_weight, 1) != static_cast<int>(packed_input_width) ||
      mlx_array_ndim(qkv_scale) != 2 ||
      mlx_array_dim(qkv_scale, 0) != static_cast<int>(qkv_width) ||
      mlx_array_dim(qkv_scale, 1) != static_cast<int>(scale_width) ||
      mlx_array_size(qkv_global_scale) != 1) {
    set_error(error, error_capacity,
              "QKV projection arrays violate the expected NVFP4 layout");
    goto cleanup;
  }
  if (mlx_load_safetensors(&z_arrays, &z_metadata, z_path, cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&z_weight, z_arrays, z_name) != 0 ||
      mlx_map_string_to_array_get(&z_scale, z_arrays,
                                  z_scale_name.c_str()) != 0 ||
      mlx_map_string_to_array_get(&z_global_scale, z_arrays,
                                  z_global_name.c_str()) != 0 ||
      z_weight.ctx == nullptr || z_scale.ctx == nullptr ||
      z_global_scale.ctx == nullptr || mlx_array_ndim(z_weight) != 2 ||
      mlx_array_dim(z_weight, 0) != static_cast<int>(z_width) ||
      mlx_array_dim(z_weight, 1) != static_cast<int>(packed_input_width) ||
      mlx_array_ndim(z_scale) != 2 ||
      mlx_array_dim(z_scale, 0) != static_cast<int>(z_width) ||
      mlx_array_dim(z_scale, 1) != static_cast<int>(scale_width) ||
      mlx_array_size(z_global_scale) != 1) {
    set_error(error, error_capacity,
              "Z projection arrays violate the expected NVFP4 layout");
    goto cleanup;
  }
  if (mlx_load_safetensors(&beta_arrays, &beta_metadata, beta_path,
                           cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&beta_weight, beta_arrays, beta_name) != 0 ||
      beta_weight.ctx == nullptr || mlx_array_ndim(beta_weight) != 2 ||
      mlx_array_dim(beta_weight, 0) != static_cast<int>(half_ba_width) ||
      mlx_array_dim(beta_weight, 1) != static_cast<int>(input_width) ||
      mlx_load_safetensors(&alpha_arrays, &alpha_metadata, alpha_path,
                           cpu_stream) != 0 ||
      mlx_map_string_to_array_get(&alpha_weight, alpha_arrays, alpha_name) != 0 ||
      alpha_weight.ctx == nullptr || mlx_array_ndim(alpha_weight) != 2 ||
      mlx_array_dim(alpha_weight, 0) != static_cast<int>(half_ba_width) ||
      mlx_array_dim(alpha_weight, 1) != static_cast<int>(input_width)) {
    set_error(error, error_capacity,
              "BA projection weights are missing or have the wrong shape");
    goto cleanup;
  }
  if (all_gdn_arguments) {
    if (mlx_load_safetensors(&conv_arrays, &conv_metadata, conv_path,
                             cpu_stream) != 0 ||
        mlx_map_string_to_array_get(&conv_weight, conv_arrays, conv_name) != 0 ||
        conv_weight.ctx == nullptr || mlx_array_ndim(conv_weight) != 3 ||
        mlx_array_dim(conv_weight, 0) != static_cast<int>(qkv_width) ||
        mlx_array_dim(conv_weight, 1) != 1 ||
        mlx_array_dim(conv_weight, 2) <= 0 ||
        mlx_load_safetensors(&dt_bias_arrays, &dt_bias_metadata, dt_bias_path,
                             cpu_stream) != 0 ||
        mlx_map_string_to_array_get(&dt_bias, dt_bias_arrays, dt_bias_name) !=
            0 ||
        dt_bias.ctx == nullptr || mlx_array_ndim(dt_bias) != 1 ||
        mlx_array_size(dt_bias) != half_ba_width ||
        mlx_load_safetensors(&a_log_arrays, &a_log_metadata, a_log_path,
                             cpu_stream) != 0 ||
        mlx_map_string_to_array_get(&a_log, a_log_arrays, a_log_name) != 0 ||
        a_log.ctx == nullptr || mlx_array_ndim(a_log) != 1 ||
        mlx_array_size(a_log) != half_ba_width) {
      set_error(error, error_capacity,
                "GDN state tensors are missing or have the wrong shape");
      goto cleanup;
    }
    conv_kernel_width =
        static_cast<size_t>(mlx_array_dim(conv_weight, 2));
    value_heads = half_ba_width;
    if (value_heads == 0 || z_width % value_heads != 0 ||
        qkv_width < z_width || (qkv_width - z_width) % 2U != 0) {
      set_error(error, error_capacity,
                "GDN projection widths do not encode valid head geometry");
      goto cleanup;
    }
    value_dim = z_width / value_heads;
    key_dim = value_dim;
    const size_t key_width = (qkv_width - z_width) / 2U;
    if (key_dim == 0 || key_width % key_dim != 0) {
      set_error(error, error_capacity,
                "GDN key width does not match the value-head dimension");
      goto cleanup;
    }
    key_heads = key_width / key_dim;
    if (conv_kernel_width < 1U || conv_kernel_width > 16U || key_heads == 0 ||
        key_heads > static_cast<size_t>(INT32_MAX) ||
        key_dim > static_cast<size_t>(INT32_MAX) ||
        value_heads > static_cast<size_t>(INT32_MAX) ||
        value_dim > static_cast<size_t>(INT32_MAX) ||
        value_heads % key_heads != 0 || key_dim % 32U != 0 ||
        value_dim % 32U != 0) {
      set_error(error, error_capacity,
                "GDN head geometry is outside the decode-kernel contract");
      goto cleanup;
    }
  }
  if (all_full_layer_arguments) {
    if (mlp_width > static_cast<size_t>(INT32_MAX) ||
        z_width % group_size_value != 0 ||
        z_width % (32U / bits_value) != 0 ||
        mlp_width % group_size_value != 0 ||
        mlp_width % (32U / bits_value) != 0 ||
        mlx_load_safetensors(&gdn_norm_arrays, &gdn_norm_metadata,
                             gdn_norm_path, cpu_stream) != 0 ||
        mlx_map_string_to_array_get(&gdn_norm_weight, gdn_norm_arrays,
                                    gdn_norm_name) != 0 ||
        gdn_norm_weight.ctx == nullptr ||
        mlx_array_ndim(gdn_norm_weight) != 1 ||
        mlx_array_size(gdn_norm_weight) != value_dim ||
        !load_nvfp4_projection(
            out_proj_path, out_proj_name, z_width, input_width, cpu_stream,
            &out_proj_arrays, &out_proj_metadata, &out_proj_weight,
            &out_proj_scale, &out_proj_global_scale, error, error_capacity) ||
        mlx_load_safetensors(&post_norm_arrays, &post_norm_metadata,
                             post_norm_path, cpu_stream) != 0 ||
        mlx_map_string_to_array_get(&post_norm_weight, post_norm_arrays,
                                    post_norm_name) != 0 ||
        post_norm_weight.ctx == nullptr ||
        mlx_array_ndim(post_norm_weight) != 1 ||
        mlx_array_size(post_norm_weight) != input_width ||
        !load_nvfp4_projection(
            mlp_gate_path, mlp_gate_name, input_width, mlp_width, cpu_stream,
            &mlp_gate_arrays, &mlp_gate_metadata, &mlp_gate_weight,
            &mlp_gate_scale, &mlp_gate_global_scale, error, error_capacity) ||
        !load_nvfp4_projection(
            mlp_up_path, mlp_up_name, input_width, mlp_width, cpu_stream,
            &mlp_up_arrays, &mlp_up_metadata, &mlp_up_weight, &mlp_up_scale,
            &mlp_up_global_scale, error, error_capacity) ||
        !load_nvfp4_projection(
            mlp_down_path, mlp_down_name, mlp_width, input_width, cpu_stream,
            &mlp_down_arrays, &mlp_down_metadata, &mlp_down_weight,
            &mlp_down_scale, &mlp_down_global_scale, error, error_capacity)) {
      if (error != nullptr && error[0] == '\0') {
        set_error(error, error_capacity,
                  "full layer-zero weights are missing or have the wrong shape");
      }
      goto cleanup;
    }
  }

  if (shift_norm_weight != 0) {
    if (mlx_astype(&one, one_f32, mlx_array_dtype(norm_weight), gpu_stream) != 0 ||
        mlx_add(&effective_norm_weight, norm_weight, one, gpu_stream) != 0) {
      set_error(error, error_capacity,
                "MLX failed to shift the normalization weight");
      goto cleanup;
    }
  } else if (mlx_array_set(&effective_norm_weight, norm_weight) != 0) {
    set_error(error, error_capacity,
              "MLX failed to retain the normalization weight");
    goto cleanup;
  }

  indices = mlx_array_new_data(&token_id, index_shape, 2, MLX_INT32);
  if (indices.ctx == nullptr ||
      mlx_take_axis(&embedding, embedding_weight, indices, 0, gpu_stream) != 0 ||
      mlx_fast_rms_norm(&normalized, embedding, effective_norm_weight, epsilon,
                        gpu_stream) != 0 ||
      mlx_vector_array_append_value(qkvz_weight_values, qkv_weight) != 0 ||
      mlx_vector_array_append_value(qkvz_weight_values, z_weight) != 0 ||
      mlx_concatenate_axis(&packed_weight, qkvz_weight_values, 0,
                           gpu_stream) != 0 ||
      mlx_vector_array_append_value(qkvz_scale_values, qkv_scale) != 0 ||
      mlx_vector_array_append_value(qkvz_scale_values, z_scale) != 0 ||
      mlx_concatenate_axis(&packed_scale, qkvz_scale_values, 0,
                           gpu_stream) != 0 ||
      mlx_broadcast_to(&expanded_qkv_global, qkv_global_scale,
                       qkv_global_shape, 1, gpu_stream) != 0 ||
      mlx_broadcast_to(&expanded_z_global, z_global_scale, z_global_shape, 1,
                       gpu_stream) != 0) {
    set_error(error, error_capacity,
              "MLX failed to build the packed layer-zero projections");
    goto cleanup;
  }
  if (mlx_vector_array_append_value(global_values, expanded_qkv_global) != 0 ||
      mlx_vector_array_append_value(global_values, expanded_z_global) != 0 ||
      mlx_concatenate_axis(&packed_global, global_values, 0, gpu_stream) != 0 ||
      mlx_quantized_matmul(&raw_qkvz, normalized, packed_weight, packed_scale,
                           no_bias, true, group_size, bits, "nvfp4",
                           gpu_stream) != 0 ||
      mlx_multiply(&scaled_qkvz_f32, raw_qkvz, packed_global, gpu_stream) != 0 ||
      mlx_astype(&mixed_qkvz, scaled_qkvz_f32,
                 mlx_array_dtype(raw_qkvz), gpu_stream) != 0 ||
      mlx_vector_array_append_value(ba_values, beta_weight) != 0 ||
      mlx_vector_array_append_value(ba_values, alpha_weight) != 0 ||
      mlx_concatenate_axis(&packed_ba_weight, ba_values, 0, gpu_stream) != 0 ||
      mlx_transpose(&packed_ba_transposed, packed_ba_weight, gpu_stream) != 0 ||
      mlx_matmul(&mixed_ba, normalized, packed_ba_transposed, gpu_stream) != 0 ||
      mlx_array_eval(qkv_global_scale) != 0 ||
      mlx_array_eval(z_global_scale) != 0 ||
      mlx_array_item_float32(&result->qkv_global_scale, qkv_global_scale) != 0 ||
      mlx_array_item_float32(&result->z_global_scale, z_global_scale) != 0 ||
      !summarize_array(normalized, gpu_stream, &result->normalized_input,
                       error, error_capacity) ||
      !summarize_array(raw_qkvz, gpu_stream, &result->raw_qkvz, error,
                       error_capacity) ||
      !summarize_array(mixed_qkvz, gpu_stream, &result->mixed_qkvz, error,
                       error_capacity) ||
      !summarize_array(mixed_ba, gpu_stream, &result->mixed_ba, error,
                       error_capacity)) {
    if (error != nullptr && error[0] == '\0') {
      set_error(error, error_capacity,
                "MLX failed to execute the layer-zero projection checkpoint");
    }
    goto cleanup;
  }

  if (all_gdn_arguments) {
    const int qkv_start[3] = {0, 0, 0};
    const int qkv_stop[3] = {1, 1, static_cast<int>(qkv_width)};
    const int unit_strides[3] = {1, 1, 1};
    const int conv_weight_shape[2] = {
        static_cast<int>(qkv_width), static_cast<int>(conv_kernel_width)};
    const int prior_conv_shape[3] = {
        1, static_cast<int>(conv_kernel_width - 1U),
        static_cast<int>(qkv_width)};
    const int conv_state_start[3] = {0, 1, 0};
    const int conv_state_stop[3] = {
        1, static_cast<int>(conv_kernel_width), static_cast<int>(qkv_width)};
    const int prior_delta_shape[4] = {
        1, static_cast<int>(value_heads), static_cast<int>(value_dim),
        static_cast<int>(key_dim)};
    if (mlx_slice(&qkv_input, mixed_qkvz, qkv_start, 3, qkv_stop, 3,
                  unit_strides, 3, gpu_stream) != 0 ||
        mlx_reshape(&conv_weight_2d, conv_weight, conv_weight_shape, 2,
                    gpu_stream) != 0 ||
        mlx_zeros(&prior_conv_state, prior_conv_shape, 3,
                  mlx_array_dtype(mixed_qkvz), gpu_stream) != 0 ||
        mlx_vector_array_append_value(conv_window_values, prior_conv_state) !=
            0 ||
        mlx_vector_array_append_value(conv_window_values, qkv_input) != 0 ||
        mlx_concatenate_axis(&conv_window, conv_window_values, 1, gpu_stream) !=
            0 ||
        !make_depthwise_conv_silu(
            &conv_out, conv_window, conv_weight_2d, 1, 1,
            static_cast<int>(qkv_width), static_cast<int>(conv_kernel_width),
            gpu_stream, error, error_capacity) ||
        mlx_slice(&conv_state, conv_window, conv_state_start, 3,
                  conv_state_stop, 3, unit_strides, 3, gpu_stream) != 0 ||
        mlx_astype(&a_log_f32, a_log, MLX_FLOAT32, gpu_stream) != 0 ||
        mlx_exp(&a_exp, a_log_f32, gpu_stream) != 0 ||
        mlx_zeros(&prior_delta_state, prior_delta_shape, 4, MLX_FLOAT32,
                  gpu_stream) != 0 ||
        !make_gated_delta_decode(
            &gdn_out, &delta_state, conv_out, mixed_ba,
            dt_bias, a_exp, prior_delta_state, 1, 1,
            static_cast<int>(key_heads), static_cast<int>(key_dim),
            static_cast<int>(value_heads), static_cast<int>(value_dim),
            gpu_stream, error, error_capacity) ||
        !summarize_array(conv_out, gpu_stream, &result->conv_out, error,
                         error_capacity) ||
        !summarize_array(conv_state, gpu_stream, &result->conv_state, error,
                         error_capacity) ||
        !summarize_array(a_exp, gpu_stream, &result->a_exp, error,
                         error_capacity) ||
        !summarize_array(gdn_out, gpu_stream, &result->gdn_out, error,
                         error_capacity) ||
        !summarize_array(delta_state, gpu_stream, &result->delta_state, error,
                         error_capacity)) {
      if (error != nullptr && error[0] == '\0') {
        set_error(error, error_capacity,
                  "MLX failed to execute the layer-zero GDN checkpoint");
      }
      goto cleanup;
    }
    result->gdn_executed = 1;
    result->conv_kernel_width = conv_kernel_width;
    result->key_heads = key_heads;
    result->key_dim = key_dim;
    result->value_heads = value_heads;
    result->value_dim = value_dim;
  }

  if (all_full_layer_arguments) {
    const int z_start[3] = {0, 0, static_cast<int>(qkv_width)};
    const int z_stop[3] = {
        1, 1, static_cast<int>(qkv_width + z_width)};
    const int unit_strides[3] = {1, 1, 1};
    const int z_shape[4] = {1, 1, static_cast<int>(value_heads),
                            static_cast<int>(value_dim)};
    const int flat_z_shape[3] = {1, 1, static_cast<int>(z_width)};
    if (shift_norm_weight != 0) {
      if (mlx_astype(&post_norm_one, one_f32,
                     mlx_array_dtype(post_norm_weight), gpu_stream) != 0 ||
          mlx_add(&effective_post_norm_weight, post_norm_weight,
                  post_norm_one, gpu_stream) != 0) {
        set_error(error, error_capacity,
                  "MLX failed to shift the post-attention norm weight");
        goto cleanup;
      }
    } else if (mlx_array_set(&effective_post_norm_weight, post_norm_weight) !=
               0) {
      set_error(error, error_capacity,
                "MLX failed to retain the post-attention norm weight");
      goto cleanup;
    }
    if (mlx_slice(&z_gate, mixed_qkvz, z_start, 3, z_stop, 3, unit_strides,
                  3, gpu_stream) != 0 ||
        mlx_reshape(&z_gate_shaped, z_gate, z_shape, 4, gpu_stream) != 0 ||
        mlx_fast_rms_norm(&gdn_normalized, gdn_out, gdn_norm_weight, epsilon,
                          gpu_stream) != 0 ||
        mlx_astype(&gdn_normalized_f32, gdn_normalized, MLX_FLOAT32,
                   gpu_stream) != 0 ||
        mlx_astype(&z_gate_f32, z_gate_shaped, MLX_FLOAT32, gpu_stream) != 0 ||
        mlx_sigmoid(&z_gate_sigmoid, z_gate_f32, gpu_stream) != 0 ||
        mlx_multiply(&z_gate_silu, z_gate_f32, z_gate_sigmoid, gpu_stream) !=
            0 ||
        mlx_multiply(&gated_f32, gdn_normalized_f32, z_gate_silu,
                     gpu_stream) != 0 ||
        mlx_astype(&gated, gated_f32, mlx_array_dtype(gdn_out), gpu_stream) !=
            0 ||
        mlx_reshape(&gated_flat, gated, flat_z_shape, 3, gpu_stream) != 0 ||
        !make_nvfp4_linear_reference(
            &attention_out, gated_flat, out_proj_weight, out_proj_scale,
            out_proj_global_scale, gpu_stream, error, error_capacity) ||
        mlx_add(&after_attention, embedding, attention_out, gpu_stream) != 0 ||
        mlx_fast_rms_norm(&post_normalized, after_attention,
                          effective_post_norm_weight, epsilon, gpu_stream) !=
            0 ||
        !make_nvfp4_linear_reference(
            &mlp_gate_out, post_normalized, mlp_gate_weight, mlp_gate_scale,
            mlp_gate_global_scale, gpu_stream, error, error_capacity) ||
        !make_nvfp4_linear_reference(
            &mlp_up_out, post_normalized, mlp_up_weight, mlp_up_scale,
            mlp_up_global_scale, gpu_stream, error, error_capacity) ||
        mlx_sigmoid(&mlp_gate_sigmoid, mlp_gate_out, gpu_stream) != 0 ||
        mlx_multiply(&mlp_gate_silu, mlp_gate_out, mlp_gate_sigmoid,
                     gpu_stream) != 0 ||
        mlx_multiply(&swiglu, mlp_gate_silu, mlp_up_out, gpu_stream) != 0 ||
        !make_nvfp4_linear_reference(
            &mlp_out, swiglu, mlp_down_weight, mlp_down_scale,
            mlp_down_global_scale, gpu_stream, error, error_capacity) ||
        mlx_add(&layer_out, after_attention, mlp_out, gpu_stream) != 0 ||
        !summarize_array(gdn_normalized, gpu_stream, &result->gdn_norm, error,
                         error_capacity) ||
        !summarize_array(gated_flat, gpu_stream, &result->gated, error,
                         error_capacity) ||
        !summarize_array(attention_out, gpu_stream, &result->attention_out,
                         error, error_capacity) ||
        !summarize_array(after_attention, gpu_stream, &result->after_attention,
                         error, error_capacity) ||
        !summarize_array(post_normalized, gpu_stream, &result->post_norm, error,
                         error_capacity) ||
        !summarize_array(mlp_gate_out, gpu_stream, &result->mlp_gate, error,
                         error_capacity) ||
        !summarize_array(mlp_up_out, gpu_stream, &result->mlp_up, error,
                         error_capacity) ||
        !summarize_array(swiglu, gpu_stream, &result->swiglu, error,
                         error_capacity) ||
        !summarize_array(mlp_out, gpu_stream, &result->mlp_out, error,
                         error_capacity) ||
        !summarize_array(layer_out, gpu_stream, &result->layer_out, error,
                         error_capacity)) {
      if (error != nullptr && error[0] == '\0') {
        set_error(error, error_capacity,
                  "MLX failed to execute the complete layer-zero checkpoint");
      }
      goto cleanup;
    }
    result->full_layer_executed = 1;
    result->mlp_width = mlp_width;
  }

  result->token_id = token_id;
  result->input_width = input_width;
  result->qkv_width = qkv_width;
  result->z_width = z_width;
  result->ba_width = ba_width;
  outcome = 0;

cleanup:
  mlx_vector_array_free(conv_window_values);
  mlx_vector_array_free(ba_values);
  mlx_vector_array_free(global_values);
  mlx_vector_array_free(qkvz_scale_values);
  mlx_vector_array_free(qkvz_weight_values);
  mlx_array_free(no_bias);
  mlx_array_free(layer_out);
  mlx_array_free(mlp_out);
  mlx_array_free(swiglu);
  mlx_array_free(mlp_gate_silu);
  mlx_array_free(mlp_gate_sigmoid);
  mlx_array_free(mlp_up_out);
  mlx_array_free(mlp_gate_out);
  mlx_array_free(post_normalized);
  mlx_array_free(effective_post_norm_weight);
  mlx_array_free(post_norm_one);
  mlx_array_free(after_attention);
  mlx_array_free(attention_out);
  mlx_array_free(gated_flat);
  mlx_array_free(gated);
  mlx_array_free(gated_f32);
  mlx_array_free(z_gate_silu);
  mlx_array_free(z_gate_sigmoid);
  mlx_array_free(z_gate_f32);
  mlx_array_free(gdn_normalized_f32);
  mlx_array_free(gdn_normalized);
  mlx_array_free(z_gate_shaped);
  mlx_array_free(z_gate);
  mlx_array_free(delta_state);
  mlx_array_free(gdn_out);
  mlx_array_free(prior_delta_state);
  mlx_array_free(a_exp);
  mlx_array_free(a_log_f32);
  mlx_array_free(conv_state);
  mlx_array_free(conv_out);
  mlx_array_free(conv_window);
  mlx_array_free(prior_conv_state);
  mlx_array_free(conv_weight_2d);
  mlx_array_free(qkv_input);
  mlx_array_free(mixed_ba);
  mlx_array_free(packed_ba_transposed);
  mlx_array_free(packed_ba_weight);
  mlx_array_free(mixed_qkvz);
  mlx_array_free(scaled_qkvz_f32);
  mlx_array_free(raw_qkvz);
  mlx_array_free(packed_global);
  mlx_array_free(expanded_z_global);
  mlx_array_free(expanded_qkv_global);
  mlx_array_free(packed_scale);
  mlx_array_free(packed_weight);
  mlx_array_free(normalized);
  mlx_array_free(embedding);
  mlx_array_free(indices);
  mlx_array_free(effective_norm_weight);
  mlx_array_free(one);
  mlx_array_free(one_f32);
  mlx_array_free(alpha_weight);
  mlx_array_free(beta_weight);
  mlx_array_free(a_log);
  mlx_array_free(dt_bias);
  mlx_array_free(conv_weight);
  mlx_array_free(mlp_down_global_scale);
  mlx_array_free(mlp_down_scale);
  mlx_array_free(mlp_down_weight);
  mlx_array_free(mlp_up_global_scale);
  mlx_array_free(mlp_up_scale);
  mlx_array_free(mlp_up_weight);
  mlx_array_free(mlp_gate_global_scale);
  mlx_array_free(mlp_gate_scale);
  mlx_array_free(mlp_gate_weight);
  mlx_array_free(post_norm_weight);
  mlx_array_free(out_proj_global_scale);
  mlx_array_free(out_proj_scale);
  mlx_array_free(out_proj_weight);
  mlx_array_free(gdn_norm_weight);
  mlx_array_free(z_global_scale);
  mlx_array_free(z_scale);
  mlx_array_free(z_weight);
  mlx_array_free(qkv_global_scale);
  mlx_array_free(qkv_scale);
  mlx_array_free(qkv_weight);
  mlx_array_free(norm_weight);
  mlx_array_free(embedding_weight);
  mlx_map_string_to_string_free(mlp_down_metadata);
  mlx_map_string_to_array_free(mlp_down_arrays);
  mlx_map_string_to_string_free(mlp_up_metadata);
  mlx_map_string_to_array_free(mlp_up_arrays);
  mlx_map_string_to_string_free(mlp_gate_metadata);
  mlx_map_string_to_array_free(mlp_gate_arrays);
  mlx_map_string_to_string_free(post_norm_metadata);
  mlx_map_string_to_array_free(post_norm_arrays);
  mlx_map_string_to_string_free(out_proj_metadata);
  mlx_map_string_to_array_free(out_proj_arrays);
  mlx_map_string_to_string_free(gdn_norm_metadata);
  mlx_map_string_to_array_free(gdn_norm_arrays);
  mlx_map_string_to_string_free(a_log_metadata);
  mlx_map_string_to_array_free(a_log_arrays);
  mlx_map_string_to_string_free(dt_bias_metadata);
  mlx_map_string_to_array_free(dt_bias_arrays);
  mlx_map_string_to_string_free(conv_metadata);
  mlx_map_string_to_array_free(conv_arrays);
  mlx_map_string_to_string_free(alpha_metadata);
  mlx_map_string_to_array_free(alpha_arrays);
  mlx_map_string_to_string_free(beta_metadata);
  mlx_map_string_to_array_free(beta_arrays);
  mlx_map_string_to_string_free(z_metadata);
  mlx_map_string_to_array_free(z_arrays);
  mlx_map_string_to_string_free(qkv_metadata);
  mlx_map_string_to_array_free(qkv_arrays);
  mlx_map_string_to_string_free(norm_metadata);
  mlx_map_string_to_array_free(norm_arrays);
  mlx_map_string_to_string_free(embedding_metadata);
  mlx_map_string_to_array_free(embedding_arrays);
  mlx_stream_free(gpu_stream);
  mlx_stream_free(cpu_stream);
  return outcome;
}

namespace {

constexpr int qwen_layer_count = 64;
constexpr int qwen_hidden_width = 5120;
constexpr int qwen_mlp_width = 17408;
constexpr int qwen_linear_qkv_width = 10240;
constexpr int qwen_linear_z_width = 6144;
constexpr int qwen_linear_ba_width = 96;
constexpr int qwen_linear_key_heads = 16;
constexpr int qwen_linear_key_dim = 128;
constexpr int qwen_linear_value_heads = 48;
constexpr int qwen_linear_value_dim = 128;
constexpr int qwen_conv_kernel_width = 4;
constexpr int qwen_attention_heads = 24;
constexpr int qwen_attention_kv_heads = 4;
constexpr int qwen_attention_head_dim = 256;
constexpr int qwen_attention_qg_width = 12288;
constexpr int qwen_attention_kv_width = 1024;
constexpr int qwen_rope_dim = 64;
constexpr int qwen_vocab_size = 248320;
constexpr int qwen_draft_special_start = 248000;
constexpr int qwen_draft_special_count =
    qwen_vocab_size - qwen_draft_special_start;
constexpr int qwen_draft_shortlist_size = 65536;
constexpr size_t qwen_draft_extra_token_limit = 256;
constexpr size_t qwen_context_length = 262144;
constexpr size_t qwen_prefill_batch_tokens = 512;
constexpr float qwen_rope_theta = 10000000.0F;
constexpr float qwen_attention_scale = 0.0625F;
constexpr float qwen_rms_epsilon = 1.0e-6F;
constexpr size_t qwen_shifted_norm_count =
    1 + 2 * qwen_layer_count + 2 * (qwen_layer_count / 4);
constexpr size_t qwen_mtp_shifted_norm_count = 7;

enum class qwen_state_schedule { eager, deferred, asynchronous };

bool qwen_layer_is_linear(int layer) { return layer % 4 != 3; }

std::string qwen_layer_prefix(int layer) {
  return "model.language_model.layers." + std::to_string(layer);
}

struct qwen_qmv_geometry {
  int default_results_per_simdgroup = 2;
  int override_output_width = 0;
  int override_input_width = 0;
  int override_results_per_simdgroup = 0;
};

thread_local qwen_qmv_geometry qwen_active_qmv_geometry;

bool qwen_valid_qmv_results(int results_per_simdgroup) {
  return results_per_simdgroup == 1 || results_per_simdgroup == 2 ||
         results_per_simdgroup == 4 || results_per_simdgroup == 8;
}

int qwen_qmv_results_for(const mlx::core::array &input,
                         const mlx::core::array &weight) {
  if (qwen_active_qmv_geometry.override_results_per_simdgroup != 0 &&
      weight.shape(-2) == qwen_active_qmv_geometry.override_output_width &&
      input.shape(-1) == qwen_active_qmv_geometry.override_input_width) {
    return qwen_active_qmv_geometry.override_results_per_simdgroup;
  }
  return qwen_active_qmv_geometry.default_results_per_simdgroup;
}

class qwen_qmv_geometry_scope {
public:
  explicit qwen_qmv_geometry_scope(int results_per_simdgroup)
      : qwen_qmv_geometry_scope(
            qwen_qmv_geometry{results_per_simdgroup, 0, 0, 0}) {}

  explicit qwen_qmv_geometry_scope(qwen_qmv_geometry geometry)
      : previous_(qwen_active_qmv_geometry) {
    qwen_active_qmv_geometry = geometry;
  }

  ~qwen_qmv_geometry_scope() {
    qwen_active_qmv_geometry = previous_;
  }

private:
  qwen_qmv_geometry previous_;
};

mlx::core::array qwen_shift_norm_weight(const mlx::core::array &weight,
                                        mlx::core::Stream stream) {
  auto one = mlx::core::astype(mlx::core::array(1.0F), weight.dtype(), stream);
  return mlx::core::add(weight, one, stream);
}

mlx::core::array qwen_rms_norm(const mlx::core::array &input,
                               const mlx::core::array &weight,
                               mlx::core::Stream stream) {
  return mlx::core::fast::rms_norm(input, weight, qwen_rms_epsilon, stream);
}

mlx::core::array qwen_nvfp4_linear(const mlx::core::array &input,
                                   const mlx::core::array &weight,
                                   const mlx::core::array &scales,
                                   const mlx::core::array &global_scale,
                                   mlx::core::Stream stream,
                                   const mlx::core::array *fused_global_scale =
                                       nullptr,
                                   bool exact_global_scale = false) {
  const int qmv_results_per_simdgroup = qwen_qmv_results_for(input, weight);
  if (fused_global_scale != nullptr) {
    auto output_shape = input.shape();
    output_shape.back() = weight.shape(-2);
    return mlx::core::array(
        std::move(output_shape), input.dtype(),
        std::make_shared<mlx::core::QuantizedMatmul>(
            stream, 16, 4, mlx::core::QuantizationMode::Nvfp4, true, true,
            exact_global_scale, qmv_results_per_simdgroup),
        std::vector<mlx::core::array>{input, weight, scales,
                                      *fused_global_scale});
  }
  auto raw = [&]() {
    if (qmv_results_per_simdgroup == 0) {
      return mlx::core::quantized_matmul(input, weight, scales, std::nullopt,
                                         true, 16, 4, "nvfp4", stream);
    }
    auto output_shape = input.shape();
    output_shape.back() = weight.shape(-2);
    return mlx::core::array(
        std::move(output_shape), input.dtype(),
        std::make_shared<mlx::core::QuantizedMatmul>(
            stream, 16, 4, mlx::core::QuantizationMode::Nvfp4, true, false,
            false, qmv_results_per_simdgroup),
        std::vector<mlx::core::array>{input, weight, scales});
  }();
  auto scaled = mlx::core::multiply(raw, global_scale, stream);
  return mlx::core::astype(scaled, raw.dtype(), stream);
}

std::vector<mlx::core::array>
qwen_silu_graph(const std::vector<mlx::core::array> &inputs) {
  auto sigmoid = mlx::core::sigmoid(inputs[0]);
  return {mlx::core::multiply(inputs[0], sigmoid)};
}

std::vector<mlx::core::array>
qwen_swiglu_graph(const std::vector<mlx::core::array> &inputs) {
  auto sigmoid = mlx::core::sigmoid(inputs[0]);
  auto silu = mlx::core::multiply(inputs[0], sigmoid);
  return {mlx::core::multiply(silu, inputs[1])};
}

mlx::core::array qwen_silu(const mlx::core::array &input,
                           mlx::core::Stream stream) {
  (void)stream;
  static auto compiled = mlx::core::compile(qwen_silu_graph, true);
  return compiled({input})[0];
}

mlx::core::array qwen_swiglu(const mlx::core::array &gate,
                             const mlx::core::array &up,
                             mlx::core::Stream stream) {
  (void)stream;
  static auto compiled = mlx::core::compile(qwen_swiglu_graph, true);
  return compiled({gate, up})[0];
}

mlx::core::array qwen_depthwise_conv_silu(
    const mlx::core::array &input, const mlx::core::array &weight,
    mlx_stream stream, char *error, size_t error_capacity) {
  if (input.ndim() != 3 || input.shape(0) != 1 ||
      input.shape(1) < qwen_conv_kernel_width ||
      input.shape(2) != qwen_linear_qkv_width) {
    throw std::runtime_error("Qwen depthwise convolution geometry changed");
  }
  const int tokens = input.shape(1) - qwen_conv_kernel_width + 1;
  mlx_array input_handle = mlx_array_new_(input);
  mlx_array weight_handle = mlx_array_new_(weight);
  mlx_array output_handle = mlx_array_new();
  const bool success = make_depthwise_conv_silu(
      &output_handle, input_handle, weight_handle, 1, tokens,
      qwen_linear_qkv_width, qwen_conv_kernel_width, stream, error,
      error_capacity);
  mlx::core::array output =
      success ? mlx_array_get_(output_handle) : input;
  mlx_array_free(output_handle);
  mlx_array_free(weight_handle);
  mlx_array_free(input_handle);
  if (!success) {
    throw std::runtime_error("failed to create Qwen depthwise conv-SiLU");
  }
  return output;
}

mlx::core::array qwen_exact_ba_projection(
    const mlx::core::array &input, const mlx::core::array &weight,
    mlx_stream stream, char *error, size_t error_capacity) {
  if (input.ndim() != 3 || input.shape(0) != 1 || input.shape(1) < 2 ||
      input.shape(1) > 8 || input.shape(2) != qwen_hidden_width ||
      weight.ndim() != 2 || weight.shape(0) != qwen_linear_ba_width ||
      weight.shape(1) != qwen_hidden_width || input.dtype() != weight.dtype()) {
    throw std::runtime_error("Qwen exact BA projection geometry changed");
  }
  mlx_array input_handle = mlx_array_new_(input);
  mlx_array weight_handle = mlx_array_new_(weight);
  mlx_array output_handle = mlx_array_new();
  const bool success = make_exact_ba_projection(
      &output_handle, input_handle, weight_handle, input.shape(1), stream,
      error, error_capacity);
  mlx::core::array output = success ? mlx_array_get_(output_handle) : input;
  mlx_array_free(output_handle);
  mlx_array_free(weight_handle);
  mlx_array_free(input_handle);
  if (!success) {
    throw std::runtime_error("failed to create exact Qwen BA projection");
  }
  return output;
}

struct qwen_gated_delta_arrays {
  mlx::core::array output;
  mlx::core::array state;
};

qwen_gated_delta_arrays qwen_gated_delta_decode(
    const mlx::core::array &packed, const mlx::core::array &ba,
    const mlx::core::array &dt_bias, const mlx::core::array &a_exp,
    const mlx::core::array &prior_state, mlx_stream stream, char *error,
    size_t error_capacity) {
  if (packed.ndim() != 3 || packed.shape(0) != 1 ||
      packed.shape(2) != qwen_linear_qkv_width || ba.ndim() != 3 ||
      ba.shape(0) != 1 || ba.shape(1) != packed.shape(1) ||
      ba.shape(2) != qwen_linear_ba_width) {
    throw std::runtime_error("Qwen gated-delta sequence geometry changed");
  }
  const int tokens = packed.shape(1);
  mlx_array packed_handle = mlx_array_new_(packed);
  mlx_array ba_handle = mlx_array_new_(ba);
  mlx_array dt_bias_handle = mlx_array_new_(dt_bias);
  mlx_array a_exp_handle = mlx_array_new_(a_exp);
  mlx_array prior_state_handle = mlx_array_new_(prior_state);
  mlx_array output_handle = mlx_array_new();
  mlx_array state_handle = mlx_array_new();
  const bool success = make_gated_delta_decode(
      &output_handle, &state_handle, packed_handle, ba_handle,
      dt_bias_handle, a_exp_handle, prior_state_handle, 1, tokens,
      qwen_linear_key_heads, qwen_linear_key_dim, qwen_linear_value_heads,
      qwen_linear_value_dim, stream, error, error_capacity);
  mlx::core::array output = success ? mlx_array_get_(output_handle) : packed;
  mlx::core::array state = success ? mlx_array_get_(state_handle) : prior_state;
  mlx_array_free(state_handle);
  mlx_array_free(output_handle);
  mlx_array_free(prior_state_handle);
  mlx_array_free(a_exp_handle);
  mlx_array_free(dt_bias_handle);
  mlx_array_free(ba_handle);
  mlx_array_free(packed_handle);
  if (!success) {
    throw std::runtime_error("failed to create Qwen gated-delta decode");
  }
  return {std::move(output), std::move(state)};
}

struct qwen_linear_prepack {
  mlx::core::array qkvz_weight;
  mlx::core::array qkvz_scales;
  mlx::core::array qkvz_global_scale;
  mlx::core::array ba_weight;
  mlx::core::array ba_transposed;
  mlx::core::array conv_weight;
  mlx::core::array a_exp;

  qwen_linear_prepack(mlx::core::array qkvz_weight,
                      mlx::core::array qkvz_scales,
                      mlx::core::array qkvz_global_scale,
                      mlx::core::array ba_weight,
                      mlx::core::array conv_weight,
                      mlx::core::array a_exp, mlx::core::Stream stream)
      : qkvz_weight(std::move(qkvz_weight)),
        qkvz_scales(std::move(qkvz_scales)),
        qkvz_global_scale(std::move(qkvz_global_scale)),
        ba_weight(std::move(ba_weight)),
        ba_transposed(mlx::core::transpose(this->ba_weight, stream)),
        conv_weight(std::move(conv_weight)), a_exp(std::move(a_exp)) {}
};

struct qwen_mlp_prepack {
  mlx::core::array gate_up_weight;
  mlx::core::array gate_up_scales;
  mlx::core::array gate_up_global_scale;
  mlx::core::array gate_weight;
  mlx::core::array up_weight;
  mlx::core::array gate_scales;
  mlx::core::array up_scales;
  mlx::core::array gate_global_scale;
  mlx::core::array up_global_scale;

  qwen_mlp_prepack(mlx::core::array gate_up_weight,
                   mlx::core::array gate_up_scales,
                   mlx::core::array gate_up_global_scale,
                   mlx::core::Stream stream)
      : gate_up_weight(std::move(gate_up_weight)),
        gate_up_scales(std::move(gate_up_scales)),
        gate_up_global_scale(std::move(gate_up_global_scale)),
        gate_weight(mlx::core::slice(
            this->gate_up_weight, {0, 0},
            {qwen_mlp_width, this->gate_up_weight.shape(1)}, stream)),
        up_weight(mlx::core::slice(
            this->gate_up_weight, {qwen_mlp_width, 0},
            {2 * qwen_mlp_width, this->gate_up_weight.shape(1)}, stream)),
        gate_scales(mlx::core::slice(
            this->gate_up_scales, {0, 0},
            {qwen_mlp_width, this->gate_up_scales.shape(1)}, stream)),
        up_scales(mlx::core::slice(
            this->gate_up_scales, {qwen_mlp_width, 0},
            {2 * qwen_mlp_width, this->gate_up_scales.shape(1)}, stream)),
        gate_global_scale(mlx::core::slice(this->gate_up_global_scale, {0},
                                           {1}, stream)),
        up_global_scale(mlx::core::slice(
            this->gate_up_global_scale, {qwen_mlp_width},
                                           {qwen_mlp_width + 1}, stream)) {}
};

struct qwen_layer_cache {
  std::optional<mlx::core::array> conv_state;
  std::optional<mlx::core::array> delta_state;
  // Compact verification-only roots used to recover a committed recurrent
  // prefix without replaying the target model's weight projections.
  std::optional<mlx::core::array> rollback_conv_window;
  std::optional<mlx::core::array> rollback_gdn_input;
  std::optional<mlx::core::array> rollback_ba;
  std::optional<mlx::core::array> keys;
  std::optional<mlx::core::array> values;
};

} // namespace

struct generalist_mlx_qwen_model {
  std::unordered_map<std::string, mlx::core::array> tensors;
  std::unordered_map<std::string, mlx::core::array> encoded_global_scales;
  std::unordered_map<std::string, mlx::core::array> shifted_norm_weights;
  std::array<std::unique_ptr<qwen_linear_prepack>, qwen_layer_count>
      linear_prepack;
  std::array<std::unique_ptr<qwen_mlp_prepack>, qwen_layer_count> mlp_prepack;
  std::unique_ptr<qwen_mlp_prepack> mtp_mlp_prepack;
  std::optional<mlx::core::array> mtp_draft_lm_head_weight;
  std::optional<mlx::core::array> mtp_draft_lm_head_scales;

  bool has_mtp() const { return mtp_mlp_prepack != nullptr; }

  const mlx::core::array &tensor(const std::string &name) const {
    auto found = tensors.find(name);
    if (found == tensors.end()) {
      throw std::runtime_error("missing Qwen tensor: " + name);
    }
    return found->second;
  }

  const mlx::core::array &encoded_global_scale(
      const std::string &name) const {
    auto found = encoded_global_scales.find(name);
    if (found == encoded_global_scales.end()) {
      throw std::runtime_error("missing encoded Qwen global scale: " + name);
    }
    return found->second;
  }

  const mlx::core::array &shifted_norm_weight(
      const std::string &name) const {
    auto found = shifted_norm_weights.find(name);
    if (found == shifted_norm_weights.end()) {
      throw std::runtime_error("missing shifted Qwen norm weight: " + name);
    }
    return found->second;
  }

  void prepare(mlx::core::Stream stream) {
    const auto &embedding = tensor("model.language_model.embed_tokens.weight");
    const auto &final_norm = tensor("model.language_model.norm.weight");
    const auto &lm_head = tensor("lm_head.weight");
    if (embedding.ndim() != 2 || embedding.shape(1) != qwen_hidden_width ||
        final_norm.ndim() != 1 || final_norm.size() != qwen_hidden_width ||
        lm_head.ndim() != 2 || lm_head.shape(0) != qwen_vocab_size ||
        lm_head.shape(1) != qwen_hidden_width * 4 / 32) {
      throw std::runtime_error("Qwen model identity or core geometry changed");
    }

    shifted_norm_weights.reserve(qwen_shifted_norm_count);
    shifted_norm_weights.emplace(
        "model.language_model.norm.weight",
        qwen_shift_norm_weight(final_norm, stream));

    for (int layer = 0; layer < qwen_layer_count; ++layer) {
      const std::string prefix = qwen_layer_prefix(layer);
      for (const std::string &name : {
               prefix + ".input_layernorm.weight",
               prefix + ".post_attention_layernorm.weight",
           }) {
        shifted_norm_weights.emplace(
            name, qwen_shift_norm_weight(tensor(name), stream));
      }
      const std::string gate_name = prefix + ".mlp.gate_proj.weight";
      const std::string up_name = prefix + ".mlp.up_proj.weight";
      const auto &gate_weight = tensor(gate_name);
      const auto &up_weight = tensor(up_name);
      const auto &down_weight = tensor(prefix + ".mlp.down_proj.weight");
      if (gate_weight.ndim() != 2 ||
          gate_weight.shape(0) != qwen_mlp_width ||
          up_weight.ndim() != 2 || up_weight.shape(0) != qwen_mlp_width ||
          down_weight.ndim() != 2 ||
          down_weight.shape(0) != qwen_hidden_width) {
        throw std::runtime_error("Qwen MLP geometry changed at layer " +
                                 std::to_string(layer));
      }
      auto gate_up_weight =
          mlx::core::concatenate({gate_weight, up_weight}, 0, stream);
      auto gate_up_scales = mlx::core::concatenate(
          {tensor(gate_name + ".scale"), tensor(up_name + ".scale")}, 0,
          stream);
      auto gate_up_global_scale = mlx::core::concatenate(
          {mlx::core::broadcast_to(tensor(gate_name + ".global_scale"),
                                   {qwen_mlp_width}, stream),
           mlx::core::broadcast_to(tensor(up_name + ".global_scale"),
                                   {qwen_mlp_width}, stream)},
          0, stream);
      mlp_prepack[layer] = std::make_unique<qwen_mlp_prepack>(
          std::move(gate_up_weight), std::move(gate_up_scales),
          std::move(gate_up_global_scale), stream);
      if (!qwen_layer_is_linear(layer)) {
        const std::string q_name = prefix + ".self_attn.q_proj.weight";
        const std::string k_name = prefix + ".self_attn.k_proj.weight";
        const std::string v_name = prefix + ".self_attn.v_proj.weight";
        const auto &q_weight = tensor(q_name);
        const auto &k_weight = tensor(k_name);
        const auto &v_weight = tensor(v_name);
        (void)tensor(prefix + ".self_attn.o_proj.weight");
        for (const std::string &name : {
                 prefix + ".self_attn.q_norm.weight",
                 prefix + ".self_attn.k_norm.weight",
             }) {
          shifted_norm_weights.emplace(
              name, qwen_shift_norm_weight(tensor(name), stream));
        }
        if (q_weight.ndim() != 2 ||
            q_weight.shape(0) != qwen_attention_qg_width ||
            k_weight.ndim() != 2 ||
            k_weight.shape(0) != qwen_attention_kv_width ||
            v_weight.ndim() != 2 ||
            v_weight.shape(0) != qwen_attention_kv_width) {
          throw std::runtime_error(
              "Qwen full-attention geometry changed at layer " +
              std::to_string(layer));
        }
        continue;
      }

      const std::string qkv_name = prefix + ".linear_attn.in_proj_qkv.weight";
      const std::string z_name = prefix + ".linear_attn.in_proj_z.weight";
      const auto &qkv_weight = tensor(qkv_name);
      const auto &z_weight = tensor(z_name);
      const auto &qkv_scales = tensor(qkv_name + ".scale");
      const auto &z_scales = tensor(z_name + ".scale");
      const auto &qkv_global = tensor(qkv_name + ".global_scale");
      const auto &z_global = tensor(z_name + ".global_scale");
      auto packed_weight =
          mlx::core::concatenate({qkv_weight, z_weight}, 0, stream);
      auto packed_scales =
          mlx::core::concatenate({qkv_scales, z_scales}, 0, stream);
      auto packed_global = mlx::core::concatenate(
          {mlx::core::broadcast_to(qkv_global, {qwen_linear_qkv_width}, stream),
           mlx::core::broadcast_to(z_global, {qwen_linear_z_width}, stream)},
          0, stream);
      auto packed_ba = mlx::core::concatenate(
          {tensor(prefix + ".linear_attn.in_proj_b.weight"),
           tensor(prefix + ".linear_attn.in_proj_a.weight")},
          0, stream);
      if (packed_ba.ndim() != 2 ||
          packed_ba.shape(0) != qwen_linear_ba_width ||
          packed_ba.shape(1) != qwen_hidden_width) {
        throw std::runtime_error(
            "Qwen linear-attention BA geometry changed at layer " +
            std::to_string(layer));
      }
      auto conv_weight = mlx::core::reshape(
          tensor(prefix + ".linear_attn.conv1d.weight"),
          {qwen_linear_qkv_width, qwen_conv_kernel_width}, stream);
      auto a_exp = mlx::core::exp(
          mlx::core::astype(tensor(prefix + ".linear_attn.A_log"),
                            mlx::core::float32, stream),
          stream);
      linear_prepack[layer] = std::make_unique<qwen_linear_prepack>(
          std::move(packed_weight), std::move(packed_scales),
          std::move(packed_global), std::move(packed_ba),
          std::move(conv_weight), std::move(a_exp), stream);
    }

    bool mtp_present = false;
    for (const auto &[name, unused_value] : tensors) {
      (void)unused_value;
      if (name.starts_with("mtp.")) {
        mtp_present = true;
        break;
      }
    }
    if (mtp_present) {
      for (const std::string &name : {
               "mtp.pre_fc_norm_embedding.weight",
               "mtp.pre_fc_norm_hidden.weight",
               "mtp.layers.0.input_layernorm.weight",
               "mtp.layers.0.post_attention_layernorm.weight",
               "mtp.layers.0.self_attn.q_norm.weight",
               "mtp.layers.0.self_attn.k_norm.weight",
               "mtp.norm.weight",
           }) {
        const auto &weight = tensor(name);
        if (weight.ndim() != 1 ||
            (name.ends_with(".q_norm.weight") ||
                     name.ends_with(".k_norm.weight")
                 ? weight.size() != qwen_attention_head_dim
                 : weight.size() != qwen_hidden_width)) {
          throw std::runtime_error("Qwen MTP norm geometry changed: " + name);
        }
        shifted_norm_weights.emplace(name,
                                     qwen_shift_norm_weight(weight, stream));
      }

      const auto &fc_weight = tensor("mtp.fc.weight");
      const auto &q_weight = tensor("mtp.layers.0.self_attn.q_proj.weight");
      const auto &k_weight = tensor("mtp.layers.0.self_attn.k_proj.weight");
      const auto &v_weight = tensor("mtp.layers.0.self_attn.v_proj.weight");
      const auto &o_weight = tensor("mtp.layers.0.self_attn.o_proj.weight");
      if (fc_weight.ndim() != 2 || fc_weight.shape(0) != qwen_hidden_width ||
          fc_weight.shape(1) != 2 * qwen_hidden_width * 4 / 32 ||
          q_weight.ndim() != 2 ||
          q_weight.shape(0) != qwen_attention_qg_width ||
          k_weight.ndim() != 2 ||
          k_weight.shape(0) != qwen_attention_kv_width ||
          v_weight.ndim() != 2 ||
          v_weight.shape(0) != qwen_attention_kv_width ||
          o_weight.ndim() != 2 || o_weight.shape(0) != qwen_hidden_width) {
        throw std::runtime_error("Qwen MTP projection geometry changed");
      }

      const std::string gate_name = "mtp.layers.0.mlp.gate_proj.weight";
      const std::string up_name = "mtp.layers.0.mlp.up_proj.weight";
      const auto &gate_weight = tensor(gate_name);
      const auto &up_weight = tensor(up_name);
      const auto &down_weight =
          tensor("mtp.layers.0.mlp.down_proj.weight");
      if (gate_weight.ndim() != 2 ||
          gate_weight.shape(0) != qwen_mlp_width ||
          up_weight.ndim() != 2 || up_weight.shape(0) != qwen_mlp_width ||
          down_weight.ndim() != 2 ||
          down_weight.shape(0) != qwen_hidden_width) {
        throw std::runtime_error("Qwen MTP MLP geometry changed");
      }
      auto gate_up_weight =
          mlx::core::concatenate({gate_weight, up_weight}, 0, stream);
      auto gate_up_scales = mlx::core::concatenate(
          {tensor(gate_name + ".scale"), tensor(up_name + ".scale")}, 0,
          stream);
      auto gate_up_global_scale = mlx::core::concatenate(
          {mlx::core::broadcast_to(tensor(gate_name + ".global_scale"),
                                   {qwen_mlp_width}, stream),
           mlx::core::broadcast_to(tensor(up_name + ".global_scale"),
                                   {qwen_mlp_width}, stream)},
          0, stream);
      mtp_mlp_prepack = std::make_unique<qwen_mlp_prepack>(
          std::move(gate_up_weight), std::move(gate_up_scales),
          std::move(gate_up_global_scale), stream);

      // Draft-only FR-Spec experiment: special/control rows precede a
      // contiguous low-ID band whose prefixes can be sliced without another
      // gather. Target verification always retains the complete LM head, so a
      // missing draft row can affect acceptance but never the committed model
      // semantics.
      std::vector<int32_t> draft_vocab_ids;
      draft_vocab_ids.reserve(qwen_draft_special_count +
                              qwen_draft_shortlist_size);
      for (int token_id = qwen_draft_special_start;
           token_id < qwen_vocab_size; ++token_id) {
        draft_vocab_ids.push_back(token_id);
      }
      for (int token_id = 0; token_id < qwen_draft_shortlist_size;
           ++token_id) {
        draft_vocab_ids.push_back(token_id);
      }
      auto draft_vocab = mlx::core::array(
          draft_vocab_ids.begin(),
          {static_cast<int>(draft_vocab_ids.size())});
      mtp_draft_lm_head_weight =
          mlx::core::take(lm_head, draft_vocab, 0, stream);
      mtp_draft_lm_head_scales = mlx::core::take(
          tensor("lm_head.weight.scale"), draft_vocab, 0, stream);
    }

    const size_t expected_shifted_norms =
        qwen_shifted_norm_count +
        (mtp_present ? qwen_mtp_shifted_norm_count : 0);
    if (shifted_norm_weights.size() != expected_shifted_norms) {
      throw std::runtime_error("Qwen shifted norm inventory changed");
    }

    for (const auto &[name, value] : tensors) {
      if (name.ends_with(".global_scale") && value.size() == 1) {
        encoded_global_scales.emplace(
            name, mlx::core::multiply(value, mlx::core::array(2688.0F),
                                      stream));
      }
    }

    // Safetensor loads plus the concatenated linear-attention and dense-MLP
    // layouts are lazy MLX arrays. Materialize the derived layouts first, then
    // discard their split source arrays so steady-state decode reads resident
    // unified memory instead of replaying 18 GB of file-backed load graphs per
    // token.
    std::vector<mlx::core::array> packed_arrays;
    packed_arrays.reserve(48 * 7 + qwen_layer_count * 3 +
                          (has_mtp() ? 5 : 0));
    for (const auto &prepack : linear_prepack) {
      if (prepack == nullptr) {
        continue;
      }
      packed_arrays.push_back(prepack->qkvz_weight);
      packed_arrays.push_back(prepack->qkvz_scales);
      packed_arrays.push_back(prepack->qkvz_global_scale);
      packed_arrays.push_back(prepack->ba_weight);
      packed_arrays.push_back(prepack->ba_transposed);
      packed_arrays.push_back(prepack->conv_weight);
      packed_arrays.push_back(prepack->a_exp);
    }
    for (const auto &prepack : mlp_prepack) {
      packed_arrays.push_back(prepack->gate_up_weight);
      packed_arrays.push_back(prepack->gate_up_scales);
      packed_arrays.push_back(prepack->gate_up_global_scale);
    }
    if (has_mtp()) {
      packed_arrays.push_back(mtp_mlp_prepack->gate_up_weight);
      packed_arrays.push_back(mtp_mlp_prepack->gate_up_scales);
      packed_arrays.push_back(mtp_mlp_prepack->gate_up_global_scale);
      packed_arrays.push_back(*mtp_draft_lm_head_weight);
      packed_arrays.push_back(*mtp_draft_lm_head_scales);
    }
    mlx::core::eval(std::move(packed_arrays));

    for (int layer = 0; layer < qwen_layer_count; ++layer) {
      if (!qwen_layer_is_linear(layer)) {
        continue;
      }
      const std::string prefix = qwen_layer_prefix(layer);
      const std::string qkv_name = prefix + ".linear_attn.in_proj_qkv.weight";
      const std::string z_name = prefix + ".linear_attn.in_proj_z.weight";
      for (const std::string &name : {
               qkv_name,
               qkv_name + ".scale",
               qkv_name + ".global_scale",
               z_name,
               z_name + ".scale",
               z_name + ".global_scale",
               prefix + ".linear_attn.in_proj_b.weight",
               prefix + ".linear_attn.in_proj_a.weight",
               prefix + ".linear_attn.conv1d.weight",
               prefix + ".linear_attn.A_log",
           }) {
        tensors.erase(name);
      }
    }

    for (int layer = 0; layer < qwen_layer_count; ++layer) {
      const std::string prefix = qwen_layer_prefix(layer);
      for (const std::string &name : {
               prefix + ".mlp.gate_proj.weight",
               prefix + ".mlp.up_proj.weight",
           }) {
        tensors.erase(name);
        tensors.erase(name + ".scale");
        tensors.erase(name + ".global_scale");
      }
    }
    if (has_mtp()) {
      for (const std::string &name : {
               "mtp.layers.0.mlp.gate_proj.weight",
               "mtp.layers.0.mlp.up_proj.weight",
           }) {
        tensors.erase(name);
        tensors.erase(name + ".scale");
        tensors.erase(name + ".global_scale");
      }
    }

    std::vector<mlx::core::array> resident_arrays;
    resident_arrays.reserve(tensors.size() + encoded_global_scales.size() +
                            shifted_norm_weights.size() + 48 * 7 +
                            qwen_layer_count * 3 +
                            (has_mtp() ? 5 : 0));
    for (const auto &[unused_name, value] : tensors) {
      (void)unused_name;
      resident_arrays.push_back(value);
    }
    for (const auto &[unused_name, value] : encoded_global_scales) {
      (void)unused_name;
      resident_arrays.push_back(value);
    }
    for (const auto &[unused_name, value] : shifted_norm_weights) {
      (void)unused_name;
      resident_arrays.push_back(value);
    }
    for (const auto &prepack : linear_prepack) {
      if (prepack == nullptr) {
        continue;
      }
      resident_arrays.push_back(prepack->qkvz_weight);
      resident_arrays.push_back(prepack->qkvz_scales);
      resident_arrays.push_back(prepack->qkvz_global_scale);
      resident_arrays.push_back(prepack->ba_weight);
      resident_arrays.push_back(prepack->ba_transposed);
      resident_arrays.push_back(prepack->conv_weight);
      resident_arrays.push_back(prepack->a_exp);
    }
    for (const auto &prepack : mlp_prepack) {
      resident_arrays.push_back(prepack->gate_up_weight);
      resident_arrays.push_back(prepack->gate_up_scales);
      resident_arrays.push_back(prepack->gate_up_global_scale);
    }
    if (has_mtp()) {
      resident_arrays.push_back(mtp_mlp_prepack->gate_up_weight);
      resident_arrays.push_back(mtp_mlp_prepack->gate_up_scales);
      resident_arrays.push_back(mtp_mlp_prepack->gate_up_global_scale);
      resident_arrays.push_back(*mtp_draft_lm_head_weight);
      resident_arrays.push_back(*mtp_draft_lm_head_scales);
    }
    mlx::core::eval(std::move(resident_arrays));

    const size_t active_memory = mlx::core::get_active_memory();
    const auto &device_info = mlx::core::device_info();
    const auto recommended =
        device_info.find("max_recommended_working_set_size");
    if (recommended != device_info.end() &&
        std::holds_alternative<size_t>(recommended->second)) {
      mlx::core::set_wired_limit(
          std::min(active_memory, std::get<size_t>(recommended->second)));
    }
  }
};

struct generalist_mlx_qwen_cache {
  size_t position = 0;
  std::array<qwen_layer_cache, qwen_layer_count> layers;
};

struct qwen_mtp_draft_head_config {
  size_t vocab_limit = 0;
  std::vector<int32_t> extra_token_ids;
  std::optional<mlx::core::array> extra_weight;
  std::optional<mlx::core::array> extra_scales;
};

struct generalist_mlx_qwen_speculative_cache {
  generalist_mlx_qwen_cache target;
  qwen_layer_cache draft;
  size_t draft_position = 0;
  bool seeded = false;
  bool decode_ready = false;
  int current_token_id = -1;
  std::optional<mlx::core::array> held_draft_hidden;
  int held_draft_token_id = -1;
  qwen_qmv_geometry verification_qmv_geometry{4, 0, 0, 0};
  // 0: independent exact-scale QMV; 1: wide QMV plus external scale;
  // 2: fused exact-scale wide QMV.
  int verification_qmv_path = 0;
  qwen_mtp_draft_head_config draft_head;
  bool direct_cache_rollback = false;
};

namespace {

template <bool use_precomputed_norms>
mlx::core::array qwen_shifted_rms_norm(
    const generalist_mlx_qwen_model &model,
    const mlx::core::array &input, const std::string &weight_name,
    mlx::core::Stream stream) {
  if constexpr (use_precomputed_norms) {
    return qwen_rms_norm(input, model.shifted_norm_weight(weight_name),
                         stream);
  }
  return qwen_rms_norm(
      input, qwen_shift_norm_weight(model.tensor(weight_name), stream),
      stream);
}

mlx::core::array qwen_quantized_by_name(
    const generalist_mlx_qwen_model &model, const std::string &name,
    const mlx::core::array &input, mlx::core::Stream stream,
    bool experimental_fused = false, bool exact_global_scale = false) {
  if (experimental_fused && exact_global_scale) {
    throw std::runtime_error(
        "approximate and exact global-scale fusion are mutually exclusive");
  }
  const auto &global_scale = model.tensor(name + ".global_scale");
  const mlx::core::array *fused_global_scale = nullptr;
  if (experimental_fused) {
    fused_global_scale =
        &model.encoded_global_scale(name + ".global_scale");
  } else if (exact_global_scale) {
    fused_global_scale = &global_scale;
  }
  return qwen_nvfp4_linear(input, model.tensor(name),
                           model.tensor(name + ".scale"),
                           global_scale, stream, fused_global_scale,
                           exact_global_scale);
}

struct qwen_draft_lm_head_result {
  mlx::core::array logits;
  mlx::core::array top;
};

int qwen_draft_token_id(uint32_t local_id,
                        const qwen_mtp_draft_head_config &draft_head) {
  if (draft_head.vocab_limit == 0) {
    return static_cast<int>(local_id);
  }
  if (local_id < static_cast<uint32_t>(qwen_draft_special_count)) {
    return qwen_draft_special_start + static_cast<int>(local_id);
  }
  local_id -= qwen_draft_special_count;
  if (local_id < draft_head.vocab_limit) {
    return static_cast<int>(local_id);
  }
  const size_t extra_index = local_id - draft_head.vocab_limit;
  if (extra_index >= draft_head.extra_token_ids.size()) {
    throw std::runtime_error("Qwen MTP draft argmax is outside its row map");
  }
  return draft_head.extra_token_ids[extra_index];
}

qwen_draft_lm_head_result qwen_draft_lm_head(
    const generalist_mlx_qwen_model &model,
    const mlx::core::array &hidden, mlx::core::Stream stream,
    bool exact_global_scale,
    const qwen_mtp_draft_head_config &draft_head) {
  if (draft_head.vocab_limit == 0) {
    auto logits = qwen_quantized_by_name(
        model, "lm_head.weight", hidden, stream, false, exact_global_scale);
    auto top = mlx::core::argmax(logits, -1, false, stream);
    return {std::move(logits), std::move(top)};
  }
  if (draft_head.vocab_limit != qwen_draft_shortlist_size ||
      model.mtp_draft_lm_head_weight == std::nullopt ||
      model.mtp_draft_lm_head_scales == std::nullopt ||
      draft_head.extra_weight.has_value() !=
          draft_head.extra_scales.has_value() ||
      draft_head.extra_weight.has_value() !=
          !draft_head.extra_token_ids.empty()) {
    throw std::runtime_error("Qwen MTP draft vocabulary limit is invalid");
  }

  const int rows = qwen_draft_special_count +
                   static_cast<int>(draft_head.vocab_limit);
  const auto &all_weight = *model.mtp_draft_lm_head_weight;
  const auto &all_scales = *model.mtp_draft_lm_head_scales;
  auto weight = mlx::core::slice(
      all_weight, {0, 0}, {rows, all_weight.shape(1)}, stream);
  auto scales = mlx::core::slice(
      all_scales, {0, 0}, {rows, all_scales.shape(1)}, stream);
  const auto &global_scale = model.tensor("lm_head.weight.global_scale");
  auto logits = qwen_nvfp4_linear(
      hidden, weight, scales, global_scale, stream,
      exact_global_scale ? &global_scale : nullptr, exact_global_scale);
  if (draft_head.extra_weight.has_value()) {
    auto extra_logits = qwen_nvfp4_linear(
        hidden, *draft_head.extra_weight, *draft_head.extra_scales,
        global_scale, stream,
        exact_global_scale ? &global_scale : nullptr, exact_global_scale);
    logits = mlx::core::concatenate(
        {std::move(logits), std::move(extra_logits)}, 2, stream);
  }
  auto top = mlx::core::argmax(logits, -1, false, stream);
  return {std::move(logits), std::move(top)};
}

template <bool use_packed_gate_up>
mlx::core::array qwen_dense_mlp(const generalist_mlx_qwen_model &model,
                                int layer,
                                const mlx::core::array &input,
                                mlx::core::Stream stream,
                                bool experimental_fused,
                                bool exact_global_scale) {
  const std::string prefix = qwen_layer_prefix(layer);
  mlx::core::array gate = input;
  mlx::core::array up = input;
  const int tokens = input.shape(1);
  if constexpr (use_packed_gate_up) {
    if (!experimental_fused) {
      const auto &prepack = *model.mlp_prepack[layer];
      auto gate_up = qwen_nvfp4_linear(
          input, prepack.gate_up_weight, prepack.gate_up_scales,
          prepack.gate_up_global_scale, stream,
          exact_global_scale ? &prepack.gate_up_global_scale : nullptr,
          exact_global_scale);
      gate = mlx::core::slice(gate_up, {0, 0, 0},
                              {1, tokens, qwen_mlp_width}, stream);
      up = mlx::core::slice(gate_up, {0, 0, qwen_mlp_width},
                            {1, tokens, 2 * qwen_mlp_width}, stream);
    } else {
      const auto &prepack = *model.mlp_prepack[layer];
      gate = qwen_nvfp4_linear(
          input, prepack.gate_weight, prepack.gate_scales,
          prepack.gate_global_scale, stream,
          &model.encoded_global_scale(prefix +
                                      ".mlp.gate_proj.weight.global_scale"));
      up = qwen_nvfp4_linear(
          input, prepack.up_weight, prepack.up_scales,
          prepack.up_global_scale, stream,
          &model.encoded_global_scale(prefix +
                                      ".mlp.up_proj.weight.global_scale"));
    }
  } else {
    const auto &prepack = *model.mlp_prepack[layer];
    gate = qwen_nvfp4_linear(input, prepack.gate_weight, prepack.gate_scales,
                             prepack.gate_global_scale, stream,
                             exact_global_scale ? &prepack.gate_global_scale
                                                : nullptr,
                             exact_global_scale);
    up = qwen_nvfp4_linear(input, prepack.up_weight, prepack.up_scales,
                           prepack.up_global_scale, stream,
                           exact_global_scale ? &prepack.up_global_scale
                                              : nullptr,
                           exact_global_scale);
  }
  auto swiglu = qwen_swiglu(gate, up, stream);
  auto output = qwen_quantized_by_name(
      model, prefix + ".mlp.down_proj.weight", swiglu, stream,
      experimental_fused, exact_global_scale);
  return output;
}

mlx::core::array qwen_mtp_dense_mlp(
    const generalist_mlx_qwen_model &model,
    const mlx::core::array &input, mlx::core::Stream stream,
    bool exact_global_scale) {
  if (!model.has_mtp()) {
    throw std::runtime_error("Qwen MTP head is not loaded");
  }
  const auto &prepack = *model.mtp_mlp_prepack;
  const int tokens = input.shape(1);
  auto gate_up = qwen_nvfp4_linear(
      input, prepack.gate_up_weight, prepack.gate_up_scales,
      prepack.gate_up_global_scale, stream,
      exact_global_scale ? &prepack.gate_up_global_scale : nullptr,
      exact_global_scale);
  auto gate = mlx::core::slice(gate_up, {0, 0, 0},
                               {1, tokens, qwen_mlp_width}, stream);
  auto up = mlx::core::slice(gate_up, {0, 0, qwen_mlp_width},
                             {1, tokens, 2 * qwen_mlp_width}, stream);
  auto swiglu = qwen_swiglu(gate, up, stream);
  return qwen_quantized_by_name(
      model, "mtp.layers.0.mlp.down_proj.weight", swiglu, stream, false,
      exact_global_scale);
}

mlx::core::array qwen_linear_attention_zero_state(
    const generalist_mlx_qwen_model &model, int layer,
    const mlx::core::array &input, mlx_stream c_stream,
    mlx::core::Stream stream, char *error, size_t error_capacity,
    bool experimental_fused, bool exact_global_scale) {
  const std::string prefix = qwen_layer_prefix(layer);
  const qwen_linear_prepack &prepack = *model.linear_prepack[layer];
  auto mixed_qkvz = qwen_nvfp4_linear(
      input, prepack.qkvz_weight, prepack.qkvz_scales,
      prepack.qkvz_global_scale, stream,
      exact_global_scale ? &prepack.qkvz_global_scale : nullptr,
      exact_global_scale);
  auto mixed_ba = mlx::core::matmul(input, prepack.ba_transposed, stream);
  auto qkv = mlx::core::slice(mixed_qkvz, {0, 0, 0},
                              {1, 1, qwen_linear_qkv_width}, stream);
  auto z = mlx::core::slice(
      mixed_qkvz, {0, 0, qwen_linear_qkv_width},
      {1, 1, qwen_linear_qkv_width + qwen_linear_z_width}, stream);
  z = mlx::core::reshape(
      z, {1, 1, qwen_linear_value_heads, qwen_linear_value_dim}, stream);

  auto prior_conv = mlx::core::zeros(
      {1, qwen_conv_kernel_width - 1, qwen_linear_qkv_width}, input.dtype(),
      stream);
  auto conv_window =
      mlx::core::concatenate({prior_conv, qkv}, 1, stream);
  auto conv_out = qwen_depthwise_conv_silu(
      conv_window, prepack.conv_weight, c_stream, error, error_capacity);
  auto prior_delta = mlx::core::zeros(
      {1, qwen_linear_value_heads, qwen_linear_value_dim,
       qwen_linear_key_dim},
      mlx::core::float32, stream);
  auto gdn = qwen_gated_delta_decode(
      conv_out, mixed_ba, model.tensor(prefix + ".linear_attn.dt_bias"),
      prepack.a_exp, prior_delta, c_stream, error, error_capacity);

  auto normalized = qwen_rms_norm(
      gdn.output, model.tensor(prefix + ".linear_attn.norm.weight"), stream);
  auto normalized_f32 =
      mlx::core::astype(normalized, mlx::core::float32, stream);
  auto z_f32 = mlx::core::astype(z, mlx::core::float32, stream);
  auto gated_f32 =
      mlx::core::multiply(normalized_f32, qwen_silu(z_f32, stream), stream);
  auto gated = mlx::core::astype(gated_f32, gdn.output.dtype(), stream);
  gated = mlx::core::reshape(gated, {1, 1, qwen_linear_z_width}, stream);
  return qwen_quantized_by_name(
      model, prefix + ".linear_attn.out_proj.weight", gated, stream,
      experimental_fused, exact_global_scale);
}

mlx::core::array qwen_full_attention_zero_context(
    const generalist_mlx_qwen_model &model, int layer,
  const mlx::core::array &input, mlx::core::Stream stream,
    bool experimental_fused, bool exact_global_scale) {
  const std::string prefix = qwen_layer_prefix(layer);
  auto qg = qwen_quantized_by_name(
      model, prefix + ".self_attn.q_proj.weight", input, stream,
      experimental_fused, exact_global_scale);
  qg = mlx::core::reshape(
      qg, {1, 1, qwen_attention_heads, 2 * qwen_attention_head_dim}, stream);
  auto gate = mlx::core::slice(
      qg, {0, 0, 0, qwen_attention_head_dim},
      {1, 1, qwen_attention_heads, 2 * qwen_attention_head_dim}, stream);
  gate = mlx::core::reshape(
      gate, {1, 1, qwen_attention_heads * qwen_attention_head_dim}, stream);

  // With an empty KV cache there is exactly one causal key, so softmax is
  // exactly one. Repeating each KV head is the exact one-token SDPA result;
  // q/k normalization and position-zero RoPE cannot affect it.
  auto value = qwen_quantized_by_name(
      model, prefix + ".self_attn.v_proj.weight", input, stream,
      experimental_fused, exact_global_scale);
  value = mlx::core::reshape(
      value, {1, 1, qwen_attention_kv_heads, qwen_attention_head_dim}, stream);
  value = mlx::core::repeat(
      value, qwen_attention_heads / qwen_attention_kv_heads, 2, stream);
  value = mlx::core::reshape(
      value, {1, 1, qwen_attention_heads * qwen_attention_head_dim}, stream);
  auto attended =
      mlx::core::multiply(value, mlx::core::sigmoid(gate, stream), stream);
  return qwen_quantized_by_name(
      model, prefix + ".self_attn.o_proj.weight", attended, stream,
      experimental_fused, exact_global_scale);
}

struct qwen_linear_attention_step {
  mlx::core::array output;
  mlx::core::array conv_state;
  mlx::core::array delta_state;
  std::optional<mlx::core::array> rollback_conv_window;
  std::optional<mlx::core::array> rollback_gdn_input;
  std::optional<mlx::core::array> rollback_ba;
};

qwen_linear_attention_step qwen_linear_attention_cached(
    const generalist_mlx_qwen_model &model, int layer,
    const mlx::core::array &input, const mlx::core::array &prior_conv,
    const mlx::core::array &prior_delta, mlx_stream c_stream,
    mlx::core::Stream stream, char *error, size_t error_capacity,
    bool exact_global_scale, bool capture_history) {
  const std::string prefix = qwen_layer_prefix(layer);
  const qwen_linear_prepack &prepack = *model.linear_prepack[layer];
  const int tokens = input.shape(1);
  auto mixed_qkvz = qwen_nvfp4_linear(
      input, prepack.qkvz_weight, prepack.qkvz_scales,
      prepack.qkvz_global_scale, stream,
      exact_global_scale ? &prepack.qkvz_global_scale : nullptr,
      exact_global_scale);
  mlx::core::array mixed_ba = input;
  if (tokens == 1) {
    mixed_ba = mlx::core::matmul(input, prepack.ba_transposed, stream);
  } else if (tokens <= GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS) {
    // Speculative verification must reproduce the ordinary one-row reduction
    // tree exactly so accepted drafts cannot perturb future recurrent state.
    mixed_ba = qwen_exact_ba_projection(
        input, prepack.ba_weight, c_stream, error, error_capacity);
  } else {
    // Prompt prefill is mathematically equivalent rather than bit-identical to
    // token-at-a-time decode. Let MLX select its throughput-oriented GEMM.
    mixed_ba = mlx::core::matmul(input, prepack.ba_transposed, stream);
  }
  auto qkv = mlx::core::slice(mixed_qkvz, {0, 0, 0},
                              {1, tokens, qwen_linear_qkv_width}, stream);
  auto z = mlx::core::slice(
      mixed_qkvz, {0, 0, qwen_linear_qkv_width},
      {1, tokens, qwen_linear_qkv_width + qwen_linear_z_width}, stream);
  z = mlx::core::reshape(
      z, {1, tokens, qwen_linear_value_heads, qwen_linear_value_dim}, stream);

  auto conv_window =
      mlx::core::concatenate({prior_conv, qkv}, 1, stream);
  auto conv_out = qwen_depthwise_conv_silu(
      conv_window, prepack.conv_weight, c_stream, error, error_capacity);
  auto next_conv = mlx::core::slice(
      conv_window, {0, tokens, 0},
      {1, tokens + qwen_conv_kernel_width - 1, qwen_linear_qkv_width},
      stream);
  auto gdn = qwen_gated_delta_decode(
      conv_out, mixed_ba, model.tensor(prefix + ".linear_attn.dt_bias"),
      prepack.a_exp, prior_delta, c_stream, error, error_capacity);

  auto normalized = qwen_rms_norm(
      gdn.output, model.tensor(prefix + ".linear_attn.norm.weight"), stream);
  auto normalized_f32 =
      mlx::core::astype(normalized, mlx::core::float32, stream);
  auto z_f32 = mlx::core::astype(z, mlx::core::float32, stream);
  auto gated_f32 =
      mlx::core::multiply(normalized_f32, qwen_silu(z_f32, stream), stream);
  auto gated = mlx::core::astype(gated_f32, gdn.output.dtype(), stream);
  gated = mlx::core::reshape(gated, {1, tokens, qwen_linear_z_width}, stream);
  auto output = qwen_quantized_by_name(
      model, prefix + ".linear_attn.out_proj.weight", gated, stream, false,
      exact_global_scale);
  return {
      std::move(output),
      std::move(next_conv),
      std::move(gdn.state),
      capture_history ? std::optional<mlx::core::array>(std::move(conv_window))
                      : std::nullopt,
      capture_history ? std::optional<mlx::core::array>(std::move(conv_out))
                      : std::nullopt,
      capture_history ? std::optional<mlx::core::array>(std::move(mixed_ba))
                      : std::nullopt,
  };
}

struct qwen_full_attention_step {
  mlx::core::array output;
  mlx::core::array key_buffer;
  mlx::core::array value_buffer;
};

template <bool use_precomputed_norms>
qwen_full_attention_step qwen_full_attention_cached(
    const generalist_mlx_qwen_model &model, const std::string &prefix,
    const mlx::core::array &input,
    const std::optional<mlx::core::array> &prior_keys,
    const std::optional<mlx::core::array> &prior_values, size_t position,
    mlx::core::Stream stream, bool exact_global_scale) {
  const int tokens = input.shape(1);
  auto qg = qwen_quantized_by_name(
      model, prefix + ".self_attn.q_proj.weight", input, stream, false,
      exact_global_scale);
  qg = mlx::core::reshape(
      qg, {1, tokens, qwen_attention_heads, 2 * qwen_attention_head_dim},
      stream);
  auto query = mlx::core::slice(
      qg, {0, 0, 0, 0},
      {1, tokens, qwen_attention_heads, qwen_attention_head_dim}, stream);
  auto gate = mlx::core::slice(
      qg, {0, 0, 0, qwen_attention_head_dim},
      {1, tokens, qwen_attention_heads, 2 * qwen_attention_head_dim}, stream);
  gate = mlx::core::reshape(
      gate, {1, tokens, qwen_attention_heads * qwen_attention_head_dim},
      stream);

  auto key = qwen_quantized_by_name(
      model, prefix + ".self_attn.k_proj.weight", input, stream, false,
      exact_global_scale);
  auto value = qwen_quantized_by_name(
      model, prefix + ".self_attn.v_proj.weight", input, stream, false,
      exact_global_scale);
  key = mlx::core::reshape(
      key, {1, tokens, qwen_attention_kv_heads, qwen_attention_head_dim},
      stream);
  value = mlx::core::reshape(
      value, {1, tokens, qwen_attention_kv_heads, qwen_attention_head_dim},
      stream);
  query = qwen_shifted_rms_norm<use_precomputed_norms>(
      model, query, prefix + ".self_attn.q_norm.weight", stream);
  key = qwen_shifted_rms_norm<use_precomputed_norms>(
      model, key, prefix + ".self_attn.k_norm.weight", stream);
  query = mlx::core::transpose(query, {0, 2, 1, 3}, stream);
  key = mlx::core::transpose(key, {0, 2, 1, 3}, stream);
  value = mlx::core::transpose(value, {0, 2, 1, 3}, stream);

  auto offsets = mlx::core::array(
      {static_cast<int32_t>(position)}, mlx::core::int32);
  query = mlx::core::fast::rope(
      query, qwen_rope_dim, false, qwen_rope_theta, 1.0F, offsets,
      std::nullopt, stream);
  key = mlx::core::fast::rope(key, qwen_rope_dim, false, qwen_rope_theta,
                              1.0F, offsets, std::nullopt, stream);

  if (prior_keys.has_value() != prior_values.has_value()) {
    throw std::runtime_error("Qwen KV cache is only partially initialized");
  }
  constexpr int kv_step = 256;
  auto key_buffer = prior_keys.has_value()
                        ? *prior_keys
                        : mlx::core::zeros(
                              {1, qwen_attention_kv_heads, kv_step,
                               qwen_attention_head_dim},
                              key.dtype(), stream);
  auto value_buffer = prior_values.has_value()
                          ? *prior_values
                          : mlx::core::zeros(
                                {1, qwen_attention_kv_heads, kv_step,
                                 qwen_attention_head_dim},
                                value.dtype(), stream);
  const int next_position = static_cast<int>(position) + tokens;
  if (next_position > key_buffer.shape(2)) {
    const int extra_capacity =
        ((next_position - key_buffer.shape(2) + kv_step - 1) / kv_step) *
        kv_step;
    auto extra_keys = mlx::core::zeros(
        {1, qwen_attention_kv_heads, extra_capacity,
         qwen_attention_head_dim},
        key.dtype(), stream);
    auto extra_values = mlx::core::zeros(
        {1, qwen_attention_kv_heads, extra_capacity,
         qwen_attention_head_dim},
        value.dtype(), stream);
    key_buffer = mlx::core::concatenate(
        {key_buffer, std::move(extra_keys)}, 2, stream);
    value_buffer = mlx::core::concatenate(
        {value_buffer, std::move(extra_values)}, 2, stream);
  }
  key_buffer = mlx::core::slice_update(
      key_buffer, key, {0, 0, static_cast<int>(position), 0},
      {1, qwen_attention_kv_heads, next_position,
       qwen_attention_head_dim},
      stream);
  value_buffer = mlx::core::slice_update(
      value_buffer, value, {0, 0, static_cast<int>(position), 0},
      {1, qwen_attention_kv_heads, next_position,
       qwen_attention_head_dim},
      stream);
  auto keys = mlx::core::slice(
      key_buffer, {0, 0, 0, 0},
      {1, qwen_attention_kv_heads, next_position,
       qwen_attention_head_dim},
      stream);
  auto values = mlx::core::slice(
      value_buffer, {0, 0, 0, 0},
      {1, qwen_attention_kv_heads, next_position,
       qwen_attention_head_dim},
      stream);
  auto attended = mlx::core::fast::scaled_dot_product_attention(
      query, keys, values, qwen_attention_scale,
      tokens > 1 ? "causal" : "", std::nullopt, std::nullopt, stream);
  attended = mlx::core::reshape(
      mlx::core::transpose(attended, {0, 2, 1, 3}, stream),
      {1, tokens, qwen_attention_heads * qwen_attention_head_dim}, stream);
  attended = mlx::core::multiply(
      attended, mlx::core::sigmoid(gate, stream), stream);
  auto output = qwen_quantized_by_name(
      model, prefix + ".self_attn.o_proj.weight", attended, stream, false,
      exact_global_scale);
  return {std::move(output), std::move(key_buffer),
          std::move(value_buffer)};
}

struct qwen_mtp_arrays {
  mlx::core::array hidden;
  mlx::core::array logits;
  int output_token_id;
  qwen_layer_cache next_layer;
};

qwen_mtp_arrays qwen_mtp_forward(
    generalist_mlx_qwen_model &model, const qwen_layer_cache &prior,
    int lookahead_token_id, const mlx::core::array &target_hidden,
    size_t position, mlx_stream gpu_stream, bool exact_global_scale,
    const qwen_mtp_draft_head_config &draft_head) {
  if (!model.has_mtp()) {
    throw std::runtime_error("Qwen MTP head is not loaded");
  }
  const auto &embedding_weight =
      model.tensor("model.language_model.embed_tokens.weight");
  if (lookahead_token_id < 0 ||
      lookahead_token_id >= embedding_weight.shape(0)) {
    throw std::runtime_error("Qwen MTP token is outside the vocabulary");
  }
  if (target_hidden.ndim() != 3 || target_hidden.shape(0) != 1 ||
      target_hidden.shape(1) != 1 ||
      target_hidden.shape(2) != qwen_hidden_width) {
    throw std::runtime_error("Qwen MTP target hidden geometry changed");
  }

  auto &stream = mlx_stream_get_(gpu_stream);
  auto raw = mlx::core::reshape(
      mlx::core::take(embedding_weight, lookahead_token_id, 0, stream),
      {1, 1, qwen_hidden_width}, stream);
  auto embedding = qwen_shifted_rms_norm<true>(
      model, raw, "mtp.pre_fc_norm_embedding.weight", stream);
  auto hidden = qwen_shifted_rms_norm<true>(
      model, target_hidden, "mtp.pre_fc_norm_hidden.weight", stream);
  auto fused_input =
      mlx::core::concatenate({embedding, hidden}, 2, stream);
  auto fused = qwen_quantized_by_name(model, "mtp.fc.weight", fused_input,
                                       stream, false, exact_global_scale);

  const std::string prefix = "mtp.layers.0";
  auto normalized = qwen_shifted_rms_norm<true>(
      model, fused, prefix + ".input_layernorm.weight", stream);
  auto attention = qwen_full_attention_cached<true>(
      model, prefix, normalized, prior.keys, prior.values, position, stream,
      exact_global_scale);
  auto after_attention = mlx::core::add(fused, attention.output, stream);
  auto post_normalized = qwen_shifted_rms_norm<true>(
      model, after_attention,
      prefix + ".post_attention_layernorm.weight", stream);
  auto mlp = qwen_mtp_dense_mlp(model, post_normalized, stream,
                                exact_global_scale);
  auto layer_output = mlx::core::add(after_attention, mlp, stream);
  auto draft_hidden = qwen_shifted_rms_norm<true>(
      model, layer_output, "mtp.norm.weight", stream);
  auto sampled = qwen_draft_lm_head(
      model, draft_hidden, stream, exact_global_scale, draft_head);
  const int output_token_id = qwen_draft_token_id(
      sampled.top.item<uint32_t>(), draft_head);

  qwen_layer_cache next;
  next.keys = std::move(attention.key_buffer);
  next.values = std::move(attention.value_buffer);
  return {std::move(draft_hidden), std::move(sampled.logits), output_token_id,
          std::move(next)};
}

struct qwen_mtp_batch_arrays {
  std::optional<mlx::core::array> hidden;
  std::optional<int> output_token_id;
  qwen_layer_cache next_layer;
};

qwen_mtp_batch_arrays qwen_mtp_batch_forward(
    generalist_mlx_qwen_model &model, const qwen_layer_cache &prior,
    const std::vector<int32_t> &lookahead_token_ids,
    const mlx::core::array &target_hiddens, size_t position,
    mlx_stream gpu_stream, bool exact_global_scale, bool sample_output,
    const qwen_mtp_draft_head_config &draft_head) {
  if (!model.has_mtp()) {
    throw std::runtime_error("Qwen MTP head is not loaded");
  }
  if (lookahead_token_ids.empty() || lookahead_token_ids.size() > 8) {
    throw std::runtime_error("Qwen MTP batch must contain 1 to 8 tokens");
  }
  const int tokens = static_cast<int>(lookahead_token_ids.size());
  if (target_hiddens.ndim() != 3 || target_hiddens.shape(0) != 1 ||
      target_hiddens.shape(1) != tokens ||
      target_hiddens.shape(2) != qwen_hidden_width) {
    throw std::runtime_error("Qwen MTP batch hidden geometry changed");
  }
  const auto &embedding_weight =
      model.tensor("model.language_model.embed_tokens.weight");
  for (int32_t token_id : lookahead_token_ids) {
    if (token_id < 0 || token_id >= embedding_weight.shape(0)) {
      throw std::runtime_error("Qwen MTP batch token is outside the vocabulary");
    }
  }

  auto &stream = mlx_stream_get_(gpu_stream);
  auto indices =
      mlx::core::array(lookahead_token_ids.begin(), {1, tokens});
  auto raw = mlx::core::take(embedding_weight, indices, 0, stream);
  auto embedding = qwen_shifted_rms_norm<true>(
      model, raw, "mtp.pre_fc_norm_embedding.weight", stream);
  auto hidden = qwen_shifted_rms_norm<true>(
      model, target_hiddens, "mtp.pre_fc_norm_hidden.weight", stream);
  auto fused_input =
      mlx::core::concatenate({embedding, hidden}, 2, stream);
  auto fused = qwen_quantized_by_name(model, "mtp.fc.weight", fused_input,
                                       stream, false, exact_global_scale);

  const std::string prefix = "mtp.layers.0";
  auto normalized = qwen_shifted_rms_norm<true>(
      model, fused, prefix + ".input_layernorm.weight", stream);
  auto attention = qwen_full_attention_cached<true>(
      model, prefix, normalized, prior.keys, prior.values, position, stream,
      exact_global_scale);
  qwen_layer_cache next;
  next.keys = std::move(attention.key_buffer);
  next.values = std::move(attention.value_buffer);
  if (!sample_output) {
    // Known prompt tokens need only the one-layer MTP attention history.
    // Query/output projection, MLP, final norm, and LM head do not contribute
    // to a future token's K/V cache and their prediction would be discarded.
    mlx::core::eval(
        std::vector<mlx::core::array>{*next.keys, *next.values});
    return {std::nullopt, std::nullopt, std::move(next)};
  }

  const int last = tokens - 1;
  auto sampled_fused = mlx::core::slice(
      fused, {0, last, 0}, {1, tokens, qwen_hidden_width}, stream);
  auto sampled_attention = mlx::core::slice(
      attention.output, {0, last, 0}, {1, tokens, qwen_hidden_width}, stream);
  auto after_attention =
      mlx::core::add(sampled_fused, sampled_attention, stream);
  auto post_normalized = qwen_shifted_rms_norm<true>(
      model, after_attention,
      prefix + ".post_attention_layernorm.weight", stream);
  auto mlp = qwen_mtp_dense_mlp(model, post_normalized, stream,
                                exact_global_scale);
  auto layer_output = mlx::core::add(after_attention, mlp, stream);
  auto draft_hidden = qwen_shifted_rms_norm<true>(
      model, layer_output, "mtp.norm.weight", stream);
  auto sampled = qwen_draft_lm_head(
      model, draft_hidden, stream, exact_global_scale, draft_head);
  mlx::core::eval(std::vector<mlx::core::array>{
      sampled.top, draft_hidden, *next.keys, *next.values});
  const int output_token_id = qwen_draft_token_id(
      sampled.top.data<uint32_t>()[0], draft_head);
  return {std::move(draft_hidden), output_token_id, std::move(next)};
}

template <bool use_precomputed_norms, bool use_packed_gate_up>
std::pair<mlx::core::array, qwen_layer_cache> qwen_layer_cached_forward(
    const generalist_mlx_qwen_model &model, int layer,
    const mlx::core::array &input, const qwen_layer_cache &prior,
    size_t position, mlx_stream c_stream, mlx::core::Stream stream,
    char *error, size_t error_capacity, bool exact_global_scale,
    bool capture_rollback_inputs = false) {
  const std::string prefix = qwen_layer_prefix(layer);
  auto normalized = qwen_shifted_rms_norm<use_precomputed_norms>(
      model, input, prefix + ".input_layernorm.weight", stream);
  qwen_layer_cache next;
  mlx::core::array attention = input;
  if (qwen_layer_is_linear(layer)) {
    auto prior_conv = prior.conv_state.has_value()
                          ? *prior.conv_state
                          : mlx::core::zeros(
                                {1, qwen_conv_kernel_width - 1,
                                 qwen_linear_qkv_width},
                                input.dtype(), stream);
    auto prior_delta = prior.delta_state.has_value()
                           ? *prior.delta_state
                           : mlx::core::zeros(
                                 {1, qwen_linear_value_heads,
                                  qwen_linear_value_dim, qwen_linear_key_dim},
                                 mlx::core::float32, stream);
    auto step = qwen_linear_attention_cached(
        model, layer, normalized, prior_conv, prior_delta, c_stream, stream,
        error, error_capacity, exact_global_scale,
        capture_rollback_inputs);
    attention = std::move(step.output);
    next.conv_state = std::move(step.conv_state);
    next.delta_state = std::move(step.delta_state);
    next.rollback_conv_window = std::move(step.rollback_conv_window);
    next.rollback_gdn_input = std::move(step.rollback_gdn_input);
    next.rollback_ba = std::move(step.rollback_ba);
  } else {
    auto step = qwen_full_attention_cached<use_precomputed_norms>(
        model, prefix, normalized, prior.keys, prior.values, position, stream,
        exact_global_scale);
    attention = std::move(step.output);
    next.keys = std::move(step.key_buffer);
    next.values = std::move(step.value_buffer);
  }
  auto after_attention = mlx::core::add(input, attention, stream);
  auto post_normalized = qwen_shifted_rms_norm<use_precomputed_norms>(
      model, after_attention, prefix + ".post_attention_layernorm.weight",
      stream);
  auto mlp = qwen_dense_mlp<use_packed_gate_up>(
      model, layer, post_normalized, stream, false, exact_global_scale);
  auto output = mlx::core::add(after_attention, mlp, stream);
  return {std::move(output), std::move(next)};
}

mlx::core::array qwen_layer_zero_state_forward(
    const generalist_mlx_qwen_model &model, int layer,
    const mlx::core::array &input, mlx_stream c_stream,
    mlx::core::Stream stream, char *error, size_t error_capacity,
    bool experimental_fused, bool exact_global_scale = false) {
  const std::string prefix = qwen_layer_prefix(layer);
  auto normalized = qwen_rms_norm(
      input, model.shifted_norm_weight(prefix + ".input_layernorm.weight"),
      stream);
  auto attention = qwen_layer_is_linear(layer)
                       ? qwen_linear_attention_zero_state(
                             model, layer, normalized, c_stream, stream, error,
                             error_capacity, experimental_fused,
                             exact_global_scale)
                       : qwen_full_attention_zero_context(model, layer,
                                                          normalized, stream,
                                                          experimental_fused,
                                                          exact_global_scale);
  auto after_attention = mlx::core::add(input, attention, stream);
  auto post_normalized = qwen_rms_norm(
      after_attention,
      model.shifted_norm_weight(prefix + ".post_attention_layernorm.weight"),
      stream);
  auto mlp = qwen_dense_mlp<true>(model, layer, post_normalized, stream,
                                  experimental_fused, exact_global_scale);
  return mlx::core::add(after_attention, mlp, stream);
}

bool summarize_core_array(const mlx::core::array &value, mlx_stream stream,
                          generalist_mlx_array_summary *summary, char *error,
                          size_t error_capacity) {
  mlx_array handle = mlx_array_new_(value);
  const bool success =
      summarize_array(handle, stream, summary, error, error_capacity);
  mlx_array_free(handle);
  return success;
}

using qwen_profile_clock = std::chrono::steady_clock;

double qwen_elapsed_milliseconds(qwen_profile_clock::time_point started) {
  return std::chrono::duration<double, std::milli>(qwen_profile_clock::now() -
                                                   started)
      .count();
}

struct qwen_zero_state_arrays {
  mlx::core::array hidden;
  mlx::core::array logits;
  int output_token_id;
};

struct qwen_cached_batch_arrays {
  mlx::core::array hidden;
  mlx::core::array logits;
  std::vector<int> output_token_ids;
};

qwen_zero_state_arrays qwen_zero_state_forward(
    generalist_mlx_qwen_model &model, int token_id, mlx_stream gpu_stream,
    char *error, size_t error_capacity, double *layer_milliseconds = nullptr,
    double *final_norm_milliseconds = nullptr,
    double *lm_head_argmax_milliseconds = nullptr,
    double *graph_milliseconds = nullptr,
    bool experimental_fused = false, bool exact_global_scale = false) {
  const auto graph_started = qwen_profile_clock::now();
  const auto &embedding_weight =
      model.tensor("model.language_model.embed_tokens.weight");
  if (token_id >= embedding_weight.shape(0)) {
    throw std::runtime_error("Qwen decode token is outside the vocabulary");
  }
  auto &stream = mlx_stream_get_(gpu_stream);
  auto hidden = mlx::core::reshape(
      mlx::core::take(embedding_weight, token_id, 0, stream),
      {1, 1, qwen_hidden_width}, stream);
  for (int layer = 0; layer < qwen_layer_count; ++layer) {
    const auto layer_started = qwen_profile_clock::now();
    hidden = qwen_layer_zero_state_forward(model, layer, hidden, gpu_stream,
                                            stream, error, error_capacity,
                                            experimental_fused,
                                            exact_global_scale);
    if (layer_milliseconds != nullptr) {
      mlx::core::eval(hidden);
      layer_milliseconds[layer] =
          qwen_elapsed_milliseconds(layer_started);
    }
  }

  const auto final_norm_started = qwen_profile_clock::now();
  hidden = qwen_rms_norm(
      hidden, model.shifted_norm_weight("model.language_model.norm.weight"),
      stream);
  if (final_norm_milliseconds != nullptr) {
    mlx::core::eval(hidden);
    *final_norm_milliseconds =
        qwen_elapsed_milliseconds(final_norm_started);
  }

  const auto lm_head_started = qwen_profile_clock::now();
  auto logits = qwen_quantized_by_name(model, "lm_head.weight", hidden,
                                        stream, experimental_fused,
                                        exact_global_scale);
  auto top = mlx::core::argmax(logits, -1, false, stream);
  const int output_token_id = static_cast<int>(top.item<uint32_t>());
  if (lm_head_argmax_milliseconds != nullptr) {
    *lm_head_argmax_milliseconds =
        qwen_elapsed_milliseconds(lm_head_started);
  }
  if (graph_milliseconds != nullptr) {
    *graph_milliseconds = qwen_elapsed_milliseconds(graph_started);
  }
  return {std::move(hidden), std::move(logits), output_token_id};
}

qwen_cached_batch_arrays qwen_cached_batch_forward(
    generalist_mlx_qwen_model &model, generalist_mlx_qwen_cache &cache,
    const std::vector<int32_t> &token_ids, mlx_stream gpu_stream,
    char *error, size_t error_capacity, bool exact_global_scale,
    bool all_output_logits, bool produce_output_logits,
    bool capture_rollback_inputs = false) {
  if (token_ids.empty() || token_ids.size() > qwen_prefill_batch_tokens) {
    throw std::runtime_error("Qwen cached batch must contain 1 to 512 tokens");
  }
  const auto &embedding_weight =
      model.tensor("model.language_model.embed_tokens.weight");
  for (int32_t token_id : token_ids) {
    if (token_id < 0 || token_id >= embedding_weight.shape(0)) {
      throw std::runtime_error("Qwen cached batch token is outside the vocabulary");
    }
  }

  auto &stream = mlx_stream_get_(gpu_stream);
  const int tokens = static_cast<int>(token_ids.size());
  auto indices = mlx::core::array(token_ids.begin(), {1, tokens});
  auto hidden = mlx::core::take(embedding_weight, indices, 0, stream);
  std::array<qwen_layer_cache, qwen_layer_count> next_layers;
  for (int layer = 0; layer < qwen_layer_count; ++layer) {
    auto step = qwen_layer_cached_forward<true, true>(
        model, layer, hidden, cache.layers[layer], cache.position, gpu_stream,
        stream, error, error_capacity, exact_global_scale,
        capture_rollback_inputs);
    hidden = std::move(step.first);
    next_layers[layer] = std::move(step.second);
  }

  hidden = qwen_shifted_rms_norm<true>(
      model, hidden, "model.language_model.norm.weight", stream);
  if (!produce_output_logits) {
    if (all_output_logits) {
      throw std::runtime_error(
          "Qwen cannot request every output while skipping prompt logits");
    }
    std::vector<mlx::core::array> evaluation;
    evaluation.reserve(1 + 5 * qwen_layer_count);
    evaluation.push_back(hidden);
    for (const auto &layer : next_layers) {
      if (layer.conv_state.has_value()) {
        evaluation.push_back(*layer.conv_state);
        evaluation.push_back(*layer.delta_state);
        if (layer.rollback_conv_window.has_value()) {
          evaluation.push_back(*layer.rollback_conv_window);
          evaluation.push_back(*layer.rollback_gdn_input);
          evaluation.push_back(*layer.rollback_ba);
        }
      } else {
        evaluation.push_back(*layer.keys);
        evaluation.push_back(*layer.values);
      }
    }
    mlx::core::eval(std::move(evaluation));
    cache.layers = std::move(next_layers);
    cache.position += token_ids.size();
    // The placeholder is intentionally unobservable: state-only callers use
    // the hidden batch and cache roots, never logits or output token IDs.
    return {hidden, hidden, {}};
  }
  auto sampled_hidden = hidden;
  if (!all_output_logits) {
    sampled_hidden = mlx::core::slice(
        hidden, {0, tokens - 1, 0}, {1, tokens, qwen_hidden_width}, stream);
  }
  auto logits = qwen_quantized_by_name(model, "lm_head.weight", sampled_hidden,
                                        stream, false, exact_global_scale);
  auto top = mlx::core::argmax(logits, -1, false, stream);

  std::vector<mlx::core::array> evaluation;
  evaluation.reserve(2 + 5 * qwen_layer_count);
  evaluation.push_back(top);
  // Speculative prefill consumes every normalized target hidden after this
  // call. Root the complete batch explicitly when logits sample only its last
  // row so MLX cannot retire the rest of the graph at the eval boundary.
  evaluation.push_back(hidden);
  for (const auto &layer : next_layers) {
    if (layer.conv_state.has_value()) {
      evaluation.push_back(*layer.conv_state);
      evaluation.push_back(*layer.delta_state);
      if (layer.rollback_conv_window.has_value()) {
        evaluation.push_back(*layer.rollback_conv_window);
        evaluation.push_back(*layer.rollback_gdn_input);
        evaluation.push_back(*layer.rollback_ba);
      }
    } else {
      evaluation.push_back(*layer.keys);
      evaluation.push_back(*layer.values);
    }
  }
  mlx::core::eval(std::move(evaluation));

  const size_t output_count = all_output_logits ? token_ids.size() : 1;
  std::vector<int> output_token_ids(output_count);
  const auto *top_data = top.data<uint32_t>();
  for (size_t index = 0; index < output_count; ++index) {
    output_token_ids[index] = static_cast<int>(top_data[index]);
  }
  cache.layers = std::move(next_layers);
  cache.position += token_ids.size();
  return {std::move(hidden), std::move(logits),
          std::move(output_token_ids)};
}

void qwen_clear_rollback_inputs(generalist_mlx_qwen_cache &cache) {
  for (auto &layer : cache.layers) {
    layer.rollback_conv_window.reset();
    layer.rollback_gdn_input.reset();
    layer.rollback_ba.reset();
  }
}

void qwen_commit_verified_prefix(
    const generalist_mlx_qwen_model &model,
    generalist_mlx_qwen_cache &cache,
    const generalist_mlx_qwen_cache &original_cache,
    size_t original_position, size_t verified_tokens,
    size_t committed_tokens, mlx_stream c_stream,
    mlx::core::Stream stream, char *error, size_t error_capacity) {
  if (committed_tokens == 0 || committed_tokens > verified_tokens ||
      cache.position != original_position + verified_tokens ||
      original_cache.position != original_position) {
    throw std::runtime_error(
        "Qwen verification rollback received an invalid prefix");
  }

  std::vector<mlx::core::array> compact_states;
  compact_states.reserve(2 * 48);
  for (int layer_index = 0; layer_index < qwen_layer_count; ++layer_index) {
    auto &layer = cache.layers[layer_index];
    if (!qwen_layer_is_linear(layer_index)) {
      continue;
    }
    if (!layer.rollback_conv_window.has_value() ||
        !layer.rollback_gdn_input.has_value() ||
        !layer.rollback_ba.has_value()) {
      throw std::runtime_error(
          "Qwen verification rollback is missing recurrent inputs");
    }
    const auto &conv_window = *layer.rollback_conv_window;
    const auto &gdn_input = *layer.rollback_gdn_input;
    const auto &ba = *layer.rollback_ba;
    if (conv_window.ndim() != 3 || conv_window.shape(0) != 1 ||
        conv_window.shape(1) !=
            static_cast<int>(verified_tokens) + qwen_conv_kernel_width - 1 ||
        conv_window.shape(2) != qwen_linear_qkv_width ||
        gdn_input.ndim() != 3 || gdn_input.shape(0) != 1 ||
        gdn_input.shape(1) != static_cast<int>(verified_tokens) ||
        gdn_input.shape(2) != qwen_linear_qkv_width || ba.ndim() != 3 ||
        ba.shape(0) != 1 ||
        ba.shape(1) != static_cast<int>(verified_tokens) ||
        ba.shape(2) != qwen_linear_ba_width) {
      throw std::runtime_error(
          "Qwen verification rollback input geometry changed");
    }

    const int committed = static_cast<int>(committed_tokens);
    layer.conv_state = mlx::core::slice(
        conv_window, {0, committed, 0},
        {1, committed + qwen_conv_kernel_width - 1,
         qwen_linear_qkv_width},
        stream);
    auto committed_gdn_input = mlx::core::slice(
        gdn_input, {0, 0, 0},
        {1, committed, qwen_linear_qkv_width}, stream);
    auto committed_ba = mlx::core::slice(
        ba, {0, 0, 0}, {1, committed, qwen_linear_ba_width}, stream);
    const auto &original_layer = original_cache.layers[layer_index];
    auto prior_delta = original_layer.delta_state.has_value()
                           ? *original_layer.delta_state
                           : mlx::core::zeros(
                                 {1, qwen_linear_value_heads,
                                  qwen_linear_value_dim,
                                  qwen_linear_key_dim},
                                 mlx::core::float32, stream);
    const std::string prefix = qwen_layer_prefix(layer_index);
    auto recovered = qwen_gated_delta_decode(
        committed_gdn_input, committed_ba,
        model.tensor(prefix + ".linear_attn.dt_bias"),
        model.linear_prepack[layer_index]->a_exp, prior_delta, c_stream,
        error, error_capacity);
    layer.delta_state = std::move(recovered.state);
    compact_states.push_back(*layer.conv_state);
    compact_states.push_back(*layer.delta_state);
  }

  // Materialize compact prefix states before releasing the verification-only
  // convolution and recurrent inputs they depend on.
  mlx::core::eval(std::move(compact_states));
  cache.position = original_position + committed_tokens;
  qwen_clear_rollback_inputs(cache);
}

template <bool use_precomputed_norms, bool use_packed_gate_up>
qwen_zero_state_arrays qwen_cached_forward(
    generalist_mlx_qwen_model &model, generalist_mlx_qwen_cache &cache,
    int token_id, mlx_stream gpu_stream, char *error,
    size_t error_capacity, qwen_state_schedule state_schedule,
    bool exact_global_scale = false) {
  const auto &embedding_weight =
      model.tensor("model.language_model.embed_tokens.weight");
  if (token_id >= embedding_weight.shape(0)) {
    throw std::runtime_error("Qwen decode token is outside the vocabulary");
  }
  auto &stream = mlx_stream_get_(gpu_stream);
  auto hidden = mlx::core::reshape(
      mlx::core::take(embedding_weight, token_id, 0, stream),
      {1, 1, qwen_hidden_width}, stream);
  std::array<qwen_layer_cache, qwen_layer_count> next_layers;
  for (int layer = 0; layer < qwen_layer_count; ++layer) {
    auto step =
        qwen_layer_cached_forward<use_precomputed_norms, use_packed_gate_up>(
        model, layer, hidden, cache.layers[layer], cache.position, gpu_stream,
        stream, error, error_capacity, exact_global_scale);
    hidden = std::move(step.first);
    next_layers[layer] = std::move(step.second);
  }

  hidden = qwen_shifted_rms_norm<use_precomputed_norms>(
      model, hidden, "model.language_model.norm.weight", stream);
  auto logits = qwen_quantized_by_name(model, "lm_head.weight", hidden,
                                        stream, false, exact_global_scale);
  auto top = mlx::core::argmax(logits, -1, false, stream);

  if (state_schedule != qwen_state_schedule::deferred) {
    // Recurrent and KV outputs branch away from the logits graph. Evaluate all
    // of them in the same synchronized transaction so a successful eager step
    // never leaves a cache containing deferred work or references to evictable
    // graph intermediates.
    std::vector<mlx::core::array> evaluation;
    evaluation.reserve(1 + 2 * qwen_layer_count);
    evaluation.push_back(top);
    for (const auto &layer : next_layers) {
      if (layer.conv_state.has_value()) {
        evaluation.push_back(*layer.conv_state);
        evaluation.push_back(*layer.delta_state);
      } else {
        evaluation.push_back(*layer.keys);
        evaluation.push_back(*layer.values);
      }
    }
    if (state_schedule == qwen_state_schedule::eager) {
      mlx::core::eval(std::move(evaluation));
    } else {
      mlx::core::async_eval(std::move(evaluation));
    }
  }
  // item() evaluates the token graph in deferred mode. The next decode then
  // consumes the retained lazy state roots, matching Ollama's pipelined cache
  // schedule while keeping the eager API available as the atomic reference.
  const int output_token_id = static_cast<int>(top.item<uint32_t>());

  cache.layers = std::move(next_layers);
  ++cache.position;
  return {std::move(hidden), std::move(logits), output_token_id};
}

bool qwen_summarize_decode(int token_id, const qwen_zero_state_arrays &arrays,
                           mlx_stream gpu_stream,
                           generalist_mlx_decode_result *result, char *error,
                           size_t error_capacity) {
  result->input_token_id = token_id;
  result->output_token_id = arrays.output_token_id;
  if (summarize_core_array(arrays.hidden, gpu_stream, &result->hidden, error,
                           error_capacity) &&
      summarize_core_array(arrays.logits, gpu_stream, &result->logits, error,
                           error_capacity)) {
    return true;
  }
  if (error != nullptr && error[0] == '\0') {
    set_error(error, error_capacity,
              "MLX failed to summarize Qwen decode checkpoint");
  }
  return false;
}

struct qwen_speculative_prefill_arrays {
  size_t position;
  int input_token_id;
  int next_token_id;
};

qwen_speculative_prefill_arrays qwen_speculative_prefill_chunk_forward(
    generalist_mlx_qwen_model &model,
    generalist_mlx_qwen_speculative_cache &cache,
    const std::vector<int32_t> &token_ids,
    std::optional<int32_t> known_next_token_id, mlx_stream gpu_stream,
    char *error, size_t error_capacity) {
  if (!model.has_mtp()) {
    throw std::runtime_error("Qwen MTP head is not loaded");
  }
  if (token_ids.empty() || token_ids.size() > qwen_prefill_batch_tokens ||
      token_ids.size() > qwen_context_length - cache.target.position) {
    throw std::runtime_error(
        "Qwen speculative prefill chunk has an invalid size");
  }
  if (cache.target.position != cache.draft_position) {
    throw std::runtime_error(
        "Qwen target and MTP caches differ before prefill append");
  }
  if (cache.seeded) {
    if (cache.current_token_id < 0 ||
        (cache.decode_ready &&
         (!cache.held_draft_hidden.has_value() ||
          cache.held_draft_token_id < 0))) {
      throw std::runtime_error(
          "Qwen speculative prefill cache is incompletely seeded");
    }
    if (token_ids.front() != cache.current_token_id) {
      throw std::runtime_error(
          "Qwen speculative prefill append does not match its held token");
    }
  } else if (cache.target.position != 0 || cache.draft_position != 0) {
    throw std::runtime_error(
        "Qwen unseeded speculative prefill cache is not empty");
  }

  auto target = qwen_cached_batch_forward(
      model, cache.target, token_ids, gpu_stream, error, error_capacity,
      false, false, !known_next_token_id.has_value());
  const int next_token_id = known_next_token_id.has_value()
                                ? *known_next_token_id
                                : target.output_token_ids.back();
  auto &stream = mlx_stream_get_(gpu_stream);

  size_t draft_offset = 0;
  while (draft_offset < token_ids.size()) {
    const size_t draft_count = std::min(
        token_ids.size() - draft_offset,
        static_cast<size_t>(
            GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS));
    std::vector<int32_t> lookahead;
    lookahead.reserve(draft_count);
    for (size_t index = 0; index < draft_count; ++index) {
      const size_t absolute = draft_offset + index;
      lookahead.push_back(absolute + 1 < token_ids.size()
                              ? token_ids[absolute + 1]
                              : next_token_id);
    }
    const int begin = static_cast<int>(draft_offset);
    const int end = static_cast<int>(draft_offset + draft_count);
    auto target_hiddens = mlx::core::slice(
        target.hidden, {0, begin, 0}, {1, end, qwen_hidden_width}, stream);
    const bool final_draft_batch =
        draft_offset + draft_count == token_ids.size();
    const bool sample_output =
        final_draft_batch && !known_next_token_id.has_value();
    auto draft = qwen_mtp_batch_forward(
        model, cache.draft, lookahead, target_hiddens,
        cache.draft_position, gpu_stream, true, sample_output,
        cache.draft_head);
    cache.draft = std::move(draft.next_layer);
    cache.draft_position += draft_count;

    if (sample_output) {
      cache.held_draft_hidden = std::move(draft.hidden);
      cache.held_draft_token_id = *draft.output_token_id;
    }
    draft_offset += draft_count;
  }

  if (cache.target.position != cache.draft_position) {
    throw std::runtime_error(
        "Qwen target and MTP caches diverged during prefill append");
  }
  cache.seeded = true;
  cache.decode_ready = !known_next_token_id.has_value();
  if (!cache.decode_ready) {
    cache.held_draft_hidden.reset();
    cache.held_draft_token_id = -1;
  }
  cache.current_token_id = next_token_id;
  return {cache.target.position - 1, token_ids.back(), next_token_id};
}

generalist_mlx_array_drift
qwen_measure_drift(const mlx::core::array &reference,
                   const mlx::core::array &candidate,
                   mlx::core::Stream stream) {
  if (reference.shape() != candidate.shape() || reference.size() == 0) {
    throw std::runtime_error("Qwen drift arrays have incompatible shapes");
  }
  auto reference_f32 =
      mlx::core::astype(reference, mlx::core::float32, stream);
  auto candidate_f32 =
      mlx::core::astype(candidate, mlx::core::float32, stream);
  auto difference =
      mlx::core::subtract(candidate_f32, reference_f32, stream);
  auto max_abs = mlx::core::max(mlx::core::abs(difference, stream), stream);
  auto mean_abs = mlx::core::mean(mlx::core::abs(difference, stream), stream);
  auto difference_squared_sum = mlx::core::sum(
      mlx::core::square(difference, stream), false, stream);
  auto reference_squared_sum = mlx::core::sum(
      mlx::core::square(reference_f32, stream), false, stream);
  auto candidate_squared_sum = mlx::core::sum(
      mlx::core::square(candidate_f32, stream), false, stream);
  auto cross_sum = mlx::core::sum(
      mlx::core::multiply(reference_f32, candidate_f32, stream), false,
      stream);
  mlx::core::eval(std::vector<mlx::core::array>{
      max_abs, mean_abs, difference_squared_sum, reference_squared_sum,
      candidate_squared_sum, cross_sum});

  const double difference_squared = difference_squared_sum.item<float>();
  const double reference_squared = reference_squared_sum.item<float>();
  const double candidate_squared = candidate_squared_sum.item<float>();
  generalist_mlx_array_drift drift{};
  drift.max_abs = max_abs.item<float>();
  drift.mean_abs = mean_abs.item<float>();
  drift.root_mean_square =
      std::sqrt(difference_squared / static_cast<double>(reference.size()));
  drift.relative_l2 = std::sqrt(difference_squared / reference_squared);
  drift.cosine_similarity = cross_sum.item<float>() /
                            std::sqrt(reference_squared * candidate_squared);
  return drift;
}

} // namespace

extern "C" int generalist_mlx_qwen_model_new(
    const generalist_mlx_model_blob_source *sources, size_t source_count,
    generalist_mlx_qwen_model **model,
    generalist_mlx_model_load_result *result, char *error,
    size_t error_capacity) {
  if (sources == nullptr || source_count == 0 || model == nullptr ||
      result == nullptr) {
    return fail(error, error_capacity,
                "Qwen model loader received an invalid argument");
  }
  *model = nullptr;
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream cpu_stream{nullptr};
  mlx_stream gpu_stream{nullptr};
  try {
    install_nonterminating_mlx_error_handler();
    cpu_stream = mlx_default_cpu_stream_new();
    gpu_stream = mlx_default_gpu_stream_new();
    auto candidate = std::make_unique<generalist_mlx_qwen_model>();
    for (size_t index = 0; index < source_count; ++index) {
      if (sources[index].path == nullptr || sources[index].path[0] == '\0') {
        throw std::runtime_error("Qwen model blob path cannot be empty");
      }
      mlx_map_string_to_array arrays = mlx_map_string_to_array_new();
      mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
      if (mlx_load_safetensors(&arrays, &metadata, sources[index].path,
                               cpu_stream) != 0) {
        mlx_map_string_to_string_free(metadata);
        mlx_map_string_to_array_free(arrays);
        throw std::runtime_error(std::string("failed to load Qwen blob: ") +
                                 sources[index].path);
      }
      for (const auto &[name, value] : mlx_map_string_to_array_get_(arrays)) {
        candidate->tensors.insert_or_assign(name, value);
      }
      mlx_map_string_to_string_free(metadata);
      mlx_map_string_to_array_free(arrays);
    }
    const size_t loaded_array_count = candidate->tensors.size();
    candidate->prepare(mlx_stream_get_(gpu_stream));
    result->blob_count = source_count;
    result->array_count = loaded_array_count;
    result->layer_count = qwen_layer_count;
    result->linear_layer_count = 48;
    result->full_attention_layer_count = 16;
    result->mtp_layer_count = candidate->has_mtp() ? 1 : 0;
    *model = candidate.release();
    mlx_stream_free(gpu_stream);
    mlx_stream_free(cpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    mlx_stream_free(cpu_stream);
    return 1;
  } catch (...) {
    set_error(error, error_capacity,
              "Qwen model loader caught an unknown native exception");
    mlx_stream_free(gpu_stream);
    mlx_stream_free(cpu_stream);
    return 1;
  }
}

extern "C" void
generalist_mlx_qwen_model_free(generalist_mlx_qwen_model *model) {
  delete model;
}

extern "C" int generalist_mlx_qwen_cache_new(
    generalist_mlx_qwen_cache **cache, char *error, size_t error_capacity) {
  if (cache == nullptr) {
    return fail(error, error_capacity,
                "Qwen cache constructor received a null output");
  }
  *cache = nullptr;
  set_error(error, error_capacity, "");
  try {
    *cache = new generalist_mlx_qwen_cache();
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    return 1;
  }
}

extern "C" void
generalist_mlx_qwen_cache_free(generalist_mlx_qwen_cache *cache) {
  delete cache;
}

extern "C" int generalist_mlx_qwen_cache_reset(
    generalist_mlx_qwen_cache *cache, char *error, size_t error_capacity) {
  if (cache == nullptr) {
    return fail(error, error_capacity,
                "Qwen cache reset received a null cache");
  }
  set_error(error, error_capacity, "");
  cache->layers = {};
  cache->position = 0;
  return 0;
}

extern "C" int generalist_mlx_qwen_speculative_cache_new(
    generalist_mlx_qwen_speculative_cache **cache, char *error,
    size_t error_capacity) {
  if (cache == nullptr) {
    return fail(error, error_capacity,
                "Qwen speculative cache constructor received a null output");
  }
  *cache = nullptr;
  set_error(error, error_capacity, "");
  try {
    *cache = new generalist_mlx_qwen_speculative_cache();
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_speculative_cache_clone(
    const generalist_mlx_qwen_speculative_cache *source,
    generalist_mlx_qwen_speculative_cache **cache, char *error,
    size_t error_capacity) {
  if (source == nullptr || cache == nullptr) {
    return fail(error, error_capacity,
                "Qwen speculative cache clone received a null argument");
  }
  *cache = nullptr;
  set_error(error, error_capacity, "");
  try {
    // MLX arrays are immutable reference-counted values. The copy preserves an
    // exact cheap checkpoint while later forwards replace, rather than mutate,
    // every cache array.
    *cache = new generalist_mlx_qwen_speculative_cache(*source);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    return 1;
  }
}

extern "C" void generalist_mlx_qwen_speculative_cache_free(
    generalist_mlx_qwen_speculative_cache *cache) {
  delete cache;
}

extern "C" int generalist_mlx_qwen_speculative_cache_reset(
    generalist_mlx_qwen_speculative_cache *cache, char *error,
    size_t error_capacity) {
  if (cache == nullptr) {
    return fail(error, error_capacity,
                "Qwen speculative cache reset received a null cache");
  }
  set_error(error, error_capacity, "");
  *cache = generalist_mlx_qwen_speculative_cache{};
  return 0;
}

extern "C" int
generalist_mlx_qwen_speculative_cache_set_verification_geometry(
    generalist_mlx_qwen_speculative_cache *cache,
    int results_per_simdgroup, char *error, size_t error_capacity) {
  if (cache == nullptr || !qwen_valid_qmv_results(results_per_simdgroup)) {
    return fail(error, error_capacity,
                "Qwen verification geometry must be 1, 2, 4, or 8");
  }
  set_error(error, error_capacity, "");
  cache->verification_qmv_geometry = {results_per_simdgroup, 0, 0, 0};
  return 0;
}

extern "C" int
generalist_mlx_qwen_speculative_cache_set_verification_shape_override(
    generalist_mlx_qwen_speculative_cache *cache,
    int default_results_per_simdgroup, int output_width, int input_width,
    int override_results_per_simdgroup, char *error, size_t error_capacity) {
  if (cache == nullptr ||
      !qwen_valid_qmv_results(default_results_per_simdgroup) ||
      output_width <= 0 || input_width <= 0 ||
      !qwen_valid_qmv_results(override_results_per_simdgroup)) {
    return fail(
        error, error_capacity,
        "Qwen verification shape override requires positive widths and 1, 2, 4, or 8 results per SIMD-group");
  }
  set_error(error, error_capacity, "");
  cache->verification_qmv_geometry = {
      default_results_per_simdgroup, output_width, input_width,
      override_results_per_simdgroup};
  return 0;
}

extern "C" int
generalist_mlx_qwen_speculative_cache_set_verification_qmv_path(
    generalist_mlx_qwen_speculative_cache *cache, int path, char *error,
    size_t error_capacity) {
  if (cache == nullptr || path < 0 || path > 2) {
    return fail(error, error_capacity,
                "Qwen verification QMV path must be between zero and two");
  }
  set_error(error, error_capacity, "");
  cache->verification_qmv_path = path;
  return 0;
}

extern "C" int generalist_mlx_qwen_speculative_cache_set_mtp_draft_vocab_limit(
    generalist_mlx_qwen_speculative_cache *cache, size_t vocab_limit,
    char *error, size_t error_capacity) {
  if (cache == nullptr ||
      (vocab_limit != 0 && vocab_limit != qwen_draft_shortlist_size)) {
    return fail(error, error_capacity,
                "Qwen MTP draft vocabulary limit must be zero or 65536");
  }
  set_error(error, error_capacity, "");
  cache->draft_head.vocab_limit = vocab_limit;
  return 0;
}

extern "C" int
generalist_mlx_qwen_speculative_cache_set_mtp_draft_extra_tokens(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, const int *token_ids,
    size_t token_count, char *error, size_t error_capacity) {
  if (model == nullptr || cache == nullptr ||
      (token_count != 0 && token_ids == nullptr) ||
      token_count > qwen_draft_extra_token_limit) {
    return fail(error, error_capacity,
                "Qwen MTP draft extras received an invalid argument");
  }
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    std::vector<int32_t> selected;
    if (token_count != 0) {
      selected.assign(token_ids, token_ids + token_count);
    }
    for (size_t index = 0; index < selected.size(); ++index) {
      if (cache->draft_head.vocab_limit != qwen_draft_shortlist_size ||
          selected[index] < qwen_draft_shortlist_size ||
          selected[index] >= qwen_draft_special_start ||
          (index != 0 && selected[index - 1] >= selected[index])) {
        throw std::runtime_error(
            "Qwen MTP draft extras must be sorted unique ordinary token IDs outside the 65536-row core");
      }
    }
    if (selected.empty()) {
      cache->draft_head.extra_token_ids.clear();
      cache->draft_head.extra_weight.reset();
      cache->draft_head.extra_scales.reset();
      mlx_stream_free(gpu_stream);
      return 0;
    }

    auto &stream = mlx_stream_get_(gpu_stream);
    auto indices = mlx::core::array(selected.begin(),
                                    {static_cast<int>(selected.size())});
    auto extra_weight = mlx::core::take(
        model->tensor("lm_head.weight"), indices, 0, stream);
    auto extra_scales = mlx::core::take(
        model->tensor("lm_head.weight.scale"), indices, 0, stream);
    mlx::core::eval(std::vector<mlx::core::array>{extra_weight, extra_scales});
    cache->draft_head.extra_token_ids = std::move(selected);
    cache->draft_head.extra_weight = std::move(extra_weight);
    cache->draft_head.extra_scales = std::move(extra_scales);
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_speculative_cache_set_direct_cache_rollback(
    generalist_mlx_qwen_speculative_cache *cache, int enabled,
    char *error, size_t error_capacity) {
  if (cache == nullptr) {
    return fail(error, error_capacity,
                "Qwen direct cache rollback received a null cache");
  }
  set_error(error, error_capacity, "");
  cache->direct_cache_rollback = enabled != 0;
  return 0;
}

extern "C" int generalist_mlx_qwen_prefill(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    const int *token_ids, size_t token_count,
    generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || token_ids == nullptr ||
      result == nullptr || token_count == 0 ||
      token_count > qwen_context_length - cache->position) {
    return fail(error, error_capacity,
                "Qwen prefill received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    auto candidate = *cache;
    int output_token_id = -1;
    size_t offset = 0;
    while (offset < token_count) {
      const size_t count = std::min(
          token_count - offset, qwen_prefill_batch_tokens);
      std::vector<int32_t> chunk(token_ids + offset,
                                 token_ids + offset + count);
      auto step = qwen_cached_batch_forward(
          *model, candidate, chunk, gpu_stream, error, error_capacity, false,
          false, offset + count == token_count);
      if (offset + count == token_count) {
        output_token_id = step.output_token_ids.back();
      }
      offset += count;
    }

    const size_t position = cache->position + token_count - 1;
    *cache = std::move(candidate);
    result->position = position;
    result->input_token_id = token_ids[token_count - 1];
    result->output_token_id = output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_speculative_prefill(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, const int *token_ids,
    size_t token_count, generalist_mlx_speculative_step_result *result,
    char *error, size_t error_capacity) {
  if (model == nullptr || cache == nullptr || token_ids == nullptr ||
      result == nullptr || token_count == 0 ||
      token_count > qwen_context_length) {
    return fail(error, error_capacity,
                "Qwen speculative prefill received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    if (cache->seeded || cache->target.position != 0 ||
        cache->draft_position != 0) {
      throw std::runtime_error(
          "Qwen speculative cache must be reset before prefill");
    }

    generalist_mlx_qwen_speculative_cache candidate;
    candidate.verification_qmv_geometry = cache->verification_qmv_geometry;
    candidate.verification_qmv_path = cache->verification_qmv_path;
    candidate.draft_head = cache->draft_head;
    candidate.direct_cache_rollback = cache->direct_cache_rollback;
    qwen_speculative_prefill_arrays final{};
    size_t offset = 0;
    while (offset < token_count) {
      const size_t count = std::min(
          token_count - offset, qwen_prefill_batch_tokens);
      std::vector<int32_t> chunk(token_ids + offset,
                                 token_ids + offset + count);
      const std::optional<int32_t> known_next =
          offset + count < token_count
              ? std::optional<int32_t>(token_ids[offset + count])
              : std::nullopt;
      final = qwen_speculative_prefill_chunk_forward(
          *model, candidate, chunk, known_next, gpu_stream, error,
          error_capacity);
      offset += count;
    }

    *cache = std::move(candidate);

    result->position = final.position;
    result->input_token_id = final.input_token_id;
    result->output_count = 1;
    result->output_token_ids[0] = final.next_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_speculative_prefill_chunk(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, const int *token_ids,
    size_t token_count, int next_token_id,
    generalist_mlx_speculative_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || token_ids == nullptr ||
      result == nullptr || token_count == 0 ||
      token_count > qwen_prefill_batch_tokens || next_token_id < -1 ||
      token_count > qwen_context_length - cache->target.position) {
    return fail(error, error_capacity,
                "Qwen speculative prefill chunk received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    auto candidate = *cache;
    std::vector<int32_t> chunk(token_ids, token_ids + token_count);
    const std::optional<int32_t> known_next =
        next_token_id >= 0 ? std::optional<int32_t>(next_token_id)
                           : std::nullopt;
    const auto final = qwen_speculative_prefill_chunk_forward(
        *model, candidate, chunk, known_next, gpu_stream, error,
        error_capacity);
    *cache = std::move(candidate);

    result->position = final.position;
    result->input_token_id = final.input_token_id;
    result->output_count = 1;
    result->output_token_ids[0] = final.next_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_speculative_seed(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, int token_id,
    generalist_mlx_speculative_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen speculative seed received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    if (!model->has_mtp()) {
      throw std::runtime_error("Qwen MTP head is not loaded");
    }
    if (cache->seeded || cache->target.position != 0 ||
        cache->draft_position != 0) {
      throw std::runtime_error(
          "Qwen speculative cache must be reset before seeding");
    }
    auto target = qwen_cached_forward<true, true>(
        *model, cache->target, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    auto draft = qwen_mtp_forward(*model, cache->draft,
                                  target.output_token_id, target.hidden, 0,
                                  gpu_stream, true,
                                  cache->draft_head);
    cache->draft = std::move(draft.next_layer);
    cache->draft_position = 1;
    cache->seeded = true;
    cache->decode_ready = true;
    cache->current_token_id = target.output_token_id;
    cache->held_draft_hidden = std::move(draft.hidden);
    cache->held_draft_token_id = draft.output_token_id;

    result->position = 0;
    result->input_token_id = token_id;
    result->output_count = 1;
    result->output_token_ids[0] = target.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_speculative_step(
    generalist_mlx_qwen_model *model,
    generalist_mlx_qwen_speculative_cache *cache, size_t draft_depth,
    size_t max_output_tokens, generalist_mlx_speculative_step_result *result,
    char *error, size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      max_output_tokens == 0 ||
      max_output_tokens > GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS) {
    return fail(error, error_capacity,
                "Qwen speculative step received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    if (!model->has_mtp() || !cache->seeded || !cache->decode_ready ||
        !cache->held_draft_hidden.has_value() ||
        cache->held_draft_token_id < 0 || cache->current_token_id < 0) {
      throw std::runtime_error("Qwen speculative cache is not seeded");
    }
    if (cache->target.position != cache->draft_position) {
      throw std::runtime_error(
          "Qwen target and MTP caches have different positions");
    }

    const size_t position = cache->target.position;
    const int input_token_id = cache->current_token_id;
    const size_t depth =
        std::min({draft_depth, max_output_tokens - 1,
                  static_cast<size_t>(
                      GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS - 1)});
    std::vector<int32_t> emitted;
    size_t accepted = 0;

    if (depth == 0) {
      const auto verification_started = qwen_profile_clock::now();
      auto target = qwen_cached_forward<true, true>(
          *model, cache->target, input_token_id, gpu_stream, error,
          error_capacity, qwen_state_schedule::eager, true);
      result->verification_milliseconds =
          qwen_elapsed_milliseconds(verification_started);
      emitted.push_back(target.output_token_id);
      const auto correction_started = qwen_profile_clock::now();
      auto corrected = qwen_mtp_batch_forward(
          *model, cache->draft, emitted, target.hidden,
          cache->draft_position, gpu_stream, true, true,
          cache->draft_head);
      result->correction_milliseconds =
          qwen_elapsed_milliseconds(correction_started);
      cache->draft = std::move(corrected.next_layer);
      ++cache->draft_position;
      cache->held_draft_hidden = std::move(*corrected.hidden);
      cache->held_draft_token_id = *corrected.output_token_id;
    } else {
      std::vector<int32_t> drafts;
      drafts.reserve(depth);
      drafts.push_back(cache->held_draft_token_id);
      auto proposal_cache = cache->draft;
      auto proposal_hidden = *cache->held_draft_hidden;
      size_t proposal_position = cache->draft_position;
      const auto proposal_started = qwen_profile_clock::now();
      while (drafts.size() < depth) {
        auto proposal = qwen_mtp_forward(
            *model, proposal_cache, drafts.back(), proposal_hidden,
            proposal_position, gpu_stream, true,
            cache->draft_head);
        proposal_cache = std::move(proposal.next_layer);
        proposal_hidden = std::move(proposal.hidden);
        drafts.push_back(proposal.output_token_id);
        ++proposal_position;
      }
      result->proposal_milliseconds =
          qwen_elapsed_milliseconds(proposal_started);

      std::vector<int32_t> target_inputs;
      target_inputs.reserve(depth + 1);
      target_inputs.push_back(input_token_id);
      target_inputs.insert(target_inputs.end(), drafts.begin(), drafts.end());
      auto candidate_cache = cache->target;
      const auto verification_started = qwen_profile_clock::now();
      // The zero geometry selects Generalist's exact-scale qmv_wide kernel.
      // It retains the BF16 -> global-scale -> BF16 rounding contract while
      // reusing each NVFP4 weight group across verification rows. Its
      // reduction tree is mathematically equivalent, not bit-identical to
      // independent one-row QMV calls. The host keeps the independent path as
      // a differential oracle while production selects this weight-reusing
      // path after token, reasoning, tool, and cache validation.
      const qwen_qmv_geometry verification_geometry =
          cache->verification_qmv_path == 2
              ? qwen_qmv_geometry{0, 0, 0, 0}
              : cache->verification_qmv_geometry;
      const bool fused_verification_scale =
          cache->verification_qmv_path != 1;
      auto verified = [&]() {
        qwen_qmv_geometry_scope geometry(verification_geometry);
        return qwen_cached_batch_forward(
            *model, candidate_cache, target_inputs, gpu_stream, error,
            error_capacity, fused_verification_scale, true, true,
            cache->direct_cache_rollback);
      }();
      result->verification_milliseconds =
          qwen_elapsed_milliseconds(verification_started);
      while (accepted < depth &&
             drafts[accepted] == verified.output_token_ids[accepted]) {
        ++accepted;
      }
      emitted.insert(emitted.end(), drafts.begin(),
                     drafts.begin() + static_cast<ptrdiff_t>(accepted));
      emitted.push_back(verified.output_token_ids[accepted]);

      mlx::core::array committed_hidden = verified.hidden;
      if (accepted == depth) {
        qwen_clear_rollback_inputs(candidate_cache);
        cache->target = std::move(candidate_cache);
      } else {
        const auto replay_started = qwen_profile_clock::now();
        const size_t committed_tokens = accepted + 1;
        if (cache->direct_cache_rollback) {
          qwen_commit_verified_prefix(
              *model, candidate_cache, cache->target, position, depth + 1,
              committed_tokens, gpu_stream, mlx_stream_get_(gpu_stream),
              error, error_capacity);
          cache->target = std::move(candidate_cache);
        } else {
          target_inputs.resize(committed_tokens);
          auto committed = [&]() {
            qwen_qmv_geometry_scope geometry(verification_geometry);
            return qwen_cached_batch_forward(
                *model, cache->target, target_inputs, gpu_stream, error,
                error_capacity, fused_verification_scale, false, false);
          }();
          committed_hidden = std::move(committed.hidden);
        }
        result->replay_milliseconds =
            qwen_elapsed_milliseconds(replay_started);
        if (cache->direct_cache_rollback) {
          committed_hidden = mlx::core::slice(
              verified.hidden, {0, 0, 0},
              {1, static_cast<int>(committed_tokens), qwen_hidden_width},
              mlx_stream_get_(gpu_stream));
        }
      }

      const auto correction_started = qwen_profile_clock::now();
      auto corrected = qwen_mtp_batch_forward(
          *model, cache->draft, emitted, committed_hidden,
          cache->draft_position, gpu_stream, true, true,
          cache->draft_head);
      result->correction_milliseconds =
          qwen_elapsed_milliseconds(correction_started);
      cache->draft = std::move(corrected.next_layer);
      cache->draft_position += emitted.size();
      cache->held_draft_hidden = std::move(*corrected.hidden);
      cache->held_draft_token_id = *corrected.output_token_id;
    }

    if (cache->target.position != cache->draft_position) {
      throw std::runtime_error(
          "Qwen target and MTP caches diverged after speculative commit");
    }
    cache->current_token_id = emitted.back();
    result->position = position;
    result->input_token_id = input_token_id;
    result->output_count = emitted.size();
    result->drafted = depth;
    result->accepted = accepted;
    for (size_t index = 0; index < emitted.size(); ++index) {
      result->output_token_ids[index] = emitted[index];
    }
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_checkpoint(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_cached_decode_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen cached decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    result->position = position;
    if (!qwen_summarize_decode(token_id, arrays, gpu_stream, &result->decode,
                               error, error_capacity)) {
      mlx_stream_free(gpu_stream);
      return 1;
    }
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_unfused_scale_checkpoint(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_cached_decode_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(
        error, error_capacity,
        "Qwen unfused-scale cached decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, false);
    result->position = position;
    if (!qwen_summarize_decode(token_id, arrays, gpu_stream, &result->decode,
                               error, error_capacity)) {
      mlx_stream_free(gpu_stream);
      return 1;
    }
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_wide_qmv_checkpoint(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_cached_decode_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen wide-QMV cached decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    qwen_qmv_geometry_scope geometry(0);
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    result->position = position;
    if (!qwen_summarize_decode(token_id, arrays, gpu_stream, &result->decode,
                               error, error_capacity)) {
      mlx_stream_free(gpu_stream);
      return 1;
    }
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_argmax(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen cached argmax decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    result->position = position;
    result->input_token_id = token_id;
    result->output_token_id = arrays.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_argmax_unfused_scale(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(
        error, error_capacity,
        "Qwen unfused-scale cached argmax decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, false);
    result->position = position;
    result->input_token_id = token_id;
    result->output_token_id = arrays.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_argmax_wide_qmv(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(
        error, error_capacity,
        "Qwen wide-QMV cached argmax decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    qwen_qmv_geometry_scope geometry(0);
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    result->position = position;
    result->input_token_id = token_id;
    result->output_token_id = arrays.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_argmax_dynamic_norm(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen dynamic-norm argmax decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<false, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    result->position = position;
    result->input_token_id = token_id;
    result->output_token_id = arrays.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_argmax_split_mlp(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen split-MLP argmax decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<true, false>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    result->position = position;
    result->input_token_id = token_id;
    result->output_token_id = arrays.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_argmax_deferred_state(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen deferred-state argmax decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::deferred, true);
    result->position = position;
    result->input_token_id = token_id;
    result->output_token_id = arrays.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_cached_decode_argmax_async_state(
    generalist_mlx_qwen_model *model, generalist_mlx_qwen_cache *cache,
    int token_id, generalist_mlx_token_step_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || cache == nullptr || result == nullptr ||
      token_id < 0) {
    return fail(error, error_capacity,
                "Qwen async-state argmax decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const size_t position = cache->position;
  try {
    auto arrays = qwen_cached_forward<true, true>(
        *model, *cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::asynchronous, true);
    result->position = position;
    result->input_token_id = token_id;
    result->output_token_id = arrays.output_token_id;
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_mtp_checkpoint(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_mtp_checkpoint_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || result == nullptr || token_id < 0) {
    return fail(error, error_capacity,
                "Qwen MTP checkpoint received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    if (!model->has_mtp()) {
      throw std::runtime_error("Qwen MTP head is not loaded");
    }
    generalist_mlx_qwen_cache target_cache;
    qwen_layer_cache draft_cache;
    qwen_mtp_draft_head_config full_draft_head;
    auto first = qwen_cached_forward<true, true>(
        *model, target_cache, token_id, gpu_stream, error, error_capacity,
        qwen_state_schedule::eager, true);
    auto draft = qwen_mtp_forward(*model, draft_cache,
                                  first.output_token_id, first.hidden, 0,
                                  gpu_stream, true, full_draft_head);
    draft_cache = std::move(draft.next_layer);
    auto second = qwen_cached_forward<true, true>(
        *model, target_cache, first.output_token_id, gpu_stream, error,
        error_capacity, qwen_state_schedule::eager, true);

    result->input_token_id = token_id;
    result->lookahead_token_id = first.output_token_id;
    result->draft_token_id = draft.output_token_id;
    result->target_token_id = second.output_token_id;
    const bool summarized =
        summarize_core_array(first.hidden, gpu_stream, &result->target_hidden,
                             error, error_capacity) &&
        summarize_core_array(draft.hidden, gpu_stream, &result->draft_hidden,
                             error, error_capacity) &&
        summarize_core_array(draft.logits, gpu_stream, &result->draft_logits,
                             error, error_capacity) &&
        summarize_core_array(second.hidden, gpu_stream,
                             &result->next_target_hidden, error,
                             error_capacity) &&
        summarize_core_array(second.logits, gpu_stream,
                             &result->next_target_logits, error,
                             error_capacity);
    (void)draft_cache;
    mlx_stream_free(gpu_stream);
    if (!summarized) {
      if (error != nullptr && error[0] == '\0') {
        set_error(error, error_capacity,
                  "MLX failed to summarize Qwen MTP checkpoint");
      }
      return 1;
    }
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_batch_decode_checkpoint(
    generalist_mlx_qwen_model *model, const int *token_ids,
    size_t token_count, generalist_mlx_batch_decode_checkpoint_result *result,
    char *error, size_t error_capacity) {
  if (model == nullptr || token_ids == nullptr || result == nullptr ||
      token_count == 0 ||
      token_count > GENERALIST_MLX_QWEN_MAX_BATCH_CHECKPOINT_TOKENS) {
    return fail(error, error_capacity,
                "Qwen batch decode checkpoint received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    std::vector<int32_t> inputs(token_ids, token_ids + token_count);
    generalist_mlx_qwen_cache batch_cache;
    generalist_mlx_qwen_cache sequential_cache;
    auto batch = qwen_cached_batch_forward(
        *model, batch_cache, inputs, gpu_stream, error, error_capacity, true,
        true, true);
    std::vector<qwen_zero_state_arrays> sequential;
    sequential.reserve(token_count);
    for (int32_t token_id : inputs) {
      sequential.push_back(qwen_cached_forward<true, true>(
          *model, sequential_cache, token_id, gpu_stream, error,
          error_capacity, qwen_state_schedule::eager, true));
    }

    auto &stream = mlx_stream_get_(gpu_stream);
    result->token_count = token_count;
    for (size_t index = 0; index < token_count; ++index) {
      result->input_token_ids[index] = inputs[index];
      result->batch_output_token_ids[index] =
          batch.output_token_ids[index];
      result->sequential_output_token_ids[index] =
          sequential[index].output_token_id;
      const int row = static_cast<int>(index);
      auto batch_hidden = mlx::core::slice(
          batch.hidden, {0, row, 0},
          {1, row + 1, qwen_hidden_width}, stream);
      auto batch_logits = mlx::core::slice(
          batch.logits, {0, row, 0},
          {1, row + 1, batch.logits.shape(2)}, stream);
      if (!summarize_core_array(batch_hidden, gpu_stream,
                                &result->batch_hidden[index], error,
                                error_capacity) ||
          !summarize_core_array(batch_logits, gpu_stream,
                                &result->batch_logits[index], error,
                                error_capacity) ||
          !summarize_core_array(sequential[index].hidden, gpu_stream,
                                &result->sequential_hidden[index], error,
                                error_capacity) ||
          !summarize_core_array(sequential[index].logits, gpu_stream,
                                &result->sequential_logits[index], error,
                                error_capacity)) {
        mlx_stream_free(gpu_stream);
        return 1;
      }
    }
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_zero_state_decode_checkpoint(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || result == nullptr || token_id < 0) {
    return fail(error, error_capacity,
                "Qwen decode checkpoint received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    auto arrays = qwen_zero_state_forward(
        *model, token_id, gpu_stream, error, error_capacity, nullptr, nullptr,
        nullptr, nullptr, false, true);
    if (!qwen_summarize_decode(token_id, arrays, gpu_stream, result, error,
                               error_capacity)) {
      mlx_stream_free(gpu_stream);
      return 1;
    }
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int
generalist_mlx_qwen_zero_state_decode_experimental_fused_checkpoint(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || result == nullptr || token_id < 0) {
    return fail(error, error_capacity,
                "experimental fused Qwen decode received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    auto arrays = qwen_zero_state_forward(
        *model, token_id, gpu_stream, error, error_capacity, nullptr, nullptr,
        nullptr, nullptr, true);
    if (!qwen_summarize_decode(token_id, arrays, gpu_stream, result, error,
                               error_capacity)) {
      mlx_stream_free(gpu_stream);
      return 1;
    }
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_compare_zero_state_fused(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_comparison_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || result == nullptr || token_id < 0) {
    return fail(error, error_capacity,
                "Qwen fused comparison received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  try {
    auto exact = qwen_zero_state_forward(
        *model, token_id, gpu_stream, error, error_capacity, nullptr, nullptr,
        nullptr, nullptr, false, true);
    auto fused = qwen_zero_state_forward(
        *model, token_id, gpu_stream, error, error_capacity, nullptr, nullptr,
        nullptr, nullptr, true);
    if (!qwen_summarize_decode(token_id, exact, gpu_stream, &result->exact,
                               error, error_capacity) ||
        !qwen_summarize_decode(token_id, fused, gpu_stream,
                               &result->experimental_fused, error,
                               error_capacity)) {
      mlx_stream_free(gpu_stream);
      return 1;
    }
    auto &stream = mlx_stream_get_(gpu_stream);
    result->hidden_drift =
        qwen_measure_drift(exact.hidden, fused.hidden, stream);
    result->logits_drift =
        qwen_measure_drift(exact.logits, fused.logits, stream);
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}

extern "C" int generalist_mlx_qwen_zero_state_decode_profile(
    generalist_mlx_qwen_model *model, int token_id,
    generalist_mlx_decode_profile_result *result, char *error,
    size_t error_capacity) {
  if (model == nullptr || result == nullptr || token_id < 0) {
    return fail(error, error_capacity,
                "Qwen decode profile received an invalid argument");
  }
  *result = {};
  set_error(error, error_capacity, "");
  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  const auto checkpoint_started = qwen_profile_clock::now();
  try {
    auto arrays = qwen_zero_state_forward(
        *model, token_id, gpu_stream, error, error_capacity,
        result->layer_milliseconds, &result->final_norm_milliseconds,
        &result->lm_head_argmax_milliseconds, &result->graph_milliseconds,
        false, true);
    if (!qwen_summarize_decode(token_id, arrays, gpu_stream, &result->decode,
                               error, error_capacity)) {
      mlx_stream_free(gpu_stream);
      return 1;
    }
    result->checkpoint_milliseconds =
        qwen_elapsed_milliseconds(checkpoint_started);
    mlx_stream_free(gpu_stream);
    return 0;
  } catch (const std::exception &exception) {
    set_error(error, error_capacity, exception.what());
    mlx_stream_free(gpu_stream);
    return 1;
  }
}
