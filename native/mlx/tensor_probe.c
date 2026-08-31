#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlx/c/mlx.h"

static void check(int status, const char *operation) {
  if (status != 0) {
    fprintf(stderr, "%s failed with MLX status %d\n", operation, status);
    exit(EXIT_FAILURE);
  }
}

static const char *dtype_name(mlx_dtype dtype) {
  switch (dtype) {
  case MLX_BOOL:
    return "bool";
  case MLX_UINT8:
    return "uint8";
  case MLX_UINT16:
    return "uint16";
  case MLX_UINT32:
    return "uint32";
  case MLX_UINT64:
    return "uint64";
  case MLX_INT8:
    return "int8";
  case MLX_INT16:
    return "int16";
  case MLX_INT32:
    return "int32";
  case MLX_INT64:
    return "int64";
  case MLX_FLOAT16:
    return "float16";
  case MLX_FLOAT32:
    return "float32";
  case MLX_FLOAT64:
    return "float64";
  case MLX_BFLOAT16:
    return "bfloat16";
  case MLX_COMPLEX64:
    return "complex64";
  }
  return "unknown";
}

static void print_shape(mlx_array value) {
  const size_t dimensions = mlx_array_ndim(value);
  const int *shape = mlx_array_shape(value);
  putchar('[');
  for (size_t index = 0; index < dimensions; ++index) {
    if (index != 0) {
      putchar(',');
    }
    printf("%d", shape[index]);
  }
  putchar(']');
}

static void evaluate_small_tensor_on_gpu(mlx_array value) {
  const size_t maximum_probe_bytes = 16U * 1024U * 1024U;
  if (mlx_array_nbytes(value) > maximum_probe_bytes) {
    printf("gpu_sum=skipped tensor_exceeds_%zu_bytes\n", maximum_probe_bytes);
    return;
  }

  mlx_stream gpu_stream = mlx_default_gpu_stream_new();
  mlx_array as_float = mlx_array_new();
  mlx_array sum = mlx_array_new();
  check(mlx_astype(&as_float, value, MLX_FLOAT32, gpu_stream), "mlx_astype");
  check(mlx_sum(&sum, as_float, false, gpu_stream), "mlx_sum");
  check(mlx_array_eval(sum), "mlx_array_eval(gpu sum)");

  float result = 0.0F;
  check(mlx_array_item_float32(&result, sum), "mlx_array_item_float32");
  printf("gpu_sum=%.9g\n", result);

  check(mlx_array_free(sum), "mlx_array_free(sum)");
  check(mlx_array_free(as_float), "mlx_array_free(as_float)");
  check(mlx_stream_free(gpu_stream), "mlx_stream_free(gpu)");
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s SAFETENSORS_FILE EXPECTED_TENSOR\n", argv[0]);
    return EXIT_FAILURE;
  }

  const int qos_status =
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  if (qos_status != 0) {
    fprintf(stderr, "pthread_set_qos_class_self_np failed: %d\n", qos_status);
    return EXIT_FAILURE;
  }

  mlx_stream cpu_stream = mlx_default_cpu_stream_new();
  mlx_map_string_to_array arrays = mlx_map_string_to_array_new();
  mlx_map_string_to_string metadata = mlx_map_string_to_string_new();
  check(mlx_load_safetensors(&arrays, &metadata, argv[1], cpu_stream),
        "mlx_load_safetensors");

  size_t metadata_count = 0;
  mlx_map_string_to_string_iterator metadata_iterator =
      mlx_map_string_to_string_iterator_new(metadata);
  for (;;) {
    const char *key = NULL;
    const char *value = NULL;
    const int status = mlx_map_string_to_string_iterator_next(
        &key, &value, metadata_iterator);
    if (status == 2) {
      break;
    }
    check(status, "mlx_map_string_to_string_iterator_next");
    printf("metadata[%s]=%s\n", key, value);
    ++metadata_count;
  }
  check(mlx_map_string_to_string_iterator_free(metadata_iterator),
        "mlx_map_string_to_string_iterator_free");

  bool expected_found = false;
  size_t tensor_count = 0;
  mlx_map_string_to_array_iterator iterator =
      mlx_map_string_to_array_iterator_new(arrays);
  for (;;) {
    const char *name = NULL;
    mlx_array value = mlx_array_new();
    const int status =
        mlx_map_string_to_array_iterator_next(&name, &value, iterator);
    if (status == 2) {
      check(mlx_array_free(value), "mlx_array_free(end sentinel)");
      break;
    }
    check(status, "mlx_map_string_to_array_iterator_next");

    printf("tensor[%zu]=%s dtype=%s shape=", tensor_count, name,
           dtype_name(mlx_array_dtype(value)));
    print_shape(value);
    printf(" elements=%zu bytes=%zu\n", mlx_array_size(value),
           mlx_array_nbytes(value));

    if (strcmp(name, argv[2]) == 0) {
      expected_found = true;
      evaluate_small_tensor_on_gpu(value);
    }
    ++tensor_count;
    check(mlx_array_free(value), "mlx_array_free(tensor)");
  }
  check(mlx_map_string_to_array_iterator_free(iterator),
        "mlx_map_string_to_array_iterator_free");

  printf("tensor_count=%zu metadata_count=%zu expected_found=%s\n",
         tensor_count, metadata_count, expected_found ? "true" : "false");

  check(mlx_map_string_to_string_free(metadata),
        "mlx_map_string_to_string_free");
  check(mlx_map_string_to_array_free(arrays),
        "mlx_map_string_to_array_free");
  check(mlx_stream_free(cpu_stream), "mlx_stream_free(cpu)");
  return expected_found ? EXIT_SUCCESS : EXIT_FAILURE;
}
