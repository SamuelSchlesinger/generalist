#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "mlx/c/mlx.h"

static void check(int status, const char *operation) {
  if (status != 0) {
    fprintf(stderr, "%s failed with MLX status %d\n", operation, status);
    exit(EXIT_FAILURE);
  }
}

int main(void) {
  const int qos_status =
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  if (qos_status != 0) {
    fprintf(stderr, "pthread_set_qos_class_self_np failed: %d\n", qos_status);
    return EXIT_FAILURE;
  }

  qos_class_t qos_class = QOS_CLASS_UNSPECIFIED;
  int relative_priority = 0;
  pthread_get_qos_class_np(pthread_self(), &qos_class, &relative_priority);

  mlx_string version = mlx_string_new();
  check(mlx_version(&version), "mlx_version");

  int gpu_count = 0;
  check(mlx_device_count(&gpu_count, MLX_GPU), "mlx_device_count");

  mlx_device device = mlx_device_new();
  check(mlx_get_default_device(&device), "mlx_get_default_device");
  mlx_string device_name = mlx_string_new();
  check(mlx_device_tostring(&device_name, device), "mlx_device_tostring");

  mlx_array left = mlx_array_new_float32(19.0f);
  mlx_array right = mlx_array_new_float32(23.0f);
  mlx_array sum = mlx_array_new();
  mlx_stream stream = mlx_default_gpu_stream_new();
  check(mlx_add(&sum, left, right, stream), "mlx_add");
  check(mlx_array_eval(sum), "mlx_array_eval");

  float value = 0.0f;
  check(mlx_array_item_float32(&value, sum), "mlx_array_item_float32");
  if (fabsf(value - 42.0f) > 0.0001f) {
    fprintf(stderr, "unexpected GPU result: %.3f\n", value);
    return EXIT_FAILURE;
  }

  printf("mlx_version=%s\n", mlx_string_data(version));
  printf("default_device=%s\n", mlx_string_data(device_name));
  printf("gpu_count=%d\n", gpu_count);
  printf("qos_class=%u relative_priority=%d\n", (unsigned)qos_class,
         relative_priority);
  printf("gpu_add=%.1f\n", value);

  check(mlx_stream_free(stream), "mlx_stream_free");
  check(mlx_array_free(sum), "mlx_array_free(sum)");
  check(mlx_array_free(right), "mlx_array_free(right)");
  check(mlx_array_free(left), "mlx_array_free(left)");
  check(mlx_string_free(device_name), "mlx_string_free(device_name)");
  check(mlx_device_free(device), "mlx_device_free");
  check(mlx_string_free(version), "mlx_string_free(version)");
  return EXIT_SUCCESS;
}
