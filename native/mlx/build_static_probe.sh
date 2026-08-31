#!/bin/sh
set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
# PINS.env is resolved relative to this script, not the caller.
# shellcheck disable=SC1091
. "$script_dir/PINS.env"

if [ -n "${GENERALIST_MLX_WORK_ROOT:-}" ]; then
  work_root=$GENERALIST_MLX_WORK_ROOT
elif [ -n "${XDG_CACHE_HOME:-}" ]; then
  work_root=$XDG_CACHE_HOME/generalist/embedded-mlx
elif [ -n "${HOME:-}" ]; then
  work_root=$HOME/Library/Caches/generalist/embedded-mlx
else
  echo "HOME is unset; set GENERALIST_MLX_WORK_ROOT to a persistent build directory" >&2
  exit 1
fi

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "embedded MLX requires Apple Silicon macOS" >&2
  exit 1
fi

variant=${GENERALIST_MLX_VARIANT:-baseline}
mlx_source=${GENERALIST_MLX_SOURCE_DIR:-$work_root/mlx}
mlx_c_source=${GENERALIST_MLX_C_SOURCE_DIR:-$work_root/mlx-c}
build_dir=${GENERALIST_MLX_BUILD_DIR:-$work_root/build/$variant}
bundle_dir=${GENERALIST_MLX_BUNDLE_DIR:-$work_root/bundle/$variant}
cpu_flags=${GENERALIST_MLX_CPU_FLAGS:--O3 -DNDEBUG}
ipo=${GENERALIST_MLX_IPO:-OFF}
metal_debug=${GENERALIST_MLX_METAL_DEBUG:-OFF}
deployment_target=${GENERALIST_MLX_DEPLOYMENT_TARGET:-$(sw_vers -productVersion)}
build_jobs=${GENERALIST_MLX_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1\n')}

case $build_jobs in
  ''|*[!0-9]*|0)
    echo "GENERALIST_MLX_BUILD_JOBS must be a positive integer" >&2
    exit 1
    ;;
esac

mkdir -p \
  "$work_root" \
  "$(dirname -- "$mlx_source")" \
  "$(dirname -- "$mlx_c_source")" \
  "$(dirname -- "$build_dir")" \
  "$(dirname -- "$bundle_dir")"

if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
  echo "Metal Toolchain is missing." >&2
  echo "Install it with: xcodebuild -downloadComponent MetalToolchain" >&2
  exit 1
fi
if ! command -v shasum >/dev/null 2>&1; then
  echo "required command is missing: shasum" >&2
  exit 1
fi

checkout_exact() {
  repository=$1
  revision=$2
  destination=$3

  if ! git -C "$destination" rev-parse --git-dir >/dev/null 2>&1; then
    git clone --filter=blob:none "$repository" "$destination"
  fi
  if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$destination" fetch origin "$revision"
  fi
  current=$(git -C "$destination" rev-parse HEAD)
  if [ "$current" != "$revision" ]; then
    if [ -n "$(git -C "$destination" status --porcelain)" ]; then
      echo "refusing to replace a dirty source checkout: $destination" >&2
      exit 1
    fi
    git -C "$destination" checkout --detach "$revision"
  fi
}

checkout_exact https://github.com/ml-explore/mlx.git "$MLX_COMMIT" "$mlx_source"
checkout_exact https://github.com/ml-explore/mlx-c.git "$MLX_C_COMMIT" "$mlx_c_source"

apply_source_patch() {
  patch_path=$1
  description=$2

  if git -C "$mlx_source" apply --reverse --check "$patch_path" 2>/dev/null; then
    return # Already applied to this exact source checkout.
  fi
  if git -C "$mlx_source" apply --check "$patch_path"; then
    git -C "$mlx_source" apply "$patch_path"
    return
  fi
  echo "Generalist's $description patch does not apply to the pinned MLX source" >&2
  exit 1
}

fused_nvfp4_patch_present() {
  grep -Fq "fused global-scale decode requires Metal" \
    "$mlx_source/mlx/backend/cpu/quantized.cpp" &&
    grep -Fq "fused global-scale decode is inference-only" \
      "$mlx_source/mlx/primitives.cpp" &&
    grep -Fq "bool has_global_scale_;" \
      "$mlx_source/mlx/primitives.h" &&
    grep -Fq "fused decode requires" \
      "$mlx_source/mlx/backend/metal/quantized.cpp"
}

exact_nvfp4_patch_present() {
  grep -Fq "_hgs_exact" \
    "$mlx_source/mlx/backend/metal/kernels/fp_quantized.metal" &&
    grep -Fq "bool exact_global_scale_;" \
      "$mlx_source/mlx/primitives.h" &&
    grep -Fq "exact fused decode requires one global scale" \
      "$mlx_source/mlx/backend/metal/quantized.cpp"
}

configurable_qmv_patch_present() {
  grep -Fq "int qmv_results_per_simdgroup_;" \
    "$mlx_source/mlx/primitives.h" &&
    grep -Fq "qmv_results_per_simdgroup_ == qm_other.qmv_results_per_simdgroup_" \
      "$mlx_source/mlx/primitives.cpp" &&
    grep -Fq "qmv_results_per_simdgroup_" \
      "$mlx_source/mlx/backend/metal/quantized.cpp"
}

batched_qmv_patch_present() {
  grep -Fq "instantiate_quantized_qmv_fast_r_exact" \
    "$mlx_source/mlx/backend/metal/kernels/fp_quantized.metal" &&
    grep -Fq "QMV geometry override must use 1, 2, 4, or 8" \
      "$mlx_source/mlx/backend/metal/quantized.cpp"
}

exact_wide_qmv_patch_present() {
  grep -Fq "instantiate_quantized_all_wide_exact" \
    "$mlx_source/mlx/backend/metal/kernels/fp_quantized.metal" &&
  grep -Fq "exact_wide_global_scale" \
    "$mlx_source/mlx/backend/metal/quantized.cpp"
}

# The exact-scale patch extends three hunks from 0001, so once both are
# present a reverse-check of 0001 alone no longer has enough unchanged
# context. Stable sentinels across all four touched concerns distinguish that
# complete state from a partial patch.
if ! fused_nvfp4_patch_present; then
  apply_source_patch \
    "$script_dir/patches/0001-fused-nvfp4-decode-global-scale.patch" \
    "fused NVFP4"
fi
apply_source_patch \
  "$script_dir/patches/0002-metal-kernel-trace-labels.patch" \
  "Metal kernel tracing"
if ! exact_nvfp4_patch_present; then
  apply_source_patch \
    "$script_dir/patches/0003-exact-nvfp4-decode-global-scale.patch" \
    "exact fused NVFP4 global-scale"
fi
if ! configurable_qmv_patch_present; then
  apply_source_patch \
    "$script_dir/patches/0004-configurable-nvfp4-qmv-geometry.patch" \
    "configurable NVFP4 QMV geometry"
fi
if ! batched_qmv_patch_present; then
  apply_source_patch \
    "$script_dir/patches/0005-batched-nvfp4-qmv-geometry.patch" \
    "batched NVFP4 QMV geometry sweep"
fi
if ! exact_wide_qmv_patch_present; then
  apply_source_patch \
    "$script_dir/patches/0006-exact-nvfp4-qmv-wide-global-scale.patch" \
    "exact NVFP4 wide-QMV global scale"
fi

verify_source_contract() {
  observed_patchset=$(
    git \
      -c color.ui=false \
      -c diff.noprefix=false \
      -c diff.mnemonicPrefix=false \
      -C "$mlx_source" \
      diff --full-index --binary --no-ext-diff --no-color --no-renames HEAD -- |
      shasum -a 256 |
      awk '{print $1}'
  )
  if [ "$observed_patchset" != "$MLX_PATCHSET_SHA256" ]; then
    echo "refusing an MLX source cache that differs from the six promoted patches" >&2
    echo "expected patch-set SHA-256: $MLX_PATCHSET_SHA256" >&2
    echo "observed patch-set SHA-256: $observed_patchset" >&2
    echo "restore or remove the cached checkout: $mlx_source" >&2
    exit 1
  fi

  untracked_mlx=$(
    git -C "$mlx_source" status --porcelain --untracked-files=all |
      sed -n 's/^?? //p'
  )
  if [ -n "$untracked_mlx" ]; then
    echo "refusing untracked files in the pinned MLX source cache:" >&2
    printf '%s\n' "$untracked_mlx" >&2
    exit 1
  fi

  dirty_mlx_c=$(git -C "$mlx_c_source" status --porcelain --untracked-files=all)
  if [ -n "$dirty_mlx_c" ]; then
    echo "refusing a modified pinned MLX-C source cache:" >&2
    printf '%s\n' "$dirty_mlx_c" >&2
    exit 1
  fi
}

verify_source_contract
native_fingerprint=$(
  {
    printf '%s\n' \
      'generalist-native-build-schema=1' \
      "variant=$variant" \
      "mlx_source=$mlx_source" \
      "mlx_c_source=$mlx_c_source" \
      "cpu_flags=$cpu_flags" \
      "ipo=$ipo" \
      "metal_debug=$metal_debug" \
      "deployment_target=$deployment_target"
    cat "$script_dir/PINS.env" "$script_dir"/patches/*.patch
    git -C "$mlx_source" rev-parse HEAD
    git -C "$mlx_source" diff --binary HEAD --
    git -C "$mlx_source" status --porcelain --untracked-files=all
    git -C "$mlx_c_source" rev-parse HEAD
    git -C "$mlx_c_source" diff --binary HEAD --
    git -C "$mlx_c_source" status --porcelain --untracked-files=all
    cmake --version
    xcrun clang --version
    xcrun -sdk macosx metal --version
  } | shasum -a 256 | awk '{print $1}'
)

mlx_archive=$build_dir/_deps/mlx-build/libmlx.a
mlxc_archive=$build_dir/libmlxc.a
jaccl_archive=$build_dir/jaccl/libjaccl.a
metallib=$build_dir/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib
native_stamp=$build_dir/.generalist-native-fingerprint

write_stamp() {
  destination=$1
  value=$2
  temporary=$destination.tmp.$$
  printf '%s\n' "$value" > "$temporary"
  mv -f "$temporary" "$destination"
}

native_artifacts_ready() {
  [ -f "$mlx_archive" ] &&
    [ -f "$mlxc_archive" ] &&
    [ -f "$jaccl_archive" ] &&
    [ -f "$metallib" ]
}

require_license_text() {
  license_file=$1
  license_marker=$2
  if ! grep -Fq "$license_marker" "$license_file"; then
    echo "native license contract changed: $license_file lacks: $license_marker" >&2
    exit 1
  fi
}

verify_native_license_contract() {
  notice_file=$script_dir/THIRD_PARTY_NOTICES.md

  # Every copyright holder whose code is retained by the static executable or
  # the embedded metallib must remain visible through `generalist --licenses`.
  for license_marker in \
    'Copyright © 2023 Apple Inc.' \
    'Copyright (c) 2023 ml-explore' \
    'Copyright (c) Ollama' \
    'Copyright (c) 2013-2022 Niels Lohmann' \
    'Victor Zverovich and {fmt} contributors' \
    'Copyright (C) 2010-2022 Max-Planck-Society' \
    'Copyright (C) 2019-2020 Peter Bell' \
    'Copyright © 2018 the V8 project authors.' \
    'Copyright (c) 2012 Jakob Progsch' \
    'Copyright (c) 2015-2023 Norbert Juffa' \
    'Copyright 2020-2024 Apple Inc.' \
    'Copyright © 2025 Apple Inc.' \
    'Copyright © 2008-2013 NVIDIA Corporation' \
    'Copyright © 2013 Filipe RNC Maia'
  do
    require_license_text "$notice_file" "$license_marker"
  done

  # Freeze the relevant upstream attributions as part of the same contract as
  # the source commits. A pin update must consciously update these checks and
  # the user-visible notice instead of silently changing the binary closure.
  require_license_text "$mlx_source/LICENSE" '2023 Apple Inc.'
  require_license_text "$mlx_c_source/LICENSE" '2023 ml-explore'
  require_license_text \
    "$mlx_source/mlx/3rdparty/pocketfft.h" \
    'Copyright (C) 2010-2022 Max-Planck-Society'
  require_license_text \
    "$mlx_source/mlx/small_vector.h" \
    '2018 the V8 project authors.'
  require_license_text \
    "$mlx_source/mlx/threadpool.h" \
    'Copyright (c) 2012 Jakob Progsch'
  require_license_text \
    "$mlx_source/mlx/backend/metal/kernels/expm1f.h" \
    'Copyright (c) 2015-2023 Norbert Juffa'
  require_license_text \
    "$mlx_source/mlx/backend/metal/kernels/cexpf.h" \
    'Copyright © 2008-2013 NVIDIA Corporation'
  require_license_text "$mlx_source/CMakeLists.txt" 'metal-cpp_26.zip'
  require_license_text "$mlx_source/CMakeLists.txt" 'v3.11.3/json.tar.xz'
  require_license_text "$mlx_source/CMakeLists.txt" 'GIT_TAG 12.1.0'

  require_license_text \
    "$build_dir/_deps/metal_cpp-src/Metal/Metal.hpp" \
    'Copyright 2020-2024 Apple Inc.'
  require_license_text \
    "$build_dir/_deps/json-src/LICENSE.MIT" \
    'Copyright (c) 2013-2022 Niels Lohmann'
  require_license_text \
    "$build_dir/_deps/fmt-src/LICENSE" \
    'Victor Zverovich and {fmt} contributors'

  fetched_dependencies=$(
    for fetched_source in "$build_dir"/_deps/*-src
    do
      [ -d "$fetched_source" ] || continue
      basename "$fetched_source"
    done | sed 's/-src$//' | LC_ALL=C sort
  )
  expected_fetched_dependencies=$(printf '%s\n' fmt json metal_cpp)
  if [ "$fetched_dependencies" != "$expected_fetched_dependencies" ]; then
    echo "native fetched dependency set changed; update the license contract" >&2
    echo "expected:" >&2
    printf '%s\n' "$expected_fetched_dependencies" >&2
    echo "actual:" >&2
    printf '%s\n' "$fetched_dependencies" >&2
    exit 1
  fi
}

native_rebuilt=0
if native_artifacts_ready &&
  [ -f "$native_stamp" ] &&
  [ "$(sed -n '1p' "$native_stamp")" = "$native_fingerprint" ]
then
  echo "Native MLX cache ready: $build_dir"
else
  cmake -S "$mlx_c_source" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment_target" \
    -DCMAKE_C_FLAGS_RELEASE="$cpu_flags" \
    -DCMAKE_CXX_FLAGS_RELEASE="$cpu_flags" \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION="$ipo" \
    -DMLX_METAL_DEBUG="$metal_debug" \
    -DBUILD_SHARED_LIBS=OFF \
    -DMLX_C_BUILD_EXAMPLES=OFF \
    -DFETCHCONTENT_SOURCE_DIR_MLX="$mlx_source" \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DMLX_BUILD_GGUF=OFF \
    -DMLX_BUILD_SAFETENSORS=ON

  cmake --build "$build_dir" --target mlxc -j "$build_jobs"
  native_artifacts_ready || {
    echo "native MLX build completed without all required artifacts" >&2
    exit 1
  }
  write_stamp "$native_stamp" "$native_fingerprint"
  native_rebuilt=1
fi

verify_native_license_contract

probe_fingerprint=$(
  {
    printf '%s\n' \
      'generalist-native-probe-schema=1' \
      "native_fingerprint=$native_fingerprint" \
      "deployment_target=$deployment_target"
    cat "$script_dir/probe.c" "$script_dir/tensor_probe.c"
    xcrun clang --version
  } | shasum -a 256 | awk '{print $1}'
)
probe_stamp=$bundle_dir/.generalist-probe-fingerprint
probe_binary=$bundle_dir/generalist-mlx-probe
tensor_probe_binary=$bundle_dir/generalist-mlx-tensor-probe
bundle_metallib=$bundle_dir/mlx.metallib
probe_rebuilt=0

if [ -f "$probe_binary" ] &&
  [ -f "$tensor_probe_binary" ] &&
  [ -f "$bundle_metallib" ] &&
  [ -f "$probe_stamp" ] &&
  [ "$(sed -n '1p' "$probe_stamp")" = "$probe_fingerprint" ]
then
  echo "Static probe cache ready: $bundle_dir"
else
  mkdir -p "$bundle_dir"

  xcrun clang -std=c17 -Wall -Wextra -Werror \
    -mmacosx-version-min="$deployment_target" \
    -I"$mlx_c_source" \
    -c "$script_dir/probe.c" \
    -o "$build_dir/generalist-mlx-probe.o"

  xcrun clang -std=c17 -Wall -Wextra -Werror \
    -mmacosx-version-min="$deployment_target" \
    -I"$mlx_c_source" \
    -c "$script_dir/tensor_probe.c" \
    -o "$build_dir/generalist-mlx-tensor-probe.o"

  xcrun clang++ \
    -mmacosx-version-min="$deployment_target" \
    "$build_dir/generalist-mlx-probe.o" \
    "$mlxc_archive" \
    "$mlx_archive" \
    "$jaccl_archive" \
    -framework Metal \
    -framework Foundation \
    -framework QuartzCore \
    -framework Accelerate \
    -o "$probe_binary"

  xcrun clang++ \
    -mmacosx-version-min="$deployment_target" \
    "$build_dir/generalist-mlx-tensor-probe.o" \
    "$mlxc_archive" \
    "$mlx_archive" \
    "$jaccl_archive" \
    -framework Metal \
    -framework Foundation \
    -framework QuartzCore \
    -framework Accelerate \
    -o "$tensor_probe_binary"

  cp "$metallib" "$bundle_metallib"
  echo "Static probe bundle: $bundle_dir"
  otool -L "$probe_binary"
  write_stamp "$probe_stamp" "$probe_fingerprint"
  probe_rebuilt=1
fi

if [ "${GENERALIST_MLX_RUN_PROBE+x}" = x ]; then
  run_probe=$GENERALIST_MLX_RUN_PROBE
elif [ "$native_rebuilt" = 1 ] || [ "$probe_rebuilt" = 1 ]; then
  run_probe=1
else
  run_probe=0
fi
case $run_probe in
  0) ;;
  1) "$probe_binary" ;;
  *)
    echo "GENERALIST_MLX_RUN_PROBE must be 0 or 1" >&2
    exit 1
    ;;
esac

if [ -n "${GENERALIST_MLX_TENSOR_PROBE_PATH:-}" ]; then
  if [ -z "${GENERALIST_MLX_TENSOR_PROBE_NAME:-}" ]; then
    echo "GENERALIST_MLX_TENSOR_PROBE_NAME is required with the tensor probe path" >&2
    exit 1
  fi
  "$tensor_probe_binary" \
    "$GENERALIST_MLX_TENSOR_PROBE_PATH" \
    "$GENERALIST_MLX_TENSOR_PROBE_NAME"
fi

if [ "$native_rebuilt" = 0 ] &&
  [ "$probe_rebuilt" = 0 ]
then
  echo "Embedded MLX native inputs are unchanged."
fi
