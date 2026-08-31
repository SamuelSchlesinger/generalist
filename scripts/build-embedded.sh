#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

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

mode=build
case ${1:-} in
  --print-cache)
    printf '%s\n' "$work_root"
    exit 0
    ;;
  --install) mode=install ;;
  '') ;;
  *)
    echo "usage: scripts/build-embedded.sh [--install|--print-cache]" >&2
    exit 2
    ;;
esac

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "embedded MLX requires Apple Silicon macOS" >&2
  exit 1
fi

for required_command in cargo cmake git shasum xcrun; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "required command is missing: $required_command" >&2
    exit 1
  fi
done

variant=${GENERALIST_MLX_VARIANT:-baseline}
mlx_source=${GENERALIST_MLX_SOURCE_DIR:-$work_root/mlx}
mlx_c_source=${GENERALIST_MLX_C_SOURCE_DIR:-$work_root/mlx-c}
build_dir=${GENERALIST_MLX_BUILD_DIR:-$work_root/build/$variant}
bundle_dir=${GENERALIST_MLX_BUNDLE_DIR:-$work_root/bundle/$variant}

mkdir -p "$work_root"

echo "Preparing pinned MLX cache in: $work_root"
GENERALIST_MLX_WORK_ROOT="$work_root" \
GENERALIST_MLX_SOURCE_DIR="$mlx_source" \
GENERALIST_MLX_C_SOURCE_DIR="$mlx_c_source" \
GENERALIST_MLX_BUILD_DIR="$build_dir" \
GENERALIST_MLX_BUNDLE_DIR="$bundle_dir" \
  "$repo_root/native/mlx/build_static_probe.sh"

cd "$repo_root"
if [ "$mode" = install ]; then
  cargo_target_dir=${CARGO_TARGET_DIR:-$repo_root/target}
  GENERALIST_MLX_SOURCE_DIR="$mlx_source" \
  GENERALIST_MLX_C_SOURCE_DIR="$mlx_c_source" \
  GENERALIST_MLX_BUILD_DIR="$build_dir" \
  CARGO_TARGET_DIR="$cargo_target_dir" \
    cargo install --path "$repo_root" --locked \
      --features embedded-mlx-native --bin generalist --force
  echo "Installed embedded Generalist with Cargo."
  echo "Run: generalist --embedded"
else
  GENERALIST_MLX_SOURCE_DIR="$mlx_source" \
  GENERALIST_MLX_C_SOURCE_DIR="$mlx_c_source" \
  GENERALIST_MLX_BUILD_DIR="$build_dir" \
    cargo build --release --locked --features embedded-mlx-native --bin generalist

  target_root=${CARGO_TARGET_DIR:-$repo_root/target}
  case $target_root in
    /*) ;;
    *) target_root=$repo_root/$target_root ;;
  esac

  echo "Embedded Generalist: $target_root/release/generalist"
  echo "Run: $target_root/release/generalist --embedded"
fi
