#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

# shellcheck source=scripts/tlc-common.sh
. "$repo_root/scripts/tlc-common.sh"

tla_jar=$(find_tla_jar)
java_bin=$(find_java)
metadata_dir=$(mktemp -d "${TMPDIR:-/tmp}/generalist-tlc.XXXXXX")
trap 'rm -rf -- "$metadata_dir"' EXIT HUP INT TERM

run_model() {
    model=$1
    config=$2
    model_metadata="$metadata_dir/$model"
    mkdir -p "$model_metadata"
    printf 'Checking %s with %s\n' "$model" "$config"
    "$java_bin" \
        -XX:+UseParallelGC \
        -cp "$tla_jar" \
        tlc2.TLC \
        -metadir "$model_metadata" \
        -workers "${TLC_WORKERS:-1}" \
        -config "$config" \
        "$model"
}

cd "$repo_root/spec"
run_model AsyncRuntime.tla AsyncRuntime.cfg
run_model MemoryRuntime.tla MemoryRuntime.cfg
run_model ArchiveScopeRuntime.tla ArchiveScopeRuntime.cfg
