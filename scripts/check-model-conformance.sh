#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

# shellcheck source=scripts/tlc-common.sh
. "$repo_root/scripts/tlc-common.sh"

tla_jar=$(find_tla_jar)
java_bin=$(find_java)
work_dir_unresolved=$(mktemp -d "${TMPDIR:-/tmp}/generalist-conformance.XXXXXX")
work_dir=$(CDPATH='' cd -- "$work_dir_unresolved" && pwd -P)
trace_json="$work_dir/implementation-traces.json"
rendered="$work_dir/rendered"
metadata="$work_dir/tlc"
trap 'rm -rf -- "$work_dir"' EXIT HUP INT TERM

cd "$repo_root"
cargo run --quiet --example model_conformance --locked -- "$trace_json"
PYTHONDONTWRITEBYTECODE=1 python3 \
    scripts/render-model-traces.py "$trace_json" "$rendered"

run_tlc() {
    model=$1
    log=$2
    mkdir -p "$metadata/$model"
    "$java_bin" \
        -XX:+UseParallelGC \
        -cp "$tla_jar" \
        tlc2.TLC \
        -metadir "$metadata/$model" \
        -workers 1 \
        -config "$rendered/$model.cfg" \
        "$rendered/$model.tla" >"$log" 2>&1
}

failed=0
while IFS= read -r model
do
    log="$work_dir/$model.log"
    if ! run_tlc "$model" "$log"; then
        printf 'Observed implementation trace failed TLA+ conformance: %s\n' "$model" >&2
        cat "$log" >&2
        failed=1
        continue
    fi
    printf 'PASS observed implementation trace: %s\n' "$model"
done < "$rendered/observed.txt"
[ "$failed" -eq 0 ] || exit 1

while IFS= read -r model
do
    log="$work_dir/$model.log"
    if run_tlc "$model" "$log"; then
        printf 'Invalid mutation unexpectedly conformed to TLA+: %s\n' "$model" >&2
        cat "$log" >&2
        exit 1
    fi
    if ! grep -Fq 'Temporal properties were violated' "$log"; then
        printf 'Invalid mutation failed for the wrong reason: %s\n' "$model" >&2
        cat "$log" >&2
        exit 1
    fi
    printf 'PASS rejected trace mutation: %s\n' "$model"
done < "$rendered/invalid.txt"

while IFS="$(printf '\t')" read -r model invariant
do
    log="$work_dir/$model.log"
    if run_tlc "$model" "$log"; then
        printf 'Model fault escaped its invariants: %s\n' "$model" >&2
        exit 1
    fi
    if ! grep -Fq "Invariant $invariant is violated" "$log"; then
        printf 'Model fault failed for the wrong reason: %s\n' "$model" >&2
        cat "$log" >&2
        exit 1
    fi
    printf 'PASS model fault: %s violates %s\n' "$model" "$invariant"
done < "$rendered/model-mutants.tsv"
