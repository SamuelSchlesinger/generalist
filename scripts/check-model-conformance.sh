#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)

find_tla_jar() {
    if [ -n "${TLA2TOOLS_JAR:-}" ] && [ -f "$TLA2TOOLS_JAR" ]; then
        printf '%s\n' "$TLA2TOOLS_JAR"
        return
    fi

    if [ -f "$repo_root/.tools/tla2tools.jar" ]; then
        printf '%s\n' "$repo_root/.tools/tla2tools.jar"
        return
    fi

    toolbox_jar="/Applications/TLA+ Toolbox.app/Contents/Eclipse/tla2tools.jar"
    if [ -f "$toolbox_jar" ]; then
        printf '%s\n' "$toolbox_jar"
        return
    fi

    printf '%s\n' \
        "TLC not found. Run 'make tla-tools' or set TLA2TOOLS_JAR." >&2
    exit 1
}

find_java() {
    if [ -n "${TLA_JAVA:-}" ] && [ -x "$TLA_JAVA" ]; then
        printf '%s\n' "$TLA_JAVA"
        return
    fi

    if command -v java >/dev/null 2>&1 && java -version >/dev/null 2>&1; then
        command -v java
        return
    fi

    for candidate in \
        /Applications/TLA+\ Toolbox.app/Contents/Eclipse/plugins/org.lamport.openjdk.*/Contents/Home/bin/java
    do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return
        fi
    done

    printf '%s\n' \
        "Java 11 or newer is required to run TLC. Set TLA_JAVA if needed." >&2
    exit 1
}

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

for model in AsyncObservedTrace MemoryObservedTrace ArchiveObservedTrace
do
    log="$work_dir/$model.log"
    if ! run_tlc "$model" "$log"; then
        printf 'Observed implementation trace failed TLA+ conformance: %s\n' "$model" >&2
        cat "$log" >&2
        exit 1
    fi
    printf 'PASS observed implementation trace: %s\n' "$model"
done

for model in AsyncInvalidTrace MemoryInvalidTrace ArchiveInvalidTrace
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
    printf 'PASS rejected refinement mutation: %s\n' "$model"
done
