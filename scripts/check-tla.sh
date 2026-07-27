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
