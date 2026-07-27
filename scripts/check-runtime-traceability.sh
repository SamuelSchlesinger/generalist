#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
trace="$repo_root/docs/runtime-traceability.md"
contributing="$repo_root/CONTRIBUTING.md"

fail=0

check_model() {
    spec=$1
    config=$2
    aggregate_start=$3

    actions=$(
        awk -v aggregate_start="$aggregate_start" '
            /^Init ==/ { in_actions = 1; next }
            in_actions && /^Next ==/ { exit }
            in_actions && aggregate_start != "" &&
                $0 ~ ("^" aggregate_start " ==") { exit }
            in_actions && /^[A-Z][A-Za-z0-9_]*(\([^)]*\))? ==$/ {
                name = $1
                sub(/\(.*/, "", name)
                print name
            }
        ' "$spec"
    )

    for action in $actions; do
        if ! grep -Fq "\`$action\`" "$trace"; then
            printf 'Missing TLA+ action in traceability matrix: %s\n' "$action" >&2
            fail=1
        fi
    done

    invariants=$(
        awk '
            $1 == "INVARIANTS" { in_invariants = 1; next }
            $1 == "PROPERTY" { in_invariants = 0 }
            in_invariants && NF { print $1 }
        ' "$config"
    )

    for invariant in $invariants; do
        if ! grep -Fq "\`$invariant\`" "$trace"; then
            printf 'Missing TLA+ invariant in traceability matrix: %s\n' "$invariant" >&2
            fail=1
        fi
    done

    properties=$(awk '$1 == "PROPERTY" { for (i = 2; i <= NF; i++) print $i }' "$config")
    for property in $properties; do
        if ! grep -Fq "\`$property\`" "$trace"; then
            printf 'Missing TLA+ property in traceability matrix: %s\n' "$property" >&2
            fail=1
        fi
    done
}

check_model \
    "$repo_root/spec/AsyncRuntime.tla" \
    "$repo_root/spec/AsyncRuntime.cfg" \
    UserAction
check_model \
    "$repo_root/spec/MemoryRuntime.tla" \
    "$repo_root/spec/MemoryRuntime.cfg" \
    ""

if ! grep -Fq 'docs/runtime-traceability.md' "$contributing"; then
    printf 'CONTRIBUTING.md must require the runtime traceability review.\n' >&2
    fail=1
fi
if ! grep -Fq 'spec/MemoryRuntime.tla' "$contributing"; then
    printf 'CONTRIBUTING.md must require the memory-runtime traceability review.\n' >&2
    fail=1
fi

acknowledgement='Yes, I have updated the TLA+ model to reflect the current architecture'
if ! grep -Fq "$acknowledgement" "$contributing"; then
    printf 'CONTRIBUTING.md is missing the commit acknowledgement text.\n' >&2
    fail=1
fi

exit "$fail"
