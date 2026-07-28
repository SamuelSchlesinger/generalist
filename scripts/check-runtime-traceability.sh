#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
trace="$repo_root/docs/runtime-traceability.md"
contributing="$repo_root/CONTRIBUTING.md"
makefile="$repo_root/Makefile"

fail=0

check_model() {
    spec=$1
    config=$2
    aggregate_start=$3
    state_start=$4
    state_end=$5
    action_start=$6
    action_end=$7
    property_start=$8
    property_end=$9

    state_section=$(
        awk -v start="$state_start" -v end="$state_end" '
            $0 == start { in_section = 1; next }
            in_section && $0 == end { exit }
            in_section { print }
        ' "$trace"
    )
    action_section=$(
        awk -v start="$action_start" -v end="$action_end" '
            $0 == start { in_section = 1; next }
            in_section && $0 == end { exit }
            in_section { print }
        ' "$trace"
    )
    property_section=$(
        awk -v start="$property_start" -v end="$property_end" '
            $0 == start { in_section = 1; next }
            in_section && $0 == end { exit }
            in_section { print }
        ' "$trace"
    )

    variables=$(
        awk '
            /^VARIABLES$/ { in_variables = 1; next }
            in_variables && /^vars ==/ { exit }
            in_variables && /^[[:space:]]+[A-Za-z][A-Za-z0-9_]*,?$/ {
                name = $1
                sub(/,$/, "", name)
                print name
            }
        ' "$spec"
    )

    for variable in $variables; do
        if ! printf '%s\n' "$state_section" | grep -Fq "\`$variable\`"; then
            printf 'Missing TLA+ state in live traceability matrix: %s\n' "$variable" >&2
            fail=1
        fi
    done

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
        if ! printf '%s\n' "$action_section" | grep -Fq "\`$action\`"; then
            printf 'Missing TLA+ action in live traceability matrix: %s\n' "$action" >&2
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
        if ! printf '%s\n' "$property_section" | grep -Fq "\`$invariant\`"; then
            printf 'Missing TLA+ invariant in live traceability matrix: %s\n' "$invariant" >&2
            fail=1
        fi
    done

    properties=$(awk '$1 == "PROPERTY" { for (i = 2; i <= NF; i++) print $i }' "$config")
    for property in $properties; do
        if ! printf '%s\n' "$property_section" | grep -Fq "\`$property\`"; then
            printf 'Missing TLA+ property in live traceability matrix: %s\n' "$property" >&2
            fail=1
        fi
    done
}

check_model \
    "$repo_root/spec/AsyncRuntime.tla" \
    "$repo_root/spec/AsyncRuntime.cfg" \
    UserAction \
    "## State mapping" \
    "## Action mapping" \
    "## Action mapping" \
    "## Property mapping" \
    "## Property mapping" \
    "## Episodic-memory model mapping"
check_model \
    "$repo_root/spec/MemoryRuntime.tla" \
    "$repo_root/spec/MemoryRuntime.cfg" \
    "" \
    "### Memory state mapping" \
    "### Memory action mapping" \
    "### Memory action mapping" \
    "### Memory property mapping" \
    "### Memory property mapping" \
    "## Archive-scope model mapping"
check_model \
    "$repo_root/spec/ArchiveScopeRuntime.tla" \
    "$repo_root/spec/ArchiveScopeRuntime.cfg" \
    "" \
    "### Archive-scope state mapping" \
    "### Archive-scope action mapping" \
    "### Archive-scope action mapping" \
    "### Archive-scope property mapping" \
    "### Archive-scope property mapping" \
    "## Durable-boundary refinement"

if ! grep -Fq 'docs/runtime-traceability.md' "$contributing"; then
    printf 'CONTRIBUTING.md must require the runtime traceability review.\n' >&2
    fail=1
fi
if ! grep -Fq 'spec/MemoryRuntime.tla' "$contributing"; then
    printf 'CONTRIBUTING.md must require the memory-runtime traceability review.\n' >&2
    fail=1
fi
if ! grep -Fq 'spec/ArchiveScopeRuntime.tla' "$contributing"; then
    printf 'CONTRIBUTING.md must require the archive-scope traceability review.\n' >&2
    fail=1
fi
for artifact in \
    'src/model_trace.rs' \
    'examples/model_conformance.rs' \
    'scripts/check-model-conformance.sh' \
    'DisclosureGrant'
do
    if ! grep -Fq "$artifact" "$trace" && ! grep -Fq "$artifact" "$contributing"; then
        printf 'Live traceability guidance must name conformance artifact: %s\n' "$artifact" >&2
        fail=1
    fi
done
if ! grep -Fq './scripts/check-model-conformance.sh' "$makefile"; then
    printf 'Makefile must run the implementation-trace conformance check.\n' >&2
    fail=1
fi
if ! grep -Fq 'make conformance' "$contributing"; then
    printf 'CONTRIBUTING.md must require sampled implementation conformance.\n' >&2
    fail=1
fi

acknowledgement='Yes, I have updated the TLA+ model to reflect the current architecture'
if ! grep -Fq "$acknowledgement" "$contributing"; then
    printf 'CONTRIBUTING.md is missing the commit acknowledgement text.\n' >&2
    fail=1
fi

exit "$fail"
