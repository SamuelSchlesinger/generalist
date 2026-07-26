#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
spec="$repo_root/spec/AsyncRuntime.tla"
config="$repo_root/spec/AsyncRuntime.cfg"
trace="$repo_root/docs/runtime-traceability.md"
contributing="$repo_root/CONTRIBUTING.md"

fail=0

actions=$(
    awk '
        /^Enqueue\(/ { in_actions = 1 }
        in_actions && /^UserAction ==/ { exit }
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

if ! grep -Fq 'docs/runtime-traceability.md' "$contributing"; then
    printf 'CONTRIBUTING.md must require the runtime traceability review.\n' >&2
    fail=1
fi

acknowledgement='Yes, I have updated the TLA+ model to reflect the current architecture'
if ! grep -Fq "$acknowledgement" "$contributing"; then
    printf 'CONTRIBUTING.md is missing the commit acknowledgement text.\n' >&2
    fail=1
fi

exit "$fail"
