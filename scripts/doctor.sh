#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        printf 'ok  %-12s %s\n' "$1" "$(command -v "$1")"
    else
        printf 'ERR %-12s missing\n' "$1" >&2
        failed=1
    fi
}

check_command cargo
check_command rustfmt
check_command shellcheck
check_command curl

hooks_path=$(git config --local --get core.hooksPath || true)
if [ "$hooks_path" = ".githooks" ]; then
    printf 'ok  %-12s %s\n' "git hooks" "$hooks_path"
else
    printf 'ERR %-12s run make hooks\n' "git hooks" >&2
    failed=1
fi

if [ -f .tools/tla2tools.jar ] ||
    [ -f "/Applications/TLA+ Toolbox.app/Contents/Eclipse/tla2tools.jar" ] ||
    [ -f "${TLA2TOOLS_JAR:-/path/that/does/not/exist}" ]; then
    printf 'ok  %-12s available\n' "TLC"
else
    printf 'ERR %-12s run make tla-tools\n' "TLC" >&2
    failed=1
fi

exit "$failed"
