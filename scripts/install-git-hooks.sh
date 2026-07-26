#!/bin/sh
set -eu

repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

git config --local core.hooksPath .githooks

configured=$(git config --local --get core.hooksPath)
if [ "$configured" != ".githooks" ]; then
    printf 'Failed to configure the checked-in Git hooks.\n' >&2
    exit 1
fi

printf 'Git hooks enabled from .githooks\n'
