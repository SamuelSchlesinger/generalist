#!/bin/sh
set -eu

version=v1.7.4
expected_sha256=936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88
url="https://github.com/tlaplus/tlaplus/releases/download/$version/tla2tools.jar"
repo_root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
destination="$repo_root/.tools/tla2tools.jar"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

if [ -f "$destination" ] &&
    [ "$(sha256_file "$destination")" = "$expected_sha256" ]; then
    printf 'TLC %s is already installed at %s\n' "$version" "$destination"
    exit 0
fi

temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/generalist-tla-install.XXXXXX")
trap 'rm -rf -- "$temporary_dir"' EXIT HUP INT TERM
download="$temporary_dir/tla2tools.jar"

printf 'Downloading TLC %s from the official TLA+ release...\n' "$version"
curl --fail --location --retry 3 --output "$download" "$url"
actual_sha256=$(sha256_file "$download")
if [ "$actual_sha256" != "$expected_sha256" ]; then
    printf 'TLC checksum mismatch: expected %s, got %s\n' \
        "$expected_sha256" "$actual_sha256" >&2
    exit 1
fi

mkdir -p "$repo_root/.tools"
mv "$download" "$destination"
printf 'Installed verified TLC %s at %s\n' "$version" "$destination"
