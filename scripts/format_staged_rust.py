#!/usr/bin/env python3
"""Format staged Rust files without broadening a partially staged commit."""

from __future__ import annotations

import os
import subprocess
import sys


def run(command: list[bytes], *, capture_stdout: bool = False) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE if capture_stdout else None,
    )


def main() -> int:
    staged = run(
        [
            b"git",
            b"diff",
            b"--cached",
            b"--name-only",
            b"--diff-filter=ACMR",
            b"-z",
            b"--",
            b"*.rs",
        ],
        capture_stdout=True,
    )
    if staged.returncode != 0:
        return staged.returncode

    paths = [path for path in staged.stdout.split(b"\0") if path]
    partially_staged: list[bytes] = []
    for path in paths:
        status = run([b"git", b"diff", b"--quiet", b"--no-ext-diff", b"--", path])
        if status.returncode == 1:
            partially_staged.append(path)
        elif status.returncode != 0:
            return status.returncode

    if partially_staged:
        print(
            "format-staged: refusing to re-stage Rust files with unstaged changes:",
            file=sys.stderr,
        )
        for path in partially_staged:
            print(f"  {os.fsdecode(path)!r}", file=sys.stderr)
        print(
            "Stage, stash, or revert those unstaged hunks before committing.",
            file=sys.stderr,
        )
        return 1

    try:
        for path in paths:
            status = run([b"rustfmt", b"--edition", b"2021", b"--", path])
            if status.returncode != 0:
                return status.returncode
        if paths:
            return run([b"git", b"add", b"--", *paths]).returncode
    except FileNotFoundError as error:
        print(f"format-staged: required executable not found: {error.filename}", file=sys.stderr)
        return 127

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
