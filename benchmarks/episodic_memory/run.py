#!/usr/bin/env python3
"""Run and preserve Generalist's explicit episodic-memory evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORPUS = HERE / "cases.json"
RESULTS = HERE / "results"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ui_probe


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_line(output: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError("evaluation executable produced no JSON")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError("evaluation executable did not produce a JSON object")
    return value


def git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def git_status_lines() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.splitlines()


def target_directory() -> pathlib.Path:
    configured = os.environ.get("CARGO_TARGET_DIR")
    if configured:
        path = pathlib.Path(configured)
        return path if path.is_absolute() else ROOT / path
    return ROOT / "target"


def build() -> tuple[pathlib.Path, pathlib.Path]:
    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--bin",
            "generalist",
            "--example",
            "memory_evaluation",
        ],
        cwd=ROOT,
        check=True,
    )
    target = target_directory() / "debug"
    return target / "generalist", target / "examples" / "memory_evaluation"


def write_jsonl(path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        for record in records:
            json.dump(record, output, ensure_ascii=False, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())


def default_output() -> pathlib.Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS / f"{stamp}-local-explicit-memory.jsonl"


def run(args: argparse.Namespace) -> tuple[pathlib.Path, dict[str, Any]]:
    if args.no_build:
        target = target_directory() / "debug"
        binary = target / "generalist"
        evaluator = target / "examples" / "memory_evaluation"
    else:
        binary, evaluator = build()
    for artifact in [binary, evaluator]:
        if not artifact.is_file():
            raise FileNotFoundError(artifact)

    started = utc_now()
    core_process = subprocess.run(
        [str(evaluator)], cwd=ROOT, text=True, capture_output=True
    )
    core = load_json_line(core_process.stdout)
    tui = None if args.skip_ui else ui_probe.run(binary)
    dirty = git_status_lines()
    result = {
        "kind": "result",
        "schema_version": 1,
        "core": core,
        "tui": tui,
        "passed": bool(core.get("passed"))
        and core_process.returncode == 0
        and (tui is None or bool(tui.get("passed"))),
    }
    metadata = {
        "kind": "run",
        "schema_version": 1,
        "started_at": started,
        "finished_at": utc_now(),
        "git_head": git(["rev-parse", "HEAD"]),
        "git_dirty": bool(dirty),
        "git_dirty_paths": [line[3:] for line in dirty],
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "sqlite": __import__("sqlite3").sqlite_version,
        "corpus_sha256": sha256_file(CORPUS),
        "binary_sha256": sha256_file(binary),
        "evaluator_sha256": sha256_file(evaluator),
        "ui_executed": tui is not None,
        "command": " ".join(sys.argv),
    }
    output = args.output or default_output()
    write_jsonl(output, [metadata, result])
    return output, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--skip-ui",
        action="store_true",
        help="run storage/crash evaluation without the tmux exact-TUI probe",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="reuse target/debug artifacts instead of rebuilding",
    )
    return parser.parse_args()


def main() -> int:
    output, result = run(parse_args())
    core = result["core"]
    b1 = core["retrieval"]["b1_episodic"]
    autosave = core["retrieval"]["history_autosave"]
    named = core["retrieval"]["history_named"]
    print(
        "memory evaluation:",
        "PASS" if result["passed"] else "FAIL",
        f"B1 recall={b1['recall']:.3f}",
        f"autosave recall={autosave['recall']:.3f}",
        f"named-save recall={named['recall']:.3f}",
    )
    print(output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
