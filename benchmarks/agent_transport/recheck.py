#!/usr/bin/env python3
"""Re-evaluate preserved benchmark responses with the current static checker."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import importlib.util
import json
import pathlib
import sys
import uuid
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
RUN_PATH = HERE / "run.py"
RUN_SPEC = importlib.util.spec_from_file_location("agent_transport_run_for_recheck", RUN_PATH)
benchmark = importlib.util.module_from_spec(RUN_SPEC)
assert RUN_SPEC.loader is not None
RUN_SPEC.loader.exec_module(benchmark)


def recheck_attempt(
    record: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    source_path: pathlib.Path,
    run_id: str,
    bridge_preloaded: bool = False,
) -> dict[str, Any]:
    derived = copy.deepcopy(record)
    original_run_id = record.get("run_id")
    original_classification = record.get("classification")
    derived["run_id"] = run_id
    derived["rechecked_at"] = benchmark.utc_now()
    derived["derived_from"] = {
        "source_file": str(source_path.resolve()),
        "run_id": original_run_id,
        "classification": original_classification,
    }
    task = tasks.get(record.get("task_id"))
    response = record.get("response")
    transport = record.get("transport")
    http_status = record.get("http_status")
    if task is None or not isinstance(response, dict) or not isinstance(http_status, int):
        derived["recheck_note"] = "original attempt lacked a known task or complete HTTP response"
        return derived
    if transport == "responses_custom":
        code, extraction = benchmark.extract_responses_code(response)
    elif transport in {"json_tool", "json_tool_legacy", "plain_text"}:
        code, extraction = benchmark.extract_chat_code(response, transport)
    else:
        derived["recheck_note"] = f"unknown transport: {transport}"
        return derived
    checker = (
        benchmark.check_source(
            code,
            task["expected_calls"],
            bridge_preloaded=bridge_preloaded and transport != "json_tool_legacy",
        )
        if code is not None
        else None
    )
    derived["code"] = code
    derived["extraction"] = extraction
    derived["checker"] = checker
    derived["classification"] = benchmark.classify(http_status, extraction, checker)
    return derived


def default_output() -> pathlib.Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return HERE / "results" / f"{timestamp}-rechecked.jsonl"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--corpus", type=pathlib.Path, default=benchmark.DEFAULT_CORPUS)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    corpus = benchmark.load_corpus(args.corpus)
    tasks = {task["id"]: task for task in corpus["tasks"]}
    output = (args.output or default_output()).resolve()
    run_id = str(uuid.uuid4())
    writer = benchmark.JsonlWriter(output, args.append)
    attempts = 0
    changed = 0
    writer.write(
        {
            "schema_version": 1,
            "record_type": "run_start",
            "run_id": run_id,
            "started_at": benchmark.utc_now(),
            "kind": "derived_recheck",
            "source_files": [
                {
                    "path": str(path.resolve()),
                    "sha256": benchmark.sha256_file(path.resolve()),
                }
                for path in args.paths
            ],
            "artifacts": {
                "runner_sha256": benchmark.sha256_file(RUN_PATH),
                "rechecker_sha256": benchmark.sha256_file(pathlib.Path(__file__).resolve()),
                "corpus_sha256": benchmark.sha256_file(args.corpus.resolve()),
            },
        }
    )
    try:
        for path in args.paths:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("record_type") != "attempt":
                        continue
                    derived = recheck_attempt(
                        record,
                        tasks,
                        path,
                        run_id,
                        bridge_preloaded=bool(corpus.get("bridge_preloaded")),
                    )
                    writer.write(derived)
                    attempts += 1
                    if derived.get("classification") != record.get("classification"):
                        changed += 1
    finally:
        writer.write(
            {
                "schema_version": 1,
                "record_type": "run_end",
                "run_id": run_id,
                "finished_at": benchmark.utc_now(),
                "attempts": attempts,
                "changed_classifications": changed,
            }
        )
        writer.close()
    print(f"rechecked {attempts} attempts ({changed} classifications changed) into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
