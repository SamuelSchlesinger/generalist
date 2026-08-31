#!/usr/bin/env python3
"""Summarize one or more local-reasoning JSONL benchmark runs."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def load_runs(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                run_id = record.get("run_id")
                if not isinstance(run_id, str):
                    raise ValueError(f"{path}:{line_number}: record has no run_id")
                if run_id not in runs:
                    runs[run_id] = {"path": str(path), "start": None, "end": None}
                    order.append(run_id)
                if record.get("record_type") == "run_start":
                    runs[run_id]["start"] = record
                elif record.get("record_type") == "run_end":
                    runs[run_id]["end"] = record
    return [runs[run_id] for run_id in order]


def cell(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(runs: list[dict[str, Any]]) -> str:
    rows = [
        "| Variant | Model | Status | Pass | Reasoning | Complete | Compact | Wall p50 ms | "
        "Prefill tok/s p50 | Decode tok/s p50 | MTP accept p50 | Draft p50 | "
        "API eval tokens p50 | Runtime passes p50 | Runtime tokens est p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        start = run.get("start") or {}
        end = run.get("end") or {}
        aggregate = end.get("aggregate") or {}
        medians = aggregate.get("medians") or {}
        attempts = aggregate.get("attempts", 0)
        rows.append(
            "| {variant} | {model} | {status} | {passed}/{attempts} | "
            "{reasoning}/{attempts} | {completed}/{attempts} | {compact}/{attempts} | "
            "{wall} | {prefill} | {decode} | {acceptance} | {draft} | {tokens} | "
            "{passes} | {runtime_tokens} |".format(
                variant=start.get("variant", "-"),
                model=start.get("model", "-"),
                status=end.get("status", "missing"),
                passed=aggregate.get("passed", 0),
                attempts=attempts,
                reasoning=aggregate.get("reasoning_present", 0),
                completed=aggregate.get("completed", 0),
                compact=aggregate.get("compact_answer", 0),
                wall=cell(medians.get("wall_ms")),
                prefill=cell(medians.get("prompt_tokens_per_second")),
                decode=cell(medians.get("decode_tokens_per_second")),
                acceptance=cell(medians.get("mtp_acceptance"), digits=2),
                draft=cell(medians.get("mtp_avg_draft"), digits=2),
                tokens=cell(medians.get("eval_count")),
                passes=cell(medians.get("runtime_inference_passes")),
                runtime_tokens=cell(medians.get("runtime_estimated_emitted_tokens")),
            )
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    print(markdown_table(load_runs(args.paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
