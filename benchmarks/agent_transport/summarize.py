#!/usr/bin/env python3
"""Summarize append-only agent-transport benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable


def read_attempts(paths: Iterable[pathlib.Path], strict: bool = False) -> list[dict[str, Any]]:
    attempts = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    message = f"{path}:{line_number}: invalid JSONL record: {error}"
                    if strict:
                        raise ValueError(message) from error
                    print(f"warning: {message}", file=sys.stderr)
                    continue
                if record.get("record_type") == "attempt":
                    attempts.append(record)
    return attempts


def numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def median_or_none(values: Iterable[Any]) -> float | None:
    present = numeric(values)
    return statistics.median(present) if present else None


def mean_or_none(values: Iterable[Any]) -> float | None:
    present = numeric(values)
    return statistics.fmean(present) if present else None


def summarize(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        key = (
            attempt.get("provider", "unknown"),
            attempt.get("model", "unknown"),
            attempt.get("transport", "unknown"),
        )
        groups[key].append(attempt)
    rows = []
    for (provider, model, transport), records in sorted(groups.items()):
        classes = Counter(record.get("classification", "unknown") for record in records)
        usage = [record.get("metrics", {}).get("usage") or {} for record in records]
        reported_costs = numeric(item.get("reported_cost") for item in usage)
        rows.append(
            {
                "provider": provider,
                "model": model,
                "transport": transport,
                "attempts": len(records),
                "passed": classes["pass"],
                "pass_rate": classes["pass"] / len(records),
                "classifications": dict(sorted(classes.items())),
                "median_latency_ms": median_or_none(
                    record.get("metrics", {}).get("latency_ms") for record in records
                ),
                "mean_input_tokens": mean_or_none(item.get("input_tokens") for item in usage),
                "mean_output_tokens": mean_or_none(item.get("output_tokens") for item in usage),
                "median_model_payload_overhead_bytes": median_or_none(
                    record.get("metrics", {}).get("model_payload_overhead_bytes")
                    for record in records
                ),
                "reported_cost_total": sum(reported_costs) if reported_costs else None,
            }
        )
    return rows


def display_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Provider / model | Transport | Pass | Rate | p50 latency | Mean input | Mean output | p50 model payload overhead | Reported cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        identity = f"{row['provider']} / {row['model']}".replace("|", "\\|")
        cost = row["reported_cost_total"]
        lines.append(
            "| "
            + " | ".join(
                [
                    identity,
                    row["transport"],
                    f"{row['passed']}/{row['attempts']}",
                    f"{100 * row['pass_rate']:.1f}%",
                    display_number(row["median_latency_ms"]) + " ms",
                    display_number(row["mean_input_tokens"]),
                    display_number(row["mean_output_tokens"]),
                    display_number(row["median_model_payload_overhead_bytes"]) + " B",
                    "—" if cost is None else f"${cost:.6f}",
                ]
            )
            + " |"
        )
    failures = []
    for row in rows:
        non_pass = {key: count for key, count in row["classifications"].items() if key != "pass"}
        if non_pass:
            failures.append(
                f"- `{row['provider']} / {row['model']} / {row['transport']}`: "
                + ", ".join(f"{key}={count}" for key, count in non_pass.items())
            )
    if failures:
        lines.extend(["", "Failure classifications", "", *failures])
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    attempts = read_attempts(args.paths, strict=args.strict)
    rows = summarize(attempts)
    if args.format == "json":
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(markdown(rows))
    return 0 if attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
