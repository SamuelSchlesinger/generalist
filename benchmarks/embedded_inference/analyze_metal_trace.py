#!/usr/bin/env python3
"""Analyze Generalist's ordered MLX Metal encoder trace.

The debug MLX patch emits one tab-separated MLX_METAL_ENCODER record per
compute encoder. An optional Xcode Instruments metal-gpu-intervals export adds
measured encoder-level GPU time. Kernel counts are exact; timing remains at the
encoder/step level because Instruments does not expose reliable per-dispatch
intervals for this workload.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

TRACE_PREFIX = "MLX_METAL_ENCODER\t"
STEP_BOUNDARY_KERNEL = "argmax_bfloat16"


@dataclass(frozen=True)
class EncoderRecord:
    stream: int
    index: int
    kernels: tuple[str, ...]


@dataclass(frozen=True)
class GpuInterval:
    encoder_id: int
    start_ns: int
    duration_ns: int

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns


def parse_encoder_log(path: Path, selected_stream: int | None = None) -> list[EncoderRecord]:
    streams: dict[int, list[EncoderRecord]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.startswith(TRACE_PREFIX):
                continue
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"{path}:{line_number}: truncated encoder record")
            try:
                stream = int(fields[1])
                index = int(fields[2])
                declared_count = int(fields[3])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: encoder metadata is not integral"
                ) from error
            kernels = tuple(fields[4:])
            if declared_count != len(kernels):
                raise ValueError(
                    f"{path}:{line_number}: declared {declared_count} dispatches "
                    f"but recorded {len(kernels)}"
                )
            if any(not kernel for kernel in kernels):
                raise ValueError(f"{path}:{line_number}: unlabeled Metal pipeline")
            streams[stream].append(EncoderRecord(stream, index, kernels))

    if not streams:
        raise ValueError(f"{path}: no {TRACE_PREFIX.rstrip()} records")
    if selected_stream is None:
        if len(streams) != 1:
            choices = ", ".join(str(stream) for stream in sorted(streams))
            raise ValueError(f"{path}: multiple streams ({choices}); select one with --stream")
        selected_stream = next(iter(streams))
    if selected_stream not in streams:
        raise ValueError(f"{path}: Metal stream {selected_stream} is absent")

    records = streams[selected_stream]
    indexes = [record.index for record in records]
    expected = list(range(len(records)))
    if indexes != expected:
        raise ValueError(
            f"{path}: stream {selected_stream} encoder indexes are not contiguous from zero"
        )
    return records


def _element_value(
    element: ElementTree.Element,
    values_by_id: dict[str, tuple[str | None, str | None]],
) -> tuple[str | None, str | None]:
    reference = element.get("ref")
    if reference is not None:
        try:
            return values_by_id[reference]
        except KeyError as error:
            raise ValueError(f"unresolved Instruments XML reference {reference}") from error
    return element.text, element.get("fmt")


def _process_name(formatted: str | None) -> str | None:
    if formatted is None:
        return None
    prefix, separator, suffix = formatted.rpartition(" (")
    if separator and suffix.endswith(")") and suffix[:-1].isdigit():
        return prefix
    return formatted


def parse_gpu_intervals(path: Path, process_name: str) -> list[GpuInterval]:
    root = ElementTree.parse(path).getroot()
    values_by_id: dict[str, tuple[str | None, str | None]] = {}
    for element in root.iter():
        identifier = element.get("id")
        if identifier is not None:
            values_by_id[identifier] = (element.text, element.get("fmt"))

    matching_nodes: list[tuple[list[str], list[ElementTree.Element]]] = []
    for node in root.iter("node"):
        schema = node.find("schema")
        if schema is None or schema.get("name") != "metal-gpu-intervals":
            continue
        columns = [
            column.findtext("mnemonic") or ""
            for column in schema.findall("col")
        ]
        matching_nodes.append((columns, node.findall("row")))
    if len(matching_nodes) != 1:
        raise ValueError(
            f"{path}: expected one metal-gpu-intervals table, found {len(matching_nodes)}"
        )

    columns, rows = matching_nodes[0]
    required = {"start", "duration", "process", "encoder-id"}
    if not required.issubset(columns):
        missing = ", ".join(sorted(required.difference(columns)))
        raise ValueError(f"{path}: GPU interval table lacks columns: {missing}")

    intervals: list[GpuInterval] = []
    for row_number, row in enumerate(rows, start=1):
        cells = list(row)
        if len(cells) != len(columns):
            raise ValueError(
                f"{path}: GPU row {row_number} has {len(cells)} cells for "
                f"{len(columns)} columns"
            )
        resolved = {
            column: _element_value(cell, values_by_id)
            for column, cell in zip(columns, cells, strict=True)
        }
        if _process_name(resolved["process"][1]) != process_name:
            continue
        try:
            start_ns = int(resolved["start"][0] or "")
            duration_ns = int(resolved["duration"][0] or "")
            encoder_id = int(resolved["encoder-id"][0] or "")
        except ValueError as error:
            raise ValueError(
                f"{path}: GPU row {row_number} has non-integral timing or encoder data"
            ) from error
        if duration_ns < 0:
            raise ValueError(f"{path}: GPU row {row_number} has negative duration")
        intervals.append(GpuInterval(encoder_id, start_ns, duration_ns))

    if not intervals:
        raise ValueError(f"{path}: no GPU intervals for process {process_name!r}")
    return intervals


def union_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if end < start:
            raise ValueError("GPU interval ends before it starts")
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def summarize_trace(
    records: list[EncoderRecord], gpu_intervals: list[GpuInterval] | None = None
) -> dict[str, object]:
    boundaries = [
        position
        for position, record in enumerate(records)
        if STEP_BOUNDARY_KERNEL in record.kernels
    ]
    if len(boundaries) < 2:
        raise ValueError("trace must contain a seed argmax and at least one cached-step argmax")

    gpu_groups: list[list[GpuInterval]] | None = None
    if gpu_intervals is not None:
        grouped: dict[int, list[GpuInterval]] = defaultdict(list)
        for interval in gpu_intervals:
            grouped[interval.encoder_id].append(interval)
        gpu_groups = sorted(
            grouped.values(), key=lambda group: min(interval.start_ns for interval in group)
        )
        if len(gpu_groups) != len(records):
            raise ValueError(
                "ordered Metal trace/GPU interval mismatch: "
                f"{len(records)} encoder records versus {len(gpu_groups)} GPU encoders"
            )

    steps: list[dict[str, object]] = []
    all_warm_kernels: Counter[str] = Counter()
    for step_number, (previous, boundary) in enumerate(
        zip(boundaries, boundaries[1:], strict=False), start=1
    ):
        start = previous + 1
        end = boundary + 1
        step_records = records[start:end]
        kernels = Counter(
            kernel for record in step_records for kernel in record.kernels
        )
        all_warm_kernels.update(kernels)
        step: dict[str, object] = {
            "step": step_number,
            "first_encoder_index": records[start].index,
            "last_encoder_index": records[end - 1].index,
            "encoder_count": len(step_records),
            "dispatch_count": sum(kernels.values()),
        }
        if gpu_groups is not None:
            step_gpu = [
                interval
                for group in gpu_groups[start:end]
                for interval in group
            ]
            start_ns = min(interval.start_ns for interval in step_gpu)
            end_ns = max(interval.end_ns for interval in step_gpu)
            step["gpu_busy_milliseconds"] = union_duration_ns(
                (interval.start_ns, interval.end_ns) for interval in step_gpu
            ) / 1_000_000.0
            step["gpu_span_milliseconds"] = (end_ns - start_ns) / 1_000_000.0
        steps.append(step)

    ordered_kernels = sorted(
        all_warm_kernels.items(), key=lambda item: (-item[1], item[0])
    )
    summary: dict[str, object] = {
        "format_version": 1,
        "stream": records[0].stream,
        "encoder_count": len(records),
        "argmax_boundaries": [records[position].index for position in boundaries],
        "warm_step_count": len(steps),
        "warm_steps": steps,
        "warm_total_dispatches": sum(all_warm_kernels.values()),
        "warm_kernel_counts": [
            {"kernel": kernel, "dispatches": count}
            for kernel, count in ordered_kernels
        ],
    }
    summary["warm_median_dispatches"] = statistics.median(
        int(step["dispatch_count"]) for step in steps
    )
    if gpu_groups is not None:
        summary["warm_median_gpu_busy_milliseconds"] = statistics.median(
            float(step["gpu_busy_milliseconds"]) for step in steps
        )
        summary["warm_median_gpu_span_milliseconds"] = statistics.median(
            float(step["gpu_span_milliseconds"]) for step in steps
        )
    return summary


def print_text(summary: dict[str, object], top: int) -> None:
    print(
        f"stream={summary['stream']} encoders={summary['encoder_count']} "
        f"argmax_boundaries={len(summary['argmax_boundaries'])} "
        f"warm_steps={summary['warm_step_count']}"
    )
    for step in summary["warm_steps"]:
        assert isinstance(step, dict)
        line = (
            f"step={step['step']} encoders={step['encoder_count']} "
            f"dispatches={step['dispatch_count']}"
        )
        if "gpu_busy_milliseconds" in step:
            line += (
                f" gpu_busy_milliseconds={step['gpu_busy_milliseconds']:.3f}"
                f" gpu_span_milliseconds={step['gpu_span_milliseconds']:.3f}"
            )
        print(line)
    aggregate = (
        f"warm_total_dispatches={summary['warm_total_dispatches']} "
        f"warm_median_dispatches={summary['warm_median_dispatches']}"
    )
    if "warm_median_gpu_busy_milliseconds" in summary:
        aggregate += (
            " warm_median_gpu_busy_milliseconds="
            f"{summary['warm_median_gpu_busy_milliseconds']:.3f}"
            " warm_median_gpu_span_milliseconds="
            f"{summary['warm_median_gpu_span_milliseconds']:.3f}"
        )
    print(aggregate)
    kernel_counts = summary["warm_kernel_counts"]
    assert isinstance(kernel_counts, list)
    for entry in kernel_counts[:top]:
        assert isinstance(entry, dict)
        print(f"kernel_dispatches={entry['dispatches']} kernel={entry['kernel']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel_log", type=Path)
    parser.add_argument("--gpu-xml", type=Path)
    parser.add_argument("--process", default="embedded_decode_trace")
    parser.add_argument("--stream", type=int)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json", action="store_true", dest="as_json")
    arguments = parser.parse_args()
    if arguments.top < 0:
        parser.error("--top must be non-negative")

    records = parse_encoder_log(arguments.kernel_log, arguments.stream)
    intervals = (
        parse_gpu_intervals(arguments.gpu_xml, arguments.process)
        if arguments.gpu_xml is not None
        else None
    )
    summary = summarize_trace(records, intervals)
    if arguments.as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary, arguments.top)


if __name__ == "__main__":
    main()
