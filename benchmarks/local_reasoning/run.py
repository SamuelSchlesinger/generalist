#!/usr/bin/env python3
"""Benchmark local reasoning quality, completion, prefill, and decode throughput."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "tasks.json"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.8:27b-mlx"
DEFAULT_OLLAMA_LOG = pathlib.Path.home() / ".ollama" / "logs" / "server.log"
SPECULATION_STATS_PATTERN = re.compile(
    r'msg="speculative decode stats".*?iterations=(?P<iterations>\d+)'
    r'.*?drafted=(?P<drafted>\d+).*?accepted=(?P<accepted>\d+)'
    r'.*?acceptance=(?P<acceptance>[0-9.]+).*?avg_draft=(?P<avg_draft>[0-9.]+)'
    r'.*?max_draft=(?P<max_draft>\d+).*?avg_accepted=(?P<avg_accepted>[0-9.]+)'
    r'.*?depth_over_time="(?P<depth_over_time>[^"]*)"'
)
SYSTEM_PROMPT = (
    "You are running a reasoning benchmark. Work through the problem in the dedicated "
    "thinking channel. Be accurate and reasonably concise. After reasoning is complete, "
    "put only `FINAL: <answer>` in the response channel, with no explanation before or "
    "after it. Do not shorten or skip the thinking work."
)
JSON_SYSTEM_PROMPT = (
    "You are running a reasoning benchmark. Work through the problem in the dedicated "
    "thinking channel. Be accurate and reasonably concise. After reasoning is complete, "
    'put only a JSON object of the form {"final":"<answer>"} in the response channel. '
    "Do not repeat the reasoning there. Do not shorten or skip the thinking work."
)
FINAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {"final": {"type": "string"}},
    "required": ["final"],
    "additionalProperties": False,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or "model"


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class JsonlWriter:
    """Flush every record so an interrupted benchmark remains useful evidence."""

    def __init__(self, path: pathlib.Path, append: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a" if append else "x", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        json.dump(record, self._file, ensure_ascii=False, separators=(",", ":"))
        self._file.write("\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def load_corpus(path: pathlib.Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        corpus = json.load(source)
    if corpus.get("schema_version") != 1:
        raise ValueError(f"unsupported corpus schema in {path}")
    tasks = corpus.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"corpus has no tasks: {path}")
    ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or not all(
            isinstance(task.get(key), str) and task[key]
            for key in ("id", "prompt", "expected")
        ):
            raise ValueError("every task needs non-empty id, prompt, and expected strings")
        if task["id"] in ids:
            raise ValueError(f"duplicate task id: {task['id']}")
        ids.add(task["id"])
    return corpus


def selected_tasks(
    corpus: dict[str, Any], task_ids: list[str], tags: list[str], max_tasks: int | None
) -> list[dict[str, Any]]:
    known_ids = {task["id"] for task in corpus["tasks"]}
    missing = set(task_ids) - known_ids
    if missing:
        raise ValueError(f"unknown task ids: {', '.join(sorted(missing))}")
    tasks = [
        task
        for task in corpus["tasks"]
        if (not task_ids or task["id"] in task_ids)
        and (not tags or set(tags).intersection(task.get("tags", [])))
    ]
    return tasks[:max_tasks] if max_tasks is not None else tasks


def context_padding(lines: int) -> str:
    if lines <= 0:
        return ""
    records = [
        f"padding record {index:05d}: alpha={(index * 17) % 1009}; "
        f"beta={(index * 43) % 1013}; unrelated=true"
        for index in range(lines)
    ]
    return (
        "The following synthetic records are intentionally unrelated to the problem. "
        "Ignore their values; they exist only to measure prompt prefill.\n"
        + "\n".join(records)
        + "\n\n"
    )


def build_request(
    task: dict[str, Any],
    model: str,
    *,
    num_ctx: int,
    num_predict: int,
    temperature: float | None,
    seed: int | None,
    keep_alive: str,
    padding_lines: int,
    mtp_mode: str,
    answer_format: str = "marker",
) -> dict[str, Any]:
    options: dict[str, Any] = {"num_ctx": num_ctx, "num_predict": num_predict}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": JSON_SYSTEM_PROMPT
                if answer_format == "json-schema"
                else SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": context_padding(padding_lines) + task["prompt"],
            },
        ],
        "stream": True,
        "think": True,
        "keep_alive": keep_alive,
        "options": options,
    }
    if answer_format == "json-schema":
        body["format"] = FINAL_JSON_SCHEMA
    if mtp_mode == "off-logprobs":
        body["logprobs"] = True
        body["top_logprobs"] = 0
    return body


def request_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> tuple[int, Any]:
    data = json_bytes(body) if body is not None else None
    headers = {
        "Accept": "application/json",
        "User-Agent": "generalist-local-reasoning-benchmark/1",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read().decode("utf-8", errors="replace")
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, {"raw": raw}


def assemble_stream(events: list[tuple[float, dict[str, Any]]]) -> dict[str, Any]:
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    first_chunk_ms: float | None = None
    first_thinking_ms: float | None = None
    first_answer_ms: float | None = None
    final: dict[str, Any] = {}
    for elapsed_ms, chunk in events:
        if first_chunk_ms is None:
            first_chunk_ms = elapsed_ms
        message = chunk.get("message")
        if isinstance(message, dict):
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking:
                if first_thinking_ms is None:
                    first_thinking_ms = elapsed_ms
                thinking_parts.append(thinking)
            content = message.get("content")
            if isinstance(content, str) and content:
                if first_answer_ms is None:
                    first_answer_ms = elapsed_ms
                answer_parts.append(content)
        if chunk.get("done") is True:
            final = chunk
    server_fields = {
        key: final.get(key)
        for key in (
            "done",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
    }
    return {
        "thinking": "".join(thinking_parts),
        "answer": "".join(answer_parts),
        "first_chunk_ms": first_chunk_ms,
        "first_thinking_ms": first_thinking_ms,
        "first_answer_ms": first_answer_ms,
        "server": server_fields,
    }


def stream_chat(
    url: str, body: dict[str, Any], timeout_seconds: float
) -> tuple[int, dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json_bytes(body),
        headers={
            "Accept": "application/x-ndjson",
            "Content-Type": "application/json",
            "User-Agent": "generalist-local-reasoning-benchmark/1",
        },
        method="POST",
    )
    started = time.monotonic()
    events: list[tuple[float, dict[str, Any]]] = []
    parse_errors: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            for raw_line in response:
                elapsed_ms = (time.monotonic() - started) * 1000
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as error:
                    parse_errors.append(str(error))
                    continue
                if isinstance(chunk, dict):
                    events.append((elapsed_ms, chunk))
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return status, {"error": parsed}, (time.monotonic() - started) * 1000
    elapsed_ms = (time.monotonic() - started) * 1000
    result = assemble_stream(events)
    result["event_count"] = len(events)
    result["parse_errors"] = parse_errors
    return status, result, elapsed_ms


def milliseconds(nanoseconds: Any) -> float | None:
    if not isinstance(nanoseconds, (int, float)):
        return None
    return round(nanoseconds / 1_000_000, 3)


def tokens_per_second(count: Any, nanoseconds: Any) -> float | None:
    if not isinstance(count, (int, float)) or not isinstance(nanoseconds, (int, float)):
        return None
    if nanoseconds <= 0:
        return None
    return round(count / (nanoseconds / 1_000_000_000), 3)


def check_quality(
    task: dict[str, Any], response: dict[str, Any], answer_format: str = "marker"
) -> dict[str, Any]:
    thinking = response.get("thinking")
    answer = response.get("answer")
    server = response.get("server") or {}
    done = server.get("done") is True
    done_reason = server.get("done_reason")
    extracted = None
    answer_contract_met = False
    if answer_format == "json-schema" and isinstance(answer, str):
        stripped_answer = answer.strip()
        object_start = stripped_answer.rfind("{")
        candidate = stripped_answer[object_start:] if object_start >= 0 else ""
        try:
            parsed_answer = json.loads(candidate)
        except json.JSONDecodeError:
            parsed_answer = None
        if isinstance(parsed_answer, dict) and set(parsed_answer) == {"final"}:
            value = parsed_answer.get("final")
            extracted = value.strip() if isinstance(value, str) else None
            answer_contract_met = (
                candidate == stripped_answer and extracted == task["expected"]
            )
    elif isinstance(answer, str):
        match = re.search(
            r"(?:^|\n)[ \t]*FINAL:[ \t]*(?P<value>[^\r\n]+?)[ \t]*\Z", answer
        )
        extracted = match.group("value").strip() if match else None
        expected_answer = f"FINAL: {task['expected']}"
        answer_contract_met = answer.strip() == expected_answer
    reasoning_present = isinstance(thinking, str) and bool(thinking.strip())
    completed = done and done_reason not in {"length", "max_tokens"}
    correct = extracted == task["expected"]
    if not done:
        failure_kind = "incomplete_stream"
    elif done_reason in {"length", "max_tokens"}:
        failure_kind = "output_limit"
    elif not reasoning_present:
        failure_kind = "missing_reasoning"
    elif extracted is None:
        failure_kind = "missing_final"
    elif not correct:
        failure_kind = "wrong_answer"
    else:
        failure_kind = None
    return {
        "passed": failure_kind is None,
        "failure_kind": failure_kind,
        "reasoning_present": reasoning_present,
        "completed": completed,
        "expected": task["expected"],
        "extracted": extracted,
        "correct": correct,
        "answer_contract_met": answer_contract_met,
        "answer_format": answer_format,
    }


def log_cursor(path: pathlib.Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        status = path.stat()
    except OSError:
        return None
    return status.st_ino, status.st_size


def parse_speculation_stats(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        match = SPECULATION_STATS_PATTERN.search(line)
        if match is None:
            continue
        records.append(
            {
                "iterations": int(match.group("iterations")),
                "drafted": int(match.group("drafted")),
                "accepted": int(match.group("accepted")),
                "acceptance": float(match.group("acceptance")),
                "avg_draft": float(match.group("avg_draft")),
                "max_draft": int(match.group("max_draft")),
                "avg_accepted": float(match.group("avg_accepted")),
                "depth_over_time": match.group("depth_over_time"),
            }
        )
    return records


def aggregate_speculation_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    iterations = sum(record["iterations"] for record in records)
    drafted = sum(record["drafted"] for record in records)
    accepted = sum(record["accepted"] for record in records)
    return {
        "passes": len(records),
        "iterations": iterations,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": round(accepted / drafted, 4) if drafted else 0.0,
        "avg_draft": round(drafted / iterations, 4) if iterations else 0.0,
        "avg_accepted": round(accepted / iterations, 4) if iterations else 0.0,
        "max_draft": max((record["max_draft"] for record in records), default=0),
        # Every speculative round emits its accepted prefix plus one target token,
        # apart from terminal/budget edge handling. This recovers a useful total
        # when Ollama reports API eval_count for only the last pass.
        "estimated_emitted_tokens": iterations + accepted,
    }


def capture_speculation_stats(
    path: pathlib.Path | None,
    cursor: tuple[int, int] | None,
    wait_seconds: float = 1.0,
) -> dict[str, Any] | None:
    if path is None or cursor is None:
        return None
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            status = path.stat()
            offset = cursor[1] if status.st_ino == cursor[0] else 0
            with path.open(encoding="utf-8", errors="replace") as source:
                source.seek(offset)
                appended = source.read()
        except OSError:
            return None
        records = parse_speculation_stats(appended)
        if records:
            return {
                "source": str(path),
                "record_count": len(records),
                "records": records,
                "aggregate": aggregate_speculation_stats(records),
                "selected": records[-1],
            }
        if time.monotonic() >= deadline:
            return {
                "source": str(path),
                "record_count": 0,
                "records": [],
                "aggregate": None,
                "selected": None,
            }
        time.sleep(0.05)


def compact_model_snapshot(show: Any, tags: Any, model: str) -> dict[str, Any]:
    if not isinstance(show, dict):
        return {"error": "show response was not an object"}
    tensors = show.get("tensors") if isinstance(show.get("tensors"), list) else []
    groups: dict[str, Counter[str]] = {
        "language": Counter(),
        "vision": Counter(),
        "mtp": Counter(),
        "other": Counter(),
    }
    full_attention_layers: set[int] = set()
    linear_attention_layers: set[int] = set()
    mtp_fc: dict[str, Any] | None = None
    for tensor in tensors:
        if not isinstance(tensor, dict):
            continue
        name = str(tensor.get("name", ""))
        lowered = name.lower()
        tensor_type = str(tensor.get("type", "unknown"))
        if lowered.startswith("mtp.") or ".mtp." in lowered:
            group = "mtp"
        elif "vision" in lowered or "visual" in lowered:
            group = "vision"
        elif "language_model" in lowered or "model.layers." in lowered or "blk." in lowered:
            group = "language"
        else:
            group = "other"
        groups[group][tensor_type] += 1
        layer_match = re.search(r"(?:layers|blk)\.(\d+)", lowered)
        if layer_match and group == "language":
            layer = int(layer_match.group(1))
            if "linear_attn" in lowered or "ssm" in lowered or "mamba" in lowered:
                linear_attention_layers.add(layer)
            elif "self_attn" in lowered or ".attn_" in lowered:
                full_attention_layers.add(layer)
        if lowered.endswith("mtp.fc.weight") or lowered == "mtp.fc.weight":
            mtp_fc = {
                "name": name,
                "type": tensor.get("type"),
                "shape": tensor.get("shape"),
            }
    tagged_model = None
    if isinstance(tags, dict) and isinstance(tags.get("models"), list):
        for candidate in tags["models"]:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("name") == model or candidate.get("model") == model:
                tagged_model = {
                    key: candidate.get(key)
                    for key in ("name", "model", "digest", "size", "modified_at")
                }
                break
    model_info = show.get("model_info") if isinstance(show.get("model_info"), dict) else {}
    selected_info = {
        key: value
        for key, value in model_info.items()
        if key == "general.architecture"
        or key == "general.parameter_count"
        or key.endswith(".context_length")
        or key.endswith(".block_count")
        or key.endswith(".embedding_length")
        or key.endswith(".attention.head_count")
        or key.endswith(".attention.head_count_kv")
        or key.endswith(".linear_attention.head_count_key")
        or key.endswith(".linear_attention.head_count_value")
        or key.endswith(".mtp_num_hidden_layers")
    }
    return {
        "details": show.get("details"),
        "capabilities": show.get("capabilities"),
        "required_ollama_version": show.get("requires"),
        "tag": tagged_model,
        "model_info": selected_info,
        "tensors": {
            "count": len(tensors),
            "groups_by_type": {
                group: dict(sorted(type_counts.items()))
                for group, type_counts in groups.items()
            },
            "full_attention_layers": sorted(full_attention_layers),
            "linear_attention_layers": sorted(linear_attention_layers),
            "mtp_fc": mtp_fc,
        },
    }


def inspect_model(base_url: str, model: str, timeout_seconds: float) -> dict[str, Any]:
    show_status, show = request_json(
        base_url.rstrip("/") + "/api/show",
        method="POST",
        body={"model": model, "verbose": True},
        timeout_seconds=timeout_seconds,
    )
    tags_status, tags = request_json(
        base_url.rstrip("/") + "/api/tags", timeout_seconds=timeout_seconds
    )
    version_status, version = request_json(
        base_url.rstrip("/") + "/api/version", timeout_seconds=timeout_seconds
    )
    if not 200 <= show_status < 300:
        raise RuntimeError(f"/api/show returned HTTP {show_status}: {show}")
    snapshot = compact_model_snapshot(show, tags, model)
    snapshot["ollama_version"] = version.get("version") if version_status == 200 else None
    snapshot["inspection_status"] = {
        "show": show_status,
        "tags": tags_status,
        "version": version_status,
    }
    return snapshot


def active_model_snapshot(base_url: str, model: str, timeout_seconds: float) -> dict[str, Any] | None:
    try:
        status, payload = request_json(
            base_url.rstrip("/") + "/api/ps", timeout_seconds=timeout_seconds
        )
    except Exception:
        return None
    if status != 200 or not isinstance(payload, dict):
        return None
    models = payload.get("models") if isinstance(payload.get("models"), list) else []
    for candidate in models:
        if isinstance(candidate, dict) and (
            candidate.get("name") == model or candidate.get("model") == model
        ):
            return {
                key: candidate.get(key)
                for key in ("name", "model", "digest", "size", "size_vram", "expires_at")
            }
    return None


def unload_model(base_url: str, model: str, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    status, response = request_json(
        base_url.rstrip("/") + "/api/generate",
        method="POST",
        body={"model": model, "keep_alive": 0},
        timeout_seconds=timeout_seconds,
    )
    return {
        "status": status,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "response": response,
    }


def unload_and_cool(
    base_url: str,
    model: str,
    timeout_seconds: float,
    cooldown_seconds: float,
) -> dict[str, Any]:
    result = unload_model(base_url, model, timeout_seconds)
    result["cooldown_seconds"] = cooldown_seconds
    if cooldown_seconds > 0:
        time.sleep(cooldown_seconds)
    return result


def run_attempt(
    *,
    task: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    repetition: int,
) -> dict[str, Any]:
    request = build_request(
        task,
        args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        temperature=args.temperature,
        seed=args.seed,
        keep_alive=args.keep_alive,
        padding_lines=args.padding_lines,
        mtp_mode=args.mtp_mode,
        answer_format=args.answer_format,
    )
    started_at = utc_now()
    reload_result = None
    try:
        if args.reload_between:
            reload_result = unload_and_cool(
                args.base_url,
                args.model,
                args.timeout_seconds,
                args.cooldown_seconds,
            )
        stats_cursor = log_cursor(args.ollama_log)
        status, response, wall_ms = stream_chat(
            args.base_url.rstrip("/") + "/api/chat", request, args.timeout_seconds
        )
        speculation = capture_speculation_stats(args.ollama_log, stats_cursor)
        if not 200 <= status < 300:
            classification = "http_error"
            quality = None
        else:
            quality = check_quality(task, response, args.answer_format)
            classification = "pass" if quality["passed"] else quality["failure_kind"]
        server = response.get("server") if isinstance(response, dict) else {}
        server = server if isinstance(server, dict) else {}
        thinking = response.get("thinking", "") if isinstance(response, dict) else ""
        answer = response.get("answer", "") if isinstance(response, dict) else ""
        selected_speculation = (speculation or {}).get("selected") or {}
        aggregate_speculation = (speculation or {}).get("aggregate") or {}
        metrics = {
            "wall_ms": round(wall_ms, 3),
            "first_chunk_ms": response.get("first_chunk_ms")
            if isinstance(response, dict)
            else None,
            "first_thinking_ms": response.get("first_thinking_ms")
            if isinstance(response, dict)
            else None,
            "first_answer_ms": response.get("first_answer_ms")
            if isinstance(response, dict)
            else None,
            "answer_tail_ms": round(
                wall_ms - response["first_answer_ms"], 3
            )
            if isinstance(response, dict)
            and isinstance(response.get("first_answer_ms"), (int, float))
            else None,
            "total_duration_ms": milliseconds(server.get("total_duration")),
            "load_duration_ms": milliseconds(server.get("load_duration")),
            "prompt_eval_count": server.get("prompt_eval_count"),
            "prompt_eval_duration_ms": milliseconds(server.get("prompt_eval_duration")),
            "prompt_tokens_per_second": tokens_per_second(
                server.get("prompt_eval_count"), server.get("prompt_eval_duration")
            ),
            "eval_count": server.get("eval_count"),
            "eval_duration_ms": milliseconds(server.get("eval_duration")),
            "decode_tokens_per_second": tokens_per_second(
                server.get("eval_count"), server.get("eval_duration")
            ),
            "thinking_chars": len(thinking) if isinstance(thinking, str) else None,
            "answer_chars": len(answer) if isinstance(answer, str) else None,
            "runtime_inference_passes": aggregate_speculation.get("passes"),
            "runtime_estimated_emitted_tokens": aggregate_speculation.get(
                "estimated_emitted_tokens"
            ),
            "mtp_acceptance": aggregate_speculation.get(
                "acceptance", selected_speculation.get("acceptance")
            ),
            "mtp_avg_draft": aggregate_speculation.get(
                "avg_draft", selected_speculation.get("avg_draft")
            ),
            "mtp_avg_accepted": aggregate_speculation.get(
                "avg_accepted", selected_speculation.get("avg_accepted")
            ),
        }
        return {
            "schema_version": 1,
            "record_type": "attempt",
            "run_id": run_id,
            "started_at": started_at,
            "variant": args.variant,
            "model": args.model,
            "task_id": task["id"],
            "task_tags": task.get("tags", []),
            "repetition": repetition,
            "http_status": status,
            "classification": classification,
            "quality": quality,
            "metrics": metrics,
            "reload": reload_result,
            "thinking": thinking,
            "answer": answer,
            "server": server,
            "stream": {
                "event_count": response.get("event_count"),
                "parse_errors": response.get("parse_errors"),
            }
            if isinstance(response, dict)
            else None,
            "speculation": speculation,
            "request_profile": {
                "think": True,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "temperature": args.temperature,
                "seed": args.seed,
                "padding_lines": args.padding_lines,
                "mtp_mode": args.mtp_mode,
                "answer_format": args.answer_format,
            },
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "record_type": "attempt",
            "run_id": run_id,
            "started_at": started_at,
            "variant": args.variant,
            "model": args.model,
            "task_id": task["id"],
            "task_tags": task.get("tags", []),
            "repetition": repetition,
            "classification": "client_error",
            "error": f"{type(error).__name__}: {error}",
            "reload": reload_result,
            "request_profile": {
                "think": True,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "temperature": args.temperature,
                "seed": args.seed,
                "padding_lines": args.padding_lines,
                "mtp_mode": args.mtp_mode,
                "answer_format": args.answer_format,
            },
        }


def median_metric(attempts: list[dict[str, Any]], key: str) -> float | None:
    values = [
        attempt["metrics"][key]
        for attempt in attempts
        if isinstance(attempt.get("metrics"), dict)
        and isinstance(attempt["metrics"].get(key), (int, float))
    ]
    return round(float(statistics.median(values)), 3) if values else None


def aggregate_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(attempt.get("classification", "unknown") for attempt in attempts)
    quality_records = [
        attempt["quality"]
        for attempt in attempts
        if isinstance(attempt.get("quality"), dict)
    ]
    return {
        "attempts": len(attempts),
        "passed": counts.get("pass", 0),
        "pass_rate": round(counts.get("pass", 0) / len(attempts), 4)
        if attempts
        else None,
        "classifications": dict(sorted(counts.items())),
        "reasoning_present": sum(
            quality.get("reasoning_present") is True for quality in quality_records
        ),
        "completed": sum(quality.get("completed") is True for quality in quality_records),
        "correct": sum(quality.get("correct") is True for quality in quality_records),
        "compact_answer": sum(
            quality.get("answer_contract_met") is True for quality in quality_records
        ),
        "medians": {
            key: median_metric(attempts, key)
            for key in (
                "wall_ms",
                "first_thinking_ms",
                "first_answer_ms",
                "answer_tail_ms",
                "load_duration_ms",
                "prompt_tokens_per_second",
                "decode_tokens_per_second",
                "eval_count",
                "runtime_inference_passes",
                "runtime_estimated_emitted_tokens",
                "thinking_chars",
                "mtp_acceptance",
                "mtp_avg_draft",
                "mtp_avg_accepted",
            )
        },
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", default="official-nvfp4")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--num-ctx", type=int, default=32_768)
    parser.add_argument("--num-predict", type=int, default=4_096)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--padding-lines", type=int, default=0)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument(
        "--mtp-mode",
        choices=("auto", "off-logprobs"),
        default="auto",
        help="keep Ollama MTP automatic or park it via native logprobs for a control",
    )
    parser.add_argument(
        "--answer-format",
        choices=("marker", "json-schema"),
        default="marker",
        help="plain FINAL marker or a schema-constrained compact final answer",
    )
    parser.add_argument(
        "--ollama-log",
        type=pathlib.Path,
        default=DEFAULT_OLLAMA_LOG,
        help="server log used to attach per-request MTP acceptance statistics",
    )
    parser.add_argument("--reload-between", action="store_true")
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0,
        help="after each unload, wait this long before the next cold load",
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--corpus", type=pathlib.Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.num_ctx < 1 or args.num_predict < 1:
        parser.error("repeat, num-ctx, and num-predict must be positive")
    if args.timeout_seconds <= 0 or args.padding_lines < 0 or args.cooldown_seconds < 0:
        parser.error(
            "timeout-seconds must be positive; padding-lines and cooldown-seconds "
            "cannot be negative"
        )
    if args.cooldown_seconds > 0 and not args.reload_between:
        parser.error("--cooldown-seconds requires --reload-between")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("temperature must be between 0 and 2")
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("max-tasks must be positive")
    return args


def default_output(model: str, variant: str) -> pathlib.Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return HERE / "results" / f"{timestamp}-{slug(model)}-{slug(variant)}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    corpus = load_corpus(args.corpus)
    tasks = selected_tasks(corpus, args.task, args.tag, args.max_tasks)
    if not tasks and not args.inspect_only:
        raise SystemExit("no tasks matched the selection")
    output = (args.output or default_output(args.model, args.variant)).resolve()
    run_id = str(uuid.uuid4())
    writer = JsonlWriter(output, args.append)
    inspection_error = None
    try:
        model_snapshot = inspect_model(args.base_url, args.model, args.timeout_seconds)
    except Exception as error:
        model_snapshot = None
        inspection_error = f"{type(error).__name__}: {error}"
    expected_attempts = 0 if args.inspect_only else args.repeat * len(tasks)
    writer.write(
        {
            "schema_version": 1,
            "record_type": "run_start",
            "run_id": run_id,
            "started_at": utc_now(),
            "variant": args.variant,
            "model": args.model,
            "base_url": args.base_url,
            "task_ids": [] if args.inspect_only else [task["id"] for task in tasks],
            "repeat": args.repeat,
            "expected_attempts": expected_attempts,
            "reasoning_contract": {
                "think": True,
                "thinking_required": True,
                "completed_response_required": True,
                "exact_final_answer_required": True,
                "reasoning_can_be_disabled": False,
            },
            "request_profile": {
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
                "temperature": args.temperature,
                "seed": args.seed,
                "padding_lines": args.padding_lines,
                "keep_alive": args.keep_alive,
                "reload_between": args.reload_between,
                "cooldown_seconds": args.cooldown_seconds,
                "mtp_mode": args.mtp_mode,
                "answer_format": args.answer_format,
                "ollama_log": str(args.ollama_log) if args.ollama_log else None,
            },
            "corpus": str(args.corpus.resolve()),
            "artifacts": {
                "runner_sha256": sha256_file(pathlib.Path(__file__).resolve()),
                "corpus_sha256": sha256_file(args.corpus.resolve()),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "model_snapshot": model_snapshot,
            "inspection_error": inspection_error,
        }
    )
    attempts: list[dict[str, Any]] = []
    interrupted = False
    try:
        if not args.inspect_only:
            for repetition in range(1, args.repeat + 1):
                for task in tasks:
                    record = run_attempt(
                        task=task, args=args, run_id=run_id, repetition=repetition
                    )
                    writer.write(record)
                    attempts.append(record)
                    decode_tps = (record.get("metrics") or {}).get(
                        "decode_tokens_per_second"
                    )
                    suffix = f" decode={decode_tps:.2f} tok/s" if decode_tps else ""
                    print(
                        f"{task['id']:<26} {record['classification']}{suffix}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        aggregate = aggregate_attempts(attempts)
        complete = not interrupted and len(attempts) == expected_attempts
        writer.write(
            {
                "schema_version": 1,
                "record_type": "run_end",
                "run_id": run_id,
                "finished_at": utc_now(),
                "status": "complete" if complete else "interrupted",
                "expected_attempts": expected_attempts,
                "aggregate": aggregate,
                "active_model": active_model_snapshot(
                    args.base_url, args.model, min(args.timeout_seconds, 10)
                ),
            }
        )
        writer.close()
    if interrupted:
        print(f"interrupted; saved {len(attempts)}/{expected_attempts} attempts to {output}")
        return 130
    print(
        f"saved {len(attempts)} attempts ({aggregate['passed']} passed) to {output}",
        flush=True,
    )
    if args.require_pass and (
        inspection_error is not None or aggregate["passed"] != expected_attempts
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
