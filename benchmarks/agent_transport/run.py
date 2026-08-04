#!/usr/bin/env python3
"""Benchmark model-facing Python transport without executing generated code."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import pathlib
import platform
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "tasks.json"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"
TRANSPORTS = ("json_tool", "plain_text", "responses_custom")
UNRESOLVED_KEY = "$unresolved"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return cleaned or "model"


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class JsonlWriter:
    """Flush every record so an interrupted benchmark remains inspectable."""

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
    if not isinstance(corpus.get("tasks"), list) or not corpus["tasks"]:
        raise ValueError(f"corpus has no tasks: {path}")
    if not isinstance(corpus.get("tool_catalog"), dict):
        raise ValueError(f"corpus has no tool catalog: {path}")
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


def bridge_docs(corpus: dict[str, Any], task: dict[str, Any]) -> str:
    rows = []
    for name in task["available_tools"]:
        definition = corpus["tool_catalog"][name]
        schema = json.dumps(definition["input_schema"], ensure_ascii=False, separators=(",", ":"))
        rows.append(f"- tools.{name}(**kwargs): {definition['description']}\n  Input schema: {schema}")
    return "\n".join(rows)


def task_prompt(task: dict[str, Any]) -> str:
    expected = json.dumps(task["expected_calls"], ensure_ascii=False, indent=2)
    return (
        f"{task['instruction']}\n\n"
        "Required bridge calls, in lexical execution order:\n"
        f"{expected}\n\n"
        "JSON escapes above describe the desired runtime values. Express fixed values as "
        "ordinary Python literals, either directly in calls or in constant assignments. "
        "Do not use base64, eval, or an embedded JSON decoder. A {$any: true} marker means "
        "that argument must be computed from earlier tool results."
    )


def system_prompt(transport: str) -> str:
    output_rule = {
        "json_tool": "Call the advertised python function tool exactly once.",
        "plain_text": "Return only raw Python source, with no Markdown fence or prose.",
        "responses_custom": "Call the advertised custom python tool exactly once.",
    }[transport]
    return (
        "You are generating one self-contained Python 3 agent script for a transport benchmark. "
        "Start with `import tools`. Call bridge functions only with keyword arguments. Do not "
        "invent tools, perform the work in prose, or print payloads. "
        + output_rule
    )


def python_tool_description(corpus: dict[str, Any], task: dict[str, Any]) -> str:
    return (
        "Execute one self-contained Python 3 script. Start with `import tools`; bridge functions "
        "return str and may raise RuntimeError. Complete all requested calls in one script.\n\n"
        "Available bridge tools:\n"
        + bridge_docs(corpus, task)
    )


def build_request(
    corpus: dict[str, Any],
    task: dict[str, Any],
    model: str,
    transport: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> tuple[str, dict[str, Any]]:
    system = system_prompt(transport)
    prompt = task_prompt(task)
    description = python_tool_description(corpus, task)
    if transport in {"json_tool", "plain_text"}:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": max_output_tokens,
        }
        if transport == "json_tool":
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "python",
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "The complete Python 3 script",
                                }
                            },
                            "required": ["code"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
            body["tool_choice"] = {"type": "function", "function": {"name": "python"}}
            body["parallel_tool_calls"] = False
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        return "/chat/completions", body

    body = {
        "model": model,
        "instructions": system,
        "input": prompt,
        "tools": [{"type": "custom", "name": "python", "description": description}],
        "tool_choice": {"type": "custom", "name": "python"},
        "parallel_tool_calls": False,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return "/responses", body


def post_json(
    url: str, body: dict[str, Any], api_key: str | None, timeout_seconds: float
) -> tuple[int, Any, float]:
    request_bytes = json_bytes(body)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "generalist-agent-transport-benchmark/1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=request_bytes, headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read().decode("utf-8", errors="replace")
    elapsed_ms = (time.monotonic() - started) * 1000
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"_raw_response": raw}
    return status, parsed, elapsed_ms


def response_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return ""


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_chat_code(response: Any, transport: str) -> tuple[str | None, dict[str, Any]]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None, {"error": "response missing choices[0].message", "tool_call_count": 0}
    if transport == "plain_text":
        text = response_text(message)
        if not text:
            return None, {"error": "assistant response contained no text", "tool_call_count": 0}
        return strip_code_fence(text), {"format": "assistant_text", "tool_call_count": 0}

    tool_calls = message.get("tool_calls") or []
    python_calls = [
        call
        for call in tool_calls
        if isinstance(call, dict) and call.get("function", {}).get("name") == "python"
    ]
    details: dict[str, Any] = {
        "format": "chat_function",
        "tool_call_count": len(tool_calls),
        "python_call_count": len(python_calls),
    }
    if len(python_calls) != 1:
        details["error"] = f"expected one python call, received {len(python_calls)}"
        return None, details
    arguments = python_calls[0].get("function", {}).get("arguments", "")
    details["raw_arguments_bytes"] = len(
        arguments.encode("utf-8") if isinstance(arguments, str) else json_bytes(arguments)
    )
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            details["error"] = f"invalid function arguments JSON: {error}"
            return None, details
    if not isinstance(arguments, dict) or not isinstance(arguments.get("code"), str):
        details["error"] = "python arguments missing string field `code`"
        return None, details
    return arguments["code"], details


def extract_responses_code(response: Any) -> tuple[str | None, dict[str, Any]]:
    output = response.get("output", []) if isinstance(response, dict) else []
    calls = [item for item in output if isinstance(item, dict) and item.get("type") == "custom_tool_call"]
    python_calls = [item for item in calls if item.get("name") == "python"]
    details: dict[str, Any] = {
        "format": "responses_custom",
        "tool_call_count": len(calls),
        "python_call_count": len(python_calls),
    }
    if len(python_calls) != 1:
        details["error"] = f"expected one custom python call, received {len(python_calls)}"
        return None, details
    code = python_calls[0].get("input")
    if not isinstance(code, str):
        details["error"] = "custom python call missing string field `input`"
        return None, details
    details["raw_arguments_bytes"] = len(code.encode("utf-8"))
    return code, details


def unresolved(node: ast.AST) -> dict[str, str]:
    try:
        source = ast.unparse(node)
    except Exception:  # ast.unparse should not make a benchmark crash.
        source = type(node).__name__
    return {UNRESOLVED_KEY: source}


def safe_value(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id, unresolved(node))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [safe_value(item, constants) for item in node.elts]
        return values
    if isinstance(node, ast.Dict):
        keys = [safe_value(item, constants) for item in node.keys]
        values = [safe_value(item, constants) for item in node.values]
        if any(isinstance(key, dict) and UNRESOLVED_KEY in key for key in keys):
            return unresolved(node)
        try:
            return dict(zip(keys, values, strict=True))
        except (TypeError, ValueError):
            return unresolved(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = safe_value(node.operand, constants)
        if isinstance(operand, (int, float)):
            return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = safe_value(node.left, constants)
        right = safe_value(node.right, constants)
        if not (isinstance(left, dict) or isinstance(right, dict)):
            try:
                return left + right
            except TypeError:
                pass
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            elif isinstance(part, ast.FormattedValue):
                value = safe_value(part.value, constants)
                if isinstance(value, dict):
                    return unresolved(node)
                parts.append(str(value))
            else:
                return unresolved(node)
        return "".join(parts)
    return unresolved(node)


def constant_bindings(tree: ast.AST) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assignments.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))
    for assignment in assignments:
        value_node = assignment.value
        value = safe_value(value_node, constants)
        if isinstance(value, dict) and UNRESOLVED_KEY in value:
            continue
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = value
    return constants


def tool_aliases(tree: ast.AST) -> tuple[set[str], dict[str, str]]:
    module_aliases = {"tools"}
    function_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tools":
                    module_aliases.add(alias.asname or "tools")
        elif isinstance(node, ast.ImportFrom) and node.module == "tools":
            for alias in node.names:
                function_aliases[alias.asname or alias.name] = alias.name
    return module_aliases, function_aliases


def tool_call_name(
    call: ast.Call, module_aliases: set[str], function_aliases: dict[str, str]
) -> str | None:
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in module_aliases
    ):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return function_aliases.get(call.func.id)
    return None


def extract_source_calls(source: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [], f"{error.msg} at line {error.lineno}, column {error.offset}"
    constants = constant_bindings(tree)
    module_aliases, function_aliases = tool_aliases(tree)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    extracted = []
    for call in calls:
        name = tool_call_name(call, module_aliases, function_aliases)
        if name is None:
            continue
        arguments = {keyword.arg: safe_value(keyword.value, constants) for keyword in call.keywords if keyword.arg}
        if call.args:
            arguments["$positional"] = [safe_value(argument, constants) for argument in call.args]
        if any(keyword.arg is None for keyword in call.keywords):
            arguments["$expanded_kwargs"] = True
        extracted.append({"name": name, "arguments": arguments, "line": call.lineno})
    return extracted, None


def value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and expected == {"$any": True}:
        return True
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(expected) == set(actual) and all(
            value_matches(value, actual[key]) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            value_matches(left, right) for left, right in zip(expected, actual, strict=True)
        )
    return expected == actual


def check_source(source: str, expected_calls: list[dict[str, Any]]) -> dict[str, Any]:
    actual_calls, syntax_error = extract_source_calls(source)
    comparable_calls = [
        {"name": call["name"], "arguments": call["arguments"]} for call in actual_calls
    ]
    if syntax_error:
        return {
            "passed": False,
            "failure_kind": "syntax_error",
            "detail": syntax_error,
            "actual_calls": comparable_calls,
        }
    if len(expected_calls) != len(comparable_calls):
        return {
            "passed": False,
            "failure_kind": "call_count",
            "detail": f"expected {len(expected_calls)} bridge calls, found {len(comparable_calls)}",
            "actual_calls": comparable_calls,
        }
    for index, (expected, actual) in enumerate(
        zip(expected_calls, comparable_calls, strict=True), start=1
    ):
        if expected["name"] != actual["name"]:
            return {
                "passed": False,
                "failure_kind": "call_name",
                "detail": f"call {index}: expected {expected['name']}, found {actual['name']}",
                "actual_calls": comparable_calls,
            }
        if not value_matches(expected["arguments"], actual["arguments"]):
            return {
                "passed": False,
                "failure_kind": "call_arguments",
                "detail": f"call {index}: arguments differ",
                "actual_calls": comparable_calls,
            }
    return {
        "passed": True,
        "failure_kind": None,
        "detail": "all expected bridge calls matched",
        "actual_calls": comparable_calls,
    }


def normalized_usage(response: Any) -> dict[str, Any] | None:
    if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
        return None
    usage = response["usage"]
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        "reported_cost": usage.get("cost"),
    }


def classify(http_status: int, extraction: dict[str, Any], checker: dict[str, Any] | None) -> str:
    if not 200 <= http_status < 300:
        return "http_error"
    if extraction.get("error"):
        return "extraction_error"
    if checker is None:
        return "internal_error"
    return "pass" if checker["passed"] else checker["failure_kind"]


def run_attempt(
    *,
    corpus: dict[str, Any],
    task: dict[str, Any],
    provider: str,
    base_url: str,
    model: str,
    transport: str,
    repetition: int,
    api_key: str | None,
    timeout_seconds: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
    run_id: str,
) -> dict[str, Any]:
    path, body = build_request(
        corpus, task, model, transport, max_output_tokens, reasoning_effort
    )
    endpoint = base_url.rstrip("/") + path
    started_at = utc_now()
    try:
        http_status, response, elapsed_ms = post_json(endpoint, body, api_key, timeout_seconds)
        if transport == "responses_custom":
            code, extraction = extract_responses_code(response)
        else:
            code, extraction = extract_chat_code(response, transport)
        checker = check_source(code, task["expected_calls"]) if code is not None else None
        metrics: dict[str, Any] = {
            "latency_ms": round(elapsed_ms, 3),
            "request_bytes": len(json_bytes(body)),
            "response_bytes": len(json_bytes(response)),
            "code_bytes": len(code.encode("utf-8")) if code is not None else None,
            "json_encoded_code_bytes": len(json.dumps(code, ensure_ascii=False).encode("utf-8"))
            if code is not None
            else None,
            "usage": normalized_usage(response),
        }
        if code is not None:
            metrics["model_payload_bytes"] = extraction.get(
                "raw_arguments_bytes", metrics["code_bytes"]
            )
            metrics["model_payload_overhead_bytes"] = (
                metrics["model_payload_bytes"] - metrics["code_bytes"]
            )
            metrics["json_escape_overhead_bytes"] = (
                metrics["json_encoded_code_bytes"] - metrics["code_bytes"]
            )
        classification = classify(http_status, extraction, checker)
        return {
            "schema_version": 1,
            "record_type": "attempt",
            "run_id": run_id,
            "started_at": started_at,
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "transport": transport,
            "task_id": task["id"],
            "task_tags": task.get("tags", []),
            "repetition": repetition,
            "http_status": http_status,
            "classification": classification,
            "extraction": extraction,
            "checker": checker,
            "metrics": metrics,
            "code": code,
            "request": body,
            "response": response,
        }
    except Exception as error:  # Preserve the rest of a matrix after one transport failure.
        return {
            "schema_version": 1,
            "record_type": "attempt",
            "run_id": run_id,
            "started_at": started_at,
            "provider": provider,
            "base_url": base_url,
            "model": model,
            "transport": transport,
            "task_id": task["id"],
            "task_tags": task.get("tags", []),
            "repetition": repetition,
            "classification": "client_error",
            "error": f"{type(error).__name__}: {error}",
            "request": body,
        }


def provider_defaults(provider: str) -> tuple[str, str | None]:
    if provider == "ollama":
        return DEFAULT_OLLAMA_URL, None
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_URL, "OPENROUTER_API_KEY"
    if provider == "openai":
        return DEFAULT_OPENAI_URL, "OPENAI_API_KEY"
    return "", "OPENAI_API_KEY"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("ollama", "openrouter", "openai", "custom"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--transport", action="append", choices=TRANSPORTS)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--reasoning-effort", choices=("minimal", "low", "medium", "high", "xhigh"))
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--corpus", type=pathlib.Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.max_output_tokens < 1 or args.timeout_seconds <= 0:
        parser.error("repeat, max-output-tokens, and timeout-seconds must be positive")
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("max-tasks must be positive")
    default_url, default_key_env = provider_defaults(args.provider)
    args.base_url = args.base_url or default_url
    if not args.base_url:
        parser.error("--base-url is required for provider=custom")
    args.api_key_env = args.api_key_env or default_key_env
    args.transports = args.transport or ["json_tool"]
    return args


def default_output(provider: str, model: str) -> pathlib.Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return HERE / "results" / f"{timestamp}-{slug(provider)}-{slug(model)}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    corpus = load_corpus(args.corpus)
    tasks = selected_tasks(corpus, args.task, args.tag, args.max_tasks)
    if not tasks:
        raise SystemExit("no tasks matched the selection")
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise SystemExit(f"required API key environment variable is absent: {args.api_key_env}")
    output = (args.output or default_output(args.provider, args.model)).resolve()
    run_id = str(uuid.uuid4())
    writer = JsonlWriter(output, args.append)
    attempts = 0
    passed = 0
    failures: dict[str, int] = {}
    writer.write(
        {
            "schema_version": 1,
            "record_type": "run_start",
            "run_id": run_id,
            "started_at": utc_now(),
            "provider": args.provider,
            "base_url": args.base_url,
            "model": args.model,
            "transports": args.transports,
            "task_ids": [task["id"] for task in tasks],
            "repeat": args.repeat,
            "max_output_tokens": args.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
            "corpus": str(args.corpus.resolve()),
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        }
    )
    try:
        for repetition in range(1, args.repeat + 1):
            for task in tasks:
                for transport in args.transports:
                    record = run_attempt(
                        corpus=corpus,
                        task=task,
                        provider=args.provider,
                        base_url=args.base_url,
                        model=args.model,
                        transport=transport,
                        repetition=repetition,
                        api_key=api_key,
                        timeout_seconds=args.timeout_seconds,
                        max_output_tokens=args.max_output_tokens,
                        reasoning_effort=args.reasoning_effort,
                        run_id=run_id,
                    )
                    writer.write(record)
                    attempts += 1
                    classification = record["classification"]
                    if classification == "pass":
                        passed += 1
                    else:
                        failures[classification] = failures.get(classification, 0) + 1
                    print(
                        f"{task['id']:<24} {transport:<16} {classification}",
                        flush=True,
                    )
    finally:
        writer.write(
            {
                "schema_version": 1,
                "record_type": "run_end",
                "run_id": run_id,
                "finished_at": utc_now(),
                "attempts": attempts,
                "passed": passed,
                "failed": attempts - passed,
                "failures": failures,
            }
        )
        writer.close()
    print(f"saved {attempts} attempts ({passed} passed) to {output}")
    return 1 if args.require_pass and passed != attempts else 0


if __name__ == "__main__":
    raise SystemExit(main())
