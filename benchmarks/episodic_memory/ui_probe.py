#!/usr/bin/env python3
"""Exact-binary tmux probe for provider- and SQLite-stall TUI liveness."""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import pathlib
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
INPUT_LAG_BUDGET_MS = 750.0
FLOOD_FRAGMENTS = 30_000
FLOOD_SIGNAL_AFTER = 20_000
FLOOD_CHUNK_FRAGMENTS = 500


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


class ProviderFixture:
    def __init__(self, stall_seconds: float = 1.5) -> None:
        self.stall_seconds = stall_seconds
        self.requests: list[dict[str, Any]] = []
        self._condition = threading.Condition()
        self._flood_prefix_sent = False
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length))
                messages = body.get("messages") or []
                user_text = next(
                    (
                        _message_text(message)
                        for message in reversed(messages)
                        if message.get("role") == "user"
                    ),
                    "",
                )
                with fixture._condition:
                    fixture.requests.append(
                        {
                            "path": self.path,
                            "user_text": user_text,
                            "stream": body.get("stream"),
                            "tool_names": [
                                tool.get("function", {}).get("name")
                                for tool in body.get("tools", [])
                            ],
                            "received_monotonic": time.monotonic(),
                        }
                    )
                    fixture._condition.notify_all()
                if "stall-provider-eval" in user_text:
                    time.sleep(fixture.stall_seconds)
                if "delta-flood-eval" in user_text:
                    first = (
                        'data: {"choices":[{"index":0,"delta":{"role":"assistant",'
                        '"content":"FLOOD-BEGIN "},"finish_reason":null}]}\n\n'
                    ).encode()
                    fragment = (
                        'data: {"choices":[{"index":0,"delta":{"content":"x"},'
                        '"finish_reason":null}]}\n\n'
                    ).encode()
                    tail = (
                        'data: {"choices":[{"index":0,"delta":{"content":" END-FLOOD"},'
                        '"finish_reason":null}]}\n\n'
                    ).encode()
                    finish = (
                        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
                        '"usage":{"prompt_tokens":1,"completion_tokens":1,'
                        '"total_tokens":2}}\n\ndata: [DONE]\n\n'
                    ).encode()
                    content_length = (
                        len(first)
                        + len(fragment) * FLOOD_FRAGMENTS
                        + len(tail)
                        + len(finish)
                    )
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("content-length", str(content_length))
                    self.end_headers()
                    try:
                        self.wfile.write(first)
                        for start in range(0, FLOOD_FRAGMENTS, FLOOD_CHUNK_FRAGMENTS):
                            count = min(
                                FLOOD_CHUNK_FRAGMENTS, FLOOD_FRAGMENTS - start
                            )
                            self.wfile.write(fragment * count)
                            self.wfile.flush()
                            sent = start + count
                            if sent == FLOOD_SIGNAL_AFTER:
                                with fixture._condition:
                                    fixture._flood_prefix_sent = True
                                    fixture._condition.notify_all()
                            if sent >= FLOOD_SIGNAL_AFTER:
                                time.sleep(0.025)
                        self.wfile.write(tail)
                        self.wfile.write(finish)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                answer = f"ACK {user_text}" if user_text else "ACK"
                payloads = [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": answer},
                                "finish_reason": None,
                            }
                        ]
                    },
                    {
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    },
                ]
                wire = "".join(
                    f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    for payload in payloads
                ) + "data: [DONE]\n\n"
                encoded = wire.encode()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(encoded)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self._thread.start()

    def wait_for_requests(self, count: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self.requests) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def wait_for_flood_prefix(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._flood_prefix_sent:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


class TmuxSession:
    def __init__(self, binary: pathlib.Path, home: pathlib.Path, base_url: str) -> None:
        tmux = shutil.which("tmux")
        if tmux is None:
            raise RuntimeError("tmux is required for the exact TUI probe")
        self.tmux = tmux
        self.socket_name = f"generalist-memory-{uuid.uuid4().hex[:12]}"
        self.session_name = "probe"
        command = " ".join(
            [
                "cd",
                shlex.quote(str(ROOT)),
                "&&",
                "env",
                f"GENERALIST_HOME={shlex.quote(str(home))}",
                f"OPENAI_BASE_URL={shlex.quote(base_url)}",
                "OPENAI_API_KEY=evaluation-only",
                shlex.quote(str(binary)),
                "--local",
                "deterministic-memory-eval",
            ]
        )
        self._run(
            "new-session",
            "-d",
            "-x",
            "120",
            "-y",
            "44",
            "-s",
            self.session_name,
            command,
        )

    def _run(
        self, *args: str, check: bool = True, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.tmux, "-L", self.socket_name, *args],
            check=check,
            text=True,
            capture_output=capture_output,
        )

    def alive(self) -> bool:
        result = self._run(
            "has-session", "-t", self.session_name, check=False, capture_output=True
        )
        return result.returncode == 0

    def pane(self) -> str:
        result = self._run(
            "capture-pane",
            "-p",
            "-J",
            "-S",
            "-80",
            "-t",
            f"{self.session_name}:0.0",
            capture_output=True,
        )
        return result.stdout

    def send_text(self, text: str) -> None:
        self._run("send-keys", "-l", "-t", f"{self.session_name}:0.0", text)

    def enter(self) -> None:
        self._run("send-keys", "-t", f"{self.session_name}:0.0", "Enter")

    def choose_next(self) -> None:
        self._run(
            "send-keys", "-t", f"{self.session_name}:0.0", "Down", "Enter"
        )

    def submit(self, text: str) -> None:
        self.send_text(text)
        self.enter()

    def clear_input(self) -> None:
        self._run("send-keys", "-t", f"{self.session_name}:0.0", "C-u")

    def wait_contains(self, needle: str, timeout: float) -> tuple[bool, float, str]:
        started = time.monotonic()
        deadline = started + timeout
        latest = ""
        while time.monotonic() < deadline:
            if not self.alive():
                return False, (time.monotonic() - started) * 1_000.0, latest
            latest = self.pane()
            if needle in latest:
                return True, (time.monotonic() - started) * 1_000.0, latest
            time.sleep(0.02)
        return False, (time.monotonic() - started) * 1_000.0, latest

    def close(self) -> None:
        if self.alive():
            self._run(
                "kill-session",
                "-t",
                self.session_name,
                check=False,
                capture_output=True,
            )
        self._run("kill-server", check=False, capture_output=True)


def _episode_markers(database: pathlib.Path) -> set[str]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute("SELECT events_json FROM episodes").fetchall()
    finally:
        connection.close()
    markers = set()
    for (events_json,) in rows:
        text = events_json
        for marker in [
            "b0-not-retained-5q",
            "stall-provider-eval",
            "queue-provider-7x",
            "queue-lock-9z",
        ]:
            if marker in text:
                markers.add(marker)
    return markers


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_flaky_mcp_fixture(home: pathlib.Path) -> None:
    """Install a stdio MCP server that fails once, then serves one tool."""
    script = home / "flaky_mcp.py"
    marker = home / "flaky_mcp_first_attempt"
    script.write_text(
        """\
import json
import pathlib
import sys

marker = pathlib.Path(sys.argv[1])
if not marker.exists():
    marker.touch()
    raise SystemExit(0)

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        continue
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "flaky", "version": "1"},
        }
    elif method == "tools/list":
        result = {
            "tools": [{
                "name": "ping",
                "description": "Flaky recovery probe",
                "inputSchema": {"type": "object", "properties": {}},
            }]
        }
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "pong"}]}
    else:
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "unknown method"},
        }
        print(json.dumps(response), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    config_directory = home / ".generalist"
    config_directory.mkdir(parents=True, exist_ok=True)
    (config_directory / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "flaky": {
                        "command": sys.executable,
                        "args": [str(script), str(marker)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def run(binary: pathlib.Path) -> dict[str, Any]:
    binary = binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    fixture = ProviderFixture()
    fixture.start()
    temporary = tempfile.TemporaryDirectory(prefix="generalist-memory-ui-")
    home = pathlib.Path(temporary.name)
    session: TmuxSession | None = None
    lock: sqlite3.Connection | None = None
    try:
        _install_flaky_mcp_fixture(home)
        session = TmuxSession(binary, home, fixture.base_url)
        ready, startup_ms, _ = session.wait_contains("Ready", 10)
        if not ready:
            raise RuntimeError("Generalist TUI did not become ready")
        requests_before_mcp = len(fixture.requests)
        session.submit("/mcp status")
        mcp_failure_visible, _, mcp_failure_pane = session.wait_contains(
            "MCP servers: 0/1 connected", 5
        )
        mcp_failed_server_visible = "flaky: failed" in mcp_failure_pane
        session.submit("/mcp retry flaky")
        mcp_retry_visible, _, _ = session.wait_contains(
            "mcp 'flaky': 1 tool(s)", 5
        )
        session.submit("/mcp status")
        mcp_connected_visible, _, _ = session.wait_contains(
            "flaky: connected · 1/1 tool(s) registered", 5
        )
        session.submit("/tools show flaky_ping")
        mcp_tool_visible, _, _ = session.wait_contains("Tool tools.flaky_ping", 5)
        session.submit("/mcp retry flaky")
        mcp_connected_retry_refused, _, _ = session.wait_contains(
            "MCP server 'flaky' is already connected", 5
        )
        session.submit("/mcp retry missing")
        mcp_unknown_retry_refused, _, _ = session.wait_contains(
            "No configured MCP server named 'missing'", 5
        )
        mcp_provider_requests = len(fixture.requests) - requests_before_mcp
        session.submit("/memory status")
        paused_by_default, _, _ = session.wait_contains("paused · 0 episode(s)", 5)
        session.submit("b0-not-retained-5q")
        b0_dispatched = fixture.wait_for_requests(1, 5)
        b0_answer_visible, _, _ = session.wait_contains("ACK b0-not-retained-5q", 5)
        session.submit("/memory search b0-not-retained-5q")
        b0_search_empty, _, _ = session.wait_contains(
            "No current-scope episodes matched “b0-not-retained-5q”", 5
        )
        session.submit("/memory resume")
        resumed, _, _ = session.wait_contains("capture enabled", 5)

        session.submit("stall-provider-eval")
        provider_stalled = fixture.wait_for_requests(2, 5)
        session.send_text("queue-provider-7x")
        visible, provider_input_lag_ms, _ = session.wait_contains(
            "queue-provider-7x", 2
        )
        session.enter()
        queued_during_provider, _, _ = session.wait_contains("queue-provider-7x", 1)
        second_dispatched = fixture.wait_for_requests(3, 8)
        second_answer_visible, _, _ = session.wait_contains("ACK queue-provider-7x", 5)

        database = home / ".generalist" / "memory" / "scoped-episodes.sqlite3"
        lock = sqlite3.connect(database, timeout=0, isolation_level=None)
        lock.execute("BEGIN IMMEDIATE")
        requests_before_lock = len(fixture.requests)
        lock_started = time.monotonic()
        session.submit("/memory pause")
        time.sleep(0.1)
        session.send_text("queue-lock-9z")
        lock_visible, lock_input_lag_ms, _ = session.wait_contains("queue-lock-9z", 2)
        session.enter()
        queued_during_lock, _, _ = session.wait_contains("queue-lock-9z", 1)
        requests_while_command_pending = len(fixture.requests) - requests_before_lock
        locked_error_visible, _, locked_pane = session.wait_contains("database is locked", 4)
        lock_elapsed_ms = (time.monotonic() - lock_started) * 1_000.0
        lock.execute("ROLLBACK")
        lock.close()
        lock = None
        third_dispatched = fixture.wait_for_requests(4, 6)
        third_answer_visible, _, _ = session.wait_contains("ACK queue-lock-9z", 5)
        session.submit("/memory search queue-lock-9z")
        search_visible, _, search_pane = session.wait_contains(
            "matched “queue-lock-9z”", 5
        )
        autosaves = list(
            (home / ".generalist" / "history" / "scopes").glob("*/autosave.json")
        )
        if len(autosaves) != 1:
            raise RuntimeError(f"expected one scoped autosave, found {len(autosaves)}")
        autosave_hash_before = _sha256(autosaves[0])
        requests_before_history = len(fixture.requests)
        session.submit("/history search b0-not-retained-5q")
        history_search_visible, _, _ = session.wait_contains(
            "1 current-scope saved session(s) matched ‘b0-not-retained-5q’", 5
        )
        session.submit("/history show autosave")
        history_show_visible, _, history_pane = session.wait_contains(
            "Saved session autosave", 5
        )
        session.submit("/save lifecycle checkpoint")
        direct_save_visible, _, _ = session.wait_contains(
            "Saved session 'lifecycle checkpoint'", 5
        )
        named_paths = list(
            (home / ".generalist" / "history" / "scopes").glob(
                "*/lifecycle checkpoint.json"
            )
        )
        named_file_created = len(named_paths) == 1
        named_hash_before_replace = None
        if named_file_created:
            replacement_fixture = json.loads(named_paths[0].read_text())
            replacement_fixture["goal"] = "preexisting-checkpoint-fixture"
            named_paths[0].write_text(
                json.dumps(replacement_fixture, indent=2), encoding="utf-8"
            )
            named_hash_before_replace = _sha256(named_paths[0])
        session.submit("/save lifecycle checkpoint")
        replace_cancel_prompt_visible, _, _ = session.wait_contains(
            "Replace 'lifecycle checkpoint'", 5
        )
        session.enter()
        replace_cancel_visible, _, _ = session.wait_contains(
            "Kept existing saved session 'lifecycle checkpoint'.", 5
        )
        named_unchanged_after_replace_cancel = (
            named_file_created
            and _sha256(named_paths[0]) == named_hash_before_replace
        )
        session.submit("/save lifecycle checkpoint")
        replace_prompt_visible, _, _ = session.wait_contains(
            "Replace 'lifecycle checkpoint'", 5
        )
        session.choose_next()
        replace_visible, _, _ = session.wait_contains(
            "Replaced saved session 'lifecycle checkpoint'", 5
        )
        named_replaced_after_confirmation = (
            named_file_created
            and _sha256(named_paths[0]) != named_hash_before_replace
        )
        session.submit("/save autosave")
        manual_autosave_refused, _, _ = session.wait_contains(
            "The live autosave name is reserved", 5
        )
        session.submit("/load lifecycle checkpoint")
        direct_load_visible, _, _ = session.wait_contains(
            "Loaded saved session 'lifecycle checkpoint'", 5
        )
        session.submit("/history forget lifecycle checkpoint")
        cancel_prompt_visible, _, _ = session.wait_contains(
            "Delete 'lifecycle checkpoint'", 5
        )
        session.enter()
        cancel_visible, _, _ = session.wait_contains(
            "Kept saved session 'lifecycle checkpoint'.", 5
        )
        retained_after_cancel = named_file_created and named_paths[0].is_file()
        session.submit("/history forget lifecycle checkpoint")
        delete_prompt_visible, _, _ = session.wait_contains(
            "Delete 'lifecycle checkpoint'", 5
        )
        session.choose_next()
        delete_visible, _, _ = session.wait_contains(
            "Deleted current-scope saved session 'lifecycle checkpoint'.", 5
        )
        absent_after_delete = named_file_created and not named_paths[0].exists()
        session.submit("/history show lifecycle checkpoint")
        deleted_show_missing, _, _ = session.wait_contains(
            "No current-scope saved session named 'lifecycle checkpoint'", 5
        )
        session.submit("/history forget autosave")
        autosave_forget_refused, _, _ = session.wait_contains(
            "The active autosave cannot be forgotten", 5
        )
        history_provider_requests = len(fixture.requests) - requests_before_history
        autosave_hash_after = _sha256(autosaves[0])
        history_content_unchanged = autosave_hash_before == autosave_hash_after
        requests_before_usage = len(fixture.requests)
        usage_autosave_hash_before = _sha256(autosaves[0])
        session.submit("/usage")
        usage_totals_visible, _, usage_pane = session.wait_contains(
            "Total: 4 attempts; 4 usage reports; 0 unreported attempts; 4 input; 4 output",
            5,
        )
        usage_cache_coverage_visible = (
            "cache read unavailable (0/4 reports)" in usage_pane
            and "cache creation unavailable (0/4 reports)" in usage_pane
        )
        usage_boundary_visible = "not a cost estimate" in usage_pane
        session.submit("/usage reset")
        usage_reset_visible, _, _ = session.wait_contains(
            "Provider usage counters reset", 5
        )
        session.submit("/usage show")
        usage_empty_visible, _, final_pane = session.wait_contains(
            "no API attempts recorded in this process", 5
        )
        usage_provider_requests = len(fixture.requests) - requests_before_usage
        usage_autosave_unchanged = (
            _sha256(autosaves[0]) == usage_autosave_hash_before
        )
        session.submit("/memory pause")
        flood_capture_paused, _, _ = session.wait_contains(
            "Episodic capture paused for the current scope", 5
        )
        requests_before_flood = len(fixture.requests)
        session.submit("delta-flood-eval")
        flood_dispatched = fixture.wait_for_requests(5, 5)
        flood_prefix_sent = fixture.wait_for_flood_prefix(5)
        session.send_text("flood-input-visible")
        flood_input_visible, flood_input_lag_ms, _ = session.wait_contains(
            "flood-input-visible", 2
        )
        session.clear_input()
        flood_tail_visible, _, flood_pane = session.wait_contains("END-FLOOD", 8)
        flood_settled, _, settled_flood_pane = session.wait_contains(
            "Enter send · Ctrl+J newline", 5
        )
        flood_provider_requests = len(fixture.requests) - requests_before_flood
        final_pane = (
            settled_flood_pane
            or flood_pane
            or final_pane
            or usage_pane
            or history_pane
            or search_pane
            or locked_pane
        )
        session.submit("/exit")
        exit_deadline = time.monotonic() + 5
        while session.alive() and time.monotonic() < exit_deadline:
            time.sleep(0.05)
        normal_exit = not session.alive()
        markers = _episode_markers(database)
        requests = list(fixture.requests)
        exact_python_catalog = all(
            request["path"] == "/v1/chat/completions"
            and request["stream"] is True
            and request["tool_names"] == ["python"]
            for request in requests
        )
        passed = all(
            [
                ready,
                mcp_failure_visible,
                mcp_failed_server_visible,
                mcp_retry_visible,
                mcp_connected_visible,
                mcp_tool_visible,
                mcp_connected_retry_refused,
                mcp_unknown_retry_refused,
                mcp_provider_requests == 0,
                paused_by_default,
                b0_dispatched,
                b0_answer_visible,
                b0_search_empty,
                resumed,
                provider_stalled,
                visible,
                queued_during_provider,
                second_dispatched,
                second_answer_visible,
                lock_visible,
                queued_during_lock,
                locked_error_visible,
                requests_while_command_pending == 0,
                third_dispatched,
                third_answer_visible,
                search_visible,
                history_search_visible,
                history_show_visible,
                direct_save_visible,
                named_file_created,
                replace_cancel_prompt_visible,
                replace_cancel_visible,
                named_unchanged_after_replace_cancel,
                replace_prompt_visible,
                replace_visible,
                named_replaced_after_confirmation,
                manual_autosave_refused,
                direct_load_visible,
                cancel_prompt_visible,
                cancel_visible,
                retained_after_cancel,
                delete_prompt_visible,
                delete_visible,
                absent_after_delete,
                deleted_show_missing,
                autosave_forget_refused,
                history_provider_requests == 0,
                history_content_unchanged,
                usage_totals_visible,
                usage_cache_coverage_visible,
                usage_boundary_visible,
                usage_reset_visible,
                usage_empty_visible,
                usage_provider_requests == 0,
                usage_autosave_unchanged,
                flood_capture_paused,
                flood_dispatched,
                flood_prefix_sent,
                flood_input_visible,
                flood_tail_visible,
                flood_settled,
                flood_provider_requests == 1,
                normal_exit,
                provider_input_lag_ms <= INPUT_LAG_BUDGET_MS,
                lock_input_lag_ms <= INPUT_LAG_BUDGET_MS,
                flood_input_lag_ms <= INPUT_LAG_BUDGET_MS,
                markers
                == {"stall-provider-eval", "queue-provider-7x", "queue-lock-9z"},
                "b0-not-retained-5q" not in markers,
                len(requests) == 5,
                exact_python_catalog,
            ]
        )
        return {
            "schema_version": 1,
            "startup_ms": startup_ms,
            "mcp_recovery": {
                "startup_failure_visible": mcp_failure_visible,
                "failed_server_visible": mcp_failed_server_visible,
                "retry_visible": mcp_retry_visible,
                "connected_status_visible": mcp_connected_visible,
                "recovered_tool_visible": mcp_tool_visible,
                "connected_retry_refused": mcp_connected_retry_refused,
                "unknown_retry_refused": mcp_unknown_retry_refused,
                "provider_requests": mcp_provider_requests,
            },
            "b0_paused": {
                "paused_by_default": paused_by_default,
                "turn_dispatched": b0_dispatched,
                "answer_visible": b0_answer_visible,
                "search_empty": b0_search_empty,
                "retained_after_exit": "b0-not-retained-5q" in markers,
            },
            "provider_stall": {
                "input_visible": visible,
                "input_lag_ms": provider_input_lag_ms,
                "queued": queued_during_provider,
                "follow_up_dispatched": second_dispatched,
            },
            "sqlite_write_lock": {
                "input_visible": lock_visible,
                "input_lag_ms": lock_input_lag_ms,
                "queued": queued_during_lock,
                "locked_error_visible": locked_error_visible,
                "elapsed_ms": lock_elapsed_ms,
                "provider_requests_while_command_pending": requests_while_command_pending,
            },
            "post_lock_search_visible": search_visible,
            "history_inspection": {
                "search_visible": history_search_visible,
                "show_visible": history_show_visible,
                "provider_requests": history_provider_requests,
                "autosave_content_unchanged": history_content_unchanged,
            },
            "saved_session_lifecycle": {
                "direct_save_visible": direct_save_visible,
                "named_file_created": named_file_created,
                "replace_cancel_prompt_visible": replace_cancel_prompt_visible,
                "replace_cancel_visible": replace_cancel_visible,
                "named_unchanged_after_replace_cancel": (
                    named_unchanged_after_replace_cancel
                ),
                "replace_prompt_visible": replace_prompt_visible,
                "replace_visible": replace_visible,
                "named_replaced_after_confirmation": (
                    named_replaced_after_confirmation
                ),
                "manual_autosave_refused": manual_autosave_refused,
                "direct_load_visible": direct_load_visible,
                "cancel_prompt_visible": cancel_prompt_visible,
                "cancel_visible": cancel_visible,
                "retained_after_cancel": retained_after_cancel,
                "delete_prompt_visible": delete_prompt_visible,
                "delete_visible": delete_visible,
                "absent_after_delete": absent_after_delete,
                "deleted_show_missing": deleted_show_missing,
                "autosave_forget_refused": autosave_forget_refused,
                "provider_requests": history_provider_requests,
            },
            "provider_usage": {
                "totals_visible": usage_totals_visible,
                "cache_coverage_visible": usage_cache_coverage_visible,
                "epistemic_boundary_visible": usage_boundary_visible,
                "reset_visible": usage_reset_visible,
                "empty_after_reset_visible": usage_empty_visible,
                "provider_requests": usage_provider_requests,
                "autosave_content_unchanged": usage_autosave_unchanged,
            },
            "event_flood": {
                "fragments": FLOOD_FRAGMENTS,
                "prefix_fragments_before_input": FLOOD_SIGNAL_AFTER,
                "capture_paused": flood_capture_paused,
                "turn_dispatched": flood_dispatched,
                "prefix_sent": flood_prefix_sent,
                "input_visible": flood_input_visible,
                "input_lag_ms": flood_input_lag_ms,
                "committed_tail_visible": flood_tail_visible,
                "turn_settled": flood_settled,
                "provider_requests": flood_provider_requests,
            },
            "captured_markers": sorted(markers),
            "provider_request_count": len(requests),
            "provider_requests_advertised_only_python": exact_python_catalog,
            "normal_exit": normal_exit,
            "final_pane_tail": "\n".join(final_pane.splitlines()[-12:]),
            "input_lag_budget_ms": INPUT_LAG_BUDGET_MS,
            "passed": passed,
        }
    finally:
        if lock is not None:
            with contextlib.suppress(sqlite3.Error):
                lock.execute("ROLLBACK")
            lock.close()
        if session is not None:
            session.close()
        fixture.close()
        temporary.cleanup()
