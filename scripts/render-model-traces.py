#!/usr/bin/env python3
"""Render redacted Rust implementation traces as TLC-checkable modules."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any


FILTERS = {"current", "global", "other_projects", "all"}


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


class FiniteIds:
    def __init__(self, values: list[str], label: str) -> None:
        self.values = values
        self.label = label
        self.mapping: dict[str, str] = {}

    def get(self, concrete: Any) -> str:
        key = str(concrete)
        if key not in self.mapping:
            if len(self.mapping) == len(self.values):
                raise ValueError(
                    f"{self.label} trace exceeds finite TLC domain {self.values}"
                )
            self.mapping[key] = self.values[len(self.mapping)]
        return self.mapping[key]


def checked_filter(value: Any) -> str:
    if value not in FILTERS:
        raise ValueError(f"unknown scope filter: {value!r}")
    return str(value)


def render_trace(events: list[tuple[Any, ...]]) -> str:
    rows = [
        "    <<" + ", ".join(render_value(value) for value in event) + ">>"
        for event in events
    ]
    return "<<\n" + ",\n".join(rows) + "\n>>"


def render_value(value: Any) -> str:
    if isinstance(value, str):
        return quoted(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, set):
        if not value:
            return "{}"
        return "{" + ", ".join(quoted(item) for item in sorted(value)) + "}"
    raise TypeError(f"unsupported trace value: {value!r}")


def normalize_async(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    prompts = FiniteIds(["p1", "p2", "p3"], "prompt")
    requests = FiniteIds(["r1", "r2"], "permission request")
    normalized: list[tuple[Any, ...]] = []
    known = {
        "enqueue",
        "dispatch_follow_up",
        "commit_start",
        "requeue_start",
        "provider_answer",
        "provider_refusal",
        "provider_failure",
        "provider_tool_batch",
        "complete_tool",
        "ask_permission",
        "allow_permission",
        "deny_permission",
        "claim_steering",
        "commit_steering",
        "requeue_steering",
        "continue_after_tools",
        "settle_turn",
        "request_cancel",
        "repair_cancelled_tool",
        "finish_cancellation",
    }
    for event in events:
        action = event["action"]
        if action not in known:
            raise ValueError(f"unknown AsyncRuntime action: {action!r}")
        prompt = "NoPrompt"
        request = "NoRequest"
        mode = "none"
        count = 0
        if action == "enqueue":
            prompt = prompts.get(event["prompt_id"])
            mode = {
                "steer": "steer",
                "follow_up": "followup",
            }[event["delivery"]]
        if action in {"ask_permission", "allow_permission", "deny_permission"}:
            request = requests.get(event["request_id"])
        if action == "provider_tool_batch":
            count = int(event["count"])
        normalized.append((action, prompt, request, mode, count))
    return normalized


def normalize_memory(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    episodes = FiniteIds(["e1", "e2"], "episode")
    normalized: list[tuple[Any, ...]] = []
    known = {
        "start_turn",
        "settle_turn",
        "record_episode",
        "skip_episode",
        "fail_episode",
        "pause_capture",
        "resume_capture",
        "forget_episode",
        "request_search",
        "deny_search",
        "approve_search",
    }
    for event in events:
        action = event["action"]
        if action not in known:
            raise ValueError(f"unknown MemoryRuntime action: {action!r}")
        episode = "NoEpisode"
        filter_name = "none"
        results: set[str] = set()
        if "episode_id" in event:
            episode = episodes.get(event["episode_id"])
        if "filter" in event:
            filter_name = checked_filter(event["filter"])
        if action == "approve_search":
            results = {episodes.get(value) for value in event["episode_ids"]}
        normalized.append((action, episode, filter_name, results))
    return normalized


def normalize_archive(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    if not events or events[0]["action"] not in {
        "select_project_scope",
        "select_global_scope",
    }:
        raise ValueError("archive trace must begin with an explicit scope selection")
    active_label = events[0].get("scope")
    histories = FiniteIds(["named", "autosave"], "history")
    memories = FiniteIds(["m1", "m2"], "memory")

    def scope(value: Any) -> str:
        if value == active_label:
            return "project"
        if value == "global":
            return "global"
        return "other"

    normalized: list[tuple[Any, ...]] = []
    known = {
        "select_project_scope",
        "select_global_scope",
        "save_history",
        "forget_history",
        "capture_memory",
        "request_search",
        "deny_search",
        "approve_empty_search",
        "approve_history_search",
        "approve_memory_search",
    }
    for event in events:
        action = event["action"]
        if action not in known:
            raise ValueError(f"unknown ArchiveScopeRuntime action: {action!r}")
        scope_name = "NoScope"
        item = "none"
        kind = "none"
        filter_name = "none"
        if action == "select_project_scope":
            scope_name = scope(event["scope"])
        elif action == "select_global_scope":
            scope_name = "global"
        elif action in {
            "save_history",
            "forget_history",
            "approve_history_search",
        }:
            item = histories.get(event["history_id"])
        elif action in {"capture_memory", "approve_memory_search"}:
            item = memories.get(event["memory_id"])
        if action in {"approve_history_search", "approve_memory_search"}:
            scope_name = scope(event["scope"])
        if action == "request_search":
            kind = event["kind"]
            if kind not in {"history", "memory"}:
                raise ValueError(f"unknown archive search kind: {kind!r}")
            filter_name = checked_filter(event["filter"])
        normalized.append((action, scope_name, item, kind, filter_name))
    return normalized


ASYNC_ACTION = r"""
TraceAction(event) ==
    CASE event[1] = "enqueue" ->
            Enqueue(event[2], event[4])
      [] event[1] = "dispatch_follow_up" -> DispatchFollowUp
      [] event[1] = "commit_start" -> CommitStart
      [] event[1] = "requeue_start" -> RequeueStart
      [] event[1] = "provider_answer" -> ProviderAnswer
      [] event[1] = "provider_refusal" -> ProviderRefusal
      [] event[1] = "provider_failure" -> ProviderFailure
      [] event[1] = "provider_tool_batch" -> ProviderToolBatch(event[5])
      [] event[1] = "complete_tool" -> CompleteTool
      [] event[1] = "ask_permission" -> AskPermission(event[3])
      [] event[1] = "allow_permission" -> AllowPermission(event[3])
      [] event[1] = "deny_permission" -> DenyPermission(event[3])
      [] event[1] = "claim_steering" -> ClaimSteering
      [] event[1] = "commit_steering" -> CommitSteering
      [] event[1] = "requeue_steering" -> RequeueSteering
      [] event[1] = "continue_after_tools" -> ContinueAfterTools
      [] event[1] = "settle_turn" -> SettleTurn
      [] event[1] = "request_cancel" -> RequestCancel
      [] event[1] = "repair_cancelled_tool" -> RepairCancelledTool
      [] event[1] = "finish_cancellation" -> FinishCancellation
      [] OTHER -> FALSE
"""

MEMORY_ACTION = r"""
TraceAction(event) ==
    CASE event[1] = "start_turn" -> StartTurn(event[2])
      [] event[1] = "settle_turn" -> SettleTurn
      [] event[1] = "record_episode" ->
            /\ Len(pendingEpisodes) > 0
            /\ Head(pendingEpisodes) = event[2]
            /\ RecordEpisode
      [] event[1] = "skip_episode" ->
            /\ Len(pendingEpisodes) > 0
            /\ Head(pendingEpisodes) = event[2]
            /\ SkipEpisode
      [] event[1] = "fail_episode" ->
            /\ Len(pendingEpisodes) > 0
            /\ Head(pendingEpisodes) = event[2]
            /\ FailEpisode
      [] event[1] = "pause_capture" -> PauseCapture
      [] event[1] = "resume_capture" -> ResumeCapture
      [] event[1] = "forget_episode" -> ForgetEpisode(event[2])
      [] event[1] = "request_search" -> RequestSearch(event[3])
      [] event[1] = "deny_search" -> DenySearch
      [] event[1] = "approve_search" -> ApproveSearch(event[4])
      [] OTHER -> FALSE
"""

ARCHIVE_ACTION = r"""
TraceAction(event) ==
    CASE event[1] = "select_project_scope" -> SelectProjectScope(event[2])
      [] event[1] = "select_global_scope" -> SelectGlobalScope
      [] event[1] = "save_history" -> SaveHistory(event[3])
      [] event[1] = "forget_history" -> ForgetHistory(event[3])
      [] event[1] = "capture_memory" -> CaptureMemory(event[3])
      [] event[1] = "request_search" -> RequestSearch(event[4], event[5])
      [] event[1] = "deny_search" -> DenySearch
      [] event[1] = "approve_empty_search" -> ApproveEmptySearch
      [] event[1] = "approve_history_search" ->
            ApproveHistorySearch(<<event[2], event[3]>>)
      [] event[1] = "approve_memory_search" ->
            ApproveMemorySearch(<<event[2], event[3]>>)
      [] OTHER -> FALSE
"""


def module_text(
    module_name: str,
    base_module: str,
    trace: list[tuple[Any, ...]],
    action_operator: str,
) -> str:
    return f"""----------------------- MODULE {module_name} -----------------------
EXTENDS {base_module}, Naturals, Sequences

Trace == {render_trace(trace)}

VARIABLE traceIndex

traceVars == <<vars, traceIndex>>

{action_operator}

TraceInit ==
    /\\ Init
    /\\ traceIndex = 1

TraceNext ==
    /\\ traceIndex <= Len(Trace)
    /\\ Next
    /\\ TraceAction(Trace[traceIndex])
    /\\ traceIndex' = traceIndex + 1

TraceSpec ==
    /\\ TraceInit
    /\\ [][TraceNext]_traceVars
    /\\ WF_traceVars(TraceNext)

TraceCompletes == <>(traceIndex = Len(Trace) + 1)

=============================================================================
"""


ASYNC_CONFIG = """SPECIFICATION TraceSpec

CONSTANTS
    PromptIds = {"p1", "p2", "p3"}
    RequestIds = {"r1", "r2"}
    NoPrompt = NoPrompt
    NoRequest = NoRequest
    MaxTools = 2
    MaxRounds = 2
    MaxFailures = 1

PROPERTY TraceCompletes

CHECK_DEADLOCK FALSE
"""

MEMORY_CONFIG = """SPECIFICATION TraceSpec

CONSTANTS
    EpisodeIds = {"e1", "e2"}
    ScopeIds = {"project", "global", "other"}
    CurrentScope = "project"
    GlobalScope = "global"
    NoEpisode = NoEpisode
    NoScope = NoScope

PROPERTY TraceCompletes

CHECK_DEADLOCK FALSE
"""

ARCHIVE_CONFIG = """SPECIFICATION TraceSpec

CONSTANTS
    ScopeIds = {"project", "global", "other"}
    HistoryIds = {"autosave", "named"}
    MemoryIds = {"m1", "m2"}
    ProjectScope = "project"
    GlobalScope = "global"
    OtherScope = "other"
    NoScope = NoScope
    NoHistory = NoHistory
    NoMemory = NoMemory
    GlobalHistory = "autosave"
    OtherHistory = "named"
    GlobalMemory = "m1"
    OtherMemory = "m2"

PROPERTY TraceCompletes

CHECK_DEADLOCK FALSE
"""


def write_model(
    directory: Path,
    name: str,
    base: str,
    trace: list[tuple[Any, ...]],
    action: str,
    config: str,
) -> None:
    (directory / f"{name}.tla").write_text(
        module_text(name, base, trace, action), encoding="utf-8"
    )
    (directory / f"{name}.cfg").write_text(config, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    data = json.loads(args.trace_json.read_text(encoding="utf-8"))
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    spec_directory = Path(__file__).resolve().parent.parent / "spec"
    for model in ("AsyncRuntime.tla", "MemoryRuntime.tla", "ArchiveScopeRuntime.tla"):
        shutil.copyfile(spec_directory / model, output / model)

    async_trace = normalize_async(data["async_runtime"])
    memory_trace = normalize_memory(data["memory_runtime"])
    archive_trace = normalize_archive(data["archive_scope_runtime"])

    write_model(
        output,
        "AsyncObservedTrace",
        "AsyncRuntime",
        async_trace,
        ASYNC_ACTION,
        ASYNC_CONFIG,
    )
    write_model(
        output,
        "MemoryObservedTrace",
        "MemoryRuntime",
        memory_trace,
        MEMORY_ACTION,
        MEMORY_CONFIG,
    )
    write_model(
        output,
        "ArchiveObservedTrace",
        "ArchiveScopeRuntime",
        archive_trace,
        ARCHIVE_ACTION,
        ARCHIVE_CONFIG,
    )

    async_invalid = copy.deepcopy(async_trace)
    complete = next(
        index for index, event in enumerate(async_invalid) if event[0] == "complete_tool"
    )
    continuation = next(
        index
        for index, event in enumerate(async_invalid)
        if event[0] == "continue_after_tools"
    )
    async_invalid[complete], async_invalid[continuation] = (
        async_invalid[continuation],
        async_invalid[complete],
    )
    write_model(
        output,
        "AsyncInvalidTrace",
        "AsyncRuntime",
        async_invalid,
        ASYNC_ACTION,
        ASYNC_CONFIG,
    )

    memory_invalid = copy.deepcopy(memory_trace)
    approval = next(
        index
        for index, event in enumerate(memory_invalid)
        if event[0] == "approve_search"
    )
    action, episode, filter_name, _ = memory_invalid[approval]
    memory_invalid[approval] = (action, episode, filter_name, {"e1"})
    write_model(
        output,
        "MemoryInvalidTrace",
        "MemoryRuntime",
        memory_invalid,
        MEMORY_ACTION,
        MEMORY_CONFIG,
    )

    archive_invalid = copy.deepcopy(archive_trace)
    request = next(
        index
        for index, event in enumerate(archive_invalid)
        if event[0] == "request_search" and event[3] == "history"
    )
    action, scope_name, item, kind, _ = archive_invalid[request]
    archive_invalid[request] = (action, scope_name, item, kind, "global")
    write_model(
        output,
        "ArchiveInvalidTrace",
        "ArchiveScopeRuntime",
        archive_invalid,
        ARCHIVE_ACTION,
        ARCHIVE_CONFIG,
    )


if __name__ == "__main__":
    main()
