#!/usr/bin/env python3
"""Render redacted Rust implementation traces as TLC-checkable modules."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


FILTERS = {"current", "global", "other_projects", "all"}


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


class FiniteIds:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.mapping: dict[str, str] = {}

    def get(self, concrete: Any) -> str:
        key = str(concrete)
        if key not in self.mapping:
            self.mapping[key] = f"{self.prefix}{len(self.mapping) + 1}"
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
    if isinstance(value, list):
        return "<<" + ", ".join(render_value(item) for item in value) + ">>"
    if isinstance(value, set):
        if not value:
            return "{}"
        return "{" + ", ".join(quoted(item) for item in sorted(value)) + "}"
    raise TypeError(f"unsupported trace value: {value!r}")


def normalize_async(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    prompts = FiniteIds("p")
    requests = FiniteIds("r")
    normalized: list[tuple[Any, ...]] = []
    known = {
        "enqueue",
        "delete_queued",
        "reclassify_queued",
        "move_queued_earlier",
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
        if action in {"enqueue", "dispatch_follow_up"}:
            prompt = prompts.get(event["prompt_id"])
        if action in {"enqueue", "reclassify_queued"}:
            mode = {
                "steer": "steer",
                "follow_up": "followup",
            }[event["delivery"]]
        if action in {"ask_permission", "allow_permission", "deny_permission"}:
            request = requests.get(event["request_id"])
        if action == "provider_tool_batch":
            count = int(event["count"])
        if action in {"delete_queued", "reclassify_queued", "move_queued_earlier"}:
            count = int(event["index"])
        claimed = [prompts.get(value) for value in event.get("prompt_ids", [])]
        normalized.append((action, prompt, request, mode, count, claimed))
    return normalized


def normalize_memory(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    episodes = FiniteIds("e")
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
    histories = FiniteIds("h")
    memories = FiniteIds("m")

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
      [] event[1] = "delete_queued" -> DeleteQueued(event[5])
      [] event[1] = "reclassify_queued" -> ReclassifyQueued(event[5], event[4])
      [] event[1] = "move_queued_earlier" -> MoveQueuedEarlier(event[5])
      [] event[1] = "dispatch_follow_up" ->
            /\ DispatchFollowUp
            /\ activeTurn' = event[2]
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
      [] event[1] = "claim_steering" ->
            /\ ClaimSteering
            /\ claimedSteers' = event[6]
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


SPEC_DIRECTORY = Path(__file__).resolve().parent.parent / "spec"


def trace_config(base: str, overrides: dict[str, str]) -> str:
    """Reuse the exhaustive model's invariants; specialize only finite constants.

    TraceCompletes checks a finite observed run, not the base Spec's fairness
    assumptions or its unbounded temporal properties.
    """
    config = (SPEC_DIRECTORY / f"{base}.cfg").read_text(encoding="utf-8")
    config = config.replace("SPECIFICATION Spec", "SPECIFICATION TraceSpec")
    config = re.sub(r"^PROPERTY .*\n?", "", config, flags=re.MULTILINE)
    for name, value in overrides.items():
        config, count = re.subn(
            rf"^([ \t]+{name} = ).+$",
            lambda match: match[1] + value,
            config,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"expected one {base} constant {name}, found {count}")
    return config + "\nPROPERTY TraceCompletes\nCHECK_DEADLOCK FALSE\n"


def domain(values: set[str], fallback: str) -> str:
    return render_value(values or {fallback})


def async_config(trace: list[tuple[Any, ...]], max_rounds: int) -> str:
    return trace_config("AsyncRuntime", {
        "PromptIds": domain({e[1] for e in trace if e[0] == "enqueue"}, "p1"),
        "RequestIds": domain({e[2] for e in trace if e[2] != "NoRequest"}, "r1"),
        "MaxTools": str(max([1] + [e[4] for e in trace if e[0] == "provider_tool_batch"])),
        "MaxRounds": str(max_rounds),
    })


def memory_config(trace: list[tuple[Any, ...]]) -> str:
    episodes = {e[1] for e in trace if e[1] != "NoEpisode"}
    episodes.update(item for e in trace for item in e[3])
    return trace_config("MemoryRuntime", {
        "EpisodeIds": domain(episodes, "e1"),
        "ScopeIds": render_value({"project", "global", "other"}),
        "CurrentScope": quoted("project"),
        "GlobalScope": quoted("global"),
    })


def archive_config(trace: list[tuple[Any, ...]]) -> str:
    histories = {e[2] for e in trace if e[0] in {
        "save_history", "forget_history", "approve_history_search"}}
    memories = {e[2] for e in trace if e[0] in {
        "capture_memory", "approve_memory_search"}}
    return trace_config("ArchiveScopeRuntime", {
        "ScopeIds": render_value({"project", "global", "other"}),
        "HistoryIds": domain(histories, "h1"),
        "MemoryIds": domain(memories, "m1"),
        "ProjectScope": quoted("project"), "GlobalScope": quoted("global"),
        "OtherScope": quoted("other"),
        "GlobalHistory": quoted(min(histories or {"h1"})),
        "OtherHistory": quoted(min(histories or {"h1"})),
        "GlobalMemory": quoted(min(memories or {"m1"})),
        "OtherMemory": quoted(min(memories or {"m1"})),
    })


def write_model(directory: Path, name: str, base: str,
                trace: list[tuple[Any, ...]], action: str, config: str) -> None:
    if not trace:
        raise ValueError(f"refusing vacuous trace: {name}")
    (directory / f"{name}.tla").write_text(
        module_text(name, base, trace, action), encoding="utf-8")
    (directory / f"{name}.cfg").write_text(config, encoding="utf-8")


def mutations(base: str, trace: list[tuple[Any, ...]]):
    """Each edit represents a concrete protocol violation, never a parse error."""
    for index, event in enumerate(trace):
        action = event[0]
        if base == "AsyncRuntime":
            if action == "continue_after_tools":
                previous = next((i for i in range(index - 1, -1, -1)
                                 if trace[i][0] == "complete_tool"), None)
                if previous is None:
                    continue
                broken = trace.copy()
                broken[previous], broken[index] = broken[index], broken[previous]
                yield "EarlyContinuation", broken
                break
    if base == "AsyncRuntime":
        for action, label in [("complete_tool", "MissingResult"),
                              ("repair_cancelled_tool", "MissingCancellationRepair"),
                              ("deny_permission", "LostDenial"),
                              ("delete_queued", "LostQueueDeletion"),
                              ("commit_steering", "LostSteeringCommit")]:
            index = next((i for i, e in enumerate(trace) if e[0] == action), None)
            if index is not None:
                yield label, trace[:index] + trace[index + 1:]
        approval = next((i for i, e in enumerate(trace)
                         if e[0] in {"allow_permission", "deny_permission"}), None)
        if approval is not None:
            broken = trace.copy()
            event = broken[approval]
            # No request ID outside RequestIds: the failure must be correlation.
            broken.insert(approval + 1, event)
            yield "ReplayedPermission", broken
    elif base == "MemoryRuntime":
        forgotten = next((e[1] for e in trace if e[0] == "forget_episode"), None)
        approval = next((i for i, e in enumerate(trace) if e[0] == "approve_search"), None)
        if forgotten is not None and approval is not None:
            broken = trace.copy()
            action, episode, filter_name, _ = broken[approval]
            broken[approval] = (action, episode, filter_name, {forgotten})
            yield "ForgottenDisclosure", broken
    elif base == "ArchiveScopeRuntime":
        request = next((i for i, e in enumerate(trace)
                        if e[0] == "request_search" and e[3] == "history"), None)
        if request is not None:
            broken = trace.copy()
            action, scope, item, kind, _ = broken[request]
            broken[request] = (action, scope, item, kind, "global")
            yield "WrongScope", broken


def write_model_mutants(output: Path) -> None:
    # These faults must violate the named invariant. A syntax/typechecker
    # failure, timeout, or unrelated temporal failure never counts as success.
    faults = [
        ("AsyncRuntime", "PrematureIdle", "SingleTurnOwnership",
         "phase' = IF toolResults' = toolUses THEN \"boundary\" ELSE \"tools\"",
         "phase' = \"idle\""),
        ("MemoryRuntime", "UndrainedEpisode", "EpisodeLifecycleIsDisjoint",
         "pendingEpisodes' = Tail(pendingEpisodes)",
         "UNCHANGED pendingEpisodes"),
        ("ArchiveScopeRuntime", "UnscopedHistory", "DisclosureMatchesRequestedScope",
         "/\\ MatchesFilter(item, pendingFilter)", "/\\ TRUE"),
    ]
    manifest = []
    for base, label, invariant, original, replacement in faults:
        source = (SPEC_DIRECTORY / f"{base}.tla").read_text(encoding="utf-8")
        # Each first occurrence belongs to CompleteTool, RecordEpisode, or
        # ApproveHistorySearch respectively. Refuse silent fixture drift.
        action = {"AsyncRuntime": "CompleteTool", "MemoryRuntime": "RecordEpisode",
                  "ArchiveScopeRuntime": "ApproveHistorySearch"}[base]
        start = re.search(rf"^{action}(?:\([^)]*\))? ==$", source, re.MULTILINE)
        if start is None:
            raise ValueError(f"missing mutation action: {base}.{action}")
        end = re.search(r"^[A-Z][A-Za-z]+(?:\([^)]*\))? ==$", source[start.end():], re.MULTILINE)
        stop = start.end() + end.start() if end else len(source)
        body = source[start.start():stop]
        if body.count(original) != 1:
            raise ValueError(f"mutation target drifted: {base}.{action}")
        source = source[:start.start()] + body.replace(original, replacement) + source[stop:]
        module = base + label
        source = source.replace(f"MODULE {base} ", f"MODULE {module} ", 1)
        (output / f"{module}.tla").write_text(source, encoding="utf-8")
        shutil.copyfile(SPEC_DIRECTORY / f"{base}.cfg", output / f"{module}.cfg")
        manifest.append(f"{module}\t{invariant}")
    (output / "model-mutants.tsv").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    scenarios = json.loads(args.trace_json.read_text(encoding="utf-8"))
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    models = [
        ("async_runtime", "AsyncRuntime", normalize_async, ASYNC_ACTION),
        ("memory_runtime", "MemoryRuntime", normalize_memory, MEMORY_ACTION),
        ("archive_scope_runtime", "ArchiveScopeRuntime", normalize_archive, ARCHIVE_ACTION),
    ]
    for _, base, _, _ in models:
        shutil.copyfile(SPEC_DIRECTORY / f"{base}.tla", output / f"{base}.tla")
    write_model_mutants(output)
    names: set[str] = set()
    covered: set[str] = set()
    observed, invalid = [], []
    for scenario in scenarios:
        name = scenario["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name) or name in names:
            raise ValueError(f"invalid or duplicate scenario name: {name!r}")
        names.add(name)
        for field, base, normalize, action in models:
            if not scenario[field]:
                continue
            covered.add(base)
            trace = normalize(scenario[field])
            config = (async_config(trace, scenario["max_rounds"])
                      if field == "async_runtime" else memory_config(trace)
                      if field == "memory_runtime" else archive_config(trace))
            module = base + name
            write_model(output, module, base, trace, action, config)
            observed.append(module)
            for label, broken in mutations(base, trace):
                mutant = module + label
                write_model(output, mutant, base, broken, action, config)
                invalid.append(mutant)
    if covered != {base for _, base, _, _ in models}:
        raise ValueError("conformance must exercise all three runtime models")
    if not observed or not invalid:
        raise ValueError("conformance requires observed traces and rejecting mutations")
    (output / "observed.txt").write_text("\n".join(observed) + "\n", encoding="utf-8")
    (output / "invalid.txt").write_text("\n".join(invalid) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
