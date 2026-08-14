//! Code mode: scripts that call tools as code APIs.
//!
//! Instead of the model emitting one JSON tool call per step, it gets one
//! model-facing `python` tool. The script imports a generated `tools` module
//! and calls any registered tool as a function (Cloudflare's "Code Mode" /
//! Anthropic's code-execution-with-MCP pattern; the AutoHarness pattern in
//! the "Code as Agent Harness" survey). The payoff:
//!
//! - one script replaces N model round-trips (loop/branch/retry in code)
//! - tool results are processed *inside* the script and never enter the
//!   model's context unless printed — state offloading for free
//!
//! Mechanics: the agent writes `tools.py` + `main.py` to a scratch dir, runs
//! `main.py` through a small wrapper that preloads `tools`, and serves tool
//! calls over a Unix socket. Every
//! bridged call goes through the same `ToolRegistry` permission gate as a
//! direct call, so interactive approval still applies.

use crate::agent::AgentEvent;
use crate::tool::{ToolCallOutcome, ToolRegistry};
use crate::types::{ContentBlock, ToolDef};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::Command;
use tokio::time::Duration;

/// The name the code-mode tool is advertised under.
pub(crate) const PY_TOOL_NAME: &str = "python";

pub(crate) use crate::subprocess::{DEFAULT_TIMEOUT_SECS, MAX_TIMEOUT_SECS};

const RUNNER_MODULE: &str = r#""""Generalist code-mode bootstrap."""
import runpy as _runpy
import sys as _sys
import tools as _tools

_runpy.run_path(_sys.argv[1], init_globals={"tools": _tools}, run_name="__main__")
"#;

fn is_python_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
        && !matches!(
            value,
            "False"
                | "None"
                | "True"
                | "and"
                | "as"
                | "assert"
                | "async"
                | "await"
                | "break"
                | "class"
                | "continue"
                | "def"
                | "del"
                | "elif"
                | "else"
                | "except"
                | "finally"
                | "for"
                | "from"
                | "global"
                | "if"
                | "import"
                | "in"
                | "is"
                | "lambda"
                | "nonlocal"
                | "not"
                | "or"
                | "pass"
                | "raise"
                | "return"
                | "try"
                | "while"
                | "with"
                | "yield"
        )
}

fn schema_type_hint(schema: &Value) -> String {
    match schema.get("type") {
        Some(Value::String(kind)) => match kind.as_str() {
            "string" => "str".to_string(),
            "integer" => "int".to_string(),
            "number" => "float".to_string(),
            "boolean" => "bool".to_string(),
            "array" => format!(
                "list[{}]",
                schema
                    .get("items")
                    .map(schema_type_hint)
                    .unwrap_or_else(|| "object".to_string())
            ),
            "object" => "dict[str, object]".to_string(),
            "null" => "None".to_string(),
            _ => "object".to_string(),
        },
        Some(Value::Array(kinds)) => kinds
            .iter()
            .filter_map(Value::as_str)
            .map(|kind| schema_type_hint(&json!({"type": kind})))
            .collect::<Vec<_>>()
            .join(" | "),
        _ => "object".to_string(),
    }
}

fn python_call_signature(tool: &ToolDef) -> String {
    let Some(properties) = tool
        .input_schema
        .get("properties")
        .and_then(Value::as_object)
    else {
        return format!("tools.{}(**kwargs) -> str", tool.name);
    };
    let required = tool
        .input_schema
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let is_required = |name: &str| required.iter().any(|value| value.as_str() == Some(name));
    if properties.keys().any(|name| !is_python_identifier(name))
        || required
            .iter()
            .filter_map(Value::as_str)
            .any(|name| !properties.contains_key(name))
    {
        return format!("tools.{}(**kwargs) -> str", tool.name);
    }

    let mut parameters = Vec::new();
    for required_pass in [true, false] {
        for (name, schema) in properties {
            if is_required(name) != required_pass {
                continue;
            }
            let hint = schema_type_hint(schema);
            parameters.push(if required_pass {
                format!("{name}: {hint}")
            } else {
                format!("{name}: {hint} | None = None")
            });
        }
    }
    if tool
        .input_schema
        .get("additionalProperties")
        .and_then(Value::as_bool)
        != Some(false)
    {
        parameters.push("**extra".to_string());
    }
    if parameters.is_empty() {
        format!("tools.{}() -> str", tool.name)
    } else {
        format!("tools.{}(*, {}) -> str", tool.name, parameters.join(", "))
    }
}

/// The sole tool definition advertised to the model when code mode is on.
///
/// Ordinary registered tools include compact call signatures and descriptions here so
/// the model can use them in its first script without a discovery round-trip.
/// `code_only_tools` are listed by name only — their full schemas live in the
/// generated module's docstrings (progressive disclosure), so heavy (e.g.
/// MCP) schemas never enter the model's context unrequested.
pub(crate) fn python_tool_def(available_tools: &[ToolDef], code_only_tools: &[ToolDef]) -> ToolDef {
    let tool_docs = if available_tools.is_empty() {
        "No ordinary bridge tools are registered; Python's standard library is still available."
            .to_string()
    } else {
        available_tools
            .iter()
            .map(|tool| {
                format!(
                    "- {signature}\n  {description}",
                    signature = python_call_signature(tool),
                    description = tool.description,
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let code_only_note = if code_only_tools.is_empty() {
        String::new()
    } else {
        format!(
            "\nAdditional progressive-disclosure tools are also available only inside scripts. \
             Their schemas are intentionally omitted here; inspect a tool at runtime with \
             print(tools.<name>.__doc__) when needed: {}.",
            code_only_tools
                .iter()
                .map(|t| format!("tools.{}", t.name))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    ToolDef {
        name: PY_TOOL_NAME.to_string(),
        description: format!(
            "Execute a Python 3 script and return its stdout/stderr. This is the only \
             model-facing capability tool while code mode is enabled: perform all capability \
             work inside the script via the pre-generated `tools` module. Host-owned control \
             tools such as `update_goal` may also be advertised separately and must be called \
             natively. A `tools.<name>` expression belongs inside the `code` string; never emit \
             `<name>` or `tools.<name>` as a native tool call. The generated module is already \
             bound to `tools` (`import tools` remains valid but is optional). Call bridge \
             functions with keyword arguments; they return str and raise RuntimeError on \
             failure. Every function's `__doc__` retains its full JSON Schema for runtime \
             inspection. Complete the largest coherent work phase in one script \
             instead of returning after one bridged call: loop, branch, retry, validate, and \
             combine results in code. Tool results stay inside the script unless printed. Print \
             only conclusions, compact evidence, and paths; write large intermediate results to \
             files. Output is truncated to the last 50,000 characters (full output is saved to a \
             temp file whose path is included). Default timeout 120s (override with \
             timeout_seconds, max 600).\n\nAvailable bridge tools:\n{}{}",
            tool_docs, code_only_note
        ),
        input_schema: json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "A self-contained Python 3 script that completes as much of the requested tool work as possible"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120, max: 600)"
                }
            },
            "required": ["code"],
            "additionalProperties": false
        }),
    }
}

/// Generate the `tools.py` bridge module for the given tool definitions.
///
/// Docstrings are embedded via JSON encoding, which is also a valid Python
/// string literal, so arbitrary descriptions can't break the module.
pub(crate) fn generate_tools_module(defs: &[ToolDef]) -> String {
    let mut module = String::from(
        r#""""Auto-generated bridge to the agent's tools.

Generalist preloads this module as `tools`; `import tools` also works.
Call tools.<name>(**kwargs) -> str. Raises RuntimeError on failure."""
import json as _json
import os as _os
import socket as _socket

_sock = None
_rfile = None


def _call(_tool, **kwargs):
    global _sock, _rfile
    if _sock is None:
        _sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        _sock.connect(_os.environ["GENERALIST_TOOL_SOCKET"])
        _rfile = _sock.makefile("r")
    _sock.sendall((_json.dumps({"tool": _tool, "input": kwargs}) + "\n").encode())
    line = _rfile.readline()
    if not line:
        raise RuntimeError("tool bridge closed unexpectedly")
    resp = _json.loads(line)
    if resp.get("is_error"):
        raise RuntimeError(f"{_tool}: {resp.get('content', 'tool failed')}")
    return resp.get("content", "")

"#,
    );
    for def in defs {
        let doc = format!(
            "{}\n\nCall: {}\nInput schema: {}",
            def.description,
            python_call_signature(def),
            serde_json::to_string(&def.input_schema).unwrap_or_default()
        );
        module.push_str(&format!(
            "\ndef {name}(**kwargs):\n    return _call(\"{name}\", **kwargs)\n\n\n{name}.__doc__ = {doc}\n",
            name = def.name,
            doc = serde_json::to_string(&doc).unwrap_or_else(|_| "\"\"".to_string()),
        ));
    }
    module
}

pub(crate) struct ScriptResult {
    pub content: String,
    pub failed: bool,
    /// At least one bridged tool call was denied by the permission policy.
    ///
    /// A script may catch the generated Python exception and still exit zero,
    /// so this signal must travel separately from process success.
    pub denied: bool,
}

/// Remove the bridge socket and its private directory when dropped.
struct RemoveOnDrop {
    socket: PathBuf,
    directory: PathBuf,
}

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

/// Run `code` with the tool bridge attached.
///
/// Returns the script's combined output; bridged tool calls are executed via
/// `registry` (permission-checked per call) and surfaced through `on_event`.
pub(crate) async fn run_script(
    code: &str,
    timeout_secs: u64,
    registry: &mut ToolRegistry,
    on_event: &mut dyn FnMut(AgentEvent),
) -> ScriptResult {
    match run_script_inner(code, timeout_secs, registry, on_event).await {
        Ok(result) => result,
        Err(message) => ScriptResult {
            content: message,
            failed: true,
            denied: false,
        },
    }
}

async fn run_script_inner(
    code: &str,
    timeout_secs: u64,
    registry: &mut ToolRegistry,
    on_event: &mut dyn FnMut(AgentEvent),
) -> Result<ScriptResult, String> {
    // A persistent scratch dir: scripts may print paths to files they wrote
    // here, which the model may read back later, so don't auto-delete.
    let script_dir: PathBuf = std::env::temp_dir().join(format!(
        "generalist-script-{}",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::create_dir_all(&script_dir)
        .map_err(|e| format!("Failed to create script dir: {}", e))?;

    std::fs::write(
        script_dir.join("tools.py"),
        generate_tools_module(&registry.get_bridge_tool_defs()),
    )
    .map_err(|e| format!("Failed to write tools.py: {}", e))?;
    let main_py = script_dir.join("main.py");
    std::fs::write(&main_py, code).map_err(|e| format!("Failed to write main.py: {}", e))?;
    let runner_py = script_dir.join("runner.py");
    std::fs::write(&runner_py, RUNNER_MODULE)
        .map_err(|e| format!("Failed to write runner.py: {}", e))?;

    // Unix socket paths are limited to ~104 bytes (SUN_LEN); macOS per-user
    // temp dirs under /var/folders are long enough to overflow it, so the
    // socket lives under /tmp with a short name — inside a freshly created
    // mode-0700 directory, because /tmp is world-listable and a bare socket
    // would otherwise be connectable by any local user.
    let bridge_dir = PathBuf::from(format!(
        "/tmp/gnl-{:.12}",
        uuid::Uuid::new_v4().simple().to_string()
    ));
    {
        use std::os::unix::fs::DirBuilderExt;
        // `create` (not `create_all`): an existing path here is unexpected
        // and must fail rather than adopt a directory someone else planted.
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&bridge_dir)
            .map_err(|e| format!("Failed to create bridge socket dir: {}", e))?;
    }
    let socket_path = bridge_dir.join("t.sock");
    // Install the guard immediately after creation: metadata inspection or
    // socket binding can fail too, and must not strand a private /tmp dir.
    let _socket_guard = RemoveOnDrop {
        socket: socket_path.clone(),
        directory: bridge_dir.clone(),
    };
    let host_uid = {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(&bridge_dir)
            .map_err(|e| format!("Failed to inspect bridge socket dir: {}", e))?
            .uid()
    };
    let listener = UnixListener::bind(&socket_path)
        .map_err(|e| format!("Failed to bind bridge socket: {}", e))?;

    // Inherit the agent's cwd so the model script reads/writes project files
    // naturally. runner.py's directory is first on sys.path, and run_path
    // gives main.py its normal file identity while preloading the bridge.
    let mut child = Command::new("python3")
        .arg(&runner_py)
        .arg(&main_py)
        .env("GENERALIST_TOOL_SOCKET", &socket_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Own process group so a timeout can kill descendants too.
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to start python3: {}. Is it installed?", e))?;

    // Drain output concurrently — with bounded memory — so a chatty script
    // can neither dead-lock against a pending tool call nor OOM the agent.
    let stdout_pipe = child.stdout.take().expect("stdout piped");
    let stderr_pipe = child.stderr.take().expect("stderr piped");
    let stdout_task = tokio::spawn(crate::subprocess::collect_bounded(
        stdout_pipe,
        "generalist-script-stdout-",
    ));
    let stderr_task = tokio::spawn(crate::subprocess::collect_bounded(
        stderr_pipe,
        "generalist-script-stderr-",
    ));

    let mut denied = false;
    let mut bridged_calls: u64 = 0;
    // The timeout budget covers script compute only. Time spent inside
    // bridged tool calls — including however long the user takes to answer a
    // permission prompt — extends the deadline, so a slow approval can never
    // kill the script underneath the user; each bridged tool bounds its own
    // execution instead.
    let started = tokio::time::Instant::now();
    let mut serving = Duration::ZERO;
    let status = loop {
        let deadline = started + Duration::from_secs(timeout_secs) + serving;
        tokio::select! {
            status = child.wait() => break status,
            accepted = listener.accept() => {
                if let Ok((stream, _)) = accepted {
                    // Belt over the 0700 directory: only this user's
                    // processes may drive the bridge.
                    let authorized = stream
                        .peer_cred()
                        .map(|cred| cred.uid() == host_uid)
                        .unwrap_or(false);
                    if !authorized {
                        continue;
                    }
                    let serve_started = tokio::time::Instant::now();
                    serve_connection(
                        stream,
                        registry,
                        on_event,
                        &mut denied,
                        &mut bridged_calls,
                    )
                    .await;
                    serving += serve_started.elapsed();
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                crate::subprocess::kill_process_group(&child);
                let _ = child.start_kill();
                return Ok(ScriptResult {
                    content: format!(
                        "Script timed out after {} seconds and was killed.",
                        timeout_secs
                    ),
                    failed: true,
                    denied,
                });
            }
        }
    };
    let status = status.map_err(|e| format!("Failed to run python: {}", e))?;

    let stdout = match stdout_task.await {
        Ok(stream) => stream.into_text(),
        Err(_) => String::new(),
    };
    let stderr = match stderr_task.await {
        Ok(stream) => stream.into_text(),
        Err(_) => String::new(),
    };

    let failed = !status.success();
    let content = crate::subprocess::combine_output(status, &stdout, &stderr);

    let mut content = crate::tools::bash::tail_truncate_with_spill(&content);
    if !failed && bridged_calls > 0 && content.trim().is_empty() {
        content = format!(
            "Script completed successfully with no output. {bridged_calls} tool call{} executed \
             through the bridge; their results stayed inside the script. Print any values the \
             next step needs.",
            if bridged_calls == 1 { "" } else { "s" }
        );
    }

    Ok(ScriptResult {
        content,
        failed,
        denied,
    })
}

/// Serve bridged tool calls on one connection until the script closes it.
async fn serve_connection(
    stream: UnixStream,
    registry: &mut ToolRegistry,
    on_event: &mut dyn FnMut(AgentEvent),
    denied: &mut bool,
    bridged_calls: &mut u64,
) {
    let (read_half, mut write_half) = stream.into_split();
    let mut lines = BufReader::new(read_half).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let response = handle_request(&line, registry, on_event, denied, bridged_calls).await;
        let mut payload = serde_json::to_string(&response).unwrap_or_else(|_| {
            "{\"is_error\": true, \"content\": \"serialization failed\"}".to_string()
        });
        payload.push('\n');
        if write_half.write_all(payload.as_bytes()).await.is_err() {
            break;
        }
    }
}

async fn handle_request(
    line: &str,
    registry: &mut ToolRegistry,
    on_event: &mut dyn FnMut(AgentEvent),
    denied: &mut bool,
    bridged_calls: &mut u64,
) -> Value {
    *bridged_calls += 1;
    let request: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return json!({"is_error": true, "content": format!("invalid request: {}", e)}),
    };
    let Some(tool) = request
        .get("tool")
        .and_then(|t| t.as_str())
        .map(str::to_string)
    else {
        return json!({"is_error": true, "content": "request missing 'tool'"});
    };
    let input = request.get("input").cloned().unwrap_or_else(|| json!({}));

    on_event(AgentEvent::ToolCallStarted {
        name: tool.clone(),
        input: input.clone(),
    });
    let call_id = format!("script_{}", uuid::Uuid::new_v4().simple());
    let result = registry.execute_tool(&tool, input, call_id).await;
    let (content, is_error) = match result.block {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => (content, is_error.unwrap_or(false)),
        _ => (String::new(), true),
    };
    on_event(AgentEvent::ToolCallFinished {
        name: tool,
        outcome: result.outcome,
        content: crate::types::truncate_middle(&content, 2_000),
    });
    *denied |= result.outcome == ToolCallOutcome::Denied;
    json!({"is_error": is_error, "content": content})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_module_is_valid_python_with_awkward_descriptions() {
        let defs = vec![ToolDef {
            name: "tricky".into(),
            description: "Contains \"quotes\", 'apostrophes',\nnewlines and \\backslashes".into(),
            input_schema: json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        }];
        let module = generate_tools_module(&defs);
        assert!(module.contains("def tricky(**kwargs):"));
        assert!(module.contains("Call: tools.tricky(*, x: str | None = None, **extra) -> str"));

        // Ask Python itself whether the module compiles.
        let dir = std::env::temp_dir().join(format!("gm-test-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tools.py");
        std::fs::write(&path, &module).unwrap();
        let status = std::process::Command::new("python3")
            .arg("-m")
            .arg("py_compile")
            .arg(&path)
            .status()
            .expect("python3 available");
        assert!(status.success(), "generated tools.py does not compile");
    }

    #[test]
    fn compact_signatures_put_required_parameters_first() {
        let tool = ToolDef {
            name: "search".into(),
            description: "Search records".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "query": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        };
        assert_eq!(
            python_call_signature(&tool),
            "tools.search(*, query: str, limit: int | None = None, tags: list[str] | None = None) -> str"
        );
        let definition = python_tool_def(&[tool], &[]);
        assert!(definition
            .description
            .contains("tools.search(*, query: str"));
        assert!(!definition.description.contains("Input schema:"));
        assert!(definition.description.contains("__doc__"));
    }
}
