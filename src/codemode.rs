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
//! Mechanics: the agent writes `tools.py` + `main.py` to an ephemeral scratch dir, runs
//! `main.py` through a small wrapper that preloads `tools`, and serves tool
//! calls over a Unix socket. Every
//! bridged call goes through the same `ToolRegistry` permission gate as a
//! direct call, so interactive approval still applies.

use crate::agent::AgentEvent;
use crate::sandbox::SandboxAccess;
use crate::tool::{ToolCallOutcome, ToolRegistry};
use crate::types::{ContentBlock, ToolDef};
use serde_json::{json, Value};
use std::os::fd::OwnedFd;
use std::os::unix::net::UnixStream as StdUnixStream;
use std::path::Path;
use std::process::Stdio;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::time::Duration;

/// The name the code-mode tool is advertised under.
pub(crate) const PY_TOOL_NAME: &str = "python";

pub(crate) use crate::subprocess::{DEFAULT_TIMEOUT_SECS, MAX_TIMEOUT_SECS};

const MAX_BRIDGE_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_BRIDGE_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_BRIDGED_TOOL_CALLS: u64 = 64;
const BRIDGE_WRITE_TIMEOUT_SECS: u64 = 2;

const RUNNER_MODULE: &str = r#""""Generalist code-mode bootstrap."""
import os as _os
import runpy as _runpy
import sys as _sys

# -I -S deliberately omits the script/project directory and site hooks.
# Add only this owner-private scaffold directory before importing the bridge.
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
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
             only conclusions, compact evidence, and paths; write durable intermediate results \
             under the current project because the runner's private scratch directory is removed \
             after execution. Output is truncated to the last 50,000 characters (full output is saved to a \
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

_bridge = _os.fdopen(0, "r+b", buffering=0)
_MAX_REQUEST_BYTES = 1048576
_MAX_RESPONSE_BYTES = 8388608


def _call(_tool, **kwargs):
    request = (_json.dumps({"tool": _tool, "input": kwargs}) + "\n").encode()
    if len(request) > _MAX_REQUEST_BYTES:
        raise RuntimeError(f"{_tool}: tool request exceeds the 1 MiB bridge limit")
    _bridge.write(request)
    line = _bridge.readline(_MAX_RESPONSE_BYTES + 1)
    if not line:
        raise RuntimeError("tool bridge closed unexpectedly")
    if len(line) > _MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
        raise RuntimeError("tool bridge response exceeds the 8 MiB bridge limit")
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

fn create_private_script_directory() -> std::io::Result<tempfile::TempDir> {
    use std::os::unix::fs::PermissionsExt;

    tempfile::Builder::new()
        .prefix("generalist-script-")
        .permissions(std::fs::Permissions::from_mode(0o700))
        .tempdir()
}

fn write_private_script_file(path: &Path, contents: impl AsRef<[u8]>) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(contents.as_ref())
}

async fn read_bridge_frame<R: AsyncBufRead + Unpin>(
    reader: &mut R,
) -> std::io::Result<Option<Vec<u8>>> {
    let mut frame = Vec::new();
    loop {
        let (consumed, complete) = {
            let available = reader.fill_buf().await?;
            if available.is_empty() {
                if frame.is_empty() {
                    return Ok(None);
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "tool bridge closed in the middle of a request",
                ));
            }
            let newline = available.iter().position(|byte| *byte == b'\n');
            let content_bytes = newline.unwrap_or(available.len());
            if frame.len().saturating_add(content_bytes) > MAX_BRIDGE_REQUEST_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "tool bridge request exceeds the 1 MiB limit",
                ));
            }
            frame.extend_from_slice(&available[..content_bytes]);
            (
                content_bytes + usize::from(newline.is_some()),
                newline.is_some(),
            )
        };
        reader.consume(consumed);
        if complete {
            return Ok(Some(frame));
        }
    }
}

struct BoundedBridgeResponse {
    bytes: Vec<u8>,
}

impl std::io::Write for BoundedBridgeResponse {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        // Reserve one byte for the newline framing delimiter consumed by the
        // generated Python bridge.
        let limit = MAX_BRIDGE_RESPONSE_BYTES - 1;
        let next = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("tool bridge response length overflow"))?;
        if next > limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "tool bridge response exceeds the 8 MiB limit",
            ));
        }
        self.bytes.try_reserve(bytes.len()).map_err(|_| {
            std::io::Error::other("failed to reserve bounded tool bridge response memory")
        })?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn serialize_bridge_response(response: &Value) -> Vec<u8> {
    const TOO_LARGE: &[u8] =
        b"{\"is_error\":true,\"content\":\"tool bridge response exceeds the 8 MiB limit\"}\n";
    let mut writer = BoundedBridgeResponse { bytes: Vec::new() };
    if serde_json::to_writer(&mut writer, response).is_err() {
        return TOO_LARGE.to_vec();
    }
    writer.bytes.push(b'\n');
    writer.bytes
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
    // Scaffold source can contain conversation data and tool schemas. Keep it
    // in an atomically created owner-only directory and remove the whole tree
    // when execution ends. Scripts that need durable artifacts write them in
    // the inherited project cwd.
    let script_directory = create_private_script_directory()
        .map_err(|e| format!("Failed to create script dir: {}", e))?;
    let script_dir = script_directory.path();

    write_private_script_file(
        &script_dir.join("tools.py"),
        generate_tools_module(&registry.get_bridge_tool_defs()),
    )
    .map_err(|e| format!("Failed to write tools.py: {}", e))?;
    let main_py = script_dir.join("main.py");
    write_private_script_file(&main_py, code)
        .map_err(|e| format!("Failed to write main.py: {}", e))?;
    let runner_py = script_dir.join("runner.py");
    write_private_script_file(&runner_py, RUNNER_MODULE)
        .map_err(|e| format!("Failed to write runner.py: {}", e))?;

    // A pre-connected socketpair is both shorter and safer than a pathname
    // socket: there is nothing for another local process to discover or dial,
    // and Linux can block creation of every new socket in the child.
    let (host_stream, child_stream) =
        StdUnixStream::pair().map_err(|e| format!("Failed to create tool bridge: {e}"))?;
    host_stream
        .set_nonblocking(true)
        .map_err(|e| format!("Failed to configure tool bridge: {e}"))?;
    let stream = UnixStream::from_std(host_stream)
        .map_err(|e| format!("Failed to start tool bridge: {e}"))?;
    let child_bridge: OwnedFd = child_stream.into();

    // Inherit the agent's cwd so the model script reads/writes project files
    // naturally. runner.py's directory is first on sys.path, and run_path
    // gives main.py its normal file identity while preloading the bridge.
    let sandbox = registry.execution_sandbox().clone();
    let mut access = SandboxAccess::default();
    access.allow_read_only(script_dir);
    let mut prepared = sandbox
        .prepare_command("python3", &access)
        .map_err(|e| format!("Failed to prepare Python sandbox: {e}"))?;
    let mut child = prepared
        .command_mut()
        // Isolated/no-site startup prevents project-controlled sitecustomize,
        // user packages, PYTHON* variables, and the cwd from executing before
        // the trusted runner establishes the bridge.
        .arg("-I")
        .arg("-S")
        .arg(&runner_py)
        .arg(&main_py)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        // Map only the pre-connected bridge into the child. Using stdin lets
        // the standard spawn plumbing perform the descriptor transfer without
        // a process-wide close-on-exec race.
        .stdin(Stdio::from(child_bridge))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Own process group so a timeout can kill descendants too.
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to start python3: {}. Is it installed?", e))?;
    let mut process_group = crate::subprocess::ProcessGroupGuard::new(child.id());

    // Drain output concurrently — with bounded memory — so a chatty script
    // can neither dead-lock against a pending tool call nor OOM the agent.
    let stdout_pipe = child.stdout.take().expect("stdout piped");
    let stderr_pipe = child.stderr.take().expect("stderr piped");
    let stdout_task = tokio::spawn(crate::subprocess::collect_bounded(
        stdout_pipe,
        "generalist-script-stdout-",
        sandbox.scratch_directory().map(Path::to_path_buf),
    ));
    let stderr_task = tokio::spawn(crate::subprocess::collect_bounded(
        stderr_pipe,
        "generalist-script-stderr-",
        sandbox.scratch_directory().map(Path::to_path_buf),
    ));

    let mut denied = false;
    let mut bridged_calls: u64 = 0;
    let mut bridge_failure = None;
    let (read_half, mut write_half) = stream.into_split();
    let mut frames = BufReader::new(read_half);
    let mut bridge_open = true;
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
            frame = read_bridge_frame(&mut frames), if bridge_open => {
                match frame {
                    Ok(Some(frame)) if bridged_calls < MAX_BRIDGED_TOOL_CALLS => {
                    let serve_started = tokio::time::Instant::now();
                    let response = handle_request(
                        &frame,
                        registry,
                        on_event,
                        &mut denied,
                        &mut bridged_calls,
                    )
                    .await;
                    let payload = serialize_bridge_response(&response);
                    let write_result = tokio::time::timeout(
                        Duration::from_secs(BRIDGE_WRITE_TIMEOUT_SECS),
                        write_half.write_all(&payload),
                    )
                    .await;
                    if !matches!(write_result, Ok(Ok(()))) {
                        bridge_failure = Some(match write_result {
                            Ok(Err(error)) => format!(
                                "Python tool bridge could not return a result: {error}"
                            ),
                            Err(_) => format!(
                                "Python tool bridge did not accept a result within \
                                 {BRIDGE_WRITE_TIMEOUT_SECS} seconds"
                            ),
                            Ok(Ok(())) => unreachable!(),
                        });
                        process_group.kill();
                        let _ = child.start_kill();
                        break child.wait().await;
                    }
                    serving += serve_started.elapsed();
                    }
                    Ok(Some(_)) => {
                        bridge_failure = Some(format!(
                            "Python tool bridge stopped after the maximum of \
                             {MAX_BRIDGED_TOOL_CALLS} calls. Split the work into another script."
                        ));
                        process_group.kill();
                        let _ = child.start_kill();
                        break child.wait().await;
                    }
                    Ok(None) => bridge_open = false,
                    Err(error) => {
                        bridge_failure = Some(format!("Python tool bridge rejected a request: {error}"));
                        process_group.kill();
                        let _ = child.start_kill();
                        break child.wait().await;
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                process_group.kill();
                let _ = child.start_kill();
                stdout_task.abort();
                stderr_task.abort();
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
    process_group.kill();
    drop(frames);
    drop(write_half);

    let (stdout, stderr) = tokio::join!(
        crate::subprocess::finish_output_collector(stdout_task),
        crate::subprocess::finish_output_collector(stderr_task)
    );
    let status = status.map_err(|e| format!("Failed to run python: {}", e))?;

    let failed = !status.success() || bridge_failure.is_some();
    let process_content = crate::subprocess::combine_output(status, &stdout, &stderr);
    let content = match bridge_failure {
        Some(message) if process_content.trim().is_empty() => message,
        Some(message) => format!("{message}\n{process_content}"),
        None => process_content,
    };

    let mut content =
        crate::tools::bash::tail_truncate_with_spill(&content, sandbox.scratch_directory());
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

async fn handle_request(
    frame: &[u8],
    registry: &mut ToolRegistry,
    on_event: &mut dyn FnMut(AgentEvent),
    denied: &mut bool,
    bridged_calls: &mut u64,
) -> Value {
    *bridged_calls += 1;
    let request: Value = match serde_json::from_slice(frame) {
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

    #[tokio::test]
    async fn bridge_frames_are_bounded_and_preserve_message_boundaries() {
        let input = b"one\ntwo\n";
        let mut reader = BufReader::new(&input[..]);
        assert_eq!(
            read_bridge_frame(&mut reader).await.unwrap().unwrap(),
            b"one"
        );
        assert_eq!(
            read_bridge_frame(&mut reader).await.unwrap().unwrap(),
            b"two"
        );
        assert!(read_bridge_frame(&mut reader).await.unwrap().is_none());

        let mut exact = vec![b'x'; MAX_BRIDGE_REQUEST_BYTES];
        exact.push(b'\n');
        let mut reader = BufReader::new(exact.as_slice());
        assert_eq!(
            read_bridge_frame(&mut reader).await.unwrap().unwrap().len(),
            MAX_BRIDGE_REQUEST_BYTES
        );

        let mut oversized = vec![b'x'; MAX_BRIDGE_REQUEST_BYTES + 1];
        oversized.push(b'\n');
        let mut reader = BufReader::new(oversized.as_slice());
        assert_eq!(
            read_bridge_frame(&mut reader).await.unwrap_err().kind(),
            std::io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn bridge_response_serialization_is_bounded_even_when_json_escaping_expands() {
        let response = json!({
            "is_error": false,
            "content": "\0".repeat(MAX_BRIDGE_RESPONSE_BYTES / 2),
        });
        let payload = serialize_bridge_response(&response);

        assert!(payload.len() <= MAX_BRIDGE_RESPONSE_BYTES);
        assert_eq!(payload.last(), Some(&b'\n'));
        let decoded: Value = serde_json::from_slice(&payload[..payload.len() - 1]).unwrap();
        assert_eq!(decoded["is_error"], true);
        assert!(decoded["content"]
            .as_str()
            .unwrap()
            .contains("exceeds the 8 MiB limit"));
    }

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
        assert!(module.contains("_os.fdopen(0"));
        assert!(!module.contains("GENERALIST_TOOL_FD"));
        assert!(!module.contains("GENERALIST_TOOL_SOCKET"));

        // Ask Python itself whether the module compiles.
        let dir = create_private_script_directory().unwrap();
        let path = dir.path().join("tools.py");
        write_private_script_file(&path, &module).unwrap();
        let status = std::process::Command::new("python3")
            .arg("-m")
            .arg("py_compile")
            .arg(&path)
            .status()
            .expect("python3 available");
        assert!(status.success(), "generated tools.py does not compile");
    }

    #[test]
    fn script_scaffold_is_private_and_ephemeral() {
        use std::os::unix::fs::PermissionsExt;

        let directory_path;
        {
            let directory = create_private_script_directory().unwrap();
            directory_path = directory.path().to_path_buf();
            assert_eq!(
                std::fs::metadata(&directory_path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o700
            );

            let script = directory_path.join("main.py");
            write_private_script_file(&script, b"print('private')\n").unwrap();
            assert_eq!(
                std::fs::metadata(script).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        assert!(!directory_path.exists());
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

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_runner_can_write_workspace_but_not_read_a_sibling() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let secret = parent.path().join("secret.txt");
        std::fs::write(&secret, "secret").unwrap();
        let output = workspace.join("output.txt");

        let mut registry = ToolRegistry::new();
        registry.set_execution_sandbox(
            crate::sandbox::ExecutionSandbox::restricted(&workspace).unwrap(),
        );
        let code = format!(
            concat!(
                "from pathlib import Path\n",
                "secret = Path({})\n",
                "output = Path({})\n",
                "try:\n secret.read_text()\n raise SystemExit('escaped')\n",
                "except PermissionError:\n pass\n",
                "output.write_text('inside')\n",
                "print('contained')\n",
            ),
            serde_json::to_string(&secret.to_string_lossy()).unwrap(),
            serde_json::to_string(&output.to_string_lossy()).unwrap(),
        );
        let result = run_script(&code, 10, &mut registry, &mut |_| {}).await;

        assert!(!result.failed, "{}", result.content);
        assert!(result.content.contains("contained"));
        assert_eq!(std::fs::read_to_string(output).unwrap(), "inside");
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_runner_ignores_project_python_startup_hooks() {
        let workspace = tempfile::tempdir().unwrap();
        let marker = workspace.path().join("startup-hook-ran.txt");
        let startup = format!(
            "from pathlib import Path\nPath({}).write_text('escaped')\n",
            serde_json::to_string(&marker.to_string_lossy()).unwrap()
        );
        std::fs::write(workspace.path().join("sitecustomize.py"), startup).unwrap();

        let mut registry = ToolRegistry::new();
        registry.set_execution_sandbox(
            crate::sandbox::ExecutionSandbox::restricted(workspace.path()).unwrap(),
        );
        let result = run_script("print('isolated')\n", 10, &mut registry, &mut |_| {}).await;

        assert!(!result.failed, "{}", result.content);
        assert!(result.content.contains("isolated"));
        assert!(
            !marker.exists(),
            "project sitecustomize.py ran before the script"
        );
    }

    #[tokio::test]
    async fn timeout_still_runs_after_a_bridged_tool_call() {
        use crate::tools::CalculatorTool;
        use std::sync::Arc;

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(CalculatorTool)).unwrap();
        let result = run_script(
            "import time\ntools.calculator(expression='1+1')\ntime.sleep(10)\n",
            1,
            &mut registry,
            &mut |_| {},
        )
        .await;

        assert!(result.failed);
        assert!(result.content.contains("timed out"), "{}", result.content);
    }

    struct LargeBridgeResultTool;

    #[async_trait::async_trait]
    impl crate::Tool for LargeBridgeResultTool {
        fn name(&self) -> &str {
            "large_bridge_result"
        }

        fn description(&self) -> &str {
            "Return a large result for bridge backpressure testing"
        }

        fn input_schema(&self) -> Value {
            json!({"type": "object", "additionalProperties": false})
        }

        async fn execute(&self, _input: Value) -> crate::Result<String> {
            Ok("x".repeat(7 * 1024 * 1024))
        }
    }

    #[tokio::test]
    async fn bridge_backpressure_cannot_hold_the_host_past_its_write_deadline() {
        use std::sync::Arc;

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(LargeBridgeResultTool)).unwrap();
        let started = tokio::time::Instant::now();
        let result = run_script(
            concat!(
                "import os,time\n",
                "os.write(0,b'{\"tool\":\"large_bridge_result\",\"input\":{}}\\n')\n",
                "time.sleep(30)\n",
            ),
            10,
            &mut registry,
            &mut |_| {},
        )
        .await;

        assert!(result.failed);
        assert!(
            result.content.contains("did not accept a result"),
            "{}",
            result.content
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "bridge write deadline was not enforced"
        );
    }

    #[tokio::test]
    async fn bridge_stops_scripts_after_the_call_budget() {
        use crate::tools::CalculatorTool;
        use std::sync::Arc;

        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(CalculatorTool)).unwrap();
        let result = run_script(
            "for _ in range(65): tools.calculator(expression='1+1')\n",
            10,
            &mut registry,
            &mut |_| {},
        )
        .await;

        assert!(result.failed);
        assert!(
            result.content.contains("maximum of 64 calls"),
            "{}",
            result.content
        );
    }

    #[tokio::test]
    async fn successful_script_cannot_leave_a_background_descendant() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("escaped.txt");
        let code = format!(
            concat!(
                "import os,time\n",
                "pid=os.fork()\n",
                "if pid: os._exit(0)\n",
                "os.close(0); os.close(1); os.close(2)\n",
                "time.sleep(1)\n",
                "open({}, 'w').write('escaped')\n",
            ),
            serde_json::to_string(&marker.to_string_lossy()).unwrap(),
        );
        let mut registry = ToolRegistry::new();
        let result = run_script(&code, 5, &mut registry, &mut |_| {}).await;
        assert!(!result.failed, "{}", result.content);
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert!(
            !marker.exists(),
            "background descendant survived tool return"
        );
    }

    #[tokio::test]
    async fn cancelling_the_script_future_kills_the_complete_process_group() {
        let directory = tempfile::tempdir().unwrap();
        let started = directory.path().join("child-started.txt");
        let escaped = directory.path().join("escaped.txt");
        let code = format!(
            concat!(
                "import os,time\n",
                "pid=os.fork()\n",
                "if pid == 0:\n",
                " open({}, 'w').write('started')\n",
                " time.sleep(1)\n",
                " open({}, 'w').write('escaped')\n",
                " os._exit(0)\n",
                "time.sleep(30)\n",
            ),
            serde_json::to_string(&started.to_string_lossy()).unwrap(),
            serde_json::to_string(&escaped.to_string_lossy()).unwrap(),
        );
        let mut registry = ToolRegistry::new();
        let mut events = |_| {};
        let mut script = Box::pin(run_script(&code, 10, &mut registry, &mut events));
        for _ in 0..200 {
            tokio::select! {
                result = &mut script => panic!("script exited before cancellation: {}", result.content),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
            if started.exists() {
                break;
            }
        }
        assert!(started.exists(), "background child never started");

        drop(script);
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert!(!escaped.exists(), "cancelled script left a live descendant");
    }
}
