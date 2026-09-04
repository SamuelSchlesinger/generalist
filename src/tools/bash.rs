use crate::sandbox::{ExecutionSandbox, SandboxAccess};
use crate::subprocess::{
    collect_bounded, combine_output, finish_output_collector, ProcessGroupGuard,
    DEFAULT_TIMEOUT_SECS, MAX_TIMEOUT_SECS,
};
use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use tokio::time::{timeout, Duration};

/// Output cap before spilling to a file: keep the *tail* (errors and results
/// live at the end of command output).
const MAX_OUTPUT_CHARS: usize = 50_000;

pub struct BashTool;

pub struct SandboxedBashTool {
    sandbox: ExecutionSandbox,
}

impl SandboxedBashTool {
    pub fn new(sandbox: ExecutionSandbox) -> Self {
        Self { sandbox }
    }
}

fn bash_description() -> &'static str {
    "Execute a bash command. Use for file operations, running programs, git, and anything \
     else a shell can do. Output is truncated to the last 50,000 characters; when truncated, \
     the full output is saved to a temp file whose path is included in the result. Commands \
     time out after 120 seconds by default (override with timeout_seconds, max 600)."
}

fn bash_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command or script to execute"
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Timeout in seconds (default: 120, max: 600)"
            }
        },
        "required": ["command"],
        "additionalProperties": false
    })
}

pub(crate) fn tail_truncate_with_spill(output: &str, spill_directory: Option<&Path>) -> String {
    let count = output.chars().count();
    if count <= MAX_OUTPUT_CHARS {
        return output.to_string();
    }
    // Preserve the full output somewhere the model can read it back.
    let spill_note = {
        let mut builder = tempfile::Builder::new();
        builder.prefix("generalist-bash-").suffix(".txt");
        match spill_directory {
            Some(directory) => builder.tempfile_in(directory),
            None => builder.tempfile(),
        }
    }
    .and_then(|mut f| {
        f.write_all(output.as_bytes())?;
        let (_file, path) = f.keep().map_err(|e| e.error)?;
        Ok(path)
    })
    .map(|path| format!(" Full output saved to: {}", path.display()))
    .unwrap_or_default();

    let tail: String = output.chars().skip(count - MAX_OUTPUT_CHARS).collect();
    format!(
        "[Output truncated: showing last {} of {} characters.{}]\n{}",
        MAX_OUTPUT_CHARS, count, spill_note, tail
    )
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        bash_description()
    }

    fn input_schema(&self) -> Value {
        bash_input_schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        execute_bash(input, &ExecutionSandbox::unrestricted()).await
    }
}

#[async_trait]
impl Tool for SandboxedBashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        bash_description()
    }

    fn input_schema(&self) -> Value {
        bash_input_schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        execute_bash(input, &self.sandbox).await
    }
}

async fn execute_bash(input: Value, sandbox: &ExecutionSandbox) -> Result<String> {
    let command = input
        .get("command")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Tool("Missing 'command' field".to_string()))?;

    let timeout_secs = input
        .get("timeout_seconds")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_TIMEOUT_SECS)
        .clamp(1, MAX_TIMEOUT_SECS);

    let spill_directory = sandbox.scratch_directory().map(Path::to_path_buf);
    let mut prepared = sandbox.prepare_command("bash", &SandboxAccess::default())?;
    let mut child = prepared
        .command_mut()
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Own process group so a timeout can kill descendants too.
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Tool(format!("Failed to start bash: {}", e)))?;
    let mut process_group = ProcessGroupGuard::new(child.id());

    let stdout_task = tokio::spawn(collect_bounded(
        child.stdout.take().expect("stdout piped"),
        "generalist-bash-stdout-",
        spill_directory.clone(),
    ));
    let stderr_task = tokio::spawn(collect_bounded(
        child.stderr.take().expect("stderr piped"),
        "generalist-bash-stderr-",
        spill_directory.clone(),
    ));
    let status = match timeout(Duration::from_secs(timeout_secs), child.wait()).await {
        Ok(status) => status,
        Err(_) => {
            // A timeout is a structured failure, never a success result.
            process_group.kill();
            let _ = child.start_kill();
            let _ = child.wait().await;
            stdout_task.abort();
            stderr_task.abort();
            return Err(Error::Tool(format!(
                "Command timed out after {} seconds and was killed.",
                timeout_secs
            )));
        }
    };
    // A command may close its pipes and exit after leaving a background child.
    // End the complete execution scope even on an ordinary successful return.
    process_group.kill();
    let (stdout, stderr) = tokio::join!(
        finish_output_collector(stdout_task),
        finish_output_collector(stderr_task)
    );
    let status = status.map_err(|e| Error::Tool(format!("Failed to run bash command: {}", e)))?;

    let combined = combine_output(status, &stdout, &stderr);
    Ok(tail_truncate_with_spill(
        &combined,
        spill_directory.as_deref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn runs_commands_and_reports_exit_codes() {
        let ok = BashTool
            .execute(json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert_eq!(ok.trim(), "hello");

        let err = BashTool
            .execute(json!({"command": "exit 3"}))
            .await
            .unwrap();
        assert!(err.contains("Exit code: 3"));
    }

    #[tokio::test]
    async fn successful_command_cannot_leave_a_background_descendant() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("escaped.txt");
        let marker_literal = serde_json::to_string(&marker.to_string_lossy()).unwrap();
        let command = format!(
            "python3 -c 'import os,time; pid=os.fork(); pid and os._exit(0); time.sleep(1); open({marker_literal}, \"w\").write(\"escaped\")'"
        );
        BashTool
            .execute(json!({"command": command, "timeout_seconds": 5}))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert!(
            !marker.exists(),
            "background descendant survived tool return"
        );
    }

    #[tokio::test]
    async fn cancelling_the_tool_future_kills_the_complete_process_group() {
        let directory = tempfile::tempdir().unwrap();
        let started = directory.path().join("child-started.txt");
        let escaped = directory.path().join("escaped.txt");
        let started_literal = serde_json::to_string(&started.to_string_lossy()).unwrap();
        let escaped_literal = serde_json::to_string(&escaped.to_string_lossy()).unwrap();
        let command = format!(
            "(printf started > {started_literal}; sleep 1; printf escaped > {escaped_literal}) & wait"
        );
        let task = tokio::spawn(async move {
            BashTool
                .execute(json!({"command": command, "timeout_seconds": 10}))
                .await
        });
        for _ in 0..200 {
            if started.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(started.exists(), "background child never started");

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert!(!escaped.exists(), "cancelled tool left a live descendant");
    }

    #[tokio::test]
    async fn times_out_and_kills() {
        let error = BashTool
            .execute(json!({"command": "sleep 30", "timeout_seconds": 1}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn tail_truncation_keeps_the_end() {
        let long = format!("{}{}", "a".repeat(100_000), "THE_END");
        let out = tail_truncate_with_spill(&long, None);
        assert!(out.ends_with("THE_END"));
        assert!(out.contains("truncated"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_shell_can_run_the_workspace_rust_toolchain() {
        let sandbox = ExecutionSandbox::restricted(env!("CARGO_MANIFEST_DIR")).unwrap();
        let output = SandboxedBashTool::new(sandbox)
            .execute(json!({
                "command": "cargo --version && cargo metadata --no-deps --offline --format-version 1 >/dev/null && git status --short >/dev/null"
            }))
            .await
            .unwrap();
        assert!(output.starts_with("cargo "), "{output}");
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_shell_reuses_private_temp_and_can_read_output_spills() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox = ExecutionSandbox::restricted(workspace.path()).unwrap();
        let spill = tail_truncate_with_spill(
            &"x".repeat(MAX_OUTPUT_CHARS + 1),
            sandbox.scratch_directory(),
        );
        let spill_path = spill
            .split("Full output saved to: ")
            .nth(1)
            .and_then(|value| value.split(']').next())
            .expect("spill path");
        let command = format!(
            "printf session > \"$HOME/state\" && test -r {}",
            serde_json::to_string(spill_path).unwrap()
        );
        let tool = SandboxedBashTool::new(sandbox);
        let first = tool.execute(json!({"command": command})).await.unwrap();
        assert!(first.trim().is_empty(), "{first}");
        let second = tool
            .execute(json!({"command": "cat \"$HOME/state\""}))
            .await
            .unwrap();
        assert_eq!(second.trim(), "session");
    }
}
