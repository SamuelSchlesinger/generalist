use crate::subprocess::{
    collect_bounded, combine_output, kill_process_group, DEFAULT_TIMEOUT_SECS, MAX_TIMEOUT_SECS,
};
use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::io::Write;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

/// Output cap before spilling to a file: keep the *tail* (errors and results
/// live at the end of command output).
const MAX_OUTPUT_CHARS: usize = 50_000;

pub struct BashTool;

pub(crate) fn tail_truncate_with_spill(output: &str) -> String {
    let count = output.chars().count();
    if count <= MAX_OUTPUT_CHARS {
        return output.to_string();
    }
    // Preserve the full output somewhere the model can read it back.
    let spill_note = tempfile::Builder::new()
        .prefix("generalist-bash-")
        .suffix(".txt")
        .tempfile()
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
        "Execute a bash command. Use for file operations, running programs, git, and anything \
         else a shell can do. Output is truncated to the last 50,000 characters; when truncated, \
         the full output is saved to a temp file whose path is included in the result. Commands \
         time out after 120 seconds by default (override with timeout_seconds, max 600)."
    }

    fn input_schema(&self) -> Value {
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

    async fn execute(&self, input: Value) -> Result<String> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Tool("Missing 'command' field".to_string()))?;

        let timeout_secs = input
            .get("timeout_seconds")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);

        let mut child = Command::new("bash")
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

        let stdout_pipe = child.stdout.take().expect("stdout piped");
        let stderr_pipe = child.stderr.take().expect("stderr piped");
        // Drain both pipes concurrently with the wait, with bounded memory,
        // so a chatty command can neither dead-lock nor OOM. The inner scope
        // ends the future's borrow of `child` before the timeout kill.
        let finished = {
            let run = async {
                tokio::join!(
                    child.wait(),
                    collect_bounded(stdout_pipe, "generalist-bash-stdout-"),
                    collect_bounded(stderr_pipe, "generalist-bash-stderr-"),
                )
            };
            tokio::pin!(run);
            timeout(Duration::from_secs(timeout_secs), &mut run)
                .await
                .ok()
        };
        let Some((status, stdout, stderr)) = finished else {
            // A timeout is a structured failure, never a success result.
            kill_process_group(&child);
            let _ = child.start_kill();
            return Err(Error::Tool(format!(
                "Command timed out after {} seconds and was killed.",
                timeout_secs
            )));
        };
        let status =
            status.map_err(|e| Error::Tool(format!("Failed to run bash command: {}", e)))?;

        let combined = combine_output(status, &stdout.into_text(), &stderr.into_text());
        Ok(tail_truncate_with_spill(&combined))
    }
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
        let out = tail_truncate_with_spill(&long);
        assert!(out.ends_with("THE_END"));
        assert!(out.contains("truncated"));
    }
}
