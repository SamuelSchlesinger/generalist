use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::io::Write;
use std::process::Stdio;
use tempfile::NamedTempFile;
use tokio::process::Command;

pub struct PatchFileTool;

#[async_trait]
impl Tool for PatchFileTool {
    fn name(&self) -> &str {
        "patch_file"
    }

    fn description(&self) -> &str {
        "Apply a unified diff to a file on the filesystem. Use this to edit existing files; the \
         diff is shown to the user for approval before it is applied."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to patch"
                },
                "diff": {
                    "type": "string",
                    "description": "The patch content in unified diff format"
                }
            },
            "required": ["path", "diff"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Tool("Missing 'path' field".to_string()))?;
        let diff = input
            .get("diff")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Tool("Missing 'diff' field".to_string()))?;

        let mut temp_file = NamedTempFile::new()
            .map_err(|e| Error::Tool(format!("Failed to create temp file: {}", e)))?;
        temp_file
            .write_all(diff.as_bytes())
            .and_then(|_| temp_file.flush())
            .map_err(|e| Error::Tool(format!("Failed to write diff: {}", e)))?;

        let output = Command::new("patch")
            .arg("-u")
            .arg(path)
            .arg("-i")
            .arg(temp_file.path())
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| Error::Tool(format!("Failed to execute patch command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(Error::Tool(format!(
                "Failed to apply patch: {} {}",
                stdout.trim(),
                stderr.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(format!("Successfully patched {}: {}", path, stdout.trim()))
    }
}
