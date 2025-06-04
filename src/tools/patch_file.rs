use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

pub struct PatchFileTool;

#[async_trait]
impl Tool for PatchFileTool {
    fn name(&self) -> &str {
        "patch_file"
    }

    fn description(&self) -> &str {
        "Apply a diff/patch to a file on the filesystem"
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
                    "description": "The diff/patch content to apply (in unified diff format)"
                }
            },
            "required": ["path", "diff"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        // First check if we got an object at all
        if !input.is_object() {
            return Err(Error::Other(format!(
                "Expected JSON object with 'path' and 'diff' fields, but got: {}. Example: {{\"path\": \"/tmp/hello.txt\", \"diff\": \"--- a/hello.txt\\n+++ b/hello.txt\\n@@ -1 +1 @@\\n-Hello\\n+Hello, world!\"}}",
                serde_json::to_string(&input).unwrap_or_else(|_| "invalid JSON".to_string())
            )));
        }

        let path = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            let keys: Vec<String> = input
                .as_object()
                .map(|obj| obj.keys().cloned().collect())
                .unwrap_or_default();
            Error::Other(format!(
                "Missing 'path' field. Got keys: {:?}. Full input: {}",
                keys,
                serde_json::to_string(&input).unwrap_or_else(|_| "invalid JSON".to_string())
            ))
        })?;

        let diff = input
            .get("diff")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("Missing 'diff' field".to_string()))?;

        // Create a temporary file with the diff content
        let mut temp_file = NamedTempFile::new()
            .map_err(|e| Error::Other(format!("Failed to create temp file: {}", e)))?;

        temp_file
            .write_all(diff.as_bytes())
            .map_err(|e| Error::Other(format!("Failed to write diff to temp file: {}", e)))?;

        temp_file
            .flush()
            .map_err(|e| Error::Other(format!("Failed to flush temp file: {}", e)))?;

        // Apply the patch using the patch command
        let output = Command::new("patch")
            .arg("-u") // Unified diff format
            .arg(path)
            .arg("-i")
            .arg(temp_file.path())
            .output()
            .map_err(|e| Error::Other(format!("Failed to execute patch command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Other(format!("Failed to apply patch: {}", stderr)));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(format!("Successfully patched {}: {}", path, stdout.trim()))
    }
}
