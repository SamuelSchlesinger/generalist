use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read content from a file on the filesystem"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read from"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            Error::Other(
                "Missing 'path' field. Example: {\"path\": \"/home/user/document.txt\"}"
                    .to_string(),
            )
        })?;

        use std::fs;

        fs::read_to_string(path).map_err(|e| Error::Other(format!("Failed to read file: {}", e)))
    }
}
