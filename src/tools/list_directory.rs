use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct ListDirectoryTool;

#[async_trait]
impl Tool for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "List files and directories in a given path"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list"
                }
            },
            "required": ["path"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
            Error::Other(
                "Missing 'path' field. Example: {\"path\": \"/home/user/documents\"}".to_string(),
            )
        })?;

        use std::fs;

        let entries = fs::read_dir(path)
            .map_err(|e| Error::Other(format!("Failed to read directory: {}", e)))?;

        let mut results = Vec::new();
        for entry in entries {
            if let Ok(entry) = entry {
                let metadata = entry.metadata();
                let file_type = if let Ok(meta) = metadata {
                    if meta.is_dir() {
                        "[DIR]"
                    } else {
                        "[FILE]"
                    }
                } else {
                    "[?]"
                };

                if let Some(name) = entry.file_name().to_str() {
                    results.push(format!("{} {}", file_type, name));
                }
            }
        }

        Ok(results.join("\n"))
    }
}
