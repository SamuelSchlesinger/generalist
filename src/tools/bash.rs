use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::process::Command;

pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }
    
    fn description(&self) -> &str {
        "Execute bash commands or scripts"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command or script to execute"
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
            .ok_or_else(|| Error::Other(
                "Missing 'command' field. Example: {\"command\": \"ls -la\"}".to_string()
            ))?;
        
        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()
            .map_err(|e| Error::Other(format!("Failed to execute bash command: {}", e)))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        if output.status.success() {
            Ok(stdout.to_string())
        } else {
            Ok(format!("Exit code: {}\nStdout:\n{}\nStderr:\n{}", 
                output.status.code().unwrap_or(-1),
                stdout,
                stderr
            ))
        }
    }
}