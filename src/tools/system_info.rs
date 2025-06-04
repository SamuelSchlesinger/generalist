use crate::{Error, Result, Tool};
use async_trait::async_trait;
use chrono::Local;
use serde_json::{json, Value};

pub struct SystemInfoTool;

#[async_trait]
impl Tool for SystemInfoTool {
    fn name(&self) -> &str {
        "system_info"
    }

    fn description(&self) -> &str {
        "Gets system information like current time, date, and OS details"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "enum": ["time", "date", "datetime", "os", "all"],
                    "description": "The type of system information to retrieve"
                }
            },
            "required": ["info_type"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let info_type = input
            .get("info_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                Error::Other(
                    "Missing 'info_type' field. Example: {\"info_type\": \"datetime\"}".to_string(),
                )
            })?;

        let result = match info_type {
            "time" => format!("Current time: {}", Local::now().format("%I:%M:%S %p")),
            "date" => format!("Current date: {}", Local::now().format("%A, %B %d, %Y")),
            "datetime" => format!(
                "Current date and time: {}",
                Local::now().format("%Y-%m-%d %I:%M:%S %p")
            ),
            "os" => {
                let os = if cfg!(target_os = "macos") {
                    "macOS"
                } else if cfg!(target_os = "linux") {
                    "Linux"
                } else if cfg!(target_os = "windows") {
                    "Windows"
                } else {
                    "Unknown"
                };
                format!("Operating System: {}", os)
            }
            "all" => {
                let os = if cfg!(target_os = "macos") {
                    "macOS"
                } else if cfg!(target_os = "linux") {
                    "Linux"
                } else if cfg!(target_os = "windows") {
                    "Windows"
                } else {
                    "Unknown"
                };
                format!(
                    "System Information:\n- {}\n- Operating System: {}",
                    Local::now().format("%A, %B %d, %Y at %I:%M:%S %p"),
                    os
                )
            }
            _ => {
                return Err(Error::Other(format!(
                    "Unknown info_type: '{}'. Valid options: time, date, datetime, os, all",
                    info_type
                )))
            }
        };

        Ok(result)
    }
}
