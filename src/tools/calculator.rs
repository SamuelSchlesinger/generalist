use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Performs mathematical calculations including basic operations, trigonometry, and more"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sin(45) * pi', 'sqrt(16)')"
                }
            },
            "required": ["expression"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let expression = input
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                Error::Other(
                    "Missing 'expression' field. Example: {\"expression\": \"2 + 2\"}".to_string(),
                )
            })?;

        // Use exmex crate for safe expression evaluation
        match exmex::eval_str::<f64>(expression) {
            Ok(result) => Ok(format!("{} = {}", expression, result)),
            Err(e) => Err(Error::Other(format!(
                "Failed to evaluate expression: {}",
                e
            ))),
        }
    }
}
