use crate::{Tool, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct ThinkTool;

#[async_trait]
impl Tool for ThinkTool {
    fn name(&self) -> &str {
        "think"
    }
    
    fn description(&self) -> &str {
        "Think more deeply about a topic or problem, exploring different angles, implications, and considerations"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic or problem to think more deeply about"
                }
            },
            "required": ["topic"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let topic = input
            .get("topic")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::Error::Other(
                "Missing 'topic' field. Example: {\"topic\": \"the implications of this design decision\"}".to_string()
            ))?;
        
        // Return a thoughtful analysis prompt
        Ok(format!(
            "Let me think more deeply about: {}\n\n\
            I should consider:\n\
            - The core aspects and underlying principles\n\
            - Potential implications and consequences\n\
            - Alternative perspectives and approaches\n\
            - Edge cases and potential challenges\n\
            - How this connects to broader patterns\n\
            - What questions I should be asking\n\n\
            Thinking...",
            topic
        ))
    }
}