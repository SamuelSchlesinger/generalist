use crate::{Tool, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct StillThinkingTool;

#[async_trait]
impl Tool for StillThinkingTool {
    fn name(&self) -> &str {
        "still_thinking"
    }
    
    fn description(&self) -> &str {
        "Generates deeper thinking prompts based on the conversation context to help explore problems more thoroughly"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The current context or problem being discussed"
                },
                "thinking_style": {
                    "type": "string",
                    "enum": ["analytical", "creative", "systematic", "critical", "exploratory"],
                    "description": "The style of thinking to apply (default: analytical)"
                },
                "depth": {
                    "type": "integer",
                    "description": "How many layers of thinking prompts to generate (1-5, default: 3)",
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["context"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let context = input
            .get("context")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::Error::Other(
                "Missing 'context' field. Example: {\"context\": \"implementing a new feature for user authentication\"}".to_string()
            ))?;
            
        let thinking_style = input
            .get("thinking_style")
            .and_then(|v| v.as_str())
            .unwrap_or("analytical");
            
        let depth = input
            .get("depth")
            .and_then(|v| v.as_i64())
            .unwrap_or(3)
            .min(5)
            .max(1) as usize;
        
        let prompts = generate_thinking_prompts(context, thinking_style, depth);
        
        Ok(format!(
            "Generated {} thinking prompts for '{}' using {} approach:\n\n{}",
            prompts.len(),
            context,
            thinking_style,
            prompts.join("\n\n")
        ))
    }
}

fn generate_thinking_prompts(context: &str, style: &str, depth: usize) -> Vec<String> {
    let mut prompts = Vec::new();
    
    match style {
        "analytical" => {
            prompts.push(format!("What are the key components and relationships in '{}'?", context));
            if depth > 1 {
                prompts.push(format!("What assumptions am I making about '{}'? Are they valid?", context));
            }
            if depth > 2 {
                prompts.push(format!("What are the potential edge cases or failure modes for '{}'?", context));
            }
            if depth > 3 {
                prompts.push("How does this relate to similar problems I've seen before?".to_string());
            }
            if depth > 4 {
                prompts.push("What would be the consequences of different approaches?".to_string());
            }
        }
        "creative" => {
            prompts.push(format!("What unconventional approaches could work for '{}'?", context));
            if depth > 1 {
                prompts.push(format!("If I had no constraints, how would I approach '{}'?", context));
            }
            if depth > 2 {
                prompts.push("What analogies from other domains might apply here?".to_string());
            }
            if depth > 3 {
                prompts.push("How might different stakeholders view this problem differently?".to_string());
            }
            if depth > 4 {
                prompts.push("What would the opposite approach look like?".to_string());
            }
        }
        "systematic" => {
            prompts.push(format!("What are all the steps needed to address '{}'?", context));
            if depth > 1 {
                prompts.push("What dependencies exist between different components?".to_string());
            }
            if depth > 2 {
                prompts.push("What is the optimal order of operations?".to_string());
            }
            if depth > 3 {
                prompts.push("How can I verify each step is working correctly?".to_string());
            }
            if depth > 4 {
                prompts.push("What fallback plans should be in place?".to_string());
            }
        }
        "critical" => {
            prompts.push(format!("What could go wrong with the current approach to '{}'?", context));
            if depth > 1 {
                prompts.push("What evidence supports or contradicts my current understanding?".to_string());
            }
            if depth > 2 {
                prompts.push("What biases might be influencing my thinking?".to_string());
            }
            if depth > 3 {
                prompts.push("What alternative explanations haven't I considered?".to_string());
            }
            if depth > 4 {
                prompts.push("How would I know if my solution is actually working?".to_string());
            }
        }
        _ => { // exploratory or default
            prompts.push(format!("What don't I know yet about '{}'?", context));
            if depth > 1 {
                prompts.push("What questions should I be asking but haven't?".to_string());
            }
            if depth > 2 {
                prompts.push("What patterns or connections am I noticing?".to_string());
            }
            if depth > 3 {
                prompts.push("What would happen if I approached this from a completely different angle?".to_string());
            }
            if depth > 4 {
                prompts.push("What insights emerge when I step back and look at the bigger picture?".to_string());
            }
        }
    }
    
    prompts
}