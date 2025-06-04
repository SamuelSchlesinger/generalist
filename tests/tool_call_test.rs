use claude::{Claude, ToolRegistry, Tool, Error, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

struct TestTool;

#[async_trait]
impl Tool for TestTool {
    fn name(&self) -> &str {
        "test_tool"
    }
    
    fn description(&self) -> &str {
        "A test tool to verify parameter passing"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                },
                "param2": {
                    "type": "number",
                    "description": "Second parameter"
                },
                "nested": {
                    "type": "object",
                    "properties": {
                        "field1": {"type": "string"},
                        "field2": {"type": "boolean"}
                    }
                }
            },
            "required": ["param1", "param2"]
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        // Log the raw input for debugging
        eprintln!("TestTool received input: {}", serde_json::to_string_pretty(&input).unwrap());
        
        // Verify we can access all parameters
        let param1 = input.get("param1")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other("Missing param1".to_string()))?;
            
        let param2 = input.get("param2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Error::Other("Missing param2".to_string()))?;
            
        let nested_info = if let Some(nested) = input.get("nested") {
            format!(", nested: {:?}", nested)
        } else {
            String::new()
        };
        
        Ok(format!("Received: param1='{}', param2={}{}", param1, param2, nested_info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claude::permissions::AlwaysAllowPermissions;
    
    #[tokio::test]
    async fn test_tool_parameter_passing() {
        // Create a tool registry
        let mut registry = ToolRegistry::with_permission_handler(
            Box::new(AlwaysAllowPermissions)
        );
        
        // Register our test tool
        registry.register(Arc::new(TestTool)).unwrap();
        
        // Test 1: Simple parameters
        let result = registry.execute_tool(
            "test_tool",
            json!({
                "param1": "hello",
                "param2": 42.5
            }),
            "test_id_1".to_string()
        ).await.unwrap();
        
        if let claude::ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("param1='hello'"));
            assert!(content.contains("param2=42.5"));
        } else {
            panic!("Expected ToolResult");
        }
        
        // Test 2: With nested object
        let result = registry.execute_tool(
            "test_tool",
            json!({
                "param1": "world",
                "param2": 123,
                "nested": {
                    "field1": "test",
                    "field2": true
                }
            }),
            "test_id_2".to_string()
        ).await.unwrap();
        
        if let claude::ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("param1='world'"));
            assert!(content.contains("param2=123"));
            assert!(content.contains("nested"));
        } else {
            panic!("Expected ToolResult");
        }
        
        // Test 3: Missing required parameter
        let result = registry.execute_tool(
            "test_tool",
            json!({
                "param1": "only_one"
            }),
            "test_id_3".to_string()
        ).await.unwrap();
        
        if let claude::ContentBlock::ToolResult { content, is_error, .. } = result {
            assert_eq!(is_error, Some(true));
            assert!(content.contains("Missing param2"));
        } else {
            panic!("Expected ToolResult with error");
        }
    }
    
    #[tokio::test]
    async fn test_tool_input_types() {
        let mut registry = ToolRegistry::with_permission_handler(
            Box::new(AlwaysAllowPermissions)
        );
        registry.register(Arc::new(TestTool)).unwrap();
        
        // Test with different number types
        let test_cases = vec![
            (json!({"param1": "test", "param2": 42}), "integer"),
            (json!({"param1": "test", "param2": 42.0}), "float"),
            (json!({"param1": "test", "param2": -10.5}), "negative"),
        ];
        
        for (input, desc) in test_cases {
            let result = registry.execute_tool(
                "test_tool",
                input,
                format!("test_{}", desc)
            ).await.unwrap();
            
            if let claude::ContentBlock::ToolResult { content, is_error, .. } = result {
                assert_eq!(is_error, None, "Failed for {}: {}", desc, content);
                assert!(content.contains("param2="), "Missing param2 in output for {}", desc);
            }
        }
    }
}