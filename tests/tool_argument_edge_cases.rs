use claude::{ToolRegistry, Tool, Error, Result, ContentBlock};
use claude::permissions::AlwaysAllowPermissions;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

/// Test tool that logs all received arguments
struct ArgumentTestTool;

#[async_trait]
impl Tool for ArgumentTestTool {
    fn name(&self) -> &str {
        "arg_test"
    }
    
    fn description(&self) -> &str {
        "Tests various argument passing scenarios"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "string_arg": {"type": "string"},
                "number_arg": {"type": "number"},
                "bool_arg": {"type": "boolean"},
                "array_arg": {"type": "array"},
                "object_arg": {"type": "object"},
                "null_arg": {"type": ["string", "null"]}
            },
            "required": []
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        // Return a JSON representation of what was received
        Ok(format!("Received: {}", serde_json::to_string(&input).unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_edge_case_arguments() {
        let mut registry = ToolRegistry::with_permission_handler(
            Box::new(AlwaysAllowPermissions)
        );
        registry.register(Arc::new(ArgumentTestTool)).unwrap();
        
        // Test 1: Empty object
        let result = registry.execute_tool(
            "arg_test",
            json!({}),
            "test_empty".to_string()
        ).await.unwrap();
        
        if let ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("{}"));
        }
        
        // Test 2: Null values
        let result = registry.execute_tool(
            "arg_test",
            json!({
                "string_arg": "test",
                "null_arg": null
            }),
            "test_null".to_string()
        ).await.unwrap();
        
        if let ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("null"));
        }
        
        // Test 3: Special characters in strings
        let result = registry.execute_tool(
            "arg_test",
            json!({
                "string_arg": "Test with \"quotes\" and 'apostrophes' and \n newlines"
            }),
            "test_special".to_string()
        ).await.unwrap();
        
        if let ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("quotes"));
            assert!(content.contains("apostrophes"));
        }
        
        // Test 4: Large numbers
        let result = registry.execute_tool(
            "arg_test",
            json!({
                "number_arg": 9007199254740992i64 // MAX_SAFE_INTEGER + 1
            }),
            "test_large_num".to_string()
        ).await.unwrap();
        
        if let ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("9007199254740992"));
        }
        
        // Test 5: Unicode strings
        let result = registry.execute_tool(
            "arg_test",
            json!({
                "string_arg": "Hello 世界 🌍 émojis"
            }),
            "test_unicode".to_string()
        ).await.unwrap();
        
        if let ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("世界"));
            assert!(content.contains("🌍"));
        }
        
        // Test 6: Deeply nested objects
        let result = registry.execute_tool(
            "arg_test",
            json!({
                "object_arg": {
                    "level1": {
                        "level2": {
                            "level3": {
                                "deep": "value"
                            }
                        }
                    }
                }
            }),
            "test_nested".to_string()
        ).await.unwrap();
        
        if let ContentBlock::ToolResult { content, .. } = result {
            assert!(content.contains("deep"));
            assert!(content.contains("value"));
        }
    }
    
    #[tokio::test]
    async fn test_tool_not_found() {
        let mut registry = ToolRegistry::new();
        
        let result = registry.execute_tool(
            "nonexistent_tool",
            json!({}),
            "test_notfound".to_string()
        ).await;
        
        match result {
            Err(Error::Other(msg)) => {
                assert!(msg.contains("not found"));
            }
            _ => panic!("Expected tool not found error"),
        }
    }
    
    #[tokio::test]
    async fn test_array_arguments() {
        let mut registry = ToolRegistry::with_permission_handler(
            Box::new(AlwaysAllowPermissions)
        );
        registry.register(Arc::new(ArgumentTestTool)).unwrap();
        
        // Test various array types
        let test_cases = vec![
            (json!({"array_arg": []}), "empty array"),
            (json!({"array_arg": [1, 2, 3]}), "number array"),
            (json!({"array_arg": ["a", "b", "c"]}), "string array"),
            (json!({"array_arg": [true, false, null]}), "mixed array"),
            (json!({"array_arg": [{"key": "value"}, {"key2": "value2"}]}), "object array"),
        ];
        
        for (input, desc) in test_cases {
            let result = registry.execute_tool(
                "arg_test",
                input.clone(),
                format!("test_{}", desc.replace(" ", "_"))
            ).await.unwrap();
            
            if let ContentBlock::ToolResult { content, .. } = result {
                let received: Value = serde_json::from_str(&content.replace("Received: ", ""))
                    .expect("Failed to parse result");
                assert_eq!(received["array_arg"], input["array_arg"], 
                    "Array mismatch for {}", desc);
            }
        }
    }
}