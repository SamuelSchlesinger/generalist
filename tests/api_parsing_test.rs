use claude::{MessageResponse, ContentBlock, Message};
use serde_json::json;

#[test]
fn test_message_response_parsing() {
    // Test parsing a typical Claude response with tool use
    let response_json = json!({
        "id": "msg_123",
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "I'll help you with that calculation."
            },
            {
                "type": "tool_use",
                "name": "calculator",
                "input": {
                    "expression": "2 + 2"
                },
                "id": "tool_calc_123"
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    });
    
    // Parse the response
    let response: MessageResponse = serde_json::from_value(response_json)
        .expect("Failed to parse response");
    
    // Verify fields
    assert_eq!(response.id, "msg_123");
    assert_eq!(response.model, "claude-3-haiku-20240307");
    assert_eq!(response.role, "assistant");
    assert_eq!(response.stop_reason, "tool_use");
    assert_eq!(response.content.len(), 2);
    
    // Check text content
    match &response.content[0] {
        ContentBlock::Text { text } => {
            assert_eq!(text, "I'll help you with that calculation.");
        }
        _ => panic!("Expected text block"),
    }
    
    // Check tool use content
    match &response.content[1] {
        ContentBlock::ToolUse { name, input, id } => {
            assert_eq!(name, "calculator");
            assert_eq!(id, "tool_calc_123");
            assert_eq!(input.get("expression").and_then(|v| v.as_str()), Some("2 + 2"));
        }
        _ => panic!("Expected tool use block"),
    }
    
    // Test Message conversion
    let message: Message = (&response).into();
    assert_eq!(message.role, "assistant");
    assert_eq!(message.content.len(), 2);
}

#[test]
fn test_complex_tool_parameters() {
    // Test with more complex nested parameters
    let response_json = json!({
        "id": "msg_456",
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "name": "write_file",
                "input": {
                    "path": "/tmp/test.json",
                    "content": "{\"nested\": {\"data\": true}}",
                    "metadata": {
                        "author": "test",
                        "timestamp": 1234567890,
                        "tags": ["json", "test", "nested"]
                    }
                },
                "id": "tool_write_456"
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": null
    });
    
    let response: MessageResponse = serde_json::from_value(response_json)
        .expect("Failed to parse complex response");
    
    match &response.content[0] {
        ContentBlock::ToolUse { name, input, id } => {
            assert_eq!(name, "write_file");
            assert_eq!(id, "tool_write_456");
            
            // Verify nested parameters are preserved
            assert_eq!(input.get("path").and_then(|v| v.as_str()), Some("/tmp/test.json"));
            assert!(input.get("content").is_some());
            
            // Check metadata object
            let metadata = input.get("metadata").expect("Missing metadata");
            assert_eq!(metadata.get("author").and_then(|v| v.as_str()), Some("test"));
            assert_eq!(metadata.get("timestamp").and_then(|v| v.as_i64()), Some(1234567890));
            
            // Check array
            let tags = metadata.get("tags").and_then(|v| v.as_array()).expect("Missing tags");
            assert_eq!(tags.len(), 3);
            assert_eq!(tags[0].as_str(), Some("json"));
        }
        _ => panic!("Expected tool use block"),
    }
}

#[test]
fn test_multiple_tool_uses() {
    // Test response with multiple tool uses
    let response_json = json!({
        "id": "msg_789",
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Let me check the weather and do some calculations."
            },
            {
                "type": "tool_use",
                "name": "weather",
                "input": {
                    "city": "London",
                    "units": "celsius"
                },
                "id": "tool_weather_789"
            },
            {
                "type": "tool_use",
                "name": "calculator",
                "input": {
                    "expression": "32 * 1.8 + 32"
                },
                "id": "tool_calc_790"
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 150,
            "output_tokens": 75
        }
    });
    
    let response: MessageResponse = serde_json::from_value(response_json)
        .expect("Failed to parse multi-tool response");
    
    assert_eq!(response.content.len(), 3);
    
    // Verify get_tool_uses method on converted Message
    let message: Message = (&response).into();
    let tool_uses = message.get_tool_uses();
    assert_eq!(tool_uses.len(), 2);
    
    assert_eq!(tool_uses[0].0, "weather");
    assert_eq!(tool_uses[0].2, "tool_weather_789");
    assert_eq!(tool_uses[0].1.get("city").and_then(|v| v.as_str()), Some("London"));
    
    assert_eq!(tool_uses[1].0, "calculator");
    assert_eq!(tool_uses[1].2, "tool_calc_790");
}

#[test]
fn test_tool_result_parsing() {
    // Test that tool results can be properly created and serialized
    let tool_result = ContentBlock::ToolResult {
        content: "The result is 42".to_string(),
        tool_use_id: "tool_123".to_string(),
        is_error: None,
    };
    
    let json = serde_json::to_value(&tool_result).expect("Failed to serialize");
    assert_eq!(json["type"], "tool_result");
    assert_eq!(json["content"], "The result is 42");
    assert_eq!(json["tool_use_id"], "tool_123");
    assert!(!json.as_object().unwrap().contains_key("is_error"));
    
    // Test with error
    let error_result = ContentBlock::ToolResult {
        content: "Failed to execute".to_string(),
        tool_use_id: "tool_456".to_string(),
        is_error: Some(true),
    };
    
    let error_json = serde_json::to_value(&error_result).expect("Failed to serialize");
    assert_eq!(error_json["is_error"], true);
}

#[test]
fn test_edge_cases() {
    // Test with null/missing optional fields
    let minimal_response = json!({
        "id": "msg_minimal",
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Simple response"
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": null
    });
    
    let response: MessageResponse = serde_json::from_value(minimal_response)
        .expect("Failed to parse minimal response");
    
    assert!(response.stop_sequence.is_none());
    assert!(response.usage.is_none());
    
    // Test empty content array (should this be allowed?)
    let empty_content = json!({
        "id": "msg_empty",
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": null
    });
    
    let empty_response: MessageResponse = serde_json::from_value(empty_content)
        .expect("Failed to parse empty content response");
    
    assert_eq!(empty_response.content.len(), 0);
}