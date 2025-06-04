use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Memory {
    id: String,
    content: String,
    tags: Vec<String>,
    created_at: DateTime<Utc>,
    accessed_at: DateTime<Utc>,
    access_count: u32,
    metadata: HashMap<String, Value>,
}

pub struct MemorySaveTool;

#[async_trait]
impl Tool for MemorySaveTool {
    fn name(&self) -> &str {
        "memory_save"
    }
    
    fn description(&self) -> &str {
        "Save information to long-term memory with tags and metadata"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to categorize this memory"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata about this memory"
                }
            },
            "required": ["content", "tags"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Other(
                "Missing 'content' field. Example: {\"content\": \"Important information to remember\", \"tags\": [\"info\", \"important\"]}".to_string()
            ))?;
        
        let tags = input
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_else(Vec::new);
        
        let metadata = input
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_else(HashMap::new);
        
        let memory_dir = get_memory_dir();
        fs::create_dir_all(&memory_dir)
            .map_err(|e| Error::Other(format!("Failed to create memory directory: {}", e)))?;
        
        // Load existing memories
        let memories_file = memory_dir.join("memories.json");
        let mut memories: Vec<Memory> = if memories_file.exists() {
            let data = fs::read_to_string(&memories_file)
                .map_err(|e| Error::Other(format!("Failed to read memories: {}", e)))?;
            serde_json::from_str(&data)
                .unwrap_or_else(|_| Vec::new())
        } else {
            Vec::new()
        };
        
        // Create new memory
        let id = format!("mem_{}", uuid::Uuid::new_v4());
        let now = Utc::now();
        let memory = Memory {
            id: id.clone(),
            content: content.to_string(),
            tags,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            metadata,
        };
        
        memories.push(memory);
        
        // Save memories
        let json_data = serde_json::to_string_pretty(&memories)
            .map_err(|e| Error::Other(format!("Failed to serialize memories: {}", e)))?;
        fs::write(&memories_file, json_data)
            .map_err(|e| Error::Other(format!("Failed to write memories: {}", e)))?;
        
        Ok(format!("Memory saved with ID: {}", id))
    }
}

pub struct MemoryRecallTool;

#[async_trait]
impl Tool for MemoryRecallTool {
    fn name(&self) -> &str {
        "memory_recall"
    }
    
    fn description(&self) -> &str {
        "Recall memories by searching content, tags, or metadata"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for memory content"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by these tags (memories must have all specified tags)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return (default: 5)"
                }
            },
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let query = input.get("query").and_then(|v| v.as_str());
        let filter_tags: Vec<String> = input
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_else(Vec::new);
        let limit = input
            .get("limit")
            .and_then(|v| v.as_i64())
            .unwrap_or(5) as usize;
        
        let memory_dir = get_memory_dir();
        let memories_file = memory_dir.join("memories.json");
        
        if !memories_file.exists() {
            return Ok("No memories found.".to_string());
        }
        
        let data = fs::read_to_string(&memories_file)
            .map_err(|e| Error::Other(format!("Failed to read memories: {}", e)))?;
        let mut memories: Vec<Memory> = serde_json::from_str(&data)
            .map_err(|e| Error::Other(format!("Failed to parse memories: {}", e)))?;
        
        // Filter memories
        let mut filtered: Vec<&mut Memory> = memories.iter_mut()
            .filter(|memory| {
                // Check tags
                if !filter_tags.is_empty() {
                    if !filter_tags.iter().all(|tag| memory.tags.contains(tag)) {
                        return false;
                    }
                }
                
                // Check query
                if let Some(q) = query {
                    let q_lower = q.to_lowercase();
                    if !memory.content.to_lowercase().contains(&q_lower) &&
                       !memory.tags.iter().any(|tag| tag.to_lowercase().contains(&q_lower)) {
                        return false;
                    }
                }
                
                true
            })
            .collect();
        
        // Sort by relevance (access count and recency)
        filtered.sort_by(|a, b| {
            let a_score = a.access_count as f64 + (a.accessed_at.timestamp() as f64 / 1_000_000.0);
            let b_score = b.access_count as f64 + (b.accessed_at.timestamp() as f64 / 1_000_000.0);
            b_score.partial_cmp(&a_score).unwrap()
        });
        
        // Update access info for recalled memories
        let now = Utc::now();
        let recalled: Vec<Memory> = filtered.into_iter()
            .take(limit)
            .map(|memory| {
                memory.accessed_at = now;
                memory.access_count += 1;
                memory.clone()
            })
            .collect();
        
        // Save updated memories
        if !recalled.is_empty() {
            let json_data = serde_json::to_string_pretty(&memories)
                .map_err(|e| Error::Other(format!("Failed to serialize memories: {}", e)))?;
            fs::write(&memories_file, json_data)
                .map_err(|e| Error::Other(format!("Failed to write memories: {}", e)))?;
        }
        
        // Format results
        if recalled.is_empty() {
            Ok("No matching memories found.".to_string())
        } else {
            let mut result = format!("Found {} memories:\n\n", recalled.len());
            for (i, memory) in recalled.iter().enumerate() {
                result.push_str(&format!(
                    "{}. [{}] {}\n   Tags: {}\n   Created: {}\n   Accessed: {} times\n\n",
                    i + 1,
                    memory.id,
                    memory.content,
                    memory.tags.join(", "),
                    memory.created_at.format("%Y-%m-%d %H:%M:%S"),
                    memory.access_count
                ));
            }
            Ok(result)
        }
    }
}

pub struct MemoryDeleteTool;

#[async_trait]
impl Tool for MemoryDeleteTool {
    fn name(&self) -> &str {
        "memory_delete"
    }
    
    fn description(&self) -> &str {
        "Delete specific memories by ID"
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of memories to delete"
                }
            },
            "required": ["memory_ids"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let memory_ids: Vec<String> = input
            .get("memory_ids")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .ok_or_else(|| Error::Other(
                "Missing 'memory_ids' field. Example: {\"memory_ids\": [\"mem_123\", \"mem_456\"]}".to_string()
            ))?;
        
        let memory_dir = get_memory_dir();
        let memories_file = memory_dir.join("memories.json");
        
        if !memories_file.exists() {
            return Ok("No memories found.".to_string());
        }
        
        let data = fs::read_to_string(&memories_file)
            .map_err(|e| Error::Other(format!("Failed to read memories: {}", e)))?;
        let mut memories: Vec<Memory> = serde_json::from_str(&data)
            .map_err(|e| Error::Other(format!("Failed to parse memories: {}", e)))?;
        
        let original_count = memories.len();
        memories.retain(|memory| !memory_ids.contains(&memory.id));
        let deleted_count = original_count - memories.len();
        
        // Save updated memories
        let json_data = serde_json::to_string_pretty(&memories)
            .map_err(|e| Error::Other(format!("Failed to serialize memories: {}", e)))?;
        fs::write(&memories_file, json_data)
            .map_err(|e| Error::Other(format!("Failed to write memories: {}", e)))?;
        
        Ok(format!("Deleted {} memories", deleted_count))
    }
}

fn get_memory_dir() -> PathBuf {
    let home_dir = std::env::home_dir().expect("Unable to determine home directory");
    home_dir.join(".chatbot_memory")
}