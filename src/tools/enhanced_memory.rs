use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Enhanced memory system with persistence, search, and tagging
pub struct EnhancedMemoryTool {
    storage: Arc<RwLock<MemoryStorage>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryEntry {
    id: String,
    content: String,
    tags: Vec<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MemoryStorage {
    entries: HashMap<String, MemoryEntry>,
    tag_index: HashMap<String, Vec<String>>, // tag -> [entry_ids]
}

impl MemoryStorage {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }
    
    fn add_entry(&mut self, entry: MemoryEntry) {
        // Update tag index
        for tag in &entry.tags {
            self.tag_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(entry.id.clone());
        }
        
        // Store entry
        self.entries.insert(entry.id.clone(), entry);
    }
    
    fn update_entry(&mut self, id: &str, content: Option<String>, tags: Option<Vec<String>>, metadata: Option<HashMap<String, String>>) -> Result<()> {
        let entry = self.entries.get_mut(id)
            .ok_or_else(|| Error::Other(format!(
                "Memory entry '{}' not found. Use 'store' to create a new entry or check available entries with 'search'", id
            )))?;
        
        // Update content if provided
        if let Some(new_content) = content {
            entry.content = new_content;
        }
        
        // Update tags if provided
        if let Some(new_tags) = tags {
            // Remove old tag associations
            for tag in &entry.tags {
                if let Some(ids) = self.tag_index.get_mut(tag) {
                    ids.retain(|entry_id| entry_id != id);
                }
            }
            
            // Add new tag associations
            for tag in &new_tags {
                self.tag_index
                    .entry(tag.clone())
                    .or_insert_with(Vec::new)
                    .push(id.to_string());
            }
            
            entry.tags = new_tags;
        }
        
        // Update metadata if provided
        if let Some(new_metadata) = metadata {
            entry.metadata = new_metadata;
        }
        
        entry.updated_at = Utc::now();
        
        Ok(())
    }
    
    fn search(&self, query: Option<&str>, tags: Option<&[String]>, limit: Option<usize>) -> Vec<MemoryEntry> {
        let mut results: Vec<&MemoryEntry> = self.entries.values().collect();
        
        // Filter by tags if provided
        if let Some(search_tags) = tags {
            results.retain(|entry| {
                search_tags.iter().any(|tag| entry.tags.contains(tag))
            });
        }
        
        // Filter by query if provided
        if let Some(q) = query {
            let q_lower = q.to_lowercase();
            results.retain(|entry| {
                entry.content.to_lowercase().contains(&q_lower) ||
                entry.tags.iter().any(|tag| tag.to_lowercase().contains(&q_lower)) ||
                entry.metadata.values().any(|v| v.to_lowercase().contains(&q_lower))
            });
        }
        
        // Sort by updated_at (most recent first)
        results.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        
        // Apply limit
        if let Some(limit) = limit {
            results.truncate(limit);
        }
        
        results.into_iter().cloned().collect()
    }
    
    fn delete(&mut self, id: &str) -> Result<()> {
        let entry = self.entries.remove(id)
            .ok_or_else(|| Error::Other(format!(
                "Memory entry '{}' not found. Use 'search' to find available entries", id
            )))?;
        
        // Remove from tag index
        for tag in &entry.tags {
            if let Some(ids) = self.tag_index.get_mut(tag) {
                ids.retain(|entry_id| entry_id != id);
            }
        }
        
        Ok(())
    }
}

impl EnhancedMemoryTool {
    pub fn new() -> Result<Self> {
        let storage = Arc::new(RwLock::new(Self::load_storage()?));
        Ok(Self { storage })
    }
    
    fn get_storage_path() -> PathBuf {
        let home_dir = std::env::home_dir()
            .unwrap_or_else(|| PathBuf::from("."));
        home_dir.join(".claude_memory.json")
    }
    
    fn load_storage() -> Result<MemoryStorage> {
        let path = Self::get_storage_path();
        
        if path.exists() {
            let data = fs::read_to_string(&path)
                .map_err(|e| Error::Other(format!("Failed to read memory file: {}", e)))?;
            
            serde_json::from_str(&data)
                .map_err(|e| Error::Other(format!("Failed to parse memory file: {}", e)))
        } else {
            Ok(MemoryStorage::new())
        }
    }
    
    async fn save_storage(&self) -> Result<()> {
        let path = Self::get_storage_path();
        let storage = self.storage.read().await;
        
        let data = serde_json::to_string_pretty(&*storage)
            .map_err(|e| Error::Other(format!("Failed to serialize memory: {}", e)))?;
        
        fs::write(&path, data)
            .map_err(|e| Error::Other(format!("Failed to write memory file: {}", e)))?;
        
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "action")]
enum MemoryAction {
    #[serde(rename = "store")]
    Store {
        content: String,
        tags: Option<Vec<String>>,
        metadata: Option<HashMap<String, String>>,
    },
    #[serde(rename = "search")]
    Search {
        query: Option<String>,
        tags: Option<Vec<String>>,
        limit: Option<usize>,
    },
    #[serde(rename = "update")]
    Update {
        id: String,
        content: Option<String>,
        tags: Option<Vec<String>>,
        metadata: Option<HashMap<String, String>>,
    },
    #[serde(rename = "delete")]
    Delete {
        id: String,
    },
    #[serde(rename = "list_tags")]
    ListTags,
}

#[async_trait]
impl Tool for EnhancedMemoryTool {
    fn name(&self) -> &str {
        "enhanced_memory"
    }
    
    fn description(&self) -> &str {
        "Advanced memory system with persistent storage, search capabilities, and tagging. Store and retrieve information across sessions."
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "search", "update", "delete", "list_tags"],
                    "description": "The memory operation to perform"
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (for store/update actions)"
                },
                "id": {
                    "type": "string",
                    "description": "Memory entry ID (for update/delete actions)"
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Tags to associate with memory or filter by"
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata as key-value pairs",
                    "additionalProperties": {
                        "type": "string"
                    }
                },
                "query": {
                    "type": "string",
                    "description": "Search query to filter memories"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["action"],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let action: MemoryAction = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!(
                "Invalid input: {}. Example: {{\"action\": \"store\", \"content\": \"Important info\", \"tags\": [\"work\", \"project\"]}}", e
            )))?;
        
        match action {
            MemoryAction::Store { content, tags, metadata } => {
                let id = Uuid::new_v4().to_string();
                let entry = MemoryEntry {
                    id: id.clone(),
                    content,
                    tags: tags.unwrap_or_default(),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    metadata: metadata.unwrap_or_default(),
                };
                
                let mut storage = self.storage.write().await;
                storage.add_entry(entry);
                drop(storage);
                
                self.save_storage().await?;
                
                Ok(json!({
                    "success": true,
                    "id": id,
                    "message": "Memory stored successfully"
                }).to_string())
            }
            
            MemoryAction::Search { query, tags, limit } => {
                let storage = self.storage.read().await;
                let results = storage.search(
                    query.as_deref(),
                    tags.as_deref(),
                    limit.or(Some(10))
                );
                
                Ok(json!({
                    "success": true,
                    "count": results.len(),
                    "results": results
                }).to_string())
            }
            
            MemoryAction::Update { id, content, tags, metadata } => {
                let mut storage = self.storage.write().await;
                storage.update_entry(&id, content, tags, metadata)?;
                drop(storage);
                
                self.save_storage().await?;
                
                Ok(json!({
                    "success": true,
                    "message": format!("Memory entry '{}' updated", id)
                }).to_string())
            }
            
            MemoryAction::Delete { id } => {
                let mut storage = self.storage.write().await;
                storage.delete(&id)?;
                drop(storage);
                
                self.save_storage().await?;
                
                Ok(json!({
                    "success": true,
                    "message": format!("Memory entry '{}' deleted", id)
                }).to_string())
            }
            
            MemoryAction::ListTags => {
                let storage = self.storage.read().await;
                let mut tags: Vec<(String, usize)> = storage.tag_index
                    .iter()
                    .map(|(tag, ids)| (tag.clone(), ids.len()))
                    .collect();
                
                // Sort by count (descending)
                tags.sort_by(|a, b| b.1.cmp(&a.1));
                
                Ok(json!({
                    "success": true,
                    "tags": tags.into_iter().map(|(tag, count)| {
                        json!({
                            "tag": tag,
                            "count": count
                        })
                    }).collect::<Vec<_>>()
                }).to_string())
            }
        }
    }
}
