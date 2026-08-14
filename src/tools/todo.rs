use crate::error::{Error, Result};
use crate::tool::Tool;
use crate::ProfilePaths;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    pub id: String,
    pub title: String,
    pub completed: bool,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoList {
    /// Tolerate a missing/`{}` file (e.g. after a truncated or partial write)
    /// by defaulting to an empty list instead of failing to parse.
    #[serde(default)]
    pub todos: Vec<Todo>,
}

impl TodoList {
    fn new() -> Self {
        TodoList { todos: Vec::new() }
    }

    fn add(&mut self, title: String) -> String {
        let id = Uuid::new_v4().to_string();
        let todo = Todo {
            id: id.clone(),
            title,
            completed: false,
            created_at: Utc::now(),
            completed_at: None,
        };
        self.todos.push(todo);
        id
    }

    fn remove(&mut self, id: &str) -> bool {
        if let Some(pos) = self.todos.iter().position(|t| t.id == id) {
            self.todos.remove(pos);
            true
        } else {
            false
        }
    }

    fn complete(&mut self, id: &str) -> bool {
        if let Some(todo) = self.todos.iter_mut().find(|t| t.id == id) {
            todo.completed = true;
            todo.completed_at = Some(Utc::now());
            true
        } else {
            false
        }
    }

    fn uncomplete(&mut self, id: &str) -> bool {
        if let Some(todo) = self.todos.iter_mut().find(|t| t.id == id) {
            todo.completed = false;
            todo.completed_at = None;
            true
        } else {
            false
        }
    }

    fn list(&self, show_completed: bool) -> Vec<&Todo> {
        self.todos
            .iter()
            .filter(|t| show_completed || !t.completed)
            .collect()
    }

    fn clear_completed(&mut self) {
        self.todos.retain(|t| !t.completed);
    }
}

#[derive(Debug, Clone)]
pub struct TodoTool {
    todo_file_path: PathBuf,
}

impl TodoTool {
    /// Build a todo tool bound to one explicit storage file.
    pub fn new(todo_file_path: impl Into<PathBuf>) -> Self {
        Self {
            todo_file_path: todo_file_path.into(),
        }
    }

    /// Build a todo tool from an already-resolved profile snapshot.
    pub fn for_profile(profile: &ProfilePaths) -> Self {
        Self::new(profile.todo_file())
    }

    fn load_todos(&self) -> Result<TodoList> {
        if !self.todo_file_path.exists() {
            return Ok(TodoList::new());
        }

        let content = fs::read_to_string(&self.todo_file_path)
            .map_err(|e| Error::Other(format!("Failed to read todo file: {}", e)))?;

        // A corrupted or partially-written todo file should not wedge the tool:
        // the list is ephemeral UX state, so fall back to an empty list rather
        // than failing every subsequent call. (The `#[serde(default)]` on the
        // field already covers the `{}` / missing-field case; this covers
        // malformed JSON or a wrong top-level shape.)
        Ok(serde_json::from_str(&content).unwrap_or_else(|_| TodoList::new()))
    }

    fn save_todos(&self, todos: &TodoList) -> Result<()> {
        if let Some(parent) = self.todo_file_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::Other(format!("Failed to create directory: {}", e)))?;
        }

        let content = serde_json::to_string_pretty(todos)
            .map_err(|e| Error::Other(format!("Failed to serialize todos: {}", e)))?;

        fs::write(&self.todo_file_path, content)
            .map_err(|e| Error::Other(format!("Failed to write todo file: {}", e)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
enum TodoAction {
    #[serde(rename = "add")]
    Add { title: String },
    #[serde(rename = "remove")]
    Remove { id: String },
    #[serde(rename = "complete")]
    Complete { id: String },
    #[serde(rename = "uncomplete")]
    Uncomplete { id: String },
    #[serde(rename = "list")]
    List { show_completed: Option<bool> },
    #[serde(rename = "clear_completed")]
    ClearCompleted,
}

#[async_trait]
impl Tool for TodoTool {
    fn name(&self) -> &'static str {
        "todo"
    }

    fn description(&self) -> &'static str {
        "Manage a simple sequential todo list. Actions: add, remove, complete, uncomplete, list, clear_completed"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "remove", "complete", "uncomplete", "list", "clear_completed"],
                    "description": "The action to perform on the todo list"
                },
                "title": {
                    "type": "string",
                    "description": "Title of the todo item (required for 'add' action)"
                },
                "id": {
                    "type": "string",
                    "description": "ID of the todo item (required for 'remove', 'complete', 'uncomplete' actions)"
                },
                "show_completed": {
                    "type": "boolean",
                    "description": "Whether to show completed items (optional for 'list' action, default: false)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let action: TodoAction = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!("Invalid parameters: {}", e)))?;

        let mut todos = self.load_todos()?;

        match action {
            TodoAction::Add { title } => {
                let id = todos.add(title.clone());
                self.save_todos(&todos)?;
                Ok(format!("Added todo '{}' with id: {}", title, id))
            }
            TodoAction::Remove { id } => {
                if todos.remove(&id) {
                    self.save_todos(&todos)?;
                    Ok(format!("Removed todo with id: {}", id))
                } else {
                    Err(Error::Other(format!("Todo with id {} not found", id)))
                }
            }
            TodoAction::Complete { id } => {
                if todos.complete(&id) {
                    self.save_todos(&todos)?;
                    Ok(format!("Marked todo {} as complete", id))
                } else {
                    Err(Error::Other(format!("Todo with id {} not found", id)))
                }
            }
            TodoAction::Uncomplete { id } => {
                if todos.uncomplete(&id) {
                    self.save_todos(&todos)?;
                    Ok(format!("Marked todo {} as incomplete", id))
                } else {
                    Err(Error::Other(format!("Todo with id {} not found", id)))
                }
            }
            TodoAction::List { show_completed } => {
                let show_completed = show_completed.unwrap_or(false);
                let items = todos.list(show_completed);

                if items.is_empty() {
                    Ok("No todos found".to_string())
                } else {
                    let mut output = String::new();
                    for todo in items {
                        let status = if todo.completed { "✓" } else { "○" };
                        let short_id = if todo.id.len() >= 8 {
                            &todo.id[0..8]
                        } else {
                            &todo.id
                        };
                        output.push_str(&format!("{} [{}] {}\n", status, short_id, todo.title));
                    }
                    Ok(output.trim_end().to_string())
                }
            }
            TodoAction::ClearCompleted => {
                let before_count = todos.todos.len();
                todos.clear_completed();
                let removed_count = before_count - todos.todos.len();
                self.save_todos(&todos)?;
                Ok(format!("Cleared {} completed todo(s)", removed_count))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn configured_todo_tool_writes_under_supplied_profile() {
        let home = tempfile::tempdir().unwrap();
        let profile = ProfilePaths::new(home.path());
        let tool = TodoTool::for_profile(&profile);

        tool.execute(serde_json::json!({"action": "add", "title": "profile-local"}))
            .await
            .unwrap();

        let saved: TodoList =
            serde_json::from_str(&fs::read_to_string(profile.todo_file()).unwrap()).unwrap();
        assert_eq!(saved.todos.len(), 1);
        assert_eq!(saved.todos[0].title, "profile-local");
    }
}
