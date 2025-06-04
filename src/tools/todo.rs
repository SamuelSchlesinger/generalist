use crate::error::{Error, Result};
use crate::tool::Tool;
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

pub struct TodoTool;

impl TodoTool {
    fn get_todo_file_path() -> PathBuf {
        let mut path = PathBuf::from(".");
        path.push("todos.json");
        path
    }

    fn load_todos() -> Result<TodoList> {
        let path = Self::get_todo_file_path();
        if !path.exists() {
            return Ok(TodoList::new());
        }

        let content = fs::read_to_string(&path)
            .map_err(|e| Error::Other(format!("Failed to read todo file: {}", e)))?;

        serde_json::from_str(&content)
            .map_err(|e| Error::Other(format!("Failed to parse todo file: {}", e)))
    }

    fn save_todos(todos: &TodoList) -> Result<()> {
        let path = Self::get_todo_file_path();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::Other(format!("Failed to create directory: {}", e)))?;
        }

        let content = serde_json::to_string_pretty(todos)
            .map_err(|e| Error::Other(format!("Failed to serialize todos: {}", e)))?;

        fs::write(&path, content)
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

        let mut todos = Self::load_todos()?;

        match action {
            TodoAction::Add { title } => {
                let id = todos.add(title.clone());
                Self::save_todos(&todos)?;
                Ok(format!("Added todo '{}' with id: {}", title, id))
            }
            TodoAction::Remove { id } => {
                if todos.remove(&id) {
                    Self::save_todos(&todos)?;
                    Ok(format!("Removed todo with id: {}", id))
                } else {
                    Err(Error::Other(format!("Todo with id {} not found", id)))
                }
            }
            TodoAction::Complete { id } => {
                if todos.complete(&id) {
                    Self::save_todos(&todos)?;
                    Ok(format!("Marked todo {} as complete", id))
                } else {
                    Err(Error::Other(format!("Todo with id {} not found", id)))
                }
            }
            TodoAction::Uncomplete { id } => {
                if todos.uncomplete(&id) {
                    Self::save_todos(&todos)?;
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
                        output.push_str(&format!(
                            "{} [{}] {}\n",
                            status,
                            &todo.id[0..8],
                            todo.title
                        ));
                    }
                    Ok(output.trim_end().to_string())
                }
            }
            TodoAction::ClearCompleted => {
                let before_count = todos.todos.len();
                todos.clear_completed();
                let removed_count = before_count - todos.todos.len();
                Self::save_todos(&todos)?;
                Ok(format!("Cleared {} completed todo(s)", removed_count))
            }
        }
    }
}
