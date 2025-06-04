use chrono::{DateTime, Utc};
use serde_json::Value;

/// Represents the execution state of a tool
///
/// Tracks the lifecycle of a tool execution from pending to completion.
///
/// # States
///
/// - `Pending`: Initial state, execution not started
/// - `Executing`: Tool is currently executing
/// - `Completed`: Execution finished successfully with result
/// - `Failed`: Execution failed with an error
/// - `Denied`: Execution was denied by permission handler
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionState {
    /// Tool execution is pending
    Pending,
    /// Tool is currently executing
    Executing,
    /// Tool execution completed successfully
    Completed {
        /// Result of the execution
        result: String,
    },
    /// Tool execution failed
    Failed {
        /// Error message
        error: String,
    },
    /// Tool execution was denied
    Denied {
        /// Reason for denial
        reason: String,
    },
}

/// Tracks the execution of a tool call
///
/// Provides detailed information about a tool execution including timing,
/// state, results, and any errors. Used by [`ToolRegistry`] to maintain
/// execution history.
///
/// # Example
///
/// ```rust
/// # use claude::ToolExecution;
/// # use chrono::Utc;
/// # let execution = ToolExecution::new(
/// #     "exec_123".to_string(),
/// #     "calculator".to_string(),
/// #     serde_json::json!({"expression": "2+2"})
/// # );
/// // Check execution state
/// if execution.is_finished() {
///     match &execution.state {
///         claude::ExecutionState::Completed { result } => println!("Success: {}", result),
///         claude::ExecutionState::Failed { error } => println!("Failed: {}", error),
///         claude::ExecutionState::Denied { reason } => println!("Denied: {}", reason),
///         _ => {}
///     }
///     
///     // Check execution time
///     if let Some(duration) = execution.duration_ms {
///         println!("Took {} ms", duration);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ToolExecution {
    /// Unique identifier for this execution
    pub id: String,
    /// Name of the tool being executed
    pub tool_name: String,
    /// Input parameters provided to the tool
    pub input: Value,
    /// Current execution state
    pub state: ExecutionState,
    /// Timestamp when execution started
    pub started_at: DateTime<Utc>,
    /// Timestamp when execution completed
    pub completed_at: Option<DateTime<Utc>>,
    /// Duration of execution in milliseconds
    pub duration_ms: Option<u64>,
}

impl ToolExecution {
    /// Create a new ToolExecution in pending state
    pub fn new(id: String, tool_name: String, input: Value) -> Self {
        Self {
            id,
            tool_name,
            input,
            state: ExecutionState::Pending,
            started_at: Utc::now(),
            completed_at: None,
            duration_ms: None,
        }
    }

    /// Mark the execution as executing
    pub fn start(&mut self) {
        self.state = ExecutionState::Executing;
        self.started_at = Utc::now();
    }

    /// Mark the execution as completed with a result
    pub fn complete(&mut self, result: Result<String, String>) {
        self.completed_at = Some(Utc::now());
        self.duration_ms =
            Some((self.completed_at.unwrap() - self.started_at).num_milliseconds() as u64);

        match result {
            Ok(output) => {
                self.state = ExecutionState::Completed { result: output };
            }
            Err(error) => {
                self.state = ExecutionState::Failed { error };
            }
        }
    }

    /// Mark the execution as denied
    pub fn deny(&mut self, reason: &str) {
        self.state = ExecutionState::Denied {
            reason: reason.to_string(),
        };
        self.completed_at = Some(Utc::now());
        self.duration_ms =
            Some((self.completed_at.unwrap() - self.started_at).num_milliseconds() as u64);
    }

    /// Check if the execution is finished (completed, failed, or denied)
    pub fn is_finished(&self) -> bool {
        matches!(
            self.state,
            ExecutionState::Completed { .. }
                | ExecutionState::Failed { .. }
                | ExecutionState::Denied { .. }
        )
    }

    /// Get the result if execution completed successfully
    pub fn result(&self) -> Option<&str> {
        match &self.state {
            ExecutionState::Completed { result } => Some(result),
            _ => None,
        }
    }

    /// Get the error if execution failed
    pub fn error(&self) -> Option<&str> {
        match &self.state {
            ExecutionState::Failed { error } => Some(error),
            ExecutionState::Denied { reason } => Some(reason),
            _ => None,
        }
    }
}
