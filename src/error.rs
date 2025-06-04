use std::error::Error as StdError;
use std::fmt;

/// Custom error type for the Claude API client
///
/// Represents various errors that can occur when interacting with the Claude API
/// or during tool execution.
///
/// # Example
///
/// ```rust
/// use claude::Error;
///
/// fn handle_api_error(error: Error) {
///     match error {
///         Error::Request(e) => eprintln!("Network error: {}", e),
///         Error::Response(msg, status) => {
///             eprintln!("API error: {} (status: {:?})", msg, status)
///         },
///         Error::Parse(e) => eprintln!("Failed to parse response: {}", e),
///         Error::Header(msg) => eprintln!("Header error: {}", msg),
///         Error::Other(msg) => eprintln!("Error: {}", msg),
///     }
/// }
/// ```
#[derive(Debug)]
pub enum Error {
    /// HTTP request error
    Request(reqwest::Error),
    /// API response error with message and optional status code
    Response(String, Option<u16>),
    /// JSON parsing error
    Parse(serde_json::Error),
    /// Header configuration error
    Header(String),
    /// Other errors
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Request(e) => write!(f, "Request error: {}", e),
            Error::Response(msg, status) => match status {
                Some(code) => write!(f, "API error (status {}): {}", code, msg),
                None => write!(f, "API error: {}", msg),
            },
            Error::Parse(e) => write!(f, "Parse error: {}", e),
            Error::Header(msg) => write!(f, "Header error: {}", msg),
            Error::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Request(e) => Some(e),
            Error::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Request(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Parse(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;