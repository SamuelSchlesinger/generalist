use std::error::Error as StdError;
use std::fmt;

/// Error type shared across providers, tools, and the agent loop.
#[derive(Debug)]
pub enum Error {
    /// Transport-level failure (connection, timeout, TLS, ...).
    Network(reqwest::Error),
    /// The provider API returned a non-success status.
    Api { status: u16, message: String },
    /// A response could not be parsed.
    Parse(serde_json::Error),
    /// A tool failed to execute.
    Tool(String),
    /// Anything else.
    Other(String),
}

impl Error {
    /// Whether the error is transient and worth retrying with backoff.
    ///
    /// Covers rate limits (429), server errors (5xx including Anthropic's 529
    /// "overloaded"), request timeouts, and connection failures.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Network(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            // Status 0 is the sentinel for mid-stream failures (premature
            // stream end, in-band stream errors) — transient in practice.
            Error::Api { status, .. } => matches!(*status, 0 | 408 | 429) || *status >= 500,
            _ => false,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Network(e) => write!(f, "Network error: {}", e),
            Error::Api { status, message } => {
                write!(f, "API error (status {}): {}", status, message)
            }
            Error::Parse(e) => write!(f, "Parse error: {}", e),
            Error::Tool(msg) => write!(f, "Tool error: {}", msg),
            Error::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Network(e) => Some(e),
            Error::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Network(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Parse(err)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_statuses() {
        for status in [0u16, 408, 429, 500, 502, 503, 529] {
            assert!(
                Error::Api {
                    status,
                    message: String::new()
                }
                .is_retryable(),
                "{} should be retryable",
                status
            );
        }
        for status in [400u16, 401, 403, 404, 413] {
            assert!(
                !Error::Api {
                    status,
                    message: String::new()
                }
                .is_retryable(),
                "{} should not be retryable",
                status
            );
        }
        assert!(!Error::Other("nope".into()).is_retryable());
    }
}
