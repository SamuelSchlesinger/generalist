use std::error::Error as StdError;
use std::fmt;

/// Error type shared across providers, tools, and the agent loop.
#[derive(Debug)]
pub enum Error {
    /// Transport-level failure (connection, timeout, TLS, ...).
    Network(reqwest::Error),
    /// The provider API returned a non-success status.
    Api {
        status: u16,
        message: String,
        /// Server-supplied `Retry-After` hint, in seconds, when present.
        retry_after: Option<u64>,
    },
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

    /// Server-suggested wait before retrying, if the response carried one.
    pub fn retry_after(&self) -> Option<u64> {
        match self {
            Error::Api { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Parse a `Retry-After` header value (delay-seconds form only).
    ///
    /// The HTTP-date form is deliberately ignored: the seconds form is what
    /// every major provider actually sends.
    pub(crate) fn parse_retry_after(value: &str) -> Option<u64> {
        value.trim().parse::<u64>().ok()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Network(e) => write!(f, "Network error: {}", e),
            Error::Api {
                status, message, ..
            } => {
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
                    message: String::new(),
                    retry_after: None
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
                    message: String::new(),
                    retry_after: None
                }
                .is_retryable(),
                "{} should not be retryable",
                status
            );
        }
        assert!(!Error::Other("nope".into()).is_retryable());
    }

    #[test]
    fn retry_after_is_carried_and_parsed() {
        let err = Error::Api {
            status: 429,
            message: "rate limited".into(),
            retry_after: Error::parse_retry_after(" 17 "),
        };
        assert_eq!(err.retry_after(), Some(17));
        assert_eq!(Error::parse_retry_after("soon"), None);
        assert_eq!(Error::parse_retry_after(""), None);
        assert_eq!(Error::Other("x".into()).retry_after(), None);
    }
}
