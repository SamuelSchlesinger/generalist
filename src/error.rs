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
        /// Machine-readable provider error type (e.g. Anthropic's
        /// `overloaded_error`), when the body carried one. Refines
        /// retryability beyond the HTTP status alone.
        error_type: Option<String>,
    },
    /// The provider's response stream failed before completing.
    ///
    /// `retryable` distinguishes transient interruptions (premature stream
    /// end, overload) from in-band errors that will fail identically on
    /// every attempt (invalid request, context overflow).
    Stream { message: String, retryable: bool },
    /// A host-enforced completion bound was exceeded (payload bytes, block
    /// count, tool calls, wire bytes). Never retryable: the remedy is a
    /// smaller request, a different model, or a raised limit.
    Limit(String),
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
    /// "overloaded"), request timeouts, connection failures, and transient
    /// stream interruptions. A provider-supplied error type refines the
    /// verdict when the HTTP status alone is ambiguous.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Network(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            Error::Api {
                status, error_type, ..
            } => provider_error_type_is_transient(
                error_type.as_deref(),
                matches!(*status, 408 | 429) || *status >= 500,
            ),
            Error::Stream { retryable, .. } => *retryable,
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

/// Whether a provider's machine-readable error type names a transient
/// condition. `default` applies when the type is absent or unrecognized.
pub(crate) fn provider_error_type_is_transient(error_type: Option<&str>, default: bool) -> bool {
    match error_type {
        Some("overloaded_error" | "api_error" | "rate_limit_error" | "timeout_error") => true,
        Some(
            "invalid_request_error"
            | "authentication_error"
            | "permission_error"
            | "not_found_error"
            | "request_too_large",
        ) => false,
        _ => default,
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
            Error::Stream { message, .. } => write!(f, "Stream error: {}", message),
            Error::Limit(msg) => write!(f, "Host limit exceeded: {}", msg),
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

    fn api(status: u16, error_type: Option<&str>) -> Error {
        Error::Api {
            status,
            message: String::new(),
            retry_after: None,
            error_type: error_type.map(str::to_string),
        }
    }

    #[test]
    fn retryable_statuses() {
        for status in [408u16, 429, 500, 502, 503, 529] {
            assert!(
                api(status, None).is_retryable(),
                "{} should be retryable",
                status
            );
        }
        for status in [400u16, 401, 403, 404, 413] {
            assert!(
                !api(status, None).is_retryable(),
                "{} should not be retryable",
                status
            );
        }
        assert!(!Error::Other("nope".into()).is_retryable());
    }

    #[test]
    fn provider_error_type_refines_retryability() {
        // An overload reported on an otherwise non-retryable status retries.
        assert!(api(400, Some("overloaded_error")).is_retryable());
        // An invalid request reported on a 5xx-ish path does not.
        assert!(!api(500, Some("invalid_request_error")).is_retryable());
        // Unknown types fall back to the status rule.
        assert!(api(500, Some("mystery_error")).is_retryable());
        assert!(!api(400, Some("mystery_error")).is_retryable());
    }

    #[test]
    fn stream_errors_carry_their_own_retryability() {
        let transient = Error::Stream {
            message: "stream ended before completion".into(),
            retryable: true,
        };
        assert!(transient.is_retryable());
        assert_eq!(
            transient.to_string(),
            "Stream error: stream ended before completion"
        );
        let permanent = Error::Stream {
            message: "context length exceeded".into(),
            retryable: false,
        };
        assert!(!permanent.is_retryable());
        assert!(!Error::Limit("blocks".into()).is_retryable());
    }

    #[test]
    fn retry_after_is_carried_and_parsed() {
        let err = Error::Api {
            status: 429,
            message: "rate limited".into(),
            retry_after: Error::parse_retry_after(" 17 "),
            error_type: None,
        };
        assert_eq!(err.retry_after(), Some(17));
        assert_eq!(Error::parse_retry_after("soon"), None);
        assert_eq!(Error::parse_retry_after(""), None);
        assert_eq!(Error::Other("x".into()).retry_after(), None);
    }
}
