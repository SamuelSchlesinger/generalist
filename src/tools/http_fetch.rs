use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Duration;

/// Maximum bytes of response body to read.
const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

pub struct HttpFetchTool;

#[derive(Debug, Deserialize)]
struct HttpFetchInput {
    url: String,
    method: Option<String>,
    headers: Option<HashMap<String, String>>,
    body: Option<String>,
    timeout_seconds: Option<u64>,
}

#[derive(Debug, Serialize)]
struct HttpFetchResponse {
    status: u16,
    headers: HashMap<String, String>,
    body: String,
    truncated: bool,
    content_type: Option<String>,
}

/// Reject URLs that point at local or private infrastructure.
///
/// This is a best-effort guard, not a security boundary: a determined
/// attacker with DNS control can still race resolution (TOCTOU). Redirects
/// are re-checked hop by hop, and hostnames are resolved and checked before
/// connecting.
fn host_is_disallowed(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") || host.to_ascii_lowercase().ends_with(".localhost") {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return ip_is_disallowed(ip);
    }
    false
}

fn ip_is_disallowed(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()          // 10/8, 172.16/12, 192.168/16
                || v4.is_link_local()       // 169.254/16 (cloud metadata lives here)
                || v4.is_unspecified()
                || v4.is_broadcast()
        }
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return ip_is_disallowed(IpAddr::V4(mapped));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // unique-local fc00::/7
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link-local fe80::/10
        }
    }
}

fn check_url(url: &reqwest::Url) -> std::result::Result<(), String> {
    match url.scheme() {
        "http" | "https" => {}
        other => return Err(format!("Unsupported URL scheme '{}'", other)),
    }
    match url.host_str() {
        Some(host) if host_is_disallowed(host) => {
            Err("Access to local or private addresses is not allowed".to_string())
        }
        Some(_) => Ok(()),
        None => Err("URL has no host".to_string()),
    }
}

#[async_trait]
impl Tool for HttpFetchTool {
    fn name(&self) -> &str {
        "http_fetch"
    }

    fn description(&self) -> &str {
        "Make an HTTP request to a public URL and return status, headers, and body. Use for \
         APIs and raw data; for web pages, prefer the firecrawl tools which return clean \
         content. Bodies are capped at 10MB."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http:// or https://)"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"],
                    "description": "HTTP method (default: GET)"
                },
                "headers": {
                    "type": "object",
                    "description": "Optional headers as key-value pairs",
                    "additionalProperties": {"type": "string"}
                },
                "body": {
                    "type": "string",
                    "description": "Optional request body (for POST, PUT, PATCH)"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Request timeout in seconds (default: 30, max: 300)"
                }
            },
            "required": ["url"],
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let params: HttpFetchInput = serde_json::from_value(input)
            .map_err(|e| Error::Tool(format!("Invalid input parameters: {}", e)))?;

        let url = reqwest::Url::parse(&params.url)
            .map_err(|e| Error::Tool(format!("Invalid URL: {}", e)))?;
        check_url(&url).map_err(Error::Tool)?;

        // Hostnames can resolve to private addresses; check what DNS says
        // before connecting.
        if let Some(host) = url.host_str() {
            if host.parse::<IpAddr>().is_err() {
                let port = url.port_or_known_default().unwrap_or(443);
                let addrs = tokio::net::lookup_host((host, port))
                    .await
                    .map_err(|e| Error::Tool(format!("DNS resolution failed: {}", e)))?;
                for addr in addrs {
                    if ip_is_disallowed(addr.ip()) {
                        return Err(Error::Tool(
                            "URL resolves to a local or private address; refusing to fetch"
                                .to_string(),
                        ));
                    }
                }
            }
        }

        let timeout = Duration::from_secs(params.timeout_seconds.unwrap_or(30).min(300));

        // Validate every redirect hop against the same policy.
        let redirect_policy = reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 5 {
                return attempt.error("too many redirects");
            }
            match check_url(attempt.url()) {
                Ok(()) => attempt.follow(),
                Err(reason) => attempt.error(reason),
            }
        });

        let client = reqwest::Client::builder()
            .timeout(timeout)
            .redirect(redirect_policy)
            .user_agent("generalist-agent/0.2")
            .build()
            .map_err(|e| Error::Tool(format!("Failed to create HTTP client: {}", e)))?;

        let method = params.method.as_deref().unwrap_or("GET").to_uppercase();
        let mut request = match method.as_str() {
            "GET" => client.get(url),
            "POST" => client.post(url),
            "PUT" => client.put(url),
            "DELETE" => client.delete(url),
            "HEAD" => client.head(url),
            "PATCH" => client.patch(url),
            other => {
                return Err(Error::Tool(format!("Unsupported HTTP method: {}", other)));
            }
        };

        if let Some(headers) = params.headers {
            for (key, value) in headers {
                let key_lower = key.to_lowercase();
                if key_lower == "host" || key_lower == "content-length" {
                    continue;
                }
                request = request.header(&key, &value);
            }
        }
        if let Some(body) = params.body {
            if matches!(method.as_str(), "POST" | "PUT" | "PATCH") {
                request = request.body(body);
            }
        }

        let mut response = request
            .send()
            .await
            .map_err(|e| Error::Tool(format!("Request failed: {}", e)))?;

        let status = response.status().as_u16();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let mut headers = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(v) = value.to_str() {
                headers.insert(key.to_string(), v.to_string());
            }
        }

        // Stream the body so the cap actually bounds memory.
        let mut body_bytes: Vec<u8> = Vec::new();
        let mut truncated = false;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|e| Error::Tool(format!("Failed to read response body: {}", e)))?
        {
            if body_bytes.len() + chunk.len() > MAX_BODY_BYTES {
                body_bytes.extend_from_slice(&chunk[..MAX_BODY_BYTES - body_bytes.len()]);
                truncated = true;
                break;
            }
            body_bytes.extend_from_slice(&chunk);
        }

        let fetch_response = HttpFetchResponse {
            status,
            headers,
            body: String::from_utf8_lossy(&body_bytes).to_string(),
            truncated,
            content_type,
        };

        serde_json::to_string_pretty(&fetch_response)
            .map_err(|e| Error::Tool(format!("Failed to serialize response: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_local_and_private_hosts() {
        for url in [
            "http://localhost/x",
            "http://sub.localhost/x",
            "http://127.0.0.1/x",
            "http://10.1.2.3/x",
            "http://172.16.0.1/x",
            "http://192.168.1.1/x",
            "http://169.254.169.254/latest/meta-data/",
            "http://0.0.0.0/x",
            "http://[::1]/x",
            "http://[fc00::1]/x",
            "http://[::ffff:127.0.0.1]/x",
        ] {
            let parsed = reqwest::Url::parse(url).unwrap();
            assert!(check_url(&parsed).is_err(), "{} should be blocked", url);
        }
    }

    #[test]
    fn allows_public_hosts_and_blocks_odd_schemes() {
        // 172.32.x is public — the old prefix check wrongly blocked all of 172.
        for url in [
            "https://example.com/",
            "http://172.32.0.1/",
            "http://8.8.8.8/",
        ] {
            let parsed = reqwest::Url::parse(url).unwrap();
            assert!(check_url(&parsed).is_ok(), "{} should be allowed", url);
        }
        let ftp = reqwest::Url::parse("ftp://example.com/").unwrap();
        assert!(check_url(&ftp).is_err());
    }
}
