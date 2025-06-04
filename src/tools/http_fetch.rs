use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

/// HTTP Fetch tool for making HTTP requests
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
    content_type: Option<String>,
    content_length: Option<usize>,
}

#[async_trait]
impl Tool for HttpFetchTool {
    fn name(&self) -> &str {
        "http_fetch"
    }
    
    fn description(&self) -> &str {
        "Make HTTP requests to fetch data from URLs. Supports GET, POST, PUT, DELETE methods with custom headers and body."
    }
    
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be http:// or https://)"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"],
                    "description": "HTTP method to use (default: GET)"
                },
                "headers": {
                    "type": "object",
                    "description": "Optional headers as key-value pairs",
                    "additionalProperties": {
                        "type": "string"
                    }
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
            .map_err(|e| Error::Other(format!(
                "Invalid input parameters: {}. Example: {{\"url\": \"https://api.example.com/data\", \"method\": \"GET\"}}", e
            )))?;
        
        // Validate URL
        if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
            return Err(Error::Other(
                "URL must start with http:// or https://. Example: {\"url\": \"https://api.example.com/data\"}".to_string()
            ));
        }
        
        // Validate URL format
        let url = reqwest::Url::parse(&params.url)
            .map_err(|e| Error::Other(format!(
                "Invalid URL: {}. Example: {{\"url\": \"https://api.example.com/data\"}}", e
            )))?;
        
        // Security: Block local addresses
        if let Some(host) = url.host_str() {
            if host == "localhost" || host == "127.0.0.1" || host.starts_with("192.168.") 
                || host.starts_with("10.") || host.starts_with("172.") {
                return Err(Error::Other(
                    "Access to local addresses is not allowed. Use external URLs like https://api.example.com".to_string()
                ));
            }
        }
        
        // Determine timeout (max 5 minutes)
        let timeout = params.timeout_seconds
            .map(|s| Duration::from_secs(s.min(300)))
            .unwrap_or(Duration::from_secs(30));
        
        // Build HTTP client
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .user_agent("Claude-RS-Bot/1.0")
            .build()
            .map_err(|e| Error::Other(format!("Failed to create HTTP client: {}", e)))?;
        
        // Determine method
        let method = params.method.as_deref().unwrap_or("GET").to_uppercase();
        
        // Build request
        let mut request = match method.as_str() {
            "GET" => client.get(&params.url),
            "POST" => client.post(&params.url),
            "PUT" => client.put(&params.url),
            "DELETE" => client.delete(&params.url),
            "HEAD" => client.head(&params.url),
            "PATCH" => client.patch(&params.url),
            _ => return Err(Error::Other(format!(
                "Unsupported HTTP method: {}. Supported methods: GET, POST, PUT, DELETE, HEAD, PATCH", method
            ))),
        };
        
        // Add headers
        if let Some(headers) = params.headers {
            for (key, value) in headers {
                // Skip potentially dangerous headers
                let key_lower = key.to_lowercase();
                if key_lower == "host" || key_lower == "content-length" {
                    continue;
                }
                request = request.header(&key, &value);
            }
        }
        
        // Add body for appropriate methods
        if let Some(body) = params.body {
            if matches!(method.as_str(), "POST" | "PUT" | "PATCH") {
                request = request.body(body);
            }
        }
        
        // Execute request
        let response = request.send().await
            .map_err(|e| Error::Other(format!("Request failed: {}", e)))?;
        
        // Extract response details
        let status = response.status().as_u16();
        let content_type = response.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        
        // Convert headers to HashMap
        let mut headers = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(v) = value.to_str() {
                headers.insert(key.to_string(), v.to_string());
            }
        }
        
        // Read body with size limit (10MB)
        let body_bytes = response.bytes().await
            .map_err(|e| Error::Other(format!("Failed to read response body: {}", e)))?;
        
        if body_bytes.len() > 10 * 1024 * 1024 {
            return Err(Error::Other("Response body too large (>10MB)".to_string()));
        }
        
        let body = String::from_utf8_lossy(&body_bytes).to_string();
        let content_length = body_bytes.len();
        
        // Create response
        let fetch_response = HttpFetchResponse {
            status,
            headers,
            body,
            content_type,
            content_length: Some(content_length),
        };
        
        // Return formatted response
        serde_json::to_string_pretty(&fetch_response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }
}