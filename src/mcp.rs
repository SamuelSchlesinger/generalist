//! Minimal MCP (Model Context Protocol) client.
//!
//! Discovered MCP tools are wrapped as regular [`Tool`]s and registered for
//! progressive disclosure. Like every tool in code mode, they are callable
//! only from scripts; unlike ordinary tools, their heavy schemas are omitted
//! from the model-facing `python` description and live only in the generated
//! `tools` module's docstrings. This is the progressive-disclosure pattern
//! from Anthropic's code-execution-with-MCP: context cost scales with what a
//! script uses, not with what a server offers.
//!
//! Transports:
//! - **Streamable HTTP** — JSON-RPC POSTs; handles both `application/json`
//!   and `text/event-stream` responses, `Mcp-Session-Id`, and the
//!   `MCP-Protocol-Version` header.
//! - **stdio** — newline-delimited JSON-RPC over a child process.

use crate::error::{Error, Result};
use crate::tool::Tool;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::time::{timeout, Duration};

const PROTOCOL_VERSION: &str = "2025-06-18";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// `~/.generalist/mcp.json`:
///
/// ```json
/// {
///   "servers": {
///     "tickerfacts": { "url": "https://tickerfacts.com/mcp" },
///     "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"] }
///   }
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, McpServerConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum McpServerConfig {
    Http {
        url: String,
    },
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
}

impl McpConfig {
    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }
}

enum Transport {
    Http {
        client: reqwest::Client,
        url: String,
        session_id: Option<String>,
        protocol_version: String,
    },
    Stdio {
        // Held to keep the process alive; killed on drop.
        _child: Child,
        stdin: ChildStdin,
        reader: BufReader<ChildStdout>,
    },
}

/// One connected MCP server, shared by all of its wrapped tools.
pub struct McpServer {
    pub name: String,
    transport: tokio::sync::Mutex<Transport>,
    next_id: AtomicI64,
}

/// Parse an SSE body: return the JSON-RPC message whose `id` matches.
fn parse_sse_response(body: &str, id: i64) -> Option<Value> {
    for line in body.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            if let Ok(value) = serde_json::from_str::<Value>(data.trim()) {
                if value.get("id").and_then(|v| v.as_i64()) == Some(id) {
                    return Some(value);
                }
            }
        }
    }
    None
}

/// Turn `server` + `tool` into a bridge-safe name: a valid Python identifier
/// that is unique per server.
pub fn exposed_tool_name(server: &str, tool: &str) -> String {
    let mut name: String = format!("{}_{}", server, tool)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if name.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        name.insert(0, '_');
    }
    name
}

impl McpServer {
    /// Connect and run the initialize handshake.
    pub async fn connect(name: &str, config: &McpServerConfig) -> Result<Arc<Self>> {
        let transport = match config {
            McpServerConfig::Http { url } => Transport::Http {
                client: reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT)
                    .build()?,
                url: url.clone(),
                session_id: None,
                protocol_version: PROTOCOL_VERSION.to_string(),
            },
            McpServerConfig::Stdio { command, args, env } => {
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(args)
                    .envs(env)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .kill_on_drop(true);
                let mut child = cmd.spawn().map_err(|e| {
                    Error::Other(format!("Failed to start MCP server '{}': {}", name, e))
                })?;
                let stdin = child.stdin.take().expect("stdin piped");
                let reader = BufReader::new(child.stdout.take().expect("stdout piped"));
                Transport::Stdio {
                    _child: child,
                    stdin,
                    reader,
                }
            }
        };

        let server = Arc::new(Self {
            name: name.to_string(),
            transport: tokio::sync::Mutex::new(transport),
            next_id: AtomicI64::new(1),
        });

        let init = server
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "generalist", "version": env!("CARGO_PKG_VERSION")},
                }),
            )
            .await?;
        // Adopt whatever protocol version the server settled on.
        if let Some(version) = init.get("protocolVersion").and_then(|v| v.as_str()) {
            if let Transport::Http {
                protocol_version, ..
            } = &mut *server.transport.lock().await
            {
                *protocol_version = version.to_string();
            }
        }
        server.notify("notifications/initialized").await?;
        Ok(server)
    }

    /// Send a request and await the matching response's `result`.
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let message = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let response = timeout(REQUEST_TIMEOUT, self.round_trip(message, Some(id)))
            .await
            .map_err(|_| Error::Other(format!("MCP '{}': {} timed out", self.name, method)))??
            .ok_or_else(|| {
                Error::Other(format!("MCP '{}': no response to {}", self.name, method))
            })?;

        if let Some(error) = response.get("error") {
            let message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(Error::Other(format!("MCP '{}': {}", self.name, message)));
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn notify(&self, method: &str) -> Result<()> {
        let message = json!({"jsonrpc": "2.0", "method": method});
        let _ = timeout(REQUEST_TIMEOUT, self.round_trip(message, None))
            .await
            .map_err(|_| Error::Other(format!("MCP '{}': notify timed out", self.name)))??;
        Ok(())
    }

    async fn round_trip(&self, message: Value, id: Option<i64>) -> Result<Option<Value>> {
        let mut transport = self.transport.lock().await;
        match &mut *transport {
            Transport::Http {
                client,
                url,
                session_id,
                protocol_version,
            } => {
                let mut request = client
                    .post(url.as_str())
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json, text/event-stream")
                    .header("MCP-Protocol-Version", protocol_version.as_str())
                    .json(&message);
                if let Some(session) = session_id.as_deref() {
                    request = request.header("Mcp-Session-Id", session);
                }
                let response = request.send().await?;
                if let Some(session) = response
                    .headers()
                    .get("mcp-session-id")
                    .and_then(|v| v.to_str().ok())
                {
                    *session_id = Some(session.to_string());
                }
                let status = response.status();
                let content_type = response
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                let body = response.text().await?;
                if !status.is_success() {
                    return Err(Error::Api {
                        status: status.as_u16(),
                        message: body,
                    });
                }
                let Some(id) = id else { return Ok(None) }; // notification
                if content_type.starts_with("text/event-stream") {
                    Ok(parse_sse_response(&body, id))
                } else if body.trim().is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(serde_json::from_str(&body)?))
                }
            }
            Transport::Stdio { stdin, reader, .. } => {
                let mut line = serde_json::to_string(&message)?;
                line.push('\n');
                stdin
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| Error::Other(format!("MCP stdio write failed: {}", e)))?;
                let Some(id) = id else { return Ok(None) };
                // Read until the matching response; skip server-initiated
                // messages and notifications.
                let mut buf = String::new();
                loop {
                    buf.clear();
                    let n = reader
                        .read_line(&mut buf)
                        .await
                        .map_err(|e| Error::Other(format!("MCP stdio read failed: {}", e)))?;
                    if n == 0 {
                        return Err(Error::Other("MCP server closed its stdout".to_string()));
                    }
                    if let Ok(value) = serde_json::from_str::<Value>(&buf) {
                        if value.get("id").and_then(|v| v.as_i64()) == Some(id) {
                            return Ok(Some(value));
                        }
                    }
                }
            }
        }
    }

    pub async fn list_tools(&self) -> Result<Vec<(String, String, Value)>> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| Error::Other(format!("MCP '{}': malformed tools/list", self.name)))?;
        Ok(tools
            .iter()
            .filter_map(|tool| {
                let name = tool.get("name")?.as_str()?.to_string();
                let description = tool
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                let schema = tool
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"}));
                Some((name, description, schema))
            })
            .collect())
    }

    pub async fn call_tool(&self, tool: &str, arguments: Value) -> Result<String> {
        let result = self
            .request("tools/call", json!({"name": tool, "arguments": arguments}))
            .await?;
        let text: String = result
            .get("content")
            .and_then(|c| c.as_array())
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        if result
            .get("isError")
            .and_then(|e| e.as_bool())
            .unwrap_or(false)
        {
            return Err(Error::Tool(if text.is_empty() {
                "MCP tool failed".into()
            } else {
                text
            }));
        }
        Ok(text)
    }
}

/// An MCP tool wrapped for the registry. Code-only: reachable from scripts
/// via the `tools` module, absent from the model-facing tool list.
pub struct McpTool {
    server: Arc<McpServer>,
    tool_name: String,
    exposed_name: String,
    description: String,
    schema: Value,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.exposed_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> Value {
        self.schema.clone()
    }

    fn code_only(&self) -> bool {
        true
    }

    async fn execute(&self, input: Value) -> Result<String> {
        self.server.call_tool(&self.tool_name, input).await
    }
}

/// Connect to every configured server and register its tools. Failures are
/// reported, not fatal — one dead server shouldn't take down the agent.
pub async fn register_servers(
    registry: &mut crate::tool::ToolRegistry,
    config: &McpConfig,
) -> Vec<String> {
    let mut report = Vec::new();
    for (name, server_config) in &config.servers {
        match McpServer::connect(name, server_config).await {
            Ok(server) => match server.list_tools().await {
                Ok(tools) => {
                    let mut registered = 0;
                    for (tool_name, description, schema) in tools {
                        let tool = McpTool {
                            server: Arc::clone(&server),
                            exposed_name: exposed_tool_name(name, &tool_name),
                            description: format!("[MCP:{}] {}", name, description),
                            schema,
                            tool_name,
                        };
                        if registry.register(Arc::new(tool)).is_ok() {
                            registered += 1;
                        }
                    }
                    report.push(format!("mcp '{}': {} tool(s)", name, registered));
                }
                Err(e) => report.push(format!("mcp '{}': tools/list failed: {}", name, e)),
            },
            Err(e) => report.push(format!("mcp '{}': connection failed: {}", name, e)),
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolRegistry;

    #[test]
    fn tool_names_become_python_identifiers() {
        assert_eq!(
            exposed_tool_name("tickerfacts", "get_fundamentals"),
            "tickerfacts_get_fundamentals"
        );
        assert_eq!(
            exposed_tool_name("my-server", "read.file"),
            "my_server_read_file"
        );
        assert_eq!(exposed_tool_name("1srv", "x"), "_1srv_x");
    }

    #[test]
    fn sse_bodies_parse_to_matching_response() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\ndata: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\n\n";
        let parsed = parse_sse_response(body, 7).unwrap();
        assert_eq!(parsed["result"]["ok"], true);
        assert!(parse_sse_response(body, 99).is_none());
    }

    #[test]
    fn config_parses_both_transport_shapes() {
        let config: McpConfig = serde_json::from_str(
            r#"{"servers": {
                "web": {"url": "https://example.com/mcp"},
                "local": {"command": "npx", "args": ["-y", "some-server"]}
            }}"#,
        )
        .unwrap();
        assert!(matches!(
            config.servers["web"],
            McpServerConfig::Http { .. }
        ));
        assert!(matches!(
            config.servers["local"],
            McpServerConfig::Stdio { .. }
        ));
    }

    /// Full stack against a fake stdio MCP server: connect, handshake,
    /// discover, register, execute through the registry.
    #[tokio::test]
    async fn stdio_server_end_to_end() {
        const FAKE_SERVER: &str = r#"
import sys, json
for line in sys.stdin:
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if i is None:
        continue
    if m == "initialize":
        r = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
             "serverInfo": {"name": "fake", "version": "0"}}
    elif m == "tools/list":
        r = {"tools": [{"name": "add", "description": "Add two numbers",
             "inputSchema": {"type": "object", "properties": {"a": {"type": "number"},
             "b": {"type": "number"}}, "required": ["a", "b"]}}]}
    elif m == "tools/call":
        a = msg["params"]["arguments"]
        r = {"content": [{"type": "text", "text": str(a["a"] + a["b"])}]}
    else:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": i,
            "error": {"code": -32601, "message": "nope"}}) + "\n")
        sys.stdout.flush(); continue
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": i, "result": r}) + "\n")
    sys.stdout.flush()
"#;
        let config = McpConfig {
            servers: [(
                "fake".to_string(),
                McpServerConfig::Stdio {
                    command: "python3".to_string(),
                    args: vec!["-c".to_string(), FAKE_SERVER.to_string()],
                    env: HashMap::new(),
                },
            )]
            .into(),
        };

        let mut registry = ToolRegistry::new();
        let report = register_servers(&mut registry, &config).await;
        assert_eq!(report, vec!["mcp 'fake': 1 tool(s)".to_string()]);
        assert!(registry.has_tool("fake_add"));

        // MCP tools are code-only: hidden from the model-facing defs,
        // present in the bridge defs.
        assert!(!registry
            .get_tool_defs()
            .iter()
            .any(|d| d.name == "fake_add"));
        assert!(registry
            .get_bridge_tool_defs()
            .iter()
            .any(|d| d.name == "fake_add"));

        let result = registry
            .execute_tool("fake_add", json!({"a": 2, "b": 3}), "id1".into())
            .await;
        assert_eq!(result.outcome, crate::tool::ToolCallOutcome::Success);
        match result.block {
            crate::types::ContentBlock::ToolResult { content, .. } => assert_eq!(content, "5"),
            other => panic!("expected tool result, got {:?}", other),
        }
    }
}
