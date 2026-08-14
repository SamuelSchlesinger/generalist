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
use std::collections::{BTreeSet, HashMap};
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
/// Cap on one JSON-RPC message (stdio line or HTTP body). A misbehaving
/// server must not grow the agent's memory without bound.
const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
/// Recent stderr retained from a stdio server for connection diagnostics.
const STDERR_TAIL_BYTES: usize = 8 * 1024;

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
#[derive(Debug, Clone, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, McpServerConfig>,
}

#[derive(Debug, Clone, Deserialize)]
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

/// Typed result of connecting one configured MCP server and registering its
/// discovered tools. The CLI retains this instead of parsing display strings
/// when deciding which servers can be retried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpRegistrationReport {
    pub server_name: String,
    pub outcome: McpRegistrationOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpRegistrationOutcome {
    Connected {
        discovered_tools: usize,
        registered_tools: usize,
    },
    ConnectionFailed {
        error: String,
    },
    ToolListFailed {
        error: String,
    },
    RegistrationFailed {
        discovered_tools: usize,
        error: String,
    },
}

impl McpRegistrationReport {
    pub fn registered_tools(&self) -> usize {
        match &self.outcome {
            McpRegistrationOutcome::Connected {
                registered_tools, ..
            } => *registered_tools,
            _ => 0,
        }
    }

    pub fn display_line(&self) -> String {
        match &self.outcome {
            McpRegistrationOutcome::Connected {
                discovered_tools,
                registered_tools,
            } if discovered_tools == registered_tools => {
                format!("mcp '{}': {} tool(s)", self.server_name, registered_tools)
            }
            McpRegistrationOutcome::Connected {
                discovered_tools,
                registered_tools,
            } => format!(
                "mcp '{}': {}/{} tool(s) registered; conflicting names were skipped",
                self.server_name, registered_tools, discovered_tools
            ),
            McpRegistrationOutcome::ConnectionFailed { error } => {
                format!("mcp '{}': connection failed: {}", self.server_name, error)
            }
            McpRegistrationOutcome::ToolListFailed { error } => {
                format!("mcp '{}': tools/list failed: {}", self.server_name, error)
            }
            McpRegistrationOutcome::RegistrationFailed {
                discovered_tools,
                error,
            } => format!(
                "mcp '{}': none of {} discovered tool(s) registered: {}",
                self.server_name, discovered_tools, error
            ),
        }
    }
}

impl McpConfig {
    /// Load a configuration while distinguishing absence from malformed or
    /// unreadable input. The interactive CLI surfaces these errors instead of
    /// silently behaving as though no servers were configured.
    pub fn load_checked(path: &Path) -> Result<Option<Self>> {
        let data = match std::fs::read_to_string(path) {
            Ok(data) => data,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(Error::Other(format!(
                    "Failed to read MCP configuration {}: {error}",
                    path.display()
                )))
            }
        };
        serde_json::from_str(&data).map(Some).map_err(|error| {
            Error::Other(format!(
                "Failed to parse MCP configuration {}: {error}",
                path.display()
            ))
        })
    }

    /// Compatibility helper for callers that intentionally treat invalid and
    /// absent configuration alike.
    pub fn load(path: &Path) -> Option<Self> {
        Self::load_checked(path).ok().flatten()
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
        /// Rolling tail of the server's stderr, for diagnostics when it dies.
        stderr_tail: Arc<std::sync::Mutex<Vec<u8>>>,
    },
}

/// Read one newline-terminated message with a hard byte cap.
///
/// tokio's `read_line` grows its buffer without bound; a server emitting one
/// enormous line must produce an error, not an allocation storm.
async fn read_line_bounded(
    reader: &mut (impl tokio::io::AsyncBufRead + Unpin),
    buf: &mut Vec<u8>,
    cap: usize,
) -> std::io::Result<usize> {
    let mut total = 0usize;
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(total); // EOF
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map(|index| index + 1).unwrap_or(available.len());
        if total + take > cap {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("message exceeded {cap} bytes"),
            ));
        }
        buf.extend_from_slice(&available[..take]);
        reader.consume(take);
        total += take;
        if newline.is_some() {
            return Ok(total);
        }
    }
}

/// Read an HTTP body with a hard byte cap, decoding lossily.
async fn read_body_bounded(mut response: reqwest::Response, cap: usize) -> Result<String> {
    if response
        .content_length()
        .is_some_and(|length| length > cap as u64)
    {
        return Err(Error::Other(format!(
            "MCP HTTP response exceeded {cap} bytes"
        )));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len().saturating_add(chunk.len()) > cap {
            return Err(Error::Other(format!(
                "MCP HTTP response exceeded {cap} bytes"
            )));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(String::from_utf8_lossy(&body).into_owned())
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
                    .stderr(Stdio::piped())
                    .kill_on_drop(true);
                let mut child = cmd.spawn().map_err(|e| {
                    Error::Other(format!("Failed to start MCP server '{}': {}", name, e))
                })?;
                let stdin = child.stdin.take().expect("stdin piped");
                let reader = BufReader::new(child.stdout.take().expect("stdout piped"));
                // Keep a rolling stderr tail so a crashing server can be
                // diagnosed instead of reporting a bare closed pipe.
                let mut stderr_pipe = child.stderr.take().expect("stderr piped");
                let stderr_tail: Arc<std::sync::Mutex<Vec<u8>>> = Arc::default();
                let tail_writer = Arc::clone(&stderr_tail);
                tokio::spawn(async move {
                    use tokio::io::AsyncReadExt;
                    let mut buf = [0u8; 4096];
                    loop {
                        match stderr_pipe.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                let mut tail = tail_writer
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                tail.extend_from_slice(&buf[..n]);
                                let length = tail.len();
                                if length > STDERR_TAIL_BYTES {
                                    tail.drain(..length - STDERR_TAIL_BYTES);
                                }
                            }
                        }
                    }
                });
                Transport::Stdio {
                    _child: child,
                    stdin,
                    reader,
                    stderr_tail,
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
                let body = read_body_bounded(response, MAX_MESSAGE_BYTES).await?;
                if !status.is_success() {
                    return Err(Error::Api {
                        status: status.as_u16(),
                        message: body,
                        retry_after: None,
                        error_type: None,
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
            Transport::Stdio {
                stdin,
                reader,
                stderr_tail,
                ..
            } => {
                let mut line = serde_json::to_string(&message)?;
                line.push('\n');
                stdin
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| Error::Other(format!("MCP stdio write failed: {}", e)))?;
                let Some(id) = id else { return Ok(None) };
                // Read until the matching response; skip server-initiated
                // messages and notifications.
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    let n = read_line_bounded(reader, &mut buf, MAX_MESSAGE_BYTES)
                        .await
                        .map_err(|e| Error::Other(format!("MCP stdio read failed: {}", e)))?;
                    if n == 0 {
                        let tail = stderr_tail
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        let stderr = String::from_utf8_lossy(&tail);
                        let stderr = stderr.trim();
                        return Err(Error::Other(if stderr.is_empty() {
                            "MCP server closed its stdout".to_string()
                        } else {
                            format!("MCP server closed its stdout. Recent stderr:\n{stderr}")
                        }));
                    }
                    if let Ok(value) = serde_json::from_slice::<Value>(&buf) {
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
    register_servers_with_progress(registry, config, |_, _| {}).await
}

/// Connect and register configured servers in stable name order, reporting
/// each completed server without waiting for the entire configuration.
///
/// `registered_total` counts only MCP tools successfully added so callers can
/// update a live bridge count while this future owns the mutable registry.
pub async fn register_servers_with_progress(
    registry: &mut crate::tool::ToolRegistry,
    config: &McpConfig,
    mut on_progress: impl FnMut(&str, usize),
) -> Vec<String> {
    register_servers_with_reports(registry, config, |report, registered_total| {
        on_progress(&report.display_line(), registered_total);
    })
    .await
    .iter()
    .map(McpRegistrationReport::display_line)
    .collect()
}

/// Connect every configured server and return typed per-server outcomes.
pub async fn register_servers_with_reports(
    registry: &mut crate::tool::ToolRegistry,
    config: &McpConfig,
    on_progress: impl FnMut(&McpRegistrationReport, usize),
) -> Vec<McpRegistrationReport> {
    register_ordered_servers(registry, ordered_servers(config), on_progress).await
}

/// Connect only the named configured servers, in the configuration's stable
/// lexical order. Unknown names are ignored here; the host command validates
/// selections before starting discovery.
pub async fn register_named_servers_with_reports(
    registry: &mut crate::tool::ToolRegistry,
    config: &McpConfig,
    names: &BTreeSet<String>,
    on_progress: impl FnMut(&McpRegistrationReport, usize),
) -> Vec<McpRegistrationReport> {
    let servers = ordered_servers(config)
        .into_iter()
        .filter(|(name, _)| names.contains(name.as_str()))
        .collect();
    register_ordered_servers(registry, servers, on_progress).await
}

async fn register_ordered_servers(
    registry: &mut crate::tool::ToolRegistry,
    servers: Vec<(&String, &McpServerConfig)>,
    mut on_progress: impl FnMut(&McpRegistrationReport, usize),
) -> Vec<McpRegistrationReport> {
    let mut reports = Vec::new();
    let mut registered_total = 0;
    for (name, server_config) in servers {
        let outcome = match McpServer::connect(name, server_config).await {
            Ok(server) => match server.list_tools().await {
                Ok(tools) => {
                    let discovered_tools = tools.len();
                    let mut registered_tools = 0;
                    let mut first_error = None;
                    for (tool_name, description, schema) in tools {
                        let tool = McpTool {
                            server: Arc::clone(&server),
                            exposed_name: exposed_tool_name(name, &tool_name),
                            description: format!("[MCP:{}] {}", name, description),
                            schema,
                            tool_name,
                        };
                        match registry.register(Arc::new(tool)) {
                            Ok(()) => registered_tools += 1,
                            Err(error) if first_error.is_none() => {
                                first_error = Some(error.to_string())
                            }
                            Err(_) => {}
                        }
                    }
                    if discovered_tools > 0 && registered_tools == 0 {
                        McpRegistrationOutcome::RegistrationFailed {
                            discovered_tools,
                            error: first_error.unwrap_or_else(|| {
                                "all discovered tool names were rejected".to_string()
                            }),
                        }
                    } else {
                        McpRegistrationOutcome::Connected {
                            discovered_tools,
                            registered_tools,
                        }
                    }
                }
                Err(error) => McpRegistrationOutcome::ToolListFailed {
                    error: error.to_string(),
                },
            },
            Err(error) => McpRegistrationOutcome::ConnectionFailed {
                error: error.to_string(),
            },
        };
        let report = McpRegistrationReport {
            server_name: name.clone(),
            outcome,
        };
        registered_total += report.registered_tools();
        on_progress(&report, registered_total);
        reports.push(report);
    }
    reports
}

fn ordered_servers(config: &McpConfig) -> Vec<(&String, &McpServerConfig)> {
    let mut servers = config.servers.iter().collect::<Vec<_>>();
    servers.sort_unstable_by_key(|(name, _)| *name);
    servers
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
        assert_eq!(
            ordered_servers(&config)
                .into_iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            vec!["local", "web"],
            "server discovery order must not depend on hash randomization"
        );
    }

    #[test]
    fn checked_config_load_distinguishes_missing_invalid_and_valid_files() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mcp.json");
        assert!(McpConfig::load_checked(&path).unwrap().is_none());

        std::fs::write(&path, "{not json").unwrap();
        assert!(McpConfig::load_checked(&path)
            .unwrap_err()
            .to_string()
            .contains("Failed to parse MCP configuration"));

        std::fs::write(
            &path,
            r#"{"servers":{"web":{"url":"https://example.com"}}}"#,
        )
        .unwrap();
        let config = McpConfig::load_checked(&path).unwrap().unwrap();
        assert!(config.servers.contains_key("web"));
    }

    #[test]
    fn typed_registration_reports_preserve_machine_state_and_display_detail() {
        let connected = McpRegistrationReport {
            server_name: "files".into(),
            outcome: McpRegistrationOutcome::Connected {
                discovered_tools: 3,
                registered_tools: 2,
            },
        };
        assert_eq!(connected.registered_tools(), 2);
        assert_eq!(
            connected.display_line(),
            "mcp 'files': 2/3 tool(s) registered; conflicting names were skipped"
        );

        let failed = McpRegistrationReport {
            server_name: "files".into(),
            outcome: McpRegistrationOutcome::ConnectionFailed {
                error: "offline".into(),
            },
        };
        assert_eq!(failed.registered_tools(), 0);
        assert_eq!(
            failed.display_line(),
            "mcp 'files': connection failed: offline"
        );
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
        let mut progress = Vec::new();
        let report = register_servers_with_progress(&mut registry, &config, |line, total| {
            progress.push((line.to_string(), total));
        })
        .await;
        assert_eq!(report, vec!["mcp 'fake': 1 tool(s)".to_string()]);
        assert_eq!(progress, vec![(report[0].clone(), 1)]);
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

        let mut typed_registry = ToolRegistry::new();
        let names = ["fake".to_string()].into_iter().collect();
        let reports =
            register_named_servers_with_reports(&mut typed_registry, &config, &names, |_, _| {})
                .await;
        assert_eq!(
            reports,
            vec![McpRegistrationReport {
                server_name: "fake".into(),
                outcome: McpRegistrationOutcome::Connected {
                    discovered_tools: 1,
                    registered_tools: 1,
                },
            }]
        );
        assert!(typed_registry.has_tool("fake_add"));
    }
}
