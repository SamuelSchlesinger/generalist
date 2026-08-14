//! End-to-end integration test: a real `Agent` talking to a fake
//! OpenAI-compatible server over localhost HTTP.
//!
//! This exercises the full transport stack the unit tests cannot: reqwest,
//! SSE assembly, streamed tool-call accumulation, the agent loop, and tool
//! dispatch — everything short of a live provider.

use generalist::provider::OpenAiProvider;
use generalist::tools::CalculatorTool;
use generalist::{Agent, AgentEvent, ToolRegistry, TurnOutcome};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Serve exactly one HTTP request with an SSE body; returns the request body.
async fn serve_one(listener: &TcpListener, sse_events: &[&str]) -> String {
    let (mut stream, _) = listener.accept().await.expect("accept");
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let (headers_end, content_length) = loop {
        let n = stream.read(&mut chunk).await.expect("read request");
        assert!(n > 0, "client closed before sending a full request");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|window| window == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buf[..pos]).to_string();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let lower = line.to_ascii_lowercase();
                    lower
                        .strip_prefix("content-length:")
                        .map(|value| value.trim().parse::<usize>().expect("content length"))
                })
                .unwrap_or(0);
            break (pos + 4, content_length);
        }
    };
    while buf.len() < headers_end + content_length {
        let n = stream.read(&mut chunk).await.expect("read body");
        assert!(n > 0, "client closed mid-body");
        buf.extend_from_slice(&chunk[..n]);
    }
    let request_body =
        String::from_utf8_lossy(&buf[headers_end..headers_end + content_length]).to_string();

    let mut body = String::new();
    for event in sse_events {
        body.push_str("data: ");
        body.push_str(event);
        body.push_str("\r\n\r\n");
    }
    body.push_str("data: [DONE]\r\n\r\n");
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream
        .write_all(response.as_bytes())
        .await
        .expect("write response");
    let _ = stream.shutdown().await;
    request_body
}

#[tokio::test]
async fn agent_completes_a_streamed_tool_round_trip_over_http() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));

    let server = async {
        // Round 1: a streamed tool call whose arguments arrive in fragments.
        let first_request = serve_one(
            &listener,
            &[
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_calc","function":{"name":"calculator","arguments":""}}]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"expression\":"}}]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"2+2\"}"}}]},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            ],
        )
        .await;
        // Round 2: the final streamed answer.
        let second_request = serve_one(
            &listener,
            &[
                r#"{"choices":[{"index":0,"delta":{"content":"The answer is "},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{"content":"4."},"finish_reason":null}]}"#,
                r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
            ],
        )
        .await;
        (first_request, second_request)
    };

    let client = async {
        let provider =
            OpenAiProvider::new("test-key".into(), base_url.clone(), "test-model".into())
                .expect("provider");
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(CalculatorTool)).expect("tool");
        let mut agent = Agent::new(Box::new(provider), registry, "You are a test agent.");
        agent.code_mode = false;

        let mut streamed_text = String::new();
        let mut tool_results = Vec::new();
        let outcome = agent
            .run_turn("What is 2+2?", &mut |event| match event {
                AgentEvent::AssistantTextDelta(delta) => streamed_text.push_str(&delta),
                AgentEvent::ToolCallFinished { name, content, .. } => {
                    tool_results.push((name, content));
                }
                _ => {}
            })
            .await
            .expect("turn");
        (agent, outcome, streamed_text, tool_results)
    };

    let ((first_request, second_request), (agent, outcome, streamed_text, tool_results)) =
        tokio::join!(server, client);

    assert_eq!(outcome, TurnOutcome::Completed);
    assert_eq!(streamed_text, "The answer is 4.");
    assert_eq!(tool_results.len(), 1);
    assert_eq!(tool_results[0].0, "calculator");
    assert!(tool_results[0].1.contains('4'), "{:?}", tool_results[0]);

    // The first request advertised the tool and streamed.
    assert!(first_request.contains("\"calculator\""));
    assert!(first_request.contains("\"stream\":true"));
    // The second request replayed the tool result under the same call id.
    assert!(second_request.contains("\"tool_call_id\":\"call_calc\""));
    assert!(second_request.contains("\"role\":\"tool\""));

    // History is a valid tool-use protocol trace of the full round trip.
    assert!(generalist::history_tool_protocol_is_valid(agent.history()));
    assert_eq!(agent.history().len(), 4);
}
