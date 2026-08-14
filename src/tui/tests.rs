use super::*;
use ratatui::backend::TestBackend;
use serde_json::json;

fn plain_text(text: &Text<'_>) -> String {
    text.lines
        .iter()
        .map(|line| {
            line.spans.iter().fold(String::new(), |mut text, span| {
                text.push_str(span.content.as_ref());
                text
            })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn highlighted_plain(lines: &[Vec<Span<'static>>]) -> String {
    lines
        .iter()
        .map(|line| {
            line.iter().fold(String::new(), |mut text, span| {
                text.push_str(span.content.as_ref());
                text
            })
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn highlighted_span<'a>(lines: &'a [Vec<Span<'static>>], content: &str) -> &'a Span<'static> {
    lines
        .iter()
        .flatten()
        .find(|span| span.content.as_ref() == content)
        .unwrap_or_else(|| panic!("missing highlighted token {content:?}"))
}

#[test]
fn editor_operations_are_unicode_safe() {
    let mut value = "a🦀c".to_string();
    let mut cursor = 2;
    insert_char(&mut value, &mut cursor, 'ß');
    assert_eq!(value, "a🦀ßc");
    backspace(&mut value, &mut cursor);
    assert_eq!(value, "a🦀c");
    backspace(&mut value, &mut cursor);
    assert_eq!(value, "ac");
    assert_eq!(cursor, 1);
}

#[test]
fn slash_completion_consumes_only_catalog_prefixes_at_the_end() {
    let mut app = AppState::new("api", "model");
    app.input = "/to".to_string();
    app.input_cursor = 3;
    assert!(app.complete_slash_command());
    assert_eq!(app.input, "/tools ");
    assert_eq!(app.input_cursor, 7);
    assert!(app.status.contains("Completed"));

    app.input = "/tools s".to_string();
    app.input_cursor = app.input.chars().count();
    assert!(app.complete_slash_command());
    assert_eq!(app.input, "/tools s");
    assert!(app.status.contains("/tools search · /tools show"));

    app.input = "/tools search archive".to_string();
    app.input_cursor = app.input.chars().count();
    assert!(!app.complete_slash_command());
    assert_eq!(app.input, "/tools search archive");

    app.input = "/goal ship 🦀 now".to_string();
    app.input_cursor = app.input.chars().count();
    assert!(!app.complete_slash_command());

    app.input = "/tools".to_string();
    app.input_cursor = 2;
    assert!(!app.complete_slash_command());
    assert_eq!(app.input, "/tools");
}

#[test]
fn command_footer_derives_stable_unique_and_ambiguous_hints() {
    assert_eq!(command_footer_hint("/to"), " Tab → /tools");
    assert_eq!(
        command_footer_hint("/tools s"),
        " /tools search  /tools show"
    );
    assert!(command_footer_hint("/tools search ").contains("prefix complete"));
    assert!(command_footer_hint("/goal ship it").contains("/permissions"));

    let mut app = AppState::new("api", "model");
    app.input = "/tools s".to_string();
    app.input_cursor = app.input.chars().count();
    let backend = TestBackend::new(100, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("/tools search"));
    assert!(rendered.contains("/tools show"));
}

#[test]
fn prompt_editor_can_replace_a_prefilled_goal_with_shell_keys() {
    let mut value = "old goal".to_string();
    let mut cursor = value.chars().count();
    assert_eq!(
        handle_prompt_editor_key(
            &mut value,
            &mut cursor,
            KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
        ),
        PromptEditorAction::Continue
    );
    assert!(value.is_empty());
    assert_eq!(cursor, 0);

    for ch in "new goal".chars() {
        handle_prompt_editor_key(
            &mut value,
            &mut cursor,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        );
    }
    assert_eq!(value, "new goal");
    assert_eq!(
        handle_prompt_editor_key(
            &mut value,
            &mut cursor,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        ),
        PromptEditorAction::Submit
    );
}

#[test]
fn bulk_paste_is_unicode_safe_and_filters_carriage_returns() {
    let mut value = "ab".to_string();
    let mut cursor = 1;
    insert_text(&mut value, &mut cursor, "\r🦀x");
    assert_eq!(value, "a🦀xb");
    assert_eq!(cursor, 3);
}

#[test]
fn untrusted_control_bytes_are_rendered_inert() {
    let sanitized = sanitize_terminal_text("red\x1b[31m\tbell\u{7}\nok");
    assert_eq!(sanitized, "red␛[31m⇥bell␇\nok");
    assert!(!sanitized.contains('\u{1b}'));
    assert!(!sanitized.chars().any(|ch| ch.is_control() && ch != '\n'));

    let (editor, _) = visible_editor("a\x1bb\n", 4, 20);
    assert_eq!(editor, "a␛b↵");

    let mut app = AppState::new("api\x1b", "model");
    app.push_chat(ChatKind::Assistant, "answer\x1b[2J");
    assert_eq!(app.api, "api␛");
    assert_eq!(app.chat[0].body, "answer␛[2J");
}

#[test]
fn terminal_cleanup_disables_every_enabled_mode() {
    let mut output = Vec::new();
    leave_terminal(&mut output);
    assert!(output.windows(8).any(|window| window == b"\x1b[?1049l"));
    assert!(output.windows(8).any(|window| window == b"\x1b[?2004l"));
    assert!(
        output.windows(8).any(|window| window == b"\x1b[?1000l")
            || output.windows(8).any(|window| window == b"\x1b[?1006l")
    );
}

#[test]
fn paused_scroll_is_anchored_and_cannot_overshoot() {
    let mut app = AppState::new("api", "model");
    app.push_chat(ChatKind::Assistant, "content");
    app.update_chat_metrics(100);
    assert_eq!(app.chat_scroll, 100);
    assert!(app.follow_latest);

    app.scroll_chat_up(10);
    assert_eq!(app.chat_scroll, 90);
    assert!(!app.follow_latest);

    // New output extends the bottom, but a paused viewport keeps the same
    // absolute top line instead of drifting with the stream.
    app.update_chat_metrics(130);
    assert_eq!(app.chat_scroll, 90);
    assert_eq!(app.lines_from_latest(), 40);

    app.scroll_chat_up(usize::MAX);
    assert_eq!(app.chat_scroll, 0);
    app.scroll_chat_down(usize::MAX);
    assert_eq!(app.chat_scroll, 130);
    assert!(app.follow_latest);
}

#[test]
fn chat_search_is_case_insensitive_and_keeps_a_stable_selected_entry() {
    let mut app = AppState::new("api", "model");
    app.push_user("alpha");
    app.push_chat(ChatKind::Assistant, "first Beta answer");
    app.push_error("second beta result");
    app.open_chat_search();

    for ch in "BETA".chars() {
        app.handle_chat_search_key(KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE));
    }
    let Some(Modal::Search(search)) = &app.modal else {
        panic!("search modal closed while entering a query");
    };
    assert_eq!(search.matches, vec![1, 2]);

    app.handle_chat_search_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
    app.push_info("later beta event");
    let Some(Modal::Search(search)) = &app.modal else {
        panic!("live chat update closed search");
    };
    assert_eq!(search.matches, vec![1, 2, 3]);
    assert_eq!(search.selected_entry(), Some(2));

    app.handle_chat_search_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    assert!(app.modal.is_none());
    assert_eq!(app.pending_chat_jump, Some(2));
    assert_eq!(app.status, "Conversation match 2 of 3");
}

#[test]
fn search_paste_is_single_line_unicode_and_rendered_jump_is_exact() {
    let mut app = AppState::new("api", "model");
    for index in 0..5 {
        app.push_info(format!(
            "earlier entry {index} with enough words to wrap across the narrow conversation"
        ));
    }
    let target_entry = app.chat.len();
    app.push_chat(ChatKind::Assistant, "needle 🦀 here");
    for index in 0..30 {
        app.push_info(format!("later entry {index} with enough words to wrap"));
    }
    app.open_chat_search();
    assert!(app.paste_chat_search("needle\r\n🦀"));
    let Some(Modal::Search(search)) = &app.modal else {
        panic!("paste closed search");
    };
    assert_eq!(search.query, "needle 🦀");
    assert_eq!(search.matches, vec![target_entry]);

    app.handle_chat_search_key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL));
    assert!(app.paste_chat_search("needle"));

    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("Find in conversation"));
    assert!(rendered.contains("needle 🦀 here"));

    app.handle_chat_search_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
    let expected_scroll = chat_entry_start(&app.chat, target_entry, 78);
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    assert_eq!(app.pending_chat_jump, None);
    assert_eq!(app.chat_scroll, expected_scroll.min(app.chat_max_scroll));
    assert!(app.chat_scroll > 0);
    assert!(app.chat_max_scroll > 0);
    assert!(!app.follow_latest);
}

#[test]
fn mouse_wheel_respects_modal_focus() {
    let mut app = AppState::new("api", "model");
    app.update_chat_metrics(100);
    app.modal = Some(Modal::Help);
    app.handle_mouse_scroll(MouseEventKind::ScrollUp);
    assert_eq!(app.chat_scroll, 100, "help leaked scroll to transcript");

    app.modal = Some(Modal::Permission {
        id: 1,
        request: ToolExecutionRequest {
            tool_use_id: "tool".into(),
            tool_name: "bash".into(),
            input: json!({"command": "true"}),
            tool_description: "run".into(),
        },
        selected: 1,
        scroll: 0,
    });
    app.handle_mouse_scroll(MouseEventKind::ScrollDown);
    assert!(matches!(
        app.modal,
        Some(Modal::Permission { scroll: 3, .. })
    ));
    assert_eq!(app.chat_scroll, 100);

    app.queue = (0..2)
        .map(|id| QueuedPrompt {
            id,
            text: id.to_string(),
            delivery: DeliveryMode::FollowUp,
            source: PromptSource::User,
        })
        .collect();
    app.modal = Some(Modal::Queue {
        selected: 0,
        editing: None,
    });
    app.handle_mouse_scroll(MouseEventKind::ScrollDown);
    assert!(matches!(app.modal, Some(Modal::Queue { selected: 1, .. })));
    assert_eq!(app.chat_scroll, 100);

    let search = ChatSearch {
        query: "content".into(),
        cursor: 7,
        matches: vec![0, 1],
        selected: 0,
    };
    app.modal = Some(Modal::Search(search));
    app.handle_mouse_scroll(MouseEventKind::ScrollDown);
    assert!(matches!(
        &app.modal,
        Some(Modal::Search(ChatSearch { selected: 1, .. }))
    ));
    assert_eq!(app.chat_scroll, 100);
}

#[test]
fn restoring_queue_text_never_overwrites_a_draft() {
    let prompt = QueuedPrompt {
        id: 7,
        text: "queued".into(),
        delivery: DeliveryMode::FollowUp,
        source: PromptSource::User,
    };
    let mut app = AppState::new("api", "model");
    app.input = "unsent draft".into();
    app.input_cursor = app.input.chars().count();

    assert!(!app.restore_to_composer(&prompt));
    assert_eq!(app.input, "unsent draft");

    app.input.clear();
    app.input_cursor = 0;
    assert!(app.restore_to_composer(&prompt));
    assert_eq!(app.input, "queued");
}

#[test]
fn automatic_goal_control_is_visible_as_info_and_completion_clears_header() {
    let mut app = AppState::new("api", "model");
    app.load_history(&[Message::user_text(crate::goal::GOAL_CONTINUATION_PROMPT)]);
    assert_eq!(app.chat[0].kind, ChatKind::User);

    app.load_history(&[Message::goal_continuation()]);
    assert_eq!(app.chat.len(), 1);
    assert_eq!(app.chat[0].kind, ChatKind::Info);
    assert!(app.chat[0].body.contains("Automatic continuation"));

    app.goal = Some("finish it".into());
    app.apply_agent_event(AgentEvent::GoalCompleted {
        goal: "finish it".into(),
    });
    assert!(app.goal.is_none());
    assert_eq!(app.chat.last().unwrap().kind, ChatKind::Info);
    assert!(app.chat.last().unwrap().body.contains("Goal completed"));
}

#[test]
fn saved_same_named_tool_results_stay_with_their_ids() {
    let history = vec![
        Message::assistant(vec![
            ContentBlock::ToolUse {
                name: "read_file".into(),
                input: json!({"path": "one"}),
                id: "one".into(),
            },
            ContentBlock::ToolUse {
                name: "read_file".into(),
                input: json!({"path": "two"}),
                id: "two".into(),
            },
        ]),
        Message::user(vec![
            ContentBlock::ToolResult {
                content: "first output".into(),
                tool_use_id: "one".into(),
                is_error: None,
            },
            ContentBlock::ToolResult {
                content: "second output".into(),
                tool_use_id: "two".into(),
                is_error: Some(true),
            },
        ]),
    ];
    let mut app = AppState::new("api", "model");
    app.load_history(&history);

    assert_eq!(app.activity[0].output, "first output");
    assert_eq!(app.activity[0].status, ActivityStatus::Success);
    assert_eq!(app.activity[1].output, "second output");
    assert_eq!(app.activity[1].status, ActivityStatus::Failed);
}

#[test]
fn saved_protocol_violation_is_labeled_rejected_not_executed() {
    let history = vec![
        Message::assistant(vec![ContentBlock::ToolUse {
            name: "tools.firecrawl_search".into(),
            input: json!({"query": "x"}),
            id: "bad-native".into(),
        }]),
        Message::user(vec![ContentBlock::ToolResult {
            content: "The provider emitted undeclared native tool call \
                      `tools.firecrawl_search`. It was not executed: code mode permits only \
                      the model-facing `python` capability tool."
                .into(),
            tool_use_id: "bad-native".into(),
            is_error: Some(true),
        }]),
    ];
    let mut app = AppState::new("api", "model");
    app.load_history(&history);

    assert_eq!(app.activity[0].status, ActivityStatus::Rejected);
}

#[test]
fn aborted_stream_is_visibly_marked_uncommitted() {
    let mut app = AppState::new("api", "model");
    app.apply_agent_event(AgentEvent::AssistantTextDelta("partial".into()));
    app.apply_agent_event(AgentEvent::StreamDisplayTruncated {
        text_bytes: 42,
        reasoning_bytes: 0,
    });
    app.apply_agent_event(AgentEvent::AssistantStreamAborted {
        reason: "interrupted before commit".into(),
    });
    app.apply_agent_event(AgentEvent::ApiCallFinished { usage: None });

    assert!(app.chat[0].body.starts_with("partial"));
    assert!(app.chat[0].body.contains("omitted 42 streamed bytes"));
    assert!(app.chat[0].body.contains("[Uncommitted stream:"));
    assert!(app.streaming_chat.is_none());
}

#[test]
fn committed_stream_replaces_a_truncated_preview_exactly() {
    let mut app = AppState::new("api", "model");
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::AssistantTextDelta("partial".into()));
    app.apply_agent_event(AgentEvent::ReasoningDelta("rough".into()));
    app.apply_agent_event(AgentEvent::StreamDisplayTruncated {
        text_bytes: 5_000,
        reasoning_bytes: 7_000,
    });
    app.apply_agent_event(AgentEvent::ApiCallFinished { usage: None });
    app.apply_agent_event(AgentEvent::StreamCommitted {
        text: Some("complete assistant response".into()),
        reasoning: Some("complete inspectable reasoning".into()),
    });

    assert_eq!(app.chat.len(), 1);
    assert_eq!(app.chat[0].body, "complete assistant response");
    assert_eq!(app.reasoning.len(), 1);
    assert_eq!(app.reasoning[0].body, "complete inspectable reasoning");
    assert!(app.reasoning[0].finished);
    assert!(app.reasoning[0].abort_reason.is_none());
    assert!(app.streaming_chat.is_none());
    assert!(app.committing_streaming_chat.is_none());
    assert!(app.committing_reasoning.is_none());
}

#[test]
fn committed_chat_and_reasoning_use_bounded_exact_projections() {
    let assistant = format!(
        "ASSISTANT-HEAD{}ASSISTANT-TAIL",
        "a".repeat(MAX_CONVERSATION_DISPLAY_CHARS * 2)
    );
    let reasoning = format!(
        "REASONING-HEAD{}REASONING-TAIL",
        "r".repeat(MAX_CONVERSATION_DISPLAY_CHARS * 2)
    );
    let mut app = AppState::new("api", "model");
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::AssistantTextDelta("preview".into()));
    app.apply_agent_event(AgentEvent::ReasoningDelta("rough".into()));
    app.apply_agent_event(AgentEvent::ApiCallFinished { usage: None });
    app.apply_agent_event(AgentEvent::StreamCommitted {
        text: Some(assistant),
        reasoning: Some(reasoning),
    });

    assert!(app.chat[0].display_capped);
    assert!(app.chat[0].body.chars().count() <= MAX_CONVERSATION_DISPLAY_CHARS);
    assert!(app.chat[0].body.starts_with("ASSISTANT-HEAD"));
    assert!(app.chat[0].body.ends_with("ASSISTANT-TAIL"));
    assert!(app.chat[0].body.contains("/copy last"));

    assert!(app.reasoning[0].display_capped);
    assert!(app.reasoning[0].body.chars().count() <= MAX_CONVERSATION_DISPLAY_CHARS);
    assert!(app.reasoning[0].body.starts_with("REASONING-HEAD"));
    assert!(app.reasoning[0].body.ends_with("REASONING-TAIL"));
    assert!(app.reasoning[0].body.contains("/copy reasoning"));
}

#[test]
fn live_projection_stops_growing_until_commit_or_abort() {
    let mut app = AppState::new("api", "model");
    app.apply_agent_event(AgentEvent::AssistantTextDelta(
        "x".repeat(MAX_CONVERSATION_DISPLAY_CHARS + 1),
    ));
    let capped = app.chat[0].body.clone();
    app.apply_agent_event(AgentEvent::AssistantTextDelta(
        "ignored-after-cap".repeat(100),
    ));

    assert_eq!(app.chat[0].body, capped);
    assert!(app.chat[0].display_capped);
    assert!(app.chat[0].body.contains("Live preview capped"));

    app.apply_agent_event(AgentEvent::AssistantStreamAborted {
        reason: "provider stopped".into(),
    });
    assert!(app.chat[0].body.chars().count() <= MAX_CONVERSATION_DISPLAY_CHARS);
    assert!(app.chat[0].body.contains("Uncommitted stream:"));
    assert!(app.chat[0].body.ends_with("preview may be incomplete.]"));
}

#[test]
fn provider_usage_tracks_reports_missing_data_and_model_buckets() {
    let mut app = AppState::new("api-a", "model-a");
    app.context_tokens = 1_234;
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::ApiCallFinished {
        usage: Some(Usage {
            input_tokens: 10,
            output_tokens: 4,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        }),
    });

    app.api = "api-b".into();
    app.model = "model-b".into();
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::ApiCallFinished { usage: None });
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::ApiCallFinished {
        usage: Some(Usage {
            input_tokens: 7,
            output_tokens: 3,
            cache_read_input_tokens: Some(5),
            cache_creation_input_tokens: Some(2),
        }),
    });

    let report = app.provider_usage_report();
    assert!(report.contains("api-a / model-a: 1 attempt; 1 usage report; 0 unreported attempts"));
    assert!(report.contains("cache read unavailable (0/1 reports)"));
    assert!(report.contains("api-b / model-b: 2 attempts; 1 usage report; 1 unreported attempt"));
    assert!(report
        .contains("Total: 3 attempts; 2 usage reports; 1 unreported attempt; 17 input; 7 output"));
    assert!(report.contains("cache read 5 (1/2 reports)"));
    assert!(report.contains("cache creation 2 (1/2 reports)"));
    assert!(report.contains("Current context estimate: 1234 tokens"));
    assert!(report.contains("not a cost estimate"));
}

#[test]
fn provider_usage_reset_preserves_only_an_active_attempt() {
    let mut app = AppState::new("api", "model");
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.reset_provider_usage();
    app.apply_agent_event(AgentEvent::ApiCallFinished {
        usage: Some(Usage {
            input_tokens: 2,
            output_tokens: 1,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        }),
    });
    let report = app.provider_usage_report();
    assert!(report.contains("1 attempt; 1 usage report; 0 unreported attempts; 2 input; 1 output"));

    app.reset_provider_usage();
    assert!(app
        .provider_usage_report()
        .contains("no API attempts recorded"));
}

#[test]
fn provider_usage_counters_saturate_instead_of_wrapping() {
    let mut totals = ProviderUsageTotals {
        attempts: u64::MAX,
        usage_reports: u64::MAX,
        input_tokens: u64::MAX,
        output_tokens: u64::MAX,
        cache_read_input_tokens: u64::MAX,
        cache_creation_input_tokens: u64::MAX,
        cache_read_reports: u64::MAX,
        cache_creation_reports: u64::MAX,
    };
    totals.record_attempt();
    totals.record_report(&Usage {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: Some(1),
        cache_creation_input_tokens: Some(1),
    });
    assert_eq!(
        totals,
        ProviderUsageTotals {
            attempts: u64::MAX,
            usage_reports: u64::MAX,
            input_tokens: u64::MAX,
            output_tokens: u64::MAX,
            cache_read_input_tokens: u64::MAX,
            cache_creation_input_tokens: u64::MAX,
            cache_read_reports: u64::MAX,
            cache_creation_reports: u64::MAX,
        }
    );
}

#[test]
fn provider_reasoning_stays_out_of_chat_and_has_a_live_inspector() {
    let mut app = AppState::new("OpenAI-compatible", "qwen");
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::ReasoningDelta("inspect the evidence".into()));
    app.apply_agent_event(AgentEvent::AssistantTextDelta("final answer".into()));
    app.apply_agent_event(AgentEvent::ApiCallFinished { usage: None });

    assert_eq!(app.chat.len(), 1);
    assert_eq!(app.chat[0].body, "final answer");
    assert_eq!(app.reasoning.len(), 1);
    assert_eq!(app.reasoning[0].body, "inspect the evidence");
    assert!(app.reasoning[0].finished);

    app.modal = Some(Modal::Reasoning {
        scroll: 0,
        max_scroll: 0,
        follow_latest: true,
    });
    let backend = TestBackend::new(100, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("Model reasoning"));
    assert!(rendered.contains("inspect the evidence"));
    assert!(rendered.contains("complete"));
    assert!(!chat_text(&app.chat)
        .to_string()
        .contains("inspect the evidence"));
}

#[test]
fn reasoning_inspector_is_explicit_when_provider_supplies_nothing() {
    let mut app = AppState::new("api", "model");
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::ApiCallFinished { usage: None });
    app.modal = Some(Modal::Reasoning {
        scroll: 0,
        max_scroll: 0,
        follow_latest: true,
    });

    let backend = TestBackend::new(100, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    assert!(terminal
        .backend()
        .to_string()
        .contains("Provider supplied no inspectable reasoning"));
}

#[test]
fn redacted_reasoning_payload_never_reaches_the_inspector() {
    let secret = "provider-private-redacted-payload";
    let history = vec![Message::assistant(vec![
        ContentBlock::RedactedThinking {
            data: secret.to_string(),
        },
        ContentBlock::Text {
            text: "answer".into(),
        },
    ])];
    let mut app = AppState::new("api", "model");
    app.load_history(&history);
    app.modal = Some(Modal::Reasoning {
        scroll: 0,
        max_scroll: 0,
        follow_latest: true,
    });

    let backend = TestBackend::new(100, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("Provider redacted this reasoning block"));
    assert!(!rendered.contains(secret));
}

#[test]
fn copy_mode_banner_explains_that_rendering_is_paused() {
    let mut app = AppState::new("api", "model");
    app.copy_mode = true;
    let backend = TestBackend::new(100, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("F3/Esc resume"));
    assert!(rendered.contains("select text"));
    assert!(rendered.contains("display paused"));
}

#[test]
fn copy_mode_can_always_resume_with_escape_without_stealing_idle_escape() {
    let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
    let f3 = KeyEvent::new(KeyCode::F(3), KeyModifiers::NONE);
    let ordinary = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE);

    assert!(copy_mode_toggle_key(false, f3));
    assert!(copy_mode_toggle_key(true, f3));
    assert!(copy_mode_toggle_key(true, escape));
    assert!(!copy_mode_toggle_key(false, escape));
    assert!(!copy_mode_toggle_key(true, ordinary));
}

#[test]
fn executable_inputs_are_reviewed_as_sanitized_source_instead_of_json() {
    let python_input = json!({
        "code": "import tools\nprint(\"ok\")\nprint(\"\u{1b}[31m\")",
        "timeout_seconds": 45
    });
    let preview = tool_input_preview("python", &python_input, 2_000);
    assert!(preview.starts_with("import tools\nprint(\"ok\")"));
    assert!(preview.contains("␛[31m"));
    assert!(!preview.contains('\u{1b}'));
    assert!(!preview.contains("\\n"));
    assert!(!preview.contains("\"code\""));

    let request = ToolExecutionRequest {
        tool_use_id: "python-call".into(),
        tool_name: "python".into(),
        input: python_input,
        tool_description: "run code".into(),
    };
    assert_eq!(permission_detail_title(&request), "Python source");
    let detail = plain_text(&permission_detail(&request));
    assert!(detail.contains("1 │ import tools"));
    assert!(detail.contains("2 │ print(\"ok\")"));
    assert!(detail.contains("3 │ print(\"␛[31m\")"));
    assert!(detail.contains("Options\n  timeout_seconds: 45"));
    assert!(!detail.contains("\\n"));

    let bash_input = json!({"command": "cargo test --locked\necho done"});
    assert_eq!(
        tool_input_preview("bash", &bash_input, 2_000),
        "cargo test --locked\necho done"
    );
    let request = ToolExecutionRequest {
        tool_use_id: "bash-call".into(),
        tool_name: "bash".into(),
        input: bash_input,
        tool_description: "run command".into(),
    };
    assert_eq!(permission_detail_title(&request), "Shell command");
    let detail = plain_text(&permission_detail(&request));
    assert!(detail.contains("1 │ cargo test --locked"));
    assert!(detail.contains("2 │ echo done"));
}

#[test]
fn python_highlighting_preserves_source_and_styles_lexical_roles() {
    let source = "@cached\nasync def greet(name: str = \"world\"):\n    # explain\n    return print(f\"hello {name}\", 3.14)\ntext = '''alpha\nreturn remains string\n''' + \"done\"";
    let highlighted = highlight_python_source(source);

    assert_eq!(highlighted_plain(&highlighted), source);
    let decorator = highlighted_span(&highlighted, "@cached");
    assert_eq!(decorator.style.fg, Some(YELLOW));
    assert!(decorator.style.add_modifier.contains(Modifier::BOLD));
    assert_eq!(
        highlighted_span(&highlighted, "async").style.fg,
        Some(PURPLE)
    );
    assert_eq!(highlighted_span(&highlighted, "def").style.fg, Some(PURPLE));
    let definition = highlighted_span(&highlighted, "greet");
    assert_eq!(definition.style.fg, Some(CYAN));
    assert!(definition.style.add_modifier.contains(Modifier::BOLD));
    assert_eq!(highlighted_span(&highlighted, "print").style.fg, Some(CYAN));
    assert_eq!(
        highlighted_span(&highlighted, "\"world\"").style.fg,
        Some(GREEN)
    );
    assert_eq!(
        highlighted_span(&highlighted, "f\"hello {name}\"").style.fg,
        Some(GREEN)
    );
    let comment = highlighted_span(&highlighted, "# explain");
    assert_eq!(comment.style.fg, Some(MUTED));
    assert!(comment.style.add_modifier.contains(Modifier::ITALIC));
    assert_eq!(highlighted_span(&highlighted, "3.14").style.fg, Some(CYAN));
    assert_eq!(highlighted_span(&highlighted, "=").style.fg, Some(YELLOW));
    assert_eq!(
        highlighted_span(&highlighted, "return remains string")
            .style
            .fg,
        Some(GREEN)
    );
    assert_eq!(
        highlighted_span(&highlighted, "'''alpha").style.fg,
        Some(GREEN)
    );
    assert_eq!(highlighted_span(&highlighted, "'''").style.fg, Some(GREEN));
}

#[test]
fn activity_keeps_source_previews_visible_after_tools_finish() {
    let mut app = AppState::new("api", "model");
    app.apply_agent_event(AgentEvent::ToolCallStarted {
        name: "python".into(),
        input: json!({"code": "import tools\nprint(\"ok\")\nprint(\"later\")"}),
    });
    app.apply_agent_event(AgentEvent::ToolCallFinished {
        name: "python".into(),
        outcome: ToolCallOutcome::Success,
        content: "script done".into(),
    });
    app.apply_agent_event(AgentEvent::ToolCallStarted {
        name: "bash".into(),
        input: json!({"command": "cargo test --locked"}),
    });
    app.apply_agent_event(AgentEvent::ToolCallFinished {
        name: "bash".into(),
        outcome: ToolCallOutcome::Success,
        content: "tests passed".into(),
    });

    let backend = TestBackend::new(100, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| render_activity(frame, &app, frame.area()))
        .unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("$ cargo test --locked"));
    assert!(rendered.contains("↳ tests passed"));
    assert!(rendered.contains("│ import tools"));
    assert!(rendered.contains("│ print(\"ok\")"));
    assert!(rendered.contains("│ …"));
    assert!(rendered.contains("↳ script done"));
    assert!(!rendered.contains("{\"code\""));
    assert!(!rendered.contains("{\"command\""));
}

#[test]
fn streaming_and_nested_tools_update_existing_entries() {
    let mut app = AppState::new("OpenAI-compatible", "qwen");
    app.apply_agent_event(AgentEvent::ApiCallStarted);
    app.apply_agent_event(AgentEvent::AssistantTextDelta("hel".into()));
    app.apply_agent_event(AgentEvent::AssistantTextDelta("lo".into()));
    assert_eq!(app.chat.len(), 1);
    assert_eq!(app.chat[0].body, "hello");

    app.apply_agent_event(AgentEvent::ToolCallStarted {
        name: "python".into(),
        input: json!({"code": "print(1)"}),
    });
    app.apply_agent_event(AgentEvent::ToolCallStarted {
        name: "read_file".into(),
        input: json!({"path": "README.md"}),
    });
    app.apply_agent_event(AgentEvent::ToolCallFinished {
        name: "read_file".into(),
        outcome: ToolCallOutcome::Success,
        content: "contents".into(),
    });
    app.apply_agent_event(AgentEvent::ToolCallFinished {
        name: "python".into(),
        outcome: ToolCallOutcome::Success,
        content: "done".into(),
    });
    assert_eq!(app.activity.len(), 2);
    assert!(app
        .activity
        .iter()
        .all(|item| item.status == ActivityStatus::Success));
    assert!(!app.activity[0].via_code_mode);
    assert!(app.activity[1].via_code_mode);
}

#[test]
fn dashboard_renders_with_test_backend() {
    let mut app = AppState::new("OpenAI-compatible", "qwen3.6:35b-a3b");
    app.bridge_count = 14;
    app.context_tokens = 12_400;
    app.goal = Some("Ship the async TUI".into());
    app.push_user("Build a better interface");
    app.push_chat(ChatKind::Assistant, "Working on it.");
    let backend = TestBackend::new(120, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("GENERALIST"));
    assert!(rendered.contains("OpenAI-compatible"));
    assert!(rendered.contains("code mode"));
    assert!(rendered.contains("Ship the async TUI"));
    assert!(rendered.contains("Conversation"));
    assert!(rendered.contains("Tool activity"));
    assert!(rendered.contains("Build a better interface"));
}

#[test]
fn slash_input_is_visibly_command_mode() {
    let mut app = AppState::new("api", "model");
    app.input = "/goal edit".into();
    app.input_cursor = app.input.chars().count();
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("Command"));
    assert!(rendered.contains("Tab complete"));
    assert!(rendered.contains("Enter run"));
    assert!(rendered.contains("/goal"));
}

#[test]
fn help_discovers_host_controls_from_the_command_catalog() {
    let backend = TestBackend::new(100, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(render_help).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("Slash commands"));
    assert!(rendered.contains("/goal"));
    assert!(rendered.contains("run/edit/show/clear objective"));
    assert!(rendered.contains("/permissions"));
    assert!(rendered.contains("inspect/reset remembered policy"));
    assert!(rendered.contains("/tools"));
    assert!(rendered.contains("inspect tools and schemas"));
    assert!(rendered.contains("/mcp"));
    assert!(rendered.contains("inspect/retry MCP connections"));
    assert!(rendered.contains("/exit"));
}

#[test]
fn word_wrapped_chat_reaches_the_real_bottom() {
    let mut app = AppState::new("api", "model");
    let long_words = (0..80)
        .map(|_| "abcdefghijklmnop")
        .collect::<Vec<_>>()
        .join(" ");
    app.push_user(long_words);
    app.push_chat(ChatKind::Assistant, "BOTTOM-MARKER");

    let backend = TestBackend::new(32, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| render_chat(frame, &mut app, frame.area()))
        .unwrap();

    assert!(app.follow_latest);
    assert!(
        terminal.backend().to_string().contains("BOTTOM-MARKER"),
        "follow-latest stopped above Ratatui's word-wrapped bottom"
    );
    assert_eq!(
        terminal
            .backend()
            .buffer()
            .cell((30, 8))
            .expect("bottom scrollbar cell")
            .symbol(),
        "▐",
        "the scrollbar must visually reach the bottom with follow-latest active"
    );

    app.scroll_chat_up(usize::MAX);
    terminal
        .draw(|frame| render_chat(frame, &mut app, frame.area()))
        .unwrap();
    assert!(!app.follow_latest);
    assert!(!terminal.backend().to_string().contains("BOTTOM-MARKER"));

    app.scroll_chat_down(usize::MAX);
    terminal
        .draw(|frame| render_chat(frame, &mut app, frame.area()))
        .unwrap();
    assert!(app.follow_latest);
    assert!(
        terminal.backend().to_string().contains("BOTTOM-MARKER"),
        "scroll-down stopped above Ratatui's word-wrapped bottom"
    );
    assert_eq!(
        terminal
            .backend()
            .buffer()
            .cell((30, 8))
            .expect("bottom scrollbar cell")
            .symbol(),
        "▐",
        "the scrollbar must return to the bottom after scrolling down"
    );
}

#[test]
fn queue_modal_keeps_a_long_selection_visible() {
    let queue = (0..14)
        .map(|id| QueuedPrompt {
            id,
            text: format!("queue-item-{id}"),
            delivery: DeliveryMode::FollowUp,
            source: PromptSource::User,
        })
        .collect::<Vec<_>>();
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| render_queue(frame, &queue, 13, None))
        .unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("queue-item-13"));
    assert!(!rendered.contains("queue-item-0"));
}

#[test]
fn dashboard_and_modals_render_on_tiny_terminals() {
    let request = ToolExecutionRequest {
        tool_use_id: "tool".into(),
        tool_name: "bash".into(),
        input: json!({"command": "printf x"}),
        tool_description: "run a command".into(),
    };
    let modals = vec![
        None,
        Some(Modal::Help),
        Some(Modal::Select {
            title: "Select".into(),
            items: vec!["one".into()],
            selected: 0,
        }),
        Some(Modal::Prompt {
            title: "Prompt".into(),
            value: "value".into(),
            cursor: 5,
        }),
        Some(Modal::Permission {
            id: 1,
            request,
            selected: 1,
            scroll: u16::MAX,
        }),
        Some(Modal::Queue {
            selected: 0,
            editing: None,
        }),
        Some(Modal::Search(ChatSearch {
            query: "needle".into(),
            cursor: 6,
            matches: Vec::new(),
            selected: 0,
        })),
        Some(Modal::Reasoning {
            scroll: 0,
            max_scroll: usize::MAX,
            follow_latest: true,
        }),
    ];
    for (width, height) in [(1, 1), (8, 3), (24, 8), (60, 16)] {
        for modal in &modals {
            let mut app = AppState::new("api", "model");
            app.modal = modal.clone();
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();
            terminal
                .draw(|frame| render(frame, &mut app))
                .unwrap_or_else(|error| panic!("{width}x{height} failed for {modal:?}: {error}"));
        }
    }
}

#[test]
fn composer_keys_distinguish_turn_steering_from_background_queueing() {
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    let alt_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT);
    let tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
    let shift_enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT);

    assert_eq!(
        submission_delivery(enter, false),
        Some(DeliveryMode::FollowUp)
    );
    assert_eq!(submission_delivery(enter, true), Some(DeliveryMode::Steer));
    assert_eq!(
        submission_delivery(alt_enter, true),
        Some(DeliveryMode::FollowUp)
    );
    assert_eq!(submission_delivery(tab, true), Some(DeliveryMode::FollowUp));
    assert_eq!(submission_delivery(shift_enter, true), None);
}

#[test]
fn background_busy_state_does_not_offer_or_create_steering() {
    let mut app = AppState::new("api", "model");
    app.busy = true;
    app.turn_active = false;
    let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(
        submission_delivery(enter, app.turn_active),
        Some(DeliveryMode::FollowUp)
    );

    let queue = PromptQueue::default();
    let id = queue.enqueue("later", DeliveryMode::FollowUp);
    assert!(!queue.toggle_delivery(id, app.turn_active));
    assert_eq!(queue.snapshot()[0].delivery, DeliveryMode::FollowUp);

    let backend = TestBackend::new(100, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, &mut app)).unwrap();
    let rendered = terminal.backend().to_string();
    assert!(rendered.contains("Enter queue"));
    assert!(!rendered.contains("Enter steer"));
}

#[test]
fn only_queue_mutations_request_terminal_event_persistence() {
    assert!(!UiAction::None.requires_queue_persist());
    assert!(!UiAction::Interrupt.requires_queue_persist());
    assert!(!UiAction::Exit.requires_queue_persist());
    assert!(UiAction::QueueChanged.requires_queue_persist());
    assert!(UiAction::Submit {
        text: "queued".into(),
        delivery: DeliveryMode::FollowUp,
    }
    .requires_queue_persist());
}

#[test]
fn interrupted_turn_retires_all_nested_activity() {
    let mut app = AppState::new("OpenAI-compatible", "model");
    app.apply_agent_event(AgentEvent::ToolCallStarted {
        name: "python".into(),
        input: json!({}),
    });
    app.apply_agent_event(AgentEvent::ToolCallStarted {
        name: "bash".into(),
        input: json!({}),
    });

    app.cancel_running_activity();

    assert!(app.active_tools.is_empty());
    assert!(app
        .activity
        .iter()
        .all(|item| item.status == ActivityStatus::Cancelled));
}
