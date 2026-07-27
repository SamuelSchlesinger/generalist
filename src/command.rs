//! Typed local slash commands shared by the reactor and TUI discovery.
//!
//! Local commands never enter model-visible conversation history. The parser
//! intentionally keeps the remainder of `/goal <text>` as a borrowed slice so
//! objective text is not re-tokenized or shell-parsed.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalCommand<'a> {
    /// Open the interactive goal editor.
    Edit,
    /// Display the complete active goal.
    Show,
    /// Remove the active goal.
    Clear,
    /// Replace the active goal directly.
    Set(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCommand<'a> {
    /// Display capture state and the current project scope.
    Status,
    /// Stop recording future settled turns.
    Pause,
    /// Opt in to recording future settled turns.
    Resume,
    /// Search retained user/assistant text and tool names.
    Search(&'a str),
    /// Display one episode by full ID or unique prefix.
    Show(&'a str),
    /// Export live episodes for the current project.
    Export,
    /// Delete one live episode by full ID or unique prefix.
    Forget(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalCommand<'a> {
    Exit,
    Help,
    Compact,
    Clear,
    Save,
    Load,
    Model,
    Goal(GoalCommand<'a>),
    Memory(MemoryCommand<'a>),
    Unknown(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandSpec {
    pub name: &'static str,
    pub usage: &'static str,
    pub description: &'static str,
    kind: CommandKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandKind {
    Exit,
    Help,
    Compact,
    Clear,
    Save,
    Load,
    Model,
    Goal,
    Memory,
}

/// The authoritative discovery order for slash commands.
pub const COMMAND_SPECS: &[CommandSpec] = &[
    CommandSpec {
        name: "/goal",
        usage: "/goal [edit|show|clear|<text>]",
        description: "edit/show/clear/set objective",
        kind: CommandKind::Goal,
    },
    CommandSpec {
        name: "/help",
        usage: "/help",
        description: "show commands and shortcuts",
        kind: CommandKind::Help,
    },
    CommandSpec {
        name: "/model",
        usage: "/model",
        description: "switch API or model",
        kind: CommandKind::Model,
    },
    CommandSpec {
        name: "/memory",
        usage: "/memory [status|pause|resume|search <query>|show <id>|export|forget <id>]",
        description: "inspect/manage explicit episodic memory",
        kind: CommandKind::Memory,
    },
    CommandSpec {
        name: "/save",
        usage: "/save",
        description: "save this session",
        kind: CommandKind::Save,
    },
    CommandSpec {
        name: "/load",
        usage: "/load",
        description: "load a saved session",
        kind: CommandKind::Load,
    },
    CommandSpec {
        name: "/compact",
        usage: "/compact",
        description: "summarize older context",
        kind: CommandKind::Compact,
    },
    CommandSpec {
        name: "/clear",
        usage: "/clear",
        description: "clear conversation history",
        kind: CommandKind::Clear,
    },
    CommandSpec {
        name: "/exit",
        usage: "/exit",
        description: "close generalist",
        kind: CommandKind::Exit,
    },
];

pub fn parse_local_command(text: &str) -> Option<LocalCommand<'_>> {
    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
        return Some(LocalCommand::Exit);
    }
    let body = trimmed.strip_prefix('/')?;
    let name_end = body.find(char::is_whitespace).unwrap_or(body.len());
    let name = &body[..name_end];
    let arguments = body[name_end..].trim();

    if name.eq_ignore_ascii_case("quit") && arguments.is_empty() {
        return Some(LocalCommand::Exit);
    }

    let Some(spec) = COMMAND_SPECS.iter().find(|spec| {
        spec.name
            .strip_prefix('/')
            .is_some_and(|catalog_name| name.eq_ignore_ascii_case(catalog_name))
    }) else {
        return Some(LocalCommand::Unknown(trimmed));
    };

    if !matches!(spec.kind, CommandKind::Goal | CommandKind::Memory) && !arguments.is_empty() {
        return Some(LocalCommand::Unknown(trimmed));
    }

    Some(match spec.kind {
        CommandKind::Exit => LocalCommand::Exit,
        CommandKind::Help => LocalCommand::Help,
        CommandKind::Compact => LocalCommand::Compact,
        CommandKind::Clear => LocalCommand::Clear,
        CommandKind::Save => LocalCommand::Save,
        CommandKind::Load => LocalCommand::Load,
        CommandKind::Model => LocalCommand::Model,
        CommandKind::Goal => {
            let goal = if arguments.is_empty() || arguments.eq_ignore_ascii_case("edit") {
                GoalCommand::Edit
            } else if arguments.eq_ignore_ascii_case("show") {
                GoalCommand::Show
            } else if arguments.eq_ignore_ascii_case("clear") {
                GoalCommand::Clear
            } else {
                GoalCommand::Set(arguments)
            };
            LocalCommand::Goal(goal)
        }
        CommandKind::Memory => {
            let argument_end = arguments
                .find(char::is_whitespace)
                .unwrap_or(arguments.len());
            let action = &arguments[..argument_end];
            let value = arguments[argument_end..].trim();
            let memory = if arguments.is_empty() || action.eq_ignore_ascii_case("status") {
                if value.is_empty() {
                    Some(MemoryCommand::Status)
                } else {
                    None
                }
            } else if action.eq_ignore_ascii_case("pause") && value.is_empty() {
                Some(MemoryCommand::Pause)
            } else if action.eq_ignore_ascii_case("resume") && value.is_empty() {
                Some(MemoryCommand::Resume)
            } else if action.eq_ignore_ascii_case("search") && !value.is_empty() {
                Some(MemoryCommand::Search(value))
            } else if action.eq_ignore_ascii_case("show") && !value.is_empty() {
                Some(MemoryCommand::Show(value))
            } else if action.eq_ignore_ascii_case("export") && value.is_empty() {
                Some(MemoryCommand::Export)
            } else if action.eq_ignore_ascii_case("forget") && !value.is_empty() {
                Some(MemoryCommand::Forget(value))
            } else {
                None
            };
            match memory {
                Some(memory) => LocalCommand::Memory(memory),
                None => LocalCommand::Unknown(trimmed),
            }
        }
    })
}

pub fn is_local_command(text: &str) -> bool {
    parse_local_command(text).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_commands_are_explicit_and_preserve_objective_text() {
        assert_eq!(
            parse_local_command("/goal"),
            Some(LocalCommand::Goal(GoalCommand::Edit))
        );
        assert_eq!(
            parse_local_command("/GOAL edit"),
            Some(LocalCommand::Goal(GoalCommand::Edit))
        );
        assert_eq!(
            parse_local_command("/goal show"),
            Some(LocalCommand::Goal(GoalCommand::Show))
        );
        assert_eq!(
            parse_local_command("/goal clear"),
            Some(LocalCommand::Goal(GoalCommand::Clear))
        );
        assert_eq!(
            parse_local_command("  /goal   ship the async TUI  "),
            Some(LocalCommand::Goal(GoalCommand::Set("ship the async TUI")))
        );
    }

    #[test]
    fn slash_and_legacy_exit_are_local_but_prompts_are_not() {
        assert_eq!(parse_local_command("/exit"), Some(LocalCommand::Exit));
        assert_eq!(parse_local_command("/quit"), Some(LocalCommand::Exit));
        assert_eq!(parse_local_command("quit"), Some(LocalCommand::Exit));
        assert_eq!(
            parse_local_command("/help unexpected"),
            Some(LocalCommand::Unknown("/help unexpected"))
        );
        assert_eq!(
            parse_local_command("/unknown value"),
            Some(LocalCommand::Unknown("/unknown value"))
        );
        assert_eq!(parse_local_command("ordinary prompt"), None);
    }

    #[test]
    fn memory_commands_preserve_queries_and_require_arguments() {
        assert_eq!(
            parse_local_command("/memory"),
            Some(LocalCommand::Memory(MemoryCommand::Status))
        );
        assert_eq!(
            parse_local_command("/MEMORY status"),
            Some(LocalCommand::Memory(MemoryCommand::Status))
        );
        assert_eq!(
            parse_local_command("/memory search exact project convention"),
            Some(LocalCommand::Memory(MemoryCommand::Search(
                "exact project convention"
            )))
        );
        assert_eq!(
            parse_local_command("/memory show deadbeef"),
            Some(LocalCommand::Memory(MemoryCommand::Show("deadbeef")))
        );
        assert_eq!(
            parse_local_command("/memory forget deadbeef"),
            Some(LocalCommand::Memory(MemoryCommand::Forget("deadbeef")))
        );
        assert_eq!(
            parse_local_command("/memory search"),
            Some(LocalCommand::Unknown("/memory search"))
        );
        assert_eq!(
            parse_local_command("/memory export elsewhere"),
            Some(LocalCommand::Unknown("/memory export elsewhere"))
        );
    }

    #[test]
    fn catalog_entries_are_unique_and_parse_as_known_commands() {
        let mut names = std::collections::HashSet::new();
        for spec in COMMAND_SPECS {
            assert!(names.insert(spec.name.to_ascii_lowercase()));
            assert!(
                !matches!(
                    parse_local_command(spec.name),
                    None | Some(LocalCommand::Unknown(_))
                ),
                "{} is discoverable but not parsed",
                spec.name
            );
        }
    }
}
