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
pub enum McpCommand<'a> {
    /// Display configured server connection state.
    Status,
    /// Retry all failed/skipped servers, or one exact configured name.
    Retry(Option<&'a str>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyCommand {
    /// Copy the most recent committed assistant text through OSC 52.
    Last,
    /// Copy the complete committed user/assistant transcript through OSC 52.
    All,
    /// Copy the most recent inspectable committed reasoning through OSC 52.
    Reasoning,
    /// Hand mouse selection to the native terminal.
    Select,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionCommand<'a> {
    /// Display every remembered per-tool decision.
    List,
    /// Return one exact tool name to ask-on-use behavior.
    Reset(&'a str),
    /// Remove every remembered decision.
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolsCommand<'a> {
    /// Display the registered bridge catalog.
    List,
    /// Filter bridge names and descriptions without calling a tool.
    Search(&'a str),
    /// Display one exact tool description and input schema.
    Show(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryCommand<'a> {
    /// Display current-scope saved-session names.
    List,
    /// Search sanitized current-scope saved-session content.
    Search(&'a str),
    /// Inspect one named session without loading it.
    Show(&'a str),
    /// Delete one named current-scope session after host confirmation.
    Forget(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsageCommand {
    /// Display process-local provider attempt and token accounting.
    Show,
    /// Clear process-local provider attempt and token accounting.
    Reset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalCommand<'a> {
    Exit,
    Help,
    Compact,
    Clear,
    Save(Option<&'a str>),
    Load(Option<&'a str>),
    Model,
    Mcp(McpCommand<'a>),
    Copy(CopyCommand),
    Permissions(PermissionCommand<'a>),
    Tools(ToolsCommand<'a>),
    History(HistoryCommand<'a>),
    Usage(UsageCommand),
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
    subcommands: &'static [SubcommandSpec],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubcommandSpec {
    name: &'static str,
    requires_value: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandCompletion {
    /// Replace the composer with this canonical command prefix.
    Replace(String),
    /// More than one catalog entry matches; keep the composer unchanged.
    Candidates(Vec<String>),
    /// The current prefix is already canonical and complete.
    Complete,
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
    Mcp,
    Copy,
    Permissions,
    Tools,
    History,
    Usage,
    Goal,
    Memory,
}

const NO_SUBCOMMANDS: &[SubcommandSpec] = &[];
const GOAL_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "edit",
        requires_value: false,
    },
    SubcommandSpec {
        name: "show",
        requires_value: false,
    },
    SubcommandSpec {
        name: "clear",
        requires_value: false,
    },
];
const MEMORY_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "status",
        requires_value: false,
    },
    SubcommandSpec {
        name: "pause",
        requires_value: false,
    },
    SubcommandSpec {
        name: "resume",
        requires_value: false,
    },
    SubcommandSpec {
        name: "search",
        requires_value: true,
    },
    SubcommandSpec {
        name: "show",
        requires_value: true,
    },
    SubcommandSpec {
        name: "export",
        requires_value: false,
    },
    SubcommandSpec {
        name: "forget",
        requires_value: true,
    },
];
const MCP_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "status",
        requires_value: false,
    },
    SubcommandSpec {
        name: "retry",
        requires_value: true,
    },
];
const COPY_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "last",
        requires_value: false,
    },
    SubcommandSpec {
        name: "all",
        requires_value: false,
    },
    SubcommandSpec {
        name: "reasoning",
        requires_value: false,
    },
    SubcommandSpec {
        name: "select",
        requires_value: false,
    },
];
const PERMISSION_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "list",
        requires_value: false,
    },
    SubcommandSpec {
        name: "reset",
        requires_value: true,
    },
    SubcommandSpec {
        name: "clear",
        requires_value: false,
    },
];
const TOOL_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "list",
        requires_value: false,
    },
    SubcommandSpec {
        name: "search",
        requires_value: true,
    },
    SubcommandSpec {
        name: "show",
        requires_value: true,
    },
];
const HISTORY_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "list",
        requires_value: false,
    },
    SubcommandSpec {
        name: "search",
        requires_value: true,
    },
    SubcommandSpec {
        name: "show",
        requires_value: true,
    },
    SubcommandSpec {
        name: "forget",
        requires_value: true,
    },
];
const USAGE_SUBCOMMANDS: &[SubcommandSpec] = &[
    SubcommandSpec {
        name: "show",
        requires_value: false,
    },
    SubcommandSpec {
        name: "reset",
        requires_value: false,
    },
];

/// The authoritative discovery order for slash commands.
pub const COMMAND_SPECS: &[CommandSpec] = &[
    CommandSpec {
        name: "/goal",
        usage: "/goal [edit|show|clear|<text>]",
        description: "run/edit/show/clear objective",
        kind: CommandKind::Goal,
        subcommands: GOAL_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/help",
        usage: "/help",
        description: "show commands and shortcuts",
        kind: CommandKind::Help,
        subcommands: NO_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/history",
        usage: "/history [list|search <query>|show <name>|forget <name>]",
        description: "inspect or forget saved sessions",
        kind: CommandKind::History,
        subcommands: HISTORY_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/model",
        usage: "/model",
        description: "switch API or model",
        kind: CommandKind::Model,
        subcommands: NO_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/mcp",
        usage: "/mcp [status|retry [server]]",
        description: "inspect/retry MCP connections",
        kind: CommandKind::Mcp,
        subcommands: MCP_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/memory",
        usage: "/memory [status|pause|resume|search <query>|show <id>|export|forget <id>]",
        description: "inspect/manage explicit episodic memory",
        kind: CommandKind::Memory,
        subcommands: MEMORY_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/permissions",
        usage: "/permissions [list|reset <tool>|clear]",
        description: "inspect/reset remembered policy",
        kind: CommandKind::Permissions,
        subcommands: PERMISSION_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/tools",
        usage: "/tools [list|search <query>|show <name>]",
        description: "inspect tools and schemas",
        kind: CommandKind::Tools,
        subcommands: TOOL_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/usage",
        usage: "/usage [show|reset]",
        description: "inspect/reset provider token reports",
        kind: CommandKind::Usage,
        subcommands: USAGE_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/save",
        usage: "/save [name]",
        description: "save this session",
        kind: CommandKind::Save,
        subcommands: NO_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/load",
        usage: "/load [name]",
        description: "load a saved session",
        kind: CommandKind::Load,
        subcommands: NO_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/compact",
        usage: "/compact",
        description: "summarize older context",
        kind: CommandKind::Compact,
        subcommands: NO_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/copy",
        usage: "/copy [last|all|reasoning|select]",
        description: "copy response/transcript/reasoning or select text",
        kind: CommandKind::Copy,
        subcommands: COPY_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/clear",
        usage: "/clear",
        description: "clear conversation history",
        kind: CommandKind::Clear,
        subcommands: NO_SUBCOMMANDS,
    },
    CommandSpec {
        name: "/exit",
        usage: "/exit",
        description: "close generalist",
        kind: CommandKind::Exit,
        subcommands: NO_SUBCOMMANDS,
    },
];

/// Complete one catalog-backed slash-command prefix.
///
/// The caller should invoke this only when the cursor is at the end of the
/// composer. Inputs containing a newline or a complete value-bearing argument
/// return `None`, preserving the caller's ordinary Tab behavior. Leading
/// horizontal whitespace is retained; canonical command/subcommand spelling
/// is lowercase.
pub fn complete_local_command(text: &str) -> Option<CommandCompletion> {
    if text.contains('\r') || text.contains('\n') {
        return None;
    }
    let body = text.trim_start();
    let leading = &text[..text.len() - body.len()];
    if !body.starts_with('/') {
        return None;
    }

    let Some(separator) = body.find(char::is_whitespace) else {
        let prefix = body.to_ascii_lowercase();
        let matches = COMMAND_SPECS
            .iter()
            .filter(|spec| spec.name.starts_with(&prefix))
            .collect::<Vec<_>>();
        let exact = matches
            .iter()
            .copied()
            .find(|spec| spec.name.eq_ignore_ascii_case(body));
        let selected = exact.or_else(|| {
            if matches.len() == 1 {
                Some(matches[0])
            } else {
                None
            }
        });
        if let Some(spec) = selected {
            let suffix = if spec.subcommands.is_empty() { "" } else { " " };
            let replacement = format!("{leading}{}{suffix}", spec.name);
            return Some(if replacement == text {
                CommandCompletion::Complete
            } else {
                CommandCompletion::Replace(replacement)
            });
        }
        return (matches.len() > 1).then(|| {
            CommandCompletion::Candidates(
                matches.iter().map(|spec| spec.name.to_string()).collect(),
            )
        });
    };

    let command_name = &body[..separator];
    let spec = COMMAND_SPECS
        .iter()
        .find(|spec| spec.name.eq_ignore_ascii_case(command_name))?;
    if spec.subcommands.is_empty() {
        return body[separator..]
            .trim()
            .is_empty()
            .then_some(CommandCompletion::Complete);
    }
    let argument = body[separator..].trim();
    if argument.chars().any(char::is_whitespace) {
        return None;
    }
    let normalized = argument.to_ascii_lowercase();
    let matches = spec
        .subcommands
        .iter()
        .filter(|subcommand| subcommand.name.starts_with(&normalized))
        .collect::<Vec<_>>();
    let exact = matches
        .iter()
        .copied()
        .find(|subcommand| subcommand.name.eq_ignore_ascii_case(argument));
    let selected = exact.or_else(|| {
        if matches.len() == 1 {
            Some(matches[0])
        } else {
            None
        }
    });
    if let Some(subcommand) = selected {
        let suffix = if subcommand.requires_value { " " } else { "" };
        let replacement = format!("{leading}{} {}{suffix}", spec.name, subcommand.name);
        return Some(if replacement == text {
            CommandCompletion::Complete
        } else {
            CommandCompletion::Replace(replacement)
        });
    }
    (matches.len() > 1).then(|| {
        CommandCompletion::Candidates(
            matches
                .iter()
                .map(|subcommand| format!("{} {}", spec.name, subcommand.name))
                .collect(),
        )
    })
}

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

    if !matches!(
        spec.kind,
        CommandKind::Goal
            | CommandKind::Memory
            | CommandKind::Mcp
            | CommandKind::Copy
            | CommandKind::Permissions
            | CommandKind::Tools
            | CommandKind::History
            | CommandKind::Usage
            | CommandKind::Save
            | CommandKind::Load
    ) && !arguments.is_empty()
    {
        return Some(LocalCommand::Unknown(trimmed));
    }

    Some(match spec.kind {
        CommandKind::Exit => LocalCommand::Exit,
        CommandKind::Help => LocalCommand::Help,
        CommandKind::Compact => LocalCommand::Compact,
        CommandKind::Clear => LocalCommand::Clear,
        CommandKind::Save => LocalCommand::Save((!arguments.is_empty()).then_some(arguments)),
        CommandKind::Load => LocalCommand::Load((!arguments.is_empty()).then_some(arguments)),
        CommandKind::Model => LocalCommand::Model,
        CommandKind::Mcp => {
            let argument_end = arguments
                .find(char::is_whitespace)
                .unwrap_or(arguments.len());
            let action = &arguments[..argument_end];
            let value = arguments[argument_end..].trim();
            let mcp = if arguments.is_empty() || action.eq_ignore_ascii_case("status") {
                value.is_empty().then_some(McpCommand::Status)
            } else if action.eq_ignore_ascii_case("retry") {
                Some(McpCommand::Retry((!value.is_empty()).then_some(value)))
            } else {
                None
            };
            match mcp {
                Some(mcp) => LocalCommand::Mcp(mcp),
                None => LocalCommand::Unknown(trimmed),
            }
        }
        CommandKind::Usage => {
            let usage = if arguments.is_empty() || arguments.eq_ignore_ascii_case("show") {
                Some(UsageCommand::Show)
            } else if arguments.eq_ignore_ascii_case("reset") {
                Some(UsageCommand::Reset)
            } else {
                None
            };
            match usage {
                Some(usage) => LocalCommand::Usage(usage),
                None => LocalCommand::Unknown(trimmed),
            }
        }
        CommandKind::Copy => {
            let copy = if arguments.is_empty() || arguments.eq_ignore_ascii_case("last") {
                Some(CopyCommand::Last)
            } else if arguments.eq_ignore_ascii_case("all") {
                Some(CopyCommand::All)
            } else if arguments.eq_ignore_ascii_case("reasoning") {
                Some(CopyCommand::Reasoning)
            } else if arguments.eq_ignore_ascii_case("select") {
                Some(CopyCommand::Select)
            } else {
                None
            };
            match copy {
                Some(copy) => LocalCommand::Copy(copy),
                None => LocalCommand::Unknown(trimmed),
            }
        }
        CommandKind::Permissions => {
            let argument_end = arguments
                .find(char::is_whitespace)
                .unwrap_or(arguments.len());
            let action = &arguments[..argument_end];
            let value = arguments[argument_end..].trim();
            let permission = if arguments.is_empty() || action.eq_ignore_ascii_case("list") {
                value.is_empty().then_some(PermissionCommand::List)
            } else if action.eq_ignore_ascii_case("clear") {
                value.is_empty().then_some(PermissionCommand::Clear)
            } else if action.eq_ignore_ascii_case("reset")
                && !value.is_empty()
                && !value.chars().any(char::is_whitespace)
            {
                Some(PermissionCommand::Reset(value))
            } else {
                None
            };
            match permission {
                Some(permission) => LocalCommand::Permissions(permission),
                None => LocalCommand::Unknown(trimmed),
            }
        }
        CommandKind::Tools => {
            let argument_end = arguments
                .find(char::is_whitespace)
                .unwrap_or(arguments.len());
            let action = &arguments[..argument_end];
            let value = arguments[argument_end..].trim();
            let tools = if arguments.is_empty() || action.eq_ignore_ascii_case("list") {
                value.is_empty().then_some(ToolsCommand::List)
            } else if action.eq_ignore_ascii_case("search") && !value.is_empty() {
                Some(ToolsCommand::Search(value))
            } else if action.eq_ignore_ascii_case("show")
                && !value.is_empty()
                && !value.chars().any(char::is_whitespace)
            {
                Some(ToolsCommand::Show(value))
            } else {
                None
            };
            match tools {
                Some(tools) => LocalCommand::Tools(tools),
                None => LocalCommand::Unknown(trimmed),
            }
        }
        CommandKind::History => {
            let argument_end = arguments
                .find(char::is_whitespace)
                .unwrap_or(arguments.len());
            let action = &arguments[..argument_end];
            let value = arguments[argument_end..].trim();
            let history = if arguments.is_empty() || action.eq_ignore_ascii_case("list") {
                value.is_empty().then_some(HistoryCommand::List)
            } else if action.eq_ignore_ascii_case("search") && !value.is_empty() {
                Some(HistoryCommand::Search(value))
            } else if action.eq_ignore_ascii_case("show") && !value.is_empty() {
                Some(HistoryCommand::Show(value))
            } else if action.eq_ignore_ascii_case("forget") && !value.is_empty() {
                Some(HistoryCommand::Forget(value))
            } else {
                None
            };
            match history {
                Some(history) => LocalCommand::History(history),
                None => LocalCommand::Unknown(trimmed),
            }
        }
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
    fn mcp_commands_support_status_and_targeted_retry() {
        assert_eq!(
            parse_local_command("/mcp"),
            Some(LocalCommand::Mcp(McpCommand::Status))
        );
        assert_eq!(
            parse_local_command("/MCP status"),
            Some(LocalCommand::Mcp(McpCommand::Status))
        );
        assert_eq!(
            parse_local_command("/mcp retry"),
            Some(LocalCommand::Mcp(McpCommand::Retry(None)))
        );
        assert_eq!(
            parse_local_command("/mcp retry flaky server"),
            Some(LocalCommand::Mcp(McpCommand::Retry(Some("flaky server"))))
        );
        assert_eq!(
            parse_local_command("/mcp status extra"),
            Some(LocalCommand::Unknown("/mcp status extra"))
        );
        assert_eq!(
            parse_local_command("/mcp reconnect"),
            Some(LocalCommand::Unknown("/mcp reconnect"))
        );
    }

    #[test]
    fn copy_commands_are_explicit_and_reject_unknown_targets() {
        assert_eq!(
            parse_local_command("/copy"),
            Some(LocalCommand::Copy(CopyCommand::Last))
        );
        assert_eq!(
            parse_local_command("/COPY last"),
            Some(LocalCommand::Copy(CopyCommand::Last))
        );
        assert_eq!(
            parse_local_command("/copy all"),
            Some(LocalCommand::Copy(CopyCommand::All))
        );
        assert_eq!(
            parse_local_command("/copy reasoning"),
            Some(LocalCommand::Copy(CopyCommand::Reasoning))
        );
        assert_eq!(
            parse_local_command("/copy select"),
            Some(LocalCommand::Copy(CopyCommand::Select))
        );
        assert_eq!(
            parse_local_command("/copy something-else"),
            Some(LocalCommand::Unknown("/copy something-else"))
        );
    }

    #[test]
    fn permission_commands_are_explicit_and_require_an_exact_tool() {
        assert_eq!(
            parse_local_command("/permissions"),
            Some(LocalCommand::Permissions(PermissionCommand::List))
        );
        assert_eq!(
            parse_local_command("/PERMISSIONS list"),
            Some(LocalCommand::Permissions(PermissionCommand::List))
        );
        assert_eq!(
            parse_local_command("/permissions reset bash"),
            Some(LocalCommand::Permissions(PermissionCommand::Reset("bash")))
        );
        assert_eq!(
            parse_local_command("/permissions clear"),
            Some(LocalCommand::Permissions(PermissionCommand::Clear))
        );
        assert_eq!(
            parse_local_command("/permissions reset"),
            Some(LocalCommand::Unknown("/permissions reset"))
        );
        assert_eq!(
            parse_local_command("/permissions reset bash extra"),
            Some(LocalCommand::Unknown("/permissions reset bash extra"))
        );
    }

    #[test]
    fn tool_catalog_commands_are_read_only_and_require_explicit_arguments() {
        assert_eq!(
            parse_local_command("/tools"),
            Some(LocalCommand::Tools(ToolsCommand::List))
        );
        assert_eq!(
            parse_local_command("/TOOLS list"),
            Some(LocalCommand::Tools(ToolsCommand::List))
        );
        assert_eq!(
            parse_local_command("/tools search archive history"),
            Some(LocalCommand::Tools(ToolsCommand::Search("archive history")))
        );
        assert_eq!(
            parse_local_command("/tools show search_memories"),
            Some(LocalCommand::Tools(ToolsCommand::Show("search_memories")))
        );
        assert_eq!(
            parse_local_command("/tools search"),
            Some(LocalCommand::Unknown("/tools search"))
        );
        assert_eq!(
            parse_local_command("/tools show one extra"),
            Some(LocalCommand::Unknown("/tools show one extra"))
        );
    }

    #[test]
    fn history_commands_preserve_names_and_require_explicit_arguments() {
        assert_eq!(
            parse_local_command("/history"),
            Some(LocalCommand::History(HistoryCommand::List))
        );
        assert_eq!(
            parse_local_command("/HISTORY list"),
            Some(LocalCommand::History(HistoryCommand::List))
        );
        assert_eq!(
            parse_local_command("/history search old project convention"),
            Some(LocalCommand::History(HistoryCommand::Search(
                "old project convention"
            )))
        );
        assert_eq!(
            parse_local_command("/history show release-notes"),
            Some(LocalCommand::History(HistoryCommand::Show("release-notes")))
        );
        assert_eq!(
            parse_local_command("/history show release notes"),
            Some(LocalCommand::History(HistoryCommand::Show("release notes")))
        );
        assert_eq!(
            parse_local_command("/history forget release notes"),
            Some(LocalCommand::History(HistoryCommand::Forget(
                "release notes"
            )))
        );
        assert_eq!(
            parse_local_command("/history search"),
            Some(LocalCommand::Unknown("/history search"))
        );
        assert_eq!(
            parse_local_command("/history forget"),
            Some(LocalCommand::Unknown("/history forget"))
        );
    }

    #[test]
    fn usage_commands_are_host_owned_and_reject_extra_arguments() {
        assert_eq!(
            parse_local_command("/usage"),
            Some(LocalCommand::Usage(UsageCommand::Show))
        );
        assert_eq!(
            parse_local_command("/USAGE show"),
            Some(LocalCommand::Usage(UsageCommand::Show))
        );
        assert_eq!(
            parse_local_command("/usage reset"),
            Some(LocalCommand::Usage(UsageCommand::Reset))
        );
        assert_eq!(
            parse_local_command("/usage reset later"),
            Some(LocalCommand::Unknown("/usage reset later"))
        );
        assert_eq!(
            parse_local_command("/usage cost"),
            Some(LocalCommand::Unknown("/usage cost"))
        );
    }

    #[test]
    fn save_and_load_accept_optional_exact_names() {
        assert_eq!(parse_local_command("/save"), Some(LocalCommand::Save(None)));
        assert_eq!(
            parse_local_command("/save release checkpoint"),
            Some(LocalCommand::Save(Some("release checkpoint")))
        );
        assert_eq!(parse_local_command("/load"), Some(LocalCommand::Load(None)));
        assert_eq!(
            parse_local_command("/LOAD release checkpoint"),
            Some(LocalCommand::Load(Some("release checkpoint")))
        );
    }

    #[test]
    fn completion_uses_the_catalog_and_preserves_argument_text() {
        assert_eq!(
            complete_local_command("/to"),
            Some(CommandCompletion::Replace("/tools ".to_string()))
        );
        assert_eq!(
            complete_local_command("  /TOO"),
            Some(CommandCompletion::Replace("  /tools ".to_string()))
        );
        assert_eq!(
            complete_local_command("/tools se"),
            Some(CommandCompletion::Replace("/tools search ".to_string()))
        );
        assert_eq!(
            complete_local_command("/memory pa"),
            Some(CommandCompletion::Replace("/memory pause".to_string()))
        );
        assert_eq!(
            complete_local_command("/mc"),
            Some(CommandCompletion::Replace("/mcp ".to_string()))
        );
        assert_eq!(
            complete_local_command("/mcp r"),
            Some(CommandCompletion::Replace("/mcp retry ".to_string()))
        );
        assert_eq!(
            complete_local_command("/exit"),
            Some(CommandCompletion::Complete)
        );
        assert_eq!(
            complete_local_command("/tools search "),
            Some(CommandCompletion::Complete)
        );
        assert_eq!(
            complete_local_command("/tools s"),
            Some(CommandCompletion::Candidates(vec![
                "/tools search".to_string(),
                "/tools show".to_string(),
            ]))
        );
        assert_eq!(
            complete_local_command("/h"),
            Some(CommandCompletion::Candidates(vec![
                "/help".to_string(),
                "/history".to_string(),
            ]))
        );
        assert_eq!(
            complete_local_command("/history se"),
            Some(CommandCompletion::Replace("/history search ".to_string()))
        );
        assert_eq!(
            complete_local_command("/history fo"),
            Some(CommandCompletion::Replace("/history forget ".to_string()))
        );
        assert_eq!(
            complete_local_command("/u"),
            Some(CommandCompletion::Replace("/usage ".to_string()))
        );
        assert_eq!(
            complete_local_command("/usage r"),
            Some(CommandCompletion::Replace("/usage reset".to_string()))
        );
        assert_eq!(
            complete_local_command("/c"),
            Some(CommandCompletion::Candidates(vec![
                "/compact".to_string(),
                "/copy".to_string(),
                "/clear".to_string(),
            ]))
        );
        assert_eq!(complete_local_command("ordinary prompt"), None);
        assert_eq!(complete_local_command("/goal ship 🦀 now"), None);
        assert_eq!(complete_local_command("/tools search archive"), None);
        assert_eq!(complete_local_command("/save release checkpoint"), None);
        assert_eq!(complete_local_command("/load release checkpoint"), None);
        assert_eq!(complete_local_command("/missing"), None);
        assert_eq!(complete_local_command("/to\nnext"), None);
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
            for subcommand in spec.subcommands {
                let candidate = if subcommand.requires_value {
                    format!("{} {} value", spec.name, subcommand.name)
                } else {
                    format!("{} {}", spec.name, subcommand.name)
                };
                assert!(
                    !matches!(
                        parse_local_command(&candidate),
                        None | Some(LocalCommand::Unknown(_))
                    ),
                    "{} {} is completable but not parsed",
                    spec.name,
                    subcommand.name
                );
            }
        }
    }
}
