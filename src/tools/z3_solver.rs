use crate::sandbox::{ExecutionSandbox, SandboxAccess};
use crate::subprocess::{
    collect_bounded, combine_output, finish_output_collector, ProcessGroupGuard,
};
use crate::tools::bash::tail_truncate_with_spill;
use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use tokio::time::{timeout, Duration};

/// Z3 SMT/SAT solver tool. Shells out to the `z3` CLI (must be installed).
pub struct Z3SolverTool;

/// Z3 constrained by the same execution boundary as shell and code mode.
pub struct SandboxedZ3SolverTool {
    sandbox: ExecutionSandbox,
}

impl SandboxedZ3SolverTool {
    pub fn new(sandbox: ExecutionSandbox) -> Self {
        Self { sandbox }
    }
}

const MAX_SMT_PROGRAM_BYTES: usize = 1024 * 1024;
const Z3_TIMEOUT_GRACE_MILLIS: u64 = 2_000;

#[derive(Debug, Deserialize, Clone)]
struct Z3Input {
    action: Option<String>,
    variables: Option<HashMap<String, String>>,
    constraints: Option<Vec<String>>,
    timeout: Option<u64>,
    logic: Option<String>,
    optimize: Option<HashMap<String, String>>,
    hypothesis: Option<Vec<String>>,
    conclusion: Option<String>,
}

#[derive(Debug, Serialize)]
struct Z3Response {
    action: String,
    result: String,
    satisfiable: bool,
    model: Option<HashMap<String, String>>,
    z3_output: String,
}

fn z3_description() -> &'static str {
    "Z3 SMT/SAT constraint solver. Use for logical constraints, satisfiability checks, \
         optimization under constraints, and proving/disproving implications. Constraints are \
         written in SMT-LIB s-expression syntax, e.g. '(> x 0)' or '(= (+ x y) 10)'."
}

fn z3_input_schema() -> Value {
    json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["solve", "optimize", "check_sat", "prove"],
                    "description": "Action to perform (default: solve)"
                },
                "variables": {
                    "type": "object",
                    "description": "Variable declarations as name->type pairs",
                    "additionalProperties": {
                        "type": "string",
                        "enum": ["Bool", "Int", "Real"]
                    }
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Constraints in SMT-LIB format, e.g. '(> x 0)'"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60000,
                    "description": "Timeout in milliseconds (default: 5000)"
                },
                "logic": {
                    "type": "string",
                    "description": "SMT-LIB logic (e.g. QF_LIA, QF_LRA)"
                },
                "optimize": {
                    "type": "object",
                    "description": "Optimization objectives as variable->direction pairs",
                    "additionalProperties": {
                        "type": "string",
                        "enum": ["minimize", "maximize"]
                    }
                },
                "hypothesis": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Premises in SMT-LIB format (for 'prove')"
                },
                "conclusion": {
                    "type": "string",
                    "description": "Conclusion in SMT-LIB format to prove (for 'prove')"
                }
            },
            "required": [],
            "additionalProperties": false
    })
}

#[async_trait]
impl Tool for Z3SolverTool {
    fn name(&self) -> &str {
        "z3_solver"
    }

    fn description(&self) -> &str {
        z3_description()
    }

    fn input_schema(&self) -> Value {
        z3_input_schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        execute_z3(input, &ExecutionSandbox::unrestricted()).await
    }
}

#[async_trait]
impl Tool for SandboxedZ3SolverTool {
    fn name(&self) -> &str {
        "z3_solver"
    }

    fn description(&self) -> &str {
        z3_description()
    }

    fn input_schema(&self) -> Value {
        z3_input_schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        execute_z3(input, &self.sandbox).await
    }
}

async fn execute_z3(input: Value, sandbox: &ExecutionSandbox) -> Result<String> {
    let params: Z3Input = serde_json::from_value(input)
        .map_err(|e| Error::Tool(format!("Invalid input parameters: {e}")))?;
    let action = params.action.clone().unwrap_or_else(|| "solve".to_string());
    let timeout_ms = params.timeout.unwrap_or(5_000).clamp(1, 60_000);
    let program = match action.as_str() {
        "solve" | "check_sat" => build_program(&params, false, false)?,
        "optimize" => build_program(&params, true, false)?,
        "prove" => build_program(&params, false, true)?,
        other => return Err(Error::Tool(format!("Unknown action: {other}"))),
    };
    if program.len() > MAX_SMT_PROGRAM_BYTES {
        return Err(Error::Tool(format!(
            "SMT program exceeds the {MAX_SMT_PROGRAM_BYTES} byte input limit"
        )));
    }
    let output = run_z3(&program, timeout_ms, sandbox).await?;
    serde_json::to_string_pretty(&interpret(&action, &output))
        .map_err(|e| Error::Tool(format!("Failed to serialize response: {e}")))
}

fn validate_simple_symbol<'a>(kind: &str, symbol: &'a str) -> Result<&'a str> {
    let mut characters = symbol.chars();
    let valid = characters
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        });
    if !valid {
        return Err(Error::Tool(format!(
            "{kind} must be a simple SMT-LIB identifier: {symbol}"
        )));
    }
    Ok(symbol)
}

/// Accept exactly one balanced SMT-LIB s-expression.
///
/// Merely checking the first/last byte lets a value close the host's `assert`
/// and inject top-level commands such as `include` or `reset`. Track strings,
/// quoted symbols, and comments so parentheses in data do not confuse the
/// boundary check.
fn validate_sexpr<'a>(kind: &str, expr: &'a str) -> Result<&'a str> {
    let trimmed = expr.trim();
    if trimmed.len() > MAX_SMT_PROGRAM_BYTES || !trimmed.starts_with('(') {
        return Err(Error::Tool(format!(
            "{kind} must be one bounded SMT-LIB s-expression: {expr}"
        )));
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ParseState {
        Normal,
        String,
        QuotedSymbol,
        Comment,
    }

    let bytes = trimmed.as_bytes();
    let mut state = ParseState::Normal;
    let mut depth = 0usize;
    let mut complete = false;
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        match state {
            ParseState::Normal => {
                if complete {
                    if !byte.is_ascii_whitespace() {
                        return Err(Error::Tool(format!(
                            "{kind} must contain exactly one SMT-LIB s-expression"
                        )));
                    }
                } else {
                    match byte {
                        b'(' => {
                            depth = depth.checked_add(1).ok_or_else(|| {
                                Error::Tool(format!("{kind} nesting depth overflowed"))
                            })?;
                        }
                        b')' => {
                            if depth == 0 {
                                return Err(Error::Tool(format!(
                                    "{kind} has an unmatched closing parenthesis"
                                )));
                            }
                            depth -= 1;
                            complete = depth == 0;
                        }
                        b'"' => state = ParseState::String,
                        b'|' => state = ParseState::QuotedSymbol,
                        b';' => state = ParseState::Comment,
                        _ => {}
                    }
                }
            }
            ParseState::String => match byte {
                // Z3 accepts backslash escapes in addition to SMT-LIB's
                // doubled-quote spelling. Either remains within the string.
                b'\\' if index + 1 < bytes.len() => index += 1,
                b'"' if index + 1 < bytes.len() && bytes[index + 1] == b'"' => index += 1,
                b'"' => state = ParseState::Normal,
                _ => {}
            },
            ParseState::QuotedSymbol => match byte {
                b'\\' if index + 1 < bytes.len() => index += 1,
                b'|' => state = ParseState::Normal,
                _ => {}
            },
            ParseState::Comment => {
                if byte == b'\n' || byte == b'\r' {
                    state = ParseState::Normal;
                }
            }
        }
        index += 1;
    }
    if !complete || depth != 0 || state != ParseState::Normal {
        return Err(Error::Tool(format!(
            "{kind} must be one balanced SMT-LIB s-expression"
        )));
    }
    Ok(trimmed)
}

fn push_program(program: &mut String, fragment: &str) -> Result<()> {
    let next = program
        .len()
        .checked_add(fragment.len())
        .ok_or_else(|| Error::Tool("SMT program length overflowed".to_string()))?;
    if next > MAX_SMT_PROGRAM_BYTES {
        return Err(Error::Tool(format!(
            "SMT program exceeds the {MAX_SMT_PROGRAM_BYTES} byte input limit"
        )));
    }
    program.push_str(fragment);
    Ok(())
}

/// Build a complete SMT-LIB program for solve/optimize/prove.
fn build_program(params: &Z3Input, optimize: bool, prove: bool) -> Result<String> {
    let mut program = String::new();

    if let Some(logic) = &params.logic {
        push_program(
            &mut program,
            &format!("(set-logic {})\n", validate_simple_symbol("Logic", logic)?),
        )?;
    }

    if let Some(variables) = &params.variables {
        // Sort for deterministic output (HashMap order is random).
        let mut vars: Vec<_> = variables.iter().collect();
        vars.sort();
        for (name, var_type) in vars {
            let name = validate_simple_symbol("Variable name", name)?;
            match var_type.as_str() {
                "Bool" | "Int" | "Real" => push_program(
                    &mut program,
                    &format!("(declare-const {name} {var_type})\n"),
                )?,
                other => return Err(Error::Tool(format!("Unsupported variable type: {}", other))),
            }
        }
    }

    if prove {
        if let Some(hypotheses) = &params.hypothesis {
            for h in hypotheses {
                push_program(
                    &mut program,
                    &format!("(assert {})\n", validate_sexpr("Hypothesis", h)?),
                )?;
            }
        }
    }

    if let Some(constraints) = &params.constraints {
        for c in constraints {
            push_program(
                &mut program,
                &format!("(assert {})\n", validate_sexpr("Constraint", c)?),
            )?;
        }
    }

    if prove {
        let conclusion = params
            .conclusion
            .as_ref()
            .ok_or_else(|| Error::Tool("Conclusion is required for proof".to_string()))?;
        // Prove by refuting the negation.
        push_program(
            &mut program,
            &format!(
                "(assert (not {}))\n",
                validate_sexpr("Conclusion", conclusion)?
            ),
        )?;
    }

    if optimize {
        if let Some(objectives) = &params.optimize {
            let mut objs: Vec<_> = objectives.iter().collect();
            objs.sort();
            for (var, direction) in objs {
                let var = validate_simple_symbol("Optimization variable", var)?;
                match direction.as_str() {
                    "minimize" => push_program(&mut program, &format!("(minimize {var})\n"))?,
                    "maximize" => push_program(&mut program, &format!("(maximize {var})\n"))?,
                    other => {
                        return Err(Error::Tool(format!(
                            "Invalid optimization direction: {}",
                            other
                        )))
                    }
                }
            }
        }
    }

    push_program(&mut program, "(check-sat)\n(get-model)\n")?;
    Ok(program)
}

async fn run_z3(program: &str, timeout_ms: u64, sandbox: &ExecutionSandbox) -> Result<String> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("generalist-z3-").suffix(".smt2");
    let mut temp = match sandbox.scratch_directory() {
        Some(directory) => builder.tempfile_in(directory),
        None => builder.tempfile(),
    }
    .map_err(|e| Error::Tool(format!("Failed to create SMT input: {e}")))?;
    temp.write_all(program.as_bytes())
        .and_then(|_| temp.flush())
        .map_err(|e| Error::Tool(format!("Failed to write SMT program: {e}")))?;

    let timeout_secs = timeout_ms.div_ceil(1000).max(1);
    let spill_directory = sandbox.scratch_directory().map(Path::to_path_buf);
    let mut access = SandboxAccess::default();
    access.allow_read_only(temp.path());
    let mut prepared = sandbox.prepare_command("z3", &access)?;
    let mut child = prepared
        .command_mut()
        .arg(format!("-T:{}", timeout_secs))
        .arg(temp.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| Error::Tool(format!("Failed to start Z3: {e}. Is Z3 installed?")))?;
    let mut process_group = ProcessGroupGuard::new(child.id());
    let stdout_task = tokio::spawn(collect_bounded(
        child.stdout.take().expect("stdout piped"),
        "generalist-z3-stdout-",
        spill_directory.clone(),
    ));
    let stderr_task = tokio::spawn(collect_bounded(
        child.stderr.take().expect("stderr piped"),
        "generalist-z3-stderr-",
        spill_directory.clone(),
    ));
    let outer_timeout = Duration::from_millis(timeout_ms.saturating_add(Z3_TIMEOUT_GRACE_MILLIS));
    let status = match timeout(outer_timeout, child.wait()).await {
        Ok(status) => status,
        Err(_) => {
            process_group.kill();
            let _ = child.start_kill();
            let _ = child.wait().await;
            stdout_task.abort();
            stderr_task.abort();
            return Err(Error::Tool(format!(
                "Z3 exceeded its {timeout_ms} ms timeout and was killed"
            )));
        }
    };
    process_group.kill();
    let (stdout, stderr) = tokio::join!(
        finish_output_collector(stdout_task),
        finish_output_collector(stderr_task)
    );
    let status = status.map_err(|e| Error::Tool(format!("Failed to wait for Z3: {e}")))?;
    let combined = combine_output(status, &stdout, &stderr);
    let combined = tail_truncate_with_spill(&combined, spill_directory.as_deref());

    if combined.trim().is_empty() && !status.success() {
        return Err(Error::Tool(format!(
            "Z3 execution failed with no output. Exit code: {}",
            status.code().unwrap_or(-1)
        )));
    }
    Ok(combined)
}

/// Read the verdict from Z3 output: the first line that is exactly
/// sat/unsat/unknown. Substring matching would misfire (e.g. "unsat" contains
/// "sat", and model variables can contain either word).
fn parse_verdict(output: &str) -> Option<&'static str> {
    for line in output.lines() {
        match line.trim() {
            "sat" => return Some("sat"),
            "unsat" => return Some("unsat"),
            "unknown" => return Some("unknown"),
            _ => {}
        }
    }
    None
}

fn interpret(action: &str, output: &str) -> Z3Response {
    let verdict = parse_verdict(output);
    let (result, satisfiable) = match (action, verdict) {
        ("prove", Some("unsat")) => ("theorem_proven".to_string(), true),
        ("prove", Some("sat")) => ("theorem_disproven_see_counterexample".to_string(), false),
        ("prove", _) => ("unknown".to_string(), false),
        ("optimize", Some("sat")) => ("optimal".to_string(), true),
        (_, Some("sat")) => ("satisfiable".to_string(), true),
        (_, Some("unsat")) => ("unsatisfiable".to_string(), false),
        (_, _) => ("unknown".to_string(), false),
    };

    let model = if verdict == Some("sat") {
        extract_model(output)
    } else {
        None
    };

    Z3Response {
        action: action.to_string(),
        result,
        satisfiable,
        model,
        z3_output: output.to_string(),
    }
}

/// Parse `(define-fun name () Type value)` entries from a Z3 model.
///
/// Z3 usually prints these across multiple lines, so scan for balanced
/// parentheses starting at each `(define-fun` and then split into tokens.
fn extract_model(output: &str) -> Option<HashMap<String, String>> {
    let mut model = HashMap::new();
    let mut search_from = 0;
    while let Some(pos) = output[search_from..].find("(define-fun") {
        let start = search_from + pos;
        let mut depth = 0usize;
        let mut end = None;
        for (i, c) in output[start..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(start + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };
        let tokens: Vec<&str> = output[start..end].split_whitespace().collect();
        // ["(define-fun", name, "()", type, value...]
        if tokens.len() >= 5 && tokens[2] == "()" {
            let name = tokens[1].to_string();
            let value = tokens[4..]
                .join(" ")
                .trim_end_matches(')')
                .trim()
                .to_string();
            model.insert(name, value);
        }
        search_from = end;
    }
    if model.is_empty() {
        None
    } else {
        Some(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_parsing_is_exact_line_match() {
        assert_eq!(parse_verdict("sat\n(model ...)"), Some("sat"));
        assert_eq!(parse_verdict("unsat\n"), Some("unsat"));
        // A variable named "sat_count" must not read as a verdict.
        assert_eq!(
            parse_verdict("(define-fun sat_count () Int 1)\nunknown"),
            Some("unknown")
        );
        assert_eq!(parse_verdict("garbage"), None);
    }

    #[test]
    fn extracts_multi_line_models() {
        let output =
            "sat\n(\n  (define-fun x () Int\n    5)\n  (define-fun y () Real\n    (/ 1.0 2.0))\n)";
        let model = extract_model(output).unwrap();
        assert_eq!(model.get("x").unwrap(), "5");
        assert!(model.get("y").unwrap().contains("1.0"));
    }

    #[test]
    fn builds_proof_program_with_negated_conclusion() {
        let params = Z3Input {
            action: Some("prove".into()),
            variables: Some([("x".to_string(), "Int".to_string())].into()),
            constraints: None,
            timeout: None,
            logic: None,
            optimize: None,
            hypothesis: Some(vec!["(> x 0)".into()]),
            conclusion: Some("(>= x 0)".into()),
        };
        let program = build_program(&params, false, true).unwrap();
        assert!(program.contains("(declare-const x Int)"));
        assert!(program.contains("(assert (> x 0))"));
        assert!(program.contains("(assert (not (>= x 0)))"));
        assert!(program.ends_with("(check-sat)\n(get-model)\n"));
    }

    #[test]
    fn rejects_non_sexpr_constraints() {
        let params = Z3Input {
            action: None,
            variables: None,
            constraints: Some(vec!["x > 0".into()]),
            timeout: None,
            logic: None,
            optimize: None,
            hypothesis: None,
            conclusion: None,
        };
        assert!(build_program(&params, false, false).is_err());
    }

    #[test]
    fn rejects_top_level_command_and_identifier_injection() {
        let mut params = Z3Input {
            action: None,
            variables: Some([("x".to_string(), "Int".to_string())].into()),
            constraints: Some(vec!["(> x 0)\n(reset)".into()]),
            timeout: None,
            logic: None,
            optimize: None,
            hypothesis: None,
            conclusion: None,
        };
        assert!(build_program(&params, false, false).is_err());

        params.constraints = Some(vec!["(= x \"a ) ; still a string\")".into()]);
        assert!(build_program(&params, false, false).is_ok());

        params.variables = Some(
            [(
                "x) (include \"/etc/passwd\") (declare-const y".to_string(),
                "Int".to_string(),
            )]
            .into(),
        );
        assert!(build_program(&params, false, false).is_err());

        params.variables = None;
        params.logic = Some("QF_LIA) (reset".into());
        assert!(build_program(&params, false, false).is_err());
    }
}
