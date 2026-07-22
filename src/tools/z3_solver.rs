use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Z3 SMT/SAT solver tool. Shells out to the `z3` CLI (must be installed).
pub struct Z3SolverTool;

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

#[async_trait]
impl Tool for Z3SolverTool {
    fn name(&self) -> &str {
        "z3_solver"
    }

    fn description(&self) -> &str {
        "Z3 SMT/SAT constraint solver. Use for logical constraints, satisfiability checks, \
         optimization under constraints, and proving/disproving implications. Constraints are \
         written in SMT-LIB s-expression syntax, e.g. '(> x 0)' or '(= (+ x y) 10)'."
    }

    fn input_schema(&self) -> Value {
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

    async fn execute(&self, input: Value) -> Result<String> {
        let params: Z3Input = serde_json::from_value(input)
            .map_err(|e| Error::Tool(format!("Invalid input parameters: {}", e)))?;

        let action = params.action.clone().unwrap_or_else(|| "solve".to_string());
        let timeout_ms = params.timeout.unwrap_or(5000).min(60_000);

        let result = tokio::task::spawn_blocking(move || -> Result<Z3Response> {
            let program = match action.as_str() {
                "solve" | "check_sat" => build_program(&params, false, false)?,
                "optimize" => build_program(&params, true, false)?,
                "prove" => build_program(&params, false, true)?,
                other => return Err(Error::Tool(format!("Unknown action: {}", other))),
            };
            let output = run_z3(&program, timeout_ms)?;
            Ok(interpret(&action, &output))
        })
        .await
        .map_err(|e| Error::Tool(format!("Task join error: {}", e)))??;

        serde_json::to_string_pretty(&result)
            .map_err(|e| Error::Tool(format!("Failed to serialize response: {}", e)))
    }
}

fn validate_sexpr(kind: &str, expr: &str) -> Result<String> {
    let trimmed = expr.trim();
    if !trimmed.starts_with('(') || !trimmed.ends_with(')') {
        return Err(Error::Tool(format!(
            "{} must be an SMT-LIB s-expression (enclosed in parentheses): {}",
            kind, expr
        )));
    }
    Ok(trimmed.to_string())
}

/// Build a complete SMT-LIB program for solve/optimize/prove.
fn build_program(params: &Z3Input, optimize: bool, prove: bool) -> Result<String> {
    let mut program = String::new();

    if let Some(logic) = &params.logic {
        program.push_str(&format!("(set-logic {})\n", logic));
    }

    if let Some(variables) = &params.variables {
        // Sort for deterministic output (HashMap order is random).
        let mut vars: Vec<_> = variables.iter().collect();
        vars.sort();
        for (name, var_type) in vars {
            match var_type.as_str() {
                "Bool" | "Int" | "Real" => {
                    program.push_str(&format!("(declare-const {} {})\n", name, var_type))
                }
                other => return Err(Error::Tool(format!("Unsupported variable type: {}", other))),
            }
        }
    }

    if prove {
        if let Some(hypotheses) = &params.hypothesis {
            for h in hypotheses {
                program.push_str(&format!("(assert {})\n", validate_sexpr("Hypothesis", h)?));
            }
        }
    }

    if let Some(constraints) = &params.constraints {
        for c in constraints {
            program.push_str(&format!("(assert {})\n", validate_sexpr("Constraint", c)?));
        }
    }

    if prove {
        let conclusion = params
            .conclusion
            .as_ref()
            .ok_or_else(|| Error::Tool("Conclusion is required for proof".to_string()))?;
        // Prove by refuting the negation.
        program.push_str(&format!(
            "(assert (not {}))\n",
            validate_sexpr("Conclusion", conclusion)?
        ));
    }

    if optimize {
        if let Some(objectives) = &params.optimize {
            let mut objs: Vec<_> = objectives.iter().collect();
            objs.sort();
            for (var, direction) in objs {
                match direction.as_str() {
                    "minimize" => program.push_str(&format!("(minimize {})\n", var)),
                    "maximize" => program.push_str(&format!("(maximize {})\n", var)),
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

    program.push_str("(check-sat)\n(get-model)\n");
    Ok(program)
}

fn run_z3(program: &str, timeout_ms: u64) -> Result<String> {
    use std::io::Write;
    use std::process::Command;

    let mut temp = tempfile::Builder::new()
        .prefix("generalist-z3-")
        .suffix(".smt2")
        .tempfile()
        .map_err(|e| Error::Tool(format!("Failed to create temp file: {}", e)))?;
    temp.write_all(program.as_bytes())
        .and_then(|_| temp.flush())
        .map_err(|e| Error::Tool(format!("Failed to write SMT program: {}", e)))?;

    let timeout_secs = timeout_ms.div_ceil(1000).max(1);
    let output = Command::new("z3")
        .arg(format!("-T:{}", timeout_secs))
        .arg(temp.path())
        .output()
        .map_err(|e| Error::Tool(format!("Failed to start Z3: {}. Is Z3 installed?", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = if stderr.is_empty() {
        stdout.to_string()
    } else {
        format!("{}\nSTDERR:\n{}", stdout, stderr)
    };

    if combined.trim().is_empty() && !output.status.success() {
        return Err(Error::Tool(format!(
            "Z3 execution failed with no output. Exit code: {}",
            output.status.code().unwrap_or(-1)
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
}
