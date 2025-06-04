use crate::{Tool, Result, Error};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Z3 SMT/SAT solver tool for constraint solving and verification
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
    execution_time_ms: u64,
    solver_info: HashMap<String, String>,
    z3_output: Option<String>,
}

#[async_trait]
impl Tool for Z3SolverTool {
    fn name(&self) -> &str {
        "z3_solver"
    }
    
    fn description(&self) -> &str {
        "Z3 SMT/SAT constraint solver for logical reasoning, optimization, and verification. Can solve boolean satisfiability, integer/real arithmetic, and constraint optimization problems."
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
                    "items": {
                        "type": "string"
                    },
                    "description": "List of constraints in SMT-LIB format"
                },
                "goal": {
                    "type": "string",
                    "enum": ["satisfiable", "unsatisfiable", "unknown"],
                    "description": "Expected satisfiability result (optional)"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60000,
                    "description": "Timeout in milliseconds (default: 5000, max: 60000)"
                },
                "logic": {
                    "type": "string",
                    "description": "SMT-LIB logic (e.g., QF_LIA, QF_LRA, QF_BOOL)"
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
                    "description": "Hypotheses/premises for proof (for 'prove' action)"
                },
                "conclusion": {
                    "type": "string",
                    "description": "Conclusion to prove (for 'prove' action)"
                }
            },
            "required": [],
            "additionalProperties": false
        })
    }
    
    async fn execute(&self, input: Value) -> Result<String> {
        let start_time = std::time::Instant::now();
        
        let params: Z3Input = serde_json::from_value(input)
            .map_err(|e| Error::Other(format!(
                "Invalid input parameters: {}. Example: {{\"constraints\": [\"(assert (> x 0))\"], \"variables\": {{\"x\": \"Int\"}}}}", e
            )))?;
        
        let action = params.action.clone().unwrap_or_else(|| "solve".to_string());
        let timeout = params.timeout.unwrap_or(5000).min(60000);
        
        // Use Z3 command-line interface for simplicity and thread safety
        let result = tokio::task::spawn_blocking(move || -> Result<Z3Response> {
            match action.as_str() {
                "solve" | "check_sat" => {
                    Self::solve_with_z3_cli(&params, timeout)
                }
                "optimize" => {
                    Self::optimize_with_z3_cli(&params, timeout)
                }
                "prove" => {
                    Self::prove_with_z3_cli(&params, timeout)
                }
                _ => Err(Error::Other(format!("Unknown action: {}", action)))
            }
        }).await.map_err(|e| Error::Other(format!("Task join error: {}", e)))??;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        let mut response = result;
        response.execution_time_ms = execution_time;
        
        serde_json::to_string_pretty(&response)
            .map_err(|e| Error::Other(format!("Failed to serialize response: {}", e)))
    }
}

impl Z3SolverTool {
    fn solve_with_z3_cli(params: &Z3Input, timeout: u64) -> Result<Z3Response> {
        let smt_program = Self::build_smt_program(params)?;
        let output = Self::run_z3(&smt_program, timeout)?;
        
        let satisfiable = output.contains("sat") && !output.contains("unsat");
        let result = if satisfiable {
            "satisfiable".to_string()
        } else if output.contains("unsat") {
            "unsatisfiable".to_string()
        } else {
            "unknown".to_string()
        };
        
        // Extract model if available
        let model = if satisfiable {
            Self::extract_model(&output)
        } else {
            None
        };
        
        let mut solver_info = HashMap::new();
        solver_info.insert("version".to_string(), "Z3 CLI".to_string());
        solver_info.insert("logic".to_string(), params.logic.clone().unwrap_or("AUTO".to_string()));
        
        Ok(Z3Response {
            action: "solve".to_string(),
            result,
            satisfiable,
            model,
            execution_time_ms: 0, // Will be set by caller
            solver_info,
            z3_output: Some(output),
        })
    }
    
    fn optimize_with_z3_cli(params: &Z3Input, timeout: u64) -> Result<Z3Response> {
        let smt_program = Self::build_optimization_program(params)?;
        let output = Self::run_z3(&smt_program, timeout)?;
        
        let satisfiable = output.contains("sat") && !output.contains("unsat");
        let result = if satisfiable {
            "optimal".to_string()
        } else if output.contains("unsat") {
            "unsatisfiable".to_string()
        } else {
            "unknown".to_string()
        };
        
        let model = if satisfiable {
            Self::extract_model(&output)
        } else {
            None
        };
        
        let mut solver_info = HashMap::new();
        solver_info.insert("version".to_string(), "Z3 Optimize".to_string());
        solver_info.insert("logic".to_string(), params.logic.clone().unwrap_or("AUTO".to_string()));
        
        Ok(Z3Response {
            action: "optimize".to_string(),
            result,
            satisfiable,
            model,
            execution_time_ms: 0,
            solver_info,
            z3_output: Some(output),
        })
    }
    
    fn prove_with_z3_cli(params: &Z3Input, timeout: u64) -> Result<Z3Response> {
        let smt_program = Self::build_proof_program(params)?;
        let output = Self::run_z3(&smt_program, timeout)?;
        
        // For proofs, unsat means theorem is proven
        let theorem_proven = output.contains("unsat");
        let result = if theorem_proven {
            "theorem_proven".to_string()
        } else if output.contains("sat") {
            "theorem_disproven".to_string()
        } else {
            "unknown".to_string()
        };
        
        // If theorem is disproven, show counterexample
        let model = if output.contains("sat") {
            Self::extract_model(&output)
        } else {
            None
        };
        
        let mut solver_info = HashMap::new();
        solver_info.insert("version".to_string(), "Z3 Theorem Prover".to_string());
        solver_info.insert("method".to_string(), "negation_satisfiability".to_string());
        
        Ok(Z3Response {
            action: "prove".to_string(),
            result,
            satisfiable: theorem_proven,
            model,
            execution_time_ms: 0,
            solver_info,
            z3_output: Some(output),
        })
    }
    
    fn build_smt_program(params: &Z3Input) -> Result<String> {
        let mut program = String::new();
        
        // Set logic
        if let Some(logic) = &params.logic {
            program.push_str(&format!("(set-logic {})\n", logic));
        }
        
        // Declare variables
        if let Some(variables) = &params.variables {
            for (name, var_type) in variables {
                let smt_type = match var_type.as_str() {
                    "Bool" => "Bool",
                    "Int" => "Int", 
                    "Real" => "Real",
                    _ => return Err(Error::Other(format!("Unsupported variable type: {}", var_type)))
                };
                program.push_str(&format!("(declare-const {} {})\n", name, smt_type));
            }
        }
        
        // Add constraints
        if let Some(constraints) = &params.constraints {
            for constraint in constraints {
                // Handle simple constraint formats and convert to SMT-LIB
                let smt_constraint = Self::convert_to_smt_lib(constraint)?;
                program.push_str(&format!("(assert {})\n", smt_constraint));
            }
        }
        
        program.push_str("(check-sat)\n");
        program.push_str("(get-model)\n");
        
        Ok(program)
    }
    
    fn build_optimization_program(params: &Z3Input) -> Result<String> {
        let mut program = String::new();
        
        // Set logic
        if let Some(logic) = &params.logic {
            program.push_str(&format!("(set-logic {})\n", logic));
        }
        
        // Declare variables
        if let Some(variables) = &params.variables {
            for (name, var_type) in variables {
                let smt_type = match var_type.as_str() {
                    "Bool" => "Bool",
                    "Int" => "Int",
                    "Real" => "Real", 
                    _ => return Err(Error::Other(format!("Unsupported variable type: {}", var_type)))
                };
                program.push_str(&format!("(declare-const {} {})\n", name, smt_type));
            }
        }
        
        // Add constraints
        if let Some(constraints) = &params.constraints {
            for constraint in constraints {
                let smt_constraint = Self::convert_to_smt_lib(constraint)?;
                program.push_str(&format!("(assert {})\n", smt_constraint));
            }
        }
        
        // Add optimization objectives
        if let Some(objectives) = &params.optimize {
            for (var_name, direction) in objectives {
                match direction.as_str() {
                    "minimize" => program.push_str(&format!("(minimize {})\n", var_name)),
                    "maximize" => program.push_str(&format!("(maximize {})\n", var_name)),
                    _ => return Err(Error::Other(format!("Invalid optimization direction: {}", direction)))
                }
            }
        }
        
        program.push_str("(check-sat)\n");
        program.push_str("(get-model)\n");
        
        Ok(program)
    }
    
    fn build_proof_program(params: &Z3Input) -> Result<String> {
        let mut program = String::new();
        
        // Set logic
        if let Some(logic) = &params.logic {
            program.push_str(&format!("(set-logic {})\n", logic));
        }
        
        // Declare variables
        if let Some(variables) = &params.variables {
            for (name, var_type) in variables {
                let smt_type = match var_type.as_str() {
                    "Bool" => "Bool",
                    "Int" => "Int",
                    "Real" => "Real",
                    _ => return Err(Error::Other(format!("Unsupported variable type: {}", var_type)))
                };
                program.push_str(&format!("(declare-const {} {})\n", name, smt_type));
            }
        }
        
        // Add hypotheses
        if let Some(hypotheses) = &params.hypothesis {
            for hypothesis in hypotheses {
                let smt_constraint = Self::convert_to_smt_lib(hypothesis)?;
                program.push_str(&format!("(assert {})\n", smt_constraint));
            }
        }
        
        // Add general constraints
        if let Some(constraints) = &params.constraints {
            for constraint in constraints {
                let smt_constraint = Self::convert_to_smt_lib(constraint)?;
                program.push_str(&format!("(assert {})\n", smt_constraint));
            }
        }
        
        // Add negation of conclusion
        if let Some(conclusion) = &params.conclusion {
            let smt_conclusion = Self::convert_to_smt_lib(conclusion)?;
            program.push_str(&format!("(assert (not {}))\n", smt_conclusion));
        } else {
            return Err(Error::Other("Conclusion is required for proof".to_string()));
        }
        
        program.push_str("(check-sat)\n");
        program.push_str("(get-model)\n");
        
        Ok(program)
    }
    
    fn convert_to_smt_lib(constraint: &str) -> Result<String> {
        let constraint = constraint.trim();
        
        // If already in SMT-LIB format (starts with parentheses), return as-is
        if constraint.starts_with('(') && constraint.ends_with(')') {
            return Ok(constraint.to_string());
        }
        
        // Convert simple infix notation to SMT-LIB
        // Handle equality: "x + y == 10" -> "(= (+ x y) 10)"
        if let Some(eq_pos) = constraint.find("==") {
            let left = constraint[..eq_pos].trim();
            let right = constraint[eq_pos + 2..].trim();
            return Ok(format!("(= {} {})", 
                Self::convert_expression_to_smt(left)?, 
                Self::convert_expression_to_smt(right)?));
        }
        
        // Handle inequalities
        for (op, smt_op) in [(">=", ">="), ("<=", "<="), (">", ">"), ("<", "<")] {
            if let Some(op_pos) = constraint.find(op) {
                let left = constraint[..op_pos].trim();
                let right = constraint[op_pos + op.len()..].trim();
                return Ok(format!("({} {} {})", 
                    smt_op,
                    Self::convert_expression_to_smt(left)?, 
                    Self::convert_expression_to_smt(right)?));
            }
        }
        
        // Handle boolean values
        if constraint == "true" {
            return Ok("true".to_string());
        }
        if constraint == "false" {
            return Ok("false".to_string());
        }
        
        // If it's a simple variable or number, return as-is
        Ok(constraint.to_string())
    }
    
    fn convert_expression_to_smt(expr: &str) -> Result<String> {
        let expr = expr.trim();
        
        // Handle numbers
        if expr.parse::<i64>().is_ok() || expr.parse::<f64>().is_ok() {
            return Ok(expr.to_string());
        }
        
        // Handle simple addition: "x + y" -> "(+ x y)"
        if let Some(plus_pos) = expr.find(" + ") {
            let left = expr[..plus_pos].trim();
            let right = expr[plus_pos + 3..].trim();
            return Ok(format!("(+ {} {})", 
                Self::convert_expression_to_smt(left)?, 
                Self::convert_expression_to_smt(right)?));
        }
        
        // Handle simple subtraction: "x - y" -> "(- x y)"
        if let Some(minus_pos) = expr.find(" - ") {
            let left = expr[..minus_pos].trim();
            let right = expr[minus_pos + 3..].trim();
            return Ok(format!("(- {} {})", 
                Self::convert_expression_to_smt(left)?, 
                Self::convert_expression_to_smt(right)?));
        }
        
        // Handle simple multiplication: "x * y" -> "(* x y)"
        if let Some(mult_pos) = expr.find(" * ") {
            let left = expr[..mult_pos].trim();
            let right = expr[mult_pos + 3..].trim();
            return Ok(format!("(* {} {})", 
                Self::convert_expression_to_smt(left)?, 
                Self::convert_expression_to_smt(right)?));
        }
        
        // Otherwise assume it's a variable
        Ok(expr.to_string())
    }
    
    fn run_z3(program: &str, timeout: u64) -> Result<String> {
        use std::process::{Command, Stdio};
        use std::fs;
        
        // Write program to temporary file since Z3 -in flag doesn't work as expected
        let temp_file = format!("/tmp/z3_input_{}.smt2", std::process::id());
        fs::write(&temp_file, program)
            .map_err(|e| Error::Other(format!("Failed to write temporary file: {}", e)))?;
        
        let mut cmd = Command::new("z3");
        cmd.arg(&temp_file);
        
        if timeout > 0 {
            cmd.arg(format!("-T:{}", timeout / 1000)); // Z3 timeout in seconds
        }
        
        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| Error::Other(format!("Failed to start Z3: {}. Make sure Z3 is installed.", e)))?;
        
        // Clean up temp file
        let _ = fs::remove_file(&temp_file);
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Z3 might return success even with some errors in stderr, so combine both
        let combined_output = if stderr.is_empty() {
            stdout.to_string()
        } else {
            format!("{}\nSTDERR:\n{}", stdout, stderr)
        };
        
        // Don't fail on non-zero exit code if we got some output, as Z3 might return
        // error codes for logic issues rather than execution failures
        if combined_output.trim().is_empty() && !output.status.success() {
            return Err(Error::Other(format!("Z3 execution failed with no output. Exit code: {}", 
                output.status.code().unwrap_or(-1))));
        }
        
        Ok(combined_output)
    }
    
    fn extract_model(output: &str) -> Option<HashMap<String, String>> {
        let mut model = HashMap::new();
        let lines: Vec<&str> = output.lines().collect();
        
        for line in lines {
            if line.trim().starts_with("(define-fun ") {
                // Parse Z3 model output: "(define-fun x () Int 5)"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let var_name = parts[1].to_string();
                    let value = parts[4].trim_end_matches(')').to_string();
                    model.insert(var_name, value);
                }
            }
        }
        
        if model.is_empty() {
            None
        } else {
            Some(model)
        }
    }
}