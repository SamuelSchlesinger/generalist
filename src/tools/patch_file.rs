use crate::sandbox::{ExecutionSandbox, SandboxAccess};
use crate::subprocess::{
    collect_bounded, combine_output, finish_output_collector, ProcessGroupGuard,
};
use crate::tools::bash::tail_truncate_with_spill;
use crate::{Error, Result, Tool};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::process::Stdio;
use tempfile::NamedTempFile;
use tokio::time::{timeout, Duration};

const MAX_PATCH_BYTES: usize = 1024 * 1024;
const PATCH_TIMEOUT_SECS: u64 = 30;

pub struct PatchFileTool;

pub struct SandboxedPatchFileTool {
    sandbox: ExecutionSandbox,
}

impl SandboxedPatchFileTool {
    pub fn new(sandbox: ExecutionSandbox) -> Self {
        Self { sandbox }
    }
}

fn description() -> &'static str {
    "Apply a unified diff to one existing file. The diff and target are shown for approval before execution; input, runtime, and output are bounded."
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The existing regular file to patch"},
            "diff": {"type": "string", "description": "A unified diff for that file (maximum 1 MiB)"}
        },
        "required": ["path", "diff"],
        "additionalProperties": false
    })
}

#[async_trait]
impl Tool for PatchFileTool {
    fn name(&self) -> &str {
        "patch_file"
    }

    fn description(&self) -> &str {
        description()
    }

    fn input_schema(&self) -> Value {
        schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        execute_patch(input, &ExecutionSandbox::unrestricted()).await
    }
}

#[async_trait]
impl Tool for SandboxedPatchFileTool {
    fn name(&self) -> &str {
        "patch_file"
    }

    fn description(&self) -> &str {
        description()
    }

    fn input_schema(&self) -> Value {
        schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        execute_patch(input, &self.sandbox).await
    }
}

fn patch_tempfile(sandbox: &ExecutionSandbox) -> std::io::Result<NamedTempFile> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("generalist-patch-").suffix(".diff");
    match sandbox.scratch_directory() {
        Some(directory) => builder.tempfile_in(directory),
        None => builder.tempfile(),
    }
}

async fn execute_patch(input: Value, sandbox: &ExecutionSandbox) -> Result<String> {
    let path = input
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Tool("Missing 'path' field".to_string()))?;
    let diff = input
        .get("diff")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Tool("Missing 'diff' field".to_string()))?;
    if diff.len() > MAX_PATCH_BYTES {
        return Err(Error::Tool(format!(
            "Patch exceeds the {MAX_PATCH_BYTES} byte input limit"
        )));
    }

    // Canonicalization prevents a leading '-' from becoming an option. The
    // OS policy remains the authoritative boundary if the path is raced.
    let target = std::fs::canonicalize(path)
        .map_err(|error| Error::Tool(format!("Failed to resolve patch target: {error}")))?;
    if !std::fs::metadata(&target)
        .map_err(|error| Error::Tool(format!("Failed to inspect patch target: {error}")))?
        .is_file()
    {
        return Err(Error::Tool(
            "patch_file only accepts an existing regular file".to_string(),
        ));
    }
    if let Some(workspace) = sandbox.workspace_root() {
        if !target.starts_with(workspace) {
            return Err(Error::Tool(format!(
                "Safe mode can patch only files inside {}",
                workspace.display()
            )));
        }
    }

    let mut patch = patch_tempfile(sandbox)
        .map_err(|error| Error::Tool(format!("Failed to create patch input: {error}")))?;
    patch
        .write_all(diff.as_bytes())
        .and_then(|_| patch.flush())
        .map_err(|error| Error::Tool(format!("Failed to write patch input: {error}")))?;

    let spill_directory = sandbox.scratch_directory().map(Path::to_path_buf);
    let mut access = SandboxAccess::default();
    access.allow_read_only(patch.path());
    let mut prepared = sandbox.prepare_command("patch", &access)?;
    let mut child = prepared
        .command_mut()
        .args(["-u", "-f", "-i"])
        .arg(patch.path())
        .arg(&target)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| Error::Tool(format!("Failed to start patch: {error}")))?;
    let mut group = ProcessGroupGuard::new(child.id());
    let stdout = tokio::spawn(collect_bounded(
        child.stdout.take().expect("stdout piped"),
        "generalist-patch-stdout-",
        spill_directory.clone(),
    ));
    let stderr = tokio::spawn(collect_bounded(
        child.stderr.take().expect("stderr piped"),
        "generalist-patch-stderr-",
        spill_directory.clone(),
    ));

    let status = match timeout(Duration::from_secs(PATCH_TIMEOUT_SECS), child.wait()).await {
        Ok(status) => status,
        Err(_) => {
            group.kill();
            let _ = child.start_kill();
            let _ = child.wait().await;
            stdout.abort();
            stderr.abort();
            return Err(Error::Tool(format!(
                "Patch timed out after {PATCH_TIMEOUT_SECS} seconds and was killed"
            )));
        }
    };
    group.kill();
    let (stdout, stderr) = tokio::join!(
        finish_output_collector(stdout),
        finish_output_collector(stderr)
    );
    let status =
        status.map_err(|error| Error::Tool(format!("Failed to wait for patch: {error}")))?;
    let output = tail_truncate_with_spill(
        &combine_output(status, &stdout, &stderr),
        spill_directory.as_deref(),
    );
    if !status.success() {
        return Err(Error::Tool(format!("Failed to apply patch: {output}")));
    }
    Ok(format!("Successfully patched {path}: {}", output.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIFF: &str = "--- before\n+++ after\n@@ -1 +1 @@\n-old\n+new\n";

    #[tokio::test]
    async fn patch_input_is_bounded() {
        let file = NamedTempFile::new().unwrap();
        let error = PatchFileTool
            .execute(json!({"path": file.path(), "diff": "x".repeat(MAX_PATCH_BYTES + 1)}))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("input limit"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn safe_patch_is_workspace_scoped() {
        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        let inside = workspace.join("inside.txt");
        let outside = parent.path().join("outside.txt");
        std::fs::write(&inside, "old\n").unwrap();
        std::fs::write(&outside, "old\n").unwrap();
        let tool = SandboxedPatchFileTool::new(ExecutionSandbox::restricted(&workspace).unwrap());

        tool.execute(json!({"path": &inside, "diff": DIFF}))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&inside).unwrap(), "new\n");
        assert!(tool
            .execute(json!({"path": &outside, "diff": DIFF}))
            .await
            .unwrap_err()
            .to_string()
            .contains("Safe mode can patch only"));
        assert_eq!(std::fs::read_to_string(&outside).unwrap(), "old\n");
    }
}
