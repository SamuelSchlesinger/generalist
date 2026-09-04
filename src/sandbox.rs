//! Fail-closed process isolation for model-authored shell and Python code.
//!
//! The CLI uses a restricted sandbox by default. Library callers keep the
//! historical unrestricted behavior unless they explicitly install one on a
//! [`crate::ToolRegistry`]. macOS uses Seatbelt through `sandbox-exec`; Linux
//! prefers native Landlock plus seccomp and falls back to Bubblewrap. A missing
//! backend is an error rather than an implicit escape from containment.

use crate::{Error, Result};
#[cfg(target_os = "linux")]
use landlock::{
    path_beneath_rules, Access, AccessFs, AccessNet, CompatLevel, Compatible, Ruleset, RulesetAttr,
    RulesetCreatedAttr, Scope, ABI,
};
use std::collections::BTreeSet;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{Seek, SeekFrom, Write};
#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd, OwnedFd};
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::sync::Arc;
use tempfile::TempDir;
use tokio::process::Command;

const SAFE_PATH_FALLBACK: &str = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";
/// A per-process descriptor ceiling keeps a model-authored fork from turning
/// one inherited soft limit (often hundreds of thousands) into an easy host
/// resource-exhaustion primitive. 1,024 remains comfortably above ordinary
/// compiler and interpreter requirements.
const SANDBOX_MAX_OPEN_FILES: libc::rlim_t = 1_024;

#[derive(Clone, Debug)]
pub struct ExecutionSandbox {
    mode: SandboxMode,
}

#[derive(Clone, Debug)]
enum SandboxMode {
    Unrestricted,
    Restricted(RestrictedSandbox),
}

#[derive(Clone, Debug)]
struct RestrictedSandbox {
    workspace_root: PathBuf,
    backend: SandboxBackend,
    scratch: Arc<TempDir>,
    scratch_path: PathBuf,
    environment: Arc<SafeEnvironment>,
}

#[derive(Clone, Debug)]
enum SandboxBackend {
    #[cfg(target_os = "macos")]
    Seatbelt { executable: PathBuf },
    #[cfg(target_os = "linux")]
    Landlock,
    #[cfg(target_os = "linux")]
    Bubblewrap {
        executable: PathBuf,
        native_userns_hardening: bool,
    },
}

/// Per-process exceptions to the common workspace policy.
#[derive(Clone, Debug, Default)]
pub(crate) struct SandboxAccess {
    read_only: Vec<PathBuf>,
}

impl SandboxAccess {
    pub(crate) fn allow_read_only(&mut self, path: impl Into<PathBuf>) {
        self.read_only.push(path.into());
    }
}

/// A command plus the private home/tmp directory its policy references.
/// Keep this value alive until the spawned child has exited.
pub(crate) struct PreparedSandboxCommand {
    command: Command,
    _scratch: Option<Arc<TempDir>>,
    #[cfg(target_os = "linux")]
    _landlock_ruleset: Option<OwnedFd>,
    #[cfg(target_os = "linux")]
    _seccomp_policy: Option<File>,
}

impl PreparedSandboxCommand {
    pub(crate) fn command_mut(&mut self) -> &mut Command {
        &mut self.command
    }
}

impl Default for ExecutionSandbox {
    fn default() -> Self {
        Self::unrestricted()
    }
}

impl ExecutionSandbox {
    /// Preserve the caller's ordinary process environment and host access.
    pub fn unrestricted() -> Self {
        Self {
            mode: SandboxMode::Unrestricted,
        }
    }

    /// Build a fail-closed workspace sandbox using the native backend.
    pub fn restricted(workspace_root: impl AsRef<Path>) -> Result<Self> {
        let workspace_root = canonical_directory(workspace_root.as_ref(), "workspace root")?;
        validate_workspace_boundary(&workspace_root)?;
        let backend = detect_backend()?;
        let scratch = Arc::new(create_private_scratch()?);
        let scratch_path = canonical_directory(scratch.path(), "sandbox scratch directory")?;
        let environment = Arc::new(SafeEnvironment::capture(&workspace_root, &scratch_path)?);
        let sandbox = Self {
            mode: SandboxMode::Restricted(RestrictedSandbox {
                workspace_root,
                backend,
                scratch,
                scratch_path,
                environment,
            }),
        };
        sandbox.validate_restricted_backend()?;
        Ok(sandbox)
    }

    pub fn is_restricted(&self) -> bool {
        matches!(self.mode, SandboxMode::Restricted(_))
    }

    pub fn workspace_root(&self) -> Option<&Path> {
        match &self.mode {
            SandboxMode::Unrestricted => None,
            SandboxMode::Restricted(sandbox) => Some(&sandbox.workspace_root),
        }
    }

    /// Private state shared by sandboxed subprocesses for this registry run.
    /// Host-side output spills live here so a later sandboxed command can read
    /// the disclosed path without exposing the host's ordinary temp directory.
    pub(crate) fn scratch_directory(&self) -> Option<&Path> {
        match &self.mode {
            SandboxMode::Unrestricted => None,
            SandboxMode::Restricted(sandbox) => Some(&sandbox.scratch_path),
        }
    }

    pub(crate) fn prepare_command(
        &self,
        program: impl AsRef<OsStr>,
        access: &SandboxAccess,
    ) -> Result<PreparedSandboxCommand> {
        let program = program.as_ref();
        let SandboxMode::Restricted(sandbox) = &self.mode else {
            return Ok(PreparedSandboxCommand {
                command: Command::new(program),
                _scratch: None,
                #[cfg(target_os = "linux")]
                _landlock_ruleset: None,
                #[cfg(target_os = "linux")]
                _seccomp_policy: None,
            });
        };

        let executable =
            resolve_executable_from(program, &sandbox.environment.executable_directories)?;
        let scratch_path = &sandbox.scratch_path;
        let environment = sandbox.environment.as_ref();

        let mut read_only = environment.read_roots.clone();
        read_only.push(executable.parent().unwrap_or(&executable).to_path_buf());
        for path in &access.read_only {
            read_only.push(canonical_existing(path, "sandbox read-only path")?);
        }
        let read_only = deduplicate_paths(read_only);
        match &sandbox.backend {
            #[cfg(target_os = "macos")]
            SandboxBackend::Seatbelt {
                executable: seatbelt,
            } => Ok(PreparedSandboxCommand {
                command: prepare_seatbelt_command(
                    seatbelt,
                    &executable,
                    &sandbox.workspace_root,
                    scratch_path,
                    &read_only,
                    environment,
                )?,
                _scratch: Some(Arc::clone(&sandbox.scratch)),
            }),
            #[cfg(target_os = "linux")]
            SandboxBackend::Landlock => {
                let (command, ruleset) = prepare_landlock_command(
                    &executable,
                    &sandbox.workspace_root,
                    scratch_path,
                    &read_only,
                    environment,
                )?;
                Ok(PreparedSandboxCommand {
                    command,
                    _scratch: Some(Arc::clone(&sandbox.scratch)),
                    _landlock_ruleset: Some(ruleset),
                    _seccomp_policy: None,
                })
            }
            #[cfg(target_os = "linux")]
            SandboxBackend::Bubblewrap {
                executable: bubblewrap,
                native_userns_hardening,
            } => {
                let (command, seccomp_policy) = prepare_bubblewrap_command(
                    bubblewrap,
                    *native_userns_hardening,
                    &executable,
                    &sandbox.workspace_root,
                    scratch_path,
                    &read_only,
                    environment,
                )?;
                Ok(PreparedSandboxCommand {
                    command,
                    _scratch: Some(Arc::clone(&sandbox.scratch)),
                    _landlock_ruleset: None,
                    _seccomp_policy: Some(seccomp_policy),
                })
            }
        }
    }

    fn validate_restricted_backend(&self) -> Result<()> {
        let mut prepared = self.prepare_command("/usr/bin/true", &SandboxAccess::default())?;
        let output = prepared
            .command
            .as_std_mut()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|error| {
                Error::Other(format!(
                    "Safe mode could not launch its restricted-process probe: {error}; use --yolo only if unrestricted execution is intended"
                ))
            })?;
        if output.status.success() {
            return Ok(());
        }
        let detail = String::from_utf8_lossy(&output.stderr);
        let detail = detail.trim();
        Err(Error::Other(if detail.is_empty() {
            format!(
                "Safe mode restricted-process probe failed (status {}); use --yolo only if unrestricted execution is intended",
                output.status
            )
        } else {
            format!(
                "Safe mode restricted-process probe failed (status {}): {detail}; use --yolo only if unrestricted execution is intended",
                output.status
            )
        }))
    }
}

fn validate_workspace_boundary(workspace: &Path) -> Result<()> {
    let broad_roots = [
        "/",
        "/Applications",
        "/Library",
        "/System",
        "/Users",
        "/bin",
        "/dev",
        "/etc",
        "/home",
        "/lib",
        "/lib64",
        "/opt",
        "/private",
        "/private/tmp",
        "/private/var",
        "/proc",
        "/run",
        "/sbin",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
    ];
    let unsafe_root = broad_roots.iter().any(|root| {
        fs::canonicalize(root)
            .map(|root| root == workspace)
            .unwrap_or(false)
    });
    let contains_home = env::var_os("HOME")
        .and_then(|home| fs::canonicalize(home).ok())
        .is_some_and(|home| home.starts_with(workspace));
    if unsafe_root || contains_home {
        return Err(Error::Other(format!(
            "Safe mode refuses the overly broad workspace {} because it would expose system or home credentials. Start Generalist inside a dedicated project directory, or use --yolo only if unrestricted execution is intended",
            workspace.display()
        )));
    }
    Ok(())
}

fn canonical_existing(path: &Path, label: &str) -> Result<PathBuf> {
    fs::canonicalize(path).map_err(|error| {
        Error::Other(format!(
            "Failed to resolve {label} {}: {error}",
            path.display()
        ))
    })
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf> {
    let canonical = canonical_existing(path, label)?;
    if !canonical.is_dir() {
        return Err(Error::Other(format!(
            "{label} {} is not a directory",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn create_private_scratch() -> Result<TempDir> {
    use std::os::unix::fs::PermissionsExt;

    tempfile::Builder::new()
        .prefix("generalist-sandbox-")
        .permissions(fs::Permissions::from_mode(0o700))
        .tempdir()
        .map_err(|error| Error::Other(format!("Failed to create sandbox scratch space: {error}")))
}

fn detect_backend() -> Result<SandboxBackend> {
    #[cfg(target_os = "macos")]
    {
        let executable = PathBuf::from("/usr/bin/sandbox-exec");
        if executable.is_file() {
            validate_backend(
                StdCommand::new(&executable)
                    .arg("-p")
                    .arg("(version 1) (allow default)")
                    .arg("/usr/bin/true"),
                "macOS Seatbelt",
            )?;
            return Ok(SandboxBackend::Seatbelt { executable });
        }
        return Err(Error::Other(
            "Safe mode requires /usr/bin/sandbox-exec on macOS; use --yolo only if unrestricted execution is intended"
                .to_string(),
        ));
    }

    #[cfg(target_os = "linux")]
    {
        let landlock = linux_landlock_abi();
        if matches!(landlock, Ok(abi) if abi >= 4) {
            return Ok(SandboxBackend::Landlock);
        }

        if let Some(executable) = trusted_bubblewrap_executable() {
            let native_userns_hardening = bubblewrap_supports_userns_hardening(&executable);
            let validation = validate_bubblewrap_backend(&executable, native_userns_hardening);
            if validation.is_ok() {
                return Ok(SandboxBackend::Bubblewrap {
                    executable,
                    native_userns_hardening,
                });
            }
            let landlock = match landlock {
                Ok(abi) => format!("Landlock ABI {abi} is older than the required ABI 4"),
                Err(error) => error,
            };
            return Err(Error::Other(format!(
                "Safe mode is unavailable: {landlock}; {}. Update to a kernel with Landlock ABI 4, or enable Bubblewrap user namespaces. On Ubuntu 24.04, the optional bwrap AppArmor profile may be required. Use --yolo only if unrestricted execution is intended",
                validation.unwrap_err()
            )));
        }
        let landlock = match landlock {
            Ok(abi) => format!("Landlock ABI {abi} is older than the required ABI 4"),
            Err(error) => error,
        };
        return Err(Error::Other(
            format!(
                "Safe mode is unavailable: {landlock}, and a trusted root-owned Bubblewrap (bwrap) executable was not found in /usr/bin, /bin, or /usr/local/bin. Update to a kernel with Landlock ABI 4 or install/configure Bubblewrap; use --yolo only if unrestricted execution is intended"
            ),
        ));
    }

    #[allow(unreachable_code)]
    Err(Error::Other(
        "Safe mode has no process-sandbox backend for this operating system; use --yolo only if unrestricted execution is intended"
            .to_string(),
    ))
}

#[cfg(target_os = "linux")]
fn validate_bubblewrap_backend(executable: &Path, native_userns_hardening: bool) -> Result<()> {
    let true_executable = canonical_executable(Path::new("/usr/bin/true"))?;
    let seccomp_policy = write_bubblewrap_seccomp_policy()?;
    let seccomp_policy_fd = seccomp_policy.as_raw_fd();
    let mut command = StdCommand::new(executable);
    command.args([
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--unshare-user",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/",
        "/",
    ]);
    if native_userns_hardening {
        command.args(["--disable-userns", "--assert-userns-disabled"]);
    }
    command.arg("--seccomp").arg(seccomp_policy_fd.to_string());
    command.arg(true_executable);
    unsafe {
        command.pre_exec(move || install_bubblewrap_launcher_restrictions(seccomp_policy_fd));
    }
    let validation = validate_backend(&mut command, "Bubblewrap");
    drop(seccomp_policy);
    validation
}

#[cfg(target_os = "linux")]
fn bubblewrap_supports_userns_hardening(executable: &Path) -> bool {
    StdCommand::new(executable)
        .arg("--help")
        .env_clear()
        .stdin(Stdio::null())
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .is_some_and(|help| {
            help.contains("--disable-userns") && help.contains("--assert-userns-disabled")
        })
}

#[cfg(target_os = "linux")]
fn linux_landlock_abi() -> std::result::Result<i32, String> {
    const LANDLOCK_CREATE_RULESET_VERSION: libc::c_uint = 1;
    let abi = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<libc::c_void>(),
            0,
            LANDLOCK_CREATE_RULESET_VERSION,
        )
    };
    if abi >= 0 {
        Ok(abi as i32)
    } else {
        Err(format!(
            "the running kernel does not provide Landlock ({})",
            std::io::Error::last_os_error()
        ))
    }
}

fn validate_backend(command: &mut StdCommand, name: &str) -> Result<()> {
    let output = command
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .output()
        .map_err(|error| Error::Other(format!("Failed to start {name}: {error}")))?;
    if output.status.success() {
        return Ok(());
    }
    let detail = String::from_utf8_lossy(&output.stderr);
    let detail = detail.trim();
    Err(Error::Other(if detail.is_empty() {
        format!(
            "Safe mode could not initialize {name} (status {}); use --yolo only if unrestricted execution is intended",
            output.status
        )
    } else {
        format!(
            "Safe mode could not initialize {name} (status {}): {detail}; use --yolo only if unrestricted execution is intended",
            output.status
        )
    }))
}

#[cfg(target_os = "linux")]
fn trusted_bubblewrap_executable() -> Option<PathBuf> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    ["/usr/bin/bwrap", "/bin/bwrap", "/usr/local/bin/bwrap"]
        .iter()
        .filter_map(|path| fs::canonicalize(path).ok())
        .find(|path| {
            path.ancestors().all(|component| {
                fs::metadata(component).is_ok_and(|metadata| {
                    (component != path || metadata.is_file())
                        && metadata.uid() == 0
                        && metadata.permissions().mode() & 0o022 == 0
                })
            })
        })
}

fn resolve_executable_from(program: &OsStr, directories: &[PathBuf]) -> Result<PathBuf> {
    let requested = Path::new(program);
    if requested.components().count() > 1 {
        return canonical_executable(requested);
    }
    for directory in directories {
        let candidate = directory.join(requested);
        if candidate.is_file() {
            return canonical_executable(&candidate);
        }
    }
    Err(Error::Other(format!(
        "Failed to locate sandboxed executable '{}' on the safe PATH",
        program.to_string_lossy()
    )))
}

fn canonical_executable(path: &Path) -> Result<PathBuf> {
    let executable = canonical_existing(path, "sandboxed executable")?;
    if !executable.is_file() {
        return Err(Error::Other(format!(
            "Sandboxed executable {} is not a file",
            executable.display()
        )));
    }
    Ok(executable)
}

#[derive(Debug)]
struct SafeEnvironment {
    variables: Vec<(OsString, OsString)>,
    read_roots: Vec<PathBuf>,
    executable_directories: Vec<PathBuf>,
}

impl SafeEnvironment {
    fn capture(workspace_root: &Path, scratch: &Path) -> Result<Self> {
        let host_path = env::var_os("PATH").unwrap_or_else(|| OsString::from(SAFE_PATH_FALLBACK));
        let mut path_directories = Vec::new();
        let mut seen = BTreeSet::new();
        #[cfg(target_os = "macos")]
        for preferred in ["/Applications/Xcode.app/Contents/Developer/usr/bin"] {
            if let Ok(directory) = fs::canonicalize(preferred) {
                if directory.is_dir() && seen.insert(directory.clone()) {
                    path_directories.push(directory);
                }
            }
        }
        for directory in env::split_paths(&host_path) {
            if !directory.is_absolute() || !directory.is_dir() {
                continue;
            }
            let Ok(directory) = fs::canonicalize(directory) else {
                continue;
            };
            if path_directory_is_overly_broad(&directory) {
                continue;
            }
            if seen.insert(directory.clone()) {
                path_directories.push(directory);
            }
        }
        if path_directories.is_empty() {
            path_directories.extend(env::split_paths(OsStr::new(SAFE_PATH_FALLBACK)));
        }
        let path = env::join_paths(&path_directories)
            .map_err(|error| Error::Other(format!("Failed to construct sandbox PATH: {error}")))?;

        let mut variables = Vec::new();
        for key in [
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TERM",
            "COLORTERM",
            "NO_COLOR",
            "TZ",
            "RUST_BACKTRACE",
        ] {
            if let Some(value) = env::var_os(key) {
                variables.push((OsString::from(key), value));
            }
        }
        variables.extend([
            (OsString::from("PATH"), path.clone()),
            (OsString::from("HOME"), scratch.as_os_str().to_os_string()),
            (OsString::from("TMPDIR"), scratch.as_os_str().to_os_string()),
            (OsString::from("TMP"), scratch.as_os_str().to_os_string()),
            (OsString::from("TEMP"), scratch.as_os_str().to_os_string()),
            (
                OsString::from("XDG_CACHE_HOME"),
                scratch.join("cache").into_os_string(),
            ),
            (
                OsString::from("XDG_CONFIG_HOME"),
                scratch.join("config").into_os_string(),
            ),
            (
                OsString::from("XDG_DATA_HOME"),
                scratch.join("data").into_os_string(),
            ),
            (OsString::from("PYTHONNOUSERSITE"), OsString::from("1")),
            (
                OsString::from("PYTHONDONTWRITEBYTECODE"),
                OsString::from("1"),
            ),
            (
                OsString::from("GIT_CONFIG_GLOBAL"),
                OsString::from("/dev/null"),
            ),
            (OsString::from("GIT_CONFIG_NOSYSTEM"), OsString::from("1")),
        ]);

        let mut read_roots = platform_runtime_roots();
        read_roots.extend(path_directories.iter().cloned());
        read_roots.push(workspace_root.to_path_buf());
        for (key, default_directory) in [("RUSTUP_HOME", ".rustup"), ("ELAN_HOME", ".elan")] {
            if let Some(directory) = toolchain_home(key, default_directory) {
                variables.push((OsString::from(key), directory.as_os_str().to_os_string()));
                read_roots.push(directory);
            }
        }
        if let Some(host_cargo_home) = toolchain_home("CARGO_HOME", ".cargo") {
            let cargo_home = scratch.join("cargo");
            fs::create_dir(&cargo_home).map_err(|error| {
                Error::Other(format!("Failed to create private Cargo home: {error}"))
            })?;
            expose_public_cargo_registry(&host_cargo_home, &cargo_home, &mut read_roots)?;
            variables.push((OsString::from("CARGO_HOME"), cargo_home.into_os_string()));
        }
        Ok(Self {
            variables,
            read_roots: deduplicate_paths(read_roots),
            executable_directories: path_directories,
        })
    }

    fn apply(&self, command: &mut Command) {
        command.env_clear();
        for (key, value) in &self.variables {
            command.env(key, value);
        }
        command.stdin(Stdio::null());
    }
}

fn expose_public_cargo_registry(
    host_cargo_home: &Path,
    private_cargo_home: &Path,
    read_roots: &mut Vec<PathBuf>,
) -> Result<()> {
    // Cargo's full registry and git caches may contain private dependency
    // source or credential-bearing remote URLs. Reuse only the two canonical
    // public crates.io cache layouts, one leaf at a time, under the private
    // CARGO_HOME. Projects with private dependencies remain fail-closed and
    // can vendor them into the workspace explicitly.
    let host_cargo_home = canonical_directory(host_cargo_home, "Cargo home")?;
    let registry_path = host_cargo_home.join("registry");
    if !registry_path.exists() {
        return Ok(());
    }
    let host_registry = canonical_directory(&registry_path, "Cargo registry directory")?;
    if !host_registry.starts_with(&host_cargo_home) {
        return Err(Error::Other(format!(
            "Refusing Cargo registry {} because it resolves outside {}",
            registry_path.display(),
            host_cargo_home.display()
        )));
    }

    for category in ["cache", "index", "src"] {
        let category_path = host_registry.join(category);
        if !category_path.exists() {
            continue;
        }
        let host_category = canonical_directory(&category_path, "Cargo registry directory")?;
        if !host_category.starts_with(&host_registry) {
            return Err(Error::Other(format!(
                "Refusing Cargo registry category {} because it resolves outside {}",
                category_path.display(),
                host_registry.display()
            )));
        }
        let Ok(entries) = fs::read_dir(&host_category) else {
            continue;
        };
        let private_category = private_cargo_home.join("registry").join(category);
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name != "index.crates.io-1949cf8c6b5b557f" && name != "github.com-1ecc6299db9ec823" {
                continue;
            }
            let source = canonical_directory(&entry.path(), "public Cargo registry cache")?;
            if !source.starts_with(&host_category) {
                return Err(Error::Other(format!(
                    "Refusing Cargo registry cache {} because it resolves outside {}",
                    entry.path().display(),
                    host_category.display()
                )));
            }
            fs::create_dir_all(&private_category).map_err(|error| {
                Error::Other(format!(
                    "Failed to create private Cargo registry directory: {error}"
                ))
            })?;
            std::os::unix::fs::symlink(&source, private_category.join(entry.file_name())).map_err(
                |error| {
                    Error::Other(format!(
                        "Failed to expose a read-only public Cargo registry cache: {error}"
                    ))
                },
            )?;
            read_roots.push(source);
        }
    }
    Ok(())
}

fn path_directory_is_overly_broad(directory: &Path) -> bool {
    if [
        "/",
        "/Users",
        "/home",
        "/private",
        "/private/tmp",
        "/tmp",
        "/var",
    ]
    .iter()
    .any(|path| fs::canonicalize(path).is_ok_and(|path| path == directory))
    {
        return true;
    }
    env::var_os("HOME")
        .and_then(|home| fs::canonicalize(home).ok())
        .is_some_and(|home| home == directory || home.starts_with(directory))
}

fn toolchain_home(key: &str, default_directory: &str) -> Option<PathBuf> {
    let configured = env::var_os(key)
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(default_directory)))?;
    canonical_directory(&configured, "toolchain directory")
        .ok()
        .filter(|directory| !path_directory_is_overly_broad(directory))
}

fn deduplicate_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    paths
        .into_iter()
        .filter(|path| seen.insert(path.clone()))
        .collect()
}

fn platform_runtime_roots() -> Vec<PathBuf> {
    #[cfg(target_os = "macos")]
    const CANDIDATES: &[&str] = &[
        "/System",
        "/usr/bin",
        "/usr/lib",
        "/usr/libexec",
        "/usr/sbin",
        "/usr/share",
        "/bin",
        "/sbin",
        "/opt/homebrew/Cellar",
        "/opt/homebrew/Frameworks",
        "/opt/homebrew/lib",
        "/opt/homebrew/share",
        "/usr/local/Cellar",
        "/usr/local/Frameworks",
        "/usr/local/lib",
        "/usr/local/share",
        "/Library/Apple",
        "/Library/Developer",
        "/Applications/Xcode.app",
        "/private/etc/ssl/openssl.cnf",
        "/private/var/db/timezone",
        "/System/Cryptexes/App/usr",
    ];
    #[cfg(target_os = "linux")]
    const CANDIDATES: &[&str] = &[
        "/usr/bin",
        "/usr/include",
        "/usr/lib",
        "/usr/lib64",
        "/usr/libexec",
        "/usr/sbin",
        "/usr/share",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/usr/local/bin",
        "/usr/local/include",
        "/usr/local/lib",
        "/usr/local/lib64",
        "/usr/local/share",
        "/etc/alternatives",
        "/etc/ca-certificates",
        "/etc/group",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/etc/localtime",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/ssl/certs",
        "/dev/null",
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
    ];
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    const CANDIDATES: &[&str] = &["/usr", "/bin", "/sbin", "/usr/local"];

    CANDIDATES
        .iter()
        .filter_map(|path| fs::canonicalize(path).ok())
        .filter(|path| path.exists())
        .collect()
}

#[cfg(target_os = "macos")]
fn prepare_seatbelt_command(
    sandbox_exec: &Path,
    executable: &Path,
    workspace: &Path,
    scratch: &Path,
    read_only: &[PathBuf],
    environment: &SafeEnvironment,
) -> Result<Command> {
    let mut profile = String::from(
        r#"(version 1)
(deny default)
(import "dyld-support.sb")
(deny syscall-unix (syscall-number SYS_setsid SYS_setpgid))
(deny process-info-setcontrol)
(allow process-fork)
(allow sysctl-read)
(allow file-read-metadata)
(allow file-read* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom") (subpath "/dev/fd") (subpath (param "SCRATCH"))"#,
    );
    for index in 0..read_only.len() {
        profile.push_str(&format!(" (subpath (param \"READ_{index}\"))"));
    }
    profile.push_str(")\n");
    profile.push_str(
        "(allow process-exec (subpath (param \"WORKSPACE\")) (subpath (param \"SCRATCH\"))",
    );
    for index in 0..read_only.len() {
        profile.push_str(&format!(" (subpath (param \"READ_{index}\"))"));
    }
    profile.push_str(")\n");
    profile.push_str(
        "(allow file-write* (literal \"/dev/null\") (subpath \"/dev/fd\") \
         (subpath (param \"WORKSPACE\")) (subpath (param \"SCRATCH\")))\n",
    );
    let mut command = Command::new(sandbox_exec);
    command.arg("-p").arg(profile);
    add_seatbelt_parameter(&mut command, "WORKSPACE", workspace)?;
    add_seatbelt_parameter(&mut command, "SCRATCH", scratch)?;
    for (index, path) in read_only.iter().enumerate() {
        add_seatbelt_parameter(&mut command, &format!("READ_{index}"), path)?;
    }
    command.arg(executable);
    let cwd = env::current_dir().unwrap_or_else(|_| workspace.to_path_buf());
    command.current_dir(if cwd.starts_with(workspace) {
        cwd
    } else {
        workspace.to_path_buf()
    });
    environment.apply(&mut command);
    // Seatbelt controls operations performed through paths, but an already-open
    // descriptor bypasses those path checks. Seal every ambient descriptor in
    // the post-fork child before sandbox-exec starts.
    unsafe {
        command.pre_exec(install_macos_process_restrictions);
    }
    Ok(command)
}

#[cfg(target_os = "macos")]
fn add_seatbelt_parameter(command: &mut Command, name: &str, path: &Path) -> Result<()> {
    let path = path.to_str().ok_or_else(|| {
        Error::Other(format!(
            "Safe mode cannot represent non-UTF-8 path {} in a Seatbelt profile",
            path.display()
        ))
    })?;
    command.arg("-D").arg(format!("{name}={path}"));
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn install_process_baseline() -> std::io::Result<()> {
    // A model-authored crash must not leave a core file containing process
    // memory, and newly created files should never become group/world-readable
    // merely because a host process had a permissive umask.
    unsafe {
        libc::umask(0o077);
        let no_core = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::setrlimit(libc::RLIMIT_CORE, &no_core) < 0 {
            return Err(std::io::Error::last_os_error());
        }

        let mut host_nofile: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut host_nofile) < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let max_open_files = host_nofile.rlim_max.min(SANDBOX_MAX_OPEN_FILES);
        let nofile = libc::rlimit {
            rlim_cur: max_open_files,
            rlim_max: max_open_files,
        };
        if libc::setrlimit(libc::RLIMIT_NOFILE, &nofile) < 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn install_macos_process_restrictions() -> std::io::Result<()> {
    seal_macos_inherited_descriptors()?;
    install_process_baseline()
}

#[cfg(target_os = "macos")]
fn seal_macos_inherited_descriptors() -> std::io::Result<()> {
    // Darwin has POSIX_SPAWN_CLOEXEC_DEFAULT, but std::process::Command does
    // not expose it. proc_pidinfo gives us a bounded, allocation-free snapshot
    // in the post-fork child. Stdio has already been remapped at this point, so
    // descriptors 0..=2 are the complete intended inheritance surface.
    const MAX_OPEN_DESCRIPTORS: usize = 4096;
    let mut descriptors: [libc::proc_fdinfo; MAX_OPEN_DESCRIPTORS] = unsafe { std::mem::zeroed() };
    let capacity = std::mem::size_of_val(&descriptors);
    let bytes = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDLISTFDS,
            0,
            descriptors.as_mut_ptr().cast(),
            capacity as libc::c_int,
        )
    };
    if bytes < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let bytes = bytes as usize;
    if bytes == capacity || !bytes.is_multiple_of(std::mem::size_of::<libc::proc_fdinfo>()) {
        return Err(std::io::Error::other(
            "refusing to launch a sandbox with an incomplete descriptor inventory",
        ));
    }
    for descriptor in &descriptors[..bytes / std::mem::size_of::<libc::proc_fdinfo>()] {
        if descriptor.proc_fd <= libc::STDERR_FILENO {
            continue;
        }
        let flags = unsafe { libc::fcntl(descriptor.proc_fd, libc::F_GETFD) };
        if flags < 0 {
            if std::io::Error::last_os_error().raw_os_error() == Some(libc::EBADF) {
                continue;
            }
            return Err(std::io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(descriptor.proc_fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn prepare_landlock_command(
    executable: &Path,
    workspace: &Path,
    scratch: &Path,
    read_only: &[PathBuf],
    environment: &SafeEnvironment,
) -> Result<(Command, OwnedFd)> {
    // V4 is our fail-closed minimum. Ask for every vetted right through V9 so
    // newer kernels additionally restrict device ioctls, out-of-sandbox Unix
    // socket resolution, signals, and abstract Unix sockets. BestEffort only
    // drops rights absent from an otherwise-accepted V4+ kernel.
    let abi = ABI::V9;
    let ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::BestEffort)
        .handle_access(AccessFs::from_all(abi))
        .and_then(|ruleset| ruleset.handle_access(AccessNet::from_all(abi)))
        .and_then(|ruleset| ruleset.scope(Scope::from_all(ABI::V6)))
        .and_then(Ruleset::create)
        .and_then(|ruleset| {
            ruleset.add_rules(path_beneath_rules(
                read_only.iter(),
                AccessFs::from_read(abi),
            ))
        })
        .and_then(|ruleset| {
            ruleset.add_rules(path_beneath_rules(
                [workspace, scratch, Path::new("/dev/null")],
                AccessFs::from_all(abi),
            ))
        })
        .map_err(|error| {
            Error::Other(format!(
                "Failed to prepare the Linux Landlock sandbox: {error}"
            ))
        })?;
    let ruleset: Option<OwnedFd> = ruleset.into();
    let ruleset = ruleset.ok_or_else(|| {
        Error::Other(
            "Linux Landlock reported success without creating an enforceable ruleset".to_string(),
        )
    })?;

    let ruleset_fd = ruleset.as_raw_fd();
    let mut command = Command::new(executable);
    let cwd = env::current_dir().unwrap_or_else(|_| workspace.to_path_buf());
    command.current_dir(if cwd.starts_with(workspace) {
        cwd
    } else {
        workspace.to_path_buf()
    });
    environment.apply(&mut command);
    let seccomp_filter = build_linux_seccomp_filter();
    unsafe {
        command.pre_exec(move || install_linux_restrictions(ruleset_fd, &seccomp_filter));
    }
    Ok((command, ruleset))
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn install_linux_restrictions(
    ruleset_fd: std::os::fd::RawFd,
    seccomp_filter: &[libc::sock_filter],
) -> std::io::Result<()> {
    install_linux_process_baseline()?;
    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    install_linux_seccomp_filter(seccomp_filter)
}

#[cfg(all(
    target_os = "linux",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
fn install_linux_restrictions(
    _ruleset_fd: std::os::fd::RawFd,
    _seccomp_filter: &[libc::sock_filter],
) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Generalist's Linux seccomp policy supports x86_64 and aarch64",
    ))
}

#[cfg(target_os = "linux")]
fn install_linux_process_baseline() -> std::io::Result<()> {
    seal_linux_inherited_descriptors()?;
    install_process_baseline()?;
    // The kernel delivers SIGKILL if the supervising Generalist process goes
    // away. Check for the narrow race in which it died just before prctl.
    if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::getppid() } == 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            "sandbox supervisor exited during process launch",
        ));
    }
    if unsafe { libc::prctl(libc::PR_SET_DUMPABLE, 0, 0, 0, 0) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    drop_linux_capabilities()?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn drop_linux_capabilities() -> std::io::Result<()> {
    // Do not inherit an unusual capability set from a file-capability or
    // privileged Generalist installation. no_new_privs is installed by each
    // backend immediately afterward, so exec cannot reacquire the dropped
    // privileges from a setuid/capability-bearing file.
    const LINUX_CAPABILITY_VERSION_3: u32 = 0x2008_0522;
    #[repr(C)]
    struct CapabilityHeader {
        version: u32,
        pid: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct CapabilityData {
        effective: u32,
        permitted: u32,
        inheritable: u32,
    }
    let mut header = CapabilityHeader {
        version: LINUX_CAPABILITY_VERSION_3,
        pid: 0,
    };
    let data = [CapabilityData {
        effective: 0,
        permitted: 0,
        inheritable: 0,
    }; 2];
    if unsafe {
        libc::syscall(
            libc::SYS_capset,
            &mut header as *mut CapabilityHeader,
            data.as_ptr(),
        )
    } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe {
        libc::prctl(
            libc::PR_CAP_AMBIENT,
            libc::PR_CAP_AMBIENT_CLEAR_ALL,
            0,
            0,
            0,
        )
    } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn seal_linux_inherited_descriptors() -> std::io::Result<()> {
    const CLOSE_RANGE_CLOEXEC: libc::c_uint = 1 << 2;
    if unsafe {
        libc::syscall(
            libc::SYS_close_range,
            (libc::STDERR_FILENO + 1) as libc::c_uint,
            libc::c_uint::MAX,
            CLOSE_RANGE_CLOEXEC,
        )
    } == 0
    {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if !matches!(
        error.raw_os_error(),
        Some(libc::ENOSYS) | Some(libc::EINVAL)
    ) {
        return Err(error);
    }

    // Bubblewrap is specifically the old-kernel fallback, so close_range may
    // be absent there. fcntl is slow but async-signal-safe and fail-closed.
    let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    const MAX_FCNTL_SCAN: libc::rlim_t = 1_048_576;
    if limit.rlim_cur == libc::RLIM_INFINITY || limit.rlim_cur > MAX_FCNTL_SCAN {
        return Err(std::io::Error::other(
            "close_range is unavailable and the descriptor limit is too large to seal safely",
        ));
    }
    for descriptor in (libc::STDERR_FILENO + 1)..(limit.rlim_cur as libc::c_int) {
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
        if flags < 0 {
            if std::io::Error::last_os_error().raw_os_error() == Some(libc::EBADF) {
                continue;
            }
            return Err(std::io::Error::last_os_error());
        }
        if unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn build_linux_seccomp_filter() -> Vec<libc::sock_filter> {
    const BPF_LD_W_ABS: u16 = 0x20;
    const BPF_JMP_JEQ_K: u16 = 0x15;
    const BPF_JMP_JSET_K: u16 = 0x45;
    const BPF_RET_K: u16 = 0x06;
    const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
    const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
    const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
    #[cfg(target_arch = "x86_64")]
    const AUDIT_ARCH_NATIVE: u32 = 0xc000_003e;
    #[cfg(target_arch = "aarch64")]
    const AUDIT_ARCH_NATIVE: u32 = 0xc000_00b7;

    let statement = |code, k| libc::sock_filter {
        code,
        jt: 0,
        jf: 0,
        k,
    };
    let jump = |code, k, jt, jf| libc::sock_filter { code, jt, jf, k };
    let mut filter = vec![
        // struct seccomp_data.arch
        statement(BPF_LD_W_ABS, 4),
        jump(BPF_JMP_JEQ_K, AUDIT_ARCH_NATIVE, 1, 0),
        statement(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
        // struct seccomp_data.nr
        statement(BPF_LD_W_ABS, 0),
        // Refuse the x32 ABI, whose syscall numbers use a high-bit tag.
        jump(BPF_JMP_JSET_K, 0x4000_0000, 0, 1),
        statement(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
    ];

    // Allow ordinary fork/thread creation but refuse namespace creation. libc
    // probes clone3 first on recent systems, so report it as unavailable and
    // let libc fall back to the classic clone syscall whose flags we inspect.
    const SECCOMP_DATA_ARGS_0: u32 = 16;
    const CLONE_NAMESPACE_FLAGS: u32 = (libc::CLONE_NEWCGROUP
        | libc::CLONE_NEWIPC
        | libc::CLONE_NEWNET
        | libc::CLONE_NEWNS
        | libc::CLONE_NEWPID
        | libc::CLONE_NEWTIME
        | libc::CLONE_NEWUSER
        | libc::CLONE_NEWUTS) as u32;
    filter.extend([
        jump(BPF_JMP_JEQ_K, libc::SYS_clone as u32, 0, 4),
        statement(BPF_LD_W_ABS, SECCOMP_DATA_ARGS_0),
        jump(BPF_JMP_JSET_K, CLONE_NAMESPACE_FLAGS, 0, 1),
        statement(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
        statement(BPF_LD_W_ABS, 0),
        jump(BPF_JMP_JEQ_K, libc::SYS_clone3 as u32, 0, 1),
        statement(BPF_RET_K, SECCOMP_RET_ERRNO | libc::ENOSYS as u32),
    ]);

    // Preserve the parent-death and non-dumpable settings installed before
    // exec. Other prctl operations remain available (including stricter
    // seccomp installation and ordinary process naming).
    filter.extend([
        jump(BPF_JMP_JEQ_K, libc::SYS_prctl as u32, 0, 10),
        statement(BPF_LD_W_ABS, SECCOMP_DATA_ARGS_0),
        jump(BPF_JMP_JEQ_K, libc::PR_SET_DUMPABLE as u32, 0, 1),
        statement(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
        jump(BPF_JMP_JEQ_K, libc::PR_SET_PTRACER as u32, 0, 1),
        statement(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
        jump(BPF_JMP_JEQ_K, libc::PR_SET_PDEATHSIG as u32, 0, 1),
        statement(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
        jump(BPF_JMP_JEQ_K, libc::PR_CAP_AMBIENT as u32, 0, 1),
        statement(BPF_RET_K, SECCOMP_RET_ERRNO | libc::EPERM as u32),
        statement(BPF_LD_W_ABS, 0),
    ]);

    // Landlock covers filesystem access and TCP bind/connect. This denylist
    // closes non-path kernel interfaces that can reach the host, other
    // same-UID processes, global IPC objects, or a fresh namespace. The
    // Python tool bridge is already connected on stdin and needs none of them.
    for syscall in [
        libc::SYS_socket,
        libc::SYS_socketpair,
        libc::SYS_bind,
        libc::SYS_connect,
        libc::SYS_listen,
        libc::SYS_accept,
        libc::SYS_accept4,
        libc::SYS_sendto,
        libc::SYS_recvfrom,
        libc::SYS_sendmsg,
        libc::SYS_recvmsg,
        libc::SYS_sendmmsg,
        libc::SYS_recvmmsg,
        libc::SYS_io_uring_setup,
        libc::SYS_io_uring_enter,
        libc::SYS_io_uring_register,
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_pidfd_open,
        libc::SYS_pidfd_getfd,
        libc::SYS_pidfd_send_signal,
        libc::SYS_kcmp,
        libc::SYS_keyctl,
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_capset,
        libc::SYS_kill,
        libc::SYS_tkill,
        libc::SYS_tgkill,
        libc::SYS_rt_sigqueueinfo,
        libc::SYS_rt_tgsigqueueinfo,
        libc::SYS_prlimit64,
        libc::SYS_sched_setaffinity,
        libc::SYS_sched_setparam,
        libc::SYS_sched_setscheduler,
        libc::SYS_sched_setattr,
        libc::SYS_setpriority,
        libc::SYS_ioprio_set,
        libc::SYS_move_pages,
        libc::SYS_migrate_pages,
        libc::SYS_process_madvise,
        libc::SYS_process_mrelease,
        libc::SYS_setpgid,
        libc::SYS_setsid,
        libc::SYS_bpf,
        libc::SYS_perf_event_open,
        libc::SYS_personality,
        libc::SYS_userfaultfd,
        libc::SYS_open_by_handle_at,
        libc::SYS_name_to_handle_at,
        libc::SYS_mount,
        libc::SYS_umount2,
        libc::SYS_pivot_root,
        libc::SYS_chroot,
        libc::SYS_setns,
        libc::SYS_unshare,
        libc::SYS_fsopen,
        libc::SYS_fsconfig,
        libc::SYS_fsmount,
        libc::SYS_fspick,
        libc::SYS_open_tree,
        libc::SYS_move_mount,
        libc::SYS_mount_setattr,
        libc::SYS_fanotify_init,
        libc::SYS_fanotify_mark,
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        libc::SYS_kexec_load,
        libc::SYS_kexec_file_load,
        libc::SYS_reboot,
        libc::SYS_swapon,
        libc::SYS_swapoff,
        libc::SYS_acct,
        libc::SYS_shmget,
        libc::SYS_shmat,
        libc::SYS_shmctl,
        libc::SYS_shmdt,
        libc::SYS_msgget,
        libc::SYS_msgsnd,
        libc::SYS_msgrcv,
        libc::SYS_msgctl,
        libc::SYS_semget,
        libc::SYS_semop,
        libc::SYS_semtimedop,
        libc::SYS_semctl,
        libc::SYS_mq_open,
        libc::SYS_mq_unlink,
        libc::SYS_mq_timedsend,
        libc::SYS_mq_timedreceive,
        libc::SYS_mq_notify,
        libc::SYS_mq_getsetattr,
        libc::SYS_syslog,
    ] {
        filter.push(jump(BPF_JMP_JEQ_K, syscall as u32, 0, 1));
        filter.push(statement(
            BPF_RET_K,
            SECCOMP_RET_ERRNO | (libc::EPERM as u32),
        ));
    }
    filter.push(statement(BPF_RET_K, SECCOMP_RET_ALLOW));
    filter
}

#[cfg(all(
    target_os = "linux",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
fn build_linux_seccomp_filter() -> Vec<libc::sock_filter> {
    Vec::new()
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn install_linux_seccomp_filter(filter: &[libc::sock_filter]) -> std::io::Result<()> {
    let filter_length = u16::try_from(filter.len()).map_err(|_| {
        std::io::Error::other("Linux seccomp policy exceeds the kernel instruction limit")
    })?;
    let program = libc::sock_fprog {
        len: filter_length,
        filter: filter.as_ptr() as *mut libc::sock_filter,
    };
    if unsafe {
        libc::prctl(
            libc::PR_SET_SECCOMP,
            2, // SECCOMP_MODE_FILTER
            &program as *const libc::sock_fprog,
        )
    } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn prepare_bubblewrap_command(
    bubblewrap: &Path,
    native_userns_hardening: bool,
    executable: &Path,
    workspace: &Path,
    scratch: &Path,
    read_only: &[PathBuf],
    environment: &SafeEnvironment,
) -> Result<(Command, File)> {
    let seccomp_policy = write_bubblewrap_seccomp_policy()?;
    let seccomp_policy_fd = seccomp_policy.as_raw_fd();
    // --dev creates a private minimal /dev, including the standard null and
    // randomness devices. Do not shadow those with read-only host bind mounts.
    let read_only = deduplicate_paths(
        read_only
            .iter()
            .filter(|path| !path.starts_with("/dev"))
            .cloned()
            .collect(),
    );

    let mut writable = vec![workspace.to_path_buf(), scratch.to_path_buf()];
    writable = deduplicate_paths(writable);
    let mut destinations = read_only.clone();
    destinations.extend(writable.clone());

    let mut parents = BTreeSet::new();
    for destination in &destinations {
        for parent in destination.ancestors().skip(1) {
            if parent != Path::new("/") {
                parents.insert(parent.to_path_buf());
            }
        }
    }

    let mut command = Command::new(bubblewrap);
    command.args([
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--unshare-user",
        "--unshare-net",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]);
    // New Bubblewrap releases can independently disable nested user
    // namespaces. Older releases remain protected by the mandatory seccomp
    // filter below, which rejects clone/clone3/unshare/setns namespace paths.
    if native_userns_hardening {
        command.args(["--disable-userns", "--assert-userns-disabled"]);
    }
    command.arg("--seccomp").arg(seccomp_policy_fd.to_string());
    for parent in parents {
        command.arg("--dir").arg(parent);
    }
    for path in read_only {
        if !writable.iter().any(|root| path.starts_with(root)) {
            command.arg("--ro-bind").arg(&path).arg(&path);
        }
    }
    for path in writable {
        command.arg("--bind").arg(&path).arg(&path);
    }
    let cwd = env::current_dir().unwrap_or_else(|_| workspace.to_path_buf());
    let cwd = if cwd.starts_with(workspace) {
        cwd
    } else {
        workspace.to_path_buf()
    };
    command.arg("--chdir").arg(cwd).arg(executable);
    environment.apply(&mut command);
    unsafe {
        command.pre_exec(move || install_bubblewrap_launcher_restrictions(seccomp_policy_fd));
    }
    Ok((command, seccomp_policy))
}

#[cfg(target_os = "linux")]
fn write_bubblewrap_seccomp_policy() -> Result<File> {
    let filter = build_linux_seccomp_filter();
    if filter.is_empty() {
        return Err(Error::Other(
            "Generalist's Linux seccomp policy supports x86_64 and aarch64".to_string(),
        ));
    }
    let mut policy = tempfile::tempfile().map_err(|error| {
        Error::Other(format!(
            "Failed to create the private Bubblewrap seccomp policy: {error}"
        ))
    })?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            filter.as_ptr().cast::<u8>(),
            filter.len() * std::mem::size_of::<libc::sock_filter>(),
        )
    };
    policy.write_all(bytes).map_err(|error| {
        Error::Other(format!(
            "Failed to write the Bubblewrap seccomp policy: {error}"
        ))
    })?;
    policy.seek(SeekFrom::Start(0)).map_err(|error| {
        Error::Other(format!(
            "Failed to rewind the Bubblewrap seccomp policy: {error}"
        ))
    })?;
    Ok(policy)
}

#[cfg(target_os = "linux")]
fn install_bubblewrap_launcher_restrictions(
    seccomp_policy_fd: std::os::fd::RawFd,
) -> std::io::Result<()> {
    install_linux_process_baseline()?;
    // The baseline marked every non-stdio descriptor close-on-exec. Preserve
    // only this anonymous policy file for Bubblewrap to consume.
    let flags = unsafe { libc::fcntl(seccomp_policy_fd, libc::F_GETFD) };
    if flags < 0
        || unsafe { libc::fcntl(seccomp_policy_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) } < 0
    {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broad_workspace_roots_are_rejected_before_backend_selection() {
        assert!(validate_workspace_boundary(Path::new("/")).is_err());
        if let Some(home) = env::var_os("HOME").and_then(|path| fs::canonicalize(path).ok()) {
            assert!(validate_workspace_boundary(&home).is_err());
            assert!(path_directory_is_overly_broad(&home));
        }
    }

    #[test]
    fn unrestricted_mode_preserves_the_legacy_boundary() {
        let sandbox = ExecutionSandbox::unrestricted();
        assert!(!sandbox.is_restricted());
        assert!(sandbox.workspace_root().is_none());
        let prepared = sandbox
            .prepare_command("bash", &SandboxAccess::default())
            .unwrap();
        assert!(prepared._scratch.is_none());
    }

    #[test]
    fn safe_environment_keeps_only_an_explicit_allowlist() {
        let workspace = tempfile::tempdir().unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let environment = SafeEnvironment::capture(workspace.path(), scratch.path()).unwrap();
        let keys = environment
            .variables
            .iter()
            .map(|(key, _)| key.to_string_lossy().into_owned())
            .collect::<BTreeSet<_>>();

        assert!(keys.contains("PATH"));
        assert!(keys.contains("HOME"));
        assert!(!keys.contains("ANTHROPIC_API_KEY"));
        assert!(!keys.contains("CEREBRAS_API_KEY"));
        assert!(!keys.contains("OPENAI_API_KEY"));
        for secret_or_injection_vector in [
            "AWS_SECRET_ACCESS_KEY",
            "DYLD_INSERT_LIBRARIES",
            "GITHUB_TOKEN",
            "GIT_ASKPASS",
            "LD_PRELOAD",
            "NODE_OPTIONS",
            "OPENROUTER_API_KEY",
            "PYTHONPATH",
            "RUBYOPT",
            "SSH_AUTH_SOCK",
        ] {
            assert!(
                !keys.contains(secret_or_injection_vector),
                "sandbox environment retained {secret_or_injection_vector}"
            );
        }
        let allowed = [
            "CARGO_HOME",
            "COLORTERM",
            "ELAN_HOME",
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_NOSYSTEM",
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "NO_COLOR",
            "PATH",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "RUSTUP_HOME",
            "RUST_BACKTRACE",
            "TEMP",
            "TERM",
            "TMP",
            "TMPDIR",
            "TZ",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let unexpected = keys
            .iter()
            .map(String::as_str)
            .filter(|key| !allowed.contains(key))
            .collect::<Vec<_>>();
        assert!(
            unexpected.is_empty(),
            "unexpected sandbox variables: {unexpected:?}"
        );
        assert!(environment
            .variables
            .iter()
            .any(|(key, value)| key == "PATH" && !value.is_empty()));
        if let Some((_, cargo_home)) = environment
            .variables
            .iter()
            .find(|(key, _)| key == "CARGO_HOME")
        {
            let cargo_home = PathBuf::from(cargo_home);
            assert!(cargo_home.starts_with(scratch.path()));
            assert!(!cargo_home.join("credentials.toml").exists());
            assert!(!cargo_home.join("git").exists());
        }
    }

    #[test]
    fn private_cargo_caches_are_not_exposed_to_the_sandbox() {
        let host = tempfile::tempdir().unwrap();
        let private = tempfile::tempdir().unwrap();
        let source = host.path().join("registry/src");
        fs::create_dir_all(source.join("index.crates.io-1949cf8c6b5b557f")).unwrap();
        fs::create_dir_all(source.join("index.crates.io-private-lookalike")).unwrap();
        fs::create_dir_all(source.join("private.example-secret")).unwrap();
        fs::create_dir_all(host.path().join("git/checkouts/private-repository")).unwrap();
        let mut read_roots = Vec::new();

        expose_public_cargo_registry(host.path(), private.path(), &mut read_roots).unwrap();

        assert!(private
            .path()
            .join("registry/src/index.crates.io-1949cf8c6b5b557f")
            .exists());
        assert!(!private
            .path()
            .join("registry/src/index.crates.io-private-lookalike")
            .exists());
        assert!(!private
            .path()
            .join("registry/src/private.example-secret")
            .exists());
        assert!(!private.path().join("git").exists());
        assert_eq!(read_roots.len(), 1);
    }

    #[test]
    fn crates_io_named_symlink_cannot_smuggle_an_outside_directory() {
        let host = tempfile::tempdir().unwrap();
        let private = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let source = host.path().join("registry/src");
        fs::create_dir_all(&source).unwrap();
        std::os::unix::fs::symlink(
            outside.path(),
            source.join("index.crates.io-1949cf8c6b5b557f"),
        )
        .unwrap();

        let error =
            expose_public_cargo_registry(host.path(), private.path(), &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("resolves outside"));
    }

    #[test]
    fn cargo_registry_category_symlink_cannot_smuggle_an_outside_directory() {
        let host = tempfile::tempdir().unwrap();
        let private = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir_all(host.path().join("registry")).unwrap();
        fs::create_dir_all(outside.path().join("index.crates.io-1949cf8c6b5b557f")).unwrap();
        std::os::unix::fs::symlink(outside.path(), host.path().join("registry/src")).unwrap();

        let error =
            expose_public_cargo_registry(host.path(), private.path(), &mut Vec::new()).unwrap_err();
        assert!(error.to_string().contains("resolves outside"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bubblewrap_fallback_carries_the_full_hardening_policy() {
        let workspace = tempfile::tempdir().unwrap();
        let scratch = tempfile::tempdir().unwrap();
        let environment = SafeEnvironment::capture(workspace.path(), scratch.path()).unwrap();
        let (command, policy) = prepare_bubblewrap_command(
            Path::new("/usr/bin/bwrap"),
            true,
            Path::new("/usr/bin/true"),
            workspace.path(),
            scratch.path(),
            &environment.read_roots,
            &environment,
        )
        .unwrap();
        let args = command
            .as_std()
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        for required in [
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--unshare-user",
            "--unshare-net",
            "--disable-userns",
            "--assert-userns-disabled",
            "--cap-drop",
            "--seccomp",
        ] {
            assert!(args.iter().any(|argument| argument == required));
        }
        assert!(args
            .windows(2)
            .any(|pair| pair[0] == "--cap-drop" && pair[1] == "ALL"));
        assert!(args
            .windows(2)
            .any(|pair| { pair[0] == "--seccomp" && pair[1] == policy.as_raw_fd().to_string() }));
        assert_eq!(
            policy.metadata().unwrap().len() as usize,
            build_linux_seccomp_filter().len() * std::mem::size_of::<libc::sock_filter>()
        );

        let (legacy_command, legacy_policy) = prepare_bubblewrap_command(
            Path::new("/usr/bin/bwrap"),
            false,
            Path::new("/usr/bin/true"),
            workspace.path(),
            scratch.path(),
            &environment.read_roots,
            &environment,
        )
        .unwrap();
        let legacy_args = legacy_command
            .as_std()
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(!legacy_args
            .iter()
            .any(|argument| argument == "--disable-userns"));
        assert!(!legacy_args
            .iter()
            .any(|argument| argument == "--assert-userns-disabled"));
        assert!(legacy_args
            .iter()
            .any(|argument| argument == "--unshare-user"));
        assert!(legacy_args.windows(2).any(|pair| {
            pair[0] == "--seccomp" && pair[1] == legacy_policy.as_raw_fd().to_string()
        }));
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_mode_allows_workspace_but_denies_sibling_files_and_network() {
        use tokio::net::TcpListener;

        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        let secret = parent.path().join("secret.txt");
        fs::write(&secret, "outside").unwrap();
        let escaped_read = workspace.join("escaped-read.txt");
        std::os::unix::fs::symlink(&secret, &escaped_read).unwrap();
        let escaped_hardlink = workspace.join("escaped-hardlink.txt");
        let outside_write = parent.path().join("escaped.txt");
        let inside_write = workspace.join("allowed.txt");
        let outside_program = parent.path().join("outside-program");
        fs::write(&outside_program, "#!/bin/bash\nprintf escaped > \"$1\"\n").unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&outside_program, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let outside_program_marker = workspace.join("outside-program-ran.txt");
        let sandbox = ExecutionSandbox::restricted(&workspace).unwrap();

        let mut prepared = sandbox
            .prepare_command("/bin/bash", &SandboxAccess::default())
            .unwrap();
        let status = prepared
            .command_mut()
            .arg("-c")
            .arg(
                "if /bin/cat \"$1\" >/dev/null 2>&1; then exit 10; fi; \
                 if /bin/cat \"$2\" >/dev/null 2>&1; then exit 11; fi; \
                 if /bin/ln \"$1\" \"$3\" 2>/dev/null; then exit 12; fi; \
                 if /usr/bin/touch \"$4\" 2>/dev/null; then exit 13; fi; \
                 /bin/echo allowed > \"$5\" || exit 14; \
                 if \"$6\" \"$7\" 2>/dev/null; then exit 15; fi",
            )
            .arg("sandbox-test")
            .arg(&secret)
            .arg(&escaped_read)
            .arg(&escaped_hardlink)
            .arg(&outside_write)
            .arg(&inside_write)
            .arg(&outside_program)
            .arg(&outside_program_marker)
            .status()
            .await
            .unwrap();
        assert!(status.success());
        assert_eq!(fs::read_to_string(&inside_write).unwrap().trim(), "allowed");
        assert!(!escaped_hardlink.exists());
        assert!(!outside_write.exists());
        assert!(!outside_program_marker.exists());

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut prepared = sandbox
            .prepare_command("python3", &SandboxAccess::default())
            .unwrap();
        let status = prepared
            .command_mut()
            .arg("-c")
            .arg(
                "import socket,sys\n".to_owned()
                    + "try:\n s=socket.socket()\n s.connect(('127.0.0.1',int(sys.argv[1])))\n"
                    + "except OSError:\n sys.exit(0)\n"
                    + "sys.exit(20)\n",
            )
            .arg(port.to_string())
            .status()
            .await
            .unwrap();
        assert!(status.success(), "sandboxed process reached a TCP listener");
        drop(listener);

        let unix_socket = workspace.join("host.sock");
        let listener = std::os::unix::net::UnixListener::bind(&unix_socket).unwrap();
        let mut prepared = sandbox
            .prepare_command("python3", &SandboxAccess::default())
            .unwrap();
        let status = prepared
            .command_mut()
            .arg("-c")
            .arg(concat!(
                "import socket,sys\n",
                "try:\n s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1])\n",
                "except OSError:\n sys.exit(0)\n",
                "sys.exit(21)\n",
            ))
            .arg(&unix_socket)
            .status()
            .await
            .unwrap();
        assert!(
            status.success(),
            "sandboxed process reached a host Unix socket"
        );
        drop(listener);

        #[cfg(target_os = "linux")]
        if matches!(
            &sandbox.mode,
            SandboxMode::Restricted(RestrictedSandbox {
                backend: SandboxBackend::Landlock,
                ..
            })
        ) {
            let mut prepared = sandbox
                .prepare_command("/bin/bash", &SandboxAccess::default())
                .unwrap();
            let status = prepared
                .command_mut()
                .arg("-c")
                .arg("! /bin/cat /proc/self/status >/dev/null 2>&1")
                .status()
                .await
                .unwrap();
            assert!(status.success(), "sandboxed process could read procfs");
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_mode_does_not_inherit_ambient_file_descriptors() {
        use std::os::fd::AsRawFd;

        let parent = tempfile::tempdir().unwrap();
        let workspace = parent.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        let secret = parent.path().join("secret.txt");
        fs::write(&secret, "secret").unwrap();
        let secret = fs::File::open(secret).unwrap();
        let descriptor = secret.as_raw_fd();
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
        assert!(flags >= 0);
        assert_eq!(
            unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags & !libc::FD_CLOEXEC) },
            0
        );

        let sandbox = ExecutionSandbox::restricted(&workspace).unwrap();
        let mut prepared = sandbox
            .prepare_command("python3", &SandboxAccess::default())
            .unwrap();
        let mut child = prepared
            .command_mut()
            .arg("-c")
            .arg(
                "import os,sys\ntry:\n os.read(int(sys.argv[1]), 1)\nexcept OSError:\n sys.exit(0)\nsys.exit(23)\n",
            )
            .arg(descriptor.to_string())
            .spawn()
            .unwrap();
        assert_eq!(
            unsafe { libc::fcntl(descriptor, libc::F_SETFD, flags | libc::FD_CLOEXEC) },
            0
        );
        let status = child.wait().await.unwrap();
        assert!(
            status.success(),
            "sandbox inherited descriptor {descriptor}"
        );
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_mode_caps_open_descriptors() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox = ExecutionSandbox::restricted(workspace.path()).unwrap();
        let mut prepared = sandbox
            .prepare_command("python3", &SandboxAccess::default())
            .unwrap();
        let output = prepared
            .command_mut()
            .arg("-I")
            .arg("-S")
            .arg("-c")
            .arg(concat!(
                "opened=[]\n",
                "try:\n",
                " while len(opened) < 2048: opened.append(open('/dev/null','rb'))\n",
                "except OSError:\n pass\n",
                "print(len(opened))\n",
            ))
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let opened = String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap();
        assert!(opened < SANDBOX_MAX_OPEN_FILES);
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_mode_cannot_signal_the_host_process() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox = ExecutionSandbox::restricted(workspace.path()).unwrap();
        let mut prepared = sandbox
            .prepare_command("python3", &SandboxAccess::default())
            .unwrap();
        let status = prepared
            .command_mut()
            .arg("-c")
            .arg(
                "import os,sys\ntry:\n os.kill(int(sys.argv[1]), 0)\nexcept OSError:\n sys.exit(0)\nsys.exit(24)\n",
            )
            .arg(std::process::id().to_string())
            .status()
            .await
            .unwrap();
        assert!(status.success(), "sandbox could signal the host process");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn restricted_linux_blocks_non_path_kernel_escape_hatches() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox = ExecutionSandbox::restricted(workspace.path()).unwrap();
        let mut prepared = sandbox
            .prepare_command("python3", &SandboxAccess::default())
            .unwrap();
        let blocked = [
            (libc::SYS_keyctl, 0_i64),
            (libc::SYS_unshare, libc::CLONE_NEWUSER as i64),
            (libc::SYS_shmget, 0),
            (libc::SYS_pidfd_open, std::process::id() as i64),
            (libc::SYS_io_uring_setup, 0),
            (libc::SYS_bpf, 0),
            (libc::SYS_prlimit64, 0),
            (libc::SYS_capset, 0),
            (libc::SYS_prctl, libc::PR_SET_PDEATHSIG as i64),
            (libc::SYS_prctl, libc::PR_CAP_AMBIENT as i64),
        ];
        let encoded = blocked
            .iter()
            .map(|(syscall, argument)| format!("{syscall}:{argument}"))
            .collect::<Vec<_>>()
            .join(",");
        let status = prepared
            .command_mut()
            .arg("-c")
            .arg(concat!(
                "import ctypes,errno,sys\n",
                "libc=ctypes.CDLL(None, use_errno=True)\n",
                "for index,item in enumerate(sys.argv[1].split(',')):\n",
                " nr,arg=map(int,item.split(':'))\n",
                " ctypes.set_errno(0)\n",
                " result=libc.syscall(nr,arg,0,0,0,0,0)\n",
                " if result != -1 or ctypes.get_errno() != errno.EPERM: sys.exit(40+index)\n",
                "ctypes.set_errno(0)\n",
                "result=libc.syscall(int(sys.argv[2]),0,0)\n",
                "sys.exit(0 if result == -1 and ctypes.get_errno() == errno.ENOSYS else 60)\n",
            ))
            .arg(encoded)
            .arg(libc::SYS_clone3.to_string())
            .status()
            .await
            .unwrap();
        assert!(
            status.success(),
            "Linux seccomp escape-hatch policy was not enforced"
        );
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[tokio::test]
    async fn restricted_mode_cannot_detach_from_its_execution_group() {
        let workspace = tempfile::tempdir().unwrap();
        let sandbox = ExecutionSandbox::restricted(workspace.path()).unwrap();
        let mut prepared = sandbox
            .prepare_command("python3", &SandboxAccess::default())
            .unwrap();
        let status = prepared
            .command_mut()
            .arg("-c")
            .arg("import os,sys\ntry:\n os.setsid()\nexcept OSError:\n sys.exit(0)\nsys.exit(25)\n")
            .status()
            .await
            .unwrap();
        assert!(
            status.success(),
            "sandboxed process detached from its execution group"
        );
    }
}
