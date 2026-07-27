//! Runtime storage scope shared by conversations and episodic memory.
//!
//! Normal runs discover the nearest Git worktree root (or use the canonical
//! working directory when no Git root exists). The global scope is never a
//! discovery fallback: callers must request it explicitly.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const GLOBAL_MEMORY_KEY: &[u8] = b"\0generalist-global-scope-v1";
const PROJECT_MEMORY_KEY_PREFIX: &[u8] = b"\0generalist-project-scope-v1\0";

/// The namespace in which a Generalist run stores conversation state and
/// captures episodic memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WorkspaceScope {
    /// An explicitly selected cross-project namespace.
    Global,
    /// A canonical project/worktree root.
    Project {
        #[serde(with = "path_bytes")]
        root: PathBuf,
    },
}

/// Explicit selection used by permissioned archive-search tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScopeFilter {
    Current,
    Global,
    OtherProjects,
    All,
}

mod path_bytes {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(path: &Path, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(path.as_os_str().as_bytes())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<PathBuf, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        Ok(PathBuf::from(OsString::from_vec(bytes)))
    }
}

impl ScopeFilter {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Global => "global",
            Self::OtherProjects => "other_projects",
            Self::All => "all",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "current" => Ok(Self::Current),
            "global" => Ok(Self::Global),
            "other_projects" => Ok(Self::OtherProjects),
            "all" => Ok(Self::All),
            other => Err(Error::Other(format!(
                "Unknown scope '{other}'; expected current, global, other_projects, or all"
            ))),
        }
    }

    pub fn includes(self, candidate: &WorkspaceScope, current: &WorkspaceScope) -> bool {
        match self {
            Self::Current => candidate == current,
            Self::Global => candidate.is_global(),
            Self::OtherProjects => !candidate.is_global() && candidate != current,
            Self::All => true,
        }
    }
}

impl WorkspaceScope {
    /// Discover a project scope from `start`.
    ///
    /// The nearest ancestor containing `.git` wins. If none exists, the
    /// canonical starting directory is its own project scope.
    pub fn discover(start: &Path) -> Result<Self> {
        Ok(Self::Project {
            root: discover_project_root(start)?,
        })
    }

    /// Construct a project scope after canonicalizing its root.
    pub fn project(root: &Path) -> Result<Self> {
        let root = fs::canonicalize(root).map_err(|error| {
            Error::Other(format!(
                "Failed to resolve project root {}: {error}",
                root.display()
            ))
        })?;
        Ok(Self::Project { root })
    }

    /// Construct the explicit global scope.
    pub fn global() -> Self {
        Self::Global
    }

    pub fn is_global(&self) -> bool {
        matches!(self, Self::Global)
    }

    pub fn project_root(&self) -> Option<&Path> {
        match self {
            Self::Global => None,
            Self::Project { root } => Some(root),
        }
    }

    /// Human-readable scope label used in the UI and exported records.
    pub fn display_name(&self) -> String {
        match self {
            Self::Global => "global".to_string(),
            Self::Project { root } => match root.to_str().filter(|root| {
                root.chars().all(|character| {
                    !character.is_control()
                        && character != '`'
                        && character != '\u{2028}'
                        && character != '\u{2029}'
                })
            }) {
                Some(root) => root.to_string(),
                None => format!(
                    "unix-path-hex:{}",
                    root.as_os_str()
                        .as_bytes()
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>()
                ),
            },
        }
    }

    /// Stable, filesystem-safe directory component for scoped history.
    pub fn storage_key(&self) -> String {
        match self {
            Self::Global => "global".to_string(),
            Self::Project { root } => {
                let id = Uuid::new_v5(&Uuid::NAMESPACE_URL, root.as_os_str().as_bytes());
                format!("project-{id}")
            }
        }
    }

    /// Typed SQLite key for the memory settings and episode rows in this scope.
    pub(crate) fn memory_key(&self) -> Vec<u8> {
        match self {
            Self::Global => GLOBAL_MEMORY_KEY.to_vec(),
            Self::Project { root } => {
                let path = root.as_os_str().as_bytes();
                let mut key = Vec::with_capacity(PROJECT_MEMORY_KEY_PREFIX.len() + path.len());
                key.extend_from_slice(PROJECT_MEMORY_KEY_PREFIX);
                key.extend_from_slice(path);
                key
            }
        }
    }
}

/// Find the nearest Git worktree root, falling back to the canonical start.
pub fn discover_project_root(start: &Path) -> Result<PathBuf> {
    let canonical = fs::canonicalize(start).map_err(|error| {
        Error::Other(format!(
            "Failed to resolve working directory {}: {error}",
            start.display()
        ))
    })?;
    let mut candidate = canonical.as_path();
    loop {
        if candidate.join(".git").exists() {
            return Ok(candidate.to_path_buf());
        }
        match candidate.parent() {
            Some(parent) => candidate = parent,
            None => return Ok(canonical),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_discovery_uses_the_nearest_git_root() {
        let temp = tempfile::tempdir().unwrap();
        let project = temp.path().join("project");
        let nested = project.join("a").join("b");
        fs::create_dir_all(project.join(".git")).unwrap();
        fs::create_dir_all(&nested).unwrap();

        assert_eq!(
            WorkspaceScope::discover(&nested).unwrap(),
            WorkspaceScope::Project {
                root: fs::canonicalize(project).unwrap()
            }
        );
    }

    #[test]
    fn global_is_explicit_and_storage_keys_are_stable_and_distinct() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("first");
        let second = temp.path().join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();

        let first = WorkspaceScope::project(&first).unwrap();
        let first_again = WorkspaceScope::project(first.project_root().unwrap()).unwrap();
        let second = WorkspaceScope::project(&second).unwrap();
        assert_eq!(first.storage_key(), first_again.storage_key());
        assert_ne!(first.storage_key(), second.storage_key());
        assert_ne!(first.storage_key(), WorkspaceScope::global().storage_key());
        assert!(WorkspaceScope::global().project_root().is_none());
    }

    #[test]
    fn non_utf8_project_labels_remain_unambiguous() {
        let first = WorkspaceScope::Project {
            root: PathBuf::from(OsString::from_vec(vec![b'/', b'a', 0x80])),
        };
        let second = WorkspaceScope::Project {
            root: PathBuf::from(OsString::from_vec(vec![b'/', b'a', 0x81])),
        };

        assert_eq!(first.display_name(), "unix-path-hex:2f6180");
        assert_eq!(second.display_name(), "unix-path-hex:2f6181");
        assert_ne!(first.display_name(), second.display_name());

        let control = WorkspaceScope::Project {
            root: PathBuf::from("/a\n`scope"),
        };
        assert_eq!(control.display_name(), "unix-path-hex:2f610a6073636f7065");
    }
}
