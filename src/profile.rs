//! Immutable paths for one Generalist profile.
//!
//! `GENERALIST_HOME` is interpreted once when a profile is discovered. The
//! resulting value owns every profile-relative path so later environment
//! changes cannot send individual stores to different roots.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

const STATE_DIRECTORY: &str = ".generalist";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilePaths {
    root: PathBuf,
    environment_file: PathBuf,
    state_directory: PathBuf,
    history_directory: PathBuf,
    history_scopes_directory: PathBuf,
    memory_database: PathBuf,
    todo_file: PathBuf,
    mcp_config: PathBuf,
    skills_directory: PathBuf,
    exports_directory: PathBuf,
}

impl ProfilePaths {
    /// Resolve the active profile from `GENERALIST_HOME`, falling back to the
    /// operating-system home directory and finally the current directory.
    pub fn discover() -> Self {
        Self::from_sources(std::env::var_os("GENERALIST_HOME"), || {
            #[allow(deprecated)]
            std::env::home_dir()
        })
    }

    /// Construct a profile rooted at an explicit path.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let state_directory = root.join(STATE_DIRECTORY);
        let history_directory = state_directory.join("history");

        Self {
            environment_file: root.join(".generalist.env"),
            history_scopes_directory: history_directory.join("scopes"),
            memory_database: state_directory
                .join("memory")
                .join("scoped-episodes.sqlite3"),
            todo_file: root.join(".generalist_todos.json"),
            mcp_config: state_directory.join("mcp.json"),
            skills_directory: state_directory.join("skills"),
            exports_directory: state_directory.join("exports"),
            root,
            state_directory,
            history_directory,
        }
    }

    fn from_sources(
        configured_home: Option<OsString>,
        fallback_home: impl FnOnce() -> Option<PathBuf>,
    ) -> Self {
        let root = configured_home
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .or_else(fallback_home)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::new(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn environment_file(&self) -> &Path {
        &self.environment_file
    }

    pub fn state_directory(&self) -> &Path {
        &self.state_directory
    }

    pub fn history_directory(&self) -> &Path {
        &self.history_directory
    }

    pub fn history_scopes_directory(&self) -> &Path {
        &self.history_scopes_directory
    }

    pub fn memory_database(&self) -> &Path {
        &self.memory_database
    }

    pub fn todo_file(&self) -> &Path {
        &self.todo_file
    }

    pub fn mcp_config(&self) -> &Path {
        &self.mcp_config
    }

    pub fn skills_directory(&self) -> &Path {
        &self.skills_directory
    }

    pub fn exports_directory(&self) -> &Path {
        &self.exports_directory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_home_wins_without_consulting_fallback() {
        let profile = ProfilePaths::from_sources(Some(OsString::from("configured-home")), || {
            panic!("fallback home must not be consulted")
        });

        assert_eq!(profile.root(), Path::new("configured-home"));
    }

    #[test]
    fn empty_configured_home_uses_fallback() {
        let profile = ProfilePaths::from_sources(Some(OsString::new()), || {
            Some(PathBuf::from("fallback-home"))
        });

        assert_eq!(profile.root(), Path::new("fallback-home"));
    }

    #[test]
    fn absent_homes_fall_back_to_current_directory() {
        let profile = ProfilePaths::from_sources(None, || None);

        assert_eq!(profile.root(), Path::new("."));
    }

    #[test]
    fn every_profile_path_is_derived_from_one_root() {
        let root = Path::new("profile-home");
        let profile = ProfilePaths::new(root);

        assert_eq!(profile.environment_file(), root.join(".generalist.env"));
        assert_eq!(profile.state_directory(), root.join(".generalist"));
        assert_eq!(
            profile.history_directory(),
            root.join(".generalist/history")
        );
        assert_eq!(
            profile.history_scopes_directory(),
            root.join(".generalist/history/scopes")
        );
        assert_eq!(
            profile.memory_database(),
            root.join(".generalist/memory/scoped-episodes.sqlite3")
        );
        assert_eq!(profile.todo_file(), root.join(".generalist_todos.json"));
        assert_eq!(profile.mcp_config(), root.join(".generalist/mcp.json"));
        assert_eq!(profile.skills_directory(), root.join(".generalist/skills"));
        assert_eq!(
            profile.exports_directory(),
            root.join(".generalist/exports")
        );
    }
}
