//! Skills: on-demand instruction files (progressive disclosure).
//!
//! A skill is a directory containing `SKILL.md`, optionally starting with
//! frontmatter:
//!
//! ```text
//! ---
//! name: deploy
//! description: How to build and deploy this project
//! ---
//! ...full instructions...
//! ```
//!
//! Only the one-line index enters the system prompt; the agent reads the
//! full file with `read_file` when a task matches. This is the SKILL.md
//! pattern used by pi and Claude Code: context cost stays flat no matter
//! how many skills exist.

use std::path::Path;

struct Skill {
    name: String,
    description: String,
    path: String,
}

fn parse_skill(dir_name: &str, path: &Path, content: &str) -> Skill {
    let mut name = dir_name.to_string();
    let mut description = String::new();

    let mut lines = content.lines().peekable();
    if lines.peek().map(|l| l.trim()) == Some("---") {
        lines.next();
        for line in lines.by_ref() {
            let line = line.trim();
            if line == "---" {
                break;
            }
            if let Some(value) = line.strip_prefix("name:") {
                name = value.trim().to_string();
            } else if let Some(value) = line.strip_prefix("description:") {
                description = value.trim().to_string();
            }
        }
    }
    if description.is_empty() {
        description = lines
            .find(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .unwrap_or("")
            .trim()
            .chars()
            .take(200)
            .collect();
    }
    Skill {
        name,
        description,
        path: path.display().to_string(),
    }
}

/// Build the system-prompt index for all skills under `dir`.
///
/// Returns `None` when the directory is missing or holds no skills.
pub fn skills_index(dir: &Path) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let skill_file = entry.path().join("SKILL.md");
        if let Ok(content) = std::fs::read_to_string(&skill_file) {
            let dir_name = entry.file_name().to_string_lossy().to_string();
            skills.push(parse_skill(&dir_name, &skill_file, &content));
        }
    }
    if skills.is_empty() {
        return None;
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    let lines: Vec<String> = skills
        .iter()
        .map(|s| {
            format!(
                "- {} — {} (read {} for the full instructions)",
                s.name, s.description, s.path
            )
        })
        .collect();
    Some(format!(
        "\n\n## Skills\n\nSpecialized instruction files are available. When a task matches one, \
         read its SKILL.md with read_file before proceeding:\n{}",
        lines.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_parses_frontmatter_and_falls_back_to_first_line() {
        let dir =
            std::env::temp_dir().join(format!("gnl-skills-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(dir.join("deploy")).unwrap();
        std::fs::write(
            dir.join("deploy/SKILL.md"),
            "---\nname: deploy\ndescription: Build and ship the project\n---\nSteps...",
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("no-frontmatter")).unwrap();
        std::fs::write(
            dir.join("no-frontmatter/SKILL.md"),
            "# Title\nFirst real line here.",
        )
        .unwrap();
        // A directory without SKILL.md is ignored.
        std::fs::create_dir_all(dir.join("empty")).unwrap();

        let index = skills_index(&dir).unwrap();
        assert!(index.contains("deploy — Build and ship the project"));
        assert!(index.contains("no-frontmatter — First real line here."));
        assert!(!index.contains("empty —"));

        assert!(skills_index(&dir.join("missing")).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
