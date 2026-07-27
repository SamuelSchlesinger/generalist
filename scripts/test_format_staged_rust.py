#!/usr/bin/env python3
"""Regression tests for scripts/format_staged_rust.py."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FORMATTER = Path(__file__).with_name("format_staged_rust.py").resolve()


def git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class FormatStagedRustTests(unittest.TestCase):
    def new_repo(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        repo = Path(temporary.name)
        git(repo, "init", "--quiet")
        return temporary, repo

    def run_formatter(self, repo: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(FORMATTER)],
            cwd=repo,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_formats_and_restages_fully_staged_paths_with_spaces(self) -> None:
        temporary, repo = self.new_repo()
        with temporary:
            source = repo / "spaced name.rs"
            source.write_text("fn main(){println!(\"ok\");}\n")
            git(repo, "add", "--", source.name)

            result = self.run_formatter(repo)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(git(repo, "diff", "--", source.name).stdout, "")
            staged = git(repo, "show", f":{source.name}").stdout
            self.assertIn('    println!("ok");', staged)

    def test_rejects_partial_staging_without_changing_the_index(self) -> None:
        temporary, repo = self.new_repo()
        with temporary:
            source = repo / "partial.rs"
            staged = "fn staged(){println!(\"staged\");}\n"
            working = staged + "fn unstaged(){println!(\"unstaged\");}\n"
            source.write_text(staged)
            git(repo, "add", "--", source.name)
            source.write_text(working)

            result = self.run_formatter(repo)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unstaged changes", result.stderr)
            self.assertEqual(git(repo, "show", f":{source.name}").stdout, staged)
            self.assertEqual(source.read_text(), working)


if __name__ == "__main__":
    unittest.main()
