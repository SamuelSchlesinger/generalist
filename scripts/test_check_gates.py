#!/usr/bin/env python3
"""Keep formal checks out of commits and in the complete publication gate."""

from pathlib import Path
import subprocess
import unittest


REPO = Path(__file__).resolve().parent.parent
FORMAL_COMMANDS = ["./scripts/check-tla.sh", "./scripts/check-model-conformance.sh"]


class CheckGateTests(unittest.TestCase):
    def commands(self, target: str) -> list[str]:
        result = subprocess.run(
            ["make", "--no-print-directory", "-n", target],
            cwd=REPO, check=True, text=True, capture_output=True,
        )
        return result.stdout.splitlines()

    def test_commit_lint_omits_formal_execution(self) -> None:
        commands = self.commands("lint")
        for command in FORMAL_COMMANDS:
            self.assertNotIn(command, commands)
        self.assertIn("cargo fmt --all -- --check", commands)
        self.assertTrue(any("cargo clippy" in command for command in commands))

    def test_full_gate_keeps_each_formal_check_and_rust_suite(self) -> None:
        commands = self.commands("check")
        for command in FORMAL_COMMANDS:
            self.assertEqual(commands.count(command), 1)
        self.assertTrue(any("cargo test --all-targets" in command for command in commands))
        self.assertTrue(any("cargo test --doc" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
