#!/usr/bin/env python3
"""Keep profile-home resolution and path layout centralized."""

from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
PROFILE_MODULE = SOURCE / "profile.rs"
ALLOW_MARKER = "profile-path-allow"
FORBIDDEN = (
    "GENERALIST_HOME",
    "env::home_dir(",
    "ProfilePaths::discover()",
    '".generalist"',
    '".generalist/',
    '".generalist_',
    '".generalist.',
)


def main() -> int:
    violations: list[str] = []
    for path in sorted(SOURCE.rglob("*.rs")):
        if path == PROFILE_MODULE:
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if ALLOW_MARKER in line:
                continue
            if any(pattern in line for pattern in FORBIDDEN):
                relative = path.relative_to(REPOSITORY)
                violations.append(f"{relative}:{line_number}: {line.strip()}")

    if violations:
        print(
            "Profile paths must be resolved through src/profile.rs. "
            f"Use '{ALLOW_MARKER}' only for intentional legacy fixtures."
        )
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
