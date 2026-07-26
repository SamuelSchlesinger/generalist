#!/usr/bin/env python3
"""Validate the local systems research corpus without network access."""

from __future__ import annotations

import re
import sys
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "index.md"
MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]*)?)\)")
CITATION = re.compile(r"\[([a-z0-9-]+)\]\[\1\]")
DEFINITION = re.compile(r"^\[([^\]]+)\]:\s+(.+)$", re.MULTILINE)


def local_links(path: Path, text: str) -> list[Path]:
    result: list[Path] = []
    for raw in MD_LINK.findall(text):
        target = raw.split("#", 1)[0]
        if "://" not in target:
            result.append((path.parent / target).resolve())
    return result


def main() -> int:
    errors: list[str] = []
    markdown = sorted(ROOT.rglob("*.md"))
    known = {path.resolve() for path in markdown}

    edges: dict[Path, list[Path]] = {}
    total_words = 0
    definition_count = 0
    citation_count = 0

    for path in markdown:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        total_words += len(re.findall(r"\b[\w’-]+\b", text))

        if "## Local References" not in text:
            errors.append(f"{rel}: missing '## Local References'")

        definitions = dict(DEFINITION.findall(text))
        definition_count += len(definitions)
        for key, value in definitions.items():
            if key != key.lower() or not re.fullmatch(r"[a-z0-9-]+", key):
                errors.append(f"{rel}: non-lowercase citation key [{key}]")
            if "(accessed 2026-07-26)" not in value:
                errors.append(f"{rel}: reference [{key}] lacks audit access date")

        used = set(CITATION.findall(text))
        citation_count += len(CITATION.findall(text))
        for key in sorted(used - definitions.keys()):
            errors.append(f"{rel}: citation [{key}][{key}] has no local definition")
        for key in sorted(definitions.keys() - used):
            errors.append(f"{rel}: local reference [{key}] is never cited")

        links = local_links(path, text)
        edges[path.resolve()] = links
        for target in links:
            if target not in known:
                errors.append(f"{rel}: missing local markdown target {target}")

    reachable: set[Path] = set()
    queue: deque[Path] = deque([ENTRY.resolve()])
    while queue:
        current = queue.popleft()
        if current in reachable or current not in known:
            continue
        reachable.add(current)
        queue.extend(edges.get(current, []))

    for path in sorted(known - reachable):
        errors.append(f"{path.relative_to(ROOT)}: not reachable from index.md")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PASS")
    print(f"markdown_files={len(markdown)}")
    print(f"words={total_words}")
    print(f"citation_uses={citation_count}")
    print(f"reference_definitions={definition_count}")
    print("all_markdown_reachable=true")
    print("all_citations_locally_defined=true")
    print("all_references_access_dated=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
