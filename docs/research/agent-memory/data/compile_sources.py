#!/usr/bin/env python3
"""Compile the corpus-wide bibliography from per-document local references."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "sources.md"
DEFINITION_RE = re.compile(r"^\[([a-z0-9]+(?:-[a-z0-9]+)*)\]:\s*(.+)$", re.MULTILINE)
URL_RE = re.compile(r"https://[^\s>)]+")
LOCAL_REFERENCES = "\n## Local References\n"


def canonical_url(value: str) -> str:
    value = value.rstrip(".,;:")
    parts = urlsplit(value)
    host = parts.netloc.lower()
    path = re.sub(r"/+", "/", unquote(parts.path))
    query = parts.query

    if host in {"dx.doi.org", "doi.org", "www.doi.org"}:
        host = "doi.org"
        path = path.rstrip("/").lower()
        query = ""
    elif host in {"arxiv.org", "www.arxiv.org"}:
        host = "arxiv.org"
        match = re.match(r"/(?:abs|pdf)/([^/]+?)(?:\.pdf)?$", path)
        if match:
            identifier = re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)
            path = f"/abs/{identifier}"
        query = ""
    elif host in {"openreview.net", "www.openreview.net"}:
        host = "openreview.net"
        path = path.rstrip("/")
    else:
        path = path.rstrip("/") or "/"

    return urlunsplit(("https", host, path, query, ""))


def parse_local_definitions(text: str) -> list[tuple[str, str]]:
    marker = text.find(LOCAL_REFERENCES)
    if marker < 0:
        return []
    return DEFINITION_RE.findall(text[marker + len(LOCAL_REFERENCES) :])


def definitions() -> dict[str, list[tuple[str, str]]]:
    by_url: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in sorted(ROOT.rglob("*.md")):
        if path == OUTPUT or ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for key, definition in parse_local_definitions(text):
            urls = URL_RE.findall(definition)
            if not urls:
                continue
            by_url[canonical_url(urls[0])].append((key, " ".join(definition.split())))
    return by_url


def choose_key(records: list[tuple[str, str]]) -> str:
    counts = Counter(key for key, _ in records)
    return sorted(counts, key=lambda key: (-counts[key], key))[0]


def choose_definition(records: list[tuple[str, str]]) -> str:
    # Prefer complete author lists and publication metadata. This is deterministic;
    # branch validators remain responsible for checking the underlying metadata.
    return sorted(
        (definition for _, definition in records),
        key=lambda definition: (-len(definition), definition),
    )[0]


def render() -> str:
    by_url = definitions()
    selected: list[tuple[str, str, str]] = []
    for url, records in by_url.items():
        selected.append((choose_key(records), choose_definition(records), url))

    used: Counter[str] = Counter()
    entries: list[tuple[str, str]] = []
    for requested_key, definition, url in sorted(
        selected, key=lambda item: (item[1].casefold(), item[2])
    ):
        used[requested_key] += 1
        key = requested_key
        if used[requested_key] > 1:
            key = f"{requested_key}-{used[requested_key]}"
        entries.append((key, definition))

    lines = [
        "# Master Bibliography",
        "",
        "This bibliography is compiled from each document's `## Local References`",
        "section by `data/compile_sources.py` and reviewed as a whole-corpus artifact.",
        "Canonical URLs are deduplicated; local reference sections remain authoritative",
        "for the claims on their page.",
        "",
        f"Canonical sources: **{len(entries)}**.",
        "",
    ]
    lines.extend(f"[{key}]: {definition}" for key, definition in entries)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
