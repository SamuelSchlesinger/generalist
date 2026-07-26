#!/usr/bin/env python3
"""Validate the self-contained safety-evaluation research branch.

This is intentionally a no-network structural validator. It checks citation
resolution and metadata consistency, local links, index coverage, and the CSV
coverage ledger. It does not establish that a bibliographic record is true;
that still requires checking the primary source.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index.md"
MATRIX = ROOT / "data" / "threat-control-matrix.csv"

GENERIC_DEFINITION_RE = re.compile(r"^\[([^\]]+)\]:\s*(.+)$", re.MULTILINE)
CITATION_RE = re.compile(r"\[([^\]\n]+)\]\[([^\]\n]+)\]")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(([^)\n]+)\)")
URL_RE = re.compile(r"https://[^\s>)]+")
LOWER_KEY_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def normalized_space(value: str) -> str:
    return " ".join(value.split())


def strip_url_punctuation(value: str) -> str:
    return value.rstrip(".,;:")


def canonical_url(value: str) -> str:
    """Return a comparison key for common scholarly URLs."""

    value = strip_url_punctuation(value)
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


def metadata_without_urls(definition: str) -> str:
    return normalized_space(URL_RE.sub("", definition)).rstrip(" .")


def fail(errors: list[str], path: Path, message: str) -> None:
    try:
        display = path.relative_to(ROOT)
    except ValueError:
        display = path
    errors.append(f"{display}: {message}")


def local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split(maxsplit=1)[0]
    if target.startswith(("https://", "http://", "mailto:", "#")):
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def validate_markdown(
    path: Path,
    global_by_key: dict[str, list[tuple[Path, str]]],
    global_by_url: dict[str, list[tuple[Path, str, str]]],
    errors: list[str],
) -> set[Path]:
    text = path.read_text(encoding="utf-8")
    heading = "\n## Local References\n"
    if heading not in f"\n{text}":
        fail(errors, path, "missing '## Local References' section")
        reference_start = len(text)
    else:
        reference_start = text.index("## Local References")

    definitions: dict[str, str] = {}
    for match in GENERIC_DEFINITION_RE.finditer(text):
        key, definition = match.groups()
        if not LOWER_KEY_RE.fullmatch(key):
            fail(errors, path, f"reference key is not lowercase kebab-case: [{key}]")
            continue
        if key in definitions:
            fail(errors, path, f"duplicate reference definition [{key}]")
            continue
        if match.start() < reference_start:
            fail(errors, path, f"reference definition [{key}] precedes Local References")
        if not URL_RE.search(definition):
            fail(errors, path, f"reference definition [{key}] has no full https URL")
        definitions[key] = normalized_space(definition)
        global_by_key[key].append((path, normalized_space(definition)))
        for url in URL_RE.findall(definition):
            canonical = canonical_url(url)
            global_by_url[canonical].append(
                (path, key, metadata_without_urls(definition))
            )

    cited_keys: set[str] = set()
    for match in CITATION_RE.finditer(text[:reference_start]):
        left, right = match.groups()
        if left != right:
            fail(errors, path, f"citation labels differ: [{left}][{right}]")
            continue
        if not LOWER_KEY_RE.fullmatch(left):
            fail(errors, path, f"citation key is not lowercase kebab-case: [{left}]")
            continue
        cited_keys.add(left)
        if left not in definitions:
            fail(errors, path, f"citation [{left}][{left}] lacks a local definition")

    for key in sorted(definitions.keys() - cited_keys):
        fail(errors, path, f"unused local reference definition [{key}]")

    linked_paths: set[Path] = set()
    for match in MARKDOWN_LINK_RE.finditer(text):
        target = local_link_target(path, match.group(1))
        if target is None:
            continue
        try:
            target.relative_to(ROOT)
        except ValueError:
            fail(errors, path, f"local link escapes branch: {match.group(1)}")
            continue
        if not target.exists():
            fail(errors, path, f"broken local link: {match.group(1)}")
        else:
            linked_paths.add(target)

    return linked_paths


def validate_global_metadata(
    global_by_key: dict[str, list[tuple[Path, str]]],
    global_by_url: dict[str, list[tuple[Path, str, str]]],
    errors: list[str],
) -> None:
    for key, records in sorted(global_by_key.items()):
        definitions = {definition for _, definition in records}
        if len(definitions) > 1:
            locations = ", ".join(str(path.relative_to(ROOT)) for path, _ in records)
            errors.append(
                f"citation key [{key}] has conflicting metadata across: {locations}"
            )

    for url, records in sorted(global_by_url.items()):
        metadata = {record_metadata for _, _, record_metadata in records}
        if len(metadata) > 1:
            locations = ", ".join(
                f"{path.relative_to(ROOT)}:[{key}]" for path, key, _ in records
            )
            errors.append(
                f"canonical URL {url} has conflicting metadata across: {locations}"
            )


def validate_index_coverage(index_links: set[Path], errors: list[str]) -> None:
    expected = {
        path.resolve()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != INDEX
        and "__pycache__" not in path.parts
        and not path.name.startswith(".")
    }
    for path in sorted(expected - index_links):
        fail(errors, INDEX, f"does not link child artifact {path.relative_to(ROOT)}")


def validate_matrix(errors: list[str]) -> None:
    expected_columns = [
        "threat_id",
        "threat",
        "attack_or_failure_path",
        "evidence_status",
        "preventive_controls",
        "detective_controls",
        "corrective_controls",
        "evaluation",
        "residual_risk",
        "primary_artifacts",
    ]
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            fail(
                errors,
                MATRIX,
                f"unexpected columns: {reader.fieldnames!r}",
            )
            return
        rows = list(reader)

    ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        missing = [key for key, value in row.items() if not value.strip()]
        if missing:
            fail(errors, MATRIX, f"line {line_number} has empty fields: {missing}")
        threat_id = row["threat_id"]
        if threat_id in ids:
            fail(errors, MATRIX, f"duplicate threat_id {threat_id}")
        ids.add(threat_id)
        for raw_artifact in row["primary_artifacts"].split(";"):
            artifact = (ROOT / raw_artifact).resolve()
            if not artifact.exists():
                fail(
                    errors,
                    MATRIX,
                    f"line {line_number} names missing artifact {raw_artifact}",
                )

    required_families = {f"T{number}" for number in range(1, 12)}
    present_families = {
        re.match(r"T\d+", threat_id).group(0)
        for threat_id in ids
        if re.match(r"T\d+", threat_id)
    }
    missing_families = sorted(required_families - present_families)
    if missing_families:
        fail(errors, MATRIX, f"missing threat families: {missing_families}")


def main() -> int:
    errors: list[str] = []
    markdown_files = sorted(ROOT.rglob("*.md"))
    global_by_key: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    global_by_url: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    index_links: set[Path] = set()

    for path in markdown_files:
        links = validate_markdown(path, global_by_key, global_by_url, errors)
        if path == INDEX:
            index_links = links

    validate_global_metadata(global_by_key, global_by_url, errors)
    validate_index_coverage(index_links, errors)
    validate_matrix(errors)

    if errors:
        print(f"FAILED: {len(errors)} validation error(s)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    definition_count = sum(len(records) for records in global_by_key.values())
    print(
        "OK: "
        f"{len(markdown_files)} markdown files, "
        f"{definition_count} local reference definitions, "
        f"{len(global_by_url)} canonical source URLs, "
        f"{sum(1 for _ in csv.DictReader(MATRIX.open(encoding='utf-8')))} matrix rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
