#!/usr/bin/env python3
"""Summarize the component and byte layout of an Ollama model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import defaultdict
from typing import Any


TENSOR_MEDIA_TYPE = "application/vnd.ollama.image.tensor"


def classify_tensor(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("mtp.") or ".mtp." in lowered:
        return "mtp"
    if "vision" in lowered or "visual" in lowered:
        return "vision"
    if lowered == "lm_head.weight" or lowered.startswith("model.language_model."):
        return "language"
    return "other"


def add_blob(target: dict[str, dict[str, int]], group: str, size: int) -> None:
    target[group]["blobs"] += 1
    target[group]["bytes"] += size


def summarize_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise ValueError("manifest layers must be a list")

    tensor_groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"blobs": 0, "bytes": 0}
    )
    non_tensor_groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"blobs": 0, "bytes": 0}
    )
    layer_bytes = 0
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("every manifest layer must be an object")
        size = layer.get("size")
        if not isinstance(size, int) or size < 0:
            raise ValueError("every manifest layer needs a non-negative integer size")
        layer_bytes += size
        media_type = str(layer.get("mediaType", "unknown"))
        if media_type == TENSOR_MEDIA_TYPE:
            add_blob(tensor_groups, classify_tensor(str(layer.get("name", ""))), size)
        else:
            add_blob(non_tensor_groups, media_type, size)

    config = manifest.get("config")
    config_bytes = config.get("size", 0) if isinstance(config, dict) else 0
    if not isinstance(config_bytes, int) or config_bytes < 0:
        raise ValueError("manifest config size must be a non-negative integer")

    def ordered(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        return {key: groups[key] for key in sorted(groups)}

    tensor_blobs = sum(group["blobs"] for group in tensor_groups.values())
    tensor_bytes = sum(group["bytes"] for group in tensor_groups.values())
    return {
        "schema_version": 1,
        "artifact": {
            "config_bytes": config_bytes,
            "layer_blobs": len(layers),
            "layer_bytes": layer_bytes,
            "referenced_bytes": config_bytes + layer_bytes,
        },
        "tensors": {
            "blobs": tensor_blobs,
            "bytes": tensor_bytes,
            "groups": ordered(tensor_groups),
        },
        "non_tensor_layers": ordered(non_tensor_groups),
    }


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.manifest.resolve()
    with path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    result = summarize_manifest(manifest)
    result["manifest"] = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
