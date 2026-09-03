#!/usr/bin/env python3
"""Regression tests for the user-facing embedded build entrypoint."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parent.parent
BUILDER = REPOSITORY / "scripts" / "build-embedded.sh"
NATIVE_BUILDER = REPOSITORY / "native" / "mlx" / "build_static_probe.sh"
NATIVE_NOTICES = REPOSITORY / "native" / "mlx" / "THIRD_PARTY_NOTICES.md"
NATIVE_PINS = REPOSITORY / "native" / "mlx" / "PINS.env"
NATIVE_SHIM = REPOSITORY / "native" / "mlx" / "shim.cpp"

REQUIRED_NATIVE_COPYRIGHTS = (
    "Copyright © 2023 Apple Inc.",
    "Copyright (c) 2023 ml-explore",
    "Copyright (c) Ollama",
    "Copyright (c) 2013-2022 Niels Lohmann",
    "Victor Zverovich and {fmt} contributors",
    "Copyright (C) 2010-2022 Max-Planck-Society",
    "Copyright (C) 2019-2020 Peter Bell",
    "Copyright © 2018 the V8 project authors.",
    "Copyright (c) 2012 Jakob Progsch",
    "Copyright (c) 2015-2023 Norbert Juffa",
    "Copyright 2020-2024 Apple Inc.",
    "Copyright © 2025 Apple Inc.",
    "Copyright © 2008-2013 NVIDIA Corporation",
    "Copyright © 2013 Filipe RNC Maia",
)


def reported_cache(**environment: str) -> str:
    process_environment = os.environ.copy()
    for variable in ("GENERALIST_MLX_WORK_ROOT", "XDG_CACHE_HOME", "HOME"):
        process_environment.pop(variable, None)
    process_environment.update(environment)
    result = subprocess.run(
        [str(BUILDER), "--print-cache"],
        cwd=REPOSITORY,
        env=process_environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


class EmbeddedBuildInterfaceTests(unittest.TestCase):
    def test_default_macos_cache_is_persistent_and_user_scoped(self) -> None:
        self.assertEqual(
            reported_cache(HOME="/Users/example"),
            "/Users/example/Library/Caches/generalist/embedded-mlx",
        )

    def test_xdg_cache_and_explicit_root_take_precedence(self) -> None:
        self.assertEqual(
            reported_cache(HOME="/Users/example", XDG_CACHE_HOME="/cache"),
            "/cache/generalist/embedded-mlx",
        )
        self.assertEqual(
            reported_cache(
                HOME="/Users/example",
                XDG_CACHE_HOME="/cache",
                GENERALIST_MLX_WORK_ROOT="/persistent/native",
            ),
            "/persistent/native",
        )

    def test_make_target_and_public_documentation_use_the_wrapper(self) -> None:
        self.assertTrue(os.access(BUILDER, os.X_OK))
        builder = BUILDER.read_text()
        makefile = (REPOSITORY / "Makefile").read_text()
        readme = (REPOSITORY / "README.md").read_text()
        self.assertIn("embedded:\n\t./scripts/build-embedded.sh", makefile)
        self.assertIn(
            "install-embedded:\n\t./scripts/build-embedded.sh --install", makefile
        )
        self.assertIn("make embedded", readme)
        self.assertIn("make install-embedded", readme)
        self.assertNotIn("/private/tmp", readme)
        self.assertNotIn("GENERALIST_MLX_SOURCE_DIR=", readme)
        self.assertIn("Preparing pinned MLX cache in:", builder)

    def test_rebuilt_probe_is_executed_by_default(self) -> None:
        builder = NATIVE_BUILDER.read_text()
        self.assertIn(
            'elif [ "$native_rebuilt" = 1 ] || [ "$probe_rebuilt" = 1 ]; then',
            builder,
        )

    def test_exact_native_patchset_is_pinned(self) -> None:
        pins = NATIVE_PINS.read_text()
        builder = NATIVE_BUILDER.read_text()
        patchset_lines = [
            line
            for line in pins.splitlines()
            if line.startswith("MLX_PATCHSET_SHA256=")
        ]
        self.assertEqual(len(patchset_lines), 1)
        digest = patchset_lines[0].partition("=")[2]
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in digest))
        self.assertIn('observed_patchset" != "$MLX_PATCHSET_SHA256', builder)
        self.assertIn("refusing a modified pinned MLX-C source cache", builder)

    def test_native_license_closure_is_embedded_and_guarded(self) -> None:
        notices = NATIVE_NOTICES.read_text()
        native_builder = NATIVE_BUILDER.read_text()
        main = (REPOSITORY / "src" / "main.rs").read_text()

        for copyright_notice in REQUIRED_NATIVE_COPYRIGHTS:
            with self.subTest(copyright_notice=copyright_notice):
                self.assertIn(copyright_notice, notices)
                self.assertIn(copyright_notice, native_builder)

        self.assertIn(
            'include_str!("../native/mlx/THIRD_PARTY_NOTICES.md")', main
        )
        self.assertIn('print!("\\n{EMBEDDED_MLX_LICENSES}")', main)
        self.assertIn("verify_native_license_contract", native_builder)
        self.assertIn("native fetched dependency set changed", native_builder)
        for fetched_dependency in ("fmt", "json", "metal_cpp"):
            self.assertIn(fetched_dependency, native_builder)

    def test_native_errors_return_instead_of_terminating_the_process(self) -> None:
        shim = NATIVE_SHIM.read_text()
        self.assertIn("mlx_set_error_handler(ignore_mlx_c_error", shim)
        self.assertIn("install_nonterminating_mlx_error_handler();", shim)
        self.assertIn(
            "Qwen model loader caught an unknown native exception", shim
        )


if __name__ == "__main__":
    unittest.main()
