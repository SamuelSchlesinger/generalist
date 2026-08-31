#!/usr/bin/env python3
"""Validate embedded-inference differential checkpoint evidence."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT = re.compile(r"^[0-9a-f]{16}$")
RMS_CHECKPOINT = "qwen3.8-27b-mlx-layer0-input-rmsnorm"
EMBEDDING_CHECKPOINT = "qwen3.8-27b-mlx-token42-embedding-layer0-rmsnorm"
NVFP4_CHECKPOINT = "qwen3.8-27b-mlx-token42-layer0-in-proj-qkv-nvfp4"
LAYER0_PROJECTION_CHECKPOINT = "qwen3.8-27b-mlx-token42-layer0-packed-projections"
LAYER0_GDN_CHECKPOINT = "qwen3.8-27b-mlx-token42-layer0-gdn-zero-state"
LAYER0_FULL_CHECKPOINT = "qwen3.8-27b-mlx-token42-layer0-full-zero-state"
FULL_DECODE_CHECKPOINT = "qwen3.8-27b-mlx-token42-full-decode-zero-state"
CACHED_DECODE_CHECKPOINT = "qwen3.8-27b-mlx-token42-1219-cached-decode"
METAL_DISPATCH_CHECKPOINT = "qwen3.8-27b-mlx-cached-metal-dispatch-profile"
QMV_GEOMETRY_SWEEP = "qwen3.8-27b-mlx-qmv-geometry-sweep"
MTP_DEPTH_CONTROL = "qwen3.8-27b-mlx-ollama-mtp-depth-control"
NATIVE_SPECULATIVE_DECODE = "qwen3.8-27b-mlx-native-speculative-decode"
ADAPTIVE_PRODUCTION = "qwen3.8-27b-mlx-adaptive-production"
EMBEDDED_OLLAMA = "qwen3.8-27b-mlx-embedded-vs-ollama"
EXPECTED_NORM_TENSOR = "model.language_model.layers.0.input_layernorm.weight"
EXPECTED_EMBEDDING_TENSOR = "model.language_model.embed_tokens.weight"
RMS_EXACT_FIELDS = {
    "output_sum",
    "output_squared_sum",
    "output_first",
    "output_middle",
    "output_last",
    "output_fingerprint",
}
EMBEDDING_STAGES = ("embedding", "normalized")
EMBEDDING_EXACT_FIELDS = {
    "sum",
    "squared_sum",
    "first",
    "middle",
    "last",
    "fingerprint",
}
NVFP4_STAGES = ("normalized_input", "raw_projection", "output")
NVFP4_EXACT_FIELDS = EMBEDDING_EXACT_FIELDS | {"dtype", "elements"}
LAYER0_PROJECTION_STAGES = ("normalized_input", "raw_qkvz", "mixed_qkvz", "mixed_ba")
LAYER0_GDN_STAGES = ("conv_out", "conv_state", "a_exp", "gdn_out", "delta_state")
LAYER0_FULL_STAGES = (
    "gdn_norm",
    "gated",
    "attention_out",
    "after_attention",
    "post_norm",
    "mlp_gate",
    "mlp_up",
    "swiglu",
    "mlp_out",
    "layer_out",
)
FULL_DECODE_STAGES = ("hidden", "logits")


def validate_checkpoint(record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError("checkpoint must be a JSON object")
    if record.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema_version")
    checkpoint = record.get("checkpoint")
    if checkpoint == RMS_CHECKPOINT:
        validate_rms_checkpoint(record)
    elif checkpoint == EMBEDDING_CHECKPOINT:
        validate_embedding_checkpoint(record)
    elif checkpoint == NVFP4_CHECKPOINT:
        validate_nvfp4_checkpoint(record)
    elif checkpoint == LAYER0_PROJECTION_CHECKPOINT:
        validate_layer0_projection_checkpoint(record)
    elif checkpoint == LAYER0_GDN_CHECKPOINT:
        validate_layer0_gdn_checkpoint(record)
    elif checkpoint == LAYER0_FULL_CHECKPOINT:
        validate_layer0_full_checkpoint(record)
    elif checkpoint == FULL_DECODE_CHECKPOINT:
        validate_full_decode_checkpoint(record)
    elif checkpoint == CACHED_DECODE_CHECKPOINT:
        validate_cached_decode_checkpoint(record)
    elif checkpoint == METAL_DISPATCH_CHECKPOINT:
        validate_metal_dispatch_checkpoint(record)
    elif checkpoint == QMV_GEOMETRY_SWEEP:
        validate_qmv_geometry_sweep(record)
    elif checkpoint == MTP_DEPTH_CONTROL:
        validate_mtp_depth_control(record)
    elif checkpoint == NATIVE_SPECULATIVE_DECODE:
        validate_native_speculative_decode(record)
    elif checkpoint == ADAPTIVE_PRODUCTION:
        validate_adaptive_production(record)
    elif checkpoint == EMBEDDED_OLLAMA:
        validate_embedded_ollama(record)
    else:
        raise ValueError("unexpected checkpoint identity")


def validate_common_provenance(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = require_object(record, "model")
    if not SHA256.fullmatch(str(model.get("manifest_sha256", ""))):
        raise ValueError("model.manifest_sha256 must be a lowercase SHA-256 digest")

    runtime = require_object(record, "runtime")
    for field in ("ollama_commit", "mlx_commit", "mlx_c_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(runtime.get(field, ""))):
            raise ValueError(f"runtime.{field} must be a full Git commit")
    return model, runtime


def validate_rms_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    if model.get("tensor_name") != EXPECTED_NORM_TENSOR:
        raise ValueError("unexpected checkpoint tensor")
    if not SHA256.fullmatch(str(model.get("tensor_sha256", ""))):
        raise ValueError("model.tensor_sha256 must be a lowercase SHA-256 digest")

    operation = require_object(record, "operation")
    if operation.get("width") != 5120:
        raise ValueError("RMSNorm checkpoint width must be 5120")
    if [operation.get(name) for name in ("input_dtype", "weight_dtype", "output_dtype")] != [
        12,
        12,
        12,
    ]:
        raise ValueError("RMSNorm checkpoint must remain BF16 end to end")
    if operation.get("epsilon") != "0.000001" or operation.get("shift_weight_by_one") is not True:
        raise ValueError("RMSNorm epsilon or weight-shift contract changed")

    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    comparison = require_object(record, "comparison")
    declared_fields = set(comparison.get("exact_fields", []))
    if declared_fields != RMS_EXACT_FIELDS or comparison.get("exact") is not True:
        raise ValueError("checkpoint does not declare the complete exact comparison")
    for field in RMS_EXACT_FIELDS:
        if generalist.get(field) != ollama.get(field):
            raise ValueError(f"Generalist/Ollama mismatch in {field}")
    if not FINGERPRINT.fullmatch(str(generalist.get("output_fingerprint", ""))):
        raise ValueError("output fingerprint must be 64-bit lowercase hexadecimal")

    reference_error = parse_finite_float(
        generalist.get("max_abs_float64_reference_error"),
        "generalist.max_abs_float64_reference_error",
    )
    if reference_error > 0.008:
        raise ValueError("BF16 RMSNorm reference error exceeds the frozen bound")
    for field in RMS_EXACT_FIELDS - {"output_fingerprint"}:
        parse_finite_float(generalist.get(field), f"generalist.{field}")


def validate_embedding_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    if model.get("embedding_name") != EXPECTED_EMBEDDING_TENSOR:
        raise ValueError("unexpected embedding tensor")
    if model.get("norm_name") != EXPECTED_NORM_TENSOR:
        raise ValueError("unexpected norm tensor")
    for field in ("embedding_sha256", "norm_sha256"):
        if not SHA256.fullmatch(str(model.get(field, ""))):
            raise ValueError(f"model.{field} must be a lowercase SHA-256 digest")

    operation = require_object(record, "operation")
    if operation.get("token_id") != 42 or operation.get("width") != 5120:
        raise ValueError("embedding checkpoint token or width changed")
    if [operation.get(name) for name in ("embedding_dtype", "normalized_dtype")] != [
        12,
        12,
    ]:
        raise ValueError("embedding checkpoint must remain BF16 end to end")
    if (
        operation.get("epsilon") != "0.000001"
        or operation.get("shift_norm_weight_by_one") is not True
    ):
        raise ValueError("embedding checkpoint norm contract changed")
    if operation.get("fingerprint") != "fnv1a64-little-endian-f32":
        raise ValueError("embedding checkpoint fingerprint contract changed")

    comparison = require_object(record, "comparison")
    if (
        comparison.get("exact") is not True
        or comparison.get("element_count") != 5120
        or comparison.get("stages") != list(EMBEDDING_STAGES)
    ):
        raise ValueError("embedding checkpoint does not declare the complete exact comparison")

    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    for stage in EMBEDDING_STAGES:
        generalist_stage = require_object(generalist, stage)
        ollama_stage = require_object(ollama, stage)
        if set(generalist_stage) != EMBEDDING_EXACT_FIELDS:
            raise ValueError(f"generalist.{stage} has an unexpected evidence shape")
        if set(ollama_stage) != EMBEDDING_EXACT_FIELDS:
            raise ValueError(f"ollama.{stage} has an unexpected evidence shape")
        for field in EMBEDDING_EXACT_FIELDS:
            if generalist_stage.get(field) != ollama_stage.get(field):
                raise ValueError(f"Generalist/Ollama mismatch in {stage}.{field}")
        if not FINGERPRINT.fullmatch(str(generalist_stage.get("fingerprint", ""))):
            raise ValueError(f"{stage} fingerprint must be 64-bit lowercase hexadecimal")
        for field in EMBEDDING_EXACT_FIELDS - {"fingerprint"}:
            parse_finite_float(generalist_stage.get(field), f"generalist.{stage}.{field}")


def validate_nvfp4_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    expected_names = {
        "embedding_name": EXPECTED_EMBEDDING_TENSOR,
        "norm_name": EXPECTED_NORM_TENSOR,
        "weight_name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "scale_name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight.scale",
        "global_scale_name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight.global_scale",
    }
    for field, expected in expected_names.items():
        if model.get(field) != expected:
            raise ValueError(f"unexpected NVFP4 {field}")
    for field in ("embedding_sha256", "norm_sha256", "projection_sha256"):
        if not SHA256.fullmatch(str(model.get(field, ""))):
            raise ValueError(f"model.{field} must be a lowercase SHA-256 digest")

    operation = require_object(record, "operation")
    exact_operation = {
        "token_id": 42,
        "input_width": 5120,
        "output_width": 10240,
        "input_dtype": 12,
        "output_dtype": 12,
        "weight_dtype": 3,
        "scale_dtype": 1,
        "global_scale_dtype": 10,
        "group_size": 16,
        "bits": 4,
        "mode": "nvfp4",
        "transpose_weight": True,
        "epsilon": "0.000001",
        "shift_norm_weight_by_one": True,
        "fingerprint": "fnv1a64-little-endian-f32",
    }
    for field, expected in exact_operation.items():
        if operation.get(field) != expected:
            raise ValueError(f"NVFP4 operation.{field} changed")
    parse_finite_float(operation.get("global_scale"), "operation.global_scale")

    comparison = require_object(record, "comparison")
    if comparison.get("exact") is not True or comparison.get("stages") != list(NVFP4_STAGES):
        raise ValueError("NVFP4 checkpoint does not declare the complete exact comparison")

    expected_elements = {"normalized_input": 5120, "raw_projection": 10240, "output": 10240}
    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    for stage in NVFP4_STAGES:
        generalist_stage = require_object(generalist, stage)
        ollama_stage = require_object(ollama, stage)
        if set(generalist_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"generalist.{stage} has an unexpected evidence shape")
        if set(ollama_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"ollama.{stage} has an unexpected evidence shape")
        if generalist_stage.get("dtype") != 12:
            raise ValueError(f"generalist.{stage} must remain BF16")
        if generalist_stage.get("elements") != expected_elements[stage]:
            raise ValueError(f"generalist.{stage} has an unexpected element count")
        for field in NVFP4_EXACT_FIELDS:
            if generalist_stage.get(field) != ollama_stage.get(field):
                raise ValueError(f"Generalist/Ollama mismatch in {stage}.{field}")
        if not FINGERPRINT.fullmatch(str(generalist_stage.get("fingerprint", ""))):
            raise ValueError(f"{stage} fingerprint must be 64-bit lowercase hexadecimal")
        for field in EMBEDDING_EXACT_FIELDS - {"fingerprint"}:
            parse_finite_float(generalist_stage.get(field), f"generalist.{stage}.{field}")


def validate_layer0_projection_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    expected_names = {
        "embedding_name": EXPECTED_EMBEDDING_TENSOR,
        "norm_name": EXPECTED_NORM_TENSOR,
        "qkv_name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "z_name": "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "beta_name": "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "alpha_name": "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    }
    for field, expected in expected_names.items():
        if model.get(field) != expected:
            raise ValueError(f"unexpected layer-zero projection {field}")
    for field in (
        "embedding_sha256",
        "norm_sha256",
        "qkv_sha256",
        "z_sha256",
        "beta_sha256",
        "alpha_sha256",
    ):
        if not SHA256.fullmatch(str(model.get(field, ""))):
            raise ValueError(f"model.{field} must be a lowercase SHA-256 digest")

    operation = require_object(record, "operation")
    exact_operation = {
        "token_id": 42,
        "input_width": 5120,
        "qkv_width": 10240,
        "z_width": 6144,
        "ba_width": 96,
        "input_dtype": 12,
        "output_dtype": 12,
        "weight_dtype": 3,
        "scale_dtype": 1,
        "group_size": 16,
        "bits": 4,
        "mode": "nvfp4",
        "transpose_weight": True,
        "epsilon": "0.000001",
        "shift_norm_weight_by_one": True,
        "fingerprint": "fnv1a64-little-endian-f32",
    }
    for field, expected in exact_operation.items():
        if operation.get(field) != expected:
            raise ValueError(f"layer-zero projection operation.{field} changed")
    for field in ("qkv_global_scale", "z_global_scale"):
        parse_finite_float(operation.get(field), f"operation.{field}")

    comparison = require_object(record, "comparison")
    if comparison.get("exact") is not True or comparison.get("stages") != list(
        LAYER0_PROJECTION_STAGES
    ):
        raise ValueError(
            "layer-zero projection checkpoint does not declare the complete exact comparison"
        )

    expected_elements = {
        "normalized_input": 5120,
        "raw_qkvz": 16384,
        "mixed_qkvz": 16384,
        "mixed_ba": 96,
    }
    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    for stage in LAYER0_PROJECTION_STAGES:
        generalist_stage = require_object(generalist, stage)
        ollama_stage = require_object(ollama, stage)
        if set(generalist_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"generalist.{stage} has an unexpected evidence shape")
        if set(ollama_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"ollama.{stage} has an unexpected evidence shape")
        if generalist_stage.get("dtype") != 12:
            raise ValueError(f"generalist.{stage} must remain BF16")
        if generalist_stage.get("elements") != expected_elements[stage]:
            raise ValueError(f"generalist.{stage} has an unexpected element count")
        for field in NVFP4_EXACT_FIELDS:
            if generalist_stage.get(field) != ollama_stage.get(field):
                raise ValueError(f"Generalist/Ollama mismatch in {stage}.{field}")
        if not FINGERPRINT.fullmatch(str(generalist_stage.get("fingerprint", ""))):
            raise ValueError(f"{stage} fingerprint must be 64-bit lowercase hexadecimal")
        for field in EMBEDDING_EXACT_FIELDS - {"fingerprint"}:
            parse_finite_float(generalist_stage.get(field), f"generalist.{stage}.{field}")


def validate_layer0_gdn_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    expected_model = {
        "projection_checkpoint": LAYER0_PROJECTION_CHECKPOINT,
        "conv_name": "model.language_model.layers.0.linear_attn.conv1d.weight",
        "dt_bias_name": "model.language_model.layers.0.linear_attn.dt_bias",
        "a_log_name": "model.language_model.layers.0.linear_attn.A_log",
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise ValueError(f"unexpected layer-zero GDN {field}")
    for field in ("conv_sha256", "dt_bias_sha256", "a_log_sha256"):
        if not SHA256.fullmatch(str(model.get(field, ""))):
            raise ValueError(f"model.{field} must be a lowercase SHA-256 digest")

    operation = require_object(record, "operation")
    exact_operation = {
        "token_id": 42,
        "batch": 1,
        "tokens": 1,
        "conv_channels": 10240,
        "conv_kernel_width": 4,
        "key_heads": 16,
        "key_dim": 128,
        "value_heads": 48,
        "value_dim": 128,
        "input_dtype": 12,
        "state_dtype": 10,
        "epsilon": "0.000001",
        "initial_conv_state": "zero",
        "initial_delta_state": "zero",
        "fingerprint": "fnv1a64-little-endian-f32",
    }
    for field, expected in exact_operation.items():
        if operation.get(field) != expected:
            raise ValueError(f"layer-zero GDN operation.{field} changed")

    comparison = require_object(record, "comparison")
    if comparison.get("exact") is not True or comparison.get("stages") != list(
        LAYER0_GDN_STAGES
    ):
        raise ValueError("layer-zero GDN checkpoint does not declare exact comparison")

    expected_layout = {
        "conv_out": (12, 10240),
        "conv_state": (12, 30720),
        "a_exp": (10, 48),
        "gdn_out": (12, 6144),
        "delta_state": (10, 786432),
    }
    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    for stage in LAYER0_GDN_STAGES:
        generalist_stage = require_object(generalist, stage)
        ollama_stage = require_object(ollama, stage)
        if set(generalist_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"generalist.{stage} has an unexpected evidence shape")
        if set(ollama_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"ollama.{stage} has an unexpected evidence shape")
        expected_dtype, expected_elements = expected_layout[stage]
        if generalist_stage.get("dtype") != expected_dtype:
            raise ValueError(f"generalist.{stage} has an unexpected dtype")
        if generalist_stage.get("elements") != expected_elements:
            raise ValueError(f"generalist.{stage} has an unexpected element count")
        for field in NVFP4_EXACT_FIELDS:
            if generalist_stage.get(field) != ollama_stage.get(field):
                raise ValueError(f"Generalist/Ollama mismatch in {stage}.{field}")
        if not FINGERPRINT.fullmatch(str(generalist_stage.get("fingerprint", ""))):
            raise ValueError(f"{stage} fingerprint must be 64-bit lowercase hexadecimal")
        for field in EMBEDDING_EXACT_FIELDS - {"fingerprint"}:
            parse_finite_float(generalist_stage.get(field), f"generalist.{stage}.{field}")


def validate_layer0_full_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    expected_model = {
        "gdn_checkpoint": LAYER0_GDN_CHECKPOINT,
        "gdn_norm_name": "model.language_model.layers.0.linear_attn.norm.weight",
        "out_proj_name": "model.language_model.layers.0.linear_attn.out_proj.weight",
        "post_norm_name": "model.language_model.layers.0.post_attention_layernorm.weight",
        "mlp_gate_name": "model.language_model.layers.0.mlp.gate_proj.weight",
        "mlp_up_name": "model.language_model.layers.0.mlp.up_proj.weight",
        "mlp_down_name": "model.language_model.layers.0.mlp.down_proj.weight",
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise ValueError(f"unexpected complete layer-zero {field}")
    for field in (
        "gdn_norm_sha256",
        "out_proj_sha256",
        "post_norm_sha256",
        "mlp_gate_sha256",
        "mlp_up_sha256",
        "mlp_down_sha256",
    ):
        if not SHA256.fullmatch(str(model.get(field, ""))):
            raise ValueError(f"model.{field} must be a lowercase SHA-256 digest")

    operation = require_object(record, "operation")
    exact_operation = {
        "token_id": 42,
        "hidden_width": 5120,
        "gdn_width": 6144,
        "mlp_width": 17408,
        "input_dtype": 12,
        "output_dtype": 12,
        "epsilon": "0.000001",
        "shift_post_norm_weight_by_one": True,
        "initial_recurrent_state": "zero",
        "fingerprint": "fnv1a64-little-endian-f32",
    }
    for field, expected in exact_operation.items():
        if operation.get(field) != expected:
            raise ValueError(f"complete layer-zero operation.{field} changed")

    comparison = require_object(record, "comparison")
    if comparison.get("exact") is not True or comparison.get("stages") != list(
        LAYER0_FULL_STAGES
    ):
        raise ValueError("complete layer-zero checkpoint does not declare exact comparison")

    expected_elements = {
        "gdn_norm": 6144,
        "gated": 6144,
        "attention_out": 5120,
        "after_attention": 5120,
        "post_norm": 5120,
        "mlp_gate": 17408,
        "mlp_up": 17408,
        "swiglu": 17408,
        "mlp_out": 5120,
        "layer_out": 5120,
    }
    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    for stage in LAYER0_FULL_STAGES:
        generalist_stage = require_object(generalist, stage)
        ollama_stage = require_object(ollama, stage)
        if set(generalist_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"generalist.{stage} has an unexpected evidence shape")
        if set(ollama_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"ollama.{stage} has an unexpected evidence shape")
        if generalist_stage.get("dtype") != 12:
            raise ValueError(f"generalist.{stage} must remain BF16")
        if generalist_stage.get("elements") != expected_elements[stage]:
            raise ValueError(f"generalist.{stage} has an unexpected element count")
        for field in NVFP4_EXACT_FIELDS:
            if generalist_stage.get(field) != ollama_stage.get(field):
                raise ValueError(f"Generalist/Ollama mismatch in {stage}.{field}")
        if not FINGERPRINT.fullmatch(str(generalist_stage.get("fingerprint", ""))):
            raise ValueError(f"{stage} fingerprint must be 64-bit lowercase hexadecimal")
        for field in EMBEDDING_EXACT_FIELDS - {"fingerprint"}:
            parse_finite_float(generalist_stage.get(field), f"generalist.{stage}.{field}")


def validate_full_decode_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    expected_model = {
        "embedding_name": EXPECTED_EMBEDDING_TENSOR,
        "final_norm_name": "model.language_model.norm.weight",
        "lm_head_name": "lm_head.weight",
        "language_blob_count": 851,
        "loaded_array_count": 1653,
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise ValueError(f"full decode model.{field} changed")

    operation = require_object(record, "operation")
    exact_operation = {
        "token_id": 42,
        "output_token_id": 1219,
        "layer_count": 64,
        "linear_attention_layer_count": 48,
        "full_attention_layer_count": 16,
        "hidden_width": 5120,
        "vocabulary_width": 248320,
        "hidden_dtype": 12,
        "logits_dtype": 12,
        "initial_conv_state": "zero",
        "initial_delta_state": "zero",
        "initial_kv_cache": "empty",
        "full_attention_first_token_specialization": "one-key-softmax-is-one",
        "fingerprint": "fnv1a64-little-endian-f32",
    }
    for field, expected in exact_operation.items():
        if operation.get(field) != expected:
            raise ValueError(f"full decode operation.{field} changed")

    comparison = require_object(record, "comparison")
    if (
        comparison.get("exact") is not True
        or comparison.get("output_token_exact") is not True
        or comparison.get("stages") != list(FULL_DECODE_STAGES)
    ):
        raise ValueError("full decode checkpoint does not declare exact comparison")

    expected_layout = {"hidden": (12, 5120), "logits": (12, 248320)}
    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    if generalist.get("output_token_id") != 1219:
        raise ValueError("Generalist full decode output token changed")
    if generalist.get("output_token_id") != ollama.get("output_token_id"):
        raise ValueError("Generalist/Ollama mismatch in output_token_id")
    for stage in FULL_DECODE_STAGES:
        generalist_stage = require_object(generalist, stage)
        ollama_stage = require_object(ollama, stage)
        if set(generalist_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"generalist.{stage} has an unexpected evidence shape")
        if set(ollama_stage) != NVFP4_EXACT_FIELDS:
            raise ValueError(f"ollama.{stage} has an unexpected evidence shape")
        expected_dtype, expected_elements = expected_layout[stage]
        if generalist_stage.get("dtype") != expected_dtype:
            raise ValueError(f"generalist.{stage} has an unexpected dtype")
        if generalist_stage.get("elements") != expected_elements:
            raise ValueError(f"generalist.{stage} has an unexpected element count")
        for field in NVFP4_EXACT_FIELDS:
            if generalist_stage.get(field) != ollama_stage.get(field):
                raise ValueError(f"Generalist/Ollama mismatch in {stage}.{field}")
        if not FINGERPRINT.fullmatch(str(generalist_stage.get("fingerprint", ""))):
            raise ValueError(f"{stage} fingerprint must be 64-bit lowercase hexadecimal")
        for field in EMBEDDING_EXACT_FIELDS - {"fingerprint"}:
            parse_finite_float(generalist_stage.get(field), f"generalist.{stage}.{field}")

    performance = require_object(record, "performance_observation")
    for field in (
        "model_load_milliseconds",
        "cold_decode_milliseconds",
        "warm_decode_milliseconds",
    ):
        if parse_finite_float(performance.get(field), f"performance_observation.{field}") <= 0:
            raise ValueError(f"performance_observation.{field} must be positive")


def validate_cached_decode_checkpoint(record: dict[str, Any]) -> None:
    model, _runtime = validate_common_provenance(record)
    expected_model = {
        "embedding_name": EXPECTED_EMBEDDING_TENSOR,
        "final_norm_name": "model.language_model.norm.weight",
        "lm_head_name": "lm_head.weight",
        "language_blob_count": 851,
        "loaded_array_count": 1653,
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise ValueError(f"cached decode model.{field} changed")

    operation = require_object(record, "operation")
    exact_operation = {
        "input_token_ids": [42, 1219],
        "output_token_ids": [1219, 73],
        "positions": [0, 1],
        "layer_count": 64,
        "linear_attention_layer_count": 48,
        "full_attention_layer_count": 16,
        "hidden_width": 5120,
        "vocabulary_width": 248320,
        "hidden_dtype": 12,
        "logits_dtype": 12,
        "conv_state_shape": [1, 3, 10240],
        "conv_state_dtype": 12,
        "delta_state_shape": [1, 48, 128, 128],
        "delta_state_dtype": 10,
        "kv_buffer_shape": [1, 4, 256, 256],
        "kv_buffer_dtype": 12,
        "kv_growth_step": 256,
        "rope_theta": "10000000",
        "rope_dimension": 64,
        "attention_scale": "0.0625",
        "full_attention_path": "generic-cached-sdpa",
        "state_commit": "single-synchronized-transaction",
        "evaluated_roots_per_step": 129,
        "silu_execution": "mlx-compiled-shapeless-graph",
        "swiglu_execution": "mlx-compiled-shapeless-graph",
        "fingerprint": "fnv1a64-little-endian-f32",
    }
    for field, expected in exact_operation.items():
        if operation.get(field) != expected:
            raise ValueError(f"cached decode operation.{field} changed")

    comparison = require_object(record, "comparison")
    if (
        comparison.get("exact") is not True
        or comparison.get("state_transition_exact") is not True
        or comparison.get("cache_reset_exact") is not True
        or comparison.get("positions") != [0, 1]
        or comparison.get("stages") != list(FULL_DECODE_STAGES)
    ):
        raise ValueError("cached decode checkpoint does not declare exact comparison")

    generalist = require_object(record, "generalist")
    ollama = require_object(record, "ollama")
    generalist_steps = generalist.get("steps")
    ollama_steps = ollama.get("steps")
    if not isinstance(generalist_steps, list) or len(generalist_steps) != 2:
        raise ValueError("generalist.steps must contain exactly two cached positions")
    if not isinstance(ollama_steps, list) or len(ollama_steps) != 2:
        raise ValueError("ollama.steps must contain exactly two cached positions")

    expected_steps = (
        (0, 42, 1219, "6dfc18bc2b1c2044", "4efefa793523b51f"),
        (1, 1219, 73, "66245d3b40e34351", "f1c93f5eefb964c8"),
    )
    expected_layout = {"hidden": (12, 5120), "logits": (12, 248320)}
    expected_step_fields = {
        "position",
        "input_token_id",
        "output_token_id",
        "hidden",
        "logits",
    }
    for index, expected in enumerate(expected_steps):
        generalist_step = generalist_steps[index]
        ollama_step = ollama_steps[index]
        if not isinstance(generalist_step, dict) or set(generalist_step) != expected_step_fields:
            raise ValueError(f"generalist.steps[{index}] has an unexpected evidence shape")
        if not isinstance(ollama_step, dict) or set(ollama_step) != expected_step_fields:
            raise ValueError(f"ollama.steps[{index}] has an unexpected evidence shape")
        position, input_token, output_token, hidden_fingerprint, logits_fingerprint = expected
        expected_metadata = {
            "position": position,
            "input_token_id": input_token,
            "output_token_id": output_token,
        }
        for field, expected_value in expected_metadata.items():
            if generalist_step.get(field) != expected_value:
                raise ValueError(f"generalist.steps[{index}].{field} changed")
            if generalist_step.get(field) != ollama_step.get(field):
                raise ValueError(f"Generalist/Ollama mismatch in steps[{index}].{field}")
        expected_fingerprints = {
            "hidden": hidden_fingerprint,
            "logits": logits_fingerprint,
        }
        for stage in FULL_DECODE_STAGES:
            generalist_stage = require_object(generalist_step, stage)
            ollama_stage = require_object(ollama_step, stage)
            if set(generalist_stage) != NVFP4_EXACT_FIELDS:
                raise ValueError(
                    f"generalist.steps[{index}].{stage} has an unexpected evidence shape"
                )
            if set(ollama_stage) != NVFP4_EXACT_FIELDS:
                raise ValueError(
                    f"ollama.steps[{index}].{stage} has an unexpected evidence shape"
                )
            expected_dtype, expected_elements = expected_layout[stage]
            if generalist_stage.get("dtype") != expected_dtype:
                raise ValueError(f"generalist.steps[{index}].{stage} has an unexpected dtype")
            if generalist_stage.get("elements") != expected_elements:
                raise ValueError(
                    f"generalist.steps[{index}].{stage} has an unexpected element count"
                )
            if generalist_stage.get("fingerprint") != expected_fingerprints[stage]:
                raise ValueError(
                    f"generalist.steps[{index}].{stage} fingerprint changed"
                )
            for field in NVFP4_EXACT_FIELDS:
                if generalist_stage.get(field) != ollama_stage.get(field):
                    raise ValueError(
                        f"Generalist/Ollama mismatch in steps[{index}].{stage}.{field}"
                    )
            for field in EMBEDDING_EXACT_FIELDS - {"fingerprint"}:
                parse_finite_float(
                    generalist_stage.get(field),
                    f"generalist.steps[{index}].{stage}.{field}",
                )

    performance = require_object(record, "performance_observation")
    if performance.get("scope") != "two-token-greedy-cached-decode-with-full-tensor-summaries":
        raise ValueError("cached decode performance scope changed")
    if performance.get("checkpoint_summaries_included") is not True:
        raise ValueError("cached decode timings must disclose checkpoint summaries")
    if performance.get("timing_status") != "accepted":
        raise ValueError("cached decode evidence must come from an accepted timing run")
    positive_fields = (
        "clean_exact_zero_state_reference_milliseconds",
        "observed_exact_zero_state_median_milliseconds",
        "maximum_accepted_control_slowdown",
        "generalist_step0_milliseconds",
        "generalist_step1_milliseconds",
        "ollama_step0_milliseconds",
        "ollama_step1_milliseconds",
        "generalist_step1_tokens_per_second",
        "ollama_step1_tokens_per_second",
        "generalist_step1_speedup",
    )
    parsed_performance = {
        field: parse_finite_float(performance.get(field), f"performance_observation.{field}")
        for field in positive_fields
    }
    for field, value in parsed_performance.items():
        if value <= 0:
            raise ValueError(f"performance_observation.{field} must be positive")
    control_ratio = (
        parsed_performance["observed_exact_zero_state_median_milliseconds"]
        / parsed_performance["clean_exact_zero_state_reference_milliseconds"]
    )
    if control_ratio > parsed_performance["maximum_accepted_control_slowdown"]:
        raise ValueError("cached decode timing control is contaminated")
    generalist_step1 = parsed_performance["generalist_step1_milliseconds"]
    ollama_step1 = parsed_performance["ollama_step1_milliseconds"]
    if not math.isclose(
        parsed_performance["generalist_step1_tokens_per_second"],
        1000.0 / generalist_step1,
        rel_tol=5e-5,
    ):
        raise ValueError("Generalist cached decode throughput is inconsistent with latency")
    if not math.isclose(
        parsed_performance["ollama_step1_tokens_per_second"],
        1000.0 / ollama_step1,
        rel_tol=5e-5,
    ):
        raise ValueError("Ollama cached decode throughput is inconsistent with latency")
    if not math.isclose(
        parsed_performance["generalist_step1_speedup"],
        ollama_step1 / generalist_step1,
        rel_tol=5e-7,
    ):
        raise ValueError("cached decode speedup is inconsistent with measured latency")


def validate_metal_dispatch_checkpoint(record: dict[str, Any]) -> None:
    model, runtime = validate_common_provenance(record)
    expected_model = {
        "ollama_tag": "qwen3.8:27b-mlx",
        "language_blob_count": 851,
        "loaded_array_count": 1653,
        "layer_count": 64,
        "linear_attention_layer_count": 48,
        "full_attention_layer_count": 16,
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise ValueError(f"Metal dispatch model.{field} changed")
    expected_commits = {
        "ollama_commit": "0f25c31bd53b64dc3fcc8fce0bde954159a67a58",
        "mlx_commit": "3abd0fd6b3eb9d9d3a34cb65e8a2189c57260399",
        "mlx_c_commit": "fba4470b89073180056c9ea46c443051375f7399",
    }
    for field, expected in expected_commits.items():
        if runtime.get(field) != expected:
            raise ValueError(f"Metal dispatch runtime.{field} changed")

    oracle = require_object(record, "oracle")
    expected_positions = {
        "position_zero": (42, 1219, "6dfc18bc2b1c2044", "4efefa793523b51f"),
        "position_one": (1219, 73, "66245d3b40e34351", "f1c93f5eefb964c8"),
        "position_128": (13, 457, "ec8df38e485727c3", "b992c48f0705f8e4"),
    }
    for name, expected in expected_positions.items():
        position = require_object(oracle, name)
        actual = (
            position.get("input_token_id"),
            position.get("output_token_id"),
            position.get("hidden_fingerprint"),
            position.get("logits_fingerprint"),
        )
        if actual != expected:
            raise ValueError(f"Metal dispatch {name} oracle changed")
    expected_continuation = [
        73,
        3706,
        23102,
        288,
        37105,
        163283,
        627,
        288,
        728,
        13373,
        84,
        1110,
        469,
        39111,
        35306,
    ]
    if oracle.get("fifteen_token_continuation") != expected_continuation:
        raise ValueError("Metal dispatch continuation oracle changed")
    if oracle.get("position_1_through_127_continuation_fingerprint") != "e1185ef3ebbb60ae":
        raise ValueError("Metal dispatch long continuation oracle changed")

    protocol = require_object(record, "trace_protocol")
    warm_steps = protocol.get("warm_cached_steps")
    if warm_steps != 8 or protocol.get("step_boundary_pipeline") != "argmax_bfloat16":
        raise ValueError("Metal dispatch trace protocol changed")

    baseline = require_object(record, "dynamic_norm_baseline")
    candidate = require_object(record, "precomputed_norm_candidate")
    expected_counts = {
        "first_step_encoders": 84,
        "steady_step_encoders": 82,
    }
    for field, expected in expected_counts.items():
        if baseline.get(field) != expected or candidate.get(field) != expected:
            raise ValueError(f"Metal dispatch {field} changed")
    if candidate.get("shifted_norm_weight_count") != 161:
        raise ValueError("Metal dispatch shifted norm inventory changed")

    baseline_first = baseline.get("first_step_dispatches")
    baseline_steady = baseline.get("steady_step_dispatches")
    candidate_first = candidate.get("first_step_dispatches")
    candidate_steady = candidate.get("steady_step_dispatches")
    if (baseline_first, baseline_steady, candidate_first, candidate_steady) != (
        2794,
        2792,
        2472,
        2470,
    ):
        raise ValueError("Metal dispatch per-step counts changed")
    if baseline.get("eight_step_dispatches") != baseline_first + (warm_steps - 1) * baseline_steady:
        raise ValueError("Metal dispatch baseline total is inconsistent")
    if candidate.get("eight_step_dispatches") != candidate_first + (warm_steps - 1) * candidate_steady:
        raise ValueError("Metal dispatch candidate total is inconsistent")

    baseline_kernels = require_object(baseline, "eight_step_selected_kernel_counts")
    candidate_kernels = require_object(candidate, "eight_step_selected_kernel_counts")
    if baseline_kernels.get("vs_Addbfloat16") != 161 * warm_steps:
        raise ValueError("Metal dispatch baseline shifted additions changed")
    if candidate_kernels.get("vs_Addbfloat16") != 0:
        raise ValueError("Metal dispatch candidate still has shifted additions")
    if (
        baseline_kernels.get("v_copyfloat32bfloat16")
        - candidate_kernels.get("v_copyfloat32bfloat16")
        != 161 * warm_steps
    ):
        raise ValueError("Metal dispatch scalar cast reduction is inconsistent")
    expected_qmv = 353 * warm_steps
    if (
        baseline_kernels.get("nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0")
        != expected_qmv
        or candidate_kernels.get("nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0")
        != expected_qmv
    ):
        raise ValueError("Metal dispatch QMV inventory changed")

    effects = require_object(record, "effects")
    effect = require_object(effects, "precomputed_norm")
    removed = baseline_steady - candidate_steady
    if effect.get("dispatches_removed_per_cached_token") != removed or removed != 322:
        raise ValueError("Metal dispatch removal count is inconsistent")
    reduction = parse_finite_float(
        effect.get("steady_dispatch_reduction"), "effect.steady_dispatch_reduction"
    )
    if not math.isclose(reduction, removed / baseline_steady, rel_tol=1e-9):
        raise ValueError("Metal dispatch reduction ratio is inconsistent")
    baseline_gpu = parse_finite_float(
        baseline.get("median_gpu_busy_milliseconds"),
        "dynamic_norm_baseline.median_gpu_busy_milliseconds",
    )
    candidate_gpu = parse_finite_float(
        candidate.get("median_gpu_busy_milliseconds"),
        "precomputed_norm_candidate.median_gpu_busy_milliseconds",
    )
    cross_run_speedup = parse_finite_float(
        effect.get("cross_run_gpu_busy_speedup"), "effect.cross_run_gpu_busy_speedup"
    )
    if not math.isclose(cross_run_speedup, baseline_gpu / candidate_gpu, rel_tol=1e-5):
        raise ValueError("Metal dispatch cross-run GPU speedup is inconsistent")

    packed = require_object(record, "packed_mlp_candidate")
    exact_scale = require_object(record, "exact_scale_candidate")
    narrow_qmv = require_object(record, "narrow_qmv_candidate")
    expected_later_counts = {
        "packed_mlp_candidate": (packed, 68, 66, 2280, 2278),
        "exact_scale_candidate": (exact_scale, 68, 66, 1413, 1411),
        "narrow_qmv_candidate": (narrow_qmv, 68, 66, 1413, 1411),
    }
    for name, (stage, first_encoders, steady_encoders, first, steady) in (
        expected_later_counts.items()
    ):
        actual = (
            stage.get("first_step_encoders"),
            stage.get("steady_step_encoders"),
            stage.get("first_step_dispatches"),
            stage.get("steady_step_dispatches"),
        )
        if actual != (first_encoders, steady_encoders, first, steady):
            raise ValueError(f"Metal dispatch {name} per-step counts changed")
        if stage.get("eight_step_dispatches") != first + (warm_steps - 1) * steady:
            raise ValueError(f"Metal dispatch {name} total is inconsistent")

    if packed.get("packed_gate_up_layer_count") != 64:
        raise ValueError("Metal dispatch packed MLP inventory changed")
    if exact_scale.get("nvfp4_matvecs_per_cached_token") != 289:
        raise ValueError("Metal dispatch exact-scale matvec inventory changed")
    if (
        narrow_qmv.get("results_per_simdgroup") != 2
        or narrow_qmv.get("minimum_output_width") != 4096
        or narrow_qmv.get("nvfp4_matvecs_per_cached_token") != 289
        or narrow_qmv.get("narrow_matvecs_per_cached_token") != 257
        or narrow_qmv.get("wide_matvecs_per_cached_token") != 32
    ):
        raise ValueError("Metal dispatch narrow QMV inventory changed")
    packed_kernels = require_object(packed, "eight_step_selected_kernel_counts")
    exact_kernels = require_object(exact_scale, "eight_step_selected_kernel_counts")
    if packed_kernels.get("nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0") != 289 * warm_steps:
        raise ValueError("Metal dispatch packed QMV inventory changed")
    if (
        exact_kernels.get("nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0_hgs_exact")
        != 289 * warm_steps
    ):
        raise ValueError("Metal dispatch exact QMV inventory changed")
    narrow_kernels = require_object(narrow_qmv, "eight_step_selected_kernel_counts")
    if (
        narrow_kernels.get(
            "nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_r_2_batch_0_hgs_exact"
        )
        != 257 * warm_steps
        or narrow_kernels.get(
            "nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0_hgs_exact"
        )
        != 32 * warm_steps
    ):
        raise ValueError("Metal dispatch narrow QMV kernel inventory changed")
    for field in ("v_copybfloat16float32", "v_copyfloat32bfloat16"):
        if packed_kernels.get(field) - exact_kernels.get(field) != 288 * warm_steps:
            raise ValueError(f"Metal dispatch exact-scale {field} reduction changed")
    if (
        packed_kernels.get("vs_Multiplyfloat32")
        + packed_kernels.get("vv_Multiplyfloat32")
        - exact_kernels.get("vs_Multiplyfloat32")
        - exact_kernels.get("vv_Multiplyfloat32")
        != 288 * warm_steps
    ):
        raise ValueError("Metal dispatch exact-scale multiply reduction changed")

    packed_effect = require_object(effects, "packed_mlp")
    packed_removed = candidate_steady - packed.get("steady_step_dispatches")
    if (
        packed_effect.get("dispatches_removed_per_cached_token") != packed_removed
        or packed_removed != 192
        or packed_effect.get("qmv_dispatches_removed_per_cached_token") != 64
    ):
        raise ValueError("Metal dispatch packed MLP effect is inconsistent")
    packed_reduction = parse_finite_float(
        packed_effect.get("steady_dispatch_reduction_from_precomputed_norm"),
        "effects.packed_mlp.steady_dispatch_reduction_from_precomputed_norm",
    )
    if not math.isclose(packed_reduction, packed_removed / candidate_steady, rel_tol=1e-9):
        raise ValueError("Metal dispatch packed MLP reduction ratio is inconsistent")

    exact_effect = require_object(effects, "exact_global_scale")
    exact_removed = packed.get("steady_step_dispatches") - exact_scale.get(
        "steady_step_dispatches"
    )
    if (
        exact_effect.get("dispatches_removed_per_cached_token") != exact_removed
        or exact_removed != 867
        or exact_effect.get("dispatches_removed_per_nvfp4_matvec") != 3
        or exact_removed != 3 * exact_scale.get("nvfp4_matvecs_per_cached_token")
    ):
        raise ValueError("Metal dispatch exact-scale effect is inconsistent")
    exact_reduction = parse_finite_float(
        exact_effect.get("steady_dispatch_reduction_from_packed_mlp"),
        "effects.exact_global_scale.steady_dispatch_reduction_from_packed_mlp",
    )
    if not math.isclose(
        exact_reduction, exact_removed / packed.get("steady_step_dispatches"), rel_tol=1e-9
    ):
        raise ValueError("Metal dispatch exact-scale reduction ratio is inconsistent")
    exact_gpu = parse_finite_float(
        exact_scale.get("median_gpu_busy_milliseconds"),
        "exact_scale_candidate.median_gpu_busy_milliseconds",
    )
    packed_gpu = parse_finite_float(
        packed.get("median_gpu_busy_milliseconds"),
        "packed_mlp_candidate.median_gpu_busy_milliseconds",
    )
    exact_gpu_speedup = parse_finite_float(
        exact_effect.get("cross_run_gpu_busy_speedup"),
        "effects.exact_global_scale.cross_run_gpu_busy_speedup",
    )
    if not math.isclose(exact_gpu_speedup, packed_gpu / exact_gpu, rel_tol=1e-5):
        raise ValueError("Metal dispatch exact-scale GPU speedup is inconsistent")

    qmv_effect = require_object(effects, "narrow_qmv_geometry")
    if (
        qmv_effect.get("dispatches_removed_per_cached_token") != 0
        or qmv_effect.get("matvecs_retiled_per_cached_token") != 257
    ):
        raise ValueError("Metal dispatch narrow QMV effect is inconsistent")
    narrow_gpu = parse_finite_float(
        narrow_qmv.get("median_gpu_busy_milliseconds"),
        "narrow_qmv_candidate.median_gpu_busy_milliseconds",
    )
    narrow_gpu_speedup = parse_finite_float(
        qmv_effect.get("cross_run_gpu_busy_speedup"),
        "effects.narrow_qmv_geometry.cross_run_gpu_busy_speedup",
    )
    if not math.isclose(narrow_gpu_speedup, exact_gpu / narrow_gpu, rel_tol=1e-5):
        raise ValueError("Metal dispatch narrow QMV GPU speedup is inconsistent")

    cumulative = require_object(effects, "cumulative")
    cumulative_removed = baseline_steady - narrow_qmv.get("steady_step_dispatches")
    if cumulative.get("dispatches_removed_per_cached_token") != cumulative_removed:
        raise ValueError("Metal dispatch cumulative removal is inconsistent")
    cumulative_reduction = parse_finite_float(
        cumulative.get("steady_dispatch_reduction_from_dynamic_norm"),
        "effects.cumulative.steady_dispatch_reduction_from_dynamic_norm",
    )
    if not math.isclose(cumulative_reduction, cumulative_removed / baseline_steady, rel_tol=1e-9):
        raise ValueError("Metal dispatch cumulative reduction ratio is inconsistent")

    wall_clock = require_object(record, "wall_clock_experiments")
    for name in ("precomputed_norm", "packed_mlp"):
        experiment = require_object(wall_clock, name)
        if experiment.get("focused_rounds") != 127 or experiment.get("timing_status") != "accepted":
            raise ValueError(f"Metal dispatch {name} focused experiment is invalid")
        candidate_time = parse_finite_float(
            experiment.get("candidate_median_milliseconds"),
            f"wall_clock_experiments.{name}.candidate_median_milliseconds",
        )
        baseline_time = parse_finite_float(
            experiment.get("baseline_median_milliseconds"),
            f"wall_clock_experiments.{name}.baseline_median_milliseconds",
        )
        median_speedup = parse_finite_float(
            experiment.get("median_speedup"),
            f"wall_clock_experiments.{name}.median_speedup",
        )
        if not math.isclose(median_speedup, baseline_time / candidate_time, rel_tol=1e-4):
            raise ValueError(f"Metal dispatch {name} median speedup is inconsistent")
        for field in ("paired_median_speedup", "paired_median_difference_microseconds"):
            parse_finite_float(experiment.get(field), f"wall_clock_experiments.{name}.{field}")
        if not 0 <= experiment.get("candidate_wins", -1) <= 127:
            raise ValueError(f"Metal dispatch {name} win count is invalid")
        for field in ("initial_control_milliseconds", "post_control_milliseconds"):
            if parse_finite_float(experiment.get(field), f"wall_clock_experiments.{name}.{field}") > (
                72.549 * 1.65
            ):
                raise ValueError(f"Metal dispatch {name} control is contaminated")

    replicas = wall_clock.get("exact_global_scale_replicas")
    if not isinstance(replicas, list) or len(replicas) != 3:
        raise ValueError("Metal dispatch evidence requires three exact-scale replicas")
    for index, replica in enumerate(replicas):
        if not isinstance(replica, dict) or replica.get("rounds") != 127:
            raise ValueError(f"Metal dispatch exact-scale replica {index} has an invalid shape")
        if replica.get("timing_status") != "accepted":
            raise ValueError(f"Metal dispatch exact-scale replica {index} is contaminated")
        candidate_time = parse_finite_float(
            replica.get("candidate_median_milliseconds"), f"replica[{index}].candidate"
        )
        baseline_time = parse_finite_float(
            replica.get("baseline_median_milliseconds"), f"replica[{index}].baseline"
        )
        median_speedup = parse_finite_float(
            replica.get("median_speedup"), f"replica[{index}].median_speedup"
        )
        paired_speedup = parse_finite_float(
            replica.get("paired_median_speedup"), f"replica[{index}].paired_speedup"
        )
        if not math.isclose(median_speedup, baseline_time / candidate_time, rel_tol=1e-4):
            raise ValueError(f"Metal dispatch exact-scale replica {index} speedup is inconsistent")
        if paired_speedup <= 1.02 or replica.get("candidate_wins", 0) < 120:
            raise ValueError(f"Metal dispatch exact-scale replica {index} did not reproduce the gain")
        for field in ("initial_control_milliseconds", "post_control_milliseconds"):
            if parse_finite_float(replica.get(field), f"replica[{index}].{field}") > 72.549 * 1.65:
                raise ValueError(f"Metal dispatch exact-scale replica {index} control is contaminated")

    qmv_replicas = wall_clock.get("narrow_qmv_replicas")
    if not isinstance(qmv_replicas, list) or len(qmv_replicas) != 4:
        raise ValueError("Metal dispatch evidence requires four narrow-QMV replicas")
    for index, replica in enumerate(qmv_replicas):
        if not isinstance(replica, dict) or replica.get("rounds") != 127:
            raise ValueError(f"Metal dispatch narrow-QMV replica {index} has an invalid shape")
        if replica.get("timing_status") != "accepted":
            raise ValueError(f"Metal dispatch narrow-QMV replica {index} is contaminated")
        candidate_time = parse_finite_float(
            replica.get("candidate_median_milliseconds"),
            f"narrow_qmv_replica[{index}].candidate",
        )
        baseline_time = parse_finite_float(
            replica.get("baseline_median_milliseconds"),
            f"narrow_qmv_replica[{index}].baseline",
        )
        median_speedup = parse_finite_float(
            replica.get("median_speedup"),
            f"narrow_qmv_replica[{index}].median_speedup",
        )
        paired_speedup = parse_finite_float(
            replica.get("paired_median_speedup"),
            f"narrow_qmv_replica[{index}].paired_speedup",
        )
        if not math.isclose(median_speedup, baseline_time / candidate_time, rel_tol=1e-4):
            raise ValueError(f"Metal dispatch narrow-QMV replica {index} speedup is inconsistent")
        if paired_speedup <= 1.005 or replica.get("candidate_wins", 0) < 100:
            raise ValueError(f"Metal dispatch narrow-QMV replica {index} did not reproduce the gain")
        for field in ("initial_control_milliseconds", "post_control_milliseconds"):
            if (
                parse_finite_float(
                    replica.get(field), f"narrow_qmv_replica[{index}].{field}"
                )
                > 72.549 * 1.65
            ):
                raise ValueError(
                    f"Metal dispatch narrow-QMV replica {index} control is contaminated"
                )

    promotion = require_object(wall_clock, "promotion_validation")
    if promotion.get("measured_rounds") != 7 or promotion.get("timing_status") != "accepted":
        raise ValueError("Metal dispatch promotion validation is invalid")
    cached_milliseconds = parse_finite_float(
        promotion.get("cached_argmax_median_milliseconds"),
        "wall_clock_experiments.promotion_validation.cached_argmax_median_milliseconds",
    )
    cached_throughput = parse_finite_float(
        promotion.get("cached_argmax_tokens_per_second"),
        "wall_clock_experiments.promotion_validation.cached_argmax_tokens_per_second",
    )
    if not math.isclose(cached_throughput, 1000.0 / cached_milliseconds, rel_tol=5e-5):
        raise ValueError("Metal dispatch promoted throughput is inconsistent")


def validate_qmv_geometry_sweep(record: dict[str, Any]) -> None:
    if not SHA256.fullmatch(str(record.get("manifest_sha256", ""))):
        raise ValueError("QMV geometry manifest must be a lowercase SHA-256 digest")
    if record.get("baseline_policy") != "r2 for output width >= 4096; r4 otherwise":
        raise ValueError("QMV geometry baseline policy changed")

    expected_shapes = [
        [16384, 5120, 48],
        [5120, 6144, 64],
        [34816, 5120, 64],
        [5120, 17408, 64],
        [12288, 5120, 16],
        [1024, 5120, 32],
        [248320, 5120, 1],
    ]
    if record.get("shape_columns") != ["output_width", "input_width", "calls_per_token"]:
        raise ValueError("QMV geometry shape columns changed")
    if record.get("shapes") != expected_shapes or sum(row[2] for row in expected_shapes) != 289:
        raise ValueError("QMV geometry shape inventory changed")

    protocol = require_object(record, "protocol")
    if (
        protocol.get("candidate_and_baseline_order") != "alternated every token"
        or protocol.get("all_runs_timing_status") != "accepted"
        or protocol.get("screen_rounds") != 15
        or protocol.get("focused_rounds") != 127
        or protocol.get("high_power_rounds") != 255
    ):
        raise ValueError("QMV geometry protocol changed")
    if parse_finite_float(protocol.get("control_limit_milliseconds"), "control limit") != 119.706:
        raise ValueError("QMV geometry control limit changed")
    for field in ("position_128_fingerprints", "position_256_fingerprints"):
        fingerprints = protocol.get(field)
        if (
            not isinstance(fingerprints, list)
            or len(fingerprints) != 3
            or any(not FINGERPRINT.fullmatch(str(value)) for value in fingerprints)
        ):
            raise ValueError(f"QMV geometry {field} changed")

    expected_columns = [
        "rounds",
        "output_width",
        "input_width",
        "results_per_simdgroup",
        "paired_median_speedup",
        "paired_median_difference_microseconds",
        "candidate_wins",
    ]
    if record.get("measurement_columns") != expected_columns:
        raise ValueError("QMV geometry measurement columns changed")

    shape_pairs = {(row[0], row[1]) for row in expected_shapes}

    def validate_rows(field: str, rounds: int, count: int) -> list[list[Any]]:
        rows = record.get(field)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"QMV geometry {field} row count changed")
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != len(expected_columns):
                raise ValueError(f"QMV geometry {field}[{index}] shape changed")
            if row[0] != rounds or (row[1], row[2]) not in shape_pairs or row[3] not in (1, 2, 4, 8):
                raise ValueError(f"QMV geometry {field}[{index}] identity changed")
            if parse_finite_float(row[4], f"{field}[{index}].speedup") <= 0:
                raise ValueError(f"QMV geometry {field}[{index}] speedup is invalid")
            parse_finite_float(row[5], f"{field}[{index}].difference")
            if not isinstance(row[6], int) or not 0 <= row[6] <= rounds:
                raise ValueError(f"QMV geometry {field}[{index}] win count is invalid")
        return rows

    screen = validate_rows("screen", 15, 21)
    focused = validate_rows("focused", 127, 9)
    high_power = validate_rows("high_power", 255, 3)
    expected_screen = {
        (output_width, input_width, results)
        for output_width, input_width, _calls in expected_shapes
        for results in ((1, 2, 8) if output_width == 1024 else (1, 4, 8))
    }
    if {(row[1], row[2], row[3]) for row in screen} != expected_screen:
        raise ValueError("QMV geometry screen coverage changed")
    if sum(row[1:4] == [248320, 5120, 1] for row in focused) != 4:
        raise ValueError("QMV geometry focused LM-head replication changed")
    if (
        any(row[1:4] != [248320, 5120, 1] for row in high_power)
        or sum(row[4] < 1.0 for row in high_power) != 2
    ):
        raise ValueError("QMV geometry high-power rejection changed")

    conclusion = require_object(record, "conclusion")
    if conclusion.get("promoted") is not False or conclusion.get("retained_policy") != record.get(
        "baseline_policy"
    ):
        raise ValueError("QMV geometry rejection conclusion changed")


def validate_mtp_depth_control(record: dict[str, Any]) -> None:
    model, runtime = validate_common_provenance(record)
    if (
        model.get("ollama_tag") != "qwen3.8:27b-mlx"
        or model.get("architecture") != "Qwen3_5ForConditionalGeneration"
        or model.get("dense") is not True
        or model.get("mtp_tensor_count") != 15
        or model.get("mtp_storage_bytes") != 238_934_656
    ):
        raise ValueError("MTP depth-control model identity changed")

    for field in ("fixed_depth_runner_sha256", "libmlxc_sha256", "metallib_sha256"):
        if not SHA256.fullmatch(str(runtime.get(field, ""))):
            raise ValueError(f"runtime.{field} must be a lowercase SHA-256 digest")
    if "OLLAMA_MLX_FIXED_DRAFT_DEPTH" not in str(runtime.get("instrumentation", "")):
        raise ValueError("MTP fixed-depth instrumentation is not identified")

    hardware = require_object(record, "hardware")
    if (
        hardware.get("model") != "MacBook Pro Mac16,8"
        or hardware.get("chip") != "Apple M4 Pro"
        or hardware.get("unified_memory_gib") != 48
        or hardware.get("cpu_cores") != 14
        or hardware.get("performance_cores") != 10
        or hardware.get("efficiency_cores") != 4
    ):
        raise ValueError("MTP depth-control hardware identity changed")

    protocol = require_object(record, "protocol")
    output_sha256 = str(protocol.get("output_sha256", ""))
    if (
        not SHA256.fullmatch(str(protocol.get("prompt_sha256", "")))
        or not SHA256.fullmatch(output_sha256)
        or protocol.get("prompt_tokens") != 36
        or protocol.get("generated_tokens") != 192
        or protocol.get("temperature") != 0
        or protocol.get("seed") != 42
        or protocol.get("replicas_per_mode") != 3
        or protocol.get("fixed_mode_warmup_tokens") != 64
        or protocol.get("adaptive_mode_warmup_tokens") != 192
        or protocol.get("mode_order")
        != ["fixed0", "fixed1", "fixed2", "fixed3", "adaptive"]
        or protocol.get("timing_status") != "accepted"
    ):
        raise ValueError("MTP depth-control protocol changed")

    expected_columns = [
        "eval_count",
        "eval_duration_ns",
        "tokens_per_second",
        "prompt_eval_duration_ns",
        "iterations",
        "drafted",
        "accepted",
        "acceptance",
        "avg_draft",
        "max_draft",
        "avg_accepted",
        "output_sha256",
    ]
    if record.get("measurement_columns") != expected_columns:
        raise ValueError("MTP depth-control measurement columns changed")

    modes = record.get("modes")
    if not isinstance(modes, list) or len(modes) != 5:
        raise ValueError("MTP depth-control mode inventory changed")
    expected_depths = {
        "fixed0": 0,
        "fixed1": 1,
        "fixed2": 2,
        "fixed3": 3,
        "adaptive": None,
    }
    medians: dict[str, float] = {}
    declared_speedups: dict[str, float] = {}
    for mode in modes:
        if not isinstance(mode, dict):
            raise ValueError("MTP depth-control mode must be an object")
        name = mode.get("name")
        if name not in expected_depths or mode.get("fixed_depth") != expected_depths[name]:
            raise ValueError("MTP depth-control mode identity changed")
        runs = mode.get("runs")
        if not isinstance(runs, list) or len(runs) != 3:
            raise ValueError(f"MTP depth-control {name} replica count changed")

        rates = []
        for index, row in enumerate(runs):
            if not isinstance(row, list) or len(row) != len(expected_columns):
                raise ValueError(f"MTP depth-control {name}[{index}] shape changed")
            (
                eval_count,
                eval_duration,
                raw_rate,
                prompt_duration,
                iterations,
                drafted,
                accepted,
                raw_acceptance,
                raw_avg_draft,
                max_draft,
                raw_avg_accepted,
                output_hash,
            ) = row
            if eval_count != 192 or not isinstance(eval_duration, int) or eval_duration <= 0:
                raise ValueError(f"MTP depth-control {name}[{index}] timing identity changed")
            if not isinstance(prompt_duration, int) or prompt_duration < 0:
                raise ValueError(f"MTP depth-control {name}[{index}] prompt timing is invalid")
            if not isinstance(iterations, int) or not 0 < iterations <= eval_count:
                raise ValueError(f"MTP depth-control {name}[{index}] iterations are invalid")
            if (
                not isinstance(drafted, int)
                or not isinstance(accepted, int)
                or drafted < 0
                or not 0 <= accepted <= drafted
            ):
                raise ValueError(f"MTP depth-control {name}[{index}] acceptance counts are invalid")
            if not isinstance(max_draft, int) or max_draft < 0:
                raise ValueError(f"MTP depth-control {name}[{index}] max draft is invalid")
            if name != "adaptive" and max_draft > expected_depths[name]:
                raise ValueError(f"MTP depth-control {name}[{index}] exceeded its fixed depth")
            if output_hash != output_sha256:
                raise ValueError("MTP depth-control semantic output mismatch")

            rate = parse_finite_float(raw_rate, f"{name}[{index}].tokens_per_second")
            expected_rate = eval_count / (eval_duration / 1_000_000_000)
            if not math.isclose(rate, expected_rate, rel_tol=1e-12):
                raise ValueError(f"MTP depth-control {name}[{index}] throughput is inconsistent")
            rates.append(rate)

            acceptance = parse_finite_float(raw_acceptance, f"{name}[{index}].acceptance")
            avg_draft = parse_finite_float(raw_avg_draft, f"{name}[{index}].avg_draft")
            avg_accepted = parse_finite_float(
                raw_avg_accepted, f"{name}[{index}].avg_accepted"
            )
            expected_acceptance = accepted / drafted if drafted else 0.0
            if not math.isclose(acceptance, expected_acceptance, abs_tol=0.006):
                raise ValueError(f"MTP depth-control {name}[{index}] acceptance is inconsistent")
            if not math.isclose(avg_draft, drafted / iterations, abs_tol=0.006):
                raise ValueError(f"MTP depth-control {name}[{index}] average draft is inconsistent")
            if not math.isclose(avg_accepted, accepted / iterations, abs_tol=0.006):
                raise ValueError(
                    f"MTP depth-control {name}[{index}] average accepted is inconsistent"
                )

        median = sorted(rates)[1]
        declared_median = parse_finite_float(
            mode.get("median_tokens_per_second"), f"{name}.median_tokens_per_second"
        )
        if not math.isclose(declared_median, median, rel_tol=1e-12):
            raise ValueError(f"MTP depth-control {name} median is inconsistent")
        medians[name] = median
        declared_speedups[name] = parse_finite_float(
            mode.get("median_speedup_over_fixed0"),
            f"{name}.median_speedup_over_fixed0",
        )

    if set(medians) != set(expected_depths):
        raise ValueError("MTP depth-control mode coverage changed")
    baseline = medians["fixed0"]
    for name, median in medians.items():
        if not math.isclose(
            declared_speedups[name], median / baseline, abs_tol=0.000_01
        ):
            raise ValueError(f"MTP depth-control {name} speedup is inconsistent")

    if (
        medians["fixed1"] / baseline < 1.5
        or medians["fixed2"] / baseline < 1.79
        or medians["fixed3"] >= medians["fixed2"]
        or medians["adaptive"] / baseline < 1.78
    ):
        raise ValueError("MTP depth-control gain or depth optimum changed")

    conclusion = require_object(record, "conclusion")
    prior = parse_finite_float(
        conclusion.get("generalist_prior_tokens_per_second"),
        "conclusion.generalist_prior_tokens_per_second",
    )
    declared_baseline = parse_finite_float(
        conclusion.get("ollama_fixed0_median_tokens_per_second"),
        "conclusion.ollama_fixed0_median_tokens_per_second",
    )
    if (
        conclusion.get("base_decode_parity") is not True
        or not math.isclose(declared_baseline, baseline, rel_tol=1e-12)
        or abs(prior / baseline - 1.0) > 0.02
        or conclusion.get("priority")
        != "implement MTP drafting plus fused multi-token target verification in Generalist"
    ):
        raise ValueError("MTP depth-control conclusion changed")


def validate_native_speculative_decode(record: dict[str, Any]) -> None:
    model, runtime = validate_common_provenance(record)
    if (
        model.get("ollama_tag") != "qwen3.8:27b-mlx"
        or model.get("architecture") != "Qwen3_5ForConditionalGeneration"
        or model.get("dense") is not True
        or model.get("language_layer_count") != 64
        or model.get("mtp_layer_count") != 1
    ):
        raise ValueError("native speculative model identity changed")
    expected_commits = {
        "ollama_commit": "0f25c31bd53b64dc3fcc8fce0bde954159a67a58",
        "mlx_commit": "3abd0fd6b3eb9d9d3a34cb65e8a2189c57260399",
        "mlx_c_commit": "fba4470b89073180056c9ea46c443051375f7399",
    }
    for field, expected in expected_commits.items():
        if runtime.get(field) != expected:
            raise ValueError(f"native speculative runtime.{field} changed")
    if not SHA256.fullmatch(str(runtime.get("metallib_sha256", ""))):
        raise ValueError("native speculative metallib hash is invalid")

    hardware = require_object(record, "hardware")
    if (
        hardware.get("model") != "MacBook Pro Mac16,8"
        or hardware.get("chip") != "Apple M4 Pro"
        or hardware.get("unified_memory_gib") != 48
    ):
        raise ValueError("native speculative hardware identity changed")

    protocol = require_object(record, "protocol")
    output_hash = str(protocol.get("output_token_sha256", ""))
    if (
        protocol.get("prompt_sha256")
        != "231763bebfa8d71fb89820343d150cefddab3554783946289c1e0f54a9be3bf4"
        or protocol.get("prompt_tokens") != 101
        or protocol.get("generated_tokens") != 192
        or protocol.get("timed_decode_tokens") != 191
        or protocol.get("replicas_per_geometry") != 3
        or protocol.get("mode_order") != "rotated one position per replica"
        or protocol.get("decoding") != "greedy argmax"
        or protocol.get("semantic_contract")
        != "every complete token-id stream equals ordinary greedy target decoding"
        or not SHA256.fullmatch(output_hash)
    ):
        raise ValueError("native speculative protocol changed")
    reasoning = require_object(protocol, "reasoning")
    if (
        reasoning.get("enabled") is not True
        or reasoning.get("effort") != "medium"
        or reasoning.get("think_template") is not True
    ):
        raise ValueError("native speculative reasoning protocol changed")

    prefill = require_object(record, "prefill_experiment")
    baseline_prefill = require_object(prefill, "baseline")
    candidate_prefill = require_object(prefill, "candidate")
    ollama_prefill = require_object(prefill, "ollama_reference")
    if (
        baseline_prefill.get("maximum_chunk_tokens") != 8
        or candidate_prefill.get("maximum_chunk_tokens") != 512
        or candidate_prefill.get("actual_chunks") != 1
        or candidate_prefill.get("output_exact") is not True
        or ollama_prefill.get("same_prompt") is not True
    ):
        raise ValueError("native speculative prefill protocol changed")
    baseline_rate = parse_finite_float(
        baseline_prefill.get("target_prompt_tokens_per_second"),
        "prefill_experiment.baseline.target_prompt_tokens_per_second",
    )
    candidate_rate = parse_finite_float(
        candidate_prefill.get("target_prompt_tokens_per_second"),
        "prefill_experiment.candidate.target_prompt_tokens_per_second",
    )
    ollama_rate = parse_finite_float(
        ollama_prefill.get("target_prompt_tokens_per_second"),
        "prefill_experiment.ollama_reference.target_prompt_tokens_per_second",
    )
    declared_prefill_speedup = parse_finite_float(
        prefill.get("candidate_speedup"), "prefill_experiment.candidate_speedup"
    )
    if not math.isclose(
        declared_prefill_speedup, candidate_rate / baseline_rate, rel_tol=1e-6
    ):
        raise ValueError("native speculative prefill speedup is inconsistent")
    if declared_prefill_speedup < 2.5 or candidate_rate < ollama_rate:
        raise ValueError("native speculative prefill gain regressed")

    stages = require_object(record, "stage_profile")
    if (
        stages.get("fixed_depth") != 2
        or stages.get("generated_tokens") != 64
        or stages.get("verification_qmv_results_per_simdgroup") != 2
    ):
        raise ValueError("native speculative stage profile changed")
    total = parse_finite_float(stages.get("total_milliseconds"), "stage_profile.total")
    stage_values = [
        parse_finite_float(stages.get(field), f"stage_profile.{field}")
        for field in (
            "proposal_milliseconds",
            "verification_milliseconds",
            "replay_milliseconds",
            "correction_milliseconds",
        )
    ]
    if total <= 0 or any(value < 0 for value in stage_values):
        raise ValueError("native speculative stage timing is invalid")
    accounted = sum(stage_values)
    if not 0.98 <= accounted / total <= 1.01:
        raise ValueError("native speculative stage accounting is inconsistent")
    declared_verification_share = parse_finite_float(
        stages.get("verification_share"), "stage_profile.verification_share"
    )
    if not math.isclose(
        declared_verification_share, stage_values[1] / total, rel_tol=1e-6
    ):
        raise ValueError("native speculative verification share is inconsistent")
    if declared_verification_share < 0.85:
        raise ValueError("native speculative target verification is no longer the bottleneck")

    geometry = require_object(record, "geometry_experiment")
    if (
        geometry.get("fixed_depth") != 2
        or geometry.get("baseline_results_per_simdgroup") != 2
        or geometry.get("candidate_results_per_simdgroup") != 4
        or geometry.get("measurement_columns")
        != [
            "r2_decode_milliseconds",
            "r4_decode_milliseconds",
            "r2_verification_milliseconds",
            "r4_verification_milliseconds",
        ]
        or geometry.get("timing_status") != "accepted paired interleaved"
    ):
        raise ValueError("native speculative geometry protocol changed")
    rows = geometry.get("runs")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("native speculative geometry replica count changed")
    decode_ratios: list[float] = []
    verification_ratios: list[float] = []
    r2_durations: list[float] = []
    r4_durations: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"native speculative geometry run {index} shape changed")
        values = [
            parse_finite_float(value, f"geometry_experiment.runs[{index}]")
            for value in row
        ]
        if any(value <= 0 for value in values):
            raise ValueError(f"native speculative geometry run {index} is invalid")
        r2_decode, r4_decode, r2_verify, r4_verify = values
        if r4_decode >= r2_decode or r4_verify >= r2_verify:
            raise ValueError("native speculative r4 did not win every paired replica")
        r2_durations.append(r2_decode)
        r4_durations.append(r4_decode)
        decode_ratios.append(r2_decode / r4_decode)
        verification_ratios.append(r2_verify / r4_verify)

    decode_speedup = sorted(decode_ratios)[1]
    verification_speedup = sorted(verification_ratios)[1]
    declared_decode_speedup = parse_finite_float(
        geometry.get("paired_median_decode_speedup"),
        "geometry_experiment.paired_median_decode_speedup",
    )
    declared_verification_speedup = parse_finite_float(
        geometry.get("paired_median_verification_speedup"),
        "geometry_experiment.paired_median_verification_speedup",
    )
    if not math.isclose(declared_decode_speedup, decode_speedup, rel_tol=1e-6):
        raise ValueError("native speculative paired decode speedup is inconsistent")
    if not math.isclose(
        declared_verification_speedup, verification_speedup, rel_tol=1e-6
    ):
        raise ValueError("native speculative paired verification speedup is inconsistent")
    if declared_decode_speedup < 1.02 or declared_verification_speedup < 1.025:
        raise ValueError("native speculative r4 gain regressed")

    timed_tokens = protocol["timed_decode_tokens"]
    r2_rate = timed_tokens / (sorted(r2_durations)[1] / 1000.0)
    r4_rate = timed_tokens / (sorted(r4_durations)[1] / 1000.0)
    if not math.isclose(
        parse_finite_float(
            geometry.get("r2_median_tokens_per_second"),
            "geometry_experiment.r2_median_tokens_per_second",
        ),
        r2_rate,
        rel_tol=1e-6,
    ) or not math.isclose(
        parse_finite_float(
            geometry.get("r4_median_tokens_per_second"),
            "geometry_experiment.r4_median_tokens_per_second",
        ),
        r4_rate,
        rel_tol=1e-6,
    ):
        raise ValueError("native speculative geometry throughput is inconsistent")
    drafted = geometry.get("drafted")
    accepted = geometry.get("accepted")
    acceptance = parse_finite_float(geometry.get("acceptance"), "geometry.acceptance")
    if (
        drafted != 128
        or accepted != 126
        or not math.isclose(acceptance, accepted / drafted, rel_tol=1e-9)
        or geometry.get("candidate_decode_wins") != 3
        or geometry.get("candidate_verification_wins") != 3
        or geometry.get("output_token_sha256") != output_hash
        or geometry.get("complete_output_exact") is not True
    ):
        raise ValueError("native speculative geometry semantic evidence changed")

    conclusion = require_object(record, "conclusion")
    if (
        conclusion.get("default_fixed_depth") != 2
        or conclusion.get("ordinary_decode_results_per_simdgroup") != 2
        or conclusion.get("batched_verification_results_per_simdgroup") != 4
        or conclusion.get("promotion_scope")
        != "multi-row target verification and rejection replay only"
        or conclusion.get("dominant_next_bottleneck") != "batched target verification"
        or conclusion.get("reasoning_remains_enabled") is not True
    ):
        raise ValueError("native speculative conclusion changed")


def validate_adaptive_production(record: dict[str, Any]) -> None:
    model, runtime = validate_common_provenance(record)
    if (
        model.get("dense") is not True
        or model.get("language_layer_count") != 64
        or model.get("mtp_layer_count") != 1
        or model.get("vocabulary_rows") != 248320
    ):
        raise ValueError("adaptive production model contract changed")
    for field in (
        "metallib_sha256",
        "speculative_benchmark_sha256",
        "provider_benchmark_sha256",
    ):
        if not SHA256.fullmatch(str(runtime.get(field, ""))):
            raise ValueError(f"runtime.{field} must be a lowercase SHA-256 digest")
    if runtime.get("patch_count") != 6:
        raise ValueError("adaptive production patch count changed")

    head = require_object(record, "prompt_adaptive_head")
    protocol = require_object(head, "protocol")
    output_hash = str(protocol.get("output_token_sha256", ""))
    if (
        protocol.get("prompt_profile") != "multilingual"
        or not SHA256.fullmatch(str(protocol.get("prompt_sha256", "")))
        or protocol.get("prompt_tokens") != 97
        or protocol.get("generated_tokens") != 64
        or protocol.get("timed_decode_tokens") != 63
        or protocol.get("replicas_per_mode") != 3
        or protocol.get("mode_order") != "rotated one position per replica"
        or protocol.get("reasoning_enabled") is not True
        or protocol.get("reasoning_effort") != "xhigh"
        or protocol.get("semantic_contract")
        != "every complete token-id stream equals ordinary greedy target decoding"
        or not SHA256.fullmatch(output_hash)
        or protocol.get("timing_status") != "stable"
    ):
        raise ValueError("adaptive draft-head protocol changed")

    modes = head.get("modes")
    if not isinstance(modes, list) or len(modes) != 2:
        raise ValueError("adaptive draft-head modes changed")
    expected_modes = [
        ("fixed4-wide-draft-full-rollback", 0, False),
        ("fixed4-wide-draft65k-prompt-rollback", 65536, True),
    ]
    durations: list[list[float]] = []
    spreads: list[float] = []
    for mode_index, (mode, expected) in enumerate(zip(modes, expected_modes)):
        if not isinstance(mode, dict):
            raise ValueError("adaptive draft-head mode must be an object")
        name, vocab_limit, prompt_rows = expected
        if (
            mode.get("name") != name
            or mode.get("draft_vocab_limit") != vocab_limit
            or mode.get("prompt_extra_rows") is not prompt_rows
            or (mode_index == 1 and mode.get("prompt_extra_row_count") != 25)
        ):
            raise ValueError("adaptive draft-head mode configuration changed")
        runs = mode.get("runs")
        if not isinstance(runs, list) or len(runs) != 3:
            raise ValueError("adaptive draft-head replica count changed")
        mode_durations: list[float] = []
        for run_index, run in enumerate(runs):
            if not isinstance(run, dict):
                raise ValueError("adaptive draft-head run must be an object")
            duration = parse_finite_float(
                run.get("decode_duration_ns"),
                f"prompt_adaptive_head.modes[{mode_index}].runs[{run_index}].decode_duration_ns",
            )
            prefill = parse_finite_float(
                run.get("prefill_duration_ns"),
                f"prompt_adaptive_head.modes[{mode_index}].runs[{run_index}].prefill_duration_ns",
            )
            stages = [
                parse_finite_float(run.get(field), field)
                for field in (
                    "proposal_milliseconds",
                    "verification_milliseconds",
                    "replay_milliseconds",
                    "correction_milliseconds",
                )
            ]
            if (
                duration <= 0
                or prefill <= 0
                or any(value < 0 for value in stages)
                or not 0.98 <= sum(stages) / (duration / 1e6) <= 1.01
            ):
                raise ValueError("adaptive draft-head timing is invalid")
            if (
                run.get("drafted") != 60
                or run.get("accepted") != 47
                or run.get("iterations") != 16
                or run.get("output_token_sha256") != output_hash
            ):
                raise ValueError("adaptive draft-head semantic output mismatch")
            mode_durations.append(duration)
        durations.append(mode_durations)
        median_duration = sorted(mode_durations)[1]
        expected_rate = protocol["timed_decode_tokens"] / (median_duration / 1e9)
        declared_rate = parse_finite_float(
            mode.get("median_tokens_per_second"), "median_tokens_per_second"
        )
        if not math.isclose(declared_rate, expected_rate, rel_tol=1e-9):
            raise ValueError("adaptive draft-head median throughput is inconsistent")
        spread = max(mode_durations) / min(mode_durations)
        declared_spread = parse_finite_float(
            mode.get("decode_duration_spread_ratio"), "decode_duration_spread_ratio"
        )
        if not math.isclose(declared_spread, spread, rel_tol=1e-6):
            raise ValueError("adaptive draft-head spread is inconsistent")
        spreads.append(spread)

    declared_max_spread = parse_finite_float(
        protocol.get("maximum_decode_spread_ratio"), "maximum_decode_spread_ratio"
    )
    if (
        not math.isclose(declared_max_spread, max(spreads), rel_tol=1e-6)
        or declared_max_spread > 1.10
    ):
        raise ValueError("adaptive draft-head timing stability regressed")
    paired = sorted(
        baseline / candidate
        for baseline, candidate in zip(durations[0], durations[1])
    )[1]
    declared_paired = parse_finite_float(
        modes[1].get("median_paired_speedup_over_full"),
        "median_paired_speedup_over_full",
    )
    if not math.isclose(declared_paired, paired, rel_tol=1e-9) or paired < 1.05:
        raise ValueError("adaptive draft-head paired speedup regressed")

    conclusion = require_object(head, "conclusion")
    if (
        conclusion.get("promoted") is not True
        or conclusion.get("target_vocabulary_complete") is not True
        or conclusion.get("production_draft_vocab_limit") != 65536
        or conclusion.get("maximum_prompt_extra_rows") != 256
        or conclusion.get("sustained_uncovered_ids_fall_back_to_full_draft_head")
        is not True
    ):
        raise ValueError("adaptive draft-head promotion contract changed")

    provider = require_object(record, "provider_path")
    provider_protocol = require_object(provider, "protocol")
    if (
        provider_protocol.get("provider") != "EmbeddedProvider::qwen"
        or provider_protocol.get("adaptive_depths") != [2, 3, 4]
        or provider_protocol.get("requested_output_tokens") != 128
        or provider_protocol.get("reasoning_enabled") is not True
        or provider_protocol.get("reasoning_effort") != "xhigh"
        or provider_protocol.get("cancellation_requested_after_milliseconds") != 50
    ):
        raise ValueError("adaptive provider protocol changed")
    measurement = require_object(provider, "measurement")
    total = parse_finite_float(measurement.get("total_milliseconds"), "provider.total")
    first_delta = parse_finite_float(
        measurement.get("first_delta_milliseconds"), "provider.first_delta"
    )
    output_tokens = measurement.get("output_tokens")
    end_to_end_rate = parse_finite_float(
        measurement.get("end_to_end_output_tokens_per_second"), "provider.end_to_end_rate"
    )
    post_first_rate = parse_finite_float(
        measurement.get("approximate_post_first_delta_tokens_per_second"),
        "provider.post_first_rate",
    )
    if (
        output_tokens != 128
        or not 0 < first_delta < total
        or not math.isclose(end_to_end_rate, output_tokens / (total / 1000), rel_tol=1e-9)
        or not math.isclose(
            post_first_rate,
            (output_tokens - 1) / ((total - first_delta) / 1000),
            rel_tol=1e-9,
        )
        or measurement.get("reasoning_bytes", 0) <= 0
        or measurement.get("stop_reason") != "MaxTokens"
        or not SHA256.fullmatch(str(measurement.get("content_sha256", "")))
    ):
        raise ValueError("adaptive provider measurement is inconsistent")

    prefix = require_object(record, "prefix_cache")
    prefix_protocol = require_object(prefix, "protocol")
    prefix_measurement = require_object(prefix, "measurement")
    if (
        prefix_protocol.get("prompt_tokens") != 256
        or prefix_protocol.get("extension_tokens") != 64
        or prefix_protocol.get("chunk_tokens") != 64
        or prefix_protocol.get("cache_read_tokens") != 255
        or prefix_protocol.get("generated_tokens_checked") != 8
        or prefix_measurement.get("complete_output_exact") is not True
    ):
        raise ValueError("adaptive prefix-cache protocol changed")
    full_extended = parse_finite_float(
        prefix_measurement.get("full_extended_milliseconds"), "prefix.full_extended"
    )
    cached_extended = parse_finite_float(
        prefix_measurement.get("cached_extended_milliseconds"), "prefix.cached_extended"
    )
    cache_speedup = parse_finite_float(
        prefix_measurement.get("cache_speedup"), "prefix.cache_speedup"
    )
    longest_transaction = parse_finite_float(
        prefix_measurement.get("longest_native_transaction_milliseconds"),
        "prefix.longest_transaction",
    )
    if (
        not math.isclose(cache_speedup, full_extended / cached_extended, rel_tol=1e-9)
        or cache_speedup < 4.0
        or longest_transaction > 600
    ):
        raise ValueError("adaptive prefix-cache performance regressed")
    cancellation = require_object(provider, "cancellation")
    cancellation_latency = parse_finite_float(
        cancellation.get("worker_shutdown_milliseconds"), "provider.cancellation"
    )
    if (
        cancellation.get("deltas_before_cancellation") != 0
        or cancellation_latency <= 0
        or cancellation_latency > longest_transaction
    ):
        raise ValueError("adaptive provider cancellation bound regressed")


def validate_embedded_ollama(record: dict[str, Any]) -> None:
    model, runtime = validate_common_provenance(record)
    expected_model = {
        "ollama_tag": "qwen3.8:27b-mlx",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "dense": True,
        "manifest_sha256": "5642e97495e1a088883805981563dcdc4a040c2f53388b7a41d1f24d3622cf7e",
        "language_layer_count": 64,
        "mtp_layer_count": 1,
        "vocabulary_rows": 248320,
    }
    if model != expected_model:
        raise ValueError("embedded/Ollama model contract changed")
    for field in (
        "metallib_sha256",
        "benchmark_source_sha256",
        "benchmark_binary_sha256",
        "captured_result_sha256",
    ):
        if not SHA256.fullmatch(str(runtime.get(field, ""))):
            raise ValueError(f"runtime.{field} must be a lowercase SHA-256 digest")
    if runtime.get("ollama_version") != "0.32.13" or runtime.get("patch_count") != 6:
        raise ValueError("embedded/Ollama runtime contract changed")

    hardware = require_object(record, "hardware")
    if hardware != {
        "model": "MacBook Pro Mac16,8",
        "chip": "Apple M4 Pro",
        "unified_memory_gib": 48,
        "macos": "26.5.2 (25F84)",
    }:
        raise ValueError("embedded/Ollama hardware provenance changed")

    boundary = require_object(record, "evidence_boundary")
    if boundary != {
        "benchmark_class": "gated-normal-desktop-load",
        "continuous_inference_telemetry": False,
        "thermal_pressure_recorded": False,
        "per_process_background_limit_enforced": False,
        "interpretation": "paired replicated performance under measured normal desktop load; not a clean-room maximum-throughput result",
    }:
        raise ValueError("embedded/Ollama evidence boundary changed")

    protocol = require_object(record, "protocol")
    sampling = require_object(protocol, "sampling")
    environment = require_object(protocol, "environment_contract")
    ollama_model = require_object(protocol, "ollama_model")
    if (
        protocol.get("model") != "qwen3.8:27b-mlx"
        or protocol.get("prompt_source")
        != "Generalist production Qwen renderer; exact bytes sent to Ollama /api/generate with raw=true"
        or protocol.get("prompt_sha256")
        != "ea4ff9e96263f0e1232c1d5053c63be21891ddeef0db2640b524968e32f48179"
        or protocol.get("prompt_bytes") != 632
        or protocol.get("reasoning_effort") != "xhigh"
        or protocol.get("reasoning_required") is not True
        or protocol.get("ollama_think") is not True
        or protocol.get("ollama_raw_prompt") is not True
        or protocol.get("context_tokens") != 32768
        or protocol.get("requested_output_tokens") != 128
        or protocol.get("replicas") != 3
        or protocol.get("order") != "AB/BA rotating by replica"
        or protocol.get("cooldown_milliseconds") != 5000
        or protocol.get("warm_measurement")
        != "second identical request with the model resident and prompt prefix reusable"
        or protocol.get("residency_isolation")
        != "every engine sample runs in a fresh child process; process exit releases embedded MLX allocations; Ollama is explicitly unloaded before and after use"
        or protocol.get("content_contract")
        != "both engines must emit reasoning and consume the fixed token budget; hashes are recorded as semantic-stability evidence but bit identity is not required"
    ):
        raise ValueError("embedded/Ollama benchmark protocol changed")
    if sampling != {
        "min_p": 0,
        "mode": "greedy",
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
    } or protocol.get("seed") != 1:
        raise ValueError("embedded/Ollama sampling protocol changed")
    if environment != {
        "maximum_post_process_gpu_memory_bytes": 2147483648,
        "maximum_pre_sample_gpu_utilization_percent": 5,
        "minimum_pre_sample_cpu_idle_percent": 80.0,
        "minimum_pre_sample_memory_free_percent": 50.0,
    }:
        raise ValueError("embedded/Ollama environment contract changed")
    if (
        ollama_model.get("digest") != expected_model["manifest_sha256"]
        or ollama_model.get("model") != "qwen3.8:27b-mlx"
        or ollama_model.get("name") != "qwen3.8:27b-mlx"
        or ollama_model.get("size") != 18174721847
        or require_object(ollama_model, "details").get("quantization_level") != "nvfp4"
        or "thinking" not in ollama_model.get("capabilities", [])
    ):
        raise ValueError("embedded/Ollama inspected model metadata changed")

    samples = record.get("samples")
    expected_order = [
        (1, 1, "embedded"),
        (1, 2, "ollama"),
        (2, 1, "ollama"),
        (2, 2, "embedded"),
        (3, 1, "embedded"),
        (3, 2, "ollama"),
    ]
    if not isinstance(samples, list) or len(samples) != len(expected_order):
        raise ValueError("embedded/Ollama sample set changed")

    rates: dict[str, list[float]] = {"embedded": [], "ollama": []}
    end_to_end_rates: dict[str, list[float]] = {"embedded": [], "ollama": []}
    first_deltas: dict[str, list[float]] = {"embedded": [], "ollama": []}
    by_replica: dict[int, dict[str, tuple[float, float]]] = {}
    environment_verified = True
    residency_verified = True
    expected_hashes = {
        ("embedded", "cold"): "d2124f21180b9fe2c06febfec3b647561c02a9267e2326e604470db011dd561d",
        ("embedded", "warm"): "d2124f21180b9fe2c06febfec3b647561c02a9267e2326e604470db011dd561d",
        ("ollama", "cold"): "2fe91be32df5463ffd2dbcfc3d68e4265ceba6e6bd3b93dfe17a9530b5ea59ac",
        ("ollama", "warm"): "e034c005dbf86f671bd5fe157af60bbfb511488c0b5cc82b5773921233f8e3e8",
    }

    for sample_index, (sample, expected) in enumerate(zip(samples, expected_order)):
        if not isinstance(sample, dict):
            raise ValueError("embedded/Ollama sample must be an object")
        replica, order_position, engine = expected
        if (
            sample.get("replica") != replica
            or sample.get("order_position") != order_position
            or sample.get("engine") != engine
        ):
            raise ValueError("embedded/Ollama AB/BA order changed")
        before = require_object(sample, "environment_before")
        after = require_object(sample, "environment_after")
        validate_environment_snapshot(before, f"samples[{sample_index}].environment_before")
        validate_environment_snapshot(after, f"samples[{sample_index}].environment_after")
        environment_verified &= (
            not before.get("observation_errors")
            and parse_finite_float(before.get("cpu_idle_percent"), "cpu_idle_percent")
            >= environment["minimum_pre_sample_cpu_idle_percent"]
            and parse_finite_float(before.get("memory_free_percent"), "memory_free_percent")
            >= environment["minimum_pre_sample_memory_free_percent"]
            and parse_finite_float(
                before.get("gpu_device_utilization_percent"), "gpu_device_utilization_percent"
            )
            <= environment["maximum_pre_sample_gpu_utilization_percent"]
            and parse_finite_float(
                before.get("gpu_in_use_system_memory_bytes"), "gpu_in_use_system_memory_bytes"
            )
            <= environment["maximum_post_process_gpu_memory_bytes"]
        )
        residency_verified &= (
            not after.get("observation_errors")
            and parse_finite_float(
                after.get("gpu_in_use_system_memory_bytes"), "gpu_in_use_system_memory_bytes"
            )
            <= environment["maximum_post_process_gpu_memory_bytes"]
        )

        result = require_object(sample, "result")
        cold = require_object(result, "cold")
        warm = require_object(result, "warm")
        cold_hash = validate_comparison_measurement(cold, engine, "cold", expected_hashes)
        warm_hash = validate_comparison_measurement(warm, engine, "warm", expected_hashes)
        hashes_match = cold_hash == warm_hash
        if result.get("cold_warm_content_match") is not hashes_match:
            raise ValueError("embedded/Ollama cold-warm stability flag is inconsistent")
        model_load = parse_finite_float(
            result.get("model_load_milliseconds"), "model_load_milliseconds"
        )
        if model_load <= 0:
            raise ValueError("embedded/Ollama model load timing must be positive")
        if engine == "embedded":
            shutdown = parse_finite_float(
                result.get("shutdown_milliseconds"), "shutdown_milliseconds"
            )
            if shutdown <= 0 or hashes_match is not True:
                raise ValueError("embedded sample lifecycle or stability changed")
        elif result.get("shutdown_milliseconds") is not None or hashes_match is not False:
            raise ValueError("Ollama sample lifecycle or stability changed")

        warm_rate = parse_finite_float(
            warm.get("approximate_post_first_delta_tokens_per_second"), "warm decode rate"
        )
        warm_end_to_end = parse_finite_float(
            warm.get("end_to_end_output_tokens_per_second"), "warm end-to-end rate"
        )
        warm_first_delta = parse_finite_float(
            warm.get("first_delta_milliseconds"), "warm first delta"
        )
        rates[engine].append(warm_rate)
        end_to_end_rates[engine].append(warm_end_to_end)
        first_deltas[engine].append(warm_first_delta)
        by_replica.setdefault(replica, {})[engine] = (warm_rate, warm_end_to_end)

    if not environment_verified or not residency_verified:
        raise ValueError("embedded/Ollama environment gate did not pass")

    median = lambda values: sorted(values)[len(values) // 2]
    spread = lambda values: max(values) / min(values)
    paired = [
        by_replica[replica]["embedded"][0] / by_replica[replica]["ollama"][0]
        for replica in range(1, 4)
    ]
    paired_end_to_end = [
        by_replica[replica]["embedded"][1] / by_replica[replica]["ollama"][1]
        for replica in range(1, 4)
    ]
    summary = require_object(record, "summary")
    expected_scalars = {
        "embedded_warm_median_tokens_per_second": median(rates["embedded"]),
        "ollama_warm_median_tokens_per_second": median(rates["ollama"]),
        "embedded_warm_median_end_to_end_tokens_per_second": median(
            end_to_end_rates["embedded"]
        ),
        "ollama_warm_median_end_to_end_tokens_per_second": median(end_to_end_rates["ollama"]),
        "embedded_warm_median_first_delta_milliseconds": median(first_deltas["embedded"]),
        "ollama_warm_median_first_delta_milliseconds": median(first_deltas["ollama"]),
        "embedded_warm_spread_ratio": spread(rates["embedded"]),
        "ollama_warm_spread_ratio": spread(rates["ollama"]),
        "median_paired_embedded_over_ollama_speedup": median(paired),
        "median_paired_embedded_over_ollama_end_to_end_speedup": median(paired_end_to_end),
    }
    for field, expected in expected_scalars.items():
        declared = parse_finite_float(summary.get(field), f"summary.{field}")
        if not math.isclose(declared, expected, rel_tol=1e-9):
            raise ValueError(f"embedded/Ollama summary.{field} is inconsistent")
    for field, expected in (
        ("paired_embedded_over_ollama_speedups", paired),
        ("paired_embedded_over_ollama_end_to_end_speedups", paired_end_to_end),
    ):
        declared = summary.get(field)
        if not isinstance(declared, list) or len(declared) != 3 or any(
            not math.isclose(parse_finite_float(value, field), target, rel_tol=1e-9)
            for value, target in zip(declared, expected)
        ):
            raise ValueError(f"embedded/Ollama summary.{field} is inconsistent")
    if (
        summary.get("evidence_grade") != "replicated-stable"
        or summary.get("environment_preconditions_verified") is not True
        or summary.get("residency_isolation_verified") is not True
        or summary.get("embedded_cold_warm_content_stable") is not True
        or summary.get("ollama_cold_warm_content_stable") is not False
        or summary.get("cross_engine_cold_content_matches") != [False, False, False]
        or spread(rates["embedded"]) > 1.05
        or spread(rates["ollama"]) > 1.05
    ):
        raise ValueError("embedded/Ollama evidence grade or stability changed")
    if median(paired) < 1.05 or median(paired_end_to_end) < 1.05:
        raise ValueError("embedded/Ollama paired speedup regressed")
    if median(first_deltas["embedded"]) >= median(first_deltas["ollama"]):
        raise ValueError("embedded/Ollama first-delta advantage regressed")


def validate_environment_snapshot(snapshot: dict[str, Any], field: str) -> None:
    load = snapshot.get("load_average")
    if not isinstance(load, list) or len(load) != 3:
        raise ValueError(f"{field}.load_average must contain three values")
    for index, value in enumerate(load):
        if parse_finite_float(value, f"{field}.load_average[{index}]") < 0:
            raise ValueError(f"{field}.load_average must be nonnegative")
    for name in (
        "cpu_idle_percent",
        "memory_free_percent",
        "gpu_device_utilization_percent",
        "gpu_renderer_utilization_percent",
        "gpu_tiler_utilization_percent",
        "gpu_in_use_system_memory_bytes",
    ):
        value = parse_finite_float(snapshot.get(name), f"{field}.{name}")
        if value < 0:
            raise ValueError(f"{field}.{name} must be nonnegative")
    errors = snapshot.get("observation_errors")
    if not isinstance(errors, list):
        raise ValueError(f"{field}.observation_errors must be a list")
    processes = snapshot.get("top_processes")
    if not isinstance(processes, list) or not processes:
        raise ValueError(f"{field}.top_processes must be nonempty")
    for process in processes:
        if (
            not isinstance(process, dict)
            or not isinstance(process.get("command"), str)
            or not process["command"]
            or not isinstance(process.get("pid"), int)
            or parse_finite_float(process.get("cpu_percent"), f"{field}.cpu_percent") < 0
            or parse_finite_float(process.get("resident_mib"), f"{field}.resident_mib") < 0
        ):
            raise ValueError(f"{field}.top_processes contains invalid telemetry")


def validate_comparison_measurement(
    measurement: dict[str, Any],
    engine: str,
    phase: str,
    expected_hashes: dict[tuple[str, str], str],
) -> str:
    total = parse_finite_float(measurement.get("total_milliseconds"), "total_milliseconds")
    first_delta = parse_finite_float(
        measurement.get("first_delta_milliseconds"), "first_delta_milliseconds"
    )
    output_tokens = measurement.get("output_tokens")
    end_to_end = parse_finite_float(
        measurement.get("end_to_end_output_tokens_per_second"), "end_to_end rate"
    )
    post_first = parse_finite_float(
        measurement.get("approximate_post_first_delta_tokens_per_second"),
        "post-first rate",
    )
    content_hash = str(measurement.get("content_sha256", ""))
    if (
        output_tokens != 128
        or not 0 < first_delta < total
        or not math.isclose(end_to_end, output_tokens / (total / 1000), rel_tol=1e-9)
        or not math.isclose(
            post_first,
            (output_tokens - 1) / ((total - first_delta) / 1000),
            rel_tol=1e-9,
        )
        or measurement.get("reasoning_bytes", 0) <= 0
        or measurement.get("text_bytes") != 0
        or content_hash != expected_hashes[(engine, phase)]
    ):
        raise ValueError("embedded/Ollama measurement is inconsistent")

    server = measurement.get("ollama_server")
    if engine == "embedded":
        expected_input = 1 if phase == "warm" else 107
        expected_cache = 106 if phase == "warm" else None
        if (
            measurement.get("input_tokens") != expected_input
            or measurement.get("cache_read_input_tokens") != expected_cache
            or measurement.get("stop_reason") != "MaxTokens"
            or server is not None
        ):
            raise ValueError("embedded prompt-cache measurement changed")
    else:
        if (
            measurement.get("input_tokens") != 107
            or measurement.get("cache_read_input_tokens") is not None
            or measurement.get("stop_reason") != "length"
            or not isinstance(server, dict)
        ):
            raise ValueError("Ollama measurement contract changed")
        eval_count = server.get("eval_count")
        eval_milliseconds = parse_finite_float(
            server.get("eval_milliseconds"), "ollama_server.eval_milliseconds"
        )
        server_rate = parse_finite_float(
            server.get("decode_tokens_per_second"), "ollama_server.decode_tokens_per_second"
        )
        server_total = parse_finite_float(
            server.get("total_milliseconds"), "ollama_server.total_milliseconds"
        )
        if (
            server.get("prompt_eval_count") != 107
            or eval_count != 128
            or eval_milliseconds <= 0
            or parse_finite_float(
                server.get("prompt_eval_milliseconds"), "ollama_server.prompt_eval_milliseconds"
            )
            <= 0
            or parse_finite_float(
                server.get("load_milliseconds"), "ollama_server.load_milliseconds"
            )
            <= 0
            or not math.isclose(server_rate, eval_count / (eval_milliseconds / 1000), rel_tol=1e-9)
            or abs(server_total - total) > 5
        ):
            raise ValueError("Ollama server timing is inconsistent")
    return content_hash


def require_object(record: dict[str, Any], field: str) -> dict[str, Any]:
    value = record.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def parse_finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    args = parser.parse_args(argv)
    with args.checkpoint.open(encoding="utf-8") as source:
        validate_checkpoint(json.load(source))
    print(f"validated {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
