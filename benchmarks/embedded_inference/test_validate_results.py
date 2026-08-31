import copy
import importlib.util
import json
import pathlib
import unittest


DIRECTORY = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "embedded_checkpoint_validator", DIRECTORY / "validate_results.py"
)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


class CheckpointValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (DIRECTORY / "results" / "rms_norm_layer0.json").open(
            encoding="utf-8"
        ) as source:
            cls.rms_record = json.load(source)
        with (DIRECTORY / "results" / "embedding_layer0.json").open(
            encoding="utf-8"
        ) as source:
            cls.embedding_record = json.load(source)
        with (DIRECTORY / "results" / "nvfp4_projection_layer0.json").open(
            encoding="utf-8"
        ) as source:
            cls.nvfp4_record = json.load(source)
        with (DIRECTORY / "results" / "layer0_projections.json").open(
            encoding="utf-8"
        ) as source:
            cls.layer0_projection_record = json.load(source)
        with (DIRECTORY / "results" / "layer0_gdn.json").open(
            encoding="utf-8"
        ) as source:
            cls.layer0_gdn_record = json.load(source)
        with (DIRECTORY / "results" / "layer0_full.json").open(
            encoding="utf-8"
        ) as source:
            cls.layer0_full_record = json.load(source)
        with (DIRECTORY / "results" / "full_decode.json").open(
            encoding="utf-8"
        ) as source:
            cls.full_decode_record = json.load(source)
        with (DIRECTORY / "results" / "cached_decode.json").open(
            encoding="utf-8"
        ) as source:
            cls.cached_decode_record = json.load(source)
        with (DIRECTORY / "results" / "metal_dispatch_profile.json").open(
            encoding="utf-8"
        ) as source:
            cls.metal_dispatch_record = json.load(source)
        with (DIRECTORY / "results" / "qmv_geometry_sweep.json").open(
            encoding="utf-8"
        ) as source:
            cls.qmv_geometry_record = json.load(source)
        with (DIRECTORY / "results" / "mtp_depth_control.json").open(
            encoding="utf-8"
        ) as source:
            cls.mtp_depth_record = json.load(source)
        with (DIRECTORY / "results" / "native_speculative_decode.json").open(
            encoding="utf-8"
        ) as source:
            cls.native_speculative_record = json.load(source)
        with (DIRECTORY / "results" / "adaptive_production.json").open(
            encoding="utf-8"
        ) as source:
            cls.adaptive_production_record = json.load(source)
        with (DIRECTORY / "results" / "embedded_vs_ollama.json").open(
            encoding="utf-8"
        ) as source:
            cls.embedded_ollama_record = json.load(source)

    def test_frozen_rms_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.rms_record)

    def test_frozen_embedding_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.embedding_record)

    def test_frozen_nvfp4_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.nvfp4_record)

    def test_frozen_layer0_projection_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.layer0_projection_record)

    def test_frozen_layer0_gdn_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.layer0_gdn_record)

    def test_frozen_complete_layer0_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.layer0_full_record)

    def test_frozen_full_decode_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.full_decode_record)

    def test_frozen_cached_decode_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.cached_decode_record)

    def test_frozen_metal_dispatch_checkpoint_is_valid(self):
        validator.validate_checkpoint(self.metal_dispatch_record)

    def test_frozen_qmv_geometry_sweep_is_valid(self):
        validator.validate_checkpoint(self.qmv_geometry_record)

    def test_frozen_mtp_depth_control_is_valid(self):
        validator.validate_checkpoint(self.mtp_depth_record)

    def test_frozen_native_speculative_decode_is_valid(self):
        validator.validate_checkpoint(self.native_speculative_record)

    def test_frozen_adaptive_production_is_valid(self):
        validator.validate_checkpoint(self.adaptive_production_record)

    def test_frozen_embedded_ollama_comparison_is_valid(self):
        validator.validate_checkpoint(self.embedded_ollama_record)

    def test_output_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.rms_record)
        changed["ollama"]["output_fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch in output_fingerprint"):
            validator.validate_checkpoint(changed)

    def test_reference_error_bound_is_enforced(self):
        changed = copy.deepcopy(self.rms_record)
        changed["generalist"]["max_abs_float64_reference_error"] = "0.009"
        with self.assertRaisesRegex(ValueError, "reference error"):
            validator.validate_checkpoint(changed)

    def test_embedding_stage_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.embedding_record)
        changed["ollama"]["normalized"]["fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch in normalized.fingerprint"):
            validator.validate_checkpoint(changed)

    def test_nvfp4_raw_stage_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.nvfp4_record)
        changed["ollama"]["raw_projection"]["fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch in raw_projection.fingerprint"):
            validator.validate_checkpoint(changed)

    def test_layer0_packed_projection_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.layer0_projection_record)
        changed["ollama"]["mixed_ba"]["fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch in mixed_ba.fingerprint"):
            validator.validate_checkpoint(changed)

    def test_layer0_delta_state_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.layer0_gdn_record)
        changed["ollama"]["delta_state"]["fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch in delta_state.fingerprint"):
            validator.validate_checkpoint(changed)

    def test_complete_layer0_output_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.layer0_full_record)
        changed["ollama"]["layer_out"]["fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch in layer_out.fingerprint"):
            validator.validate_checkpoint(changed)

    def test_full_decode_logits_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.full_decode_record)
        changed["ollama"]["logits"]["fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "mismatch in logits.fingerprint"):
            validator.validate_checkpoint(changed)

    def test_full_decode_token_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.full_decode_record)
        changed["ollama"]["output_token_id"] = 0
        with self.assertRaisesRegex(ValueError, "mismatch in output_token_id"):
            validator.validate_checkpoint(changed)

    def test_cached_decode_step1_logits_mismatch_is_rejected(self):
        changed = copy.deepcopy(self.cached_decode_record)
        changed["ollama"]["steps"][1]["logits"]["fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, r"mismatch in steps\[1\].logits.fingerprint"):
            validator.validate_checkpoint(changed)

    def test_cached_decode_contaminated_timing_is_rejected(self):
        changed = copy.deepcopy(self.cached_decode_record)
        changed["performance_observation"][
            "observed_exact_zero_state_median_milliseconds"
        ] = "226.980"
        with self.assertRaisesRegex(ValueError, "timing control is contaminated"):
            validator.validate_checkpoint(changed)

    def test_metal_dispatch_regression_is_rejected(self):
        changed = copy.deepcopy(self.metal_dispatch_record)
        changed["precomputed_norm_candidate"]["steady_step_dispatches"] = 2471
        with self.assertRaisesRegex(ValueError, "per-step counts changed"):
            validator.validate_checkpoint(changed)

    def test_exact_scale_dispatch_regression_is_rejected(self):
        changed = copy.deepcopy(self.metal_dispatch_record)
        changed["exact_scale_candidate"]["steady_step_dispatches"] = 1412
        with self.assertRaisesRegex(ValueError, "per-step counts changed"):
            validator.validate_checkpoint(changed)

    def test_exact_scale_long_oracle_regression_is_rejected(self):
        changed = copy.deepcopy(self.metal_dispatch_record)
        changed["oracle"]["position_128"]["logits_fingerprint"] = "0000000000000000"
        with self.assertRaisesRegex(ValueError, "position_128 oracle changed"):
            validator.validate_checkpoint(changed)

    def test_exact_scale_paired_gain_regression_is_rejected(self):
        changed = copy.deepcopy(self.metal_dispatch_record)
        changed["wall_clock_experiments"]["exact_global_scale_replicas"][0][
            "paired_median_speedup"
        ] = "1.0"
        with self.assertRaisesRegex(ValueError, "did not reproduce the gain"):
            validator.validate_checkpoint(changed)

    def test_narrow_qmv_inventory_regression_is_rejected(self):
        changed = copy.deepcopy(self.metal_dispatch_record)
        changed["narrow_qmv_candidate"]["narrow_matvecs_per_cached_token"] = 256
        with self.assertRaisesRegex(ValueError, "narrow QMV inventory changed"):
            validator.validate_checkpoint(changed)

    def test_narrow_qmv_paired_gain_regression_is_rejected(self):
        changed = copy.deepcopy(self.metal_dispatch_record)
        changed["wall_clock_experiments"]["narrow_qmv_replicas"][0][
            "paired_median_speedup"
        ] = "1.0"
        with self.assertRaisesRegex(ValueError, "did not reproduce the gain"):
            validator.validate_checkpoint(changed)

    def test_qmv_geometry_promotion_is_rejected(self):
        changed = copy.deepcopy(self.qmv_geometry_record)
        changed["conclusion"]["promoted"] = True
        with self.assertRaisesRegex(ValueError, "rejection conclusion changed"):
            validator.validate_checkpoint(changed)

    def test_mtp_semantic_drift_is_rejected(self):
        changed = copy.deepcopy(self.mtp_depth_record)
        changed["modes"][2]["runs"][1][-1] = "0" * 64
        with self.assertRaisesRegex(ValueError, "semantic output mismatch"):
            validator.validate_checkpoint(changed)

    def test_mtp_depth_optimum_regression_is_rejected(self):
        changed = copy.deepcopy(self.mtp_depth_record)
        changed["modes"][3]["runs"][1][1] = 6_000_000_000
        changed["modes"][3]["runs"][1][2] = "32.0"
        changed["modes"][3]["median_tokens_per_second"] = "27.223896180054364"
        changed["modes"][3]["median_speedup_over_fixed0"] = "1.834105"
        with self.assertRaisesRegex(ValueError, "depth optimum changed"):
            validator.validate_checkpoint(changed)

    def test_native_speculative_reasoning_disable_is_rejected(self):
        changed = copy.deepcopy(self.native_speculative_record)
        changed["protocol"]["reasoning"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "reasoning protocol changed"):
            validator.validate_checkpoint(changed)

    def test_native_speculative_prefill_regression_is_rejected(self):
        changed = copy.deepcopy(self.native_speculative_record)
        changed["prefill_experiment"]["candidate"][
            "target_prompt_tokens_per_second"
        ] = changed["prefill_experiment"]["baseline"][
            "target_prompt_tokens_per_second"
        ]
        changed["prefill_experiment"]["candidate_speedup"] = "1.0"
        with self.assertRaisesRegex(ValueError, "prefill gain regressed"):
            validator.validate_checkpoint(changed)

    def test_native_speculative_geometry_loss_is_rejected(self):
        changed = copy.deepcopy(self.native_speculative_record)
        changed["geometry_experiment"]["runs"][0][1] = changed[
            "geometry_experiment"
        ]["runs"][0][0]
        with self.assertRaisesRegex(ValueError, "did not win every paired replica"):
            validator.validate_checkpoint(changed)

    def test_native_speculative_output_drift_is_rejected(self):
        changed = copy.deepcopy(self.native_speculative_record)
        changed["geometry_experiment"]["output_token_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "semantic evidence changed"):
            validator.validate_checkpoint(changed)

    def test_adaptive_production_output_drift_is_rejected(self):
        changed = copy.deepcopy(self.adaptive_production_record)
        changed["prompt_adaptive_head"]["modes"][1]["runs"][0][
            "output_token_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "semantic output mismatch"):
            validator.validate_checkpoint(changed)

    def test_adaptive_production_gain_regression_is_rejected(self):
        changed = copy.deepcopy(self.adaptive_production_record)
        changed["prompt_adaptive_head"]["modes"][1][
            "median_paired_speedup_over_full"
        ] = "1.0"
        with self.assertRaisesRegex(ValueError, "paired speedup"):
            validator.validate_checkpoint(changed)

    def test_adaptive_production_cancellation_regression_is_rejected(self):
        changed = copy.deepcopy(self.adaptive_production_record)
        changed["provider_path"]["cancellation"][
            "worker_shutdown_milliseconds"
        ] = "700"
        with self.assertRaisesRegex(ValueError, "cancellation bound"):
            validator.validate_checkpoint(changed)

    def test_embedded_ollama_missing_cache_reuse_is_rejected(self):
        changed = copy.deepcopy(self.embedded_ollama_record)
        changed["samples"][0]["result"]["warm"]["cache_read_input_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "prompt-cache measurement"):
            validator.validate_checkpoint(changed)

    def test_embedded_ollama_cpu_contamination_is_rejected(self):
        changed = copy.deepcopy(self.embedded_ollama_record)
        changed["samples"][0]["environment_before"]["cpu_idle_percent"] = 79.9
        with self.assertRaisesRegex(ValueError, "environment gate"):
            validator.validate_checkpoint(changed)

    def test_embedded_ollama_speedup_regression_is_rejected(self):
        changed = copy.deepcopy(self.embedded_ollama_record)
        changed["summary"]["median_paired_embedded_over_ollama_speedup"] = 1.0
        with self.assertRaisesRegex(ValueError, "summary.*inconsistent"):
            validator.validate_checkpoint(changed)

    def test_embedded_ollama_stability_regression_is_rejected(self):
        changed = copy.deepcopy(self.embedded_ollama_record)
        changed["summary"]["embedded_cold_warm_content_stable"] = False
        with self.assertRaisesRegex(ValueError, "evidence grade or stability"):
            validator.validate_checkpoint(changed)


if __name__ == "__main__":
    unittest.main()
