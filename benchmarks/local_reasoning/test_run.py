import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("local_reasoning_run", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = benchmark.load_corpus(benchmark.DEFAULT_CORPUS)

    def test_exact_answers_are_nonempty_and_ids_are_unique(self):
        ids = [task["id"] for task in self.corpus["tasks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(task["expected"] for task in self.corpus["tasks"]))

    def test_selection_rejects_unknown_task(self):
        with self.assertRaisesRegex(ValueError, "unknown task ids"):
            benchmark.selected_tasks(self.corpus, ["missing"], [], None)


class RequestContractTests(unittest.TestCase):
    def setUp(self):
        self.task = {"id": "fixture", "prompt": "Solve it.", "expected": "7"}

    def test_reasoning_is_unconditionally_enabled(self):
        body = benchmark.build_request(
            self.task,
            "model",
            num_ctx=32768,
            num_predict=2048,
            temperature=None,
            seed=None,
            keep_alive="10m",
            padding_lines=0,
            mtp_mode="auto",
        )
        self.assertIs(body["think"], True)
        self.assertNotIn("temperature", body["options"])
        self.assertNotIn("seed", body["options"])

    def test_controlled_sampling_is_recorded_without_changing_reasoning(self):
        body = benchmark.build_request(
            self.task,
            "model",
            num_ctx=32768,
            num_predict=2048,
            temperature=0.6,
            seed=1,
            keep_alive="10m",
            padding_lines=2,
            mtp_mode="auto",
        )
        self.assertIs(body["think"], True)
        self.assertEqual(body["options"]["temperature"], 0.6)
        self.assertEqual(body["options"]["seed"], 1)
        self.assertIn("padding record 00001", body["messages"][1]["content"])

    def test_mtp_off_control_preserves_reasoning_and_requests_native_logprobs(self):
        body = benchmark.build_request(
            self.task,
            "model",
            num_ctx=32768,
            num_predict=2048,
            temperature=0.0,
            seed=1,
            keep_alive="10m",
            padding_lines=0,
            mtp_mode="off-logprobs",
        )
        self.assertIs(body["think"], True)
        self.assertIs(body["logprobs"], True)
        self.assertEqual(body["top_logprobs"], 0)

    def test_json_schema_constrains_only_the_final_answer(self):
        body = benchmark.build_request(
            self.task,
            "model",
            num_ctx=32768,
            num_predict=2048,
            temperature=0.0,
            seed=1,
            keep_alive="10m",
            padding_lines=0,
            mtp_mode="auto",
            answer_format="json-schema",
        )
        self.assertIs(body["think"], True)
        self.assertEqual(body["format"], benchmark.FINAL_JSON_SCHEMA)
        self.assertIn("dedicated thinking channel", body["messages"][0]["content"])


class StreamAndQualityTests(unittest.TestCase):
    def setUp(self):
        self.task = {"id": "fixture", "prompt": "Solve it.", "expected": "135"}

    def test_stream_keeps_reasoning_and_answer_timings_separate(self):
        response = benchmark.assemble_stream(
            [
                (12.0, {"message": {"thinking": "work ", "content": ""}}),
                (20.0, {"message": {"thinking": "done", "content": ""}}),
                (24.0, {"message": {"thinking": "", "content": "FINAL: 135"}}),
                (
                    25.0,
                    {
                        "message": {"thinking": "", "content": ""},
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 50,
                        "prompt_eval_duration": 500_000_000,
                        "eval_count": 100,
                        "eval_duration": 2_000_000_000,
                    },
                ),
            ]
        )
        self.assertEqual(response["thinking"], "work done")
        self.assertEqual(response["answer"], "FINAL: 135")
        self.assertEqual(response["first_thinking_ms"], 12.0)
        self.assertEqual(response["first_answer_ms"], 24.0)
        self.assertTrue(benchmark.check_quality(self.task, response)["passed"])

    def test_missing_reasoning_cannot_pass_on_a_correct_answer(self):
        response = {
            "thinking": "",
            "answer": "FINAL: 135",
            "server": {"done": True, "done_reason": "stop"},
        }
        quality = benchmark.check_quality(self.task, response)
        self.assertFalse(quality["passed"])
        self.assertEqual(quality["failure_kind"], "missing_reasoning")

    def test_output_limit_precedes_missing_final(self):
        response = {
            "thinking": "still working",
            "answer": "",
            "server": {"done": True, "done_reason": "length"},
        }
        quality = benchmark.check_quality(self.task, response)
        self.assertEqual(quality["failure_kind"], "output_limit")

    def test_wrong_final_answer_is_classified(self):
        response = {
            "thinking": "work",
            "answer": "Explanation.\nFINAL: 136",
            "server": {"done": True, "done_reason": "stop"},
        }
        quality = benchmark.check_quality(self.task, response)
        self.assertEqual(quality["failure_kind"], "wrong_answer")

    def test_explanation_is_a_compactness_signal_not_a_reasoning_failure(self):
        response = {
            "thinking": "complete reasoning",
            "answer": "Here is the result.\nFINAL: 135",
            "server": {"done": True, "done_reason": "stop"},
        }
        quality = benchmark.check_quality(self.task, response)
        self.assertTrue(quality["passed"])
        self.assertFalse(quality["answer_contract_met"])

    def test_schema_constrained_answer_passes_with_reasoning(self):
        response = {
            "thinking": "complete reasoning",
            "answer": '{"final":"135"}',
            "server": {"done": True, "done_reason": "stop"},
        }
        quality = benchmark.check_quality(self.task, response, "json-schema")
        self.assertTrue(quality["passed"])
        self.assertTrue(quality["answer_contract_met"])

    def test_schema_constrained_answer_still_requires_reasoning(self):
        response = {
            "thinking": "",
            "answer": '{"final":"135"}',
            "server": {"done": True, "done_reason": "stop"},
        }
        quality = benchmark.check_quality(self.task, response, "json-schema")
        self.assertFalse(quality["passed"])
        self.assertEqual(quality["failure_kind"], "missing_reasoning")

    def test_schema_suffix_is_correct_but_not_schema_adherent(self):
        response = {
            "thinking": "complete reasoning",
            "answer": 'Repeated explanation.\n{"final":"135"}',
            "server": {"done": True, "done_reason": "stop"},
        }
        quality = benchmark.check_quality(self.task, response, "json-schema")
        self.assertTrue(quality["passed"])
        self.assertFalse(quality["answer_contract_met"])


class InspectionAndEvidenceTests(unittest.TestCase):
    def test_unload_and_cool_records_and_applies_the_idle_interval(self):
        with mock.patch.object(
            benchmark,
            "unload_model",
            return_value={"status": 200, "elapsed_ms": 12.0, "response": {}},
        ) as unload, mock.patch.object(benchmark.time, "sleep") as sleep:
            result = benchmark.unload_and_cool("http://local", "model", 10, 2.5)

        unload.assert_called_once_with("http://local", "model", 10)
        sleep.assert_called_once_with(2.5)
        self.assertEqual(result["cooldown_seconds"], 2.5)

    def test_cooldown_requires_cold_reload_mode(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(["--cooldown-seconds", "1"])

    def test_speculation_log_parser_extracts_direct_acceptance_measurements(self):
        line = (
            'time=now level=INFO source=speculate_stats.go:62 '
            'msg="speculative decode stats" iterations=327 drafted=990 accepted=875 '
            'acceptance=0.88 avg_draft=3.03 max_draft=4 avg_accepted=2.68 '
            'depth_over_time="3.0/3 3.1/4"'
        )
        records = benchmark.parse_speculation_stats(line)
        self.assertEqual(records[0]["accepted"], 875)
        self.assertEqual(records[0]["acceptance"], 0.88)
        self.assertEqual(records[0]["avg_draft"], 3.03)
        self.assertEqual(records[0]["depth_over_time"], "3.0/3 3.1/4")

    def test_speculation_aggregate_combines_multi_pass_reasoning(self):
        records = [
            {
                "iterations": 103,
                "drafted": 397,
                "accepted": 366,
                "max_draft": 6,
            },
            {
                "iterations": 80,
                "drafted": 244,
                "accepted": 215,
                "max_draft": 4,
            },
        ]
        aggregate = benchmark.aggregate_speculation_stats(records)
        self.assertEqual(aggregate["passes"], 2)
        self.assertEqual(aggregate["estimated_emitted_tokens"], 764)
        self.assertEqual(aggregate["max_draft"], 6)
        self.assertAlmostEqual(aggregate["acceptance"], 581 / 641, places=4)

    def test_capture_reads_only_records_appended_after_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "server.log"
            path.write_text("old line\n")
            cursor = benchmark.log_cursor(path)
            with path.open("a") as destination:
                destination.write(
                    'msg="speculative decode stats" iterations=2 drafted=3 accepted=2 '
                    'acceptance=0.67 avg_draft=1.50 max_draft=2 avg_accepted=1.00 '
                    'depth_over_time="1.0/1"\n'
                )
            capture = benchmark.capture_speculation_stats(path, cursor, wait_seconds=0)
        self.assertEqual(capture["record_count"], 1)
        self.assertEqual(capture["selected"]["drafted"], 3)
        self.assertEqual(capture["aggregate"]["estimated_emitted_tokens"], 4)

    def test_model_snapshot_exposes_mtp_precision_and_layer_mix(self):
        show = {
            "details": {"parameter_size": "27.8B", "quantization_level": "NVFP4"},
            "capabilities": ["completion", "thinking"],
            "requires": "0.32.12",
            "model_info": {
                "general.architecture": "qwen3_5",
                "general.parameter_count": 27_781_081_984,
                "qwen3_5.context_length": 262144,
                "irrelevant": "omitted",
            },
            "tensors": [
                {
                    "name": "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                    "type": "NVFP4",
                    "shape": [3, 4],
                },
                {
                    "name": "model.language_model.layers.3.self_attn.q_proj.weight",
                    "type": "NVFP4",
                    "shape": [3, 4],
                },
                {"name": "model.visual.blocks.0.attn.qkv.weight", "type": "BF16"},
                {"name": "mtp.fc.weight", "type": "NVFP4", "shape": [5120, 10240]},
            ],
        }
        tags = {
            "models": [
                {"name": "qwen3.8:27b-mlx", "digest": "abc", "size": 18_000_000_000}
            ]
        }
        snapshot = benchmark.compact_model_snapshot(show, tags, "qwen3.8:27b-mlx")
        self.assertEqual(snapshot["tensors"]["mtp_fc"]["type"], "NVFP4")
        self.assertEqual(snapshot["tensors"]["linear_attention_layers"], [0])
        self.assertEqual(snapshot["tensors"]["full_attention_layers"], [3])
        self.assertNotIn("irrelevant", snapshot["model_info"])

    def test_aggregate_keeps_quality_and_speed_separate(self):
        attempts = [
            {
                "classification": "pass",
                "quality": {
                    "reasoning_present": True,
                    "completed": True,
                    "correct": True,
                },
                "metrics": {"wall_ms": 100.0, "decode_tokens_per_second": 40.0},
            },
            {
                "classification": "wrong_answer",
                "quality": {
                    "reasoning_present": True,
                    "completed": True,
                    "correct": False,
                },
                "metrics": {"wall_ms": 50.0, "decode_tokens_per_second": 80.0},
            },
        ]
        aggregate = benchmark.aggregate_attempts(attempts)
        self.assertEqual(aggregate["passed"], 1)
        self.assertEqual(aggregate["correct"], 1)
        self.assertEqual(aggregate["medians"]["decode_tokens_per_second"], 60.0)

    def test_inspect_only_writes_a_complete_evidence_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "inspection.jsonl"
            with mock.patch.object(benchmark, "inspect_model", return_value={"ok": True}), mock.patch.object(
                benchmark, "active_model_snapshot", return_value=None
            ):
                exit_code = benchmark.main(["--inspect-only", "--output", str(output)])
            self.assertEqual(exit_code, 0)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(records[0]["reasoning_contract"]["think"], True)
            self.assertEqual(records[-1]["status"], "complete")
            self.assertEqual(records[-1]["aggregate"]["attempts"], 0)


if __name__ == "__main__":
    unittest.main()
