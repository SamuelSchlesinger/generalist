import importlib.util
import json
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("local_reasoning_summarize", MODULE_PATH)
summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_table_reports_quality_next_to_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "run.jsonl"
            records = [
                {
                    "record_type": "run_start",
                    "run_id": "one",
                    "variant": "bf16-mtp",
                    "model": "qwen",
                },
                {
                    "record_type": "run_end",
                    "run_id": "one",
                    "status": "complete",
                    "aggregate": {
                        "attempts": 3,
                        "passed": 3,
                        "reasoning_present": 3,
                        "completed": 3,
                        "medians": {
                            "wall_ms": 1000.0,
                            "prompt_tokens_per_second": 120.0,
                            "decode_tokens_per_second": 50.0,
                            "mtp_acceptance": 0.88,
                            "mtp_avg_draft": 3.0,
                            "eval_count": 200.0,
                            "runtime_inference_passes": 2.0,
                            "runtime_estimated_emitted_tokens": 450.0,
                        },
                    },
                },
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            table = summary.markdown_table(summary.load_runs([path]))
        self.assertIn("bf16-mtp", table)
        self.assertIn("3/3", table)
        self.assertIn("50.0", table)
        self.assertIn("0.88", table)
        self.assertIn("450.0", table)


if __name__ == "__main__":
    unittest.main()
