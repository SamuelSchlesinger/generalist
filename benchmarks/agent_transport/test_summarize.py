import importlib.util
import json
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("summarize.py")
SPEC = importlib.util.spec_from_file_location("agent_transport_summarize", MODULE_PATH)
summary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_groups_attempts_and_keeps_failure_classes(self):
        attempts = [
            {
                "provider": "ollama",
                "model": "local",
                "transport": "json_tool",
                "classification": "pass",
                "metrics": {
                    "latency_ms": 10,
                    "model_payload_overhead_bytes": 4,
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            },
            {
                "provider": "ollama",
                "model": "local",
                "transport": "json_tool",
                "classification": "syntax_error",
                "metrics": {
                    "latency_ms": 30,
                    "model_payload_overhead_bytes": 8,
                    "usage": {"input_tokens": 120, "output_tokens": 40},
                },
            },
        ]
        rows = summary.summarize(attempts)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["passed"], 1)
        self.assertEqual(rows[0]["pass_rate"], 0.5)
        self.assertEqual(rows[0]["median_latency_ms"], 20)
        self.assertEqual(rows[0]["classifications"]["syntax_error"], 1)

    def test_reader_skips_a_truncated_record_but_preserves_prior_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "partial.jsonl"
            path.write_text(
                json.dumps({"record_type": "attempt", "classification": "pass"})
                + "\n{\"record_type\":",
                encoding="utf-8",
            )
            attempts = summary.read_attempts([path])
            self.assertEqual(len(attempts), 1)


if __name__ == "__main__":
    unittest.main()
