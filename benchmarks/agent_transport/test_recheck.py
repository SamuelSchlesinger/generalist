import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("recheck.py")
SPEC = importlib.util.spec_from_file_location("agent_transport_recheck", MODULE_PATH)
recheck = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(recheck)


class RecheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = recheck.benchmark.load_corpus(recheck.benchmark.DEFAULT_CORPUS)
        cls.tasks = {task["id"]: task for task in corpus["tasks"]}

    def test_reclassifies_a_previously_accepted_missing_import(self):
        code = "tools.record_text(text='agent transport works')"
        record = {
            "schema_version": 1,
            "record_type": "attempt",
            "run_id": "old-run",
            "task_id": "simple_ascii",
            "transport": "plain_text",
            "http_status": 200,
            "classification": "pass",
            "response": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": code},
                    }
                ]
            },
        }
        derived = recheck.recheck_attempt(
            record, self.tasks, pathlib.Path("old.jsonl"), "new-run"
        )
        self.assertEqual(derived["run_id"], "new-run")
        self.assertEqual(derived["classification"], "missing_tools_import")
        self.assertEqual(derived["derived_from"]["run_id"], "old-run")
        self.assertEqual(derived["derived_from"]["classification"], "pass")

    def test_reclassifies_custom_tool_downgrade(self):
        record = {
            "schema_version": 1,
            "record_type": "attempt",
            "run_id": "old-run",
            "task_id": "simple_ascii",
            "transport": "responses_custom",
            "http_status": 200,
            "classification": "extraction_error",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "name": "python",
                        "arguments": '{"text":"agent transport works"}',
                    }
                ],
            },
        }
        derived = recheck.recheck_attempt(
            record, self.tasks, pathlib.Path("old.jsonl"), "new-run"
        )
        self.assertEqual(derived["classification"], "custom_tool_degraded")


if __name__ == "__main__":
    unittest.main()
