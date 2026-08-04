import importlib.util
import json
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("agent_transport_run", MODULE_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def replace_wildcards(value):
    if value == {"$any": True}:
        return "computed from a prior result"
    if isinstance(value, dict):
        return {key: replace_wildcards(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_wildcards(item) for item in value]
    return value


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = benchmark.load_corpus(benchmark.DEFAULT_CORPUS)

    def test_every_task_references_known_tools(self):
        catalog = self.corpus["tool_catalog"]
        ids = set()
        for task in self.corpus["tasks"]:
            self.assertNotIn(task["id"], ids)
            ids.add(task["id"])
            self.assertTrue(task["available_tools"])
            self.assertTrue(task["expected_calls"])
            for name in task["available_tools"]:
                self.assertIn(name, catalog)
            for call in task["expected_calls"]:
                self.assertIn(call["name"], task["available_tools"])

    def test_literal_rendering_can_satisfy_every_task(self):
        for task in self.corpus["tasks"]:
            lines = ["import tools"]
            for call in task["expected_calls"]:
                arguments = replace_wildcards(call["arguments"])
                keywords = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
                lines.append(f"tools.{call['name']}({keywords})")
            result = benchmark.check_source("\n".join(lines), task["expected_calls"])
            self.assertTrue(result["passed"], f"{task['id']}: {result}")

    def test_task_selection_rejects_unknown_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown task ids"):
            benchmark.selected_tasks(self.corpus, ["missing"], [], None)


class SourceCheckerTests(unittest.TestCase):
    def test_constants_aliases_and_dynamic_values(self):
        source = """\
import tools as bridge
key = "alpha"
value = bridge.fetch(key=key)
bridge.publish(body=value + "\\nnext", source_count=1)
"""
        expected = [
            {"name": "fetch", "arguments": {"key": "alpha"}},
            {
                "name": "publish",
                "arguments": {"body": {"$any": True}, "source_count": 1},
            },
        ]
        result = benchmark.check_source(source, expected)
        self.assertTrue(result["passed"], result)

    def test_argument_mismatch_is_classified(self):
        result = benchmark.check_source(
            "import tools\ntools.record_text(text='wrong')",
            [{"name": "record_text", "arguments": {"text": "right"}}],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_kind"], "call_arguments")

    def test_syntax_error_is_classified(self):
        result = benchmark.check_source("import tools\ntools.fetch(key=", [])
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_kind"], "syntax_error")

    def test_non_tool_code_is_never_executed(self):
        source = "__import__('os').system('touch should-not-exist')"
        result = benchmark.check_source(
            source, [{"name": "record_text", "arguments": {"text": "x"}}]
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["actual_calls"], [])

    def test_tools_reference_without_import_is_rejected(self):
        result = benchmark.check_source(
            "tools.record_text(text='x')",
            [{"name": "record_text", "arguments": {"text": "x"}}],
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_kind"], "missing_tools_import")

    def test_preloaded_bridge_accepts_tools_without_import(self):
        result = benchmark.check_source(
            "tools.record_text(text='x')",
            [{"name": "record_text", "arguments": {"text": "x"}}],
            bridge_preloaded=True,
        )
        self.assertTrue(result["passed"], result)


class WireFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = benchmark.load_corpus(benchmark.DEFAULT_CORPUS)
        cls.task = cls.corpus["tasks"][0]

    def test_json_tool_request_uses_current_object_wrapper(self):
        path, body = benchmark.build_request(
            self.corpus, self.task, "model", "json_tool", 800, None
        )
        self.assertEqual(path, "/chat/completions")
        function = body["tools"][0]["function"]
        self.assertEqual(function["name"], "python")
        self.assertEqual(function["parameters"]["required"], ["code"])
        self.assertEqual(
            body["tool_choice"], {"type": "function", "function": {"name": "python"}}
        )
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["seed"], 1)

    def test_custom_request_uses_freeform_tool(self):
        path, body = benchmark.build_request(
            self.corpus, self.task, "model", "responses_custom", 800, "low"
        )
        self.assertEqual(path, "/responses")
        self.assertEqual(body["tools"][0]["type"], "custom")
        self.assertNotIn("parameters", body["tools"][0])
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertEqual(body["temperature"], 0.0)
        self.assertNotIn("seed", body)

    def test_compact_signature_preserves_required_and_optional_types(self):
        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "query": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }
        self.assertEqual(
            benchmark.python_call_signature("search", schema),
            "tools.search(*, query: str, limit: int | None = None) -> str",
        )

    def test_extracts_json_wrapped_code(self):
        code = "import tools\ntools.record_text(text='x')"
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "python",
                                    "arguments": json.dumps({"code": code}),
                                }
                            }
                        ]
                    }
                }
            ]
        }
        actual, details = benchmark.extract_chat_code(response, "json_tool")
        self.assertEqual(actual, code)
        self.assertEqual(details["python_call_count"], 1)

    def test_rejects_malformed_json_arguments(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "python", "arguments": "{bad"}}
                        ]
                    }
                }
            ]
        }
        code, details = benchmark.extract_chat_code(response, "json_tool")
        self.assertIsNone(code)
        self.assertIn("invalid function arguments JSON", details["error"])

    def test_output_limit_is_not_reported_as_a_transport_error(self):
        response = {
            "choices": [
                {"finish_reason": "length", "message": {"content": "", "tool_calls": []}}
            ]
        }
        code, details = benchmark.extract_chat_code(response, "json_tool")
        self.assertIsNone(code)
        self.assertEqual(benchmark.classify(200, details, None), "output_limit")

    def test_extracts_custom_freeform_code(self):
        response = {
            "output": [
                {
                    "type": "custom_tool_call",
                    "name": "python",
                    "input": "import tools\n",
                }
            ]
        }
        code, details = benchmark.extract_responses_code(response)
        self.assertEqual(code, "import tools\n")
        self.assertEqual(details["python_call_count"], 1)

    def test_classifies_custom_tool_downgrade(self):
        response = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": "python",
                    "arguments": '{"text":"not freeform code"}',
                }
            ],
        }
        code, details = benchmark.extract_responses_code(response)
        self.assertIsNone(code)
        self.assertEqual(
            benchmark.classify(200, details, None), "custom_tool_degraded"
        )

    def test_jsonl_writer_preserves_prior_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "result.jsonl"
            writer = benchmark.JsonlWriter(path, append=False)
            writer.write({"n": 1})
            writer.close()
            writer = benchmark.JsonlWriter(path, append=True)
            writer.write({"n": 2})
            writer.close()
            self.assertEqual(
                [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()],
                [{"n": 1}, {"n": 2}],
            )

    def test_artifact_hash_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "artifact"
            path.write_bytes(b"generalist\n")
            self.assertEqual(
                benchmark.sha256_file(path),
                "985bfe71d0f386c5bbce96722eb5dece3b132953561da58d05fc5bf093c0f466",
            )


if __name__ == "__main__":
    unittest.main()
