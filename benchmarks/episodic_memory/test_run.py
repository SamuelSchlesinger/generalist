import importlib.util
import json
import pathlib
import unittest


HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load_module("episodic_memory_run", "run.py")
ui_probe = load_module("episodic_memory_ui_probe", "ui_probe.py")


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads((HERE / "cases.json").read_text())

    def test_corpus_ids_and_references_are_closed(self):
        self.assertEqual(self.corpus["schema_version"], 1)
        record_ids = [record["id"] for record in self.corpus["records"]]
        query_ids = [query["id"] for query in self.corpus["queries"]]
        self.assertEqual(len(record_ids), len(set(record_ids)))
        self.assertEqual(len(query_ids), len(set(query_ids)))
        for query in self.corpus["queries"]:
            self.assertTrue(set(query["relevant_record_ids"]).issubset(record_ids))
        self.assertTrue(
            any(not query["relevant_record_ids"] for query in self.corpus["queries"])
        )

    def test_other_project_uses_a_deliberate_collision(self):
        marker = self.corpus["queries"][0]["query"]
        self.assertIn(marker, self.corpus["other_project_record"]["user"])


class RunnerTests(unittest.TestCase):
    def test_json_line_loader_uses_final_nonempty_line(self):
        self.assertEqual(runner.load_json_line("noise\n\n{\"passed\":true}\n"), {"passed": True})
        with self.assertRaisesRegex(ValueError, "no JSON"):
            runner.load_json_line("\n")

    def test_provider_fixture_extracts_string_and_block_text(self):
        self.assertEqual(ui_probe._message_text({"content": "plain"}), "plain")
        self.assertEqual(
            ui_probe._message_text(
                {"content": [{"text": "first"}, {"type": "ignored"}, {"text": "last"}]}
            ),
            "first\nlast",
        )


if __name__ == "__main__":
    unittest.main()
