import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("inspect_manifest.py")
SPEC = importlib.util.spec_from_file_location("local_reasoning_manifest", MODULE_PATH)
inspector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(inspector)


class ManifestSummaryTests(unittest.TestCase):
    def test_components_and_referenced_bytes_are_kept_separate(self):
        manifest = {
            "config": {"size": 7},
            "layers": [
                {
                    "mediaType": inspector.TENSOR_MEDIA_TYPE,
                    "name": "model.language_model.layers.0.self_attn.q_proj.weight",
                    "size": 100,
                },
                {
                    "mediaType": inspector.TENSOR_MEDIA_TYPE,
                    "name": "mtp.fc.weight",
                    "size": 20,
                },
                {
                    "mediaType": inspector.TENSOR_MEDIA_TYPE,
                    "name": "model.visual.blocks.0.attn.qkv.weight",
                    "size": 30,
                },
                {
                    "mediaType": "application/vnd.ollama.image.json",
                    "name": "config.json",
                    "size": 4,
                },
            ],
        }

        result = inspector.summarize_manifest(manifest)

        self.assertEqual(result["artifact"]["layer_blobs"], 4)
        self.assertEqual(result["artifact"]["layer_bytes"], 154)
        self.assertEqual(result["artifact"]["referenced_bytes"], 161)
        self.assertEqual(result["tensors"]["blobs"], 3)
        self.assertEqual(result["tensors"]["bytes"], 150)
        self.assertEqual(
            result["tensors"]["groups"],
            {
                "language": {"blobs": 1, "bytes": 100},
                "mtp": {"blobs": 1, "bytes": 20},
                "vision": {"blobs": 1, "bytes": 30},
            },
        )

    def test_unknown_tensor_is_not_silently_counted_as_language(self):
        result = inspector.summarize_manifest(
            {
                "layers": [
                    {
                        "mediaType": inspector.TENSOR_MEDIA_TYPE,
                        "name": "unexpected.weight",
                        "size": 11,
                    }
                ]
            }
        )

        self.assertEqual(
            result["tensors"]["groups"]["other"], {"blobs": 1, "bytes": 11}
        )

    def test_malformed_layer_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer size"):
            inspector.summarize_manifest(
                {"layers": [{"mediaType": inspector.TENSOR_MEDIA_TYPE, "size": -1}]}
            )


if __name__ == "__main__":
    unittest.main()
