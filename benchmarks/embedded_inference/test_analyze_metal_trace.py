import tempfile
import unittest
from pathlib import Path

import analyze_metal_trace as analyzer


class MetalTraceAnalyzerTests(unittest.TestCase):
    def write_temporary(self, contents: str) -> Path:
        temporary = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        self.addCleanup(Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(contents)
        return Path(temporary.name)

    def test_segments_warm_steps_at_argmax(self):
        path = self.write_temporary(
            "program output\n"
            "MLX_METAL_ENCODER\t1\t0\t1\tprepare\n"
            "MLX_METAL_ENCODER\t1\t1\t2\tqmv\targmax_bfloat16\n"
            "MLX_METAL_ENCODER\t1\t2\t1\tqmv\n"
            "MLX_METAL_ENCODER\t1\t3\t2\tadd\targmax_bfloat16\n"
            "MLX_METAL_ENCODER\t1\t4\t1\tqmv\n"
            "MLX_METAL_ENCODER\t1\t5\t1\targmax_bfloat16\n"
        )
        summary = analyzer.summarize_trace(analyzer.parse_encoder_log(path))
        self.assertEqual(summary["argmax_boundaries"], [1, 3, 5])
        self.assertEqual(
            [step["dispatch_count"] for step in summary["warm_steps"]], [3, 2]
        )
        self.assertEqual(
            summary["warm_kernel_counts"],
            [
                {"kernel": "argmax_bfloat16", "dispatches": 2},
                {"kernel": "qmv", "dispatches": 2},
                {"kernel": "add", "dispatches": 1},
            ],
        )

    def test_rejects_declared_dispatch_count_mismatch(self):
        path = self.write_temporary("MLX_METAL_ENCODER\t1\t0\t2\tone\n")
        with self.assertRaisesRegex(ValueError, "declared 2 dispatches"):
            analyzer.parse_encoder_log(path)

    def test_resolves_instruments_references_and_unions_nested_intervals(self):
        path = self.write_temporary(
            """<?xml version="1.0"?>
<trace-query-result><node><schema name="metal-gpu-intervals">
<col><mnemonic>start</mnemonic></col><col><mnemonic>duration</mnemonic></col>
<col><mnemonic>process</mnemonic></col><col><mnemonic>encoder-id</mnemonic></col>
</schema>
<row><start-time id="1">100</start-time><duration id="2">10</duration>
<process id="3" fmt="embedded_decode_trace (42)"/><metal-command-buffer-id id="4">11</metal-command-buffer-id></row>
<row><start-time id="5">102</start-time><duration id="6">5</duration>
<process ref="3"/><metal-command-buffer-id ref="4"/></row>
<row><start-time id="7">120</start-time><duration id="8">20</duration>
<process ref="3"/><metal-command-buffer-id id="9">12</metal-command-buffer-id></row>
<row><start-time id="10">200</start-time><duration ref="2"/>
<process id="11" fmt="another_process (7)"/><metal-command-buffer-id id="12">99</metal-command-buffer-id></row>
</node></trace-query-result>
"""
        )
        intervals = analyzer.parse_gpu_intervals(path, "embedded_decode_trace")
        self.assertEqual(len(intervals), 3)
        first_encoder = [interval for interval in intervals if interval.encoder_id == 11]
        self.assertEqual(
            analyzer.union_duration_ns(
                (interval.start_ns, interval.end_ns) for interval in first_encoder
            ),
            10,
        )
        records = [
            analyzer.EncoderRecord(1, 0, ("argmax_bfloat16",)),
            analyzer.EncoderRecord(1, 1, ("argmax_bfloat16",)),
        ]
        summary = analyzer.summarize_trace(records, intervals)
        self.assertEqual(summary["warm_steps"][0]["gpu_busy_milliseconds"], 0.00002)
        self.assertEqual(summary["warm_steps"][0]["gpu_span_milliseconds"], 0.00002)


if __name__ == "__main__":
    unittest.main()
