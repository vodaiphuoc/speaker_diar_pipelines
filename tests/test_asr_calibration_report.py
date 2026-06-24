import json
import tempfile
import unittest
from pathlib import Path

from SDP.onnx.asr.utils.calibration_report import (
    build_asr_calibration_report,
    compare_words,
    write_asr_calibration_report,
)


class ASRCalibrationReportTest(unittest.TestCase):
    def test_compare_words_reports_equal_replace_insert_and_delete(self):
        self.assertEqual(
            compare_words("xin chào thế giới", "xin chào thế giới")["same"],
            True,
        )

        replacement = compare_words("xin chào thế giới", "xin chào bạn")
        self.assertFalse(replacement["same"])
        self.assertIn(
            {
                "op": "replace",
                "native_words": ["thế", "giới"],
                "onnx_words": ["bạn"],
                "native_range": [2, 4],
                "onnx_range": [2, 3],
                "native_timestamps": None,
                "onnx_timestamps": None,
            },
            replacement["operations"],
        )

        insertion = compare_words("xin chào", "xin chào bạn")
        self.assertIn("insert", [op["op"] for op in insertion["operations"]])

        deletion = compare_words("xin chào bạn", "xin chào")
        self.assertIn("delete", [op["op"] for op in deletion["operations"]])

    def test_build_report_contains_transcripts_tokens_timestamps_and_diff(self):
        report = build_asr_calibration_report(
            audio_file="tests/fixtures/bacsidatnhkhoavitadoc_1.wav",
            native_text="xin chào thế giới",
            native_token_ids=(1, 2),
            native_token_timestamps=(10, 11),
            onnx_text="xin chào bạn",
            onnx_token_ids=(1, 3),
            onnx_token_times=((0.8, 0.88), (0.88, 0.96)),
        )

        self.assertEqual(
            report["audio_file"],
            "tests/fixtures/bacsidatnhkhoavitadoc_1.wav",
        )
        self.assertEqual(report["native_nemo"]["full_text"], "xin chào thế giới")
        self.assertEqual(report["native_nemo"]["token_ids"], [1, 2])
        self.assertEqual(report["native_nemo"]["token_timestamps"], [10, 11])
        self.assertEqual(report["onnx_streaming"]["full_text"], "xin chào bạn")
        self.assertEqual(report["onnx_streaming"]["token_ids"], [1, 3])
        self.assertEqual(
            report["onnx_streaming"]["token_times"],
            [[0.8, 0.88], [0.88, 0.96]],
        )
        self.assertFalse(report["word_diff"]["same"])
        self.assertFalse(report["exact_match"]["text"])
        self.assertFalse(report["exact_match"]["token_ids"])

    def test_write_report_creates_json_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "asr_report.json"
            report = build_asr_calibration_report(
                audio_file="audio.wav",
                native_text="a",
                native_token_ids=(1,),
                native_token_timestamps=None,
                onnx_text="a",
                onnx_token_ids=(1,),
                onnx_token_times=(),
            )

            write_asr_calibration_report(output_path, report)

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                report,
            )


if __name__ == "__main__":
    unittest.main()
