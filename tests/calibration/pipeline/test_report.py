import unittest

from SDP.pipeline import MergedSpeechSegment
from SDP.pipeline.calibration_report import build_pipeline_calibration_report


class PipelineCalibrationReportTest(unittest.TestCase):
    def test_report_serializes_segments_and_word_diff(self):
        report = build_pipeline_calibration_report(
            audio_file="sample.wav",
            native_segments=(
                MergedSpeechSegment(
                    stream_id="native",
                    sequence_id=0,
                    speaker_id=0,
                    start=0.0,
                    end=1.0,
                    text="xin chào",
                ),
            ),
            onnx_segments=(
                MergedSpeechSegment(
                    stream_id="onnx",
                    sequence_id=0,
                    speaker_id=0,
                    start=0.02,
                    end=1.04,
                    text="xin bạn",
                ),
            ),
            timestamp_tolerance=0.1,
        )

        self.assertEqual(report["audio_file"], "sample.wav")
        self.assertTrue(report["exact_match"]["speaker_ids"])
        self.assertTrue(report["exact_match"]["timestamps_within_tolerance"])
        self.assertFalse(report["exact_match"]["text"])
        self.assertEqual(report["native_pipeline"]["segments"][0]["text"], "xin chào")
        self.assertEqual(report["onnx_pipeline"]["segments"][0]["text"], "xin bạn")
        self.assertFalse(report["word_diff"]["same"])
        self.assertEqual(report["word_diff"]["native_words"], ["xin", "chào"])
        self.assertEqual(report["word_diff"]["onnx_words"], ["xin", "bạn"])


if __name__ == "__main__":
    unittest.main()
