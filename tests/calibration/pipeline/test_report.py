import unittest

from SDP.onnx.asr import StreamingASREvent
from SDP.onnx.streaming_service import StreamingDiarizationEvent
from SDP.pipeline import MergedSpeechSegment
from SDP.pipeline.calibration_report import (
    build_pipeline_calibration_report,
    build_pipeline_raw_events_report,
)


class PipelineCalibrationReportTest(unittest.TestCase):
    def test_report_serializes_segments_and_word_diff(self):
        report = build_pipeline_calibration_report(
            audio_file="sample.wav",
            alignment_mode="asr_timeline",
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
        self.assertEqual(report["alignment_mode"], "asr_timeline")
        self.assertTrue(report["exact_match"]["speaker_ids"])
        self.assertTrue(report["exact_match"]["timestamps_within_tolerance"])
        self.assertFalse(report["exact_match"]["text"])
        self.assertEqual(report["native_pipeline"]["segments"][0]["text"], "xin chào")
        self.assertEqual(report["onnx_pipeline"]["segments"][0]["text"], "xin bạn")
        self.assertFalse(report["word_diff"]["same"])
        self.assertEqual(report["word_diff"]["native_words"], ["xin", "chào"])
        self.assertEqual(report["word_diff"]["onnx_words"], ["xin", "bạn"])

    def test_report_includes_raw_native_and_onnx_events(self):
        native_diarization_event = StreamingDiarizationEvent(
            stream_id="native",
            sequence_id=0,
            speaker_id=1,
            start=0.0,
            end=1.2,
        )
        native_asr_event = StreamingASREvent(
            stream_id="native",
            sequence_id=0,
            token_ids=(11, 12),
            text_delta="xin chào",
            full_text="xin chào",
            token_times=((0.08, 0.16), (0.16, 0.24)),
            start=0.08,
            end=0.24,
            is_final=False,
        )
        onnx_diarization_event = StreamingDiarizationEvent(
            stream_id="onnx",
            sequence_id=2,
            speaker_id=1,
            start=0.0,
            end=1.0,
        )
        onnx_asr_event = StreamingASREvent(
            stream_id="onnx",
            sequence_id=3,
            token_ids=(21,),
            text_delta="xin",
            full_text="xin",
            token_times=((0.1, 0.2),),
            start=0.1,
            end=0.2,
            is_final=True,
        )

        report = build_pipeline_calibration_report(
            audio_file="sample.wav",
            alignment_mode="diarization_timeline",
            native_segments=(),
            onnx_segments=(),
            native_diarization_events=(native_diarization_event,),
            native_asr_events=(native_asr_event,),
            onnx_diarization_events=(onnx_diarization_event,),
            onnx_asr_events=(onnx_asr_event,),
        )

        self.assertEqual(
            report["raw_events"]["native"]["diarization_events"][0]["speaker_id"],
            1,
        )
        self.assertEqual(
            report["raw_events"]["native"]["asr_events"][0]["token_times"],
            [[0.08, 0.16], [0.16, 0.24]],
        )
        self.assertEqual(
            report["raw_events"]["onnx"]["diarization_events"][0]["sequence_id"],
            2,
        )
        self.assertEqual(
            report["raw_events"]["onnx"]["asr_events"][0]["full_text"],
            "xin",
        )
        self.assertEqual(report["raw_events"]["native"]["asr_full_text"], "xin chào")
        self.assertEqual(report["raw_events"]["onnx"]["asr_text_delta_joined"], "xin")

    def test_builds_standalone_raw_events_report(self):
        native_asr_event = StreamingASREvent(
            stream_id="native",
            sequence_id=0,
            token_ids=(11,),
            text_delta="xin",
            full_text="xin",
            token_times=((0.08, 0.16),),
            start=0.08,
            end=0.16,
            is_final=True,
        )

        report = build_pipeline_raw_events_report(
            audio_file="sample.wav",
            alignment_mode="asr_timeline",
            native_diarization_events=(),
            native_asr_events=(native_asr_event,),
            onnx_diarization_events=(),
            onnx_asr_events=(),
        )

        self.assertEqual(report["audio_file"], "sample.wav")
        self.assertEqual(report["alignment_mode"], "asr_timeline")
        self.assertEqual(
            report["raw_events"]["native"]["asr_events"][0]["text_delta"],
            "xin",
        )
        self.assertEqual(report["raw_events"]["onnx"]["asr_events"], [])


if __name__ == "__main__":
    unittest.main()
