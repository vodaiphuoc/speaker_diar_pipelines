import unittest

import torch

from SDP.onnx.diarization.nemo_vad_utils import ts_vad_post_processing
from SDP.onnx.diarization.post_processing import StreamingDiarizationPostProcessor
from SDP.onnx.diarization.types import PostProcessingParams


def flatten_speaker_outputs(outputs, num_spks):
    merged = [[] for _ in range(num_spks)]
    for output in outputs:
        for spk in range(num_spks):
            merged[spk].extend(output[spk])
    return merged


def round_segments(segments):
    return [[round(start, 2), round(end, 2)] for start, end in segments]


class StreamingDiarizationPostProcessorTest(unittest.TestCase):
    def test_buffered_window_matches_offline_post_processing_after_flush(self):
        cfg = PostProcessingParams(
            onset=0.5,
            offset=0.5,
            pad_onset=0.0,
            pad_offset=0.0,
            min_duration_on=0.02,
            min_duration_off=0.03,
        )
        probs = torch.tensor(
            [
                [0.1, 0.1],
                [0.9, 0.1],
                [0.9, 0.8],
                [0.1, 0.8],
                [0.1, 0.1],
                [0.9, 0.1],
                [0.9, 0.9],
                [0.1, 0.9],
                [0.1, 0.1],
                [0.1, 0.1],
            ],
            dtype=torch.float32,
        )

        processor = StreamingDiarizationPostProcessor(
            cfg_vad_params=cfg,
            num_spks=2,
            unit_10ms_frame_count=1,
            processing_mode="buffered_window",
            buffer_window_sec=1.0,
            commit_delay_sec=0.03,
        )

        outputs = [processor.process_chunk(chunk) for chunk in probs.split(3)]
        outputs.append(processor.flush())
        actual = flatten_speaker_outputs(outputs, num_spks=2)

        expected = []
        for spk in range(2):
            offline = ts_vad_post_processing(
                probs[:, spk],
                cfg_vad_params=cfg,
                unit_10ms_frame_count=1,
                bypass_postprocessing=False,
            )
            expected.append(round_segments(offline.tolist()))

        self.assertEqual(round_segments(actual[0]), expected[0])
        self.assertEqual(round_segments(actual[1]), expected[1])

    def test_incremental_mode_waits_to_merge_short_gap_across_chunks(self):
        cfg = PostProcessingParams(
            onset=0.5,
            offset=0.5,
            pad_onset=0.0,
            pad_offset=0.0,
            min_duration_on=0.0,
            min_duration_off=0.03,
        )
        processor = StreamingDiarizationPostProcessor(
            cfg_vad_params=cfg,
            num_spks=1,
            unit_10ms_frame_count=1,
            processing_mode="incremental",
        )

        first = torch.tensor([[0.9], [0.9], [0.1]], dtype=torch.float32)
        second = torch.tensor([[0.1], [0.9], [0.9], [0.1], [0.1], [0.1]], dtype=torch.float32)

        self.assertEqual(processor.process_chunk(first), [[]])
        self.assertEqual(round_segments(processor.process_chunk(second)[0]), [[0.0, 0.06]])
        self.assertEqual(processor.flush(), [[]])

    def test_buffered_window_delays_output_until_commit_watermark(self):
        cfg = PostProcessingParams(
            onset=0.5,
            offset=0.5,
            pad_onset=0.0,
            pad_offset=0.0,
            min_duration_on=0.0,
            min_duration_off=0.03,
        )
        processor = StreamingDiarizationPostProcessor(
            cfg_vad_params=cfg,
            num_spks=1,
            unit_10ms_frame_count=1,
            processing_mode="buffered_window",
            buffer_window_sec=1.0,
            commit_delay_sec=0.03,
        )

        first = torch.tensor([[0.9], [0.9], [0.1]], dtype=torch.float32)
        second = torch.tensor([[0.1], [0.9], [0.9]], dtype=torch.float32)
        third = torch.tensor([[0.1], [0.1], [0.1], [0.1]], dtype=torch.float32)

        self.assertEqual(processor.process_chunk(first), [[]])
        self.assertEqual(processor.process_chunk(second), [[]])
        self.assertEqual(round_segments(processor.process_chunk(third)[0]), [[0.0, 0.06]])
        self.assertEqual(processor.flush(), [[]])

    def test_flush_emits_active_tail_segment(self):
        cfg = PostProcessingParams(onset=0.5, offset=0.5)
        processor = StreamingDiarizationPostProcessor(
            cfg_vad_params=cfg,
            num_spks=1,
            unit_10ms_frame_count=1,
            processing_mode="buffered_window",
            buffer_window_sec=1.0,
            commit_delay_sec=0.5,
        )

        self.assertEqual(
            processor.process_chunk(torch.tensor([[0.1], [0.9], [0.9]], dtype=torch.float32)),
            [[]],
        )
        self.assertEqual(round_segments(processor.flush()[0]), [[0.01, 0.02]])

    def test_reset_clears_state_and_invalid_shapes_raise(self):
        cfg = PostProcessingParams(onset=0.5, offset=0.5)
        processor = StreamingDiarizationPostProcessor(
            cfg_vad_params=cfg,
            num_spks=1,
            unit_10ms_frame_count=1,
            processing_mode="buffered_window",
        )

        processor.process_chunk(torch.tensor([[0.9], [0.1], [0.1]], dtype=torch.float32))
        processor.reset()
        self.assertEqual(processor.flush(), [[]])

        with self.assertRaises(ValueError):
            processor.process_chunk(torch.tensor([0.1, 0.9], dtype=torch.float32))
        with self.assertRaises(ValueError):
            processor.process_chunk(torch.tensor([[0.1, 0.2]], dtype=torch.float32))
        with self.assertRaises(ValueError):
            StreamingDiarizationPostProcessor(
                cfg_vad_params=cfg,
                num_spks=1,
                processing_mode="unsupported",
            )


if __name__ == "__main__":
    unittest.main()
