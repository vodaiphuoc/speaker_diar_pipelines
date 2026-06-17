import unittest
from types import SimpleNamespace

import numpy as np
import torch

from SDP.onnx.preprocess.feature_buffer import CacheFeatureBufferer


class FakePreprocessor:
    def __call__(self, input_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        signal = input_signal[0, : int(length[0].item())].float()
        return signal.view(1, 1, -1)


class CacheFeatureBuffererContextTest(unittest.TestCase):
    def make_bufferer(self):
        cfg = SimpleNamespace(features=1, window_stride=1.0, log=False)
        return CacheFeatureBufferer(
            sample_rate=1,
            buffer_size_in_secs=8.0,
            chunk_size_in_secs=4.0,
            preprocessor_cfg=cfg,
            preprocessor=FakePreprocessor(),
            device=torch.device("cpu"),
            left_context_in_secs=2.0,
            right_context_in_secs=2.0,
        )

    def test_waits_for_right_context_before_emitting_first_chunk(self):
        bufferer = self.make_bufferer()
        bufferer.update(np.arange(5, dtype=np.float32))

        self.assertIsNone(bufferer.pop_ready_feature_chunk())

        bufferer.update(np.array([5], dtype=np.float32))
        chunk = bufferer.pop_ready_feature_chunk()

        self.assertEqual(chunk.left_offset, 0)
        self.assertEqual(chunk.right_offset, 2)
        self.assertEqual(chunk.center_frame_count, 4)
        self.assertEqual(chunk.features.tolist(), [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

    def test_later_chunks_have_left_and_right_context(self):
        bufferer = self.make_bufferer()
        bufferer.update(np.arange(10, dtype=np.float32))

        first = bufferer.pop_ready_feature_chunk()
        second = bufferer.pop_ready_feature_chunk()

        self.assertEqual(first.features.tolist(), [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
        self.assertEqual(second.left_offset, 2)
        self.assertEqual(second.right_offset, 2)
        self.assertEqual(second.center_frame_count, 4)
        self.assertEqual(
            second.features.tolist(),
            [[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]],
        )

    def test_flush_emits_final_partial_chunk_with_reduced_right_context(self):
        bufferer = self.make_bufferer()
        bufferer.update(np.arange(10, dtype=np.float32))
        bufferer.pop_ready_feature_chunk()
        bufferer.pop_ready_feature_chunk()

        tail = bufferer.flush_ready_feature_chunks()

        self.assertEqual(len(tail), 1)
        self.assertEqual(tail[0].left_offset, 2)
        self.assertEqual(tail[0].right_offset, 0)
        self.assertEqual(tail[0].center_frame_count, 2)
        self.assertEqual(tail[0].features.tolist(), [[6.0, 7.0, 8.0, 9.0]])


if __name__ == "__main__":
    unittest.main()
