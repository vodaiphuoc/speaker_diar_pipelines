import unittest

import numpy as np
import torch

from SDP.onnx.asr.streaming import (
    ASRDecoderJointONNXRunner,
    ASREncoderONNXRunner,
    ASRFeatureChunk,
    RNNTDecoderState,
    StatefulGreedyRNNTDecoder,
    StreamingASREvent,
    StreamingASRFeatureBuffer,
    StreamingASRSession,
    load_asr_runtime_config,
)
from SDP.onnx.asr.types import EncoderConfig


class FakePreprocessor:
    def __call__(self, input_signal: torch.Tensor, length: torch.Tensor):
        feature_count = int(length[0].item())
        features = input_signal[:, :feature_count].unsqueeze(1)
        return features, torch.tensor([feature_count], dtype=torch.int64)


class FakeDecoderJoint:
    def __init__(self, predictions):
        self.predictions = list(predictions)
        self.calls = []

    def initial_states(self, batch_size):
        return (
            np.zeros((2, batch_size, 1), dtype=np.float32),
            np.zeros((2, batch_size, 1), dtype=np.float32),
        )

    def __call__(self, encoder_frame, target, target_length, states):
        self.calls.append(
            {
                "target": target.copy(),
                "states": tuple(state.copy() for state in states),
            }
        )
        prediction = self.predictions.pop(0)
        logits = np.full((1, 1, 1, 4), -10.0, dtype=np.float32)
        logits[..., prediction] = 10.0
        next_states = tuple(state + 1 for state in states)
        return logits, next_states


class FakeFeatureBuffer:
    def __init__(self, ready_chunks=None, flush_chunks=None):
        self.ready_chunks = list(ready_chunks or [])
        self.flush_chunks = list(flush_chunks or [])
        self.updates = []

    def update(self, audio):
        self.updates.append(audio.copy())

    def pop_ready_chunk(self):
        if not self.ready_chunks:
            return None
        return self.ready_chunks.pop(0)

    def flush(self):
        chunks = self.flush_chunks
        self.flush_chunks = []
        return chunks


class FakeEncoder:
    def initial_cache_state(self, batch_size):
        return (
            np.zeros((batch_size, 1, 1, 1), dtype=np.float32),
            np.zeros((batch_size, 1, 1, 1), dtype=np.float32),
            np.zeros((batch_size,), dtype=np.int64),
        )

    def __call__(self, features, length, cache):
        encoded = np.ones((1, 1, 1), dtype=np.float32)
        return encoded, np.array([1], dtype=np.int64), cache


class FakePrompt:
    def __init__(self):
        self.indices = []

    def __call__(self, encoded, prompt_index):
        self.indices.append(prompt_index)
        return encoded


class FakeTokenizer:
    def decode(self, token_ids):
        return "".join({1: "hello", 2: " world"}[token] for token in token_ids)


class StreamingASRFeatureBufferTest(unittest.TestCase):
    def test_builds_every_encoder_input_from_history_plus_current_frames(self):
        buffer = StreamingASRFeatureBuffer(
            preprocessor=FakePreprocessor(),
            feature_dim=1,
            sample_rate=1,
            window_stride=1.0,
            n_fft=2,
            current_frames=2,
            history_frames=1,
            subsampling_factor=1,
        )

        buffer.update(np.arange(4, dtype=np.float32))

        first = buffer.pop_ready_chunk()
        self.assertIsInstance(first, ASRFeatureChunk)
        self.assertEqual(first.features.tolist(), [[[0.0, 0.0, 1.0]]])
        self.assertEqual(first.length.tolist(), [3])
        self.assertEqual(first.frame_offset, 0)
        self.assertEqual(first.valid_current_frames, 2)

        buffer.update(np.array([4.0, 5.0], dtype=np.float32))
        second = buffer.pop_ready_chunk()

        self.assertEqual(second.features.tolist(), [[[1.0, 2.0, 3.0]]])
        self.assertEqual(second.frame_offset, 2)
        self.assertEqual(second.valid_current_frames, 2)

    def test_flush_pads_only_the_uncommitted_current_frames(self):
        buffer = StreamingASRFeatureBuffer(
            preprocessor=FakePreprocessor(),
            feature_dim=1,
            sample_rate=1,
            window_stride=1.0,
            n_fft=2,
            current_frames=2,
            history_frames=1,
            subsampling_factor=1,
        )
        buffer.update(np.arange(4, dtype=np.float32))
        buffer.pop_ready_chunk()

        chunks = buffer.flush()

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].features.tolist(), [[[1.0, 2.0, 3.0]]])
        self.assertEqual(chunks[0].length.tolist(), [3])
        self.assertEqual(chunks[0].valid_current_frames, 2)

    def test_discards_waveform_that_is_older_than_preprocessor_context(self):
        buffer = StreamingASRFeatureBuffer(
            preprocessor=FakePreprocessor(),
            feature_dim=1,
            sample_rate=1,
            window_stride=1.0,
            n_fft=2,
            current_frames=2,
            history_frames=1,
            subsampling_factor=1,
        )

        buffer.update(np.arange(100, dtype=np.float32))
        while buffer.pop_ready_chunk() is not None:
            pass

        self.assertLessEqual(buffer.retained_sample_count, 3)


class StatefulGreedyRNNTDecoderTest(unittest.TestCase):
    def test_blank_does_not_advance_decoder_state_or_last_label(self):
        joint = FakeDecoderJoint(predictions=[1, 3])
        decoder = StatefulGreedyRNNTDecoder(
            decoder_joint=joint,
            blank_id=3,
            max_symbols_per_step=4,
        )
        state = RNNTDecoderState.create(joint.initial_states(batch_size=1), blank_id=3)

        token_ids, token_frames = decoder.decode(
            encoded=np.zeros((1, 1, 1), dtype=np.float32),
            encoded_length=1,
            state=state,
            frame_offset=7,
        )

        self.assertEqual(token_ids, [1])
        self.assertEqual(token_frames, [7])
        self.assertEqual(state.last_label, 1)
        np.testing.assert_array_equal(state.states[0], np.ones((2, 1, 1)))
        self.assertEqual(joint.calls[0]["target"].tolist(), [[3]])
        self.assertEqual(joint.calls[1]["target"].tolist(), [[1]])

    def test_blank_on_first_prediction_keeps_zero_state(self):
        joint = FakeDecoderJoint(predictions=[3])
        decoder = StatefulGreedyRNNTDecoder(joint, blank_id=3)
        state = RNNTDecoderState.create(joint.initial_states(batch_size=1), blank_id=3)

        token_ids, _ = decoder.decode(
            encoded=np.zeros((1, 1, 1), dtype=np.float32),
            encoded_length=1,
            state=state,
            frame_offset=0,
        )

        self.assertEqual(token_ids, [])
        self.assertEqual(state.last_label, 3)
        np.testing.assert_array_equal(state.states[0], np.zeros((2, 1, 1)))


class StreamingASRSessionTest(unittest.TestCase):
    def make_chunk(self, frame_offset=0):
        return ASRFeatureChunk(
            features=torch.zeros((1, 1, 3), dtype=torch.float32),
            length=torch.tensor([3], dtype=torch.int64),
            frame_offset=frame_offset,
            valid_current_frames=2,
        )

    def test_process_pcm_emits_stable_delta_and_token_timing(self):
        feature_buffer = FakeFeatureBuffer(ready_chunks=[self.make_chunk(frame_offset=5)])
        prompt = FakePrompt()
        joint = FakeDecoderJoint(predictions=[1, 3])
        session = StreamingASRSession(
            feature_buffer=feature_buffer,
            encoder=FakeEncoder(),
            prompt_projection=prompt,
            decoder_joint=joint,
            tokenizer=FakeTokenizer(),
            blank_id=3,
            prompt_index=33,
            frame_duration=0.08,
        )

        events = session.process_pcm(
            np.array([32767, -32768], dtype=np.int16).tobytes(),
            stream_id="call-1",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0],
            StreamingASREvent(
                stream_id="call-1",
                sequence_id=0,
                token_ids=(1,),
                text_delta="hello",
                full_text="hello",
                token_times=((0.4, 0.48),),
                start=0.4,
                end=0.48,
                is_final=False,
            ),
        )
        np.testing.assert_allclose(
            feature_buffer.updates[0],
            np.array([32767 / 32768.0, -1.0], dtype=np.float32),
        )
        self.assertEqual(prompt.indices, [33])

    def test_flush_emits_final_event_even_without_new_tokens(self):
        session = StreamingASRSession(
            feature_buffer=FakeFeatureBuffer(),
            encoder=FakeEncoder(),
            prompt_projection=FakePrompt(),
            decoder_joint=FakeDecoderJoint(predictions=[]),
            tokenizer=FakeTokenizer(),
            blank_id=3,
            prompt_index=33,
        )

        events = session.flush(stream_id="default")

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].is_final)
        self.assertEqual(events[0].text_delta, "")
        self.assertEqual(events[0].full_text, "")

    def test_rejects_misaligned_pcm_and_stream_id_changes(self):
        session = StreamingASRSession(
            feature_buffer=FakeFeatureBuffer(),
            encoder=FakeEncoder(),
            prompt_projection=FakePrompt(),
            decoder_joint=FakeDecoderJoint(predictions=[]),
            tokenizer=FakeTokenizer(),
            blank_id=3,
            prompt_index=33,
        )

        with self.assertRaisesRegex(ValueError, "16-bit"):
            session.process_pcm(b"\x00")

        session.process_pcm(b"\x00\x00", stream_id="first")
        with self.assertRaisesRegex(ValueError, "one active stream"):
            session.process_pcm(b"\x00\x00", stream_id="second")


class ONNXRunnerStateTest(unittest.TestCase):
    def test_encoder_cache_shapes_match_exported_batch_first_contract(self):
        runner = ASREncoderONNXRunner.__new__(ASREncoderONNXRunner)
        runner.encoder_config = EncoderConfig()

        channel, time, length = runner.initial_cache_state(batch_size=2)

        self.assertEqual(channel.shape, (2, 24, 56, 1024))
        self.assertEqual(time.shape, (2, 24, 1024, 8))
        self.assertEqual(length.shape, (2,))
        self.assertEqual(channel.dtype, np.float32)
        self.assertEqual(length.dtype, np.int64)

    def test_decoder_initial_states_match_two_layer_lstm_contract(self):
        runner = ASRDecoderJointONNXRunner.__new__(ASRDecoderJointONNXRunner)
        runner.pred_rnn_layers = 2
        runner.pred_hidden = 640

        states = runner.initial_states(batch_size=3)

        self.assertEqual(len(states), 2)
        self.assertEqual(states[0].shape, (2, 3, 640))
        self.assertEqual(states[1].shape, (2, 3, 640))

    def test_runtime_config_resolves_vietnamese_prompt_and_blank_id(self):
        config = load_asr_runtime_config(
            "configs/asr_pretrained_config.yaml",
            target_language="vi-VN",
        )

        self.assertEqual(config.prompt_index, 33)
        self.assertEqual(config.blank_id, 13087)
        self.assertEqual(config.feature_dim, 128)
        self.assertEqual(config.sample_rate, 16000)

    def test_runtime_config_rejects_unknown_language(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ASR language"):
            load_asr_runtime_config(
                "configs/asr_pretrained_config.yaml",
                target_language="not-a-language",
            )


if __name__ == "__main__":
    unittest.main()
