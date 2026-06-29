import unittest

import numpy as np

from SDP.onnx.asr import StreamingASREvent
from SDP.qwen3_asr import Qwen3ASRModalConfig, Qwen3ASRModalSession


class FakeQwen3RemoteActor:
    def __init__(self, texts):
        self.texts = list(texts)
        self.transcribe_calls = []
        self.finish_calls = 0

    def transcribe(self, samples):
        self.transcribe_calls.append(np.asarray(samples, dtype=np.float32).copy())
        text = self.texts.pop(0) if self.texts else ""
        return {"language": "Vietnamese", "text": text}

    def finish(self):
        self.finish_calls += 1
        text = self.texts.pop(0) if self.texts else ""
        return {"language": "Vietnamese", "text": text}


class Qwen3ASRModalSessionTest(unittest.TestCase):
    def test_config_defaults_match_modal_trial(self):
        config = Qwen3ASRModalConfig()

        self.assertEqual(config.model_name, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(config.app_name, "sdp-qwen3-asr-streaming")
        self.assertEqual(config.volume_name, "speaker_diar_qwen3_asr_streaming_cache")
        self.assertEqual(config.gpu, "A10G")
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.step_ms, 1000)

    def test_process_buffers_until_step_size_before_calling_remote(self):
        remote = FakeQwen3RemoteActor(texts=["xin chào"])
        session = Qwen3ASRModalSession(
            remote_actor=remote,
            config=Qwen3ASRModalConfig(step_ms=1000),
        )

        events = session.process_samples(np.zeros(8000, dtype=np.float32))

        self.assertEqual(events, [])
        self.assertEqual(remote.transcribe_calls, [])

    def test_process_emits_standard_asr_event_from_cumulative_qwen_text(self):
        remote = FakeQwen3RemoteActor(texts=["xin chào"])
        session = Qwen3ASRModalSession(
            remote_actor=remote,
            config=Qwen3ASRModalConfig(step_ms=1000),
        )

        events = session.process_samples(np.ones(16000, dtype=np.float32), stream_id="s1")

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], StreamingASREvent)
        self.assertEqual(
            events[0],
            StreamingASREvent(
                stream_id="s1",
                sequence_id=0,
                token_ids=(),
                text_delta="xin chào",
                full_text="xin chào",
                token_times=(),
                start=0.0,
                end=1.0,
                is_final=False,
            ),
        )
        np.testing.assert_allclose(remote.transcribe_calls[0], np.ones(16000))

    def test_process_computes_delta_from_cumulative_text(self):
        remote = FakeQwen3RemoteActor(texts=["xin", "xin chào"])
        session = Qwen3ASRModalSession(
            remote_actor=remote,
            config=Qwen3ASRModalConfig(step_ms=500),
        )

        first = session.process_samples(np.ones(8000, dtype=np.float32), stream_id="s1")
        second = session.process_samples(np.ones(8000, dtype=np.float32), stream_id="s1")

        self.assertEqual(first[0].text_delta, "xin")
        self.assertEqual(first[0].start, 0.0)
        self.assertEqual(first[0].end, 0.5)
        self.assertEqual(second[0].text_delta, " chào")
        self.assertEqual(second[0].full_text, "xin chào")
        self.assertEqual(second[0].start, 0.5)
        self.assertEqual(second[0].end, 1.0)

    def test_flush_sends_remaining_audio_and_finalizes_remote(self):
        remote = FakeQwen3RemoteActor(texts=["xin", "xin chào"])
        session = Qwen3ASRModalSession(
            remote_actor=remote,
            config=Qwen3ASRModalConfig(step_ms=1000),
        )
        session.process_samples(np.ones(8000, dtype=np.float32), stream_id="s1")

        events = session.flush(stream_id="s1")

        self.assertEqual(remote.finish_calls, 1)
        self.assertEqual(len(remote.transcribe_calls), 1)
        self.assertEqual(events[-1].is_final, True)
        self.assertEqual(events[-1].full_text, "xin chào")
        self.assertEqual(events[-1].text_delta, " chào")
        self.assertEqual(events[-1].start, 0.0)
        self.assertEqual(events[-1].end, 0.5)

    def test_process_pcm_validates_int16_alignment(self):
        session = Qwen3ASRModalSession(
            remote_actor=FakeQwen3RemoteActor(texts=[]),
            config=Qwen3ASRModalConfig(step_ms=1000),
        )

        with self.assertRaisesRegex(ValueError, "aligned 16-bit samples"):
            session.process_pcm(b"\x00")

    def test_stream_id_and_flush_lifecycle_match_existing_asr_session(self):
        session = Qwen3ASRModalSession(
            remote_actor=FakeQwen3RemoteActor(texts=[]),
            config=Qwen3ASRModalConfig(step_ms=1000),
        )
        session.process_samples(np.zeros(1, dtype=np.float32), stream_id="a")

        with self.assertRaisesRegex(ValueError, "one active stream"):
            session.process_samples(np.zeros(1, dtype=np.float32), stream_id="b")

        session.flush(stream_id="a")
        with self.assertRaisesRegex(RuntimeError, "after ASR flush"):
            session.process_samples(np.zeros(1, dtype=np.float32), stream_id="a")


if __name__ == "__main__":
    unittest.main()
