import unittest
import asyncio
import inspect
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import torch

from SDP.onnx.diarization.types import StreamingSortformerState
from SDP.onnx.preprocess.feature_buffer import FeatureBufferChunk
from SDP.onnx.streaming_service import (
    StreamingDiarizationASROnnxService,
    StreamingDiarizationEvent,
    StreamingDiarizerOnnxService,
    StreamingPipelineResult,
)
from SDP.onnx.asr import StreamingASREvent
from SDP.pipeline import MergedSpeechSegment


class FakeFeatureBufferer:
    def __init__(self):
        self.update_calls = []
        self.ready_chunks = []
        self.flush_chunks = []

    def update(self, audio):
        self.update_calls.append(audio.copy())

    def pop_ready_feature_chunk(self):
        if not self.ready_chunks:
            return None
        return self.ready_chunks.pop(0)

    def flush_ready_feature_chunks(self):
        chunks = self.flush_chunks
        self.flush_chunks = []
        return chunks


class FakeDiarizer:
    def __init__(self):
        self.calls = []
        self.update_calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch, frames, _ = kwargs["chunk"].shape
        preds = torch.zeros((batch, frames, 1), dtype=torch.float32)
        encoded_frames = frames // 2
        embs = torch.zeros((batch, encoded_frames, 1), dtype=torch.float32)
        lengths = torch.tensor([encoded_frames], dtype=torch.long)
        return preds, embs, lengths

    def streaming_update_async(self, **kwargs):
        self.update_calls.append(kwargs)
        chunk = kwargs["chunk"]
        lc = kwargs["lc"]
        rc = kwargs["rc"]
        committed = chunk.shape[1] - lc - rc
        preds = torch.full((1, committed, 1), 0.9, dtype=torch.float32)
        return kwargs["streaming_state"], preds


class FakePostProcessor:
    def __init__(self):
        self.chunks = []
        self.flush_called = False

    def process_chunk(self, chunk):
        self.chunks.append(chunk.clone())
        return [[[0.1, 0.2], [0.3, 0.4]]]

    def flush(self):
        self.flush_called = True
        return [[[0.5, 0.6]]]


class StreamingServiceContextTest(unittest.TestCase):
    def make_service(self):
        service = StreamingDiarizerOnnxService.__new__(StreamingDiarizerOnnxService)
        service.device = torch.device("cpu")
        service.encoder_config = SimpleNamespace(subsampling_factor=2)
        service.sortformer_config = SimpleNamespace(num_spks=1)
        service.streaming_state = StreamingSortformerState(
            spkcache=torch.zeros((1, 0, 1)),
            spkcache_lengths=torch.zeros((1,), dtype=torch.long),
            spkcache_preds=torch.zeros((1, 0, 1)),
            fifo=torch.zeros((1, 0, 1)),
            fifo_lengths=torch.zeros((1,), dtype=torch.long),
            fifo_preds=None,
            spk_perm=None,
            mean_sil_emb=torch.zeros((1, 1)),
            n_sil_frames=torch.zeros((1,), dtype=torch.long),
        )
        service._feature_bufferer = FakeFeatureBufferer()
        service._diarizer = FakeDiarizer()
        service._post_diar_processor = FakePostProcessor()
        service._event_queue = None
        service._async_event_queue = None
        service._next_sequence_id = 0
        service._init_event_queues(enable_async_queue=False, async_queue_maxsize=0)
        return service

    def test_diarize_waits_until_feature_chunk_is_ready(self):
        service = self.make_service()

        result = service.diarize(np.array([1, 2], dtype=np.int16).tobytes())

        self.assertEqual(result, [])
        self.assertEqual(service.drain_events(), [])
        self.assertEqual(len(service._diarizer.calls), 0)
        self.assertEqual(len(service._feature_bufferer.update_calls), 1)

    def test_diarize_uses_ready_chunk_context_offsets_and_returns_events(self):
        service = self.make_service()
        feature_chunk = FeatureBufferChunk(
            features=torch.arange(8, dtype=torch.float32).view(1, 8),
            left_offset=2,
            right_offset=2,
            center_frame_count=4,
            center_diar_frame_count=2,
        )
        service._feature_bufferer.ready_chunks.append(feature_chunk)

        result = service.diarize(np.array([1], dtype=np.int16).tobytes())

        self.assertEqual(
            result,
            [
                StreamingDiarizationEvent(
                    stream_id="default",
                    sequence_id=0,
                    speaker_id=0,
                    start=0.1,
                    end=0.2,
                ),
                StreamingDiarizationEvent(
                    stream_id="default",
                    sequence_id=1,
                    speaker_id=0,
                    start=0.3,
                    end=0.4,
                ),
            ],
        )
        self.assertEqual(service.drain_events(), result)
        self.assertEqual(service.drain_events(), [])
        model_input = service._diarizer.calls[0]["chunk"]
        self.assertEqual(tuple(model_input.shape), (1, 8, 1))
        update_call = service._diarizer.update_calls[0]
        self.assertEqual(update_call["lc"], 1)
        self.assertEqual(update_call["rc"], 1)
        self.assertEqual(service._post_diar_processor.chunks[0].shape[0], 2)

    def test_flush_processes_tail_chunks_then_post_processor_tail(self):
        service = self.make_service()
        service._feature_bufferer.flush_chunks.append(
            FeatureBufferChunk(
                features=torch.arange(4, dtype=torch.float32).view(1, 4),
                left_offset=2,
                right_offset=0,
                center_frame_count=2,
                center_diar_frame_count=1,
            )
        )

        result = service.flush()

        self.assertEqual(
            result,
            [
                StreamingDiarizationEvent(
                    stream_id="default",
                    sequence_id=0,
                    speaker_id=0,
                    start=0.1,
                    end=0.2,
                ),
                StreamingDiarizationEvent(
                    stream_id="default",
                    sequence_id=1,
                    speaker_id=0,
                    start=0.3,
                    end=0.4,
                ),
                StreamingDiarizationEvent(
                    stream_id="default",
                    sequence_id=2,
                    speaker_id=0,
                    start=0.5,
                    end=0.6,
                ),
            ],
        )
        self.assertTrue(service._post_diar_processor.flush_called)
        self.assertEqual(service._diarizer.update_calls[0]["lc"], 1)
        self.assertEqual(service._diarizer.update_calls[0]["rc"], 0)

    def test_async_queue_receives_same_events(self):
        service = self.make_service()
        service._init_event_queues(enable_async_queue=True, async_queue_maxsize=0)
        service._feature_bufferer.ready_chunks.append(
            FeatureBufferChunk(
                features=torch.arange(8, dtype=torch.float32).view(1, 8),
                left_offset=2,
                right_offset=2,
                center_frame_count=4,
                center_diar_frame_count=2,
            )
        )

        result = service.diarize(np.array([1], dtype=np.int16).tobytes(), stream_id="s1")
        first = asyncio.run(service.get_event())
        second = asyncio.run(service.get_event())

        self.assertEqual([first, second], result)
        self.assertEqual(first.stream_id, "s1")
        self.assertEqual(first.sequence_id, 0)
        self.assertEqual(second.sequence_id, 1)


class FakeStreamingBranch:
    def __init__(self, *events):
        self.events = list(events)
        self.sample_calls = []
        self.flush_calls = []

    def process_samples(self, samples, stream_id):
        self.sample_calls.append((samples.copy(), stream_id))
        return list(self.events)

    def flush(self, stream_id):
        self.flush_calls.append(stream_id)
        return list(self.events)


class CombinedStreamingServiceTest(unittest.TestCase):
    def test_combined_service_public_methods_have_type_hints(self):
        init_hints = get_type_hints(StreamingDiarizationASROnnxService.__init__)
        process_hints = get_type_hints(StreamingDiarizationASROnnxService.process)
        flush_hints = get_type_hints(StreamingDiarizationASROnnxService.flush)

        self.assertIn("diarization_service", init_hints)
        self.assertIn("asr_session", init_hints)
        self.assertIs(init_hints["return"], type(None))
        self.assertIs(process_hints["audio"], bytes)
        self.assertIs(process_hints["return"], StreamingPipelineResult)
        self.assertIs(flush_hints["return"], StreamingPipelineResult)
        self.assertIsNot(
            inspect.signature(
                StreamingDiarizationASROnnxService._validate_stream
            ).return_annotation,
            inspect.Signature.empty,
        )

    def test_process_decodes_pcm_once_and_fans_same_samples_to_both_branches(self):
        diar_event = SimpleNamespace(event_type="diarization")
        asr_event = SimpleNamespace(event_type="asr")
        diarizer = FakeStreamingBranch(diar_event)
        asr = FakeStreamingBranch(asr_event)
        service = StreamingDiarizationASROnnxService(
            diarization_service=diarizer,
            asr_session=asr,
        )
        pcm = np.array([32767, -32768], dtype=np.int16).tobytes()

        result = service.process(pcm, stream_id="stream-1")

        self.assertEqual(
            result,
            StreamingPipelineResult(
                diarization_events=(diar_event,),
                asr_events=(asr_event,),
            ),
        )
        self.assertEqual(len(diarizer.sample_calls), 1)
        self.assertEqual(len(asr.sample_calls), 1)
        np.testing.assert_array_equal(
            diarizer.sample_calls[0][0],
            asr.sample_calls[0][0],
        )
        np.testing.assert_allclose(
            diarizer.sample_calls[0][0],
            np.array([32767 / 32768.0, -1.0], dtype=np.float32),
        )

    def test_flush_returns_independent_branch_events(self):
        diar_event = SimpleNamespace(event_type="diarization")
        asr_event = SimpleNamespace(event_type="asr")
        service = StreamingDiarizationASROnnxService(
            diarization_service=FakeStreamingBranch(diar_event),
            asr_session=FakeStreamingBranch(asr_event),
        )

        result = service.flush(stream_id="stream-1")

        self.assertEqual(result.diarization_events, (diar_event,))
        self.assertEqual(result.asr_events, (asr_event,))

    def test_process_returns_merged_segments_when_asr_covers_diarization(self):
        diar_event = StreamingDiarizationEvent(
            stream_id="stream-1",
            sequence_id=0,
            speaker_id=1,
            start=0.0,
            end=1.0,
        )
        asr_event = StreamingASREvent(
            stream_id="stream-1",
            sequence_id=0,
            token_ids=(7,),
            text_delta="xin chào",
            full_text="xin chào",
            token_times=((0.1, 1.1),),
            start=0.1,
            end=1.1,
            is_final=False,
        )
        service = StreamingDiarizationASROnnxService(
            diarization_service=FakeStreamingBranch(diar_event),
            asr_session=FakeStreamingBranch(asr_event),
        )

        result = service.process(
            np.array([1, 2], dtype=np.int16).tobytes(), stream_id="stream-1"
        )

        self.assertEqual(
            result.merged_segments,
            (
                MergedSpeechSegment(
                    stream_id="stream-1",
                    sequence_id=0,
                    speaker_id=1,
                    start=0.0,
                    end=1.0,
                    text="xin chào",
                    token_ids=(7,),
                    token_times=((0.1, 1.1),),
                ),
            ),
        )

    def test_flush_returns_remaining_merged_segments(self):
        diar_event = StreamingDiarizationEvent(
            stream_id="stream-1",
            sequence_id=0,
            speaker_id=1,
            start=0.0,
            end=1.0,
        )
        service = StreamingDiarizationASROnnxService(
            diarization_service=FakeStreamingBranch(diar_event),
            asr_session=FakeStreamingBranch(),
        )
        service.process(np.array([1], dtype=np.int16).tobytes(), stream_id="stream-1")

        result = service.flush(stream_id="stream-1")

        self.assertEqual(
            result.merged_segments,
            (
                MergedSpeechSegment(
                    stream_id="stream-1",
                    sequence_id=0,
                    speaker_id=1,
                    start=0.0,
                    end=1.0,
                    text="",
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
