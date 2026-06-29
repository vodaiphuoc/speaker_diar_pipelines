import asyncio
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from SDP.onnx.artifacts import (
    load_asr_artifact_manifest,
    load_diarization_artifact_manifest,
)
from SDP.onnx.asr.streaming import (
    StreamingASREvent,
    StreamingASRSession,
    create_nemotron_streaming_session_from_manifest,
)
from SDP.qwen3_asr import create_qwen3_asr_modal_session

# from SDP.onnx.diarization.nemo_vad_utils import ts_vad_post_processing
from SDP.onnx.diarization.post_processing import StreamingDiarizationPostProcessor
from SDP.onnx.diarization.streaming import SortformerONNXRunner
from SDP.onnx.diarization.types import (
    EncoderModuleConfig,
    PostProcessingParams,
    PreProcessorConfig,
    SortformerModuleConfig,
)
from SDP.onnx.diarization.utils import (
    load_encoder_modules_config,
    load_preprocessor_config,
    load_sortformer_modules_config,
)
from SDP.onnx.preprocess.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorOnnxRunner,
)
from SDP.onnx.preprocess.feature_buffer import CacheFeatureBufferer, FeatureBufferChunk
from SDP.pipeline import (
    AlignmentMode,
    MergedSpeechSegment,
    StreamingPipelineEventMerger,
)


@dataclass(frozen=True)
class StreamingDiarizationEvent:
    stream_id: str
    sequence_id: int
    speaker_id: int
    start: float
    end: float
    event_type: Literal["diarization"] = "diarization"
    is_final: bool = True


@dataclass(frozen=True)
class StreamingPipelineResult:
    diarization_events: tuple[StreamingDiarizationEvent, ...]
    asr_events: tuple[StreamingASREvent, ...]
    merged_segments: tuple[MergedSpeechSegment, ...] = ()


class StreamingDiarizerOnnxService(object):
    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        device: Literal["cpu", "cuda"] = "cpu",
        post_processing_config: PostProcessingParams = PostProcessingParams(),
        frame_len_in_secs: float = 0.08,
        sample_rate: int = 16000,
        left_offset: int = 8,
        right_offset: int = 8,
        enable_async_queue: bool = False,
        async_queue_maxsize: int = 0,
    ) -> "StreamingDiarizerOnnxService":
        artifact = load_diarization_artifact_manifest(manifest_path)
        config_path = str(artifact.config)
        return cls(
            modal_ckpt_path=str(artifact.sortformer.onnx),
            preprocessor_ckpt_path=str(artifact.preprocessor.onnx),
            device=device,
            encoder_config=load_encoder_modules_config(config_path),
            sortformer_config=load_sortformer_modules_config(config_path),
            preprocessor_config=load_preprocessor_config(config_path),
            post_processing_config=post_processing_config,
            frame_len_in_secs=frame_len_in_secs,
            sample_rate=sample_rate,
            left_offset=left_offset,
            right_offset=right_offset,
            enable_async_queue=enable_async_queue,
            async_queue_maxsize=async_queue_maxsize,
        )

    def __init__(
        self,
        modal_ckpt_path: str,
        preprocessor_ckpt_path: str,
        device: Literal["cpu", "cuda"],
        encoder_config: EncoderModuleConfig,
        sortformer_config: SortformerModuleConfig,
        preprocessor_config: PreProcessorConfig,
        post_processing_config: PostProcessingParams = PostProcessingParams(),
        frame_len_in_secs: float = 0.08,
        sample_rate: int = 16000,
        left_offset: int = 8,
        right_offset: int = 8,
        enable_async_queue: bool = False,
        async_queue_maxsize: int = 0,
    ):
        self.frame_len_in_secs = frame_len_in_secs
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.chunk_size = sortformer_config.chunk_len
        self.device = torch.device(device)
        self.encoder_config = encoder_config
        self.sortformer_config = sortformer_config
        self.post_processing_config = post_processing_config

        self._diarizer = SortformerONNXRunner(
            onnx_path=modal_ckpt_path,
            device=device,
            sortformer_config=sortformer_config,
        )

        preprocessor = AudioToMelSpectrogramPreprocessorOnnxRunner(
            onnx_path=preprocessor_ckpt_path, device=device
        )

        self.buffer_size_in_secs = (
            sortformer_config.chunk_len * self.frame_len_in_secs
            + (self.left_offset + self.right_offset) * 0.01
        )

        self._feature_bufferer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=sortformer_config.chunk_len * self.frame_len_in_secs,
            preprocessor_cfg=preprocessor_config,
            preprocessor=preprocessor,
            device=self.device,
            left_context_in_secs=self.left_offset * 0.01,
            right_context_in_secs=self.right_offset * 0.01,
        )

        self.streaming_state = self._diarizer.init_streaming_state(
            batch_size=1, device=self.device
        )

        self._post_diar_processor = StreamingDiarizationPostProcessor(
            cfg_vad_params=self.post_processing_config,
            num_spks=sortformer_config.num_spks,
            unit_10ms_frame_count=int(encoder_config.subsampling_factor),
            processing_mode="incremental",
        )
        self._init_event_queues(
            enable_async_queue=enable_async_queue,
            async_queue_maxsize=async_queue_maxsize,
        )

    def _init_event_queues(
        self, enable_async_queue: bool = False, async_queue_maxsize: int = 0
    ):
        self._event_queue = deque()
        self._async_event_queue = (
            asyncio.Queue(maxsize=async_queue_maxsize) if enable_async_queue else None
        )
        self._next_sequence_id = 0

    def diarize(
        self, audio: bytes, stream_id: str = "default"
    ) -> list[StreamingDiarizationEvent]:
        r"""
        Main entrypoint to be call from websocket endpoint
        or processing each chunk audio from a stream
        """
        if len(audio) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM input must contain aligned 16-bit samples")
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return self.process_samples(audio_array, stream_id=stream_id)

    def process_samples(
        self, audio_array: np.ndarray, stream_id: str = "default"
    ) -> list[StreamingDiarizationEvent]:
        self._feature_bufferer.update(audio_array)

        events = []
        while (
            feature_chunk := self._feature_bufferer.pop_ready_feature_chunk()
        ) is not None:
            chunk_outputs = self._process_feature_chunk(feature_chunk)
            events.extend(self._enqueue_diarization_outputs(chunk_outputs, stream_id))

        return events

    def flush(self, stream_id: str = "default"):
        events = []
        for feature_chunk in self._feature_bufferer.flush_ready_feature_chunks():
            chunk_outputs = self._process_feature_chunk(feature_chunk)
            events.extend(self._enqueue_diarization_outputs(chunk_outputs, stream_id))

        tail_outputs = self._post_diar_processor.flush()
        events.extend(self._enqueue_diarization_outputs(tail_outputs, stream_id))
        return events

    def drain_events(self) -> list[StreamingDiarizationEvent]:
        events = list(self._event_queue)
        self._event_queue.clear()
        return events

    async def get_event(self) -> StreamingDiarizationEvent:
        if self._async_event_queue is None:
            raise RuntimeError("Async event queue is not enabled")
        return await self._async_event_queue.get()

    def _enqueue_diarization_outputs(
        self, outputs, stream_id: str
    ) -> list[StreamingDiarizationEvent]:
        events = []
        for spk, segments in enumerate(outputs):
            for start, end in segments:
                event = StreamingDiarizationEvent(
                    stream_id=stream_id,
                    sequence_id=self._next_sequence_id,
                    speaker_id=spk,
                    start=round((start), 2),
                    end=round(float(end), 2),
                )
                self._next_sequence_id += 1
                self._event_queue.append(event)
                if self._async_event_queue is not None:
                    try:
                        self._async_event_queue.put_nowait(event)
                    except asyncio.QueueFull as exc:
                        raise RuntimeError("Async event queue is full") from exc
                events.append(event)
        return events

    def _process_feature_chunk(self, feature_chunk: FeatureBufferChunk):
        features = feature_chunk.features
        feature_buffers = features.unsqueeze(0)  # add batch dimension
        feature_buffers = feature_buffers.transpose(
            1, 2
        )  # [batch, feature, time] -> [batch, time, feature]
        feature_buffer_lens = torch.tensor(
            [feature_buffers.shape[1]], device=self.device
        )

        (spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths) = (
            self._diarizer(
                chunk=feature_buffers,
                chunk_lengths=feature_buffer_lens,
                spkcache=self.streaming_state.spkcache,
                spkcache_lengths=self.streaming_state.spkcache_lengths,
                fifo=self.streaming_state.fifo,
                fifo_lengths=self.streaming_state.fifo_lengths,
            )
        )

        (self.streaming_state, chunk_preds) = self._diarizer.streaming_update_async(
            streaming_state=self.streaming_state,
            chunk=chunk_pre_encode_embs,
            chunk_lengths=chunk_pre_encode_lengths,
            preds=spkcache_fifo_chunk_preds,
            lc=round(
                feature_chunk.left_offset / self.encoder_config.subsampling_factor
            ),
            rc=math.ceil(
                feature_chunk.right_offset / self.encoder_config.subsampling_factor
            ),
        )

        return self._post_diar_processor.process_chunk(chunk_preds[0])


class StreamingDiarizationASROnnxService:
    """Fan one decoded PCM stream into independent diarization and ASR branches."""

    def __init__(
        self,
        diarization_service: StreamingDiarizerOnnxService,
        asr_session: StreamingASRSession,
        alignment_mode: AlignmentMode = "diarization_timeline",
        merger: StreamingPipelineEventMerger | None = None,
    ) -> None:
        self.diarization_service = diarization_service
        self.asr_session = asr_session
        self._merger = merger or StreamingPipelineEventMerger(
            alignment_mode=alignment_mode
        )
        self._stream_id: str | None = None
        self._flushed: bool = False

    @classmethod
    def from_manifests(
        cls,
        diarization_manifest_path: str | Path,
        asr_manifest_path: str | Path | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        target_language: str = "vi-VN",
        post_processing_config: PostProcessingParams = PostProcessingParams(),
        frame_len_in_secs: float = 0.08,
        sample_rate: int = 16000,
        left_offset: int = 8,
        right_offset: int = 8,
        enable_async_queue: bool = False,
        async_queue_maxsize: int = 0,
        alignment_mode: AlignmentMode = "diarization_timeline",
        asr_backend: Literal["nemotron_onnx", "qwen3_modal"] = "nemotron_onnx",
        qwen3_asr_config=None,
        qwen3_asr_remote_actor=None,
    ) -> "StreamingDiarizationASROnnxService":
        if asr_backend not in ("nemotron_onnx", "qwen3_modal"):
            raise ValueError("asr_backend must be 'nemotron_onnx' or 'qwen3_modal'")
        if asr_backend == "nemotron_onnx":
            if asr_manifest_path is None:
                raise ValueError(
                    "asr_manifest_path is required for nemotron_onnx ASR backend"
                )
            diarization_artifact = load_diarization_artifact_manifest(
                diarization_manifest_path
            )
            asr_artifact = load_asr_artifact_manifest(asr_manifest_path)
            if diarization_artifact.preprocessor.onnx == asr_artifact.preprocessor.onnx:
                raise ValueError(
                    "ASR and diarization require independent preprocessors; "
                    "their manifests resolve to the same ONNX file"
                )

        diarization_service = StreamingDiarizerOnnxService.from_manifest(
            diarization_manifest_path,
            device=device,
            post_processing_config=post_processing_config,
            frame_len_in_secs=frame_len_in_secs,
            sample_rate=sample_rate,
            left_offset=left_offset,
            right_offset=right_offset,
            enable_async_queue=enable_async_queue,
            async_queue_maxsize=async_queue_maxsize,
        )
        if asr_backend == "nemotron_onnx":
            asr_session = create_nemotron_streaming_session_from_manifest(
                asr_manifest_path,
                device=device,
                target_language=target_language,
            )
        else:
            asr_session = create_qwen3_asr_modal_session(
                config=qwen3_asr_config,
                remote_actor=qwen3_asr_remote_actor,
            )
        return cls(
            diarization_service=diarization_service,
            asr_session=asr_session,
            alignment_mode=alignment_mode,
        )

    def process(
        self, audio: bytes, stream_id: str = "default"
    ) -> StreamingPipelineResult:
        self._validate_stream(stream_id)
        if self._flushed:
            raise RuntimeError("Cannot process audio after pipeline flush")
        if len(audio) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM input must contain aligned 16-bit samples")
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        diarization_events = self.diarization_service.process_samples(
            samples, stream_id=stream_id
        )
        asr_events = self.asr_session.process_samples(samples, stream_id=stream_id)
        merged_segments = self._merger.consume(
            diarization_events=diarization_events,
            asr_events=asr_events,
        )
        return StreamingPipelineResult(
            diarization_events=tuple(diarization_events),
            asr_events=tuple(asr_events),
            merged_segments=merged_segments,
        )

    def flush(self, stream_id: str = "default") -> StreamingPipelineResult:
        self._validate_stream(stream_id)
        if self._flushed:
            return StreamingPipelineResult((), (), ())
        self._flushed = True
        diarization_events = tuple(self.diarization_service.flush(stream_id=stream_id))
        asr_events = tuple(self.asr_session.flush(stream_id=stream_id))
        ready_segments = self._merger.consume(
            diarization_events=diarization_events,
            asr_events=asr_events,
        )
        remaining_segments = self._merger.flush()
        return StreamingPipelineResult(
            diarization_events=diarization_events,
            asr_events=asr_events,
            merged_segments=ready_segments + remaining_segments,
        )

    def _validate_stream(self, stream_id: str) -> None:
        if self._stream_id is None:
            self._stream_id = stream_id
        elif self._stream_id != stream_id:
            raise ValueError(
                "StreamingDiarizationASROnnxService supports one active stream per instance"
            )
