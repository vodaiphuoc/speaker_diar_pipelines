from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from SDP.onnx.asr import StreamingASREvent


@dataclass(frozen=True)
class Qwen3ASRModalConfig:
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    app_name: str = "sdp-qwen3-asr-streaming"
    volume_name: str = "speaker_diar_qwen3_asr_streaming_cache"
    gpu: str = "A10G"
    sample_rate: int = 16000
    step_ms: int = 1000
    gpu_memory_utilization: float = 0.8
    max_new_tokens: int = 32
    unfixed_chunk_num: int = 2
    unfixed_token_num: int = 5
    chunk_size_sec: float = 2.0

    @classmethod
    def from_env(cls) -> "Qwen3ASRModalConfig":
        return cls(
            model_name=os.environ.get("SDP_QWEN3_ASR_MODEL", cls.model_name),
            app_name=os.environ.get("SDP_QWEN3_ASR_APP", cls.app_name),
            volume_name=os.environ.get("SDP_QWEN3_ASR_VOLUME", cls.volume_name),
            gpu=os.environ.get("SDP_QWEN3_ASR_GPU", cls.gpu),
            step_ms=int(os.environ.get("SDP_QWEN3_ASR_STEP_MS", cls.step_ms)),
            max_new_tokens=int(
                os.environ.get("SDP_QWEN3_ASR_MAX_NEW_TOKENS", cls.max_new_tokens)
            ),
        )


class Qwen3ASRRemoteActor(Protocol):
    def transcribe(self, samples: np.ndarray) -> dict:
        ...

    def finish(self) -> dict:
        ...


class Qwen3ASRModalSession:
    """StreamingASRSession-compatible Qwen3-ASR client backed by Modal GPU."""

    def __init__(
        self,
        remote_actor: Qwen3ASRRemoteActor | None = None,
        config: Qwen3ASRModalConfig | None = None,
    ) -> None:
        self.config = config or Qwen3ASRModalConfig.from_env()
        if self.config.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.config.step_ms <= 0:
            raise ValueError("step_ms must be positive")

        self.remote_actor = remote_actor or _create_default_remote_actor(self.config)
        self._step_samples = max(
            1, int(round(self.config.step_ms / 1000.0 * self.config.sample_rate))
        )
        self._buffer = np.zeros((0,), dtype=np.float32)
        self._buffer_start_sample = 0
        self._full_text = ""
        self._next_sequence_id = 0
        self._stream_id: str | None = None
        self._flushed = False

    @property
    def full_text(self) -> str:
        return self._full_text

    def process_pcm(
        self, audio: bytes, stream_id: str = "default"
    ) -> list[StreamingASREvent]:
        if len(audio) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM input must contain aligned 16-bit samples")
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return self.process_samples(samples, stream_id=stream_id)

    def process_samples(
        self, samples: np.ndarray, stream_id: str = "default"
    ) -> list[StreamingASREvent]:
        self._validate_stream(stream_id)
        if self._flushed:
            raise RuntimeError("Cannot process audio after ASR flush")

        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        if incoming.size:
            self._buffer = np.concatenate((self._buffer, incoming))

        events: list[StreamingASREvent] = []
        while self._buffer.shape[0] >= self._step_samples:
            segment = self._buffer[: self._step_samples]
            start_sample = self._buffer_start_sample
            end_sample = start_sample + segment.shape[0]
            self._buffer = self._buffer[self._step_samples :]
            self._buffer_start_sample = end_sample
            result = self.remote_actor.transcribe(segment)
            event = self._make_event_from_result(
                result,
                start_sample=start_sample,
                end_sample=end_sample,
                stream_id=stream_id,
                is_final=False,
                force_emit=False,
            )
            if event is not None:
                events.append(event)
        return events

    def flush(self, stream_id: str = "default") -> list[StreamingASREvent]:
        self._validate_stream(stream_id)
        if self._flushed:
            return []
        self._flushed = True

        events: list[StreamingASREvent] = []
        final_start_sample: int | None = None
        final_end_sample: int | None = None
        if self._buffer.shape[0]:
            segment = self._buffer
            final_start_sample = self._buffer_start_sample
            final_end_sample = final_start_sample + segment.shape[0]
            self._buffer = np.zeros((0,), dtype=np.float32)
            self._buffer_start_sample = final_end_sample
            result = self.remote_actor.transcribe(segment)
            event = self._make_event_from_result(
                result,
                start_sample=final_start_sample,
                end_sample=final_end_sample,
                stream_id=stream_id,
                is_final=False,
                force_emit=False,
            )
            if event is not None:
                events.append(event)

        result = self.remote_actor.finish()
        final_event = self._make_event_from_result(
            result,
            start_sample=final_start_sample,
            end_sample=final_end_sample,
            stream_id=stream_id,
            is_final=True,
            force_emit=True,
        )
        if final_event is not None:
            events.append(final_event)
        return events

    def _validate_stream(self, stream_id: str) -> None:
        if self._stream_id is None:
            self._stream_id = stream_id
        elif stream_id != self._stream_id:
            raise ValueError("Qwen3ASRModalSession supports one active stream per instance")

    def _make_event_from_result(
        self,
        result: object,
        start_sample: int | None,
        end_sample: int | None,
        stream_id: str,
        is_final: bool,
        force_emit: bool,
    ) -> StreamingASREvent | None:
        text = _extract_text(result)
        if text.startswith(self._full_text):
            text_delta = text[len(self._full_text) :]
        else:
            text_delta = text
        if not text_delta and not force_emit:
            return None
        self._full_text = text

        start = (
            round(start_sample / float(self.config.sample_rate), 2)
            if start_sample is not None
            else None
        )
        end = (
            round(end_sample / float(self.config.sample_rate), 2)
            if end_sample is not None
            else None
        )
        event = StreamingASREvent(
            stream_id=stream_id,
            sequence_id=self._next_sequence_id,
            token_ids=(),
            text_delta=text_delta,
            full_text=self._full_text,
            token_times=(),
            start=start,
            end=end,
            is_final=is_final,
        )
        self._next_sequence_id += 1
        return event


def create_qwen3_asr_modal_session(
    config: Qwen3ASRModalConfig | None = None,
    remote_actor: Qwen3ASRRemoteActor | None = None,
) -> Qwen3ASRModalSession:
    return Qwen3ASRModalSession(remote_actor=remote_actor, config=config)


def _extract_text(result: object) -> str:
    if isinstance(result, dict):
        return str(result.get("text") or "")
    return str(getattr(result, "text", "") or "")


def _create_default_remote_actor(config: Qwen3ASRModalConfig) -> Qwen3ASRRemoteActor:
    from SDP.qwen3_asr.modal_runtime import create_qwen3_asr_modal_actor

    return create_qwen3_asr_modal_actor(config)
