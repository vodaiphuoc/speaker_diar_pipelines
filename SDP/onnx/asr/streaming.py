from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal, Protocol, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from SDP.onnx.asr.types import EncoderConfig
from SDP.onnx.base import BaseOnnxRunner


class AudioPreprocessor(Protocol):
    def process(
        self, input_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class DecoderJoint(Protocol):
    def initial_states(self, batch_size: int) -> tuple[np.ndarray, ...]: ...

    def __call__(
        self,
        encoder_frame: np.ndarray,
        target: np.ndarray,
        target_length: np.ndarray,
        states: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]: ...


class StreamingEncoder(Protocol):
    def initial_cache_state(self, batch_size: int) -> tuple[np.ndarray, ...]: ...

    def __call__(
        self,
        features: torch.Tensor,
        length: torch.Tensor,
        cache: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]: ...


class PromptProjection(Protocol):
    def __call__(self, encoded: np.ndarray, prompt_index: int) -> np.ndarray: ...


class Tokenizer(Protocol):
    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class ASRFeatureChunk:
    features: torch.Tensor
    length: torch.Tensor
    frame_offset: int
    valid_current_frames: int


class StreamingASRFeatureBuffer:
    """Build fixed-size cache-aware encoder inputs from incrementally received PCM."""

    def __init__(
        self,
        preprocessor: AudioPreprocessor,
        feature_dim: int = 128,
        sample_rate: int = 16000,
        window_stride: float = 0.01,
        n_fft: int = 512,
        current_frames: int = 16,
        history_frames: int = 9,
        subsampling_factor: int = 8,
    ):
        self.preprocessor = preprocessor
        self.feature_dim = feature_dim
        self.hop_samples = int(round(sample_rate * window_stride))
        self.n_fft = n_fft
        self.current_frames = current_frames
        self.history_frames = history_frames
        self.subsampling_factor = subsampling_factor

        self._audio = np.empty(0, dtype=np.float32)
        self._audio_start_sample = 0
        self._received_samples = 0
        self._pending = torch.empty((feature_dim, 0), dtype=torch.float32)
        self._history = torch.zeros(
            (feature_dim, history_frames), dtype=torch.float32
        )
        self._extracted_frames = 0
        self._consumed_frames = 0
        self._flushed = False

    def update(self, audio: np.ndarray) -> None:
        if self._flushed:
            raise RuntimeError("Cannot update a flushed ASR feature buffer")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            self._audio = np.concatenate((self._audio, audio))
            self._received_samples += audio.size

    @property
    def retained_sample_count(self) -> int:
        return self._audio.size

    def pop_ready_chunk(self) -> ASRFeatureChunk | None:
        self._extract_available(flush=False)
        if self._pending.shape[-1] < self.current_frames:
            return None
        return self._pop_chunk(self.current_frames)

    def flush(self) -> list[ASRFeatureChunk]:
        if self._flushed:
            return []
        self._flushed = True
        self._extract_available(flush=True)

        chunks = []
        while self._pending.shape[-1] >= self.current_frames:
            chunks.append(self._pop_chunk(self.current_frames))
        if self._pending.shape[-1]:
            chunks.append(self._pop_chunk(self._pending.shape[-1]))
        return chunks

    def _available_frame_count(self, flush: bool) -> int:
        sample_count = self._received_samples
        if flush:
            return sample_count // self.hop_samples
        right_context = self.n_fft // 2
        if sample_count < right_context:
            return 0
        return (sample_count - right_context) // self.hop_samples + 1

    def _extract_available(self, flush: bool) -> None:
        available = self._available_frame_count(flush)
        if available <= self._extracted_frames:
            return

        context_frames = max(1, (self.n_fft // 2) // self.hop_samples + 1)
        segment_start_frame = max(0, self._extracted_frames - context_frames)
        segment_start_sample = segment_start_frame * self.hop_samples
        local_start_sample = segment_start_sample - self._audio_start_sample
        if local_start_sample < 0:
            raise RuntimeError("ASR waveform context was discarded too early")

        signal = torch.from_numpy(self._audio[local_start_sample:].copy()).unsqueeze(0)
        length = torch.tensor([signal.shape[-1]], dtype=torch.int64)
        result = (
            self.preprocessor.process(signal, length)
            if hasattr(self.preprocessor, "process")
            else self.preprocessor(signal, length)
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("ASR preprocessor must return features and feature lengths")
        processed, processed_length = result
        local_first_frame = self._extracted_frames - segment_start_frame
        requested_count = available - self._extracted_frames
        local_valid_length = min(int(processed_length[0]), processed.shape[-1])
        local_end_frame = min(
            local_first_frame + requested_count, local_valid_length
        )
        if local_end_frame <= local_first_frame:
            return

        new_features = processed[0, :, local_first_frame:local_end_frame].detach().cpu()
        if new_features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} ASR features, got {new_features.shape[0]}"
            )
        self._pending = torch.cat((self._pending, new_features), dim=-1)
        self._extracted_frames += new_features.shape[-1]
        self._trim_audio(context_frames)

    def _trim_audio(self, context_frames: int) -> None:
        keep_from_frame = max(0, self._extracted_frames - context_frames)
        keep_from_sample = keep_from_frame * self.hop_samples
        trim_count = keep_from_sample - self._audio_start_sample
        if trim_count > 0:
            self._audio = self._audio[trim_count:]
            self._audio_start_sample = keep_from_sample

    def _pop_chunk(self, valid_current_frames: int) -> ASRFeatureChunk:
        current = self._pending[:, :valid_current_frames]
        self._pending = self._pending[:, valid_current_frames:]
        if valid_current_frames < self.current_frames:
            current = torch.nn.functional.pad(
                current, (0, self.current_frames - valid_current_frames)
            )

        features = torch.cat((self._history, current), dim=-1).unsqueeze(0)
        chunk = ASRFeatureChunk(
            features=features,
            length=torch.tensor(
                [self.history_frames + valid_current_frames], dtype=torch.int64
            ),
            frame_offset=self._consumed_frames // self.subsampling_factor,
            valid_current_frames=valid_current_frames,
        )

        history_source = torch.cat(
            (self._history, current[:, :valid_current_frames]), dim=-1
        )
        self._history = history_source[:, -self.history_frames :].clone()
        self._consumed_frames += valid_current_frames
        return chunk


@dataclass
class RNNTDecoderState:
    states: tuple[np.ndarray, ...]
    last_label: int

    @classmethod
    def create(
        cls, states: Sequence[np.ndarray], blank_id: int
    ) -> "RNNTDecoderState":
        return cls(states=tuple(np.asarray(state) for state in states), last_label=blank_id)


class StatefulGreedyRNNTDecoder:
    """Greedy RNNT decoding that preserves prediction-network state across chunks."""

    def __init__(
        self,
        decoder_joint: DecoderJoint,
        blank_id: int,
        max_symbols_per_step: int = 10,
    ):
        self.decoder_joint = decoder_joint
        self.blank_id = blank_id
        self.max_symbols_per_step = max_symbols_per_step

    def decode(
        self,
        encoded: np.ndarray,
        encoded_length: int,
        state: RNNTDecoderState,
        frame_offset: int,
    ) -> tuple[list[int], list[int]]:
        token_ids: list[int] = []
        token_frames: list[int] = []
        frame_count = min(int(encoded_length), encoded.shape[-1])

        for time_idx in range(frame_count):
            encoder_frame = encoded[:, :, time_idx : time_idx + 1]
            for _ in range(self.max_symbols_per_step):
                target = np.array([[state.last_label]], dtype=np.int32)
                target_length = np.ones((encoded.shape[0],), dtype=np.int32)
                logits, candidate_states = self.decoder_joint(
                    encoder_frame,
                    target,
                    target_length,
                    state.states,
                )
                prediction = int(np.argmax(logits[0, 0, 0]))
                if prediction == self.blank_id:
                    break

                state.last_label = prediction
                state.states = tuple(candidate_states)
                token_ids.append(prediction)
                token_frames.append(frame_offset + time_idx)

        return token_ids, token_frames


def _as_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


class ASREncoderONNXRunner(BaseOnnxRunner):
    def __init__(
        self,
        onnx_path: str,
        device: str,
        encoder_config: EncoderConfig | None = None,
    ):
        self.encoder_config = encoder_config or EncoderConfig()
        super().__init__(onnx_path=onnx_path, device=device)

    @property
    def input_names(self) -> list[str]:
        return [
            "audio_signal",
            "length",
            "cache_last_channel",
            "cache_last_time",
            "cache_last_channel_len",
        ]

    @property
    def output_names(self) -> list[str]:
        return [
            "outputs",
            "encoded_lengths",
            "cache_last_channel_next",
            "cache_last_time_next",
            "cache_last_channel_next_len",
        ]

    def initial_cache_state(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.encoder_config
        left_context = cfg.att_context_size[0]
        return (
            np.zeros(
                (batch_size, cfg.n_layers, left_context, cfg.d_model),
                dtype=np.float32,
            ),
            np.zeros(
                (
                    batch_size,
                    cfg.n_layers,
                    cfg.d_model,
                    cfg.conv_context_size[0],
                ),
                dtype=np.float32,
            ),
            np.zeros((batch_size,), dtype=np.int64),
        )

    def __call__(
        self,
        features: torch.Tensor,
        length: torch.Tensor,
        cache: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        channel, time, channel_length = cache
        outputs = self.session.run(
            self.output_names,
            {
                "audio_signal": _as_numpy(features, np.float32),
                "length": _as_numpy(length, np.int64),
                "cache_last_channel": _as_numpy(channel, np.float32),
                "cache_last_time": _as_numpy(time, np.float32),
                "cache_last_channel_len": _as_numpy(channel_length, np.int64),
            },
        )
        return outputs[0], outputs[1], tuple(outputs[2:])


class ASRPromptProjectionONNXRunner(BaseOnnxRunner):
    @property
    def input_names(self) -> list[str]:
        return ["encoded", "prompt_index"]

    @property
    def output_names(self) -> list[str]:
        return ["outputs"]

    def __call__(self, encoded: np.ndarray, prompt_index: int) -> np.ndarray:
        batch_size = encoded.shape[0]
        (output,) = self.session.run(
            self.output_names,
            {
                "encoded": _as_numpy(encoded, np.float32),
                "prompt_index": np.full(
                    (batch_size,), prompt_index, dtype=np.int64
                ),
            },
        )
        return output


class ASRDecoderJointONNXRunner(BaseOnnxRunner):
    def __init__(
        self,
        onnx_path: str,
        device: str,
        pred_rnn_layers: int = 2,
        pred_hidden: int = 640,
    ):
        self.pred_rnn_layers = pred_rnn_layers
        self.pred_hidden = pred_hidden
        super().__init__(onnx_path=onnx_path, device=device)

    @property
    def input_names(self) -> list[str]:
        return [
            "encoder_outputs",
            "targets",
            "target_length",
            "input_states_1",
            "input_states_2",
        ]

    @property
    def output_names(self) -> list[str]:
        return [
            "outputs",
            "prednet_lengths",
            "output_states_1",
            "output_states_2",
        ]

    def initial_states(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        shape = (self.pred_rnn_layers, batch_size, self.pred_hidden)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def __call__(
        self,
        encoder_frame: np.ndarray,
        target: np.ndarray,
        target_length: np.ndarray,
        states: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if len(states) != 2:
            raise ValueError(f"RNNT decoder expects two LSTM states, got {len(states)}")
        outputs = self.session.run(
            self.output_names,
            {
                "encoder_outputs": _as_numpy(encoder_frame, np.float32),
                "targets": _as_numpy(target, np.int32),
                "target_length": _as_numpy(target_length, np.int32),
                "input_states_1": _as_numpy(states[0], np.float32),
                "input_states_2": _as_numpy(states[1], np.float32),
            },
        )
        return outputs[0], (outputs[2], outputs[3])


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is required to decode Nemotron token IDs"
            ) from exc
        self._processor = spm.SentencePieceProcessor(model_file=model_path)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._processor.decode([int(token) for token in token_ids])

    @property
    def vocabulary_size(self) -> int:
        return int(self._processor.vocab_size())


@dataclass(frozen=True)
class ASRRuntimeConfig:
    prompt_index: int
    blank_id: int
    feature_dim: int
    sample_rate: int
    window_stride: float
    n_fft: int


@dataclass(frozen=True)
class ASRModelPaths:
    preprocessor: str
    encoder: str
    prompt_projection: str
    decoder_joint: str
    tokenizer: str


def load_asr_runtime_config(
    config_path: str, target_language: str = "vi-VN"
) -> ASRRuntimeConfig:
    config = OmegaConf.load(config_path)
    prompt_dictionary = OmegaConf.to_container(
        config.model_defaults.prompt_dictionary, resolve=True
    )
    if target_language not in prompt_dictionary:
        raise ValueError(
            f"Unsupported ASR language {target_language!r}; "
            f"available keys include {list(prompt_dictionary)[:10]}"
        )
    return ASRRuntimeConfig(
        prompt_index=int(prompt_dictionary[target_language]),
        blank_id=int(config.decoder.vocab_size),
        feature_dim=int(config.preprocessor.features),
        sample_rate=int(config.preprocessor.sample_rate),
        window_stride=float(config.preprocessor.window_stride),
        n_fft=int(config.preprocessor.n_fft),
    )


@dataclass(frozen=True)
class StreamingASREvent:
    stream_id: str
    sequence_id: int
    token_ids: tuple[int, ...]
    text_delta: str
    full_text: str
    token_times: tuple[tuple[float, float], ...]
    start: float | None
    end: float | None
    is_final: bool
    event_type: Literal["asr"] = "asr"


class StreamingASRSession:
    """One-stream state owner for PCM preprocessing, cached encoding and RNNT decoding."""

    def __init__(
        self,
        feature_buffer: StreamingASRFeatureBuffer,
        encoder: StreamingEncoder,
        prompt_projection: PromptProjection,
        decoder_joint: DecoderJoint,
        tokenizer: Tokenizer,
        blank_id: int,
        prompt_index: int,
        frame_duration: float = 0.08,
        max_symbols_per_step: int = 10,
        strip_language_tags: bool = True,
    ):
        self.feature_buffer = feature_buffer
        self.encoder = encoder
        self.prompt_projection = prompt_projection
        self.tokenizer = tokenizer
        self.prompt_index = prompt_index
        self.frame_duration = frame_duration
        self.strip_language_tags = strip_language_tags

        self._decoder = StatefulGreedyRNNTDecoder(
            decoder_joint=decoder_joint,
            blank_id=blank_id,
            max_symbols_per_step=max_symbols_per_step,
        )
        self._encoder_cache = tuple(encoder.initial_cache_state(batch_size=1))
        self._decoder_state = RNNTDecoderState.create(
            decoder_joint.initial_states(batch_size=1), blank_id=blank_id
        )
        self._token_ids: list[int] = []
        self._token_frames: list[int] = []
        self._full_text = ""
        self._next_sequence_id = 0
        self._stream_id: str | None = None
        self._flushed = False

    @property
    def full_text(self) -> str:
        return self._full_text

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self._token_ids)

    def process_pcm(
        self, audio: bytes, stream_id: str = "default"
    ) -> list[StreamingASREvent]:
        if len(audio) % np.dtype(np.int16).itemsize:
            raise ValueError("PCM input must contain aligned 16-bit samples")
        samples = (
            np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return self.process_samples(samples, stream_id=stream_id)

    def process_samples(
        self, samples: np.ndarray, stream_id: str = "default"
    ) -> list[StreamingASREvent]:
        self._validate_stream(stream_id)
        if self._flushed:
            raise RuntimeError("Cannot process audio after ASR flush")
        self.feature_buffer.update(np.asarray(samples, dtype=np.float32))

        events = []
        while (chunk := self.feature_buffer.pop_ready_chunk()) is not None:
            event = self._process_feature_chunk(chunk, stream_id)
            if event is not None:
                events.append(event)
        return events

    def flush(self, stream_id: str = "default") -> list[StreamingASREvent]:
        self._validate_stream(stream_id)
        if self._flushed:
            return []
        self._flushed = True

        events = []
        for chunk in self.feature_buffer.flush():
            event = self._process_feature_chunk(chunk, stream_id)
            if event is not None:
                events.append(event)
        events.append(self._make_event((), (), stream_id=stream_id, is_final=True))
        return events

    def _validate_stream(self, stream_id: str) -> None:
        if self._stream_id is None:
            self._stream_id = stream_id
        elif stream_id != self._stream_id:
            raise ValueError("StreamingASRSession supports one active stream per instance")

    def _process_feature_chunk(
        self, chunk: ASRFeatureChunk, stream_id: str
    ) -> StreamingASREvent | None:
        encoded, encoded_length, self._encoder_cache = self.encoder(
            chunk.features, chunk.length, self._encoder_cache
        )
        encoded = self.prompt_projection(encoded, self.prompt_index)
        token_ids, token_frames = self._decoder.decode(
            encoded=encoded,
            encoded_length=int(encoded_length[0]),
            state=self._decoder_state,
            frame_offset=chunk.frame_offset,
        )
        if not token_ids:
            return None

        self._token_ids.extend(token_ids)
        self._token_frames.extend(token_frames)
        return self._make_event(
            token_ids, token_frames, stream_id=stream_id, is_final=False
        )

    def _decode_full_text(self) -> str:
        text = self.tokenizer.decode(self._token_ids)
        if self.strip_language_tags:
            text = re.sub(r"\s*<[a-z]{2}-[A-Z]{2}>", "", text)
        return text

    def _make_event(
        self,
        token_ids: Sequence[int],
        token_frames: Sequence[int],
        stream_id: str,
        is_final: bool,
    ) -> StreamingASREvent:
        full_text = self._decode_full_text()
        if full_text.startswith(self._full_text):
            text_delta = full_text[len(self._full_text) :]
        else:
            text_delta = full_text
        self._full_text = full_text

        token_times = tuple(
            (
                round(frame * self.frame_duration, 2),
                round((frame + 1) * self.frame_duration, 2),
            )
            for frame in token_frames
        )
        event = StreamingASREvent(
            stream_id=stream_id,
            sequence_id=self._next_sequence_id,
            token_ids=tuple(int(token) for token in token_ids),
            text_delta=text_delta,
            full_text=full_text,
            token_times=token_times,
            start=token_times[0][0] if token_times else None,
            end=token_times[-1][1] if token_times else None,
            is_final=is_final,
        )
        self._next_sequence_id += 1
        return event


def create_nemotron_streaming_session(
    model_paths: ASRModelPaths,
    config_path: str = "configs/asr_pretrained_config.yaml",
    device: Literal["cpu", "cuda"] = "cpu",
    target_language: str = "vi-VN",
) -> StreamingASRSession:
    from SDP.onnx.preprocess.audio_preprocessing import (
        AudioToMelSpectrogramPreprocessorOnnxRunner,
    )

    missing = [
        path
        for path in (
            model_paths.preprocessor,
            model_paths.encoder,
            model_paths.prompt_projection,
            model_paths.decoder_joint,
            model_paths.tokenizer,
        )
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing ASR runtime assets: {missing}")

    runtime_config = load_asr_runtime_config(config_path, target_language)
    tokenizer = SentencePieceTokenizer(model_paths.tokenizer)
    if tokenizer.vocabulary_size != runtime_config.blank_id:
        raise ValueError(
            "Tokenizer vocabulary size must equal the RNNT blank index: "
            f"{tokenizer.vocabulary_size} != {runtime_config.blank_id}"
        )

    preprocessor = AudioToMelSpectrogramPreprocessorOnnxRunner(
        onnx_path=model_paths.preprocessor,
        device=device,
    )
    feature_buffer = StreamingASRFeatureBuffer(
        preprocessor=preprocessor,
        feature_dim=runtime_config.feature_dim,
        sample_rate=runtime_config.sample_rate,
        window_stride=runtime_config.window_stride,
        n_fft=runtime_config.n_fft,
    )
    encoder = ASREncoderONNXRunner(
        onnx_path=model_paths.encoder,
        device=device,
        encoder_config=EncoderConfig(
            feat_in=runtime_config.feature_dim,
            att_context_size=[56, 1],
        ),
    )
    prompt_projection = ASRPromptProjectionONNXRunner(
        onnx_path=model_paths.prompt_projection,
        device=device,
    )
    decoder_joint = ASRDecoderJointONNXRunner(
        onnx_path=model_paths.decoder_joint,
        device=device,
    )
    return StreamingASRSession(
        feature_buffer=feature_buffer,
        encoder=encoder,
        prompt_projection=prompt_projection,
        decoder_joint=decoder_joint,
        tokenizer=tokenizer,
        blank_id=runtime_config.blank_id,
        prompt_index=runtime_config.prompt_index,
    )


def create_nemotron_streaming_session_from_manifest(
    manifest_path: str | Path,
    device: Literal["cpu", "cuda"] = "cpu",
    target_language: str = "vi-VN",
) -> StreamingASRSession:
    from SDP.onnx.artifacts import load_asr_artifact_manifest

    artifact = load_asr_artifact_manifest(manifest_path)
    return create_nemotron_streaming_session(
        ASRModelPaths(
            preprocessor=str(artifact.preprocessor.onnx),
            encoder=str(artifact.encoder.onnx),
            prompt_projection=str(artifact.prompt_projection.onnx),
            decoder_joint=str(artifact.decoder_joint.onnx),
            tokenizer=str(artifact.tokenizer),
        ),
        config_path=str(artifact.config),
        device=device,
        target_language=target_language,
    )
