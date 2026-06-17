import math
from dataclasses import dataclass

import numpy as np
import torch

from SDP.onnx.diarization.types import PreProcessorConfig
from SDP.onnx.preprocess.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorOnnxRunner,
)

LOG_MEL_ZERO = -16.635


@dataclass
class FeatureBufferChunk:
    features: torch.Tensor
    left_offset: int
    right_offset: int
    center_frame_count: int
    center_diar_frame_count: int


class AudioBufferer:
    def __init__(self, sample_rate: int, buffer_size_in_secs: float):
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.sample_buffer = torch.zeros(self.buffer_size, dtype=torch.float32)

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.zero_()

    def update(self, audio: np.ndarray) -> None:
        """
        Update the buffer with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """
        if not isinstance(audio, torch.Tensor):
            audio: torch.Tensor = torch.from_numpy(audio)

        audio_size = audio.shape[0]
        if audio_size > self.buffer_size:
            raise ValueError(
                f"Frame size ({audio_size}) exceeds buffer size ({self.buffer_size})"
            )

        shift = audio_size
        self.sample_buffer[:-shift] = self.sample_buffer[shift:].clone()
        self.sample_buffer[-shift:] = audio.clone()

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current buffer
        Returns:
            torch.Tensor: current state of the buffer
        """
        return self.sample_buffer.clone()

    def is_buffer_empty(self) -> bool:
        """
        Check if the buffer is empty
        Returns:
            bool: True if the buffer is empty, False otherwise
        """
        return self.sample_buffer.sum().item() == 0


class CacheFeatureBufferer:
    def __init__(
        self,
        sample_rate: int,
        buffer_size_in_secs: float,
        chunk_size_in_secs: float,
        preprocessor_cfg: PreProcessorConfig,
        preprocessor: AudioToMelSpectrogramPreprocessorOnnxRunner,
        device: torch.device,
        fill_value: float = LOG_MEL_ZERO,
        left_context_in_secs: float | None = None,
        right_context_in_secs: float | None = None,
    ):

        if buffer_size_in_secs < chunk_size_in_secs:
            raise ValueError(
                f"Buffer size ({buffer_size_in_secs}s) should be no less than chunk size ({chunk_size_in_secs}s)"
            )

        self.sample_rate = sample_rate
        self.buffer_size_in_secs = buffer_size_in_secs
        self.chunk_size_in_secs = chunk_size_in_secs
        self.device = device
        extra_context_secs = max(0.0, buffer_size_in_secs - chunk_size_in_secs)
        if left_context_in_secs is None:
            left_context_in_secs = extra_context_secs / 2
        if right_context_in_secs is None:
            right_context_in_secs = extra_context_secs - left_context_in_secs

        self.left_context_in_secs = left_context_in_secs
        self.right_context_in_secs = right_context_in_secs
        self.left_context_samples = int(round(left_context_in_secs * sample_rate))
        self.right_context_samples = int(round(right_context_in_secs * sample_rate))

        if hasattr(preprocessor_cfg, "log") and preprocessor_cfg.log:
            self.ZERO_LEVEL_SPEC_DB_VAL = (
                LOG_MEL_ZERO  # Log-Mel spectrogram value for zero signals
            )
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = fill_value

        self.n_feat = preprocessor_cfg.features
        self.timestep_duration = preprocessor_cfg.window_stride
        self.n_chunk_look_back = int(self.timestep_duration * self.sample_rate)
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)
        self.audio_buffer = AudioBufferer(sample_rate, buffer_size_in_secs)

        self.feature_buffer_len = int(buffer_size_in_secs / self.timestep_duration)
        self.feature_chunk_len = int(chunk_size_in_secs / self.timestep_duration)
        self.feature_buffer = torch.full(
            [self.n_feat, self.feature_buffer_len],
            self.ZERO_LEVEL_SPEC_DB_VAL,
            dtype=torch.float32,
            device=self.device,
        )
        self.preprocessor = preprocessor
        self.received_audio = torch.empty(0, dtype=torch.float32)
        self.next_chunk_start_sample = 0

    def is_buffer_empty(self) -> bool:
        """
        Check if the buffer is empty
        Returns:
            bool: True if the buffer is empty, False otherwise
        """
        return self.audio_buffer.is_buffer_empty()

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.audio_buffer.reset()
        self.feature_buffer.fill_(self.ZERO_LEVEL_SPEC_DB_VAL)
        self.received_audio = torch.empty(0, dtype=torch.float32)
        self.next_chunk_start_sample = 0

    def _update_feature_buffer(self, feat_chunk: torch.Tensor) -> None:
        """
        Add an extracted feature to `feature_buffer`
        """

        # push `self.feature_chunk_len` columns to the 'left'
        self.feature_buffer[:, : -self.feature_chunk_len] = self.feature_buffer[
            :, self.feature_chunk_len :
        ].clone()
        self.feature_buffer[:, -self.feature_chunk_len :] = feat_chunk.clone()

    def _preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the audio signal using the preprocessor
        Args:
            audio_signal (torch.Tensor): audio signal
        Returns:
            torch.Tensor: preprocessed features
                Shape (128, 1008)
        """
        audio_signal = audio_signal.unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features = self.preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len,
        )
        if features.ndim == 3:
            features = features.squeeze(0)
        elif features.ndim == 1:
            features = features.unsqueeze(0)
        return features

    def update(self, audio: np.ndarray) -> None:
        """
        Update the sample anf feature buffers with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """

        if not isinstance(audio, torch.Tensor):
            audio_tensor = torch.from_numpy(audio).to(torch.float32)
        else:
            audio_tensor = audio.detach().cpu().to(torch.float32)

        self.received_audio = torch.cat([self.received_audio, audio_tensor])

        # Update the sample buffer with the new frame
        rolling_audio = audio_tensor
        if rolling_audio.shape[0] > self.audio_buffer.buffer_size:
            rolling_audio = rolling_audio[-self.audio_buffer.buffer_size :]
        self.audio_buffer.update(rolling_audio)

        if math.isclose(self.buffer_size_in_secs, self.chunk_size_in_secs):
            # If the buffer size is equal to the chunk size, just take the whole buffer
            samples = self.audio_buffer.sample_buffer.clone()
        else:
            # Add look_back to have context for the first feature
            samples = self.audio_buffer.sample_buffer[
                -(self.n_chunk_look_back + self.chunk_size) :
            ]

        # Get the mel spectrogram
        features = self._preprocess(samples)

        # If the features are longer than supposed to be, drop the last frames
        # Drop the last diff frames because they might be incomplete
        if (diff := features.shape[1] - self.feature_chunk_len - 1) > 0:
            features = features[:, :-diff]

        # Update the feature buffer with the new features
        self._update_feature_buffer(features[:, -self.feature_chunk_len :])

    def pop_ready_feature_chunk(self) -> FeatureBufferChunk | None:
        center_end = self.next_chunk_start_sample + self.chunk_size
        required_samples = center_end + self.right_context_samples
        if self.received_audio.shape[0] < required_samples:
            return None

        return self._pop_feature_chunk(finalize=False)

    def flush_ready_feature_chunks(self) -> list[FeatureBufferChunk]:
        chunks = []
        while self.next_chunk_start_sample < self.received_audio.shape[0]:
            chunk = self._pop_feature_chunk(finalize=True)
            if chunk is None:
                break
            chunks.append(chunk)
        return chunks

    def _pop_feature_chunk(self, finalize: bool) -> FeatureBufferChunk | None:
        stream_len = self.received_audio.shape[0]
        center_start = self.next_chunk_start_sample
        if center_start >= stream_len:
            return None

        center_end = center_start + self.chunk_size
        if not finalize and stream_len < center_end + self.right_context_samples:
            return None

        actual_center_end = min(center_end, stream_len) if finalize else center_end
        slice_start = max(0, center_start - self.left_context_samples)
        requested_slice_end = center_end + self.right_context_samples
        slice_end = min(requested_slice_end, stream_len)
        if slice_end <= slice_start:
            return None

        left_offset_samples = center_start - slice_start
        right_offset_samples = max(0, slice_end - actual_center_end)
        center_samples = actual_center_end - center_start
        features = self._preprocess(self.received_audio[slice_start:slice_end])

        chunk = FeatureBufferChunk(
            features=features,
            left_offset=self._samples_to_feature_frames(left_offset_samples),
            right_offset=self._samples_to_feature_frames(right_offset_samples),
            center_frame_count=self._samples_to_feature_frames(center_samples),
            center_diar_frame_count=self._samples_to_feature_frames(center_samples),
        )
        self.next_chunk_start_sample += center_samples
        return chunk

    def _samples_to_feature_frames(self, samples: int) -> int:
        if samples <= 0:
            return 0
        return int(round(samples / (self.timestep_duration * self.sample_rate)))

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current sample buffer
        Returns:
            torch.Tensor: current state of the buffer
        """
        return self.audio_buffer.get_buffer()

    def get_feature_buffer(self) -> torch.Tensor:
        """
        Get the current feature buffer
        Returns:
            torch.Tensor: current state of the feature buffer
        """
        return self.feature_buffer.clone()
