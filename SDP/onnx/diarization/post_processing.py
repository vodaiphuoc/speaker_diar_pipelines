from dataclasses import asdict
from typing import List, Optional, Tuple

import torch


class StreamingDiarizationPostProcessor:
    """
    Stateful VAD-style post-processing for streaming diarization outputs.

    `process_chunk` accepts only the model output for the current chunk and returns
    newly finalized segments. `incremental` mode finalizes from a per-frame state
    machine. `buffered_window` mode reprocesses a bounded rolling history and
    emits only segments older than a commit watermark.
    """

    def __init__(
        self,
        cfg_vad_params,
        num_spks: int,
        unit_10ms_frame_count: int = 8,
        frame_length_in_sec: float = 0.01,
        processing_mode: str = "buffered_window",
        buffer_window_sec: float = 30.0,
        commit_delay_sec: Optional[float] = None,
    ):
        if processing_mode not in {"incremental", "buffered_window"}:
            raise ValueError(
                "processing_mode must be either 'incremental' or 'buffered_window'"
            )

        self.params = asdict(cfg_vad_params)
        self.num_spks = num_spks
        self.unit_10ms_frame_count = unit_10ms_frame_count
        self.frame_length_in_sec = self.params.get(
            "frame_length_in_sec", frame_length_in_sec
        )
        self.processing_mode = processing_mode
        self.buffer_window_sec = buffer_window_sec

        self.onset = self.params.get("onset", 0.5)
        self.offset = self.params.get("offset", 0.5)
        self.pad_onset = self.params.get("pad_onset", 0.0)
        self.pad_offset = self.params.get("pad_offset", 0.0)
        self.min_dur_on = self.params.get("min_duration_on", 0.0)
        self.min_dur_off = self.params.get("min_duration_off", 0.0)
        self.commit_delay_sec = (
            max(1.0, self.min_dur_off)
            if commit_delay_sec is None
            else commit_delay_sec
        )
        self.filter_speech_first = bool(self.params.get("filter_speech_first", 1.0))

        self.global_frame_idx = 0
        self.is_speech = [False] * num_spks
        self.start_times = [0.0] * num_spks
        self.pending_segments: List[List[List[float]]] = [
            [] for _ in range(num_spks)
        ]
        self.raw_chunk_buffer: List[Tuple[int, torch.Tensor]] = []
        self.emitted_until_sec = [0.0] * num_spks

    def reset(self):
        self.global_frame_idx = 0
        self.is_speech = [False] * self.num_spks
        self.start_times = [0.0] * self.num_spks
        self.pending_segments = [[] for _ in range(self.num_spks)]
        self.raw_chunk_buffer = []
        self.emitted_until_sec = [0.0] * self.num_spks

    def process_chunk(self, ts_vad_chunk: torch.Tensor) -> List[List[List[float]]]:
        """
        Process current chunk diarization probabilities.

        Args:
            ts_vad_chunk: Tensor with shape `(chunk_len, num_spks)`.

        Returns:
            Per-speaker list of newly finalized `[start, end]` segments.
        """
        ts_vad_chunk = self._validate_chunk(ts_vad_chunk)
        if self.processing_mode == "incremental":
            return self._process_incremental_chunk(ts_vad_chunk)
        return self._process_buffered_window_chunk(ts_vad_chunk)

    def flush(self) -> List[List[List[float]]]:
        """
        Finalize all currently buffered segments.

        Call this only when the stream ends. During regular streaming,
        `process_chunk` intentionally keeps uncertain tail segments buffered.
        """
        if self.processing_mode == "buffered_window":
            flushed_output = self._emit_buffered_segments(float("inf"))
            self.raw_chunk_buffer = []
            return flushed_output

        current_stream_time = self.global_frame_idx * self.frame_length_in_sec
        flushed_output = [[] for _ in range(self.num_spks)]

        for spk in range(self.num_spks):
            if self.is_speech[spk]:
                start_time = self._to_float32_time(
                    max(0.0, self.start_times[spk] - self.pad_onset)
                )
                end_time = self._to_float32_time(
                    current_stream_time + self.pad_offset
                )
                if end_time > start_time:
                    self.pending_segments[spk].append([start_time, end_time])
                self.is_speech[spk] = False

            flushed_output[spk] = self._filter_segments(self.pending_segments[spk])
            self.pending_segments[spk] = []

        return flushed_output

    def _validate_chunk(self, ts_vad_chunk: torch.Tensor) -> torch.Tensor:
        if ts_vad_chunk.ndim != 2:
            raise ValueError(
                f"Expected shape (chunk_len, num_spks), got {tuple(ts_vad_chunk.shape)}"
            )
        if ts_vad_chunk.shape[1] != self.num_spks:
            raise ValueError(
                f"Expected {self.num_spks} speakers, got {ts_vad_chunk.shape[1]}"
            )
        return ts_vad_chunk.detach().cpu()

    def _process_incremental_chunk(
        self, ts_vad_chunk: torch.Tensor
    ) -> List[List[List[float]]]:
        ts_vad_frames = torch.repeat_interleave(
            ts_vad_chunk, self.unit_10ms_frame_count, dim=0
        )
        chunk_frames = ts_vad_frames.shape[0]
        finalized_output = [[] for _ in range(self.num_spks)]

        for spk in range(self.num_spks):
            sequence = ts_vad_frames[:, spk]
            for frame_offset, score in enumerate(sequence):
                current_time = self._to_time(self.global_frame_idx + frame_offset)

                if self.is_speech[spk]:
                    if score < self.offset:
                        end_time = self._to_float32_time(
                            current_time + self.pad_offset
                        )
                        start_time = self._to_float32_time(
                            max(0.0, self.start_times[spk] - self.pad_onset)
                        )
                        if end_time > start_time:
                            self.pending_segments[spk].append([start_time, end_time])
                        self.start_times[spk] = current_time
                        self.is_speech[spk] = False
                elif score > self.onset:
                    self.start_times[spk] = current_time
                    self.is_speech[spk] = True

            current_stream_time = (
                self.global_frame_idx + chunk_frames
            ) * self.frame_length_in_sec
            finalized_output[spk] = self._finalize_safe_segments(
                spk, current_stream_time
            )

        self.global_frame_idx += chunk_frames
        return finalized_output

    def _process_buffered_window_chunk(
        self, ts_vad_chunk: torch.Tensor
    ) -> List[List[List[float]]]:
        chunk_frames = ts_vad_chunk.shape[0] * self.unit_10ms_frame_count
        self.raw_chunk_buffer.append((self.global_frame_idx, ts_vad_chunk))
        self.global_frame_idx += chunk_frames

        current_stream_time = self.global_frame_idx * self.frame_length_in_sec
        emit_before_sec = current_stream_time - self.commit_delay_sec
        finalized_output = self._emit_buffered_segments(emit_before_sec)
        self._prune_raw_chunk_buffer(current_stream_time)
        return finalized_output

    def _emit_buffered_segments(self, emit_before_sec: float) -> List[List[List[float]]]:
        output = [[] for _ in range(self.num_spks)]
        if not self.raw_chunk_buffer:
            return output

        window_start_frame_idx = self.raw_chunk_buffer[0][0]
        window_start_sec = self._to_time(window_start_frame_idx)
        window = torch.cat([chunk for _, chunk in self.raw_chunk_buffer], dim=0)

        for spk in range(self.num_spks):
            segments = self._post_process_sequence(window[:, spk])
            for start, end in segments:
                abs_start = self._to_float32_time(window_start_sec + start)
                abs_end = self._to_float32_time(window_start_sec + end)
                if abs_end >= emit_before_sec - 1e-8:
                    continue
                if abs_end <= self.emitted_until_sec[spk]:
                    continue
                if abs_start < self.emitted_until_sec[spk]:
                    abs_start = self.emitted_until_sec[spk]
                if abs_end > abs_start:
                    output[spk].append([abs_start, abs_end])
                    self.emitted_until_sec[spk] = abs_end

        return output

    def _post_process_sequence(self, sequence: torch.Tensor) -> List[List[float]]:
        frames = torch.repeat_interleave(sequence, self.unit_10ms_frame_count)
        raw_segments = self._binarize_frames(frames)
        return self._filter_segments(raw_segments)

    def _binarize_frames(self, sequence: torch.Tensor) -> List[List[float]]:
        speech = False
        start = 0.0
        last_frame_idx = 0
        segments = []

        for frame_idx, score in enumerate(sequence):
            last_frame_idx = frame_idx
            current_time = self._to_time(frame_idx)
            if speech:
                if score < self.offset:
                    start_time = self._to_float32_time(max(0.0, start - self.pad_onset))
                    end_time = self._to_float32_time(current_time + self.pad_offset)
                    if end_time > start_time:
                        segments.append([start_time, end_time])
                    start = current_time
                    speech = False
            elif score > self.onset:
                start = current_time
                speech = True

        if speech:
            start_time = self._to_float32_time(max(0.0, start - self.pad_onset))
            end_time = self._to_float32_time(
                last_frame_idx * self.frame_length_in_sec + self.pad_offset
            )
            if end_time > start_time:
                segments.append([start_time, end_time])

        return self._merge_overlaps(segments) if segments else []

    def _prune_raw_chunk_buffer(self, current_stream_time: float):
        keep_after_sec = current_stream_time - self.buffer_window_sec
        while len(self.raw_chunk_buffer) > 1:
            start_frame_idx, chunk = self.raw_chunk_buffer[0]
            end_frame_idx = (
                start_frame_idx + chunk.shape[0] * self.unit_10ms_frame_count
            )
            end_sec = end_frame_idx * self.frame_length_in_sec
            if end_sec >= keep_after_sec:
                break
            self.raw_chunk_buffer.pop(0)

    def _finalize_safe_segments(
        self, spk: int, current_stream_time: float
    ) -> List[List[float]]:
        pending = self.pending_segments[spk]
        if not pending:
            return []

        safe_cutoff = current_stream_time - self.min_dur_off
        pending = self._merge_overlaps(sorted(pending, key=lambda segment: segment[0]))
        groups = self._group_by_short_gaps(pending)
        if self.is_speech[spk] and groups:
            active_start = self._to_float32_time(
                max(0.0, self.start_times[spk] - self.pad_onset)
            )
            last_closed_end = groups[-1][-1][1]
            if active_start - last_closed_end < self.min_dur_off:
                safe_cutoff = min(
                    safe_cutoff, last_closed_end - self.frame_length_in_sec
                )

        finalized_groups = []
        remaining_groups = []
        reached_uncertain_tail = False

        for group in groups:
            if not reached_uncertain_tail and group[-1][1] <= safe_cutoff:
                finalized_groups.append(group)
            else:
                reached_uncertain_tail = True
                remaining_groups.append(group)

        finalized_candidates = [
            segment for group in finalized_groups for segment in group
        ]
        self.pending_segments[spk] = [
            segment for group in remaining_groups for segment in group
        ]
        return self._filter_segments(finalized_candidates)

    def _filter_segments(self, segments: List[List[float]]) -> List[List[float]]:
        if not segments:
            return []

        sorted_segments = sorted(segments, key=lambda segment: segment[0])
        merged_segments = self._merge_overlaps(sorted_segments)

        if self.filter_speech_first:
            filtered = self._remove_short_speech(merged_segments)
            return self._merge_short_gaps(filtered)

        gap_merged = self._merge_short_gaps(merged_segments)
        return self._remove_short_speech(gap_merged)

    def _merge_overlaps(self, segments: List[List[float]]) -> List[List[float]]:
        if not segments:
            return []

        merged = [segments[0].copy()]
        for start, end in segments[1:]:
            last = merged[-1]
            if start <= last[1]:
                last[1] = max(last[1], end)
            else:
                merged.append([start, end])
        return merged

    def _remove_short_speech(self, segments: List[List[float]]) -> List[List[float]]:
        if self.min_dur_on <= 0.0:
            return [segment.copy() for segment in segments]
        return [
            segment.copy()
            for segment in segments
            if segment[1] - segment[0] >= self.min_dur_on - 1e-8
        ]

    def _merge_short_gaps(self, segments: List[List[float]]) -> List[List[float]]:
        if self.min_dur_off <= 0.0 or len(segments) <= 1:
            return [segment.copy() for segment in segments]

        merged = [segments[0].copy()]
        for start, end in segments[1:]:
            last = merged[-1]
            if start - last[1] < self.min_dur_off + 1e-8:
                last[1] = max(last[1], end)
            else:
                merged.append([start, end])
        return merged

    def _group_by_short_gaps(
        self, segments: List[List[float]]
    ) -> List[List[List[float]]]:
        if not segments:
            return []
        if self.min_dur_off <= 0.0:
            return [[segment.copy()] for segment in segments]

        groups = [[segments[0].copy()]]
        for segment in segments[1:]:
            previous = groups[-1][-1]
            if segment[0] - previous[1] < self.min_dur_off + 1e-8:
                groups[-1].append(segment.copy())
            else:
                groups.append([segment.copy()])
        return groups

    def _to_time(self, frame_idx: int) -> float:
        return self._to_float32_time(frame_idx * self.frame_length_in_sec)

    def _to_float32_time(self, value: float) -> float:
        return float(torch.tensor(value, dtype=torch.float32).item())
