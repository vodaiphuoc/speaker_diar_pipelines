from dataclasses import asdict
from typing import Dict, List

import torch


class StreamingDiarizationPostProcessor:
    def __init__(self, cfg_vad_params, num_spks: int, unit_10ms_frame_count: int = 8):
        self.params = asdict(cfg_vad_params)
        self.num_spks = num_spks
        self.unit_10ms_frame_count = unit_10ms_frame_count

        # Extract params
        self.frame_length_in_sec = self.params.get("frame_length_in_sec", 0.01)
        self.onset = self.params.get("onset", 0.5)
        self.offset = self.params.get("offset", 0.5)
        self.pad_onset = self.params.get("pad_onset", 0.0)
        self.pad_offset = self.params.get("pad_offset", 0.0)

        self.min_dur_on = self.params.get("min_duration_on", 0.0)
        self.min_dur_off = self.params.get("min_duration_off", 0.0)

        # 1. STATEFUL VARIABLES (Tracked globally across chunks)
        self.global_frame_idx = 0
        self.is_speech = [False] * num_spks
        self.start_times = [0.0] * num_spks

        # 2. BUFFERS FOR FILTERING
        # Holds segments that are finished but waiting for gap/duration validation
        self.pending_segments = [[] for _ in range(num_spks)]

    def process_chunk(self, ts_vad_chunk: torch.Tensor) -> List[List[List[float]]]:
        """
        Process a streaming chunk of shape (chunk_len, num_spks)
        Returns a list (per speaker) of finalized segments [[start, end], ...]
        """
        # Upsample chunk to match your 10ms frame logic
        ts_vad_chunk = torch.repeat_interleave(
            ts_vad_chunk, self.unit_10ms_frame_count, dim=0
        )
        chunk_frames = ts_vad_chunk.shape[0]

        finalized_output = [[] for _ in range(self.num_spks)]

        # Process binarization per speaker
        for spk in range(self.num_spks):
            sequence = ts_vad_chunk[:, spk]

            for i in range(chunk_frames):
                current_time = (self.global_frame_idx + i) * self.frame_length_in_sec

                if self.is_speech[spk]:
                    # Check for offset (speech ending)
                    if sequence[i] < self.offset:
                        end_time = current_time + self.pad_offset
                        start_time = max(0.0, self.start_times[spk] - self.pad_onset)

                        # Instead of emitting immediately, push to pending buffer for filtering
                        self._add_to_pending(spk, start_time, end_time)
                        self.is_speech[spk] = False
                else:
                    # Check for onset (speech starting)
                    if sequence[i] > self.onset:
                        self.start_times[spk] = current_time
                        self.is_speech[spk] = True

            # Run streaming filter over pending segments for this speaker
            current_global_time = (
                self.global_frame_idx + chunk_frames
            ) * self.frame_length_in_sec
            valid_segments = self._apply_streaming_filters(spk, current_global_time)
            finalized_output[spk].extend(valid_segments)

        # Increment global time
        self.global_frame_idx += chunk_frames

        return finalized_output

    def _add_to_pending(self, spk: int, start: float, end: float):
        """Adds a completed segment to the pending buffer and merges overlaps."""
        if not self.pending_segments[spk]:
            self.pending_segments[spk].append([start, end])
            return

        last_start, last_end = self.pending_segments[spk][-1]

        # Merge overlapping segments caused by padding
        if start <= last_end:
            self.pending_segments[spk][-1][1] = max(last_end, end)
        else:
            self.pending_segments[spk].append([start, end])

    def _apply_streaming_filters(
        self, spk: int, current_time: float
    ) -> List[List[float]]:
        """
        Applies min_duration_off and min_duration_on logic.
        Only emits segments that are "safe" (i.e., we are sure no future chunk will cause a merge).
        """
        safe_segments = []
        pending = self.pending_segments[spk]

        i = 0
        while i < len(pending) - 1:
            curr_seg = pending[i]
            next_seg = pending[i + 1]
            gap = next_seg[0] - curr_seg[1]

            # 1. Merge short gaps (min_duration_off)
            if gap < self.min_dur_off:
                # Merge next_seg into curr_seg
                next_seg[0] = curr_seg[0]
                pending.pop(i)  # remove the current one
                continue

            # 2. Filter short speech (min_duration_on)
            duration = curr_seg[1] - curr_seg[0]
            if duration >= self.min_dur_on:
                safe_segments.append(curr_seg)

            # Move forward
            i += 1

        # Keep the very last segment in the buffer because we don't know
        # if the gap to the next chunk's segment will be smaller than min_dur_off!
        if len(pending) > 0:
            last_seg = pending[-1]
            time_since_last_seg = current_time - last_seg[1]

            # If enough silence has passed, it's safe to evaluate the final segment
            if time_since_last_seg >= self.min_dur_off:
                duration = last_seg[1] - last_seg[0]
                if duration >= self.min_dur_on:
                    safe_segments.append(last_seg)
                self.pending_segments[spk] = []
            else:
                self.pending_segments[spk] = [last_seg]
        else:
            self.pending_segments[spk] = []

        return safe_segments
