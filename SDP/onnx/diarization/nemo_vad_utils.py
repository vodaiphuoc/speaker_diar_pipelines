from dataclasses import asdict
from typing import Dict

import torch

from SDP.onnx.diarization.types import PostProcessingParams


@torch.jit.script
def filter_short_segments(segments: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Remove segments which duration is smaller than a threshold.
    For example,
    torch.Tensor([[0, 1.5], [1, 3.5], [4, 7]]) and threshold = 2.0
    ->
    torch.Tensor([[1, 3.5], [4, 7]])
    """
    return segments[segments[:, 1] - segments[:, 0] >= threshold]


@torch.jit.script
def get_gap_segments(segments: torch.Tensor) -> torch.Tensor:
    """
    Get the gap segments.
    For example,
    torch.Tensor([[start1, end1], [start2, end2], [start3, end3]]) -> torch.Tensor([[end1, start2], [end2, start3]])
    """
    segments = segments[segments[:, 0].sort()[1]]
    return torch.column_stack((segments[:-1, 1], segments[1:, 0]))


@torch.jit.script
def remove_segments(
    original_segments: torch.Tensor, to_be_removed_segments: torch.Tensor
) -> torch.Tensor:
    """
    Remove speech segments list in to_be_removed_segments from original_segments.
    (Example) Remove torch.Tensor([[start2, end2],[start4, end4]])
              from torch.Tensor([[start1, end1],[start2, end2],[start3, end3], [start4, end4]]),
              ->
              torch.Tensor([[start1, end1],[start3, end3]])
    """
    for y in to_be_removed_segments:
        original_segments = original_segments[
            original_segments.eq(y).all(dim=1).logical_not()
        ]
    return original_segments


@torch.jit.script
def merge_overlap_segment(segments: torch.Tensor) -> torch.Tensor:
    """
    Merged the given overlapped segments.
    For example:
    torch.Tensor([[0, 1.5], [1, 3.5]]) -> torch.Tensor([0, 3.5])
    """
    if (
        segments.shape == torch.Size([0])
        or segments.shape == torch.Size([0, 2])
        or segments.shape == torch.Size([1, 2])
    ):
        return segments

    segments = segments[segments[:, 0].sort()[1]]
    merge_boundary = segments[:-1, 1] >= segments[1:, 0]
    head_padded = torch.nn.functional.pad(
        merge_boundary, [1, 0], mode="constant", value=0.0
    )
    head = segments[~head_padded, 0]
    tail_padded = torch.nn.functional.pad(
        merge_boundary, [0, 1], mode="constant", value=0.0
    )
    tail = segments[~tail_padded, 1]
    merged = torch.stack((head, tail), dim=1)
    return merged


@torch.jit.script
def filtering(
    speech_segments: torch.Tensor, per_args: Dict[str, float]
) -> torch.Tensor:
    """
    Filter out short non-speech and speech segments.

    Reference:
        Paper: Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of RNN-based Voice
        Activity Detection", InterSpeech 2015.
        Implementation: see the equivalent reference implementation in the
        External Annotation Library's audio toolkit (``utils/signal.py``).

    Args:
        speech_segments (torch.Tensor):
            A tensor of speech segments in the format
            torch.Tensor([[start1, end1], [start2, end2]]).
        per_args:
            min_duration_on (float):
                Threshold for short speech segment deletion.
            min_duration_off (float):
                Threshold for small non-speech deletion.
            filter_speech_first (float):
                Whether to perform short speech segment deletion first. Use 1.0 to represent True.

    Returns:
        speech_segments (torch.Tensor):
            A tensor of filtered speech segments in the format
            torch.Tensor([[start1, end1], [start2, end2]]).
    """
    if speech_segments.shape == torch.Size([0]):
        return speech_segments

    min_duration_on = per_args.get("min_duration_on", 0.0)
    min_duration_off = per_args.get("min_duration_off", 0.0)
    filter_speech_first = per_args.get("filter_speech_first", 1.0)

    if filter_speech_first == 1.0:
        # Filter out the shorter speech segments
        if min_duration_on > 0.0:
            speech_segments = filter_short_segments(speech_segments, min_duration_on)
        # Filter out the shorter non-speech segments and return to be as speech segments
        if min_duration_off > 0.0:
            # Find non-speech segments
            non_speech_segments = get_gap_segments(speech_segments)
            # Find shorter non-speech segments
            short_non_speech_segments = remove_segments(
                non_speech_segments,
                filter_short_segments(non_speech_segments, min_duration_off),
            )
            # Return shorter non-speech segments to be as speech segments
            speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)

            # Merge the overlapped speech segments
            speech_segments = merge_overlap_segment(speech_segments)
    else:
        if min_duration_off > 0.0:
            # Find non-speech segments
            non_speech_segments = get_gap_segments(speech_segments)
            # Find shorter non-speech segments
            short_non_speech_segments = remove_segments(
                non_speech_segments,
                filter_short_segments(non_speech_segments, min_duration_off),
            )

            speech_segments = torch.cat((speech_segments, short_non_speech_segments), 0)

            # Merge the overlapped speech segments
            speech_segments = merge_overlap_segment(speech_segments)
        if min_duration_on > 0.0:
            speech_segments = filter_short_segments(speech_segments, min_duration_on)

    return speech_segments


@torch.jit.script
def binarization(sequence: torch.Tensor, per_args: Dict[str, float]) -> torch.Tensor:
    """
    Binarize predictions to speech and non-speech

    Reference
    Paper: Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of RNN-based Voice
           Activity Detection", InterSpeech 2015.
    Implementation: see the equivalent reference implementation in the External
    Annotation Library's audio toolkit (``utils/signal.py``).

    Args:
        sequence (torch.Tensor) : A tensor of frame level predictions.
        per_args:
            onset (float): onset threshold for detecting the beginning and end of a speech
            offset (float): offset threshold for detecting the end of a speech.
            pad_onset (float): adding durations before each speech segment
            pad_offset (float): adding durations after each speech segment;
            frame_length_in_sec (float): length of frame.

    Returns:
        speech_segments(torch.Tensor): A tensor of speech segment in the form of:
                                      `torch.Tensor([[start1, end1], [start2, end2]])`.
    """
    frame_length_in_sec = per_args.get("frame_length_in_sec", 0.01)

    onset = per_args.get("onset", 0.5)
    offset = per_args.get("offset", 0.5)
    pad_onset = per_args.get("pad_onset", 0.0)
    pad_offset = per_args.get("pad_offset", 0.0)

    speech = False
    start = 0.0
    i = 0

    speech_segments = torch.empty(0)

    for i in range(0, len(sequence)):
        # Current frame is speech
        if speech:
            # Switch from speech to non-speech
            if sequence[i] < offset:
                if i * frame_length_in_sec + pad_offset > max(0, start - pad_onset):
                    new_seg = torch.tensor(
                        [
                            max(0, start - pad_onset),
                            i * frame_length_in_sec + pad_offset,
                        ]
                    ).unsqueeze(0)
                    speech_segments = torch.cat((speech_segments, new_seg), 0)

                start = i * frame_length_in_sec
                speech = False

        # Current frame is non-speech
        else:
            # Switch from non-speech to speech
            if sequence[i] > onset:
                start = i * frame_length_in_sec
                speech = True

    # if it's speech at the end, add final segment
    if speech:
        new_seg = torch.tensor(
            [max(0, start - pad_onset), i * frame_length_in_sec + pad_offset]
        ).unsqueeze(0)
        speech_segments = torch.cat((speech_segments, new_seg), 0)

    # Merge the overlapped speech segments due to padding
    speech_segments = merge_overlap_segment(speech_segments)  # not sorted
    return speech_segments


def ts_vad_post_processing(
    ts_vad_binary_vec: torch.Tensor,
    cfg_vad_params: PostProcessingParams,
    unit_10ms_frame_count: int = 8,
    bypass_postprocessing: bool = False,
):
    """
    Post-processing on diarization results using VAD style post-processing methods.
    These post-processing methods are inspired by the following paper:
    Medennikov, Ivan, et al. "Target-Speaker Voice Activity Detection:
                              a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario." (2020).

    Args:
        ts_vad_binary_vec (Tensor):
            Sigmoid values of each frame and each speaker.
            Dimension: (num_frames,)
        cfg_vad_params (PostProcessingParams):
            Configuration (omega config) of VAD parameters.
        unit_10ms_frame_count (int, optional):
            an integer indicating the number of 10ms frames in a unit.
            For example, if unit_10ms_frame_count is 8, then each frame is 0.08 seconds.
        bypass_postprocessing (bool, optional):
            If True, diarization post-processing will be bypassed.

    Returns:
        speech_segments (Tensor):
            start and end of each speech segment.
            Dimension: (num_segments, 2)

            Example:
                tensor([[  0.0000,   3.0400],
                        [  6.0000,   6.0800],
                        ...
                        [587.3600, 591.0400],
                        [591.1200, 597.7600]])
    """
    ts_vad_binary_frames = torch.repeat_interleave(
        ts_vad_binary_vec, unit_10ms_frame_count
    )
    if not bypass_postprocessing:
        speech_segments = binarization(ts_vad_binary_frames, asdict(cfg_vad_params))
        speech_segments = filtering(speech_segments, asdict(cfg_vad_params))
    else:
        cfg_vad_params.onset = 0.5
        cfg_vad_params.offset = 0.5
        cfg_vad_params.pad_onset = 0.0
        cfg_vad_params.pad_offset = 0.0
        speech_segments = binarization(ts_vad_binary_frames, asdict(cfg_vad_params))
    return speech_segments
