from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from .alignment import MergedSpeechSegment


def build_pipeline_calibration_report(
    *,
    audio_file: str,
    native_segments: Sequence[MergedSpeechSegment],
    onnx_segments: Sequence[MergedSpeechSegment],
    timestamp_tolerance: float = 0.1,
) -> dict:
    native_full_text = _join_segment_text(native_segments)
    onnx_full_text = _join_segment_text(onnx_segments)
    return {
        "audio_file": audio_file,
        "timestamp_tolerance": float(timestamp_tolerance),
        "exact_match": {
            "segment_count": len(native_segments) == len(onnx_segments),
            "speaker_ids": _speaker_ids(native_segments) == _speaker_ids(onnx_segments),
            "timestamps_within_tolerance": _timestamps_within_tolerance(
                native_segments, onnx_segments, timestamp_tolerance
            ),
            "text": native_full_text == onnx_full_text,
        },
        "native_pipeline": {
            "full_text": native_full_text,
            "segments": _serialize_segments(native_segments),
        },
        "onnx_pipeline": {
            "full_text": onnx_full_text,
            "segments": _serialize_segments(onnx_segments),
        },
        "word_diff": _word_diff(native_full_text, onnx_full_text),
        "segment_diff": _segment_diff(
            native_segments, onnx_segments, timestamp_tolerance
        ),
    }


def write_pipeline_calibration_report(path: str | Path, report: dict) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _serialize_segments(segments: Sequence[MergedSpeechSegment]) -> list[dict]:
    return [
        {
            "stream_id": segment.stream_id,
            "sequence_id": segment.sequence_id,
            "speaker_id": segment.speaker_id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
            "token_ids": list(segment.token_ids),
            "token_times": [
                [float(start), float(end)] for start, end in segment.token_times
            ],
        }
        for segment in segments
    ]


def _speaker_ids(segments: Sequence[MergedSpeechSegment]) -> tuple[int, ...]:
    return tuple(segment.speaker_id for segment in segments)


def _timestamps_within_tolerance(
    native_segments: Sequence[MergedSpeechSegment],
    onnx_segments: Sequence[MergedSpeechSegment],
    tolerance: float,
) -> bool:
    if len(native_segments) != len(onnx_segments):
        return False
    return all(
        abs(native.start - onnx.start) <= tolerance
        and abs(native.end - onnx.end) <= tolerance
        for native, onnx in zip(native_segments, onnx_segments)
    )


def _join_segment_text(segments: Sequence[MergedSpeechSegment]) -> str:
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def _word_diff(native_text: str, onnx_text: str) -> dict:
    native_words = native_text.split()
    onnx_words = onnx_text.split()
    matcher = SequenceMatcher(a=native_words, b=onnx_words)
    return {
        "native_words": native_words,
        "onnx_words": onnx_words,
        "same": native_words == onnx_words,
        "operations": [
            {
                "op": tag,
                "native": native_words[i1:i2],
                "onnx": onnx_words[j1:j2],
            }
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        ],
    }


def _segment_diff(
    native_segments: Sequence[MergedSpeechSegment],
    onnx_segments: Sequence[MergedSpeechSegment],
    tolerance: float,
) -> list[dict]:
    rows = []
    for index, (native, onnx) in enumerate(zip(native_segments, onnx_segments)):
        rows.append(
            {
                "index": index,
                "speaker_match": native.speaker_id == onnx.speaker_id,
                "start_delta": round(onnx.start - native.start, 4),
                "end_delta": round(onnx.end - native.end, 4),
                "timestamps_within_tolerance": (
                    abs(onnx.start - native.start) <= tolerance
                    and abs(onnx.end - native.end) <= tolerance
                ),
                "text_match": native.text == onnx.text,
                "word_diff": _word_diff(native.text, onnx.text),
            }
        )
    if len(native_segments) != len(onnx_segments):
        rows.append(
            {
                "index": len(rows),
                "segment_count_mismatch": {
                    "native": len(native_segments),
                    "onnx": len(onnx_segments),
                },
            }
        )
    return rows
