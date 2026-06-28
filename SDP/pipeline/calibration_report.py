from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from .alignment import AlignmentMode, MergedSpeechSegment


def build_pipeline_calibration_report(
    *,
    audio_file: str,
    native_segments: Sequence[MergedSpeechSegment],
    onnx_segments: Sequence[MergedSpeechSegment],
    native_diarization_events: Sequence[object] = (),
    native_asr_events: Sequence[object] = (),
    onnx_diarization_events: Sequence[object] = (),
    onnx_asr_events: Sequence[object] = (),
    alignment_mode: AlignmentMode = "diarization_timeline",
    timestamp_tolerance: float = 0.1,
) -> dict:
    native_full_text = _join_segment_text(native_segments)
    onnx_full_text = _join_segment_text(onnx_segments)
    return {
        "audio_file": audio_file,
        "alignment_mode": alignment_mode,
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
        "raw_events": _build_raw_events_section(
            native_diarization_events=native_diarization_events,
            native_asr_events=native_asr_events,
            onnx_diarization_events=onnx_diarization_events,
            onnx_asr_events=onnx_asr_events,
        ),
        "word_diff": _word_diff(native_full_text, onnx_full_text),
        "segment_diff": _segment_diff(
            native_segments, onnx_segments, timestamp_tolerance
        ),
    }


def build_pipeline_raw_events_report(
    *,
    audio_file: str,
    alignment_mode: AlignmentMode,
    native_diarization_events: Sequence[object],
    native_asr_events: Sequence[object],
    onnx_diarization_events: Sequence[object],
    onnx_asr_events: Sequence[object],
) -> dict:
    return {
        "audio_file": audio_file,
        "alignment_mode": alignment_mode,
        "raw_events": _build_raw_events_section(
            native_diarization_events=native_diarization_events,
            native_asr_events=native_asr_events,
            onnx_diarization_events=onnx_diarization_events,
            onnx_asr_events=onnx_asr_events,
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


def _build_raw_events_section(
    *,
    native_diarization_events: Sequence[object],
    native_asr_events: Sequence[object],
    onnx_diarization_events: Sequence[object],
    onnx_asr_events: Sequence[object],
) -> dict:
    return {
        "native": _build_pipeline_raw_events(
            diarization_events=native_diarization_events,
            asr_events=native_asr_events,
        ),
        "onnx": _build_pipeline_raw_events(
            diarization_events=onnx_diarization_events,
            asr_events=onnx_asr_events,
        ),
    }


def _build_pipeline_raw_events(
    *,
    diarization_events: Sequence[object],
    asr_events: Sequence[object],
) -> dict:
    return {
        "diarization_event_count": len(diarization_events),
        "asr_event_count": len(asr_events),
        "asr_full_text": _last_asr_full_text(asr_events),
        "asr_text_delta_joined": "".join(
            str(getattr(event, "text_delta", "")) for event in asr_events
        ).strip(),
        "diarization_events": _serialize_diarization_events(diarization_events),
        "asr_events": _serialize_asr_events(asr_events),
    }


def _serialize_diarization_events(events: Sequence[object]) -> list[dict]:
    return [
        {
            "stream_id": str(getattr(event, "stream_id")),
            "sequence_id": int(getattr(event, "sequence_id")),
            "speaker_id": int(getattr(event, "speaker_id")),
            "start": float(getattr(event, "start")),
            "end": float(getattr(event, "end")),
            "event_type": str(getattr(event, "event_type", "diarization")),
            "is_final": bool(getattr(event, "is_final", True)),
        }
        for event in sorted(events, key=lambda event: float(getattr(event, "start")))
    ]


def _serialize_asr_events(events: Sequence[object]) -> list[dict]:
    return [
        {
            "stream_id": str(getattr(event, "stream_id")),
            "sequence_id": int(getattr(event, "sequence_id")),
            "token_ids": [int(token_id) for token_id in getattr(event, "token_ids", ())],
            "text_delta": str(getattr(event, "text_delta")),
            "full_text": str(getattr(event, "full_text")),
            "token_times": [
                [float(start), float(end)]
                for start, end in getattr(event, "token_times", ())
            ],
            "start": _optional_float(getattr(event, "start")),
            "end": _optional_float(getattr(event, "end")),
            "event_type": str(getattr(event, "event_type", "asr")),
            "is_final": bool(getattr(event, "is_final")),
        }
        for event in events
    ]


def _last_asr_full_text(events: Sequence[object]) -> str:
    for event in reversed(events):
        full_text = str(getattr(event, "full_text", ""))
        if full_text:
            return full_text
    return ""


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


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
