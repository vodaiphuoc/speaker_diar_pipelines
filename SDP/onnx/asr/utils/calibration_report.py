from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence


def normalize_words(text: str) -> list[str]:
    """Return lowercase word tokens for transcript-level comparison."""
    return re.findall(r"\w+(?:['’-]\w+)?", text.lower(), flags=re.UNICODE)


def compare_words(
    native_text: str,
    onnx_text: str,
) -> dict[str, Any]:
    native_words = normalize_words(native_text)
    onnx_words = normalize_words(onnx_text)
    operations: list[dict[str, Any]] = []

    matcher = SequenceMatcher(a=native_words, b=onnx_words, autojunk=False)
    for tag, native_start, native_end, onnx_start, onnx_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        operations.append(
            {
                "op": tag,
                "native_words": native_words[native_start:native_end],
                "onnx_words": onnx_words[onnx_start:onnx_end],
                "native_range": [native_start, native_end],
                "onnx_range": [onnx_start, onnx_end],
                "native_timestamps": None,
                "onnx_timestamps": None,
            }
        )

    return {
        "same": not operations,
        "native_words": native_words,
        "onnx_words": onnx_words,
        "operations": operations,
    }


def token_frames_to_token_times(
    token_frames: Sequence[int] | None,
    *,
    frame_duration: float = 0.08,
) -> list[list[float]] | None:
    if token_frames is None:
        return None
    return [
        [
            round(int(frame) * frame_duration, 2),
            round((int(frame) + 1) * frame_duration, 2),
        ]
        for frame in token_frames
    ]


def _serialize_token_times(
    token_times: Sequence[Sequence[float]] | None,
) -> list[list[float]] | None:
    if token_times is None:
        return None
    return [[float(start), float(end)] for start, end in token_times]


def build_asr_calibration_report(
    *,
    audio_file: str,
    native_text: str,
    native_token_ids: Sequence[int],
    native_token_times: Sequence[Sequence[float]] | None,
    onnx_text: str,
    onnx_token_ids: Sequence[int],
    onnx_token_times: Sequence[Sequence[float]],
) -> dict[str, Any]:
    native_token_id_list = [int(token) for token in native_token_ids]
    onnx_token_id_list = [int(token) for token in onnx_token_ids]
    return {
        "audio_file": audio_file,
        "native_nemo": {
            "full_text": native_text,
            "token_ids": native_token_id_list,
            "token_times": _serialize_token_times(native_token_times),
        },
        "onnx_streaming": {
            "full_text": onnx_text,
            "token_ids": onnx_token_id_list,
            "token_times": _serialize_token_times(onnx_token_times),
        },
        "word_diff": compare_words(native_text, onnx_text),
        "exact_match": {
            "text": native_text == onnx_text,
            "token_ids": native_token_id_list == onnx_token_id_list,
        },
    }


def write_asr_calibration_report(
    output_path: str | Path,
    report: dict[str, Any],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
