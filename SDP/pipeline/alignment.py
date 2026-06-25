from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence


class DiarizationEventLike(Protocol):
    stream_id: str
    sequence_id: int
    speaker_id: int
    start: float
    end: float


class ASREventLike(Protocol):
    stream_id: str
    sequence_id: int
    token_ids: tuple[int, ...]
    text_delta: str
    token_times: tuple[tuple[float, float], ...]
    start: float | None
    end: float | None


@dataclass(frozen=True)
class MergedSpeechSegment:
    stream_id: str
    sequence_id: int
    speaker_id: int
    start: float
    end: float
    text: str
    token_ids: tuple[int, ...] = ()
    token_times: tuple[tuple[float, float], ...] = ()


def merge_diarization_asr_events(
    diarization_events: Sequence[DiarizationEventLike],
    asr_events: Sequence[ASREventLike],
) -> tuple[MergedSpeechSegment, ...]:
    segments: list[MergedSpeechSegment] = []
    valid_diarization_events = [
        event for event in diarization_events if _is_valid_diarization_event(event)
    ]
    valid_asr_events = [event for event in asr_events if _is_valid_asr_event(event)]

    for diarization_event in valid_diarization_events:
        assigned_asr_events = tuple(
            asr_event
            for asr_event in valid_asr_events
            if _asr_event_belongs_to_diarization_event(
                asr_event=asr_event,
                diarization_event=diarization_event,
            )
        )
        segments.append(_build_merged_segment(diarization_event, assigned_asr_events))

    return tuple(segments)


class StreamingDiarizationASRMerger:
    def __init__(self) -> None:
        self._diarization_events: list[DiarizationEventLike] = []
        self._asr_events: list[ASREventLike] = []
        self._seen_diarization_sequence_ids: set[int] = set()
        self._emitted_diarization_sequence_ids: set[int] = set()
        self._latest_asr_end: float | None = None

    def consume(
        self,
        diarization_events: Sequence[DiarizationEventLike],
        asr_events: Sequence[ASREventLike],
    ) -> tuple[MergedSpeechSegment, ...]:
        for event in diarization_events:
            if not _is_valid_diarization_event(event):
                continue
            sequence_id = int(event.sequence_id)
            if sequence_id in self._seen_diarization_sequence_ids:
                continue
            self._seen_diarization_sequence_ids.add(sequence_id)
            self._diarization_events.append(event)
        valid_asr_events = [
            event for event in asr_events if _is_valid_asr_event(event)
        ]
        self._asr_events.extend(valid_asr_events)
        self._latest_asr_end = _max_asr_end(
            event.end for event in valid_asr_events
        ) or self._latest_asr_end

        ready_diarization_events = tuple(
            event
            for event in self._diarization_events
            if event.sequence_id not in self._emitted_diarization_sequence_ids
            and self._latest_asr_end is not None
            and float(event.end) <= self._latest_asr_end
        )
        return self._emit(ready_diarization_events)

    def flush(self) -> tuple[MergedSpeechSegment, ...]:
        remaining_diarization_events = tuple(
            event
            for event in self._diarization_events
            if event.sequence_id not in self._emitted_diarization_sequence_ids
        )
        return self._emit(remaining_diarization_events)

    def _emit(
        self, diarization_events: Sequence[DiarizationEventLike]
    ) -> tuple[MergedSpeechSegment, ...]:
        segments = merge_diarization_asr_events(
            diarization_events=diarization_events,
            asr_events=tuple(self._asr_events),
        )
        self._emitted_diarization_sequence_ids.update(
            int(segment.sequence_id) for segment in segments
        )
        return segments


def _build_merged_segment(
    diarization_event: DiarizationEventLike,
    asr_events: Sequence[ASREventLike],
) -> MergedSpeechSegment:
    return MergedSpeechSegment(
        stream_id=str(diarization_event.stream_id),
        sequence_id=int(diarization_event.sequence_id),
        speaker_id=int(diarization_event.speaker_id),
        start=float(diarization_event.start),
        end=float(diarization_event.end),
        text="".join(str(event.text_delta) for event in asr_events).strip(),
        token_ids=tuple(
            int(token_id)
            for event in asr_events
            for token_id in getattr(event, "token_ids", ())
        ),
        token_times=tuple(
            (float(start), float(end))
            for event in asr_events
            for start, end in getattr(event, "token_times", ())
        ),
    )


def _asr_event_belongs_to_diarization_event(
    asr_event: ASREventLike,
    diarization_event: DiarizationEventLike,
) -> bool:
    midpoint = (float(asr_event.start) + float(asr_event.end)) / 2.0
    diarization_start = float(diarization_event.start)
    diarization_end = float(diarization_event.end)
    return diarization_start <= midpoint < diarization_end


def _is_valid_diarization_event(event: object) -> bool:
    return all(
        hasattr(event, attr)
        for attr in ("stream_id", "sequence_id", "speaker_id", "start", "end")
    )


def _is_valid_asr_event(event: object) -> bool:
    if not all(
        hasattr(event, attr)
        for attr in ("stream_id", "sequence_id", "text_delta", "start", "end")
    ):
        return False
    return getattr(event, "start") is not None and getattr(event, "end") is not None


def _max_asr_end(values: Iterable[float | None]) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)
