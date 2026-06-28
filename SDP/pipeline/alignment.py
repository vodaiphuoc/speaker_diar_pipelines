from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Protocol, Sequence


AlignmentMode = Literal["diarization_timeline", "asr_timeline"]
UNKNOWN_SPEAKER_ID = -1


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


def merge_asr_to_diarization_timeline(
    diarization_events: Sequence[DiarizationEventLike],
    asr_events: Sequence[ASREventLike],
) -> tuple[MergedSpeechSegment, ...]:
    segments: list[MergedSpeechSegment] = []
    valid_diarization_events = _sorted_valid_diarization_events(diarization_events)
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
        segments.append(
            _build_diarization_timeline_segment(
                diarization_event, assigned_asr_events
            )
        )

    return tuple(segments)


def merge_diarization_to_asr_timeline(
    diarization_events: Sequence[DiarizationEventLike],
    asr_events: Sequence[ASREventLike],
) -> tuple[MergedSpeechSegment, ...]:
    valid_diarization_events = _sorted_valid_diarization_events(diarization_events)
    valid_asr_events = [event for event in asr_events if _is_valid_asr_event(event)]
    return tuple(
        _build_asr_timeline_segment(asr_event, valid_diarization_events)
        for asr_event in valid_asr_events
    )


def merge_pipeline_events(
    diarization_events: Sequence[DiarizationEventLike],
    asr_events: Sequence[ASREventLike],
    alignment_mode: AlignmentMode = "diarization_timeline",
) -> tuple[MergedSpeechSegment, ...]:
    if alignment_mode == "diarization_timeline":
        return merge_asr_to_diarization_timeline(diarization_events, asr_events)
    if alignment_mode == "asr_timeline":
        return merge_diarization_to_asr_timeline(diarization_events, asr_events)
    raise ValueError(
        "alignment_mode must be 'diarization_timeline' or 'asr_timeline'"
    )


def merge_diarization_asr_events(
    diarization_events: Sequence[DiarizationEventLike],
    asr_events: Sequence[ASREventLike],
) -> tuple[MergedSpeechSegment, ...]:
    return merge_asr_to_diarization_timeline(diarization_events, asr_events)


class StreamingPipelineEventMerger:
    def __init__(
        self, alignment_mode: AlignmentMode = "diarization_timeline"
    ) -> None:
        if alignment_mode not in ("diarization_timeline", "asr_timeline"):
            raise ValueError(
                "alignment_mode must be 'diarization_timeline' or 'asr_timeline'"
            )
        self.alignment_mode = alignment_mode
        self._diarization_events: list[DiarizationEventLike] = []
        self._asr_events: list[ASREventLike] = []
        self._seen_diarization_sequence_ids: set[int] = set()
        self._seen_asr_sequence_ids: set[int] = set()
        self._emitted_diarization_sequence_ids: set[int] = set()
        self._emitted_asr_sequence_ids: set[int] = set()
        self._latest_asr_end: float | None = None
        self._latest_diarization_end: float | None = None

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
        self._latest_diarization_end = _max_asr_end(
            event.end for event in diarization_events if _is_valid_diarization_event(event)
        ) or self._latest_diarization_end

        valid_asr_events = [
            event for event in asr_events if _is_valid_asr_event(event)
        ]
        for event in valid_asr_events:
            sequence_id = int(event.sequence_id)
            if sequence_id in self._seen_asr_sequence_ids:
                continue
            self._seen_asr_sequence_ids.add(sequence_id)
            self._asr_events.append(event)
        self._latest_asr_end = _max_asr_end(
            event.end for event in valid_asr_events
        ) or self._latest_asr_end

        if self.alignment_mode == "asr_timeline":
            return self._consume_asr_timeline()
        return self._consume_diarization_timeline()

    def _consume_diarization_timeline(self) -> tuple[MergedSpeechSegment, ...]:
        ready_diarization_events = tuple(
            event
            for event in self._diarization_events
            if event.sequence_id not in self._emitted_diarization_sequence_ids
            and self._latest_asr_end is not None
            and float(event.end) <= self._latest_asr_end
        )
        return self._emit(ready_diarization_events)

    def _consume_asr_timeline(self) -> tuple[MergedSpeechSegment, ...]:
        ready_asr_events = tuple(
            event
            for event in self._asr_events
            if event.sequence_id not in self._emitted_asr_sequence_ids
            and self._latest_diarization_end is not None
            and float(event.end) <= self._latest_diarization_end
        )
        return self._emit_asr_events(ready_asr_events)

    def flush(self) -> tuple[MergedSpeechSegment, ...]:
        if self.alignment_mode == "asr_timeline":
            remaining_asr_events = tuple(
                event
                for event in self._asr_events
                if event.sequence_id not in self._emitted_asr_sequence_ids
            )
            return self._emit_asr_events(remaining_asr_events)

        remaining_diarization_events = tuple(
            event
            for event in self._diarization_events
            if event.sequence_id not in self._emitted_diarization_sequence_ids
        )
        return self._emit(remaining_diarization_events)

    def _emit(
        self, diarization_events: Sequence[DiarizationEventLike]
    ) -> tuple[MergedSpeechSegment, ...]:
        segments = merge_asr_to_diarization_timeline(
            diarization_events=diarization_events,
            asr_events=tuple(self._asr_events),
        )
        self._emitted_diarization_sequence_ids.update(
            int(segment.sequence_id) for segment in segments
        )
        return segments

    def _emit_asr_events(
        self, asr_events: Sequence[ASREventLike]
    ) -> tuple[MergedSpeechSegment, ...]:
        segments = merge_diarization_to_asr_timeline(
            diarization_events=tuple(self._diarization_events),
            asr_events=asr_events,
        )
        self._emitted_asr_sequence_ids.update(
            int(segment.sequence_id) for segment in segments
        )
        return segments


class StreamingDiarizationASRMerger(StreamingPipelineEventMerger):
    def __init__(self) -> None:
        super().__init__(alignment_mode="diarization_timeline")


def _build_diarization_timeline_segment(
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


def _build_asr_timeline_segment(
    asr_event: ASREventLike,
    diarization_events: Sequence[DiarizationEventLike],
) -> MergedSpeechSegment:
    speaker_id = _select_speaker_for_asr_event(asr_event, diarization_events)
    return MergedSpeechSegment(
        stream_id=str(asr_event.stream_id),
        sequence_id=int(asr_event.sequence_id),
        speaker_id=speaker_id,
        start=float(asr_event.start),
        end=float(asr_event.end),
        text=str(asr_event.text_delta).strip(),
        token_ids=tuple(int(token_id) for token_id in getattr(asr_event, "token_ids", ())),
        token_times=tuple(
            (float(start), float(end))
            for start, end in getattr(asr_event, "token_times", ())
        ),
    )


def _select_speaker_for_asr_event(
    asr_event: ASREventLike,
    diarization_events: Sequence[DiarizationEventLike],
) -> int:
    candidates = []
    for diarization_event in diarization_events:
        overlap = _time_overlap(
            float(asr_event.start),
            float(asr_event.end),
            float(diarization_event.start),
            float(diarization_event.end),
        )
        if overlap <= 0.0:
            continue
        candidates.append(
            (
                overlap,
                -float(diarization_event.start),
                -int(diarization_event.sequence_id),
                int(diarization_event.speaker_id),
            )
        )
    if not candidates:
        return UNKNOWN_SPEAKER_ID
    return max(candidates)[3]


def _asr_event_belongs_to_diarization_event(
    asr_event: ASREventLike,
    diarization_event: DiarizationEventLike,
) -> bool:
    midpoint = (float(asr_event.start) + float(asr_event.end)) / 2.0
    diarization_start = float(diarization_event.start)
    diarization_end = float(diarization_event.end)
    return diarization_start <= midpoint < diarization_end


def _time_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _is_valid_diarization_event(event: object) -> bool:
    return all(
        hasattr(event, attr)
        for attr in ("stream_id", "sequence_id", "speaker_id", "start", "end")
    )


def _sorted_valid_diarization_events(
    events: Sequence[DiarizationEventLike],
) -> list[DiarizationEventLike]:
    return sorted(
        (event for event in events if _is_valid_diarization_event(event)),
        key=lambda event: float(event.start),
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
