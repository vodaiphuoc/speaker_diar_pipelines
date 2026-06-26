import unittest

from SDP.onnx.asr import StreamingASREvent
from SDP.onnx.streaming_service import StreamingDiarizationEvent
from SDP.pipeline import (
    StreamingDiarizationASRMerger,
    StreamingPipelineEventMerger,
    merge_asr_to_diarization_timeline,
    merge_diarization_asr_events,
    merge_diarization_to_asr_timeline,
    merge_pipeline_events,
)


def diar_event(sequence_id, speaker_id, start, end):
    return StreamingDiarizationEvent(
        stream_id="s1",
        sequence_id=sequence_id,
        speaker_id=speaker_id,
        start=start,
        end=end,
    )


def asr_event(sequence_id, text_delta, start, end, token_ids=(1,)):
    return StreamingASREvent(
        stream_id="s1",
        sequence_id=sequence_id,
        token_ids=token_ids,
        text_delta=text_delta,
        full_text=text_delta,
        token_times=((start, end),),
        start=start,
        end=end,
        is_final=False,
    )


class PipelineAlignmentTest(unittest.TestCase):
    def test_diarization_timeline_is_main_stream_for_merged_segments(self):
        segments = merge_asr_to_diarization_timeline(
            diarization_events=(
                diar_event(0, speaker_id=0, start=0.0, end=1.0),
                diar_event(1, speaker_id=1, start=1.0, end=2.0),
            ),
            asr_events=(
                asr_event(0, "xin ", start=0.2, end=0.4, token_ids=(10,)),
                asr_event(1, "chào", start=1.2, end=1.4, token_ids=(11,)),
            ),
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker_id, 0)
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 1.0)
        self.assertEqual(segments[0].text, "xin")
        self.assertEqual(segments[0].token_ids, (10,))
        self.assertEqual(segments[1].speaker_id, 1)
        self.assertEqual(segments[1].text, "chào")

    def test_legacy_merge_name_preserves_diarization_timeline_behavior(self):
        self.assertEqual(
            merge_diarization_asr_events(
                diarization_events=(diar_event(0, speaker_id=0, start=0.0, end=1.0),),
                asr_events=(asr_event(0, "xin", start=0.2, end=0.4),),
            ),
            merge_asr_to_diarization_timeline(
                diarization_events=(diar_event(0, speaker_id=0, start=0.0, end=1.0),),
                asr_events=(asr_event(0, "xin", start=0.2, end=0.4),),
            ),
        )

    def test_midpoint_assignment_prevents_boundary_duplication(self):
        segments = merge_asr_to_diarization_timeline(
            diarization_events=(
                diar_event(0, speaker_id=0, start=0.0, end=1.0),
                diar_event(1, speaker_id=1, start=1.0, end=2.0),
            ),
            asr_events=(
                asr_event(0, "boundary", start=0.9, end=1.1, token_ids=(20,)),
            ),
        )

        self.assertEqual(segments[0].text, "")
        self.assertEqual(segments[1].text, "boundary")
        self.assertEqual(segments[0].token_ids, ())
        self.assertEqual(segments[1].token_ids, (20,))

    def test_asr_timeline_uses_asr_event_times_and_max_overlap_speaker(self):
        segments = merge_diarization_to_asr_timeline(
            diarization_events=(
                diar_event(0, speaker_id=0, start=0.0, end=1.0),
                diar_event(1, speaker_id=2, start=0.8, end=2.0),
            ),
            asr_events=(
                asr_event(4, "xin chào", start=0.7, end=1.5, token_ids=(40,)),
            ),
        )

        self.assertEqual(
            segments[0],
            type(segments[0])(
                stream_id="s1",
                sequence_id=4,
                speaker_id=2,
                start=0.7,
                end=1.5,
                text="xin chào",
                token_ids=(40,),
                token_times=((0.7, 1.5),),
            ),
        )

    def test_asr_timeline_uses_unknown_speaker_when_no_diarization_overlap(self):
        segments = merge_diarization_to_asr_timeline(
            diarization_events=(diar_event(0, speaker_id=0, start=0.0, end=1.0),),
            asr_events=(asr_event(0, "late", start=2.0, end=2.5, token_ids=(50,)),),
        )

        self.assertEqual(segments[0].speaker_id, -1)
        self.assertEqual(segments[0].start, 2.0)
        self.assertEqual(segments[0].end, 2.5)

    def test_asr_timeline_overlap_tie_breaks_by_earlier_diarization_then_sequence(self):
        segments = merge_diarization_to_asr_timeline(
            diarization_events=(
                diar_event(4, speaker_id=4, start=1.0, end=2.0),
                diar_event(1, speaker_id=1, start=0.0, end=1.0),
            ),
            asr_events=(asr_event(0, "tie", start=0.5, end=1.5),),
        )

        self.assertEqual(segments[0].speaker_id, 1)

    def test_merge_pipeline_events_selects_alignment_mode(self):
        diarization_events = (diar_event(0, speaker_id=0, start=0.0, end=1.0),)
        asr_events = (asr_event(0, "xin", start=0.2, end=0.4),)

        self.assertEqual(
            merge_pipeline_events(
                diarization_events=diarization_events,
                asr_events=asr_events,
                alignment_mode="diarization_timeline",
            )[0].end,
            1.0,
        )
        self.assertEqual(
            merge_pipeline_events(
                diarization_events=diarization_events,
                asr_events=asr_events,
                alignment_mode="asr_timeline",
            )[0].end,
            0.4,
        )

    def test_streaming_merger_holds_segments_until_asr_covers_them(self):
        merger = StreamingDiarizationASRMerger()

        first = merger.consume(
            diarization_events=(diar_event(0, speaker_id=0, start=0.0, end=1.0),),
            asr_events=(asr_event(0, "xin", start=0.1, end=0.4),),
        )
        second = merger.consume(
            diarization_events=(),
            asr_events=(asr_event(1, " chào", start=1.1, end=1.2),),
        )
        flushed = merger.flush()
        flushed_again = merger.flush()

        self.assertEqual(first, ())
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].text, "xin")
        self.assertEqual(flushed, ())
        self.assertEqual(flushed_again, ())

    def test_flush_emits_remaining_diarization_segments_even_without_asr_text(self):
        merger = StreamingDiarizationASRMerger()
        merger.consume(
            diarization_events=(diar_event(0, speaker_id=0, start=0.0, end=1.0),),
            asr_events=(),
        )

        segments = merger.flush()

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].speaker_id, 0)
        self.assertEqual(segments[0].text, "")

    def test_asr_timeline_streaming_merger_holds_asr_until_diarization_covers_it(self):
        merger = StreamingPipelineEventMerger(alignment_mode="asr_timeline")

        first = merger.consume(
            diarization_events=(),
            asr_events=(asr_event(0, "xin", start=0.1, end=0.4),),
        )
        second = merger.consume(
            diarization_events=(diar_event(0, speaker_id=3, start=0.0, end=0.5),),
            asr_events=(),
        )

        self.assertEqual(first, ())
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].speaker_id, 3)
        self.assertEqual(second[0].start, 0.1)
        self.assertEqual(second[0].end, 0.4)


if __name__ == "__main__":
    unittest.main()
