import gc
import os
import re
import unittest
from pathlib import Path

import torch

from SDP import wav_to_mono_pcm16_bytes
from SDP.onnx.asr import StreamingASREvent
from SDP.onnx.asr.utils.calibration_report import token_frames_to_token_times
from SDP.onnx.streaming_service import (
    StreamingDiarizationASROnnxService,
    StreamingDiarizationEvent,
    StreamingPipelineResult,
)
from SDP.pipeline import MergedSpeechSegment, merge_pipeline_events
from SDP.pipeline.calibration_report import (
    build_pipeline_calibration_report,
    build_pipeline_raw_events_report,
    write_pipeline_calibration_report,
)
from tests.calibration.support import (
    extract_native_transcription_texts,
    int_sequence_or_none,
    move_tensor_to_device,
    resolve_native_device,
)


def _speaker_id(value) -> int:
    if isinstance(value, str):
        matches = re.findall(r"\d+", value)
        if not matches:
            raise ValueError(f"Cannot parse speaker id from {value!r}")
        return int(matches[-1])
    return int(value)


def _parse_native_diarization_segment(segment) -> tuple[float, float, int]:
    if isinstance(segment, dict):
        start = segment.get("start", segment.get("begin"))
        end = segment.get("end", segment.get("stop"))
        speaker = segment.get("speaker_id", segment.get("speaker"))
        if start is None or end is None or speaker is None:
            raise ValueError(f"Unsupported diarization segment dict: {segment!r}")
        return float(start), float(end), _speaker_id(speaker)

    if isinstance(segment, (list, tuple)) and len(segment) >= 3:
        return float(segment[0]), float(segment[1]), _speaker_id(segment[2])

    if isinstance(segment, str):
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", segment)
        if len(numbers) < 3:
            raise ValueError(f"Unsupported diarization segment string: {segment!r}")
        return float(numbers[0]), float(numbers[1]), int(float(numbers[-1]))

    raise TypeError(f"Unsupported diarization segment type: {type(segment)!r}")


def _native_diarization_segments_to_events(
    segments, stream_id: str = "native"
) -> tuple[StreamingDiarizationEvent, ...]:
    events = []
    for sequence_id, segment in enumerate(segments):
        start, end, speaker_id = _parse_native_diarization_segment(segment)
        events.append(
            StreamingDiarizationEvent(
                stream_id=stream_id,
                sequence_id=sequence_id,
                speaker_id=speaker_id,
                start=round(start, 2),
                end=round(end, 2),
            )
        )
    return tuple(events)


def _run_native_diarization_events(audio_path: Path):
    from nemo.collections.asr.models import SortformerEncLabelModel

    native_device = resolve_native_device()
    diar_model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1",
        map_location=native_device,
    )
    diar_model = diar_model.to(device=native_device)
    diar_model.eval()
    diar_model.sortformer_modules.chunk_len = 6
    diar_model.sortformer_modules.chunk_right_context = 7
    diar_model.sortformer_modules.fifo_len = 188
    diar_model.sortformer_modules.spkcache_update_period = 144
    diar_model.sortformer_modules.spkcache_len = 188

    predicted_segments = diar_model.diarize(audio=[str(audio_path)], batch_size=1)
    del diar_model
    gc.collect()

    return _native_diarization_segments_to_events(
        predicted_segments[0], stream_id="native"
    )


def _run_native_asr_events(audio_path: Path):
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    native_device = resolve_native_device()
    native = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        map_location=native_device,
    )
    native = native.to(device=native_device)
    native.eval()
    native.encoder.setup_streaming_params(att_context_size=[56, 1])
    native.set_inference_prompt("vi-VN")
    native.decoding.set_strip_lang_tags(True)
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=native,
        online_normalization=False,
        pad_and_drop_preencoded=True,
    )
    streaming_buffer.append_audio_file(audio_path)

    caches = tuple(
        move_tensor_to_device(cache, native_device)
        for cache in native.encoder.get_initial_cache_state(batch_size=1)
    )
    pred_out_stream = None
    transcribed_texts = None
    previous_hypotheses = None
    previous_text = ""
    previous_token_count = 0
    events = []

    with torch.inference_mode():
        for chunk, chunk_length in iter(streaming_buffer):
            chunk = move_tensor_to_device(chunk, native_device)
            chunk_length = move_tensor_to_device(chunk_length, native_device)
            (
                pred_out_stream,
                transcribed_texts,
                channel_cache,
                time_cache,
                channel_cache_length,
                previous_hypotheses,
            ) = native.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_length,
                cache_last_channel=caches[0],
                cache_last_time=caches[1],
                cache_last_channel_len=caches[2],
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=native.encoder.streaming_cfg.drop_extra_pre_encoded,
                return_transcription=True,
            )
            caches = channel_cache, time_cache, channel_cache_length
            transcriptions = extract_native_transcription_texts(transcribed_texts)
            current_text = transcriptions[0] if transcriptions else previous_text
            text_delta = (
                current_text[len(previous_text) :]
                if current_text.startswith(previous_text)
                else current_text
            )

            token_ids = ()
            token_times = ()
            if previous_hypotheses:
                hypothesis = previous_hypotheses[0]
                current_token_ids = tuple(int(token) for token in hypothesis.y_sequence)
                token_ids = current_token_ids[previous_token_count:]
                token_frames = int_sequence_or_none(
                    getattr(hypothesis, "timestamp", None)
                )
                if token_frames is not None:
                    token_times = token_frames_to_token_times(
                        token_frames[previous_token_count:]
                    )
                previous_token_count = len(current_token_ids)

            if text_delta or token_ids:
                events.append(
                    StreamingASREvent(
                        stream_id="native",
                        sequence_id=len(events),
                        token_ids=tuple(token_ids),
                        text_delta=text_delta,
                        full_text=current_text,
                        token_times=tuple(token_times or ()),
                        start=token_times[0][0] if token_times else None,
                        end=token_times[-1][1] if token_times else None,
                        is_final=False,
                    )
                )
            previous_text = current_text

    del previous_hypotheses
    del caches
    del streaming_buffer
    del native
    gc.collect()
    return tuple(events)


def _run_onnx_pipeline_events(
    audio_path: Path, alignment_mode: str
) -> tuple[
    tuple[StreamingDiarizationEvent, ...],
    tuple[StreamingASREvent, ...],
    tuple[MergedSpeechSegment, ...],
]:
    asr_asset_dir = Path(os.environ.get("ASR_ASSET_DIR", ".onnx_ckpt/asr"))
    diar_asset_dir = Path(os.environ.get("DIAR_ASSET_DIR", ".onnx_ckpt/diar"))
    service = StreamingDiarizationASROnnxService.from_manifests(
        diarization_manifest_path=diar_asset_dir / "diarization_artifact.json",
        asr_manifest_path=asr_asset_dir / "asr_artifact.json",
        device=os.environ.get("ONNX_PIPELINE_DEVICE", "cpu"),
        target_language="vi-VN",
        alignment_mode=alignment_mode,
    )
    pcm = wav_to_mono_pcm16_bytes(audio_path, target_sr=16000)
    bytes_per_chunk = 16000 // 10 * 2
    diarization_events = []
    asr_events = []
    for offset in range(0, len(pcm), bytes_per_chunk):
        result = service.process(
            pcm[offset : offset + bytes_per_chunk], stream_id="onnx"
        )
        diarization_events.extend(result.diarization_events)
        asr_events.extend(result.asr_events)
    result = service.flush(stream_id="onnx")
    diarization_events.extend(result.diarization_events)
    asr_events.extend(result.asr_events)
    return (
        tuple(diarization_events),
        tuple(asr_events),
        merge_pipeline_events(
            diarization_events=diarization_events,
            asr_events=asr_events,
            alignment_mode=alignment_mode,
        ),
    )


def _raw_events_report_path(calibration_report_path: str | Path) -> Path:
    report_path = Path(calibration_report_path)
    alignment_mode = _resolve_pipeline_alignment_mode()
    return report_path.with_name(f"pipeline_raw_events_{alignment_mode}.json")


def _resolve_pipeline_alignment_mode():
    mode = os.environ.get("PIPELINE_ALIGNMENT_MODE", "diarization_timeline").strip()
    if mode not in ("diarization_timeline", "asr_timeline"):
        raise ValueError(
            "PIPELINE_ALIGNMENT_MODE must be 'diarization_timeline' or 'asr_timeline'"
        )
    return mode


def _merge_pipeline_events_for_calibration(diarization_events, asr_events):
    return merge_pipeline_events(
        diarization_events=diarization_events,
        asr_events=asr_events,
        alignment_mode=_resolve_pipeline_alignment_mode(),
    )


class NativeDiarizationSegmentParsingTest(unittest.TestCase):
    def test_parses_documented_string_segment_format(self):
        self.assertEqual(
            _parse_native_diarization_segment("[0.12, 1.23, speaker_2]"),
            (0.12, 1.23, 2),
        )

    def test_parses_sequence_segment_format(self):
        self.assertEqual(
            _parse_native_diarization_segment((0.1, 0.2, 3)), (0.1, 0.2, 3)
        )

    def test_parses_dict_segment_format(self):
        self.assertEqual(
            _parse_native_diarization_segment(
                {"start": 0.1, "end": 0.2, "speaker": "speaker_4"}
            ),
            (0.1, 0.2, 4),
        )


class PipelineAlignmentModeResolutionTest(unittest.TestCase):
    def test_defaults_to_diarization_timeline(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _resolve_pipeline_alignment_mode(),
                "diarization_timeline",
            )

    def test_accepts_asr_timeline(self):
        with unittest.mock.patch.dict(
            os.environ, {"PIPELINE_ALIGNMENT_MODE": "asr_timeline"}
        ):
            self.assertEqual(_resolve_pipeline_alignment_mode(), "asr_timeline")

    def test_rejects_unknown_mode(self):
        with unittest.mock.patch.dict(os.environ, {"PIPELINE_ALIGNMENT_MODE": "bad"}):
            with self.assertRaisesRegex(ValueError, "PIPELINE_ALIGNMENT_MODE"):
                _resolve_pipeline_alignment_mode()


class OnnxPipelineEventCollectionTest(unittest.TestCase):
    def test_collects_raw_onnx_events_before_merge(self):
        diarization_event = StreamingDiarizationEvent(
            stream_id="onnx",
            sequence_id=0,
            speaker_id=2,
            start=0.0,
            end=1.0,
        )
        asr_event = StreamingASREvent(
            stream_id="onnx",
            sequence_id=0,
            token_ids=(101,),
            text_delta="xin",
            full_text="xin",
            token_times=((0.1, 0.2),),
            start=0.1,
            end=0.2,
            is_final=False,
        )

        class FakePipelineService:
            def process(self, audio, stream_id="default"):
                return StreamingPipelineResult(
                    diarization_events=(diarization_event,),
                    asr_events=(),
                    merged_segments=(),
                )

            def flush(self, stream_id="default"):
                return StreamingPipelineResult(
                    diarization_events=(),
                    asr_events=(asr_event,),
                    merged_segments=(),
                )

        with (
            unittest.mock.patch(
                f"{__name__}.wav_to_mono_pcm16_bytes",
                return_value=b"\x00\x00",
            ),
            unittest.mock.patch.object(
                StreamingDiarizationASROnnxService,
                "from_manifests",
                return_value=FakePipelineService(),
            ),
        ):
            diarization_events, asr_events, segments = _run_onnx_pipeline_events(
                Path("sample.wav"),
                "asr_timeline",
            )

        self.assertEqual(diarization_events, (diarization_event,))
        self.assertEqual(asr_events, (asr_event,))
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].speaker_id, 2)
        self.assertEqual(segments[0].text, "xin")


@unittest.skipUnless(
    os.environ.get("RUN_PIPELINE_CALIBRATION") == "1",
    "Set RUN_PIPELINE_CALIBRATION=1 to run pipeline-level NeMo/ONNX calibration",
)
class NativeVsOnnxPipelineCalibrationTest(unittest.TestCase):
    def test_merged_speaker_transcript_segments_are_reported(self):
        audio_path = Path(
            os.environ.get(
                "PIPELINE_CALIBRATION_WAV",
                "tests/fixtures/bacsidatnhkhoavitadoc_1.wav",
            )
        )
        native_diarization_events = _run_native_diarization_events(audio_path)
        native_asr_events = _run_native_asr_events(audio_path)
        alignment_mode = _resolve_pipeline_alignment_mode()
        native_segments = _merge_pipeline_events_for_calibration(
            native_diarization_events, native_asr_events
        )
        (
            onnx_diarization_events,
            onnx_asr_events,
            onnx_segments,
        ) = _run_onnx_pipeline_events(audio_path, alignment_mode)

        report = build_pipeline_calibration_report(
            audio_file=str(audio_path),
            alignment_mode=alignment_mode,
            native_segments=native_segments,
            onnx_segments=onnx_segments,
            native_diarization_events=native_diarization_events,
            native_asr_events=native_asr_events,
            onnx_diarization_events=onnx_diarization_events,
            onnx_asr_events=onnx_asr_events,
        )
        calibration_report_path = os.environ.get(
            "PIPELINE_CALIBRATION_REPORT",
            "ci-logs/pipeline_calibration_report.json",
        )
        write_pipeline_calibration_report(calibration_report_path, report)
        write_pipeline_calibration_report(
            os.environ.get(
                "PIPELINE_RAW_EVENTS_REPORT",
                str(_raw_events_report_path(calibration_report_path)),
            ),
            build_pipeline_raw_events_report(
                audio_file=str(audio_path),
                alignment_mode=alignment_mode,
                native_diarization_events=native_diarization_events,
                native_asr_events=native_asr_events,
                onnx_diarization_events=onnx_diarization_events,
                onnx_asr_events=onnx_asr_events,
            ),
        )

        self.assertGreater(
            len(native_segments),
            0,
            "Native pipeline produced no merged diarization/ASR segments.",
        )
        self.assertGreater(
            len(onnx_segments),
            0,
            "ONNX pipeline produced no merged diarization/ASR segments.",
        )


if __name__ == "__main__":
    unittest.main()
