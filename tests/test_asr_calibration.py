import gc
import os
import unittest
import wave
from pathlib import Path

import numpy as np
import torch

from SDP.onnx.asr.utils.calibration_report import (
    build_asr_calibration_report,
    write_asr_calibration_report,
)
from SDP.onnx.asr import create_nemotron_streaming_session_from_manifest


def _int_sequence_or_none(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return None


@unittest.skipUnless(
    os.environ.get("RUN_NEMOTRON_CALIBRATION") == "1",
    "Set RUN_NEMOTRON_CALIBRATION=1 to run the NeMo/ONNX parity test",
)
class NemotronONNXCalibrationTest(unittest.TestCase):
    """Opt-in parity check based on NeMo's cache-aware streaming example."""

    def test_final_tokens_and_text_match_native_cache_aware_streaming(self):
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        audio_path = Path(
            os.environ.get(
                "NEMOTRON_CALIBRATION_WAV",
                "tests/fixtures/bacsidatnhkhoavitadoc_1.wav",
            )
        )
        with wave.open(str(audio_path), "rb") as wav_file:
            # self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            pcm = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        native = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
            map_location="cpu",
        )
        native.eval()
        native.encoder.setup_streaming_params(att_context_size=[56, 1])
        native.set_inference_prompt("vi-VN")
        native.decoding.set_strip_lang_tags(True)
        streaming_buffer = CacheAwareStreamingAudioBuffer(
            model=native,
            online_normalization=False,
            pad_and_drop_preencoded=True,
        )
        streaming_buffer.append_audio(audio)

        caches = native.encoder.get_initial_cache_state(batch_size=1)
        previous_hypotheses = None
        with torch.inference_mode():
            for chunk, chunk_length in streaming_buffer:
                (
                    _,
                    _,
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
                    drop_extra_pre_encoded=(
                        native.encoder.streaming_cfg.drop_extra_pre_encoded
                    ),
                    return_transcription=True,
                )
                caches = channel_cache, time_cache, channel_cache_length

        self.assertIsNotNone(previous_hypotheses)
        native_hypothesis = previous_hypotheses[0]
        native_tokens = tuple(int(token) for token in native_hypothesis.y_sequence)
        native_text = native_hypothesis.text
        native_token_timestamps = _int_sequence_or_none(
            getattr(native_hypothesis, "timestamp", None)
        )

        del native_hypothesis
        del previous_hypotheses
        del caches
        del streaming_buffer
        del native
        gc.collect()

        asset_dir = Path(os.environ.get("ASR_ASSET_DIR", ".onnx_ckpt/asr"))
        onnx_session = create_nemotron_streaming_session_from_manifest(
            asset_dir / "asr_artifact.json",
            target_language="vi-VN",
        )
        bytes_per_chunk = 16000 // 10 * 2
        onnx_events = []
        for offset in range(0, len(pcm), bytes_per_chunk):
            onnx_events.extend(
                onnx_session.process_pcm(pcm[offset : offset + bytes_per_chunk])
            )
        onnx_events.extend(onnx_session.flush())

        onnx_token_times = tuple(
            token_time
            for event in onnx_events
            for token_time in event.token_times
        )
        report = build_asr_calibration_report(
            audio_file=str(audio_path),
            native_text=native_text,
            native_token_ids=native_tokens,
            native_token_timestamps=native_token_timestamps,
            onnx_text=onnx_session.full_text,
            onnx_token_ids=onnx_session.token_ids,
            onnx_token_times=onnx_token_times,
        )
        write_asr_calibration_report(
            os.environ.get(
                "NEMOTRON_CALIBRATION_REPORT",
                "ci-logs/asr_calibration_report.json",
            ),
            report,
        )

        self.assertEqual(onnx_session.token_ids, native_tokens)
        self.assertEqual(onnx_session.full_text, native_text)


if __name__ == "__main__":
    unittest.main()
