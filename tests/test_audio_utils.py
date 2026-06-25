import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from SDP.utils import (
    float32_to_pcm16_bytes,
    resample_audio_like_nemo,
    wav_to_mono_pcm16_bytes,
)


class AudioUtilsTest(unittest.TestCase):
    def test_resample_audio_like_nemo_returns_float32_without_resampling_same_rate(
        self,
    ):
        samples = np.array([0, 1000, -1000], dtype=np.int16)

        with mock.patch("librosa.core.resample") as resample:
            output = resample_audio_like_nemo(samples, orig_sr=16000, target_sr=16000)

        resample.assert_not_called()
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_allclose(output, samples.astype(np.float32))

    def test_resample_audio_like_nemo_uses_librosa_core_resample_for_rate_change(
        self,
    ):
        samples = np.array([0.0, 0.5, -0.5], dtype=np.float32)

        with mock.patch(
            "librosa.core.resample",
            return_value=np.array([0.0, 0.25, 0.5, -0.25, -0.5], dtype=np.float32),
        ) as resample:
            output = resample_audio_like_nemo(samples, orig_sr=8000, target_sr=16000)

        resample.assert_called_once()
        _, kwargs = resample.call_args
        self.assertEqual(kwargs["orig_sr"], 8000)
        self.assertEqual(kwargs["target_sr"], 16000)
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_allclose(output, [0.0, 0.25, 0.5, -0.25, -0.5])

    def test_float32_to_pcm16_bytes_clips_and_converts(self):
        pcm = float32_to_pcm16_bytes(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))

        decoded = np.frombuffer(pcm, dtype=np.int16)
        np.testing.assert_array_equal(
            decoded,
            np.array([-32768, -32768, 0, 32767, 32767], dtype=np.int16),
        )

    def test_wav_to_mono_pcm16_bytes_resamples_non_16k_stereo_wav(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "stereo_8k.wav"
            # Two stereo frames. Averaging gives mono samples [0.25, -0.25].
            stereo = np.array(
                [
                    [0.5, 0.0],
                    [-0.5, 0.0],
                ],
                dtype=np.float32,
            )
            stereo_pcm = (stereo * 32767).astype(np.int16)
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(stereo_pcm.tobytes())

            with mock.patch(
                "librosa.core.resample",
                return_value=np.array([0.25, 0.0, -0.25, 0.0], dtype=np.float32),
            ):
                pcm = wav_to_mono_pcm16_bytes(audio_path, target_sr=16000)

        decoded = np.frombuffer(pcm, dtype=np.int16)
        self.assertEqual(len(decoded), 4)
        np.testing.assert_array_equal(
            decoded,
            np.array([8191, 0, -8192, 0], dtype=np.int16),
        )


if __name__ == "__main__":
    unittest.main()
