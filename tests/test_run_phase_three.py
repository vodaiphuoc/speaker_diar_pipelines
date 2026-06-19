import importlib.util
import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).parents[1] / "run_phase_three.py"


def load_run_phase_three_module():
    fake_sdp = types.ModuleType("SDP")
    fake_sdp.StreamingDiarizerOnnxService = lambda **kwargs: object()
    fake_sdp.load_encoder_modules_config = lambda path: object()
    fake_sdp.load_preprocessor_config = lambda path: object()
    fake_sdp.load_sortformer_modules_config = lambda path: object()

    spec = importlib.util.spec_from_file_location("run_phase_three_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)

    original_sdp = sys.modules.get("SDP")
    sys.modules["SDP"] = fake_sdp
    try:
        spec.loader.exec_module(module)
    finally:
        if original_sdp is None:
            del sys.modules["SDP"]
        else:
            sys.modules["SDP"] = original_sdp
    return module


class RunPhaseThreeTest(unittest.TestCase):
    def test_import_does_not_start_audio_processing(self):
        fake_sdp = types.ModuleType("SDP")
        fake_sdp.StreamingDiarizerOnnxService = lambda **kwargs: object()
        fake_sdp.load_encoder_modules_config = lambda path: object()
        fake_sdp.load_preprocessor_config = lambda path: object()
        fake_sdp.load_sortformer_modules_config = lambda path: object()

        spec = importlib.util.spec_from_file_location("run_phase_three_test", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)

        original_sdp = sys.modules.get("SDP")
        sys.modules["SDP"] = fake_sdp
        try:
            with patch(
                "wave.open", side_effect=AssertionError("audio processing started")
            ):
                spec.loader.exec_module(module)
        finally:
            if original_sdp is None:
                del sys.modules["SDP"]
            else:
                sys.modules["SDP"] = original_sdp

    def test_initialize_results_file_overwrites_existing_content(self):
        module = load_run_phase_three_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results_phase3.json"
            output_path.write_text('{"old": "content"}', encoding="utf-8")

            module.initialize_results_file(
                output_path,
                [
                    "data/part1/b.wav",
                    "data/part1/a.wav",
                ],
            )

            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [
                    {
                        "audio_file": "data/part1/b.wav",
                        "results": None,
                    },
                    {
                        "audio_file": "data/part1/a.wav",
                        "results": None,
                    },
                ],
            )

    def test_discover_audio_files_returns_sorted_wav_files(self):
        module = load_run_phase_three_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_dir = Path(temp_dir)
            (audio_dir / "b.wav").touch()
            (audio_dir / "a.WAV").touch()
            (audio_dir / "a.wav").touch()
            (audio_dir / "notes.txt").touch()
            (audio_dir / "nested").mkdir()
            (audio_dir / "nested" / "c.wav").touch()

            audio_files = module.discover_audio_files(audio_dir)

            self.assertEqual(
                audio_files,
                [
                    audio_dir / "a.wav",
                    audio_dir / "b.wav",
                ],
            )

    def test_stream_audio_file_bytes_accepts_path_object(self):
        module = load_run_phase_three_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "example.wav"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 2560)

            chunks = list(module.stream_audio_file_bytes(audio_path))

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(len(chunks[0][1]), 5120)

    def test_update_event_results_persists_each_event(self):
        module = load_run_phase_three_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results_phase3.json"
            module.initialize_results_file(
                output_path,
                [
                    "data/part1/first.wav",
                    "data/part1/second.wav",
                ],
            )
            module.set_audio_results(output_path, "data/part1/second.wav", [])
            first_event = SimpleNamespace(start=0.08, end=1.2, speaker_id=0)
            second_event = SimpleNamespace(start=1.28, end=2.4, speaker_id=1)

            def events():
                yield first_event
                persisted = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    persisted[1]["results"],
                    [
                        {
                            "start": 0.08,
                            "end": 1.2,
                            "text": None,
                            "speaker": "speaker_0",
                        }
                    ],
                )
                yield second_event

            module.update_event_results(
                events(),
                output_path,
                "data/part1/second.wav",
            )

            persisted = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted,
                [
                    {
                        "audio_file": "data/part1/first.wav",
                        "results": None,
                    },
                    {
                        "audio_file": "data/part1/second.wav",
                        "results": [
                            {
                                "start": 0.08,
                                "end": 1.2,
                                "text": None,
                                "speaker": "speaker_0",
                            },
                            {
                                "start": 1.28,
                                "end": 2.4,
                                "text": None,
                                "speaker": "speaker_1",
                            },
                        ],
                    }
                ],
            )

    def test_process_audio_files_continues_after_failure_with_fresh_services(self):
        module = load_run_phase_three_module()

        class FakeService:
            def __init__(self, audio_file):
                self.audio_file = audio_file

            def diarize(self, audio, stream_id):
                if self.audio_file.name == "first.wav":
                    raise RuntimeError("broken audio")
                return [SimpleNamespace(start=0.0, end=1.0, speaker_id=1)]

            def flush(self, stream_id):
                return []

            def drain_events(self):
                return []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results_phase3.json"
            audio_files = [
                Path("data/part1/first.wav"),
                Path("data/part1/second.wav"),
            ]
            created_for = []

            def service_factory(audio_file):
                created_for.append(audio_file)
                return FakeService(audio_file)

            def audio_streamer(audio_file):
                yield 0, b"audio"

            module.initialize_results_file(output_path, audio_files)
            module.process_audio_files(
                audio_files,
                output_path,
                service_factory,
                audio_streamer,
            )

            self.assertEqual(created_for, audio_files)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                [
                    {
                        "audio_file": "data/part1/first.wav",
                        "results": None,
                    },
                    {
                        "audio_file": "data/part1/second.wav",
                        "results": [
                            {
                                "start": 0.0,
                                "end": 1.0,
                                "text": None,
                                "speaker": "speaker_1",
                            }
                        ],
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()
