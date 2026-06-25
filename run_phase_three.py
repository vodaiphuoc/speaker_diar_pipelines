import json
import os
import tempfile
from pathlib import Path

from SDP import StreamingDiarizerOnnxService, wav_to_mono_pcm16_bytes


def stream_audio_file_bytes(
    file_path: str, chunk_duration_ms=160, target_sample_rate=16000
):
    """
    Reads a WAV file, resamples it to target_sample_rate (16kHz) if needed,
    and yields raw PCM bytes in chunks of chunk_duration_ms.
    """
    pcm_bytes = wav_to_mono_pcm16_bytes(file_path, target_sr=target_sample_rate)

    bytes_per_sample = 2
    samples_per_chunk = int(target_sample_rate * (chunk_duration_ms / 1000))
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    # Yield the chunks
    for ith, i in enumerate(range(0, len(pcm_bytes), bytes_per_chunk)):
        chunk = pcm_bytes[i : i + bytes_per_chunk]
        # Skip the final chunk if it's incomplete (optional, depending on your model requirements)
        if len(chunk) == bytes_per_chunk:
            yield ith, chunk


def _write_results_file(output_path, data):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            json.dump(data, temporary_file, ensure_ascii=False, indent=4)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def discover_audio_files(audio_dir):
    return sorted(Path(audio_dir).glob("*.wav"))


def initialize_results_file(output_path, audio_files):
    _write_results_file(
        output_path,
        [
            {
                "audio_file": str(audio_file),
                "results": None,
            }
            for audio_file in audio_files
        ],
    )


def _find_audio_result(data, audio_file):
    audio_file = str(audio_file)
    for audio_result in data:
        if audio_result["audio_file"] == audio_file:
            return audio_result
    raise ValueError(f"Audio file is not present in results: {audio_file}")


def set_audio_results(output_path, audio_file, results):
    output_path = Path(output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    _find_audio_result(data, audio_file)["results"] = results
    _write_results_file(output_path, data)


def update_event_results(events, output_path, audio_file):
    output_path = Path(output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    audio_result = _find_audio_result(data, audio_file)

    for event in events:
        audio_result["results"].append(
            {
                "start": event.start,
                "end": event.end,
                "text": None,
                "speaker": f"speaker_{event.speaker_id}",
            }
        )
        _write_results_file(output_path, data)


def create_streaming_service(manifest_path):
    return StreamingDiarizerOnnxService.from_manifest(
        manifest_path,
        device="cpu",
        enable_async_queue=True,
    )


def process_audio_files(
    audio_files,
    output_path,
    service_factory,
    audio_streamer=stream_audio_file_bytes,
):
    for audio_file in audio_files:
        try:
            set_audio_results(output_path, audio_file, [])
            service = service_factory(audio_file)
            stream_id = audio_file.name

            for _, audio_bytes in audio_streamer(audio_file):
                events = service.diarize(audio=audio_bytes, stream_id=stream_id)
                if events:
                    update_event_results(events, output_path, audio_file)

            final_events = service.flush(stream_id=stream_id)
            if final_events:
                update_event_results(final_events, output_path, audio_file)

            service.drain_events()
        except Exception:
            set_audio_results(output_path, audio_file, None)


def main():
    manifest_path = ".onnx_ckpt/diar/diarization_artifact.json"
    audio_files = discover_audio_files("data/part1")
    output_path = Path("data_results/part1/results_phase3.json")

    initialize_results_file(output_path, audio_files)
    process_audio_files(
        audio_files,
        output_path,
        service_factory=lambda _: create_streaming_service(manifest_path),
    )


if __name__ == "__main__":
    main()
