import wave

import numpy as np
from scipy.signal import resample_poly

from SDP import (
    StreamingDiarizerOnnxService,
    load_encoder_modules_config,
    load_preprocessor_config,
    load_sortformer_modules_config,
)


def stream_audio_file_bytes(
    file_path: str, chunk_duration_ms=160, target_sample_rate=16000
):
    """
    Reads a WAV file, resamples it to target_sample_rate (16kHz) if needed,
    and yields raw PCM bytes in chunks of chunk_duration_ms.
    """
    with wave.open(file_path, "rb") as wav_file:
        original_sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()

        raw_data = wav_file.readframes(num_frames)

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    audio_data = np.frombuffer(raw_data, dtype=dtype)

    # multi-channel audio stereo to mono by averaging channels
    if num_channels > 1:
        audio_data = audio_data.reshape(-1, num_channels)
        audio_data = audio_data.mean(axis=1).astype(dtype)

    # Resample if the rate doesn't match 16,000 Hz
    if original_sample_rate != target_sample_rate:
        print(
            f"Resampling audio from {original_sample_rate}Hz to {target_sample_rate}Hz..."
        )
        audio_data = resample_poly(
            audio_data, target_sample_rate, original_sample_rate
        ).astype(dtype)

    # Convert back to raw bytes for streaming simulation
    resampled_bytes = audio_data.tobytes()

    bytes_per_sample = dtype().itemsize
    samples_per_chunk = int(target_sample_rate * (chunk_duration_ms / 1000))
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    print(f"Streaming chunks: {bytes_per_chunk} bytes per {chunk_duration_ms}ms chunk.")

    # Yield the chunks
    for ith, i in enumerate(range(0, len(resampled_bytes), bytes_per_chunk)):
        chunk = resampled_bytes[i : i + bytes_per_chunk]
        # Skip the final chunk if it's incomplete (optional, depending on your model requirements)
        if len(chunk) == bytes_per_chunk:
            yield ith, chunk
        else:
            print(f"error in ith {ith}: {len(chunk)} vs {bytes_per_chunk}")


def print_events(events, prefix: str = "event"):
    for event in events:
        data = {
            "start": event.start,
            "end": event.end,
            "speaker": f"speaker_{event.speaker_id}",
        }
        print(f"{prefix}: {data}")


config_path = "configs/pretrained_config.yaml"

s = StreamingDiarizerOnnxService(
    modal_ckpt_path=".onnx_ckpt/model.onnx",
    preprocessor_ckpt_path=".onnx_ckpt/preprocessor.onnx",
    device="cpu",
    encoder_config=load_encoder_modules_config(config_path),
    sortformer_config=load_sortformer_modules_config(config_path),
    preprocessor_config=load_preprocessor_config(config_path),
    enable_async_queue=True,
)

stream_id = "bacsidatnhkhoavitadoc_1"

for chunk_idx, audio_bytes in stream_audio_file_bytes(f"data/part1/{stream_id}"):
    events = s.diarize(audio=audio_bytes, stream_id=stream_id)
    if events:
        print_events(events, prefix=f"chunk ith: {chunk_idx}")

final_events = s.flush(stream_id=stream_id)
if final_events:
    print_events(final_events, prefix="flush")

queued_events = s.drain_events()
print(f"total queued diarization events: {len(queued_events)}")
