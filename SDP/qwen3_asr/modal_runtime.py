from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from SDP.qwen3_asr.session import Qwen3ASRModalConfig

try:
    import modal
except ImportError as exc:  # pragma: no cover - exercised only without modal installed.
    modal = None
    _MODAL_IMPORT_ERROR = exc
else:
    _MODAL_IMPORT_ERROR = None


DEFAULT_CONFIG = Qwen3ASRModalConfig.from_env()
CACHE_ROOT = Path("/mnt/qwen3_asr_cache")

if modal is not None:
    qwen3_cache_volume = modal.Volume.from_name(
        DEFAULT_CONFIG.volume_name,
        create_if_missing=True,
    )

    qwen3_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install("numpy", "soundfile", "qwen-asr[vllm]")
    )

    app = modal.App(DEFAULT_CONFIG.app_name)

    @app.cls(
        image=qwen3_image,
        gpu=DEFAULT_CONFIG.gpu,
        timeout=60 * 60,
        volumes={str(CACHE_ROOT): qwen3_cache_volume},
    )
    class Qwen3ASRStreamingActor:
        @modal.enter()
        def load(self) -> None:
            from qwen_asr import Qwen3ASRModel

            _configure_cache_env()
            self._asr = Qwen3ASRModel.LLM(
                model=DEFAULT_CONFIG.model_name,
                gpu_memory_utilization=DEFAULT_CONFIG.gpu_memory_utilization,
                max_new_tokens=DEFAULT_CONFIG.max_new_tokens,
            )
            self._state = self._asr.init_streaming_state(
                unfixed_chunk_num=DEFAULT_CONFIG.unfixed_chunk_num,
                unfixed_token_num=DEFAULT_CONFIG.unfixed_token_num,
                chunk_size_sec=DEFAULT_CONFIG.chunk_size_sec,
            )

        @modal.method()
        def transcribe(self, samples: list[float]) -> dict:
            wav = np.asarray(samples, dtype=np.float32)
            self._asr.streaming_transcribe(wav, self._state)
            return {
                "language": getattr(self._state, "language", None),
                "text": getattr(self._state, "text", "") or "",
            }

        @modal.method()
        def finish(self) -> dict:
            self._asr.finish_streaming_transcribe(self._state)
            qwen3_cache_volume.commit()
            return {
                "language": getattr(self._state, "language", None),
                "text": getattr(self._state, "text", "") or "",
            }


class Qwen3ASRModalActorHandle:
    def __init__(self, actor) -> None:
        self._actor = actor

    def transcribe(self, samples: np.ndarray) -> dict:
        return self._actor.transcribe.remote(
            np.asarray(samples, dtype=np.float32).tolist()
        )

    def finish(self) -> dict:
        return self._actor.finish.remote()


def create_qwen3_asr_modal_actor(
    config: Qwen3ASRModalConfig | None = None,
) -> Qwen3ASRModalActorHandle:
    if modal is None:
        raise RuntimeError("modal is required for Qwen3-ASR remote execution") from (
            _MODAL_IMPORT_ERROR
        )
    if config is not None and config != DEFAULT_CONFIG:
        _validate_supported_runtime_config(config)
    return Qwen3ASRModalActorHandle(Qwen3ASRStreamingActor())


def _validate_supported_runtime_config(config: Qwen3ASRModalConfig) -> None:
    static_fields = (
        "model_name",
        "app_name",
        "volume_name",
        "gpu",
        "gpu_memory_utilization",
        "max_new_tokens",
        "unfixed_chunk_num",
        "unfixed_token_num",
        "chunk_size_sec",
    )
    mismatches = [
        field
        for field in static_fields
        if getattr(config, field) != getattr(DEFAULT_CONFIG, field)
    ]
    if mismatches:
        raise ValueError(
            "Qwen3 Modal runtime config is fixed when the module is imported; "
            f"set SDP_QWEN3_ASR_* env vars before import for {mismatches}"
        )


def _configure_cache_env() -> None:
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        str(CACHE_ROOT / "huggingface" / "transformers"),
    )
    os.environ.setdefault("VLLM_CACHE_ROOT", str(CACHE_ROOT / "vllm"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
    for cache_dir in (
        CACHE_ROOT / "huggingface",
        CACHE_ROOT / "huggingface" / "transformers",
        CACHE_ROOT / "vllm",
        CACHE_ROOT / "xdg",
    ):
        cache_dir.mkdir(parents=True, exist_ok=True)
