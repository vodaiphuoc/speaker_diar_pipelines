from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from SDP.qwen3_asr.session import Qwen3ASRModalConfig


class Qwen3ASRInProcessActor:
    """Qwen3 streaming actor for processes that already run on a Modal GPU."""

    def __init__(
        self,
        config: Qwen3ASRModalConfig | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        from qwen_asr import Qwen3ASRModel

        self.config = config or Qwen3ASRModalConfig.from_env()
        self.cache_root = Path(cache_root or "/app/.modal_ci/qwen3_asr_cache")
        self._configure_cache_env()
        self._asr = Qwen3ASRModel.LLM(
            model=self.config.model_name,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_new_tokens=self.config.max_new_tokens,
        )
        self._state = self._asr.init_streaming_state(
            unfixed_chunk_num=self.config.unfixed_chunk_num,
            unfixed_token_num=self.config.unfixed_token_num,
            chunk_size_sec=self.config.chunk_size_sec,
        )

    def transcribe(self, samples: np.ndarray) -> dict:
        wav = np.asarray(samples, dtype=np.float32)
        self._asr.streaming_transcribe(wav, self._state)
        return {
            "language": getattr(self._state, "language", None),
            "text": getattr(self._state, "text", "") or "",
        }

    def finish(self) -> dict:
        self._asr.finish_streaming_transcribe(self._state)
        return {
            "language": getattr(self._state, "language", None),
            "text": getattr(self._state, "text", "") or "",
        }

    def _configure_cache_env(self) -> None:
        os.environ.setdefault("HF_HOME", str(self.cache_root / "huggingface"))
        os.environ.setdefault(
            "TRANSFORMERS_CACHE",
            str(self.cache_root / "huggingface" / "transformers"),
        )
        os.environ.setdefault("VLLM_CACHE_ROOT", str(self.cache_root / "vllm"))
        os.environ.setdefault("XDG_CACHE_HOME", str(self.cache_root / "xdg"))
        for cache_dir in (
            self.cache_root / "huggingface",
            self.cache_root / "huggingface" / "transformers",
            self.cache_root / "vllm",
            self.cache_root / "xdg",
        ):
            cache_dir.mkdir(parents=True, exist_ok=True)
