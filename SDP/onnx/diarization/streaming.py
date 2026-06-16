from typing import Literal, Unpack

import numpy as np
import torch

from SDP.onnx.base import BaseOnnxRunner
from SDP.onnx.diarization.nemo_sortformer import SortformerModules
from SDP.onnx.diarization.types import (
    SortformerModuleConfig,
    StreamingInputArgs,
    StreamingSortformerState,
)


class SortformerONNXRunner(BaseOnnxRunner, SortformerModules):
    def __init__(
        self,
        onnx_path: str,
        device: Literal["cpu", "cuda"],
        sortformer_config: SortformerModuleConfig,
    ):
        BaseOnnxRunner.__init__(self, onnx_path=onnx_path, device=device)
        SortformerModules.__init__(self, sortformer_config=sortformer_config)

        self._sortformer_config = sortformer_config

    @property
    def input_names(self) -> list[str]:
        return list(StreamingInputArgs.__annotations__.keys())

    @property
    def output_names(self):
        return [
            "spkcache_fifo_chunk_preds",
            "chunk_pre_encode_embs",
            "chunk_pre_encode_lengths",
        ]

    def __call__(self, **kwargs: Unpack[StreamingInputArgs]) -> list[torch.Tensor]:
        r"""
        Call method for running onnx model
        """
        allowed = StreamingInputArgs.__annotations__.keys()
        unknown = set(kwargs) - set(allowed)
        if unknown:
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        for k, v in kwargs.items():
            kwargs[k] = v.numpy()

        outputs = self.session.run(self.output_names, kwargs)
        return [torch.from_numpy(ele) for ele in outputs]

    def init_streaming_state(
        self, batch_size: int, device: torch.device
    ) -> StreamingSortformerState:
        """
        Initializes StreamingSortformerState with empty tensors or zero-valued tensors.

        Args:
            batch_size (int): Batch size for tensors in streaming state
            device (torch.device): Device for tensors in streaming state

        Returns:
            streaming_state (SortformerStreamingState): initialized streaming state
        """
        return StreamingSortformerState(
            spkcache=torch.zeros(
                (
                    batch_size,
                    self._sortformer_config.spkcache_len,
                    self._sortformer_config.fc_d_model,
                ),
                device=device,
            ),
            spkcache_lengths=torch.zeros(
                (batch_size,), dtype=torch.long, device=device
            ),
            spkcache_preds=torch.zeros(
                (
                    batch_size,
                    self._sortformer_config.spkcache_len,
                    self._sortformer_config.num_spks,
                ),
                device=device,
            ),
            fifo=torch.zeros(
                (
                    batch_size,
                    self._sortformer_config.fifo_len,
                    self._sortformer_config.fc_d_model,
                ),
                device=device,
            ),
            fifo_lengths=torch.zeros((batch_size,), dtype=torch.long, device=device),
            fifo_preds=None,
            spk_perm=None,
            mean_sil_emb=torch.zeros(
                (batch_size, self._sortformer_config.fc_d_model), device=device
            ),
            n_sil_frames=torch.zeros((batch_size,), dtype=torch.long, device=device),
        )
