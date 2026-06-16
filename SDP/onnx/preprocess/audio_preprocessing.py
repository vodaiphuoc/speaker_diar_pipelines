from typing import Literal

import torch

from SDP.onnx.base import BaseOnnxRunner


class AudioToMelSpectrogramPreprocessorOnnxRunner(BaseOnnxRunner):
    def __init__(
        self,
        onnx_path: str,
        device: Literal["cpu", "cuda"],
    ):
        super().__init__(onnx_path=onnx_path, device=device)

    def __call__(self, input_signal: torch.Tensor, length: torch.Tensor):
        (features, _) = self.session.run(
            output_names=self.output_names,
            input_feed={"input_signal": input_signal.numpy(), "length": length.numpy()},
        )
        return torch.from_numpy(features)

    @property
    def input_names(self) -> list[str]:
        return ["input_signal", "length"]

    @property
    def output_names(self):
        return ["processed_signal", "processed_length"]
