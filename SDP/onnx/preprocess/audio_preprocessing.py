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
        features, _ = self.process(input_signal, length)
        return features

    def process(
        self, input_signal: torch.Tensor, length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, processed_length = self.session.run(
            output_names=self.output_names,
            input_feed={
                "input_signal": input_signal.detach().cpu().numpy(),
                "length": length.detach().cpu().numpy(),
            },
        )
        return torch.from_numpy(features), torch.from_numpy(processed_length)

    @property
    def input_names(self) -> list[str]:
        return ["input_signal", "length"]

    @property
    def output_names(self):
        return ["processed_signal", "processed_length"]
