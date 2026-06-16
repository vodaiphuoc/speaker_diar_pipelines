from abc import ABC, abstractmethod
from typing import Literal

import onnxruntime as ort
import torch


class MissmatchONNXModelKey(Exception):
    pass


class BaseOnnxRunner(ABC):
    def __init__(
        self,
        onnx_path: str,
        device: Literal["cpu", "cuda"],
    ):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # 0 = Verbose, 1 = Info, 2 = Warning, 3 = Error
        opts.log_verbosity_level = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        self.session = ort.InferenceSession(
            path_or_bytes=onnx_path, providers=providers, sess_options=opts
        )

        unknown_inputs = set(self.input_names) - set(
            [inp.name for inp in self.session.get_inputs()]
        )
        if unknown_inputs:
            raise MissmatchONNXModelKey(f"Unexpected keys: {unknown_inputs}")

        unknown_outputs = set(self.output_names) - set(
            [inp.name for inp in self.session.get_outputs()]
        )
        if unknown_outputs:
            raise MissmatchONNXModelKey(f"Unexpected keys: {unknown_outputs}")

    @abstractmethod
    def __call__(self, *args, **kwargs) -> list[torch.Tensor]:
        pass

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        pass
