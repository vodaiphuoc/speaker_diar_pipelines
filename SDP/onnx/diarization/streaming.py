import numpy as np
import torch
from SDP.onnx.diarization.types import StreamingInputArgs, StreamingSortformerState
import onnxruntime as ort
from typing import Unpack, Literal
from SDP.onnx.diarization.types import SortformerModuleConfig

class MissmatchONNXModelKey(Exception):
    pass


class SortformerONNXRunner:
    def __init__(
            self, 
            onnx_path: str, 
            device: Literal['cpu','cuda'], 
            sortformer_config: SortformerModuleConfig
        ):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.torch_device = torch.device
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers
        )

        allowed = StreamingInputArgs.__annotations__.keys()
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        unknown = set(self.input_names) - set(allowed)
        if unknown:
            raise MissmatchONNXModelKey(f"Unexpected keys: {unknown}")

        self._sortformer_config = sortformer_config

        
        # self.caches = {}

    def _get_empty_caches(self, batch_size=1):
        r"""
        Constructs the zeroed-out initial states for the FastConformer 
        and the Speaker caches. Shapes depend on your specific model config.
        Usually, the NeMo ONNX graph handles dynamic zero-init if you pass empty arrays.
        """
        # Note: The exact names and shapes of these caches depend on the NeMo version
        # used during export. Typically, they look like 'fifo_cache', 'spk_cache', etc.
        # Check self.input_names to map them correctly.
        
        cache_inputs = {}
        for name in self.input_names:
            if "audio" in name or "signal" in name:
                continue # Skip the main audio input
            
            # Initialize empty caches (shapes dictated by ONNX dynamic axes)
            # Example standard initialization for NeMo caches:
            cache_inputs[name] = np.zeros((batch_size, 0, 0), dtype=np.float32) 
            
        return cache_inputs

    def onnx_forward(self,**kwargs: Unpack[StreamingInputArgs]):
        allowed = StreamingInputArgs.__annotations__.keys()
        unknown = set(kwargs) - set(allowed)
        if unknown:
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        outputs = self.session.run(self.output_names, kwargs)

    
    def init_streaming_state(self, batch_size: int = 1, device: torch.device = None):
        """
        Initializes StreamingSortformerState with empty tensors or zero-valued tensors.

        Args:
            batch_size (int): Batch size for tensors in streaming state
            async_streaming (bool): True for asynchronous update, False for synchronous update
            device (torch.device): Device for tensors in streaming state

        Returns:
            streaming_state (SortformerStreamingState): initialized streaming state
        """
        streaming_state = StreamingSortformerState()
        
        streaming_state.spkcache = torch.zeros(
            (batch_size, self._sortformer_config.spkcache_len, self._sortformer_config.fc_d_model), 
            device=device
        )
        
        streaming_state.spkcache_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        
        streaming_state.spkcache_preds = torch.zeros(
            (batch_size, self._sortformer_config.spkcache_len, self._sortformer_config.num_spks),
            device=device
        )
        
        streaming_state.fifo = torch.zeros(
            (batch_size, self._sortformer_config.fifo_len, self._sortformer_config.fc_d_model), 
            device=device
        )
        streaming_state.fifo_lengths = torch.zeros((batch_size,), dtype=torch.long, device=device)
        
        streaming_state.mean_sil_emb = torch.zeros(
            (batch_size, self._sortformer_config.fc_d_model), 
            device=device
        )
        streaming_state.n_sil_frames = torch.zeros((batch_size,), dtype=torch.long, device=device)
        return streaming_state

