# Speaker Diarization:
#    - pyannote
#    - diar_sortformer_4spk-v1
#    - diar_streaming_sortformer_4spk-v2.1
# ASR:
#    - zipformer (TODO)
#    -nvidia/nemotron-3.5-asr-streaming-0.6b


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from nemo.collections.asr.models import SortformerEncLabelModel
import torch
import numpy as np
from typing import List
import types


class Speaker_Diarization_Offline(object):
    r"""
    Use of offline inference for full one audio 
    """
    def __init__(
            self,
            model_id: str = "nvidia/diar_sortformer_4spk-v1",
            device:str = "cuda"
        ):
        self._diar_model = SortformerEncLabelModel.from_pretrained(
            model_id,
            map_location = torch.device(device)
        )
        self._diar_model.eval()

    def forward(
            self,
            audio_array: np.ndarray,
            sampling_rate: int
        )->list[dict[str,str|float]]:
        r"""
        Run inference for one audio file
        Args:
            - audio_array (np.ndarray): audio data as in numpy array shape
            - sampling_rate (int): sampling of the audio
        
        Returns:
            - list of dictionary with keys:
                - "start" (float): start time
                - "end" (float): end time
                - "speaker_str" (str): name of speaker
            which is sorted by "start" key
        """
        predicted_segments: list[str] = self._diar_model.diarize(
            audio=audio_array, 
            sample_rate = sampling_rate,
            batch_size=1,
            include_tensor_outputs=False,
            verbose = False
        )[0]

        results = []
        
        for ele in predicted_segments:
            start_str, end_str, speaker_str = ele.split()
            start_timestamp, end_timestamp = float(start_str), float(end_str)
            assert start_timestamp < end_timestamp, f"Found start_timestamp: {start_timestamp}, end_timestamp: {end_timestamp}"
            results.append({
                "start": start_timestamp,
                "end": end_timestamp,
                "speaker_str": speaker_str,
            })
        
        assert len(results) != 0, f"Found empty results"
        return sorted(results, key=lambda x: x['start'])

class Speaker_Diarization_Online(object):
    r"""
    Streaming inference which take chunk of audio input
    TODO:
        - export model to onnx format (Done)
        - inference with onnx format, not in pytorch
        - create chunk audio generator as in example, not in pipeline
    """
    def __init__(
            self,
            model_id: str = "nvidia/diar_sortformer_4spk-v1",
            device:str = "cuda"
        ):
        self._diar_model = SortformerEncLabelModel.from_pretrained(
            model_id,
            map_location = torch.device(device)
        )
        self._diar_model.eval()
        self._diar_model.sortformer_modules.chunk_len = 340
        self._diar_model.sortformer_modules.chunk_right_context = 40
        self._diar_model.sortformer_modules.fifo_len = 40
        self._diar_model.sortformer_modules.spkcache_update_period = 300
        self._diar_model.sortformer_modules.log = True

    def custom_export(self, onnx_checkpoint: str = "/mydata/test.onnx"):
        r"""
        Export model to onnx.
        Require custom/modify the original implement in Nemo to prevent
        shape missmatch

        """
    
        def custom_streaming_input_examples(self):
                """Input tensor examples for exporting streaming version of model"""
                print("run custom example streaming input")
                batch_size = 4
                
                # modify shape parameters to match current inputs
                chunk = torch.rand([batch_size, 1008, 128]).to(self.device)
                chunk_lengths = torch.tensor([1008] * batch_size).to(self.device)
                spkcache = torch.randn([batch_size, 188, 512]).to(self.device)
                spkcache_lengths = torch.tensor([188] * batch_size).to(self.device)
                fifo = torch.randn([batch_size, 124, 512]).to(self.device)
                fifo_lengths = torch.tensor([124] * batch_size).to(self.device)
                return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths
            
        def custom_concat_and_pad(embs: List[torch.Tensor], lengths: List[torch.Tensor]):
            """
            Concatenates lengths[i] first embeddings of embs[i], and pads the rest elements with zeros.

            Args:
                embs: List of embeddings Tensors of (batch_size, n_frames, emb_dim) shape
                lengths: List of lengths Tensors of (batch_size,) shape

            Returns:
                output: concatenated embeddings Tensor of (batch_size, n_frames, emb_dim) shape
                total_lengths: output lengths Tensor of (batch_size,) shape
            """
            print('run custom_concat_and_pad')
            # Error handling for mismatched list lengths
            if len(embs) != len(lengths):
                raise ValueError(
                    f"Length lists must have the same length, but got len(embs) - {len(embs)} "
                    f"and len(lengths) - {len(lengths)}."
                )
            # Handle empty lists
            if len(embs) == 0 or len(lengths) == 0:
                raise ValueError(
                    f"Cannot concatenate empty lists of embeddings or lengths: embs - {len(embs)}, lengths - {len(lengths)}"
                )

            device, dtype = embs[0].device, embs[0].dtype
            batch_size, emb_dim = embs[0].shape[0], embs[0].shape[2]

            total_lengths = torch.sum(torch.stack(lengths), dim=0)
            sig_length = total_lengths.max().item()

            output = torch.zeros(batch_size, sig_length, emb_dim, device=device, dtype=dtype)
            start_indices = torch.zeros(batch_size, dtype=torch.int64, device=device)

            for emb, length in zip(embs, lengths):
                end_indices = start_indices + length
                for batch_idx in range(batch_size):
                    
                    s = start_indices[batch_idx].view(())
                    e = end_indices[batch_idx].view(())
                    l = length[batch_idx].view(())
                    output[batch_idx, s : e] = emb[
                        batch_idx, : l
                    ]
                    
                start_indices = end_indices

            return output, total_lengths

        self._diar_model.streaming_input_examples = types.MethodType(custom_streaming_input_examples, self._diar_model)
        self._diar_model.concat_and_pad_script = torch.jit.script(custom_concat_and_pad)

        self._diar_model.streaming_export(onnx_checkpoint)
