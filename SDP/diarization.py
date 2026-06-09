# Speaker Diarization:
#    - pyannote
#    - diar_sortformer_4spk-v1
#    - diar_streaming_sortformer_4spk-v2.1
# ASR:
#    - zipformer
#    -nvidia/nemotron-3.5-asr-streaming-0.6b



import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from nemo.collections.asr.models import SortformerEncLabelModel
import torch
import numpy as np


class Speaker_Diarization_Offline(object):
    def __init__(self, device:str = "cuda"):
        
        # self._diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1", map_location = device)
        self._diar_model = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2.1",
            map_location = torch.device(device)
        )
        self._diar_model.eval()
        self._diar_model.sortformer_modules.chunk_len = 340
        self._diar_model.sortformer_modules.chunk_right_context = 40
        self._diar_model.sortformer_modules.fifo_len = 40
        self._diar_model.sortformer_modules.spkcache_update_period = 300
        self._diar_model.sortformer_modules.log = True

    def forward(self, audio_array: np.ndarray, sampling_rate: int)->list[dict[str,str|float]]:
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

