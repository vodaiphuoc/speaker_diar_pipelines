from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np
import nemo.collections.asr as nemo_asr

class ASR_Offine_Wrapper(object):
    def __init__(
            self,
            model_id:str = "large-v3",
            device:str = "cuda",
            compute_type: str = "float16"
        ):
        model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type
        )
        self._model = BatchedInferencePipeline(model=model)

    def forward(
            self, 
            audio_array: np.ndarray, 
            speech_timestamps: list[dict[str, float]]
        ):
        r"""
        Args:
            - audio_array (np.ndarray): audio array of the audio file
        """
        segments, _ = self._model.transcribe(
            audio = audio_array,
            beam_size=len(speech_timestamps),
            vad_filter=False,
            clip_timestamps = speech_timestamps
        )

        return {
            segment.id: {
                "text": segment.text
            }
            for segment in segments
        }



# class ASR_Online_Wrapper(object):
#     r"""
#     Streaming ASR
#     """
#     def __init__(
#             self,
#             model_id:str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
#             device:str = "cuda",
#             compute_type: str = "float16"
#         ):
#         self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
#             model_name = model_id,
#             map_location=device
#         )


#     def forward(self, audio_array: np.ndarray, speech_timestamps: list[dict[str, float]]):
#         self.asr_model.conformer_stream_step()

#         # return {
#         #     segment.id: {
#         #         "text": segment.text
#         #     }
#         #     for segment in segments
#         # }



