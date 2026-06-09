from SDP.diarization import Speaker_Diarization_Offline
from SDP.asr import ASR_Offine_Wrapper
from SDP.utils import decode_audio
import logging


logger = logging.getLogger(__name__)

class Pipeline(object):
    def __init__(self):
        self._sd = Speaker_Diarization_Offline()
        self._asr = ASR_Offine_Wrapper()

    def forward(self, audio_path:str = "examples/part1/bacsidatnhkhoavitadoc_1.wav", sampling_rate: int = 16000):
        try:
            # 1. read audio
            audio = decode_audio(input_file = audio_path, sampling_rate = sampling_rate)
            
            # 2. speaker diarization
            segments = self._sd.forward(audio_array = audio, sampling_rate = sampling_rate)

            # # 3. asr
            # asr_results = self._asr.forward(
            #     audio, 
            #     speech_timestamps= [
            #         {
            #             "start": ele["start"],
            #             "end": ele["end"]
            #         }
            #         for ele in segments
            #     ]
            # )

            # return [
            #     {   
            #         "start": segment["start"],
            #         "end": segment['end'],
            #         "text": asr_results[_ith]["text"],
            #         "speaker": segment['speaker_str']
            #     }
            #     for _ith, segment in enumerate(segments, start=1)
            # ]
            return []
        except Exception as e:
            logger.error(f"Error at {audio_path},  detail: {e}")
            return None
