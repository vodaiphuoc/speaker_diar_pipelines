import math
from typing import Literal

import numpy as np
import torch

from SDP.onnx.diarization.nemo_vad_utils import ts_vad_post_processing
from SDP.onnx.diarization.post_processing import StreamingDiarizationPostProcessor
from SDP.onnx.diarization.streaming import SortformerONNXRunner
from SDP.onnx.diarization.types import (
    EncoderModuleConfig,
    PostProcessingParams,
    PreProcessorConfig,
    SortformerModuleConfig,
)
from SDP.onnx.preprocess.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorOnnxRunner,
)
from SDP.onnx.preprocess.feature_buffer import CacheFeatureBufferer


class StreamingDiarizerOnnxService(object):
    def __init__(
        self,
        modal_ckpt_path: str,
        preprocessor_ckpt_path: str,
        device: Literal["cpu", "cuda"],
        encoder_config: EncoderModuleConfig,
        sortformer_config: SortformerModuleConfig,
        preprocessor_config: PreProcessorConfig,
        post_processing_config: PostProcessingParams = PostProcessingParams(),
        frame_len_in_secs: float = 0.08,
        sample_rate: int = 16000,
        left_offset: int = 8,
        right_offset: int = 8,
    ):
        self.frame_len_in_secs = frame_len_in_secs
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.chunk_size = sortformer_config.chunk_len
        self.device = torch.device(device)
        self.encoder_config = encoder_config
        self.sortformer_config = sortformer_config
        self.post_processing_config = post_processing_config

        self._diarizer = SortformerONNXRunner(
            onnx_path=modal_ckpt_path,
            device=device,
            sortformer_config=sortformer_config,
        )

        preprocessor = AudioToMelSpectrogramPreprocessorOnnxRunner(
            onnx_path=preprocessor_ckpt_path, device=device
        )

        self.buffer_size_in_secs = (
            sortformer_config.chunk_len * self.frame_len_in_secs
            + (self.left_offset + self.right_offset) * 0.01
        )

        self._feature_bufferer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=sortformer_config.chunk_len * self.frame_len_in_secs,
            preprocessor_cfg=preprocessor_config,
            preprocessor=preprocessor,
            device=self.device,
        )

        self.streaming_state = self._diarizer.init_streaming_state(
            batch_size=1, device=self.device
        )

        self._post_diar_processor = StreamingDiarizationPostProcessor(
            cfg_vad_params=self.post_processing_config,
            num_spks=4,
        )

    def diarize(self, audio: bytes, stream_id: str = "default"):
        r"""
        Main entrypoint to be call from websocket endpoint
        or processing each chunk audio from a stream
        """
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        self._feature_bufferer.update(audio_array)

        features = self._feature_bufferer.get_feature_buffer()
        print("features: ", features.shape)
        feature_buffers = features.unsqueeze(0)  # add batch dimension
        feature_buffers = feature_buffers.transpose(
            1, 2
        )  # [batch, feature, time] -> [batch, time, feature]
        feature_buffer_lens = torch.tensor(
            [feature_buffers.shape[1]], device=self.device
        )

        print("feature_buffers input: ", feature_buffers.shape)
        (spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths) = (
            self._diarizer(
                chunk=feature_buffers,
                chunk_lengths=feature_buffer_lens,
                spkcache=self.streaming_state.spkcache,
                spkcache_lengths=self.streaming_state.spkcache_lengths,
                fifo=self.streaming_state.fifo,
                fifo_lengths=self.streaming_state.fifo_lengths,
            )
        )

        (self.streaming_state, chunk_preds) = self._diarizer.streaming_update_async(
            streaming_state=self.streaming_state,
            chunk=chunk_pre_encode_embs,
            chunk_lengths=chunk_pre_encode_lengths,
            preds=spkcache_fifo_chunk_preds,
            lc=round(self.left_offset / self.encoder_config.subsampling_factor),
            rc=math.ceil(self.right_offset / self.encoder_config.subsampling_factor),
        )

        diar_result = chunk_preds[:, -self.chunk_size :, :].clone()
        print(diar_result.shape)
        return self._post_diar_processor.process_chunk(diar_result[0])

    def _diarize_output_processing(self, outputs: torch.Tensor):
        """
        Processes the diarization outputs and generates RTTM (Real-time Text Markup) files.
        TODO: Currently, this function is not included in mixin test because of
              `ts_vad_post_processing` function.
              (1) Implement a test-compatible function
              (2) `vad_utils.py` has `predlist_to_timestamps` function that is close to this function.
                  Needs to consolute differences and implement the test-compatible function.

        Args:
            outputs (torch.Tensor): Sorted tensor containing Sigmoid values for predicted speaker labels.
                Shape: (batch_size, diar_frame_count, num_speakers)
            uniq_ids (List[str]): List of unique identifiers for each audio file.
            diarcfg (DiarizeConfig): Configuration object for diarization.

        Returns:
            diar_output_lines_list (List[List[str]]): A list of lists, where each inner list contains
                                                      the RTTM lines for a single audio file.
            preds_list (List[torch.Tensor]): A list of tensors containing the diarization outputs
                                             for each audio file.
        """
        preds_list = []
        if outputs.shape[0] == 1:  # batch size = 1
            preds_list.append(outputs)
        else:
            preds_list.extend(torch.split(outputs, [1] * outputs.shape[0]))

        uniq_ids = ["121u3ou3r2o832"]
        for sample_idx, uniq_id in enumerate(uniq_ids):
            offset = 0
            speaker_assign_mat = preds_list[sample_idx].squeeze(dim=0)
            print(speaker_assign_mat.shape)
            speaker_timestamps = [[] for _ in range(speaker_assign_mat.shape[-1])]
            print("speaker_assign_mat.shape[-1]: ", speaker_assign_mat.shape[-1])
            for spk_id in range(speaker_assign_mat.shape[-1]):
                ts_mat = ts_vad_post_processing(
                    speaker_assign_mat[:, spk_id],
                    cfg_vad_params=self.post_processing_config,
                    unit_10ms_frame_count=int(self.encoder_config.subsampling_factor),
                    bypass_postprocessing=False,
                )
                print("ts_mat:", ts_mat)
                ts_mat = ts_mat + offset
                ts_seg_raw_list = ts_mat.tolist()
                ts_seg_list = [
                    [round(stt, 2), round(end, 2)] for (stt, end) in ts_seg_raw_list
                ]
                speaker_timestamps[spk_id].extend(ts_seg_list)
            print(speaker_timestamps)
