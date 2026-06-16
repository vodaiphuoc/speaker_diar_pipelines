from dataclasses import dataclass
from typing import TypedDict

import torch


@dataclass
class EncoderModuleConfig:
    _target_: str
    feat_in: int
    feat_out: int
    n_layers: int
    d_model: int
    subsampling: str
    subsampling_factor: int
    subsampling_conv_channels: int
    causal_downsampling: bool
    ff_expansion_factor: int
    self_attention_model: str
    n_heads: int
    att_context_size: list[int]
    att_context_style: str
    xscaling: bool
    untie_biases: bool
    pos_emb_max_len: int
    conv_kernel_size: int
    conv_norm_type: str
    conv_context_size: list[int] | None
    dropout: float
    dropout_pre_encoder: float
    dropout_emb: float
    dropout_att: float
    stochastic_depth_drop_prob: float
    stochastic_depth_mode: str
    stochastic_depth_start_layer: int


@dataclass
class SortformerModuleConfig:
    _target_: str
    num_spks: int = 4
    dropout_rate: float = 0.5
    fc_d_model: int = 512
    tf_d_model: int = 192
    spkcache_len: int = 188
    fifo_len: int = 0
    chunk_len: int = 188
    spkcache_update_period: int = 188
    chunk_left_context: int = 1
    chunk_right_context: int = 1
    spkcache_sil_frames_per_spk: int = 3
    scores_add_rnd: int = 0
    causal_attn_rate: float = 0.5
    causal_attn_rc: int = 7


@dataclass
class PreProcessorConfig:
    _target_: str
    normalize: str | None
    log: bool = True
    window_size: float = 0.025
    sample_rate: int = 16000
    window_stride: float = 0.01
    window: str = "hann"
    features: int = 128
    n_fft: int = 512
    frame_splicing: int = 1
    dither: float = 1.0e-05


class StreamingInputArgs(TypedDict):
    r"""
    Typehint for keyword args input for forward of onnx
    Attributes:
        chunk (torch.Tensor):
            Shape (batch_size, diar frame count, feat_dim)
        chunk_lengths (torch.Tensor):
            Shape (batch_size,)
        spkcache (torch.Tensor):
        spkcache_lengths (torch.Tensor):
        fifo (torch.Tensor):
        fifo_lengths (torch.Tensor):
    """

    chunk: torch.Tensor
    chunk_lengths: torch.Tensor
    spkcache: torch.Tensor
    spkcache_lengths: torch.Tensor
    fifo: torch.Tensor
    fifo_lengths: torch.Tensor


@dataclass
class StreamingSortformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Sortformer model.

    Attributes:
        spkcache (torch.Tensor): Speaker cache to store embeddings from start.
            Shape (B, spkcache_len, fc_d_model)

        spkcache_lengths (torch.Tensor): Lengths of the speaker cache.
            Shape (B, )

        spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
            Shape (B, spkcache_len, num_spks)

        fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks.
            Shape (B, fifo_len, fc_d_model)

        fifo_lengths (torch.Tensor): Lengths of the FIFO queue.
            Shape (B,)

        fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
        spk_perm (torch.Tensor): Speaker permutation information for the speaker cache

        mean_sil_emb (torch.Tensor): Mean silence embedding.
            Shape (B, fc_d_model)

        n_sil_frames (torch.Tensor): Number of silence frames.
            Shape (B,)
    """

    spkcache: torch.Tensor
    spkcache_lengths: torch.Tensor
    spkcache_preds: torch.Tensor
    fifo: torch.Tensor
    fifo_lengths: torch.Tensor
    fifo_preds: torch.Tensor | None
    spk_perm: torch.Tensor | None
    mean_sil_emb: torch.Tensor
    n_sil_frames: torch.Tensor

    def to(self, device):
        if self.spkcache is not None:
            self.spkcache = self.spkcache.to(device)
        if self.spkcache_lengths is not None:
            self.spkcache_lengths = self.spkcache_lengths.to(device)
        if self.spkcache_preds is not None:
            self.spkcache_preds = self.spkcache_preds.to(device)
        if self.fifo is not None:
            self.fifo = self.fifo.to(device)
        if self.fifo_lengths is not None:
            self.fifo_lengths = self.fifo_lengths.to(device)
        if self.fifo_preds is not None:
            self.fifo_preds = self.fifo_preds.to(device)
        if self.spk_perm is not None:
            self.spk_perm = self.spk_perm.to(device)
        if self.mean_sil_emb is not None:
            self.mean_sil_emb = self.mean_sil_emb.to(device)
        if self.n_sil_frames is not None:
            self.n_sil_frames = self.n_sil_frames.to(device)


@dataclass
class PostProcessingParams:
    """
    Postprocessing parameters for end-to-end speaker diarization models.
    These parameters can significantly affect DER performance depending on the evaluation style and the dataset.
    It is recommended to tune these parameters based on the evaluation style and the dataset
    to achieve the desired DER performance.
    """

    onset: float = (
        0.5  # Onset threshold for detecting the beginning and end of a speech
    )
    offset: float = 0.5  # Offset threshold for detecting the end of a speech
    pad_onset: float = 0.0  # Adding durations before each speech segment
    pad_offset: float = 0.0  # Adding durations after each speech segment
    min_duration_on: float = 0.0  # Threshold for short speech segment deletion
    min_duration_off: float = 0.0  # Threshold for small non-speech deletion
