import torch
from typing import TypedDict
from dataclasses import dataclass


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

class StreamingInputArgs(TypedDict, total=False):
    r"""
    Typehint for keyword args input for forward of onnx
    Args:
        chunk (torch.Tensor):
        chunk_lengths (torch.Tensor):
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

    spkcache = None  # Speaker cache to store embeddings from start
    spkcache_lengths = None  #
    spkcache_preds = None  # speaker cache predictions
    fifo = None  # to save the embedding from the latest chunks
    fifo_lengths = None
    fifo_preds = None
    spk_perm = None
    mean_sil_emb = None
    n_sil_frames = None

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

