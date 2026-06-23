from dataclasses import dataclass, field


@dataclass
class EncoderCacheAwareStreamingConfig:
    chunk_size: int = 0  # the size of each chunk at each step, it can be a list of two integers to specify different chunk sizes for the first step and others
    shift_size: int = 0  # the size of the shift in each step, it can be a list of two integers to specify different shift sizes for the first step and others

    cache_drop_size: int = 0  # the number of steps to drop from the cache
    last_channel_cache_size: int = (
        0  # the size of the needed cache for last channel layers
    )

    valid_out_len: int = 0  # the number of the steps in the final output which are valid (have the same value as in the offline mode)

    pre_encode_cache_size: int = 0  # the size of the needed cache for the pre-encoding part of the model to avoid caching inside the pre-encoding layers
    drop_extra_pre_encoded: int = (
        0  # the number of steps to get dropped after the pre-encoding layer
    )

    last_channel_num: int = 0  # number of the last channel layers (like MHA layers) which need caching in the model
    last_time_num: int = 0  # number of the last time layers (like convolutions) which need caching in the model


@dataclass
class EncoderConfig:
    feat_in: int = 128
    feat_out: int = -1
    n_layers: int = 24
    d_model: int = 1024
    use_bias: bool = False
    subsampling: str = "dw_striding"
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    causal_downsampling: bool = True
    reduction: str | None = None
    reduction_position: int | None = None
    reduction_factor: int = 1
    ff_expansion_factor: int = 4
    self_attention_model: str = "rel_pos"
    n_heads: int = 8
    att_context_size: list[int] = field(default_factory=lambda: [56, 1])
    att_context_style: str = "chunked_limited"
    xscaling: bool = False
    untie_biases: bool = True
    pos_emb_max_len: int = 5000
    conv_kernel_size: int = 9
    conv_norm_type: str = "layer_norm"
    conv_context_size: list[int] = field(default_factory=lambda: [8, 0])
    dropout: float = 0.1
    dropout_pre_encoder: float = 0.1
    dropout_emb: float = 0.0
    dropout_att: float = 0.1
    stochastic_depth_drop_prob: float = 0.0
    stochastic_depth_mode: str = "linear"
    stochastic_depth_start_layer: int = 1
