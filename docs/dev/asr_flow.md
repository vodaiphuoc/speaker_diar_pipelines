# Inheritance of classes for Nemotron streaming ASR

### Main model

`ASR model`\
`├── `[`ASRModuleMixin`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/mixins/mixins.py)\
`│   └── conformer_stream_step`\
`│       ├── self.encoder.cache_aware_stream_step`\
`│       └── self.decoding.rnnt_decoder_predictions_tensor`\
`└── EncDecRNNTModel`\
`    └── EncDecRNNTBPEModel`\
`        └── `[`EncDecRNNTBPEModelWithPrompt`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/models/rnnt_bpe_models_prompt.py)\
`            └── self.encoder: ConformerEncoder`

### Encoder

`Encoder`\
`├── `[`StreamingEncoder`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/mixins/streaming.py)\
`│   └── cache_aware_stream_step`\
`│       ├── self() (__call__ == forward)`\
`│       └── streaming_post_process`\
`└── `[`ConformerEncoder`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py)

### Decoding

`Decoding`\
`└── `[`AbstractRNNTDecoding`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_decoding.py)\
`    ├── rnnt_decoder_predictions_tensor`\
`    │   └── self.decoding: `[`GreedyBatchedRNNTInfer`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py)\
`    └── `[`RNNTBPEDecoding`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_decoding.py)
