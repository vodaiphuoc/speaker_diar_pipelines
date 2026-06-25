# Inheritance of classes for Nemotron streaming ASR

### Main model

[`ASRModuleMixin`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/mixins/mixins.py#L597)\
`│   └── conformer_stream_step()`\
`│       ├── self.encoder.cache_aware_stream_step()`\
`│       └── self.decoding.rnnt_decoder_predictions_tensor()`\
`│`\
`└── `[`EncDecRNNTModel`](nemo/collections/asr/models/rnnt_models.py#L53)\
`      │`\
`      └── `[`EncDecRNNTBPEModel`](nemo/collections/asr/models/rnnt_bpe_models.py#L38)\
`             │`\
`             └── `[`EncDecRNNTBPEModelWithPrompt`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/models/rnnt_bpe_models_prompt.py#L56)\
`                     └── self.encoder: ConformerEncoder`\
`                     └── self.decoding: RNNTBPEDecoding`\
`                     └── self.decoder_joint: RNNTDecoderJoint`

### Encoder module

[`StreamingEncoder`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/mixins/streaming.py)\
`│   └── cache_aware_stream_step()`\
`│       ├── self() (__call__ == forward)`\
`│       └── streaming_post_process`\
`│`\
`└── `[`ConformerEncoder`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py)

### Decoder Joint module

[`RNNTDecoderJoint`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/modules/rnnt.py#L1875)\
`│   └── forward(self, 
                encoder_outputs, 
                targets, 
                target_length, 
                input_states_1, 
                input_states_2)`\

### Decoding module

[`AbstractRNNTDecoding`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L719)\
`│   ├── self.decoding: GreedyBatchedRNNTInfer`\
`│   │`\
`│   └── rnnt_decoder_predictions_tensor()`\
`│          │`\
`│          └── self.decoding(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                partial_hypotheses=partial_hypotheses
            )`
            
`│`\
`└── `[`RNNTBPEDecoding`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_decoding.py#L1519)

### Greedy Decoding module

[`_GreedyRNNTInfer`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L99)\
`│   └── __call__()`\
`│          │`\
`│          └── self.forward()`\
`│`\
`└── `[`GreedyBatchedRNNTInfer`](https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py#L529)\
`│          │`\
`│          └── forward(encoder_output, encoded_lengths, partial_hypotheses)`\
`│                  ... ` \
`│                  └── self._greedy_decode()` \
