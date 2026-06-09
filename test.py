import torch
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml
from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, InternalDiarizeConfig
import tempfile

CHUNK_SIZE = 124
RIGHT_CONTEXT = 1
FIFO_SIZE = 124
UPDATE_PERIOD = 124
SPEAKER_CACHE_SIZE = 188

# init
diar_model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    map_location="cpu"
)

# set eval
diar_model.eval()

print(type(diar_model.preprocessor))

# config
# diar_model.sortformer_modules.chunk_len = CHUNK_SIZE
# diar_model.sortformer_modules.chunk_right_context = RIGHT_CONTEXT
# diar_model.sortformer_modules.fifo_len = FIFO_SIZE
# diar_model.sortformer_modules.spkcache_update_period = UPDATE_PERIOD
# diar_model.sortformer_modules.spkcache_len = SPEAKER_CACHE_SIZE
# diar_model.sortformer_modules._check_streaming_parameters()



# workflow
# SortformerEncLabelModel
# method:
#   - process_signal
#   - sortformer_modules.init_streaming_state
#   
#
#

# audio:str = ["data/part1/bacsidatnhkhoavitadoc_1.wav", "data/part1/bacsidatnhkhoavitadoc_2.wav"]

# postprocessing_yaml = None
# postprocessing_params = load_postprocessing_from_yaml(postprocessing_yaml)
# diarize_cfg = DiarizeConfig(
#     batch_size=2,
#     num_workers=1,
#     verbose=False,
#     include_tensor_outputs=False,
#     postprocessing_yaml=postprocessing_yaml,
#     postprocessing_params=postprocessing_params,
#     sample_rate=16000,
#     **{},
# )

# # Add new internal config
# if diarize_cfg._internal is None:
#     diarize_cfg._internal = InternalDiarizeConfig()
# else:
#     # Check if internal config is valid
#     if not isinstance(diarize_cfg._internal, InternalDiarizeConfig):
#         raise ValueError(
#             "`diarize_cfg._internal` must be of an object of type InternalDiarizeConfig or " "its subclass"
#         )

# with tempfile.TemporaryDirectory() as tmpdir:
#     diarize_cfg._internal.temp_dir = tmpdir
#     dataloader = diar_model._diarize_input_processing(audio, diarize_cfg)

#     for batch in dataloader:
#         audio_signal, audio_signal_length = batch[0], batch[1]
#         print(audio_signal.shape, audio_signal_length)
#         with torch.no_grad():
#             processed_signal, processed_signal_length = diar_model.process_signal(
#                 audio_signal=audio_signal, audio_signal_length=audio_signal_length
#             )
#             processed_signal = processed_signal[:, :, : processed_signal_length.max()]
#             print(processed_signal.shape, processed_signal_length)