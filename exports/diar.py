# Before running the script manually, make sure `speaker_diar_model_export`
# exists, or set DIAR_EXPORT_VOLUME_NAME to another Modal volume.
# If not, use `modal volume create`.
# to run this script, in root project directory:
# ```terminal
# modal run exports/diar.py
# ```
# After runing the script, expect the following files are placed
# in `speaker_diar_model_export` volume
# ```terminal
# modal volume ls speaker_diar_model_export --json
# ```
# Outputs
# [
#   {
#     "filename": "preprocessor.onnx",
#     "type": "file",
#     "created_modified": "2026-06-19 10:33 +07",
#     "size": "180.3 KiB"
#   },
#   {
#     "filename": "model.onnx",
#     "type": "file",
#     "created_modified": "2026-06-19 10:33 +07",
#     "size": "469.4 MiB"
#   },
#   {
#     "filename": "diar_pretrained_config.yaml",
#     "type": "file",
#     "created_modified": "2026-06-19 10:32 +07",
#     "size": "3.3 KiB"
#   },
#   {
#     "filename": "diarization_artifact.json",
#     "type": "file"
#   }
# ]
# Download the manifest and every file it references into `.onnx_ckpt/diar`.
# Relative manifest paths require the exported runtime assets to stay together.

import os
from pathlib import Path

import modal

ROOT_DIR = Path(__file__).resolve().parent.parent

dockerfile_image = modal.Image.from_dockerfile(
    ROOT_DIR / "docker" / "Dockerfile",
    context_dir=ROOT_DIR,
    ignore="./.dockerignore",
)

DIAR_EXPORT_VOLUME_NAME = os.environ.get(
    "DIAR_EXPORT_VOLUME_NAME",
    "speaker_diar_model_export",
)
vol = modal.Volume.from_name(DIAR_EXPORT_VOLUME_NAME, create_if_missing=True)
app = modal.App("pipeline-inference-v4")


@app.function(image=dockerfile_image, timeout=60 * 20, volumes={"/mydata": vol})
def run():
    import os
    import subprocess

    subprocess.run(
        ["pip", "install", "typing_extensions>=4.14.0", "--ignore-installed"],
        check=True,
    )

    os.environ["LOG_LEVEL"] = "INFO"
    import contextlib

    import torch

    @contextlib.contextmanager
    def use_forward_for_export(model: torch.nn.Module):
        """
        Temporarily replaces the 'forward' method with 'forward_for_export'
        for the top-level model and all nested sub-modules.
        """
        original_forwards = {}

        # 1. Walk through the entire module tree
        for name, module in model.named_modules():
            # Check if this specific module has the export alternative
            if hasattr(module, "forward_for_export"):
                # Save the original eager forward method
                original_forwards[module] = module.forward
                # Overwrite 'forward' with the export-friendly version
                module.forward = module.forward_for_export

        try:
            # Yield control back to the export block
            yield model
        finally:
            # 2. Revert everything back to normal after exporting
            for module, orig_forward in original_forwards.items():
                module.forward = orig_forward

    def export():
        import types
        from typing import List, Tuple

        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
        from SDP.onnx.artifacts import write_diarization_artifact_manifest

        CHUNK_SIZE = 124
        RIGHT_CONTEXT = 1
        FIFO_SIZE = 124
        UPDATE_PERIOD = 124
        SPEAKER_CACHE_SIZE = 188

        # init
        diar_model: SortformerEncLabelModel = SortformerEncLabelModel.from_pretrained(
            "nvidia/diar_streaming_sortformer_4spk-v2", map_location="cpu"
        )

        # config
        diar_model.sortformer_modules.chunk_len = CHUNK_SIZE
        diar_model.sortformer_modules.chunk_right_context = RIGHT_CONTEXT
        diar_model.sortformer_modules.fifo_len = FIFO_SIZE
        diar_model.sortformer_modules.spkcache_update_period = UPDATE_PERIOD
        diar_model.sortformer_modules.spkcache_len = SPEAKER_CACHE_SIZE
        diar_model.sortformer_modules._check_streaming_parameters()

        diar_model.sortformer_modules.log = True
        diar_model.async_streaming = True

        # set eval
        diar_model.eval()
        print(
            "check `encoder_proj` in sortformer_modules: ",
            diar_model.sortformer_modules.encoder_proj,
        )

        from omegaconf import OmegaConf

        with open("/mydata/diar_pretrained_config.yaml", "w+") as fp:
            OmegaConf.save(config=diar_model._cfg, f=fp.name)

        def custom_streaming_input_examples(self):
            """Input tensor examples for exporting streaming version of model"""
            print("run custom example streaming input")
            batch_size = 4
            # chunk = torch.rand([batch_size, 120, 80]).to(self.device)
            # chunk_lengths = torch.tensor([120] * batch_size).to(self.device)
            # spkcache = torch.randn([batch_size, 188, 512]).to(self.device)
            # spkcache_lengths = torch.tensor([40, 188, 0, 68]).to(self.device)
            # fifo = torch.randn([batch_size, 188, 512]).to(self.device)
            # fifo_lengths = torch.tensor([50, 88, 0, 90]).to(self.device)

            chunk = torch.rand([batch_size, 1520, 128]).to(self.device)
            chunk_lengths = torch.tensor([1520] * batch_size).to(self.device)
            spkcache = torch.randn([batch_size, 188, 512]).to(self.device)
            spkcache_lengths = torch.tensor([188] * batch_size).to(self.device)
            fifo = torch.randn([batch_size, FIFO_SIZE, 512]).to(self.device)
            fifo_lengths = torch.tensor([124] * batch_size).to(self.device)
            return chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths

        def custom_concat_and_pad(
            embs: List[torch.Tensor],
            lengths: List[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:

            total_lengths = torch.sum(torch.stack(lengths), dim=0)
            max_len = total_lengths.max().to(torch.int64).item()

            total_lengths = torch.stack(lengths).sum(0)

            B = embs[0].shape[0]
            D = embs[0].shape[2]

            # out = embs[0].new_zeros((B, max_len, D))
            device, dtype = embs[0].device, embs[0].dtype
            out = torch.zeros(B, max_len, D, device=device, dtype=dtype)

            start = torch.zeros_like(total_lengths)

            for emb, length in zip(embs, lengths):
                T = emb.shape[1]

                frame_idx = torch.arange(T, device=emb.device)[None, :]
                valid = frame_idx < length[:, None]

                dst_idx = start[:, None] + frame_idx

                batch_idx = torch.arange(B, device=emb.device)[:, None].expand(B, T)

                out[batch_idx[valid], dst_idx[valid]] = emb[valid]
                start = start + length

            return out, total_lengths

        # assign custom models avoid break export process
        diar_model.streaming_input_examples = types.MethodType(
            custom_streaming_input_examples, diar_model
        )
        diar_model.concat_and_pad_script = torch.jit.script(custom_concat_and_pad)

        from nemo.utils.export_utils import (
            wrap_forward_method,
        )

        forward_method, old_forward_method = wrap_forward_method(diar_model)

        dynamic_axes = {
            "chunk": {0: "batch_dim", 1: "chunk__1"},
            "chunk_lengths": {0: "batch_dim"},
            "spkcache": {
                0: "batch_dim",
                1: "spkcache__1",
            },
            "spkcache_lengths": {0: "batch_dim"},
            "fifo": {0: "batch_dim", 1: "fifo__1"},
            "fifo_lengths": {0: "batch_dim"},
        }
        print("diar_model.input_names: ", diar_model.input_names)
        # set dynamo = False to use legacy api for main model
        # with use_forward_for_export(diar_model):
        torch.onnx.export(
            model=diar_model,
            args=diar_model.streaming_input_examples(),
            f="/mydata/model.onnx",
            input_names=diar_model.input_names,
            output_names=diar_model.output_names,
            dynamo=False,
            verbose=False,
            report=True,
            optimize=True,
            opset_version=20,
            external_data=False,
            export_params=True,
            export_modules_as_functions=False,
            # dynamic_shapes=dynamic_shapes,
            dynamic_axes=dynamic_axes,
        )

        def export_preprocessor(
            model: SortformerEncLabelModel,
            path_to_save: str,
            max_batch: int = 8,
            max_dim: int = 32000,
            min_length: int = 200,
        ):
            r"""
            Export preprocessor to onnx format using torch.onnx.export API
            with dynamo
            """
            # config for dynamic shape
            batch_dim = torch.export.Dim("batch", min=1, max=max_batch)
            dynamic_shapes = {
                "input_signal": {
                    0: batch_dim,
                    1: torch.export.Dim("input_signal__1", min=1, max=max_dim),
                },
                "length": {0: batch_dim},
            }

            preprocessor_instance: AudioToMelSpectrogramPreprocessor = (
                model.preprocessor
            )
            print(preprocessor_instance.input_names)
            print(preprocessor_instance.output_names)
            # set dynamo = True for pre-processing model
            onnx_program = torch.onnx.export(
                model=preprocessor_instance,
                args=preprocessor_instance.input_example(),
                # f=path_to_save,
                input_names=preprocessor_instance.input_names,
                output_names=preprocessor_instance.output_names,
                dynamo=True,
                report=True,
                optimize=True,
                external_data=False,
                export_params=True,
                export_modules_as_functions=False,
                dynamic_shapes=dynamic_shapes,
            )
            assert onnx_program is not None
            onnx_program.save(
                destination=path_to_save,
                external_data=False,
            )

        export_preprocessor(diar_model, "/mydata/preprocessor.onnx")
        write_diarization_artifact_manifest(
            output_dir=Path("/mydata"),
            source_model="nvidia/diar_streaming_sortformer_4spk-v2",
            preprocessor=Path("/mydata/preprocessor.onnx"),
            sortformer=Path("/mydata/model.onnx"),
            config=Path("/mydata/diar_pretrained_config.yaml"),
        )

    export()
    return None


@app.local_entrypoint()
def main():
    run.remote()
