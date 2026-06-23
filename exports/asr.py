"""Export all production assets required by the Nemotron streaming ONNX runtime."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import torch


class PromptProjectionWrapper(torch.nn.Module):
    def __init__(self, prompt_kernel: torch.nn.Module, num_prompts: int):
        super().__init__()
        self.prompt_kernel = prompt_kernel
        self.num_prompts = num_prompts

    def forward(
        self, encoded: torch.Tensor, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        encoded_time_major = encoded.transpose(1, 2)
        prompt = torch.nn.functional.one_hot(
            prompt_index.to(torch.int64), num_classes=self.num_prompts
        )
        prompt = (
            prompt.to(encoded.dtype)
            .unsqueeze(1)
            .expand(-1, encoded_time_major.shape[1], -1)
        )
        projected = self.prompt_kernel(torch.cat((encoded_time_major, prompt), dim=-1))
        return projected.transpose(1, 2)


def _find_sentencepiece_model(model) -> Path:
    candidates = [
        getattr(model.tokenizer, "model_path", None),
        getattr(getattr(model.tokenizer, "tokenizer", None), "model_file", None),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Could not locate the restored SentencePiece model")


def export_assets(output_dir: Path, model_name: str) -> None:
    import nemo.collections.asr as nemo_asr
    import onnx
    from omegaconf import OmegaConf

    output_dir.mkdir(parents=True, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_name, map_location="cpu"
    )
    model.eval()
    model.encoder.setup_streaming_params(att_context_size=[56, 1])
    model.encoder.export_cache_support = True

    OmegaConf.save(model._cfg, output_dir / "asr_pretrained_config.yaml")

    export_prefix = output_dir / "nemotron_streaming.onnx"
    model.export(
        str(export_prefix),
        onnx_opset_version=20,
        check_trace=False,
        do_constant_folding=True,
    )

    encoder_source = output_dir / "encoder-nemotron_streaming.onnx"
    encoder_target = output_dir / "final_encoder-exported_asr.onnx"
    encoder_weights = output_dir / "final_encoder_weight-exported_asr.data"
    encoder_model = onnx.load(str(encoder_source), load_external_data=True)
    onnx.save_model(
        encoder_model,
        str(encoder_target),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        size_threshold=0,
        location=encoder_weights.name,
    )
    encoder_source.unlink()

    decoder_source = output_dir / "decoder_joint-nemotron_streaming.onnx"
    decoder_source.replace(output_dir / "decoder_joint-exported_asr.onnx")

    preprocessor = copy.deepcopy(model.preprocessor).cpu().eval()
    preprocessor.featurizer.dither = 0.0
    preprocessor.featurizer.pad_to = 0
    signal = torch.zeros((1, 32000), dtype=torch.float32)
    signal_length = torch.tensor([32000], dtype=torch.int64)
    torch.onnx.export(
        preprocessor,
        kwargs={
            "input_signal": signal,
            "length": signal_length,
        },
        f=str(output_dir / "preprocessor.onnx"),
        input_names=["input_signal", "length"],
        output_names=["processed_signal", "processed_length"],
        dynamic_axes={
            "input_signal": {0: "batch", 1: "samples"},
            "length": {0: "batch"},
            "processed_signal": {0: "batch", 2: "frames"},
            "processed_length": {0: "batch"},
        },
        opset_version=20,
        dynamo=False,
    )

    prompt_projection = PromptProjectionWrapper(
        copy.deepcopy(model.prompt_kernel).cpu().eval(),
        int(model.num_prompts),
    )
    torch.onnx.export(
        prompt_projection,
        (
            torch.zeros((1, 1024, 2), dtype=torch.float32),
            torch.tensor([33], dtype=torch.int64),
        ),
        str(output_dir / "prompt_projection.onnx"),
        input_names=["encoded", "prompt_index"],
        output_names=["outputs"],
        dynamic_axes={
            "encoded": {0: "batch", 2: "frames"},
            "prompt_index": {0: "batch"},
            "outputs": {0: "batch", 2: "frames"},
        },
        opset_version=20,
        dynamo=False,
    )

    shutil.copy2(_find_sentencepiece_model(model), output_dir / "tokenizer.model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(".onnx_ckpt/asr"))
    parser.add_argument(
        "--model-name",
        default="nvidia/nemotron-3.5-asr-streaming-0.6b",
    )
    args = parser.parse_args()
    export_assets(args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
