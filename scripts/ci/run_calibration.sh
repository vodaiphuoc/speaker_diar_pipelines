#!/usr/bin/env bash
set -euo pipefail

ASR_ASSET_DIR="${ASR_ASSET_DIR:-/app/.onnx_ckpt/asr}"
CALIBRATION_WAV="${NEMOTRON_CALIBRATION_WAV:-/app/tests/fixtures/asr_calibration_vi.wav}"

mkdir -p "${ASR_ASSET_DIR}"

python exports/asr.py --output-dir "${ASR_ASSET_DIR}"

required_assets=(
  asr_pretrained_config.yaml
  preprocessor.onnx
  final_encoder-exported_asr.onnx
  final_encoder_weight-exported_asr.data
  prompt_projection.onnx
  decoder_joint-exported_asr.onnx
  tokenizer.model
)

for asset in "${required_assets[@]}"; do
  test -s "${ASR_ASSET_DIR}/${asset}"
done

ruff check \
  SDP/__init__.py \
  SDP/onnx/asr \
  SDP/onnx/preprocess/audio_preprocessing.py \
  SDP/onnx/streaming_service.py \
  exports/asr.py \
  tests \
  run_phase_three.py
python -m compileall -q SDP exports tests run_phase_three.py

RUN_NEMOTRON_CALIBRATION=1 \
NEMOTRON_CALIBRATION_WAV="${CALIBRATION_WAV}" \
python -m unittest discover -s tests -p "test_*.py" -v
