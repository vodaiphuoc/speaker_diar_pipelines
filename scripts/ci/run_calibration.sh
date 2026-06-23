#!/usr/bin/env bash
set -euo pipefail

ASR_ASSET_DIR="${ASR_ASSET_DIR:-/app/.onnx_ckpt/asr}"
CALIBRATION_WAV="${NEMOTRON_CALIBRATION_WAV:-/app/tests/fixtures/asr_calibration_vi.wav}"

mkdir -p "${ASR_ASSET_DIR}"

python exports/asr.py --output-dir "${ASR_ASSET_DIR}"

ASR_MANIFEST="${ASR_ASSET_DIR}/asr_artifact.json"
test -s "${ASR_MANIFEST}"
python - "${ASR_MANIFEST}" <<'PY'
import sys

from SDP.onnx.artifacts import load_asr_artifact_manifest

load_asr_artifact_manifest(sys.argv[1])
PY

ruff check \
  SDP/__init__.py \
  SDP/onnx/artifacts.py \
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
