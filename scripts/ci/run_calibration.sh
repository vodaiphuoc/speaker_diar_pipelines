#!/usr/bin/env bash
set -euo pipefail

ASR_ASSET_DIR="${ASR_ASSET_DIR:-/app/.onnx_ckpt/asr}"
CALIBRATION_WAV="${NEMOTRON_CALIBRATION_WAV:-/app/tests/fixtures/bacsidatnhkhoavitadoc_1.wav}"
CALIBRATION_REPORT="${NEMOTRON_CALIBRATION_REPORT:-/app/ci-logs/asr_calibration_report.json}"
CALIBRATION_TEST_TARGET="${NEMOTRON_CALIBRATION_TEST_TARGET:-tests.test_asr_calibration.NemotronONNXCalibrationTest}"

mkdir -p "${ASR_ASSET_DIR}"

python -m exports.asr --output-dir "${ASR_ASSET_DIR}"

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
NEMOTRON_CALIBRATION_REPORT="${CALIBRATION_REPORT}" \
python -m unittest "${CALIBRATION_TEST_TARGET}" -v
