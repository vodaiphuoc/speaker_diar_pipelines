#!/usr/bin/env bash
set -euo pipefail

ASR_ASSET_DIR="${ASR_ASSET_DIR:-/app/.onnx_ckpt/asr}"
CALIBRATION_WAV="${NEMOTRON_CALIBRATION_WAV:-/app/tests/fixtures/bacsidatnhkhoavitadoc_1.wav}"
CALIBRATION_REPORT="${NEMOTRON_CALIBRATION_REPORT:-/app/ci-logs/asr_calibration_report.json}"
CALIBRATION_TEST_TARGET="${NEMOTRON_CALIBRATION_TEST_TARGET:-tests.calibration.asr.test_model_calibration.NemotronONNXCalibrationTest}"

echo "Exporting latest ASR ONNX artifacts into ASR calibration volume path: ${ASR_ASSET_DIR}"
rm -rf "${ASR_ASSET_DIR}"
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
  SDP/pipeline \
  exports/asr.py \
  tests \
  run_phase_three.py
python -m compileall -q SDP exports tests run_phase_three.py

echo "=== Calibration unittest environment ==="
echo "cwd=$(pwd)"
echo "python=$(command -v python)"
echo "CALIBRATION_TEST_TARGET=${CALIBRATION_TEST_TARGET}"
echo "CALIBRATION_WAV=${CALIBRATION_WAV}"
echo "CALIBRATION_REPORT=${CALIBRATION_REPORT}"
echo "ASR_ASSET_DIR=${ASR_ASSET_DIR}"
python - <<'PY'
import importlib.util
import sys

print(f"sys.executable={sys.executable}")
print(f"sys.path[:5]={sys.path[:5]}")
for module_name in (
    "SDP",
    "tests.calibration.asr.test_model_calibration",
    "nemo",
    "onnxruntime",
):
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        print(f"{module_name}=ERROR {type(exc).__name__}: {exc}")
    else:
        print(f"{module_name}={spec.origin if spec is not None else 'NOT_FOUND'}")
PY

RUN_NEMOTRON_CALIBRATION=1 \
NEMOTRON_CALIBRATION_WAV="${CALIBRATION_WAV}" \
NEMOTRON_CALIBRATION_REPORT="${CALIBRATION_REPORT}" \
python -m unittest "${CALIBRATION_TEST_TARGET}" -v
