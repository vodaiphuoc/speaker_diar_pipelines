#!/usr/bin/env bash
set -euo pipefail

ASR_ASSET_DIR="${ASR_ASSET_DIR:-/app/.onnx_ckpt/asr}"
DIAR_ASSET_DIR="${DIAR_ASSET_DIR:-/app/.onnx_ckpt/diar}"
CALIBRATION_WAV="${PIPELINE_CALIBRATION_WAV:-/app/tests/fixtures/bacsidatnhkhoavitadoc_1.wav}"
ALIGNMENT_MODE="${PIPELINE_ALIGNMENT_MODE:-diarization_timeline}"
ASR_BACKEND="${PIPELINE_ASR_BACKEND:-nemotron_onnx}"
CALIBRATION_REPORT="${PIPELINE_CALIBRATION_REPORT:-/app/ci-logs/pipeline_calibration_report_${ALIGNMENT_MODE}.json}"
RAW_EVENTS_REPORT="${PIPELINE_RAW_EVENTS_REPORT:-/app/ci-logs/pipeline_raw_events_${ALIGNMENT_MODE}.json}"
CALIBRATION_TEST_TARGET="${PIPELINE_CALIBRATION_TEST_TARGET:-tests.calibration.pipeline.test_calibration.NativeVsOnnxPipelineCalibrationTest}"

case "${ALIGNMENT_MODE}" in
  diarization_timeline|asr_timeline) ;;
  *)
    echo "PIPELINE_ALIGNMENT_MODE must be 'diarization_timeline' or 'asr_timeline', got: ${ALIGNMENT_MODE}" >&2
    exit 1
    ;;
esac

case "${ASR_BACKEND}" in
  nemotron_onnx)
    echo "Exporting latest ASR ONNX artifacts into pipeline calibration volume path: ${ASR_ASSET_DIR}"
    rm -rf "${ASR_ASSET_DIR}"
    mkdir -p "${ASR_ASSET_DIR}" "$(dirname "${CALIBRATION_REPORT}")"
    python -m exports.asr --output-dir "${ASR_ASSET_DIR}"
    ;;
  qwen3_modal)
    echo "Skipping ASR ONNX export for Qwen3 ASR backend."
    mkdir -p "$(dirname "${CALIBRATION_REPORT}")"
    ;;
  *)
    echo "PIPELINE_ASR_BACKEND must be 'nemotron_onnx' or 'qwen3_modal', got: ${ASR_BACKEND}" >&2
    exit 1
    ;;
esac

ASR_MANIFEST="${ASR_ASSET_DIR}/asr_artifact.json"
DIAR_MANIFEST="${DIAR_ASSET_DIR}/diarization_artifact.json"

if [[ "${ASR_BACKEND}" == "nemotron_onnx" ]]; then
  test -s "${ASR_MANIFEST}"
fi
if [[ ! -s "${DIAR_MANIFEST}" ]]; then
  echo "Missing diarization ONNX artifact manifest: ${DIAR_MANIFEST}" >&2
  echo "Expected the pipeline calibration Modal volume to contain diarization_artifact.json and referenced ONNX assets." >&2
  echo "Volume name: ${MODAL_PIPELINE_CALIBRATION_VOLUME:-<unset>}" >&2
  echo "DIAR_ASSET_DIR listing:" >&2
  ls -la "${DIAR_ASSET_DIR}" >&2 || true
  exit 1
fi

python - "${ASR_MANIFEST}" "${DIAR_MANIFEST}" <<'PY'
import os
import sys

from SDP.onnx.artifacts import (
    load_asr_artifact_manifest,
    load_diarization_artifact_manifest,
)

if os.environ.get("PIPELINE_ASR_BACKEND", "nemotron_onnx") == "nemotron_onnx":
    load_asr_artifact_manifest(sys.argv[1])
load_diarization_artifact_manifest(sys.argv[2])
PY

ruff check \
  SDP/__init__.py \
  SDP/onnx/artifacts.py \
  SDP/onnx/asr \
  SDP/onnx/preprocess/audio_preprocessing.py \
  SDP/onnx/streaming_service.py \
  SDP/qwen3_asr \
  SDP/pipeline \
  exports/asr.py \
  tests \
  run_phase_three.py
python -m compileall -q SDP exports tests run_phase_three.py

echo "=== Pipeline calibration unittest environment ==="
echo "cwd=$(pwd)"
echo "python=$(command -v python)"
echo "CALIBRATION_TEST_TARGET=${CALIBRATION_TEST_TARGET}"
echo "CALIBRATION_WAV=${CALIBRATION_WAV}"
echo "CALIBRATION_REPORT=${CALIBRATION_REPORT}"
echo "RAW_EVENTS_REPORT=${RAW_EVENTS_REPORT}"
echo "PIPELINE_ALIGNMENT_MODE=${ALIGNMENT_MODE}"
echo "PIPELINE_ASR_BACKEND=${ASR_BACKEND}"
echo "ASR_ASSET_DIR=${ASR_ASSET_DIR}"
echo "DIAR_ASSET_DIR=${DIAR_ASSET_DIR}"
python - <<'PY'
import importlib.util
import sys

print(f"sys.executable={sys.executable}")
print(f"sys.path[:5]={sys.path[:5]}")
for module_name in (
    "SDP",
    "SDP.pipeline",
    "tests.calibration.pipeline.test_calibration",
    "nemo",
    "onnxruntime",
    "qwen_asr",
):
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        print(f"{module_name}=ERROR {type(exc).__name__}: {exc}")
    else:
        print(f"{module_name}={spec.origin if spec is not None else 'NOT_FOUND'}")
PY

RUN_PIPELINE_CALIBRATION=1 \
PIPELINE_ALIGNMENT_MODE="${ALIGNMENT_MODE}" \
PIPELINE_ASR_BACKEND="${ASR_BACKEND}" \
PIPELINE_CALIBRATION_WAV="${CALIBRATION_WAV}" \
PIPELINE_CALIBRATION_REPORT="${CALIBRATION_REPORT}" \
PIPELINE_RAW_EVENTS_REPORT="${RAW_EVENTS_REPORT}" \
python -m unittest "${CALIBRATION_TEST_TARGET}" -v
