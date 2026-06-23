#!/usr/bin/env bash
set -euo pipefail

ruff check \
  SDP/__init__.py \
  SDP/onnx/artifacts.py \
  SDP/onnx/asr \
  SDP/onnx/preprocess/audio_preprocessing.py \
  SDP/onnx/streaming_service.py \
  tests \
  run_phase_three.py
python -m compileall -q SDP tests run_phase_three.py
python -m unittest discover -s tests -p "test_*.py" -v
