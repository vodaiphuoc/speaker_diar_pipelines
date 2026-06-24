"""Run the native NeMo versus ONNX ASR calibration test on Modal GPU."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import modal

ROOT_DIR = Path(__file__).resolve().parents[2]
CALIBRATION_LOG_PATH = Path("ci-logs/calibration.log")
CALIBRATION_REPORT_PATH = Path("ci-logs/asr_calibration_report.json")
CALIBRATION_COMMAND_DISPLAY = "bash scripts/ci/run_calibration.sh"

dockerfile_image = modal.Image.from_dockerfile(
    ROOT_DIR / "docker" / "Dockerfile_calibration_gpu",
    context_dir=ROOT_DIR,
    ignore="./.dockerignore",
)

app = modal.App("speaker-diar-asr-calibration")


class CalibrationResult(TypedDict):
    returncode: int
    stdout: str
    stderr: str
    report: str


@app.function(
    image=dockerfile_image,
    gpu=os.environ.get("MODAL_CALIBRATION_GPU", "T4"),
    timeout=60 * 60,
)
def run_calibration_remote() -> CalibrationResult:
    import os
    import subprocess
    from pathlib import Path

    logs_dir = Path("/app/ci-logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / "asr_calibration_report.json"

    env = os.environ.copy()
    env.update(
        {
            "ASR_ASSET_DIR": "/app/.onnx_ckpt/asr",
            "RUN_NEMOTRON_CALIBRATION": "1",
            "NEMOTRON_NATIVE_DEVICE": "cuda",
            "NEMOTRON_CALIBRATION_WAV": (
                "/app/tests/fixtures/bacsidatnhkhoavitadoc_1.wav"
            ),
            "NEMOTRON_CALIBRATION_REPORT": str(report_path),
        }
    )

    completed = subprocess.run(
        ["bash", "scripts/ci/run_calibration.sh"],
        cwd="/app",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": report_path.read_text(encoding="utf-8")
        if report_path.exists()
        else "",
    }


@app.local_entrypoint()
def main() -> None:
    CALIBRATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = run_calibration_remote.remote()

    log_text = (
        f"Remote command: {CALIBRATION_COMMAND_DISPLAY}\n"
        f"Return code: {result['returncode']}\n\n"
        f"STDOUT:\n{result['stdout']}\n"
        f"STDERR:\n{result['stderr']}\n"
    )
    CALIBRATION_LOG_PATH.write_text(log_text, encoding="utf-8")

    if result["report"]:
        CALIBRATION_REPORT_PATH.write_text(result["report"], encoding="utf-8")

    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])
