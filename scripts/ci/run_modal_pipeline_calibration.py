"""Run native NeMo versus ONNX diarization+ASR pipeline calibration on Modal GPU."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import modal

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CALIBRATION_LOG_PATH = Path("ci-logs/pipeline_calibration.log")
CALIBRATION_COMMAND_DISPLAY = "bash scripts/ci/run_pipeline_calibration.sh"
PIPELINE_CALIBRATION_VOLUME_NAME = os.environ.get(
    "MODAL_PIPELINE_CALIBRATION_VOLUME",
    "speaker_diar_ci_pipeline_calibration",
)
PIPELINE_ALIGNMENT_MODE = os.environ.get(
    "PIPELINE_ALIGNMENT_MODE",
    "diarization_timeline",
)

dockerfile_image = modal.Image.from_dockerfile(
    ROOT_DIR / "docker" / "Dockerfile_calibration_gpu",
    context_dir=ROOT_DIR,
    ignore="./.dockerignore",
)

pipeline_calibration_volume = modal.Volume.from_name(
    PIPELINE_CALIBRATION_VOLUME_NAME,
    create_if_missing=True,
)
app = modal.App("speaker-diar-pipeline-calibration")


class PipelineCalibrationResult(TypedDict):
    returncode: int
    command: str
    cwd: str
    env_summary: dict[str, str]
    stdout: str
    stderr: str
    report_path: str
    report_exists: bool
    report: str
    raw_events_report_path: str
    raw_events_report_exists: bool
    raw_events_report: str


@app.function(
    image=dockerfile_image,
    gpu=os.environ.get("MODAL_CALIBRATION_GPU", "T4"),
    timeout=60 * 60,
    volumes={"/app/.modal_ci/pipeline_calibration": pipeline_calibration_volume},
)
def run_pipeline_calibration_remote() -> PipelineCalibrationResult:
    import os
    import subprocess
    from pathlib import Path

    volume_root = Path("/app/.modal_ci/pipeline_calibration")
    logs_dir = volume_root / "ci-logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    alignment_mode = PIPELINE_ALIGNMENT_MODE
    report_path = logs_dir / f"pipeline_calibration_report_{alignment_mode}.json"
    raw_events_report_path = logs_dir / f"pipeline_raw_events_{alignment_mode}.json"

    env = os.environ.copy()
    env.update(
        {
            "ASR_ASSET_DIR": str(volume_root / "onnx_ckpt" / "asr" / alignment_mode),
            "DIAR_ASSET_DIR": str(volume_root),
            "MODAL_PIPELINE_CALIBRATION_VOLUME": PIPELINE_CALIBRATION_VOLUME_NAME,
            "PIPELINE_ALIGNMENT_MODE": alignment_mode,
            "RUN_PIPELINE_CALIBRATION": "1",
            "NEMOTRON_NATIVE_DEVICE": "cuda",
            "PIPELINE_CALIBRATION_WAV": (
                "/app/tests/fixtures/bacsidatnhkhoavitadoc_1.wav"
            ),
            "PIPELINE_CALIBRATION_REPORT": str(report_path),
            "PIPELINE_RAW_EVENTS_REPORT": str(raw_events_report_path),
            "POST_PROCESSING_CONFIG": "/app/configs/post_processing.yaml",
        }
    )
    env_summary_keys = (
        "ASR_ASSET_DIR",
        "DIAR_ASSET_DIR",
        "MODAL_PIPELINE_CALIBRATION_VOLUME",
        "PIPELINE_ALIGNMENT_MODE",
        "RUN_PIPELINE_CALIBRATION",
        "NEMOTRON_NATIVE_DEVICE",
        "PIPELINE_CALIBRATION_WAV",
        "PIPELINE_CALIBRATION_REPORT",
        "PIPELINE_RAW_EVENTS_REPORT",
        "PIPELINE_CALIBRATION_TEST_TARGET",
        "PYTHONPATH",
    )
    env_summary = {key: env.get(key, "<unset>") for key in env_summary_keys}

    completed = subprocess.run(
        ["bash", "scripts/ci/run_pipeline_calibration.sh"],
        cwd="/app",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    report_exists = report_path.exists()
    raw_events_report_exists = raw_events_report_path.exists()

    return {
        "returncode": completed.returncode,
        "command": CALIBRATION_COMMAND_DISPLAY,
        "cwd": "/app",
        "env_summary": env_summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report_path": str(report_path),
        "report_exists": report_exists,
        "report": report_path.read_text(encoding="utf-8") if report_exists else "",
        "raw_events_report_path": str(raw_events_report_path),
        "raw_events_report_exists": raw_events_report_exists,
        "raw_events_report": (
            raw_events_report_path.read_text(encoding="utf-8")
            if raw_events_report_exists
            else ""
        ),
    }


@app.local_entrypoint()
def main() -> None:
    CALIBRATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = run_pipeline_calibration_remote.remote()
    return_code = result["returncode"]
    stdout = result["stdout"] or "<empty>"
    stderr = result["stderr"] or "<empty>"
    combined_output = f"{stdout}\n{stderr}"
    failure_hint = ""
    if return_code != 0 and (
        "ModuleNotFoundError" in combined_output or "ImportError" in combined_output
    ):
        failure_hint = (
            "\nFailure hint: import/module error detected. Check the module "
            "availability lines in the unittest environment section above.\n"
        )
    env_summary = "\n".join(
        f"{key}={value}" for key, value in sorted(result["env_summary"].items())
    )
    log_text = (
        f"Remote command: {result['command']}\n"
        f"Remote cwd: {result['cwd']}\n"
        f"Return code: {return_code}\n\n"
        f"Environment summary:\n{env_summary}\n\n"
        f"Report path: {result['report_path']}\n"
        f"Report exists: {result['report_exists']}\n"
        f"Raw events report path: {result['raw_events_report_path']}\n"
        f"Raw events report exists: {result['raw_events_report_exists']}\n"
        f"{failure_hint}\n"
        f"STDOUT:\n{stdout}\n"
        f"STDERR:\n{stderr}\n"
    )
    CALIBRATION_LOG_PATH.write_text(log_text, encoding="utf-8")

    if result["report"]:
        report_path = Path(
            f"ci-logs/pipeline_calibration_report_{PIPELINE_ALIGNMENT_MODE}.json"
        )
        report_path.write_text(result["report"], encoding="utf-8")
    if result["raw_events_report"]:
        raw_events_report_path = Path(
            f"ci-logs/pipeline_raw_events_{PIPELINE_ALIGNMENT_MODE}.json"
        )
        raw_events_report_path.write_text(
            result["raw_events_report"],
            encoding="utf-8",
        )

    if result["returncode"] != 0:
        raise SystemExit(return_code)
