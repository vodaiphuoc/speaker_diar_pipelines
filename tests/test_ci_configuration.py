import ast
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


class DependencyBoundaryTest(unittest.TestCase):
    def test_onnx_package_does_not_import_nemo(self):
        violations = []
        for source_path in sorted((PROJECT_ROOT / "SDP" / "onnx").rglob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported_names = [node.module or ""]
                else:
                    continue
                if any(
                    name == "nemo" or name.startswith("nemo.")
                    for name in imported_names
                ):
                    violations.append(str(source_path.relative_to(PROJECT_ROOT)))

        self.assertEqual(violations, [])

    def test_dependency_profiles_are_layered(self):
        expected_includes = {
            "requirements/test.txt": "-r onnx.txt",
            "requirements/calibration.txt": "-r test.txt",
            "requirements/dev.txt": "-r calibration.txt",
        }
        for filename, include in expected_includes.items():
            contents = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn(include, contents, filename)

        production_requirements = (
            PROJECT_ROOT / "requirements" / "onnx.txt"
        ).read_text(encoding="utf-8")
        self.assertNotIn("nemo_toolkit", production_requirements)
        self.assertNotRegex(production_requirements, r"(?m)^onnx(?:[<=>]|$)")
        self.assertIn(
            "onnx>=",
            (PROJECT_ROOT / "requirements" / "test.txt").read_text(),
        )
        self.assertIn(
            "nemo_toolkit[asr]",
            (PROJECT_ROOT / "requirements" / "calibration.txt").read_text(),
        )
        for old_path in (
            "onnx_requirements.txt",
            "test_requirements.txt",
            "calibration_requirements.txt",
            "dev_requirements.txt",
            "requirements.txt",
        ):
            self.assertFalse((PROJECT_ROOT / old_path).exists(), old_path)


# class CalibrationFixtureTest(unittest.TestCase):
#     def test_calibration_fixture_is_short_mono_16khz_pcm(self):
#         fixture = PROJECT_ROOT / "tests" / "fixtures" / "bacsidatnhkhoavitadoc_1.wav"
#         self.assertTrue(fixture.is_file())

#         with wave.open(str(fixture), "rb") as wav_file:
#             self.assertEqual(wav_file.getframerate(), 16000)
#             self.assertEqual(wav_file.getnchannels(), 1)
#             self.assertEqual(wav_file.getsampwidth(), 2)
#             duration = wav_file.getnframes() / wav_file.getframerate()
#         self.assertGreaterEqual(duration, 7.5)
#         self.assertLessEqual(duration, 8.5)


class DockerAndWorkflowTest(unittest.TestCase):
    def test_cpu_dockerfiles_install_the_correct_dependency_profiles(self):
        dockerfiles = {
            path.name: path.read_text(encoding="utf-8")
            for path in (PROJECT_ROOT / "docker").glob("Dockerfile*")
        }
        try:
            onnx_dockerfile = dockerfiles["Dockerfile_onnx_cpu"]
        except Exception as e:
            print(dockerfiles)
            raise e
        calibration_dockerfile = dockerfiles["Dockerfile_calibration_cpu"]
        onnx_test_script = (
            PROJECT_ROOT / "scripts" / "ci" / "run_onnx_tests.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("requirements/test.txt", onnx_dockerfile)
        self.assertIn("run_onnx_tests.sh", onnx_dockerfile)
        self.assertIn("python -m unittest discover", onnx_test_script)
        self.assertIn("requirements/calibration.txt", calibration_dockerfile)
        self.assertNotIn("COPY .onnx_ckpt", calibration_dockerfile)
        calibration_gpu_dockerfile = dockerfiles["Dockerfile_calibration_gpu"]
        self.assertIn("pytorch/pytorch", calibration_gpu_dockerfile)
        self.assertIn("requirements/calibration.txt", calibration_gpu_dockerfile)
        self.assertIn("onnxruntime-gpu==1.26.0", calibration_gpu_dockerfile)
        for name, contents in dockerfiles.items():
            self.assertIn("COPY requirements/ ./requirements/", contents, name)
            self.assertIn("COPY exports/ ./exports/", contents, name)
            self.assertIn(
                'org.opencontainers.image.ref.name="development"', contents, name
            )
            self.assertIn("org.opencontainers.image.description=", contents, name)

    def test_calibration_script_exports_transcript_report_path(self):
        calibration_script = (
            PROJECT_ROOT / "scripts" / "ci" / "run_calibration.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'CALIBRATION_REPORT="${NEMOTRON_CALIBRATION_REPORT:-/app/ci-logs/asr_calibration_report.json}"',
            calibration_script,
        )
        self.assertIn(
            'NEMOTRON_CALIBRATION_REPORT="${CALIBRATION_REPORT}"',
            calibration_script,
        )
        self.assertIn(
            'CALIBRATION_TEST_TARGET="${NEMOTRON_CALIBRATION_TEST_TARGET:-tests.calibration.asr.test_model_calibration.NemotronONNXCalibrationTest}"',
            calibration_script,
        )
        self.assertIn(
            'python -m unittest "${CALIBRATION_TEST_TARGET}" -v',
            calibration_script,
        )
        self.assertIn("=== Calibration unittest environment ===", calibration_script)
        self.assertIn("importlib.util.find_spec", calibration_script)
        self.assertNotIn(
            'python -m unittest discover -s tests -p "test_*.py"',
            calibration_script,
        )

    def test_modal_calibration_runner_uses_gpu_dockerfile_and_device_env(self):
        modal_runner = (
            PROJECT_ROOT / "scripts" / "ci" / "run_modal_asr_calibration.py"
        ).read_text(encoding="utf-8")

        self.assertIn("modal.Image.from_dockerfile", modal_runner)
        self.assertIn("Dockerfile_calibration_gpu", modal_runner)
        self.assertIn("MODAL_CALIBRATION_GPU", modal_runner)
        self.assertIn("NEMOTRON_NATIVE_DEVICE", modal_runner)
        self.assertIn("cuda", modal_runner)
        self.assertIn("bash scripts/ci/run_calibration.sh", modal_runner)
        self.assertIn("ci-logs/asr_calibration_report.json", modal_runner)
        self.assertIn("Environment summary", modal_runner)
        self.assertIn("Report exists", modal_runner)
        self.assertNotIn('print("result: "', modal_runner)

    def test_pipeline_calibration_supports_qwen3_asr_backend(self):
        calibration_script = (
            PROJECT_ROOT / "scripts" / "ci" / "run_pipeline_calibration.sh"
        ).read_text(encoding="utf-8")
        modal_runner = (
            PROJECT_ROOT / "scripts" / "ci" / "run_modal_pipeline_calibration.py"
        ).read_text(encoding="utf-8")
        old_workflow = (
            PROJECT_ROOT / ".github" / "workflows" / "pipeline-calibration.yml"
        ).read_text(encoding="utf-8")
        qwen3_workflow = (
            PROJECT_ROOT / ".github" / "workflows" / "qwen3-pipeline-calibration.yml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'ASR_BACKEND="${PIPELINE_ASR_BACKEND:-nemotron_onnx}"',
            calibration_script,
        )
        self.assertIn('qwen3_modal)', calibration_script)
        self.assertIn("Skipping ASR ONNX export for Qwen3", calibration_script)
        self.assertIn("Qwen3PipelineCalibrationTest", calibration_script)
        self.assertIn("PIPELINE_ASR_BACKEND", modal_runner)
        self.assertIn('"qwen3_modal"', modal_runner)
        self.assertIn("print(log_text, flush=True)", modal_runner)
        self.assertNotIn("qwen3_modal", old_workflow)
        self.assertNotIn("qwen3-pipeline-calibration", old_workflow)
        self.assertIn(
            "if: github.event_name != 'pull_request' || github.head_ref != 'qwen_asr'",
            old_workflow,
        )
        self.assertIn("name: Qwen3 Pipeline Calibration", qwen3_workflow)
        self.assertIn("pull_request:\n    branches:\n      - main", qwen3_workflow)
        self.assertIn("push:\n    branches:\n      - qwen_asr", qwen3_workflow)
        self.assertIn(
            "if: github.event_name != 'pull_request' || github.head_ref == 'qwen_asr'",
            qwen3_workflow,
        )
        self.assertNotIn("workflow_dispatch", qwen3_workflow)
        self.assertNotIn("schedule:", qwen3_workflow)
        self.assertIn("PIPELINE_ASR_BACKEND: qwen3_modal", qwen3_workflow)
        self.assertIn(
            "PIPELINE_CALIBRATION_TEST_TARGET: tests.calibration.pipeline.test_calibration.Qwen3PipelineCalibrationTest",
            qwen3_workflow,
        )
        self.assertIn(
            "MODAL_PIPELINE_CALIBRATION_VOLUME: speaker_diar_ci_pipeline_qwen3_calibration",
            qwen3_workflow,
        )
        self.assertIn(
            'DIAR_EXPORT_VOLUME_NAME="${MODAL_PIPELINE_CALIBRATION_VOLUME}"',
            qwen3_workflow,
        )
        self.assertNotIn("MODAL_PIPELINE_QWEN3_CALIBRATION_VOLUME", qwen3_workflow)

if __name__ == "__main__":
    unittest.main()
