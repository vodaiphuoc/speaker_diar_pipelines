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
            'CALIBRATION_TEST_TARGET="${NEMOTRON_CALIBRATION_TEST_TARGET:-tests.test_asr_calibration.NemotronONNXCalibrationTest}"',
            calibration_script,
        )
        self.assertIn('python -m unittest "${CALIBRATION_TEST_TARGET}" -v', calibration_script)
        self.assertNotIn('python -m unittest discover -s tests -p "test_*.py"', calibration_script)

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

    def test_workflow_runs_calibration_on_modal_and_uploads_logs(self):
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("MODAL_TOKEN_ID", workflow)
        self.assertIn("MODAL_TOKEN_SECRET", workflow)
        self.assertIn("scripts/ci/run_modal_asr_calibration.py", workflow)
        self.assertIn("if: always()", workflow)
        self.assertNotIn("Dockerfile_calibration_cpu", workflow)

if __name__ == "__main__":
    unittest.main()
