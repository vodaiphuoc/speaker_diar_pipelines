import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from jsonschema import Draft202012Validator
from onnx import helper, numpy_helper

from SDP.onnx.artifacts import (
    ArtifactManifestError,
    discover_onnx_external_data,
    load_asr_artifact_manifest,
    load_diarization_artifact_manifest,
    write_asr_artifact_manifest,
    write_diarization_artifact_manifest,
)
from SDP.onnx.asr.streaming import (
    ASRModelPaths,
    create_nemotron_streaming_session_from_manifest,
)
from SDP.onnx.streaming_service import (
    StreamingDiarizationASROnnxService,
    StreamingDiarizerOnnxService,
)


PROJECT_ROOT = Path(__file__).parents[1]
SCHEMA_PATH = (
    PROJECT_ROOT / "SDP" / "onnx" / "schemas" / "artifact_manifest.schema.json"
)


def write_bytes(path: Path, contents: bytes = b"runtime-asset") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def write_onnx(path: Path, external_data_name: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    weight = numpy_helper.from_array(
        np.arange(16, dtype=np.float32).reshape(4, 4), name="weight"
    )
    graph = helper.make_graph(
        nodes=[],
        name="artifact-test",
        inputs=[],
        outputs=[],
        initializer=[weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20)],
    )
    if external_data_name is None:
        onnx.save_model(model, str(path))
    else:
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            size_threshold=0,
            location=external_data_name,
        )
    return path


class ArtifactFixtures:
    def make_asr_assets(self, root: Path) -> dict[str, Path]:
        return {
            "preprocessor": write_onnx(root / "asr_preprocessor.onnx"),
            "encoder": write_onnx(
                root / "encoder.onnx", "encoder_weights.generated.data"
            ),
            "prompt_projection": write_onnx(root / "prompt_projection.onnx"),
            "decoder_joint": write_onnx(root / "decoder_joint.onnx"),
            "config": write_bytes(root / "asr_config.yaml"),
            "tokenizer": write_bytes(root / "tokenizer.model"),
        }

    def make_diarization_assets(self, root: Path) -> dict[str, Path]:
        return {
            "preprocessor": write_onnx(root / "diar_preprocessor.onnx"),
            "sortformer": write_onnx(root / "sortformer.onnx"),
            "config": write_bytes(root / "diar_config.yaml"),
        }


class ArtifactManifestTest(ArtifactFixtures, unittest.TestCase):
    def test_discovers_external_data_from_onnx_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            onnx_path = write_onnx(
                root / "preprocessor.onnx", "preprocessor.onnx.data"
            )

            external_data = discover_onnx_external_data(onnx_path)

            self.assertEqual(external_data, (root / "preprocessor.onnx.data",))

    def test_asr_export_manifest_is_schema_valid_and_loads_relative_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_asr_assets(root)

            manifest_path = write_asr_artifact_manifest(
                output_dir=root,
                source_model="nvidia/nemotron-test",
                preprocessor=assets["preprocessor"],
                encoder=assets["encoder"],
                prompt_projection=assets["prompt_projection"],
                decoder_joint=assets["decoder_joint"],
                config=assets["config"],
                tokenizer=assets["tokenizer"],
            )
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            Draft202012Validator(schema).validate(data)
            artifact = load_asr_artifact_manifest(manifest_path)

            self.assertEqual(manifest_path, root / "asr_artifact.json")
            self.assertEqual(artifact.pipeline, "asr")
            self.assertEqual(artifact.preprocessor.onnx, assets["preprocessor"])
            self.assertEqual(
                artifact.encoder.external_data,
                (root / "encoder_weights.generated.data",),
            )
            self.assertEqual(artifact.config, assets["config"])
            self.assertEqual(artifact.tokenizer, assets["tokenizer"])
            self.assertFalse(Path(data["components"]["encoder"]["onnx"]).is_absolute())

    def test_diarization_export_manifest_is_separate_and_schema_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_diarization_assets(root)

            manifest_path = write_diarization_artifact_manifest(
                output_dir=root,
                source_model="nvidia/diar-test",
                preprocessor=assets["preprocessor"],
                sortformer=assets["sortformer"],
                config=assets["config"],
            )
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            Draft202012Validator(schema).validate(data)
            artifact = load_diarization_artifact_manifest(manifest_path)

            self.assertEqual(manifest_path, root / "diarization_artifact.json")
            self.assertEqual(artifact.pipeline, "diarization")
            self.assertEqual(artifact.preprocessor.onnx, assets["preprocessor"])
            self.assertEqual(artifact.sortformer.onnx, assets["sortformer"])
            self.assertNotIn("tokenizer", data["runtime_assets"])

    def test_loader_rejects_wrong_pipeline_and_unsupported_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_diarization_assets(root)
            manifest_path = write_diarization_artifact_manifest(
                output_dir=root,
                source_model="nvidia/diar-test",
                preprocessor=assets["preprocessor"],
                sortformer=assets["sortformer"],
                config=assets["config"],
            )

            with self.assertRaisesRegex(ArtifactManifestError, "pipeline"):
                load_asr_artifact_manifest(manifest_path)

            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["format_version"] = 2
            manifest_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactManifestError, "format_version"):
                load_diarization_artifact_manifest(manifest_path)

    def test_loader_rejects_absolute_traversal_missing_and_empty_assets(self):
        invalid_paths = (
            "/tmp/model.onnx",
            "../outside/model.onnx",
            "missing.onnx",
            "empty.onnx",
        )
        for invalid_path in invalid_paths:
            with self.subTest(path=invalid_path), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                assets = self.make_diarization_assets(root)
                manifest_path = write_diarization_artifact_manifest(
                    output_dir=root,
                    source_model="nvidia/diar-test",
                    preprocessor=assets["preprocessor"],
                    sortformer=assets["sortformer"],
                    config=assets["config"],
                )
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                data["components"]["sortformer"]["onnx"] = invalid_path
                if invalid_path == "empty.onnx":
                    (root / invalid_path).touch()
                manifest_path.write_text(json.dumps(data), encoding="utf-8")

                with self.assertRaises(ArtifactManifestError):
                    load_diarization_artifact_manifest(manifest_path)

    def test_loader_rejects_windows_style_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_diarization_assets(root)
            manifest_path = write_diarization_artifact_manifest(
                output_dir=root,
                source_model="nvidia/diar-test",
                preprocessor=assets["preprocessor"],
                sortformer=assets["sortformer"],
                config=assets["config"],
            )
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["components"]["sortformer"]["onnx"] = "..\\outside\\model.onnx"
            manifest_path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ArtifactManifestError, "escapes"):
                load_diarization_artifact_manifest(manifest_path)

    def test_loader_rejects_unexpected_top_level_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_diarization_assets(root)
            manifest_path = write_diarization_artifact_manifest(
                output_dir=root,
                source_model="nvidia/diar-test",
                preprocessor=assets["preprocessor"],
                sortformer=assets["sortformer"],
                config=assets["config"],
            )
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["training_checkpoint"] = "model.nemo"
            manifest_path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ArtifactManifestError, "manifest keys"):
                load_diarization_artifact_manifest(manifest_path)

    def test_writer_fails_when_onnx_external_data_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_asr_assets(root)
            (root / "encoder_weights.generated.data").unlink()

            with self.assertRaisesRegex(ArtifactManifestError, "external data"):
                write_asr_artifact_manifest(
                    output_dir=root,
                    source_model="nvidia/nemotron-test",
                    preprocessor=assets["preprocessor"],
                    encoder=assets["encoder"],
                    prompt_projection=assets["prompt_projection"],
                    decoder_joint=assets["decoder_joint"],
                    config=assets["config"],
                    tokenizer=assets["tokenizer"],
                )


class ManifestFactoryTest(ArtifactFixtures, unittest.TestCase):
    def test_asr_factory_maps_complete_manifest_to_existing_constructor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_asr_assets(root)
            manifest_path = write_asr_artifact_manifest(
                output_dir=root,
                source_model="nvidia/nemotron-test",
                preprocessor=assets["preprocessor"],
                encoder=assets["encoder"],
                prompt_projection=assets["prompt_projection"],
                decoder_joint=assets["decoder_joint"],
                config=assets["config"],
                tokenizer=assets["tokenizer"],
            )
            sentinel = object()

            with patch(
                "SDP.onnx.asr.streaming.create_nemotron_streaming_session",
                return_value=sentinel,
            ) as create_session:
                result = create_nemotron_streaming_session_from_manifest(
                    manifest_path,
                    device="cpu",
                    target_language="vi-VN",
                )

            self.assertIs(result, sentinel)
            create_session.assert_called_once_with(
                ASRModelPaths(
                    preprocessor=str(assets["preprocessor"]),
                    encoder=str(assets["encoder"]),
                    prompt_projection=str(assets["prompt_projection"]),
                    decoder_joint=str(assets["decoder_joint"]),
                    tokenizer=str(assets["tokenizer"]),
                ),
                config_path=str(assets["config"]),
                device="cpu",
                target_language="vi-VN",
            )

    def test_diarization_factory_loads_config_and_uses_its_own_preprocessor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = self.make_diarization_assets(root)
            manifest_path = write_diarization_artifact_manifest(
                output_dir=root,
                source_model="nvidia/diar-test",
                preprocessor=assets["preprocessor"],
                sortformer=assets["sortformer"],
                config=assets["config"],
            )
            encoder_config = object()
            sortformer_config = object()
            preprocessor_config = object()

            with (
                patch(
                    "SDP.onnx.streaming_service.load_encoder_modules_config",
                    return_value=encoder_config,
                ) as load_encoder,
                patch(
                    "SDP.onnx.streaming_service.load_sortformer_modules_config",
                    return_value=sortformer_config,
                ) as load_sortformer,
                patch(
                    "SDP.onnx.streaming_service.load_preprocessor_config",
                    return_value=preprocessor_config,
                ) as load_preprocessor,
                patch.object(
                    StreamingDiarizerOnnxService, "__init__", return_value=None
                ) as init_service,
            ):
                service = StreamingDiarizerOnnxService.from_manifest(
                    manifest_path,
                    device="cpu",
                    enable_async_queue=True,
                )

            self.assertIsInstance(service, StreamingDiarizerOnnxService)
            for loader in (load_encoder, load_sortformer, load_preprocessor):
                loader.assert_called_once_with(str(assets["config"]))
            init_service.assert_called_once()
            kwargs = init_service.call_args.kwargs
            self.assertEqual(kwargs["modal_ckpt_path"], str(assets["sortformer"]))
            self.assertEqual(
                kwargs["preprocessor_ckpt_path"], str(assets["preprocessor"])
            )
            self.assertIs(kwargs["encoder_config"], encoder_config)
            self.assertIs(kwargs["sortformer_config"], sortformer_config)
            self.assertIs(kwargs["preprocessor_config"], preprocessor_config)
            self.assertTrue(kwargs["enable_async_queue"])

    def test_combined_factory_builds_branches_from_separate_manifests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asr_root = root / "asr"
            diar_root = root / "diar"
            asr_assets = self.make_asr_assets(asr_root)
            diar_assets = self.make_diarization_assets(diar_root)
            asr_manifest = write_asr_artifact_manifest(
                output_dir=asr_root,
                source_model="nvidia/nemotron-test",
                preprocessor=asr_assets["preprocessor"],
                encoder=asr_assets["encoder"],
                prompt_projection=asr_assets["prompt_projection"],
                decoder_joint=asr_assets["decoder_joint"],
                config=asr_assets["config"],
                tokenizer=asr_assets["tokenizer"],
            )
            diar_manifest = write_diarization_artifact_manifest(
                output_dir=diar_root,
                source_model="nvidia/diar-test",
                preprocessor=diar_assets["preprocessor"],
                sortformer=diar_assets["sortformer"],
                config=diar_assets["config"],
            )
            diar_service = object()
            asr_session = object()

            with (
                patch.object(
                    StreamingDiarizerOnnxService,
                    "from_manifest",
                    return_value=diar_service,
                ) as create_diar,
                patch(
                    "SDP.onnx.streaming_service."
                    "create_nemotron_streaming_session_from_manifest",
                    return_value=asr_session,
                ) as create_asr,
            ):
                service = StreamingDiarizationASROnnxService.from_manifests(
                    diarization_manifest_path=diar_manifest,
                    asr_manifest_path=asr_manifest,
                    device="cpu",
                    target_language="vi-VN",
                )

            self.assertIs(service.diarization_service, diar_service)
            self.assertIs(service.asr_session, asr_session)
            create_diar.assert_called_once()
            self.assertEqual(create_diar.call_args.args[0], diar_manifest)
            create_asr.assert_called_once_with(
                asr_manifest,
                device="cpu",
                target_language="vi-VN",
            )

    def test_combined_factory_rejects_shared_preprocessor_onnx(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asr_assets = self.make_asr_assets(root)
            diar_assets = self.make_diarization_assets(root)
            shared_preprocessor = asr_assets["preprocessor"]
            asr_manifest = write_asr_artifact_manifest(
                output_dir=root,
                source_model="nvidia/nemotron-test",
                preprocessor=shared_preprocessor,
                encoder=asr_assets["encoder"],
                prompt_projection=asr_assets["prompt_projection"],
                decoder_joint=asr_assets["decoder_joint"],
                config=asr_assets["config"],
                tokenizer=asr_assets["tokenizer"],
            )
            diar_manifest = write_diarization_artifact_manifest(
                output_dir=root,
                source_model="nvidia/diar-test",
                preprocessor=shared_preprocessor,
                sortformer=diar_assets["sortformer"],
                config=diar_assets["config"],
            )

            with self.assertRaisesRegex(ValueError, "independent preprocessors"):
                StreamingDiarizationASROnnxService.from_manifests(
                    diarization_manifest_path=diar_manifest,
                    asr_manifest_path=asr_manifest,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
