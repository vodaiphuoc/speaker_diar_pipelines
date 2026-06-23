from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal


ARTIFACT_FORMAT_VERSION = 1
ASR_ARTIFACT_FILENAME = "asr_artifact.json"
DIARIZATION_ARTIFACT_FILENAME = "diarization_artifact.json"


class ArtifactManifestError(ValueError):
    """Raised when an ONNX runtime artifact manifest is invalid."""


@dataclass(frozen=True)
class OnnxComponentArtifact:
    onnx: Path
    external_data: tuple[Path, ...]


@dataclass(frozen=True)
class ASRArtifactManifest:
    manifest_path: Path
    source_model: str
    preprocessor: OnnxComponentArtifact
    encoder: OnnxComponentArtifact
    prompt_projection: OnnxComponentArtifact
    decoder_joint: OnnxComponentArtifact
    config: Path
    tokenizer: Path
    format_version: int = ARTIFACT_FORMAT_VERSION
    pipeline: Literal["asr"] = "asr"


@dataclass(frozen=True)
class DiarizationArtifactManifest:
    manifest_path: Path
    source_model: str
    preprocessor: OnnxComponentArtifact
    sortformer: OnnxComponentArtifact
    config: Path
    format_version: int = ARTIFACT_FORMAT_VERSION
    pipeline: Literal["diarization"] = "diarization"


def discover_onnx_external_data(onnx_path: str | Path) -> tuple[Path, ...]:
    """Return external tensor files referenced by an ONNX model."""
    import onnx

    model_path = Path(onnx_path).resolve()
    model = onnx.load(str(model_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {entry.key: entry.value for entry in tensor.external_data}
        location = metadata.get("location")
        if not location:
            raise ArtifactManifestError(
                f"ONNX external data entry has no location: {model_path}"
            )
        locations.add(location)
    return tuple(
        sorted((model_path.parent / location).resolve() for location in locations)
    )


def write_asr_artifact_manifest(
    *,
    output_dir: str | Path,
    source_model: str,
    preprocessor: str | Path,
    encoder: str | Path,
    prompt_projection: str | Path,
    decoder_joint: str | Path,
    config: str | Path,
    tokenizer: str | Path,
) -> Path:
    root = Path(output_dir).resolve()
    data = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "pipeline": "asr",
        "source_model": source_model,
        "components": {
            "preprocessor": _component_document(root, preprocessor),
            "encoder": _component_document(root, encoder),
            "prompt_projection": _component_document(root, prompt_projection),
            "decoder_joint": _component_document(root, decoder_joint),
        },
        "runtime_assets": {
            "config": _runtime_path(root, config, "config"),
            "tokenizer": _runtime_path(root, tokenizer, "tokenizer"),
        },
    }
    return _write_manifest(root / ASR_ARTIFACT_FILENAME, data)


def write_diarization_artifact_manifest(
    *,
    output_dir: str | Path,
    source_model: str,
    preprocessor: str | Path,
    sortformer: str | Path,
    config: str | Path,
) -> Path:
    root = Path(output_dir).resolve()
    data = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "pipeline": "diarization",
        "source_model": source_model,
        "components": {
            "preprocessor": _component_document(root, preprocessor),
            "sortformer": _component_document(root, sortformer),
        },
        "runtime_assets": {
            "config": _runtime_path(root, config, "config"),
        },
    }
    return _write_manifest(root / DIARIZATION_ARTIFACT_FILENAME, data)


def load_asr_artifact_manifest(
    manifest_path: str | Path,
) -> ASRArtifactManifest:
    path, data = _read_manifest(manifest_path, "asr")
    components = _require_mapping(data, "components")
    runtime_assets = _require_mapping(data, "runtime_assets")
    _require_keys(
        components,
        {"preprocessor", "encoder", "prompt_projection", "decoder_joint"},
        "components",
    )
    _require_keys(runtime_assets, {"config", "tokenizer"}, "runtime_assets")
    root = path.parent
    return ASRArtifactManifest(
        manifest_path=path,
        source_model=_require_nonempty_string(data, "source_model"),
        preprocessor=_load_component(root, components, "preprocessor"),
        encoder=_load_component(root, components, "encoder"),
        prompt_projection=_load_component(root, components, "prompt_projection"),
        decoder_joint=_load_component(root, components, "decoder_joint"),
        config=_resolve_runtime_path(root, runtime_assets.get("config"), "config"),
        tokenizer=_resolve_runtime_path(
            root, runtime_assets.get("tokenizer"), "tokenizer"
        ),
    )


def load_diarization_artifact_manifest(
    manifest_path: str | Path,
) -> DiarizationArtifactManifest:
    path, data = _read_manifest(manifest_path, "diarization")
    components = _require_mapping(data, "components")
    runtime_assets = _require_mapping(data, "runtime_assets")
    _require_keys(components, {"preprocessor", "sortformer"}, "components")
    _require_keys(runtime_assets, {"config"}, "runtime_assets")
    root = path.parent
    return DiarizationArtifactManifest(
        manifest_path=path,
        source_model=_require_nonempty_string(data, "source_model"),
        preprocessor=_load_component(root, components, "preprocessor"),
        sortformer=_load_component(root, components, "sortformer"),
        config=_resolve_runtime_path(root, runtime_assets.get("config"), "config"),
    )


def _component_document(root: Path, onnx_path: str | Path) -> dict[str, Any]:
    model_path = _validate_export_path(root, onnx_path, "ONNX model")
    external_paths = discover_onnx_external_data(model_path)
    external_paths = tuple(
        _validate_export_path(root, external_path, "ONNX external data")
        for external_path in external_paths
    )
    return {
        "onnx": model_path.relative_to(root).as_posix(),
        "external_data": [
            path.relative_to(root).as_posix() for path in external_paths
        ],
    }


def _runtime_path(root: Path, path: str | Path, label: str) -> str:
    return _validate_export_path(root, path, label).relative_to(root).as_posix()


def _validate_export_path(root: Path, path: str | Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactManifestError(
            f"{label} must be inside artifact directory {resolved_root}: {resolved}"
        ) from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ArtifactManifestError(f"Missing or empty {label}: {resolved}")
    return resolved


def _write_manifest(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            json.dump(data, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def _read_manifest(
    manifest_path: str | Path, expected_pipeline: str
) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ArtifactManifestError(f"Artifact manifest does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"Invalid artifact manifest: {path}") from exc
    if not isinstance(data, dict):
        raise ArtifactManifestError("Artifact manifest root must be an object")
    _require_keys(
        data,
        {
            "format_version",
            "pipeline",
            "source_model",
            "components",
            "runtime_assets",
        },
        "manifest",
    )
    if data.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ArtifactManifestError(
            f"Unsupported format_version: {data.get('format_version')!r}"
        )
    if data.get("pipeline") != expected_pipeline:
        raise ArtifactManifestError(
            f"Expected pipeline {expected_pipeline!r}, got {data.get('pipeline')!r}"
        )
    return path, data


def _load_component(
    root: Path, components: dict[str, Any], name: str
) -> OnnxComponentArtifact:
    value = components.get(name)
    if not isinstance(value, dict):
        raise ArtifactManifestError(f"Component {name!r} must be an object")
    _require_keys(value, {"onnx", "external_data"}, f"component {name!r}")
    external_values = value.get("external_data")
    if not isinstance(external_values, list) or not all(
        isinstance(item, str) for item in external_values
    ):
        raise ArtifactManifestError(
            f"Component {name!r} external_data must be an array of paths"
        )
    return OnnxComponentArtifact(
        onnx=_resolve_runtime_path(root, value.get("onnx"), f"{name} ONNX"),
        external_data=tuple(
            _resolve_runtime_path(root, item, f"{name} external data")
            for item in external_values
        ),
    )


def _resolve_runtime_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactManifestError(f"{label} path must be a non-empty string")
    candidate = Path(value)
    windows_candidate = PureWindowsPath(value)
    if candidate.is_absolute() or windows_candidate.is_absolute():
        raise ArtifactManifestError(f"{label} path must be relative: {value}")
    if ".." in candidate.parts or ".." in windows_candidate.parts:
        raise ArtifactManifestError(
            f"{label} path escapes artifact directory: {value}"
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactManifestError(
            f"{label} path escapes artifact directory: {value}"
        ) from exc
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ArtifactManifestError(f"Missing or empty {label}: {resolved}")
    return resolved


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ArtifactManifestError(f"{key} must be an object")
    return value


def _require_nonempty_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactManifestError(f"{key} must be a non-empty string")
    return value


def _require_keys(
    value: dict[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ArtifactManifestError(
            f"{label} keys must be {sorted(expected)}, got {sorted(actual)}"
        )
