import os

import torch


def int_sequence_or_none(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return tuple(int(item) for item in value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return None


def resolve_native_device():
    requested_device = os.environ.get("NEMOTRON_NATIVE_DEVICE", "cpu").strip()
    if not requested_device:
        requested_device = "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for native NeMo calibration via "
            "NEMOTRON_NATIVE_DEVICE, but torch.cuda.is_available() is false."
        )
    return device


def move_tensor_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def extract_native_transcription_texts(transcriptions):
    if transcriptions is None:
        return []
    if isinstance(transcriptions, str):
        return [transcriptions]
    if hasattr(transcriptions, "text"):
        return [transcriptions.text]

    texts = []
    for transcription in transcriptions:
        if hasattr(transcription, "text"):
            texts.append(transcription.text)
        else:
            texts.append(transcription)
    return texts
