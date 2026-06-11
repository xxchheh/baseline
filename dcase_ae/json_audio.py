import json
from math import gcd
import wave
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_JSON_SAMPLE_RATE = 44800
SAMPLE_KEYS = ("audio", "samples", "waveform", "signal", "values", "data")
SAMPLE_RATE_KEYS = ("sample_rate", "sampling_rate", "sr", "fs", "rate")


def load_json_audio(
    path: str | Path,
    default_sample_rate: int = DEFAULT_JSON_SAMPLE_RATE,
    target_sample_rate: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = _read_json(path)
    sample_rate = _find_sample_rate(payload) or default_sample_rate
    samples = _find_samples(payload)
    waveform = _to_mono_float(samples)

    original_sample_rate = int(sample_rate)
    resampled = False
    if target_sample_rate is not None and target_sample_rate != original_sample_rate:
        waveform = resample_audio(
            waveform,
            original_sample_rate,
            target_sample_rate,
        )
        sample_rate = int(target_sample_rate)
        resampled = True

    metadata = {
        "input_sample_rate": original_sample_rate,
        "sample_rate": int(sample_rate),
        "resampled": resampled,
        "num_samples": int(waveform.shape[0]),
    }
    return waveform, metadata


def write_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("write_wav expects a mono waveform.")
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return path


def resample_audio(waveform: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if original_sample_rate == target_sample_rate:
        return waveform.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        divisor = gcd(int(original_sample_rate), int(target_sample_rate))
        up = int(target_sample_rate) // divisor
        down = int(original_sample_rate) // divisor
        return resample_poly(waveform, up, down).astype(np.float32, copy=False)
    except ImportError:
        duration = waveform.shape[0] / float(original_sample_rate)
        target_length = max(1, int(round(duration * target_sample_rate)))
        source_positions = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, waveform).astype(np.float32, copy=False)


def _read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_sample_rate(payload: Any) -> int | None:
    if isinstance(payload, dict):
        for key in SAMPLE_RATE_KEYS:
            if key in payload:
                return int(payload[key])
        for value in payload.values():
            sample_rate = _find_sample_rate(value)
            if sample_rate is not None:
                return sample_rate
    return None


def _find_samples(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("JSON audio must be a list or an object containing audio samples.")
    for key in SAMPLE_KEYS:
        if key in payload:
            candidate = payload[key]
            if isinstance(candidate, dict):
                try:
                    return _find_samples(candidate)
                except ValueError:
                    pass
            else:
                return candidate
    for value in payload.values():
        if isinstance(value, (dict, list)):
            try:
                return _find_samples(value)
            except ValueError:
                continue
    raise ValueError(
        "Could not find audio samples in JSON. Expected one of: "
        + ", ".join(SAMPLE_KEYS)
    )


def _to_mono_float(samples: Any) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.size == 0:
        raise ValueError("JSON audio samples are empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("JSON audio samples must be numeric.")
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    elif arr.ndim != 1:
        arr = arr.reshape(-1)

    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 1.0:
        arr = arr / _guess_pcm_scale(max_abs)
    return arr.astype(np.float32, copy=False)


def _guess_pcm_scale(max_abs: float) -> float:
    if max_abs <= 128:
        return 128.0
    if max_abs <= 32768:
        return 32768.0
    if max_abs <= 2147483648:
        return 2147483648.0
    return max_abs
