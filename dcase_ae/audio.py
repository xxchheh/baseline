from pathlib import Path

import librosa
import numpy as np


def load_audio(
    path: str | Path,
    mono: bool = True,
    sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    path = Path(path)
    y, sr = librosa.load(path, sr=sample_rate, mono=mono)
    if not mono and y.ndim > 1:
        y = y[0]
    return y.astype(np.float32, copy=False), sr
