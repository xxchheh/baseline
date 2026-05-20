from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    path = Path(path)
    y, sr = librosa.load(path, sr=None, mono=mono)
    if not mono and y.ndim > 1:
        y = y[0]
    return y, sr
