import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from dcase_ae.audio import load_audio


@dataclass(frozen=True)
class FeatureConfig:
    n_mels: int = 128
    frames: int = 5
    frame_hop_length: int = 1
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0
    fmin: float = 0.0
    fmax: float | None = None
    win_length: int | None = None
    mono: bool = True

    @property
    def input_dim(self) -> int:
        return self.n_mels * self.frames


def file_to_vectors(file_name: str | Path, cfg: FeatureConfig) -> np.ndarray:
    y, sr = load_audio(file_name, mono=cfg.mono)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=cfg.power,
        fmax=cfg.fmax,
        fmin=cfg.fmin,
        win_length=cfg.win_length,
    )
    log_mel_spectrogram = 20.0 / cfg.power * np.log10(
        np.maximum(mel_spectrogram, sys.float_info.epsilon)
    )

    n_vectors = log_mel_spectrogram.shape[-1] - cfg.frames + 1
    if n_vectors < 1:
        return np.empty((0, cfg.input_dim), dtype=np.float32)

    vectors = np.zeros((n_vectors, cfg.input_dim), dtype=np.float32)
    for t in range(cfg.frames):
        vectors[:, cfg.n_mels * t : cfg.n_mels * (t + 1)] = log_mel_spectrogram[
            :, t : t + n_vectors
        ].T
    return vectors[:: cfg.frame_hop_length].astype(np.float32, copy=False)
