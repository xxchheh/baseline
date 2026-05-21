from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dcase_ae.features import FeatureConfig, file_to_vectors
from dcase_ae.audio import load_audio


WAV_EXTENSIONS = {".wav", ".wave"}


@dataclass(frozen=True)
class FileFeatures:
    path: Path
    vectors: np.ndarray


def list_wav_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in WAV_EXTENSIONS
    )


def load_file_features(directory: str | Path, cfg: FeatureConfig) -> list[FileFeatures]:
    files = list_wav_files(directory)
    if not files:
        raise FileNotFoundError(f"No wav files found under: {directory}")

    features = []
    for path in files:
        vectors = file_to_vectors(path, cfg)
        if len(vectors) == 0:
            continue
        features.append(FileFeatures(path=path, vectors=vectors))
    if not features:
        raise RuntimeError(f"All wav files under {directory} were too short for feature extraction.")
    return features


class LocalDCASEFrameDataset(Dataset):
    def __init__(self, directory: str | Path, cfg: FeatureConfig):
        self.file_features = load_file_features(directory, cfg)
        self.data = np.concatenate([item.vectors for item in self.file_features], axis=0)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.data[index]).float()


class LocalDCASEFileDataset(Dataset):
    def __init__(self, directory: str | Path, cfg: FeatureConfig):
        self.file_features = load_file_features(directory, cfg)

    def __len__(self) -> int:
        return len(self.file_features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        item = self.file_features[index]
        return torch.from_numpy(item.vectors).float(), item.path.name


class LocalDCASEAudioDataset(Dataset):
    def __init__(
        self,
        directory: str | Path,
        sample_rate: int = 16000,
        mono: bool = True,
    ):
        self.files = list_wav_files(directory)
        if not self.files:
            raise FileNotFoundError(f"No wav files found under: {directory}")
        self.sample_rate = sample_rate
        self.mono = mono

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        path = self.files[index]
        waveform, _ = load_audio(path, mono=self.mono, sample_rate=self.sample_rate)
        return waveform, path.name


def audio_collate_fn(batch: list[tuple[np.ndarray, str]]) -> tuple[list[np.ndarray], list[str]]:
    waveforms, basenames = zip(*batch)
    return list(waveforms), list(basenames)


class LocalDCASEDataModule:
    def __init__(
        self,
        data_dir: str | Path,
        feature_cfg: FeatureConfig,
        batch_size: int = 256,
        validation_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 0,
        seed: int = 13711,
    ):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"
        self.feature_cfg = feature_cfg
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed

        if not self.train_dir.is_dir():
            raise FileNotFoundError(f"Expected train directory: {self.train_dir}")
        if not self.test_dir.is_dir():
            raise FileNotFoundError(f"Expected test directory: {self.test_dir}")

    def train_valid_loaders(self) -> tuple[DataLoader, DataLoader]:
        dataset = LocalDCASEFrameDataset(self.train_dir, self.feature_cfg)
        valid_len = int(len(dataset) * self.validation_split)
        train_len = len(dataset) - valid_len
        if train_len <= 0:
            raise RuntimeError("Training split is empty. Reduce validation_split or add more wav files.")

        if valid_len == 0:
            train_dataset = dataset
            valid_dataset = dataset
        else:
            generator = torch.Generator().manual_seed(self.seed)
            train_dataset, valid_dataset = random_split(
                dataset,
                [train_len, valid_len],
                generator=generator,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, valid_loader

    def test_loader(self) -> DataLoader:
        dataset = LocalDCASEFileDataset(self.test_dir, self.feature_cfg)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def train_audio_loader(self, sample_rate: int, mono: bool = True) -> DataLoader:
        dataset = LocalDCASEAudioDataset(self.train_dir, sample_rate=sample_rate, mono=mono)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=audio_collate_fn,
        )

    def test_audio_loader(self, sample_rate: int, mono: bool = True) -> DataLoader:
        dataset = LocalDCASEAudioDataset(self.test_dir, sample_rate=sample_rate, mono=mono)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=audio_collate_fn,
        )
