import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str | Path, model: torch.nn.Module, config: dict, epoch: int) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(Path(path), map_location=device)
