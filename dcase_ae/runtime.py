import os
from dataclasses import dataclass
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    libraries_root: Path
    data_train_root: Path
    enable_library_training: bool = False
    use_cuda: bool = True
    default_library_name: str = "Default_libraries"

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        root = _project_root()
        return cls(
            libraries_root=Path(os.getenv("DCASE_LIBRARIES_ROOT", root / "health_libraries")),
            data_train_root=Path(os.getenv("DCASE_DATA_TRAIN_ROOT", root / "data_train")),
            enable_library_training=_env_bool("DCASE_ENABLE_LIBRARY_TRAINING", False),
            use_cuda=_env_bool("DCASE_USE_CUDA", True),
            default_library_name=os.getenv("DCASE_DEFAULT_LIBRARY_NAME", "Default_libraries"),
        )

    def library_dir(self, machine_type: str, library_name: str | None = None) -> Path:
        return self.libraries_root / (library_name or self.default_library_name) / machine_type

    def training_dir(self, machine_type: str, library_name: str | None = None) -> Path:
        return self.data_train_root / (library_name or self.default_library_name) / machine_type
