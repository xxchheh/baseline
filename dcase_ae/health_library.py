import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from dcase_ae.json_audio import DEFAULT_JSON_SAMPLE_RATE, load_json_audio
from dcase_ae.knn_detector import KNNAnomalyDetector
from dcase_ae.scoring import RISK_THRESHOLDS, anomaly_result, health_score_from_reference


LIBRARY_VERSION = 1
REQUIRED_LIBRARY_FILES = (
    "metadata.json",
    "manifest.csv",
    "detector.joblib",
    "reference_scores.npy",
    "reference_embeddings.npy",
)


@dataclass(frozen=True)
class HealthLibraryConfig:
    pretrained_model_name: str = "microsoft/wavlm-base"
    input_sample_rate: int = DEFAULT_JSON_SAMPLE_RATE
    target_sample_rate: int = 16000
    knn_neighbors: int = 5
    pca_dim: int = 64
    batch_size: int = 8
    use_cuda: bool = True
    recursive: bool = True
    pattern: str = "*.json"


@dataclass(frozen=True)
class HealthAnalysis:
    status: str
    detector_type: str
    filename: str
    input_path: str
    raw_knn_distance: float
    abnormality_score: float
    abnormality_unit: str
    risk_level: str
    health_score: float
    health_level: str
    health_score_meaning: str
    score_meaning: str
    audio: dict[str, Any]
    library: dict[str, Any]
    risk_thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_health_library(
    reference_dir: str | Path,
    library_dir: str | Path,
    config: HealthLibraryConfig,
) -> dict[str, Any]:
    reference_dir = Path(reference_dir)
    library_dir = _ensure_dir(library_dir)
    files = _iter_json_files(reference_dir, config.pattern, config.recursive)
    if len(files) < 2:
        raise ValueError(f"Need at least two JSON reference files in {reference_dir}")

    from dcase_ae.embedder import PretrainedAudioEmbedder
    from dcase_ae.utils import get_device

    device = get_device(config.use_cuda)
    embedder = PretrainedAudioEmbedder(
        model_name=config.pretrained_model_name,
        sample_rate=config.target_sample_rate,
        device=device,
    )

    all_embeddings = []
    manifest_rows = []
    for start in range(0, len(files), config.batch_size):
        batch_files = files[start : start + config.batch_size]
        waveforms = []
        batch_metadata = []
        for path in batch_files:
            waveform, metadata = load_json_audio(
                path,
                default_sample_rate=config.input_sample_rate,
                target_sample_rate=config.target_sample_rate,
            )
            waveforms.append(waveform)
            batch_metadata.append(metadata)

        batch_embeddings = embedder.extract(waveforms)
        all_embeddings.append(batch_embeddings)
        for path, metadata in zip(batch_files, batch_metadata):
            manifest_rows.append(
                {
                    "index": len(manifest_rows),
                    "file_path": str(path),
                    "input_sample_rate": metadata["input_sample_rate"],
                    "sample_rate": metadata["sample_rate"],
                    "num_samples": metadata["num_samples"],
                    "resampled": metadata["resampled"],
                }
            )
        print(f"Extracted embeddings {min(start + len(batch_files), len(files))}/{len(files)}", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    detector = KNNAnomalyDetector.fit(
        embeddings,
        n_neighbors=config.knn_neighbors,
        pca_dim=config.pca_dim,
    )
    reference_scores = detector.reference_scores().astype(np.float32, copy=False)

    joblib.dump(
        {
            "detector_type": "knn",
            "detector": detector,
        },
        library_dir / "detector.joblib",
    )
    np.save(library_dir / "reference_scores.npy", reference_scores)
    np.save(library_dir / "reference_embeddings.npy", detector.normal_embeddings.astype(np.float32, copy=False))
    _write_manifest(library_dir / "manifest.csv", manifest_rows)

    metadata = {
        "library_version": LIBRARY_VERSION,
        "detector_type": "knn",
        "method": "pretrained-audio-embedding + scaler + PCA + KNN",
        "reference_dir": str(reference_dir),
        "reference_count": len(files),
        "embedding_dim": int(embeddings.shape[1]),
        "transformed_embedding_dim": int(detector.normal_embeddings.shape[1]),
        "pretrained_model_name": config.pretrained_model_name,
        "input_sample_rate": int(config.input_sample_rate),
        "target_sample_rate": int(config.target_sample_rate),
        "knn_neighbors": int(config.knn_neighbors),
        "pca_dim": int(config.pca_dim),
        "batch_size": int(config.batch_size),
        "recursive": bool(config.recursive),
        "pattern": config.pattern,
        "score_reference": _reference_summary(reference_scores),
    }
    (library_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


class HealthLibraryError(RuntimeError):
    def __init__(self, status_code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "error",
            "status_code": self.status_code,
            "message": self.message,
        }


class HealthAnalyzer:
    def __init__(
        self,
        library_dir: str | Path,
        *,
        use_cuda: bool = True,
    ):
        self.library_dir = Path(library_dir)
        self._validate_library()
        self.metadata = json.loads((self.library_dir / "metadata.json").read_text(encoding="utf-8"))
        if int(self.metadata.get("library_version", 0)) != LIBRARY_VERSION:
            raise HealthLibraryError(
                "health_library_version_mismatch",
                f"Unsupported health library version: {self.metadata.get('library_version')}",
            )

        checkpoint = joblib.load(self.library_dir / "detector.joblib")
        if not isinstance(checkpoint, dict) or checkpoint.get("detector_type") != "knn":
            raise HealthLibraryError("health_library_invalid_detector", "detector.joblib is not a KNN detector.")
        self.detector = checkpoint["detector"]
        self.reference_scores = np.load(self.library_dir / "reference_scores.npy")
        from dcase_ae.embedder import PretrainedAudioEmbedder
        from dcase_ae.utils import get_device

        self.embedder = PretrainedAudioEmbedder(
            model_name=str(self.metadata["pretrained_model_name"]),
            sample_rate=int(self.metadata["target_sample_rate"]),
            device=get_device(use_cuda),
        )

    def analyze_json_file(self, input_path: str | Path) -> HealthAnalysis:
        input_path = Path(input_path)
        waveform, audio_metadata = load_json_audio(
            input_path,
            default_sample_rate=int(self.metadata["input_sample_rate"]),
            target_sample_rate=int(self.metadata["target_sample_rate"]),
        )
        embeddings = self.embedder.extract([waveform])
        raw_score = float(self.detector.score_samples(embeddings)[0])
        anomaly = anomaly_result(raw_score, self.reference_scores)
        health = health_score_from_reference(raw_score, self.reference_scores)
        return HealthAnalysis(
            status="ok",
            detector_type="knn",
            filename=input_path.name,
            input_path=str(input_path),
            raw_knn_distance=anomaly.raw_score,
            abnormality_score=anomaly.abnormality_score,
            abnormality_unit="percentile_0_100",
            risk_level=anomaly.risk_level,
            health_score=health.health_score,
            health_level=health.health_level,
            health_score_meaning=(
                "equipment health score derived from KNN distance against normal reference distribution; "
                "higher is healthier"
            ),
            score_meaning="mean distance to nearest normal embeddings; larger means more abnormal",
            audio=audio_metadata,
            library={
                "library_dir": str(self.library_dir),
                "reference_count": int(self.metadata["reference_count"]),
                "pretrained_model_name": self.metadata["pretrained_model_name"],
                "target_sample_rate": int(self.metadata["target_sample_rate"]),
            },
            risk_thresholds=RISK_THRESHOLDS,
        )

    def _validate_library(self) -> None:
        missing = [name for name in REQUIRED_LIBRARY_FILES if not (self.library_dir / name).exists()]
        if missing:
            raise HealthLibraryError(
                "health_library_incomplete",
                f"Health library is missing required files: {', '.join(missing)}",
            )


def validate_health_library(library_dir: str | Path) -> dict[str, Any]:
    library_dir = Path(library_dir)
    missing = [name for name in REQUIRED_LIBRARY_FILES if not (library_dir / name).exists()]
    if missing:
        return {
            "status": "error",
            "status_code": "health_library_incomplete",
            "message": f"Health library is missing required files: {', '.join(missing)}",
            "library_dir": str(library_dir),
        }
    metadata_path = library_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "error",
            "status_code": "health_library_bad_metadata",
            "message": str(exc),
            "library_dir": str(library_dir),
        }
    return {
        "status": "ok",
        "library_dir": str(library_dir),
        "metadata": metadata,
    }


def error_result(status_code: str, message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "status_code": status_code,
        "message": message,
    }


def _iter_json_files(reference_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = reference_dir.rglob(pattern) if recursive else reference_dir.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["index", "file_path", "input_sample_rate", "sample_rate", "num_samples", "resampled"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reference_summary(reference_scores: np.ndarray) -> dict[str, Any]:
    return {
        "source": "train",
        "method": "leave-one-out-knn-distance",
        "count": int(reference_scores.shape[0]),
        "mean": float(np.mean(reference_scores)),
        "std": float(np.std(reference_scores)),
        "p50": float(np.quantile(reference_scores, 0.50)),
        "p90": float(np.quantile(reference_scores, 0.90)),
        "p95": float(np.quantile(reference_scores, 0.95)),
        "p99": float(np.quantile(reference_scores, 0.99)),
    }
