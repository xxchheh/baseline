import csv
from pathlib import Path

import joblib

from configs import Config
from dcase_ae.dataset import LocalDCASEDataModule
from dcase_ae.embedder import PretrainedAudioEmbedder
from dcase_ae.features import FeatureConfig
from dcase_ae.metrics import (
    METRIC_HEADER,
    ScoreRecord,
    compute_metrics,
    format_metric,
    parse_domain_from_filename,
    parse_label_from_filename,
)
from dcase_ae.utils import ensure_dir, get_device


class AEEvaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.use_cuda)
        self.feature_cfg = FeatureConfig(
            n_mels=cfg.n_mels,
            frames=cfg.frames,
            frame_hop_length=cfg.frame_hop_length,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            power=cfg.power,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            win_length=cfg.win_length,
            mono=cfg.mono,
        )
        self.data = LocalDCASEDataModule(
            data_dir=cfg.data_dir,
            feature_cfg=self.feature_cfg,
            batch_size=cfg.batch_size,
            validation_split=cfg.validation_split,
            shuffle=False,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )
        self.embedder = PretrainedAudioEmbedder(
            model_name=cfg.pretrained_model_name,
            sample_rate=cfg.embedding_sample_rate,
            device=self.device,
        )

    def evaluate(self) -> tuple[Path, Path | None]:
        checkpoint = joblib.load(self.cfg.checkpoint_path)
        gmm = checkpoint["gmm"] if isinstance(checkpoint, dict) else checkpoint

        output_dir = ensure_dir(self.cfg.output_dir)
        scores_path = output_dir / "scores.csv"
        metrics_path = output_dir / "metrics.csv"
        rows = [["filename", "anomaly_score", "label", "domain"]]
        records = []

        for batch_idx, (waveforms, basenames) in enumerate(
            self.data.test_audio_loader(
                sample_rate=self.cfg.embedding_sample_rate,
                mono=True,
            )
        ):
            embeddings = self.embedder.extract(waveforms)
            anomaly_scores = -gmm.score_samples(embeddings)
            for filename, score_value in zip(basenames, anomaly_scores):
                label = parse_label_from_filename(filename)
                domain = parse_domain_from_filename(filename)
                rows.append([
                    filename,
                    f"{float(score_value):.10f}",
                    _format_label(label),
                    domain or "",
                ])
                records.append(
                    ScoreRecord(
                        filename=filename,
                        score=float(score_value),
                        label=label,
                        domain=domain,
                    )
                )
            if batch_idx % self.cfg.log_interval == 0:
                print(f"Score test embeddings batch={batch_idx}")

        with open(scores_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        if any(record.label is not None for record in records):
            metric_values = compute_metrics(
                records=records,
                max_fpr=self.cfg.max_fpr,
                threshold=self.cfg.decision_threshold,
            )
            with open(metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(METRIC_HEADER)
                writer.writerow([format_metric(metric_values[name]) for name in METRIC_HEADER])
        else:
            metrics_path = None
        return scores_path, metrics_path


def _format_label(label: int | None) -> str:
    if label is None:
        return ""
    return "anomaly" if label == 1 else "normal"
