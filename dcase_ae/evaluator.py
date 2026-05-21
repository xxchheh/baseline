import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from configs import Config
from dcase_ae.dataset import LocalDCASEDataModule
from dcase_ae.features import FeatureConfig
from dcase_ae.metrics import (
    METRIC_HEADER,
    ScoreRecord,
    compute_metrics,
    format_metric,
    parse_domain_from_filename,
    parse_label_from_filename,
)
from dcase_ae.utils import ensure_dir, get_device, load_checkpoint
from networks.dcase2023t2_ae.network import AENet


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
        self.model = AENet(
            input_dim=cfg.input_dim,
            block_size=cfg.n_mels,
            num_conditions=0,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        ).to(self.device)

    def evaluate(self) -> tuple[Path, Path | None]:
        checkpoint = load_checkpoint(self.cfg.checkpoint_path, self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        output_dir = ensure_dir(self.cfg.output_dir)
        scores_path = output_dir / "scores.csv"
        metrics_path = output_dir / "metrics.csv"
        rows = [["filename", "anomaly_score", "label", "domain"]]
        records = []

        with torch.no_grad():
            for vectors, basename in self.data.test_loader():
                vectors = vectors.squeeze(0).to(self.device).float()
                recon, _ = self.model(vectors)
                frame_scores = F.mse_loss(recon, vectors.view(recon.shape), reduction="none").mean(dim=1)
                score = self._aggregate_frame_scores(frame_scores)
                filename = basename[0]
                score_value = float(score.item())
                label = parse_label_from_filename(filename)
                domain = parse_domain_from_filename(filename)
                rows.append([
                    filename,
                    f"{score_value:.10f}",
                    _format_label(label),
                    domain or "",
                ])
                records.append(
                    ScoreRecord(
                        filename=filename,
                        score=score_value,
                        label=label,
                        domain=domain,
                    )
                )

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

    def _aggregate_frame_scores(self, frame_scores: torch.Tensor) -> torch.Tensor:
        if frame_scores.numel() == 0:
            return frame_scores.new_tensor(0.0)
        mean_score = frame_scores.mean()
        tail_score = torch.quantile(frame_scores, self.cfg.file_score_quantile)
        return (
            (1.0 - self.cfg.file_score_tail_weight) * mean_score
            + self.cfg.file_score_tail_weight * tail_score
        )


def _format_label(label: int | None) -> str:
    if label is None:
        return ""
    return "anomaly" if label == 1 else "normal"
