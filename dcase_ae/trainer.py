from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

from configs import Config
from dcase_ae.dataset import LocalDCASEDataModule
from dcase_ae.embedder import PretrainedAudioEmbedder
from dcase_ae.features import FeatureConfig
from dcase_ae.utils import ensure_dir, get_device


class GMMTrainer:
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

    def fit(self) -> Path:
        ensure_dir(self.cfg.output_dir)
        embeddings = self._extract_train_embeddings()
        gmm = GaussianMixture(
            n_components=self.cfg.gmm_components,
            covariance_type=self.cfg.gmm_covariance_type,
            reg_covar=self.cfg.gmm_reg_covar,
            max_iter=self.cfg.gmm_max_iter,
            random_state=self.cfg.seed,
        )
        print(f"Fitting GMM on embeddings: {embeddings.shape}")
        gmm.fit(embeddings)
        normal_reference_scores = -gmm.score_samples(embeddings)

        checkpoint_path = Path(self.cfg.checkpoint_path)
        ensure_dir(checkpoint_path.parent)
        joblib.dump(
            {
                "gmm": gmm,
                "config": self._checkpoint_config(),
                "embedding_dim": int(embeddings.shape[1]),
                "normal_reference_scores": normal_reference_scores.astype(np.float32, copy=False),
                "score_reference": {
                    "source": "train",
                    "count": int(normal_reference_scores.shape[0]),
                    "mean": float(np.mean(normal_reference_scores)),
                    "std": float(np.std(normal_reference_scores)),
                    "p90": float(np.quantile(normal_reference_scores, 0.90)),
                    "p95": float(np.quantile(normal_reference_scores, 0.95)),
                    "p99": float(np.quantile(normal_reference_scores, 0.99)),
                },
            },
            checkpoint_path,
        )
        return checkpoint_path

    def _extract_train_embeddings(self) -> np.ndarray:
        embeddings = []
        loader = self.data.train_audio_loader(
            sample_rate=self.cfg.embedding_sample_rate,
            mono=True,
        )
        for batch_idx, (waveforms, basenames) in enumerate(loader):
            batch_embeddings = self.embedder.extract(waveforms)
            embeddings.append(batch_embeddings)
            if batch_idx % self.cfg.log_interval == 0:
                print(
                    f"Extract train embeddings batch={batch_idx}/{len(loader)} "
                    f"batch_size={len(basenames)}"
                )
        if not embeddings:
            raise RuntimeError("No train embeddings were extracted.")
        return np.concatenate(embeddings, axis=0)

    def _checkpoint_config(self) -> dict:
        config = asdict(self.cfg)
        config["data_dir"] = str(self.cfg.data_dir)
        config["checkpoint_path"] = str(self.cfg.checkpoint_path)
        config["output_dir"] = str(self.cfg.output_dir)
        return config
