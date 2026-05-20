from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from configs import Config
from networks.dcase2023t2_ae.network import AENet
from dcase_ae.dataset import LocalDCASEDataModule
from dcase_ae.features import FeatureConfig
from dcase_ae.utils import ensure_dir, get_device, save_checkpoint


class AETrainer:
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
            shuffle=cfg.shuffle,
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
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    def fit(self) -> Path:
        train_loader, valid_loader = self.data.train_valid_loaders()
        ensure_dir(self.cfg.output_dir)

        best_valid_loss = float("inf")
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            valid_loss = self._valid_epoch(valid_loader)
            print(
                f"Epoch {epoch:03d}/{self.cfg.epochs} "
                f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}"
            )
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(
                    self.cfg.checkpoint_path,
                    self.model,
                    config=self._checkpoint_config(),
                    epoch=epoch,
                )
        return Path(self.cfg.checkpoint_path)

    def _train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_count = 0

        for batch_idx, data in enumerate(loader):
            data = data.to(self.device).float()
            self.optimizer.zero_grad()
            recon, _ = self.model(self._corrupt_input(data))
            loss = F.smooth_l1_loss(recon, data.view(recon.shape))
            loss.backward()
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            batch_size = data.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            if batch_idx % self.cfg.log_interval == 0:
                print(f"Train epoch={epoch} batch={batch_idx}/{len(loader)} loss={loss.item():.6f}")

        return total_loss / max(total_count, 1)

    def _valid_epoch(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device).float()
                recon, _ = self.model(data)
                loss = F.mse_loss(recon, data.view(recon.shape), reduction="none")
                loss = loss.mean(dim=1)
                total_loss += float(loss.sum().item())
                total_count += data.size(0)
        return total_loss / max(total_count, 1)

    def _corrupt_input(self, data: torch.Tensor) -> torch.Tensor:
        if self.cfg.input_noise <= 0:
            return data
        scale = data.detach().std(dim=1, keepdim=True).clamp_min(1.0)
        return data + torch.randn_like(data) * scale * self.cfg.input_noise

    def _checkpoint_config(self) -> dict:
        config = asdict(self.cfg)
        config["data_dir"] = str(self.cfg.data_dir)
        config["checkpoint_path"] = str(self.cfg.checkpoint_path)
        config["output_dir"] = str(self.cfg.output_dir)
        return config
