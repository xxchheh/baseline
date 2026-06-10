from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    checkpoint_path: Path = Path("checkpoints/gmm.joblib")
    output_dir: Path = Path("outputs")
    seed: int = 13711
    use_cuda: bool = True
    num_workers: int = 4
    pretrained_model_name: str = "microsoft/wavlm-base"
    embedding_sample_rate: int = 16000

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

    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    validation_split: float = 0.1
    shuffle: bool = True

    hidden_dim: int = 512
    latent_dim: int = 64
    dropout: float = 0.1
    input_noise: float = 0.05
    file_score_quantile: float = 0.9
    file_score_tail_weight: float = 0.5
    max_fpr: float = 0.1
    decision_threshold: float | None = None
    gmm_components: int = 4
    gmm_covariance_type: str = "full"
    gmm_reg_covar: float = 1e-6
    gmm_max_iter: int = 100
    log_interval: int = 100

    @property
    def input_dim(self) -> int:
        return self.n_mels * self.frames
