import argparse
from pathlib import Path

from configs import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score wav files with a pretrained-audio-embedding + GMM baseline.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Dataset root with train/ and test/ wav folders.")
    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoints/gmm.joblib"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=13711)
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrained_model_name", type=str, default="microsoft/wavlm-base")
    parser.add_argument("--embedding_sample_rate", type=int, default=16000)

    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--frame_hop_length", type=int, default=1)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--win_length", type=int, default=None)
    parser.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--file_score_quantile", type=float, default=0.9)
    parser.add_argument("--file_score_tail_weight", type=float, default=0.5)
    parser.add_argument("--max_fpr", type=float, default=0.1)
    parser.add_argument("--decision_threshold", type=float, default=None)
    parser.add_argument("--gmm_components", type=int, default=4)
    parser.add_argument("--gmm_covariance_type", type=str, default="full")
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-6)
    parser.add_argument("--gmm_max_iter", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from dcase_ae.evaluator import AEEvaluator
    from dcase_ae.utils import set_seed

    cfg = Config(**vars(args))
    set_seed(cfg.seed)
    scores_path, metrics_path = AEEvaluator(cfg).evaluate()
    print(f"Saved scores: {scores_path}")
    if metrics_path is not None:
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
