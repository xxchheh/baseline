import argparse
import json
import os
from pathlib import Path

import joblib

from configs import Config

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one JSON audio file and print JSON result.")
    parser.add_argument("--input_path", type=Path, required=True, help="Input JSON audio file.")
    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoints/knn.joblib"))
    parser.add_argument("--pretrained_model_name", type=str, default="microsoft/wavlm-base")
    parser.add_argument("--embedding_sample_rate", type=int, default=16000)
    parser.add_argument("--json_sample_rate", type=int, default=44800)
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--downsample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wav_output_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from dcase_ae.embedder import PretrainedAudioEmbedder
    from dcase_ae.json_audio import load_json_audio, write_wav
    from dcase_ae.scoring import RISK_THRESHOLDS, anomaly_result
    from dcase_ae.utils import get_device

    checkpoint = joblib.load(args.checkpoint_path)
    if not isinstance(checkpoint, dict) or checkpoint.get("detector_type") != "knn":
        raise RuntimeError("Expected a KNN checkpoint. Please retrain with train.py on this branch.")
    if "normal_reference_scores" not in checkpoint:
        raise RuntimeError(
            "Checkpoint is missing normal_reference_scores. "
            "Please retrain with the updated train.py first."
        )
    detector = checkpoint["detector"]

    target_sample_rate = args.embedding_sample_rate if args.downsample else None
    waveform, audio_metadata = load_json_audio(
        args.input_path,
        default_sample_rate=args.json_sample_rate,
        target_sample_rate=target_sample_rate,
    )

    wav_output_path = None
    if args.wav_output_path is not None:
        wav_output_path = write_wav(
            args.wav_output_path,
            waveform,
            audio_metadata["sample_rate"],
        )

    cfg = Config(
        data_dir=Path("."),
        checkpoint_path=args.checkpoint_path,
        pretrained_model_name=args.pretrained_model_name,
        embedding_sample_rate=audio_metadata["sample_rate"],
        use_cuda=args.use_cuda,
    )
    embedder = PretrainedAudioEmbedder(
        model_name=cfg.pretrained_model_name,
        sample_rate=cfg.embedding_sample_rate,
        device=get_device(cfg.use_cuda),
    )

    embeddings = embedder.extract([waveform])
    raw_score = float(detector.score_samples(embeddings)[0])
    result = anomaly_result(raw_score, checkpoint["normal_reference_scores"])

    output = {
        "filename": args.input_path.name,
        "input_path": str(args.input_path),
        "detector_type": "knn",
        "score_meaning": "mean distance to nearest normal embeddings; larger means more abnormal",
        "raw_knn_distance": result.raw_score,
        "abnormality_score": result.abnormality_score,
        "abnormality_unit": "percentile_0_100",
        "risk_level": result.risk_level,
        "risk_thresholds": RISK_THRESHOLDS,
        "audio": audio_metadata,
        "wav_output_path": str(wav_output_path) if wav_output_path is not None else None,
        "reference": checkpoint.get("score_reference", {"source": "train"}),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
