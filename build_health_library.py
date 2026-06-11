import argparse
import json
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deployable KNN health library from normal JSON audio.")
    parser.add_argument("--reference_dir", type=Path, required=True, help="Directory containing normal JSON files.")
    parser.add_argument("--library_dir", type=Path, required=True, help="Output health library directory.")
    parser.add_argument("--pretrained_model_name", type=str, default="microsoft/wavlm-base")
    parser.add_argument("--input_sample_rate", type=int, default=44800)
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--knn_neighbors", type=int, default=5)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pattern", type=str, default="*.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from dcase_ae.health_library import HealthLibraryConfig, build_health_library, validate_health_library

    config = HealthLibraryConfig(
        pretrained_model_name=args.pretrained_model_name,
        input_sample_rate=args.input_sample_rate,
        target_sample_rate=args.target_sample_rate,
        knn_neighbors=args.knn_neighbors,
        pca_dim=args.pca_dim,
        batch_size=args.batch_size,
        use_cuda=args.use_cuda,
        recursive=args.recursive,
        pattern=args.pattern,
    )
    metadata = build_health_library(args.reference_dir, args.library_dir, config)
    validation = validate_health_library(args.library_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "library_dir": str(args.library_dir),
                "metadata": metadata,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
