import argparse
import json
from pathlib import Path

from dcase_ae.json_audio import load_json_audio, write_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON audio files to mono 16 kHz WAV files.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Dataset root containing train/test JSON files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dataset root for converted WAV files.")
    parser.add_argument("--input_sample_rate", type=int, default=44800)
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Subdirectories to convert.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAV files.")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Optional JSONL manifest path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converted = 0
    skipped = 0
    failed = 0

    manifest_file = None
    if args.manifest_path is not None:
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_file = open(args.manifest_path, "w", encoding="utf-8")

    try:
        for split in args.splits:
            input_split_dir = args.input_dir / split
            output_split_dir = args.output_dir / split
            if not input_split_dir.is_dir():
                print(f"Skip missing split: {input_split_dir}")
                continue

            for json_path in sorted(input_split_dir.rglob("*.json")):
                relative_path = json_path.relative_to(input_split_dir).with_suffix(".wav")
                wav_path = output_split_dir / relative_path
                if wav_path.exists() and not args.overwrite:
                    skipped += 1
                    continue

                try:
                    waveform, metadata = load_json_audio(
                        json_path,
                        default_sample_rate=args.input_sample_rate,
                        target_sample_rate=args.target_sample_rate,
                    )
                    write_wav(wav_path, waveform, metadata["sample_rate"])
                    converted += 1
                    if manifest_file is not None:
                        manifest_file.write(
                            json.dumps(
                                {
                                    "input_path": str(json_path),
                                    "output_path": str(wav_path),
                                    "split": split,
                                    "input_sample_rate": metadata["input_sample_rate"],
                                    "sample_rate": metadata["sample_rate"],
                                    "num_samples": metadata["num_samples"],
                                    "resampled": metadata["resampled"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                except Exception as exc:
                    failed += 1
                    print(f"Failed: {json_path} ({exc})")
    finally:
        if manifest_file is not None:
            manifest_file.close()

    print(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "target_sample_rate": args.target_sample_rate,
                "converted": converted,
                "skipped": skipped,
                "failed": failed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
