from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pd_speech.audio_features import (
    extract_audio_features,
    label_from_parent,
    subject_id_from_filename,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="data/figshare_features.csv")
    parser.add_argument("--patterns", nargs="+", default=["HC_*/*.wav", "PD_*/*.wav"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    wav_paths: list[Path] = []
    for pattern in args.patterns:
        wav_paths.extend(root.glob(pattern))
    wav_paths = sorted(set(wav_paths))

    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under {root} with patterns {args.patterns}")

    rows = []
    for path in tqdm(wav_paths, desc="extracting audio features"):
        label_name, label = label_from_parent(path)
        row = {
            "recording_id": path.stem,
            "subject_id": subject_id_from_filename(path),
            "file_path": str(path),
            "group": label_name,
            "pd_label": label,
            "utterance": path.parent.name.split("_", 1)[1] if "_" in path.parent.name else path.parent.name,
        }
        row.update(extract_audio_features(path))
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows and {len(fieldnames)} columns to {out_path}")


if __name__ == "__main__":
    main()
