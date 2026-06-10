from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/features.csv")
    parser.add_argument("--subjects", type=int, default=120)
    parser.add_argument("--visits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    rows = []

    for subject_idx in range(args.subjects):
        subject_id = f"p{subject_idx:04d}"
        has_pd = int(rng.random() < 0.55)
        base_impairment = rng.gauss(0.0, 0.4) + has_pd * rng.gauss(1.1, 0.25)

        for visit in range(1, args.visits + 1):
            progression = has_pd * (visit - 1) * rng.gauss(0.35, 0.08)
            severity = base_impairment + progression
            stage = 0 if not has_pd else (2 if severity < 1.45 else 3)

            rows.append(
                {
                    "subject_id": subject_id,
                    "visit": visit,
                    "jitter": 0.012 + 0.006 * severity + rng.gauss(0, 0.002),
                    "shimmer": 0.035 + 0.012 * severity + rng.gauss(0, 0.004),
                    "hnr": 24.0 - 2.2 * severity + rng.gauss(0, 1.0),
                    "nhr": 0.12 + 0.05 * severity + rng.gauss(0, 0.02),
                    "rpde": 0.38 + 0.08 * severity + rng.gauss(0, 0.03),
                    "dfa": 0.63 + 0.04 * severity + rng.gauss(0, 0.02),
                    "ppe": 0.16 + 0.06 * severity + rng.gauss(0, 0.02),
                    "mfcc_1": rng.gauss(0.2 * severity, 0.5),
                    "mfcc_2": rng.gauss(-0.15 * severity, 0.5),
                    "mfcc_3": rng.gauss(0.1 * severity, 0.5),
                    "pd_label": has_pd,
                    "stage": stage,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
