from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Parkinson telemonitoring paper pipeline."
    )
    parser.add_argument("--csv", default="parkinsons+telemonitoring/parkinsons_updrs.data")
    parser.add_argument("--out-dir", default="runs/paper_reproduction")
    parser.add_argument("--target-col", default="motor_UPDRS")
    parser.add_argument("--stage-thresholds", default="10,20,30")
    parser.add_argument("--drop-stage-zero", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--mlp-epochs", type=int, default=1000)
    parser.add_argument("--lstm-epochs", type=int, default=1000)
    parser.add_argument("--lstm-batch-size", type=int, default=64)
    parser.add_argument("--smote-k", type=int, default=5)
    parser.add_argument("--run-lstm", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--show-convergence-warnings", action="store_true")
    return parser.parse_args()


def apply_smoke_limits(args: argparse.Namespace) -> argparse.Namespace:
    if args.smoke:
        args.mlp_epochs = min(args.mlp_epochs, 2)
        args.lstm_epochs = min(args.lstm_epochs, 2)
    return args
