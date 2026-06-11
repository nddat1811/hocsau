from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from paper_reproduction.config import apply_smoke_limits, parse_args
from paper_reproduction.data import load_dataframe, make_tabular_data, write_distribution
from paper_reproduction.detection import run_stage_detection
from paper_reproduction.lstm import run_lstm_forecasts


def main() -> None:
    args = apply_smoke_limits(parse_args())
    if not args.show_convergence_warnings:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    df = load_dataframe(args.csv, args.target_col, args.stage_thresholds, args.drop_stage_zero)
    data = make_tabular_data(df, args.target_col)

    write_distribution(df, out_dir, args.seed, args.test_size)
    if not args.skip_detection:
        run_stage_detection(data, out_dir, args, rng)
    if args.run_lstm:
        run_lstm_forecasts(df, data.features, out_dir, args)
