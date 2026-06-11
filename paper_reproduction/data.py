from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PAPER_FEATURES = [
    "age",
    "sex",
    "test_time",
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer:APQ5",
    "NHR",
    "DFA",
]


@dataclass
class TabularData:
    x: np.ndarray
    y_stage: np.ndarray
    y_motor: np.ndarray
    features: list[str]


def load_dataframe(
    csv_path: str,
    target_col: str,
    thresholds: str,
    drop_stage_zero: bool,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna().copy()
    limits = [float(item.strip()) for item in thresholds.split(",") if item.strip()]
    bins = [-np.inf, *limits, np.inf]
    df["pd_stage"] = pd.cut(df[target_col], bins=bins, labels=False).astype(int)
    if drop_stage_zero:
        df = df[df["pd_stage"] > 0].copy()
    return df


def make_tabular_data(df: pd.DataFrame, target_col: str) -> TabularData:
    feature_cols = [col for col in PAPER_FEATURES if col in df.columns and col != target_col]
    return TabularData(
        df[feature_cols].to_numpy(dtype=np.float32),
        df["pd_stage"].to_numpy(dtype=np.int64),
        df[target_col].to_numpy(dtype=np.float32),
        feature_cols,
    )


def write_distribution(df: pd.DataFrame, out_dir: Path, seed: int, test_size: float) -> None:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["pd_stage"],
    )
    labels = sorted(df["pd_stage"].unique())
    distribution = {
        "train": {str(label): int((train_df["pd_stage"] == label).sum()) for label in labels},
        "test": {str(label): int((test_df["pd_stage"] == label).sum()) for label in labels},
        "total": {str(label): int((df["pd_stage"] == label).sum()) for label in labels},
    }
    (out_dir / "stage_distribution.json").write_text(
        json.dumps(distribution, indent=2),
        encoding="utf-8",
    )
