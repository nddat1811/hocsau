from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from paper_reproduction.models import PaperLSTM


class SequenceDataset(Dataset):
    def __init__(self, xs: list[np.ndarray], ys: list[float]) -> None:
        self.xs = [torch.tensor(x, dtype=torch.float32) for x in xs]
        self.ys = torch.tensor(ys, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, idx: int):
        return self.xs[idx], torch.tensor(len(self.xs[idx]), dtype=torch.long), self.ys[idx]


def run_lstm_forecasts(
    df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    scaler = MinMaxScaler().fit(df[feature_cols])
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    results = {}
    for target_stage in [2, 3]:
        xs, ys = build_forecast_samples(df, feature_cols, args.target_col, target_stage)
        if len(xs) < 4:
            results[str(target_stage)] = {"error": "not enough samples"}
            continue
        train_idx, val_idx = train_test_split(
            np.arange(len(xs)),
            test_size=args.test_size,
            random_state=args.seed,
        )
        target_mean = float(np.mean([ys[i] for i in train_idx]))
        target_std = float(np.std([ys[i] for i in train_idx]) or 1.0)
        train_ds = SequenceDataset(
            [xs[i] for i in train_idx],
            [(ys[i] - target_mean) / target_std for i in train_idx],
        )
        val_ds = SequenceDataset(
            [xs[i] for i in val_idx],
            [(ys[i] - target_mean) / target_std for i in val_idx],
        )
        history = train_lstm(
            train_ds,
            val_ds,
            len(feature_cols),
            args.lstm_epochs,
            args.lstm_batch_size,
            target_mean,
            target_std,
        )
        results[str(target_stage)] = {
            "samples": len(xs),
            "target_mean": target_mean,
            "target_std": target_std,
            "history": history,
        }
        write_lstm_results(out_dir, results)
    write_lstm_results(out_dir, results)


def write_lstm_results(out_dir: Path, results: dict) -> None:
    (out_dir / "lstm_forecast_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def build_forecast_samples(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    target_stage: int,
) -> tuple[list[np.ndarray], list[float]]:
    xs = []
    ys = []
    for _, sdf in df.groupby("subject#"):
        sdf = sdf.sort_values("test_time")
        values = sdf[feature_cols].to_numpy(dtype=np.float32)
        targets = sdf[target_col].to_numpy(dtype=np.float32)
        stages = sdf["pd_stage"].to_numpy(dtype=np.int64)
        for idx in range(1, len(sdf)):
            if int(stages[idx]) == target_stage:
                xs.append(values[:idx])
                ys.append(float(targets[idx]))
    return xs, ys


def collate_sequences(batch):
    xs, lengths, ys = zip(*batch)
    return (
        nn.utils.rnn.pad_sequence(xs, batch_first=True),
        torch.stack(lengths),
        torch.stack(ys),
    )


def train_lstm(
    train_ds: SequenceDataset,
    val_ds: SequenceDataset,
    input_dim: int,
    epochs: int,
    batch_size: int,
    target_mean: float,
    target_std: float,
) -> list[dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaperLSTM(input_dim=input_dim, hidden_dim=150).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_sequences)
    report_epochs = {1, 50, 100, 150, 200, 400, 600, 800, 1000}
    history = []
    for epoch in tqdm(range(1, epochs + 1), desc="paper-lstm"):
        model.train()
        for x, lengths, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(x.to(device), lengths.to(device))
            loss = criterion(pred, y.to(device))
            loss.backward()
            optimizer.step()
        if epoch in report_epochs or epoch == epochs:
            metrics = evaluate_lstm(model, val_loader, device, target_mean, target_std)
            history.append({"epoch": float(epoch), **metrics})
    return history


def evaluate_lstm(model, val_loader, device, target_mean: float, target_std: float) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, lengths, y in val_loader:
            pred = model(x.to(device), lengths.to(device)).cpu().numpy()
            preds.append(pred)
            targets.append(y.numpy())
    pred = np.concatenate(preds)
    true = np.concatenate(targets)
    raw_pred = pred * target_std + target_mean
    raw_true = true * target_std + target_mean
    mse = float(mean_squared_error(raw_true, raw_pred))
    return {
        "validation_rmse": float(math.sqrt(mse)),
        "validation_mse": mse,
        "validation_loss": float(mean_squared_error(true, pred)),
    }
