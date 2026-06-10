from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset


@dataclass
class TabularSplit:
    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    scaler: StandardScaler
    feature_cols: list[str]
    classes: np.ndarray
    target_mean: float = 0.0
    target_std: float = 1.0


@dataclass
class SequenceSplit:
    train: list[tuple[np.ndarray, int]]
    val: list[tuple[np.ndarray, int]]
    train_flat_x: np.ndarray
    train_flat_y: np.ndarray
    scaler: StandardScaler
    feature_cols: list[str]
    classes: np.ndarray
    target_mean: float = 0.0
    target_std: float = 1.0


class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, target_dtype: torch.dtype = torch.long) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=target_dtype)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SequenceDataset(Dataset):
    def __init__(self, items: list[tuple[np.ndarray, int | float]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.items[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(len(x), dtype=torch.long),
            torch.tensor(y),
        )


def infer_feature_columns(
    df: pd.DataFrame,
    excluded: set[str],
    explicit_features: list[str] | None = None,
) -> list[str]:
    if explicit_features:
        return explicit_features

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


def load_detection_split(
    csv_path: str,
    target_col: str,
    feature_cols: list[str] | None,
    test_size: float,
    seed: int,
) -> TabularSplit:
    df = pd.read_csv(csv_path).dropna()
    features = infer_feature_columns(df, {target_col}, feature_cols)

    x = df[features].to_numpy(dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target_col].to_numpy())

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )


def load_tabular_regression_split(
    csv_path: str,
    target_col: str,
    feature_cols: list[str] | None,
    test_size: float,
    seed: int,
) -> TabularSplit:
    df = pd.read_csv(csv_path).dropna()
    features = infer_feature_columns(df, {target_col}, feature_cols)

    x = df[features].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
    )
    scaler = StandardScaler().fit(x_train)
    target_mean = float(y_train.mean())
    target_std = float(y_train.std() if y_train.std() > 0 else 1.0)
    y_train_scaled = (y_train - target_mean) / target_std
    y_val_scaled = (y_val - target_mean) / target_std
    return TabularSplit(
        scaler.transform(x_train),
        scaler.transform(x_val),
        y_train_scaled,
        y_val_scaled,
        scaler,
        features,
        np.array([target_col]),
        target_mean,
        target_std,
    )
    scaler = StandardScaler().fit(x_train)
    return TabularSplit(
        scaler.transform(x_train),
        scaler.transform(x_val),
        y_train,
        y_val,
        scaler,
        features,
        encoder.classes_,
    )


def load_sequence_split(
    csv_path: str,
    subject_col: str,
    time_col: str,
    target_col: str,
    feature_cols: list[str] | None,
    test_size: float,
    seed: int,
) -> SequenceSplit:
    df = pd.read_csv(csv_path).dropna()
    features = infer_feature_columns(df, {subject_col, time_col, target_col}, feature_cols)

    encoder = LabelEncoder()
    df = df.copy()
    df["_target"] = encoder.fit_transform(df[target_col].to_numpy())

    subjects = df[subject_col].drop_duplicates().to_numpy()
    subject_targets = []
    for subject in subjects:
        sdf = df[df[subject_col] == subject].sort_values(time_col)
        subject_targets.append(int(sdf["_target"].iloc[-1]))

    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=seed,
        stratify=subject_targets if len(np.unique(subject_targets)) > 1 else None,
    )


def load_sequence_regression_split(
    csv_path: str,
    subject_col: str,
    time_col: str,
    target_col: str,
    feature_cols: list[str] | None,
    test_size: float,
    seed: int,
) -> SequenceSplit:
    df = pd.read_csv(csv_path).dropna()
    features = infer_feature_columns(df, {subject_col, time_col, target_col}, feature_cols)

    subjects = df[subject_col].drop_duplicates().to_numpy()
    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=seed,
    )

    train_df = df[df[subject_col].isin(train_subjects)]
    scaler = StandardScaler().fit(train_df[features])
    target_mean = float(train_df[target_col].mean())
    target_std = float(train_df[target_col].std() if train_df[target_col].std() > 0 else 1.0)

    def build_items(selected_subjects: np.ndarray) -> list[tuple[np.ndarray, float]]:
        items: list[tuple[np.ndarray, float]] = []
        for subject in selected_subjects:
            sdf = df[df[subject_col] == subject].sort_values(time_col)
            x = scaler.transform(sdf[features]).astype(np.float32)
            y = float((sdf[target_col].iloc[-1] - target_mean) / target_std)
            items.append((x, y))
        return items

    train_flat = scaler.transform(train_df[features]).astype(np.float32)
    train_flat_y = ((train_df[target_col].to_numpy(dtype=np.float32) - target_mean) / target_std)

    return SequenceSplit(
        build_items(train_subjects),
        build_items(val_subjects),
        train_flat,
        train_flat_y,
        scaler,
        features,
        np.array([target_col]),
        target_mean,
        target_std,
    )

    scaler = StandardScaler().fit(df[df[subject_col].isin(train_subjects)][features])

    def build_items(selected_subjects: np.ndarray) -> list[tuple[np.ndarray, int]]:
        items: list[tuple[np.ndarray, int]] = []
        for subject in selected_subjects:
            sdf = df[df[subject_col] == subject].sort_values(time_col)
            x = scaler.transform(sdf[features]).astype(np.float32)
            y = int(sdf["_target"].iloc[-1])
            items.append((x, y))
        return items

    train_flat = scaler.transform(df[df[subject_col].isin(train_subjects)][features]).astype(np.float32)
    train_flat_y = df[df[subject_col].isin(train_subjects)]["_target"].to_numpy()

    return SequenceSplit(
        build_items(train_subjects),
        build_items(val_subjects),
        train_flat,
        train_flat_y,
        scaler,
        features,
        encoder.classes_,
    )


def collate_sequences(batch):
    xs, lengths, ys = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    return padded, torch.stack(lengths), torch.stack(ys)
