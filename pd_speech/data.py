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
    train: list[tuple[np.ndarray, int | float] | tuple[np.ndarray, int | float, int]]
    val: list[tuple[np.ndarray, int | float] | tuple[np.ndarray, int | float, int]]
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
    def __init__(
        self,
        items: list[tuple[np.ndarray, int | float] | tuple[np.ndarray, int | float, int]],
    ) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        x, y = item[:2]
        output = [
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(len(x), dtype=torch.long),
            torch.tensor(y),
        ]
        if len(item) == 3:
            output.append(torch.tensor(item[2], dtype=torch.long))
        return tuple(output)


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


def parse_stage_thresholds(value: str | list[float] | tuple[float, ...] | None) -> list[float]:
    if value is None:
        return [10.0, 20.0, 30.0]
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def add_derived_stage(
    df: pd.DataFrame,
    stage_col: str | None,
    stage_source_col: str,
    stage_thresholds: list[float],
) -> pd.DataFrame:
    df = df.copy()
    if stage_col and stage_col in df.columns:
        df["_pd_stage"] = df[stage_col].astype(int)
        return df
    if stage_source_col not in df.columns:
        raise ValueError(
            f"Cannot derive PD stage because '{stage_source_col}' is not present and "
            f"stage column '{stage_col}' was not found."
        )
    bins = [-np.inf, *stage_thresholds, np.inf]
    labels = list(range(len(bins) - 1))
    df["_pd_stage"] = pd.cut(df[stage_source_col], bins=bins, labels=labels).astype(int)
    return df


def load_paper_stage_forecast_split(
    csv_path: str,
    subject_col: str,
    time_col: str,
    target_col: str,
    feature_cols: list[str] | None,
    test_size: float,
    seed: int,
    stage_col: str | None,
    stage_source_col: str,
    stage_thresholds: str | list[float] | tuple[float, ...] | None,
    forecast_target_stage: int,
    sequence_window: int | None,
    min_history: int,
) -> SequenceSplit:
    df = pd.read_csv(csv_path).dropna()
    thresholds = parse_stage_thresholds(stage_thresholds)
    df = add_derived_stage(df, stage_col, stage_source_col, thresholds)
    features = infer_feature_columns(df, {subject_col, time_col, target_col, "_pd_stage"}, feature_cols)

    subjects = df[subject_col].drop_duplicates().to_numpy()
    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=seed,
    )

    train_df = df[df[subject_col].isin(train_subjects)]
    scaler = StandardScaler().fit(train_df[features])

    def raw_items(selected_subjects: np.ndarray) -> list[tuple[np.ndarray, float, int]]:
        items: list[tuple[np.ndarray, float, int]] = []
        for subject in selected_subjects:
            sdf = df[df[subject_col] == subject].sort_values(time_col)
            values = scaler.transform(sdf[features]).astype(np.float32)
            targets = sdf[target_col].to_numpy(dtype=np.float32)
            stages = sdf["_pd_stage"].to_numpy(dtype=np.int64)
            for idx in range(max(1, min_history), len(sdf)):
                stage = int(stages[idx])
                if stage != forecast_target_stage:
                    continue
                start = 0 if sequence_window is None or sequence_window <= 0 else max(0, idx - sequence_window)
                history = values[start:idx]
                if len(history) < min_history:
                    continue
                items.append((history, float(targets[idx]), stage))
        return items

    train_raw = raw_items(train_subjects)
    val_raw = raw_items(val_subjects)
    if not train_raw or not val_raw:
        raise ValueError(
            "No paper-stage forecast samples were created. Adjust stage thresholds, "
            "forecast target stage, sequence window, or validation split."
        )

    train_targets = np.array([item[1] for item in train_raw], dtype=np.float32)
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std() if train_targets.std() > 0 else 1.0)

    def scale_items(items: list[tuple[np.ndarray, float, int]]) -> list[tuple[np.ndarray, float, int]]:
        return [(x, float((y - target_mean) / target_std), stage) for x, y, stage in items]

    train_flat = scaler.transform(train_df[features]).astype(np.float32)
    train_flat_y = train_df[target_col].to_numpy(dtype=np.float32)

    return SequenceSplit(
        scale_items(train_raw),
        scale_items(val_raw),
        train_flat,
        train_flat_y,
        scaler,
        features,
        np.array([f"stage_{forecast_target_stage}"]),
        target_mean,
        target_std,
    )


def collate_sequences(batch):
    xs = [item[0] for item in batch]
    lengths = [item[1] for item in batch]
    ys = [item[2] for item in batch]
    padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    if len(batch[0]) == 4:
        stages = [item[3] for item in batch]
        return padded, torch.stack(lengths), torch.stack(ys), torch.stack(stages)
    return padded, torch.stack(lengths), torch.stack(ys)
