from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from pd_speech.data import (
    SequenceDataset,
    TabularDataset,
    collate_sequences,
    load_detection_split,
    load_sequence_regression_split,
    load_sequence_split,
    load_tabular_regression_split,
)
from pd_speech.features import select_features
from pd_speech.models import (
    LSTMProgressionClassifier,
    LSTMProgressionRegressor,
    MLPClassifier,
    MLPRegressor,
    SpMambaProgressionClassifier,
    SpMambaProgressionRegressor,
)
from pd_speech.trainers import make_device, train_classifier, train_regressor


def parse_feature_cols(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=["detection", "regression", "progression"], default=None)
    parser.add_argument("--problem-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--target-col", default="pd_label")
    parser.add_argument("--subject-col", default="subject_id")
    parser.add_argument("--time-col", default="visit")
    parser.add_argument("--feature-cols", default=None)
    parser.add_argument("--feature-selection", choices=["none", "relief", "sfs", "mutual_info"], default="none")
    parser.add_argument("--num-features", type=int, default=None)
    parser.add_argument("--sequence-backbone", choices=["lstm", "spmamba"], default="lstm")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="runs")
    return parser


def apply_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not args.config:
        if args.task is None or args.csv is None:
            parser.error("--task and --csv are required unless --config is provided")
        return args

    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    defaults = vars(parser.parse_args([]))
    cli_values = vars(args)
    merged = defaults.copy()
    merged.update(config)

    for key, value in cli_values.items():
        if key == "config":
            merged[key] = value
            continue
        if value != defaults.get(key):
            merged[key] = value

    if merged.get("task") is None or merged.get("csv") is None:
        parser.error("Config must define task and csv")
    return argparse.Namespace(**merged)


def train_detection(args: argparse.Namespace) -> None:
    split = load_detection_split(
        args.csv,
        args.target_col,
        parse_feature_cols(args.feature_cols),
        args.test_size,
        args.seed,
    )
    selected = select_features(
        split.x_train,
        split.y_train,
        args.feature_selection,
        args.num_features,
    )
    feature_cols = [split.feature_cols[i] for i in selected]
    train_ds = TabularDataset(split.x_train[:, selected], split.y_train)
    val_ds = TabularDataset(split.x_val[:, selected], split.y_val)

    model = MLPClassifier(
        input_dim=len(selected),
        num_classes=len(split.classes),
        hidden_dims=(args.hidden_dim, max(args.hidden_dim // 2, 8)),
        dropout=args.dropout,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    result = train_classifier(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        args.weight_decay,
        make_device(args.device),
        [str(c) for c in split.classes],
        str(run_paths(args)["log_path"]),
    )
    save_run(args, model, feature_cols, [str(c) for c in split.classes], result)


def train_tabular_regression(args: argparse.Namespace) -> None:
    split = load_tabular_regression_split(
        args.csv,
        args.target_col,
        parse_feature_cols(args.feature_cols),
        args.test_size,
        args.seed,
    )
    selected = select_features(
        split.x_train,
        split.y_train,
        args.feature_selection,
        args.num_features,
    )
    feature_cols = [split.feature_cols[i] for i in selected]
    train_ds = TabularDataset(split.x_train[:, selected], split.y_train, target_dtype=torch.float32)
    val_ds = TabularDataset(split.x_val[:, selected], split.y_val, target_dtype=torch.float32)

    model = MLPRegressor(
        input_dim=len(selected),
        hidden_dims=(args.hidden_dim, max(args.hidden_dim // 2, 8)),
        dropout=args.dropout,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    result = train_regressor(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        args.weight_decay,
        make_device(args.device),
        split.target_mean,
        split.target_std,
        str(run_paths(args)["log_path"]),
    )
    save_run(args, model, feature_cols, [args.target_col], result)


def train_progression(args: argparse.Namespace) -> None:
    is_regression = args.problem_type == "regression"
    loader = load_sequence_regression_split if is_regression else load_sequence_split
    split = loader(
        args.csv,
        args.subject_col,
        args.time_col,
        args.target_col,
        parse_feature_cols(args.feature_cols),
        args.test_size,
        args.seed,
    )
    selected = select_features(
        split.train_flat_x,
        split.train_flat_y,
        args.feature_selection,
        args.num_features,
    )
    feature_cols = [split.feature_cols[i] for i in selected]
    train_items = [(x[:, selected], y) for x, y in split.train]
    val_items = [(x[:, selected], y) for x, y in split.val]

    if args.sequence_backbone == "lstm":
        if is_regression:
            model = LSTMProgressionRegressor(
                input_dim=len(selected),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        else:
            model = LSTMProgressionClassifier(
                input_dim=len(selected),
                num_classes=len(split.classes),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
    else:
        if is_regression:
            model = SpMambaProgressionRegressor(
                input_dim=len(selected),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        else:
            model = SpMambaProgressionClassifier(
                input_dim=len(selected),
                num_classes=len(split.classes),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )

    train_loader = DataLoader(
        SequenceDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        SequenceDataset(val_items),
        batch_size=args.batch_size,
        collate_fn=collate_sequences,
    )
    if is_regression:
        result = train_regressor(
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            args.weight_decay,
            make_device(args.device),
            split.target_mean,
            split.target_std,
            str(run_paths(args)["log_path"]),
        )
    else:
        result = train_classifier(
            model,
            train_loader,
            val_loader,
            args.epochs,
            args.lr,
            args.weight_decay,
            make_device(args.device),
            [str(c) for c in split.classes],
            str(run_paths(args)["log_path"]),
        )
    save_run(args, model, feature_cols, [str(c) for c in split.classes], result)


def run_paths(args) -> dict[str, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.sequence_backbone if args.task == "progression" else "mlp"
    stem = f"{args.task}_{model_name}"
    return {
        "out_dir": out_dir,
        "model_path": out_dir / f"{stem}.pt",
        "metrics_path": out_dir / f"{stem}_metrics.json",
        "log_path": out_dir / f"{stem}_train.log",
    }


def save_run(args, model, feature_cols, class_names, result) -> None:
    paths = run_paths(args)
    model_path = paths["model_path"]
    metrics_path = paths["metrics_path"]

    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "feature_cols": feature_cols,
            "class_names": class_names,
            "checkpoint_contents": {
                "model_state": "PyTorch model weights.",
                "args": "Training arguments after merging YAML config and CLI overrides.",
                "feature_cols": "Feature columns used by the model, after feature selection.",
                "class_names": "Label names in model output order.",
            },
        },
        model_path,
    )
    metrics = {
        "best_val_accuracy": result.best_val_accuracy,
        "best_val_f1": result.best_val_f1,
        "best_val_rmse": result.best_val_rmse,
        "best_val_mae": result.best_val_mae,
        "best_val_r2": result.best_val_r2,
        "classification_report": result.report,
        "confusion_matrix": result.confusion.tolist(),
        "model_path": str(model_path),
        "train_log_path": str(paths["log_path"]),
        "feature_cols": feature_cols,
        "class_names": class_names,
        "history": result.history,
        "checkpoint_contents": {
            "model_state": "PyTorch model weights.",
            "args": "Training arguments after merging YAML config and CLI overrides.",
            "feature_cols": "Feature columns used by the model, after feature selection.",
            "class_names": "Label names in model output order.",
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = build_parser()
    args = apply_config(parser.parse_args(), parser)
    torch.manual_seed(args.seed)
    if args.task == "detection":
        train_detection(args)
    elif args.task == "regression":
        train_tabular_regression(args)
    else:
        train_progression(args)
