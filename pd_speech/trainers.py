from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainResult:
    best_val_accuracy: float
    best_val_f1: float
    report: str
    confusion: np.ndarray
    history: list[dict[str, float]]
    best_val_rmse: float | None = None
    best_val_mae: float | None = None
    best_val_r2: float | None = None


def make_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            try:
                torch.empty(1, device="cuda")
                return torch.device("cuda")
            except Exception as exc:
                print(f"CUDA is visible but unusable with this PyTorch build; falling back to CPU. Reason: {exc}")
        return torch.device("cpu")
    return torch.device(requested)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
                logits = model(x.to(device))
            else:
                x, lengths, y = batch
                logits = model(x.to(device), lengths.to(device))
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
                outputs = model(x.to(device))
            else:
                x, lengths, y = batch
                outputs = model(x.to(device), lengths.to(device))
            preds.append(outputs.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    return np.concatenate(targets), np.concatenate(preds)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_mean: float, target_std: float) -> dict[str, float]:
    y_true_raw = y_true * target_std + target_mean
    y_pred_raw = y_pred * target_std + target_mean
    rmse = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
    mae = float(mean_absolute_error(y_true_raw, y_pred_raw))
    r2 = float(r2_score(y_true_raw, y_pred_raw))
    mse_scaled = float(mean_squared_error(y_true, y_pred))
    return {
        "val_rmse": rmse,
        "val_mae": mae,
        "val_r2": r2,
        "val_loss": mse_scaled,
    }


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    class_names: list[str],
    log_path: str | None = None,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_acc = -1.0
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    log_file = Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("epoch,train_loss,val_accuracy,val_f1\n", encoding="utf-8")

    for epoch in tqdm(range(1, epochs + 1), desc="training"):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if len(batch) == 2:
                x, y = batch
                logits = model(x.to(device))
            else:
                x, lengths, y = batch
                logits = model(x.to(device), lengths.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(y.size(0))
            total_examples += int(y.size(0))

        y_true, y_pred = evaluate(model, val_loader, device)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        train_loss = total_loss / max(total_examples, 1)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_accuracy": float(acc),
            "val_f1": float(f1),
        }
        history.append(row)
        if log_file:
            with log_file.open("a", encoding="utf-8") as file:
                file.write(f"{epoch},{train_loss:.8f},{acc:.8f},{f1:.8f}\n")
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = evaluate(model, val_loader, device)
    return TrainResult(
        best_val_accuracy=accuracy_score(y_true, y_pred),
        best_val_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        report=classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
        confusion=confusion_matrix(y_true, y_pred),
        history=history,
    )


def train_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    target_mean: float,
    target_std: float,
    log_path: str | None = None,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_rmse = float("inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    log_file = Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("epoch,train_loss,val_loss,val_rmse,val_mae,val_r2\n", encoding="utf-8")

    for epoch in tqdm(range(1, epochs + 1), desc="training"):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if len(batch) == 2:
                x, y = batch
                outputs = model(x.to(device))
            else:
                x, lengths, y = batch
                outputs = model(x.to(device), lengths.to(device))
            y = y.float().to(device)
            loss = criterion(outputs, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(y.size(0))
            total_examples += int(y.size(0))

        y_true, y_pred = evaluate_regression(model, val_loader, device)
        metrics = regression_metrics(y_true, y_pred, target_mean, target_std)
        train_loss = total_loss / max(total_examples, 1)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            **metrics,
        }
        history.append(row)
        if log_file:
            with log_file.open("a", encoding="utf-8") as file:
                file.write(
                    f"{epoch},{train_loss:.8f},{metrics['val_loss']:.8f},"
                    f"{metrics['val_rmse']:.8f},{metrics['val_mae']:.8f},{metrics['val_r2']:.8f}\n"
                )
        if metrics["val_rmse"] < best_rmse:
            best_rmse = metrics["val_rmse"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = evaluate_regression(model, val_loader, device)
    final_metrics = regression_metrics(y_true, y_pred, target_mean, target_std)
    best_metrics = best_metrics or final_metrics
    report = (
        f"RMSE: {final_metrics['val_rmse']:.4f}\n"
        f"MAE: {final_metrics['val_mae']:.4f}\n"
        f"R2: {final_metrics['val_r2']:.4f}"
    )
    return TrainResult(
        best_val_accuracy=0.0,
        best_val_f1=0.0,
        report=report,
        confusion=np.empty((0, 0)),
        history=history,
        best_val_rmse=best_metrics["val_rmse"],
        best_val_mae=best_metrics["val_mae"],
        best_val_r2=best_metrics["val_r2"],
    )
