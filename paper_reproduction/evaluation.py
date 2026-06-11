from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

from paper_reproduction.models import make_regressor


def fit_and_score_regressor(model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    rounded = np.rint(test_pred).astype(int)
    rounded = np.clip(rounded, min(labels), max(labels))
    return {
        "metrics": {
            "train_mse": float(mean_squared_error(y_train, train_pred)),
            "train_r2": float(r2_score(y_train, train_pred)),
            "test_mse": float(mean_squared_error(y_test, test_pred)),
            "test_r2": float(r2_score(y_test, test_pred)),
            "test_accuracy_rounded": float((rounded == y_test).mean()),
        },
        "confusion": confusion_matrix(y_test, rounded, labels=labels),
    }


def cross_validate_regressor(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    max_iter: int,
) -> dict:
    _, class_counts = np.unique(y, return_counts=True)
    folds = min(10, int(class_counts.min()))
    if folds < 2:
        raise ValueError("Need at least two samples per class for cross-validation.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    mse_scores = []
    r2_scores = []
    for train_idx, val_idx in splitter.split(x, y):
        model = make_regressor(model_name, seed, max_iter)
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[val_idx])
        mse_scores.append(mean_squared_error(y[val_idx], pred))
        r2_scores.append(r2_score(y[val_idx], pred))
    return {
        "cv_folds": float(folds),
        "cv_mse_mean": float(np.mean(mse_scores)),
        "cv_mse_std": float(np.std(mse_scores)),
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
    }
