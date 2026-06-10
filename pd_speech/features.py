from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_regression
from sklearn.neighbors import KNeighborsClassifier


def relief_f_scores(x: np.ndarray, y: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
    """Small Relief-F implementation for numeric tabular features."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)
    n_samples, n_features = x.shape
    scores = np.zeros(n_features, dtype=np.float32)

    ranges = x.max(axis=0) - x.min(axis=0)
    ranges[ranges == 0] = 1.0
    x_norm = (x - x.min(axis=0)) / ranges

    classes, class_counts = np.unique(y, return_counts=True)
    priors = {label: count / n_samples for label, count in zip(classes, class_counts)}

    for i in range(n_samples):
        distances = np.linalg.norm(x_norm - x_norm[i], axis=1)
        distances[i] = np.inf

        same_mask = y == y[i]
        same_mask[i] = False
        hit_idx = np.argsort(np.where(same_mask, distances, np.inf))[:n_neighbors]
        hit_idx = hit_idx[np.isfinite(distances[hit_idx])]

        if len(hit_idx) > 0:
            scores -= np.abs(x_norm[i] - x_norm[hit_idx]).mean(axis=0) / n_samples

        miss_weight_total = 1.0 - priors[y[i]]
        if miss_weight_total <= 0:
            continue

        for cls in classes:
            if cls == y[i]:
                continue
            cls_mask = y == cls
            miss_idx = np.argsort(np.where(cls_mask, distances, np.inf))[:n_neighbors]
            miss_idx = miss_idx[np.isfinite(distances[miss_idx])]
            if len(miss_idx) == 0:
                continue
            weight = priors[cls] / miss_weight_total
            scores += weight * np.abs(x_norm[i] - x_norm[miss_idx]).mean(axis=0) / n_samples

    return scores


def select_features(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    num_features: int | None,
) -> np.ndarray:
    if method == "none" or num_features is None or num_features >= x.shape[1]:
        return np.arange(x.shape[1])

    if method == "relief":
        scores = relief_f_scores(x, y)
        return np.argsort(scores)[::-1][:num_features]

    if method == "mutual_info":
        scores = mutual_info_regression(x, y, random_state=42)
        return np.argsort(scores)[::-1][:num_features]

    if method == "sfs":
        estimator = KNeighborsClassifier(n_neighbors=5)
        selector = SequentialFeatureSelector(
            estimator,
            n_features_to_select=num_features,
            direction="forward",
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
        )
        selector.fit(x, y)
        return np.flatnonzero(selector.get_support())

    raise ValueError(f"Unknown feature selection method: {method}")
