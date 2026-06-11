from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from paper_reproduction.data import TabularData
from paper_reproduction.evaluation import cross_validate_regressor, fit_and_score_regressor
from paper_reproduction.features import select_relief, select_sfs
from paper_reproduction.models import make_regressor
from paper_reproduction.preprocessing import simple_smote


def run_stage_detection(
    data: TabularData,
    out_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        data.x,
        data.y_stage,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data.y_stage,
    )
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_train_bal, y_train_bal = simple_smote(x_train, y_train, args.smote_k, rng)

    selectors = {
        "none": np.arange(x_train_bal.shape[1]),
        "relief": select_relief(x_train_bal, y_train_bal, args.num_features),
        "sfs": select_sfs(x_train_bal, y_train_bal, args.num_features),
    }

    rows = []
    matrices = {}
    labels = sorted(np.unique(np.concatenate([y_train_bal, y_test])).tolist())
    label_names = stage_label_names(labels)
    for selector_name, selected in selectors.items():
        names = [data.features[i] for i in selected]
        for model_name in ["mlp", "svm"]:
            model = make_regressor(model_name, args.seed, args.mlp_epochs)
            result = fit_and_score_regressor(
                model,
                x_train_bal[:, selected],
                y_train_bal.astype(np.float32),
                x_test[:, selected],
                y_test.astype(np.float32),
            )
            cv = cross_validate_regressor(
                model_name,
                x_train_bal[:, selected],
                y_train_bal,
                args.seed,
                args.mlp_epochs,
            )
            rows.append(
                {
                    "feature_selection": selector_name,
                    "model": model_name,
                    "features": names,
                    **result["metrics"],
                    **cv,
                }
            )
            matrices[f"{model_name}_{selector_name}"] = {
                "labels": label_names,
                "matrix": result["confusion"].tolist(),
            }

    (out_dir / "stage_detection_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    (out_dir / "stage_detection_confusion_matrices.json").write_text(
        json.dumps(matrices, indent=2),
        encoding="utf-8",
    )


def stage_label_names(labels: list[int]) -> list[str]:
    if labels == [1, 2, 3]:
        return ["Level 1", "Level 2", "Level 3/4"]
    return [f"Stage {label}" for label in labels]
