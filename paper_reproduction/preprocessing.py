from __future__ import annotations

import numpy as np


def simple_smote(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    xs = [x]
    ys = [y]
    labels, counts = np.unique(y, return_counts=True)
    target_count = int(counts.max())
    for label, count in zip(labels, counts):
        need = target_count - int(count)
        if need <= 0:
            continue
        class_x = x[y == label]
        synthetic = []
        for _ in range(need):
            i = int(rng.integers(0, len(class_x)))
            distances = np.linalg.norm(class_x - class_x[i], axis=1)
            neighbor_order = np.argsort(distances)
            neighbor_pool = neighbor_order[1 : min(k + 1, len(neighbor_order))]
            j = i if len(neighbor_pool) == 0 else int(rng.choice(neighbor_pool))
            gap = float(rng.random())
            synthetic.append(class_x[i] + gap * (class_x[j] - class_x[i]))
        xs.append(np.asarray(synthetic, dtype=np.float32))
        ys.append(np.full(need, label, dtype=y.dtype))
    return np.vstack(xs), np.concatenate(ys)
