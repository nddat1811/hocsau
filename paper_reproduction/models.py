from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def make_regressor(model_name: str, seed: int, max_iter: int):
    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(25, 25, 25),
            activation="relu",
            solver="lbfgs",
            alpha=0.0,
            learning_rate_init=0.1,
            max_iter=max_iter,
            max_fun=max(15000, max_iter * 50),
            random_state=seed,
        )
    if model_name == "svm":
        gamma = 1.0 / (2.0 * 2.0**2)
        return SVR(kernel="rbf", gamma=gamma, C=1.0)
    raise ValueError(f"Unknown model: {model_name}")


class PaperLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 150) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1]).squeeze(-1)
