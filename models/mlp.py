"""
MLP Predictor
=============
[8 -> 64 -> 32 -> 1]  with ReLU, Dropout, BatchNorm, Sigmoid output.
Predicts association probability from concatenated NMF pair features.
"""

import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    def __init__(self, input_dim: int = 8, hidden1: int = 64,
                 hidden2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
