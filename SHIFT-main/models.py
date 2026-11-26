"""PyTorch MLP: state (B, 4) -> acceleration (B, 2)."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim=4, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),  # no final activation
        )

    def forward(self, x):
        return self.net(x)