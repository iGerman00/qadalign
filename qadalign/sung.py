"""Sung / not-sung head.

Classifier over the aligner's encoder output (128-dim), so one forward pass
yields phone posteriors and a per-frame probability that anything is being sung.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SungHead(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(),
                                 nn.Linear(dim // 2, 1))

    def forward(self, x):                     # (T, dim) -> (T,)
        return self.net(x).squeeze(-1)
