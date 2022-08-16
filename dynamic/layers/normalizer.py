from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# TODO find better name
class Normalizer(nn.Module):
    """Normalize the features to have zero mean and unit variance."""

    mean: Tensor
    std: Tensor

    def __init__(self, features: int, beta: float = 0.9):
        super().__init__()

        # parameters
        self.features = features
        self.beta = beta

        # layers
        self.register_buffer("mean", torch.zeros(1, features, 1, 1))
        self.register_buffer("std", torch.ones(1, features, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True)

            self.mean *= self.beta
            self.std *= self.beta

            self.mean += (1 - self.beta) * mean
            self.std += (1 - self.beta) * std
        else:
            mean = self.mean
            std = self.std

        return (x - mean) / std
