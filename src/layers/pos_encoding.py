from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from ..utils import auto_grad


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        features: int,
        ndim: Literal[1] = 1,  # TODO implement support for higher dimensions
        requires_grad: bool = False,
    ):
        super().__init__()

        # parameters
        self.features = features
        self.ndim = ndim

        # layers
        idx = torch.arange(features)

        A = torch.ones(1, features)
        freq = 10_000 ** idx.div(-features)
        phase = idx.fmod(2).mul(torch.pi / 2)

        self.A = Parameter(A.view(1, 1, -1), requires_grad)
        self.freq = Parameter(freq.view(1, 1, -1), requires_grad)
        self.phase = Parameter(phase.view(1, 1, -1), requires_grad)

    @property
    def requires_grad(self) -> bool:
        for param in self.parameters():
            if param.requires_grad:
                return True
        return False

    @auto_grad
    def embedding_like(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        time = torch.arange(T, device=x.device).view(1, -1, 1)
        emb = self.A * torch.cos(self.freq * time + self.phase)

        return emb

    @auto_grad
    def add(self, x: Tensor) -> Tensor:
        return x + self.embedding_like(x)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        args = ", ".join(
            [
                f"features={self.features}",
                f"ndim={self.ndim}",
                f"requires_grad={self.requires_grad}",
            ]
        )

        return f"{name}({args})"
