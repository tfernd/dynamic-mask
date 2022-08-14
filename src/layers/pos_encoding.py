from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Positional Encoding."""
    
    def __init__(
        self,
        features: int,
        *,
        ndim: int = 1,
        requires_grad: bool = False,
    ):
        super().__init__()

        # parameters
        self.features = features
        self.ndim = ndim

        # layers
        idx = torch.arange(features)

        A = torch.ones(ndim, features)
        gamma = torch.linspace(8_000, 12_000, ndim).view(ndim, 1)
        freq = gamma ** idx.div(-features)
        phase = idx.fmod(2).mul(torch.pi / 2).expand(ndim, features)

        # add spatial dimensions
        shape = [1] * ndim
        A = A.unsqueeze(1).unflatten(1, shape)
        freq = freq.unsqueeze(1).unflatten(1, shape)
        phase = phase.unsqueeze(1).unflatten(1, shape)

        self.A = Parameter(A, requires_grad=requires_grad)
        self.freq = Parameter(freq, requires_grad=requires_grad)
        self.phase = Parameter(phase, requires_grad=requires_grad)

    @property
    def requires_grad(self) -> bool:
        for param in self.parameters():
            if param.requires_grad:
                return True
        return False

    def forward(self, x: Tensor) -> Tensor:
        N, *shape, C = x.shape

        idx = torch.stack(
            torch.meshgrid(
                [torch.arange(i, device=x.device) for i in shape],
                indexing="ij",
            )
        ).unsqueeze(-1)

        arg = self.freq.mul(idx).add(self.phase)
        out = self.A.mul(arg.sin())
        out = out.mean(0, keepdim=True)

        return out


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
