from __future__ import annotations
from typing import Optional

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor


class Block(nn.Module):
    """A super-block with affine transformation, gated/squeeze-extitation activation, and skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        ratio: float = 1,
        full: bool = True,
    ) -> None:
        super().__init__()

        assert ratio > 0

        out_channels = out_channels or in_channels
        mid_channels = math.ceil(ratio * max(in_channels, out_channels))

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.full = full

        self.mid_channels = mid_channels

        ## layers
        # affine transformation
        self.scale = Parameter(torch.ones(in_channels))
        self.shift = Parameter(torch.zeros(in_channels))

        self.main = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
        )

        # squeeze-excitation / gated activation
        self.gate = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
            nn.Sigmoid(),
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        elif full:
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.skip = nn.Sequential(
                nn.Linear(in_channels, mid_channels),
                nn.Linear(mid_channels, out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        # affine transformation
        xn = x * self.scale + self.shift

        xmean = average_pool(xn)

        return self.skip(x) + self.main(xn) * self.gate(xmean)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return (
            f"{name}("
            f"{self.in_channels} -> "
            f"{self.mid_channels} -> "
            f"{self.out_channels}"
            ")"
        )


def average_pool(x: Tensor) -> Tensor:
    """Average pooling of 'spatial' dimensions."""

    B, *shape, C = x.shape

    sdims = len(shape)
    if sdims == 0:
        return x

    return x.mean(dim=tuple(range(1, sdims + 1)), keepdim=True)
