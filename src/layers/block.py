from __future__ import annotations
from typing import Optional

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from einops.layers.torch import Reduce


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        /,
        *,
        ratio: float = 1,
    ) -> None:
        super().__init__()

        assert ratio > 0

        out_channels = out_channels or in_channels
        mid_channels = math.ceil(ratio * max(in_channels, out_channels))

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio

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
        self.squeeze = nn.Sequential(
            Reduce("b h w c -> b () () c", "mean"),
            nn.Linear(in_channels, mid_channels),
            nn.GELU(),
            nn.Linear(mid_channels, out_channels),
            nn.Sigmoid(),
        )

        # ! Can be costy!
        self.skip = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, /) -> Tensor:
        # affine transformation
        xn = x * self.scale + self.shift

        return self.skip(x) + self.main(xn) * self.squeeze(xn)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return (
            f"{name}("
            f"{self.in_channels} -> "
            f"{self.mid_channels} -> "
            f"{self.out_channels}"
            ")"
        )
