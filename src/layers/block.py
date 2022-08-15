from __future__ import annotations
from typing import Optional

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from einops.layers.torch import Reduce


class Block(nn.Module):
    """A super-block with affine transformation, gated/squeeze-extitation activation, and skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        /,
        *,
        ratio: float | tuple[float, float] = 1,
        full: bool = True,
    ) -> None:
        super().__init__()

        ratio = ratio if isinstance(ratio, tuple) else (ratio, ratio)

        assert ratio[0] > 0 and ratio[1] > 0

        out_channels = out_channels or in_channels

        max_channels = max(in_channels, out_channels)
        mid_channels = (
            math.ceil(ratio[0] * max_channels),
            math.ceil(ratio[1] * max_channels),
        )

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
            nn.Linear(in_channels, mid_channels[0]),
            nn.GELU(),
            nn.Linear(mid_channels[0], out_channels),
        )

        # squeeze-excitation / gated activation
        self.gate = nn.Sequential(
            nn.Linear(in_channels, mid_channels[1]),
            Reduce("b h w c -> b 1 1 c", "mean"),
            nn.GELU(),
            nn.Linear(mid_channels[1], out_channels),
            nn.Sigmoid(),
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        elif full:
            self.skip = nn.Linear(in_channels, out_channels)
        else:
            self.skip = nn.Sequential(
                nn.Linear(in_channels, mid_channels[0]),
                nn.Linear(mid_channels[0], out_channels),
            )

    def forward(self, x: Tensor, /) -> Tensor:
        # affine transformation
        xn = x * self.scale + self.shift

        return self.skip(x) + self.main(xn) * self.gate(xn)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        if self.mid_channels[0] == self.mid_channels[1]:
            mid_channels = self.mid_channels[0]
        else:
            mid_channels = f"{self.mid_channels[0]}/{self.mid_channels[1]}"

        return (
            f"{name}("
            f"{self.in_channels} -> "
            f"{mid_channels} -> "
            f"{self.out_channels}"
            ")"
        )
