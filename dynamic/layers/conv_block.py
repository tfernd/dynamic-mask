from __future__ import annotations
from typing import Optional

from functools import partial

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from einops.layers.torch import Reduce


class ConvBlock(nn.Module):
    """A convolutional block that uses a squeeze-excitation layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *,
        kernel_size: int = 1,
        ratio: float = 1,
        full: bool = True,
    ) -> None:
        super().__init__()

        assert ratio > 0
        assert kernel_size % 2 == 1

        out_channels = out_channels or in_channels

        max_channels = max(in_channels, out_channels)
        mid_channels = math.ceil(ratio * max_channels)

        # recompute ratio
        ratio = mid_channels / max_channels

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.ratio = ratio
        self.full = full

        self.mid_channels = mid_channels

        # functions helpers
        convk = partial(nn.Conv2d, kernel_size=kernel_size, padding=kernel_size // 2)
        conv1 = partial(nn.Conv2d, kernel_size=1)

        # TODO Add normalization?

        ## layers
        # affine transformation
        self.scale = Parameter(torch.ones(1, in_channels, 1, 1))
        self.shift = Parameter(torch.zeros(1, in_channels, 1, 1))

        self.main = nn.Sequential(
            convk(in_channels, mid_channels),
            nn.GELU(),
            convk(mid_channels, out_channels),
        )

        # squeeze-excitation / gated activation
        self.gate = nn.Sequential(
            conv1(in_channels, mid_channels),
            Reduce("b c h w -> b c 1 1", "mean"),
            nn.GELU(),
            conv1(mid_channels, out_channels),
            nn.Sigmoid(),
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        elif full:
            self.skip = conv1(in_channels, out_channels)
        else:
            self.skip = nn.Sequential(
                conv1(in_channels, mid_channels),
                conv1(mid_channels, out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        xn = x * self.scale + self.shift

        return self.skip(x) + self.main(xn) * self.gate(xn)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        out = (
            f"{name}("
            f"{self.in_channels} -> "
            f"{self.mid_channels} -> "
            f"{self.out_channels}; "
            f"kernel_size={self.kernel_size}"
            ")"
        )

        return out
