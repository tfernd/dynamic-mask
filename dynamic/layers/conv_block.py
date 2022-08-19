#%%
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
        /,
        *,
        kernel_size: int = 1,
        ratio: float = 1,
        groups: int = 1,
        full: bool = True,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        assert ratio > 0
        assert in_channels % groups == 0
        assert out_channels % groups == 0

        max_channels = max(in_channels, out_channels)
        mid_channels = math.ceil(ratio * max_channels / groups) * groups

        # recompute ratio
        ratio = mid_channels / max_channels

        # parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.ratio = ratio
        self.groups = groups
        self.full = full

        self.mid_channels = mid_channels

        # functions helpers
        convk = partial(
            nn.Conv2d,
            kernel_size=kernel_size,
            padding="same",
            groups=groups,
        )
        conv1 = partial(
            nn.Conv2d,
            kernel_size=1,
            groups=groups,
        )

        # TODO find best position for normalization
        # norm = partial(nn.GroupNorm, groups)

        ## layers
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

        self.scale = Parameter(torch.zeros(1, out_channels, 1, 1))

        # ! is Identity always used?
        if in_channels == out_channels:
            self.skip = nn.Identity()
        elif full:
            self.skip = conv1(in_channels, out_channels)
        else:
            self.skip = nn.Sequential(
                conv1(in_channels, mid_channels),
                conv1(mid_channels, out_channels),
            )

    def forward(self, x: Tensor, /) -> Tensor:
        return self.skip(x) + self.main(x) * self.gate(x) * self.scale

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        out = (
            f"{name}("
            f"{self.in_channels} -> "
            f"{self.mid_channels} -> "
            f"{self.out_channels}; "
            f"kernel_size={self.kernel_size}, "
            f"groups={self.groups}"
            ")"
        )

        return out


class Block(nn.Module):
    def __init__(
        self,
        features: int,
        /,
        *,
        kernel_size: int = 3,
        groups: int = 1,
    ) -> None:
        super().__init__()

        # parameters
        assert features % groups == 0

        self.features = features
        self.kernel_size = kernel_size
        self.groups = groups

        # helpers
        conv = partial(nn.Conv2d, features, features, padding="same")

        # layers
        self.local_avg = conv(kernel_size, groups=features)
        self.global_avg = Reduce("b c h w -> b c 1 1", "mean")

        self.layer1 = conv(kernel_size=1, groups=groups)
        self.layer2 = conv(kernel_size=1, groups=groups)

        # initialize weights
        W = self.local_avg.weight
        W.data.fill_(1 / kernel_size**2)

    def forward(self, x: Tensor, /) -> Tensor:
        x = x + self.layer1(x) * self.local_avg(x).sigmoid()
        x = x + self.layer2(x) * self.global_avg(x).sigmoid()

        return x

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        out = (
            f"{name}("
            f"features={self.features}, "
            f"kernel_size={self.kernel_size}, "
            f"groups={self.groups}"
            ")"
        )

        return out
