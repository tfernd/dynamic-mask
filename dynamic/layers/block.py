from __future__ import annotations

from functools import partial

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Reduce


class Block(nn.Module):
    """A squeeze-excitation block uses global and local features to gate the features flow."""

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
        conv = partial(
            nn.Conv2d,
            features,
            features,
            kernel_size=1,
            groups=groups,
            padding="same",
        )
        norm = partial(nn.GroupNorm, groups, features)

        # layers
        local_avg = conv(kernel_size=kernel_size, groups=features)
        global_avg = Reduce("b c h w -> b c 1 1", "mean")

        # initialize weights
        W = local_avg.weight
        W.data.fill_(1 / kernel_size**2)

        self.layer1 = nn.Sequential(norm(), conv())
        self.layer2 = nn.Sequential(norm(), conv())

        self.gate1 = nn.Sequential(local_avg, norm(), conv(), nn.Sigmoid())
        self.gate2 = nn.Sequential(global_avg, norm(), conv(), nn.Sigmoid())

    def forward(self, x: Tensor, /) -> Tensor:
        x = x + self.gate1(x) * self.layer1(x)
        x = x + self.gate2(x) * self.layer2(x)

        return x

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        args = ", ".join(
            [
                f"features={self.features}",
                f"kernel_size={self.kernel_size}",
                f"groups={self.groups}",
            ]
        )

        return f"{name}({args})"
