from __future__ import annotations
from typing import Optional

import math

import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange


class MaskLatent(nn.Module):
    """Randomly mask the latent space and possibly crop it."""

    masks: Tensor

    def __init__(self, features: int) -> None:
        super().__init__()

        assert features >= 1

        # parameters
        self.features = features

        # layers
        masks = ~torch.eye(features + 1).cumsum(0).bool()
        masks = masks[:, 1:]
        self.register_buffer("masks", masks)

        self.rearrange = Rearrange("b h w c -> b c h w")

    def forward(self, z: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        """Mask the latent space."""

        if not self.training:
            return z, None

        B, C, H, W = z.shape

        idx = torch.randint(0, self.masks.size(0), (B, H, W), device=z.device)
        mask = self.rearrange(self.masks[idx])
        z = z.masked_fill(mask, 0)

        return z, mask

    def crop(
        self,
        z: Tensor,
        n: int | float = 1.0,
    ) -> Tensor:
        """Crop the latent space."""

        if not isinstance(n, int):
            assert 0 < n <= 1
            n = math.ceil(n * self.features)

        assert 1 <= n <= self.features

        return z[:, :n]

    def expand(self, z: Tensor) -> Tensor:
        """Expand the latent space."""

        B, C, H, W = z.shape

        if C == self.features:
            return z

        assert C < self.features

        zeros = torch.zeros(B, self.features - C, H, W, device=z.device)
        z = torch.cat([z, zeros], dim=1)

        return z

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        args = ", ".join(
            [
                f"features={self.features}",
            ]
        )

        return f"{name}({args})"
