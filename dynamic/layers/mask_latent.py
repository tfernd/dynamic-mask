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

    def __init__(
        self,
        features: int,
        groups: int = 1,
    ) -> None:
        super().__init__()

        assert features >= 1
        assert groups >= 1
        assert features % groups == 0

        # parameters
        self.features = features
        self.groups = groups

        # layers
        masks = ~torch.eye(features + 1).cumsum(0).bool()
        masks = masks[:, 1:]

        select = masks.sum(1) % groups == 0
        masks = masks[select]

        self.register_buffer("masks", masks)

        self.rearrange = Rearrange("b h w c -> b c h w")

    def forward(
        self,
        z: Tensor,
        /,
        *,
        mask: Optional[Tensor] = None,
        n: Optional[int] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Mask the latent space and possibly crop it."""

        if self.training:
            if mask is None:
                B, C, H, W = z.shape

                idx = torch.randint(0, self.masks.size(0), (B, H, W), device=z.device)
                mask = self.rearrange(self.masks[idx])

        if mask is not None:
            z = z.masked_fill(mask, 0)

        if n is not None:
            assert 0 < n <= 1
            assert n % self.groups == 0

            z = z[:, :n]

        return z, mask

    def expand(self, z: Tensor, /) -> Tensor:
        """Expand the latent space."""

        B, C, H, W = z.shape

        if C == self.features:
            return z

        assert C < self.features

        zeros = z.new_zeros((B, self.features - C, H, W))
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
