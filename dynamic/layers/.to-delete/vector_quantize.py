from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from einops import rearrange


class VectorQuantize(nn.Module):
    def __init__(
        self,
        features: int,
        codes: int,
    ):
        super().__init__()

        # parameters
        self.features = features
        self.codes = codes

        # layers
        self.codebook = Parameter(torch.randn(codes, features).div(features))

    def encode(self, idx: Tensor) -> Tensor:
        zq = self.codebook[idx]
        zq = rearrange(zq, "b h w c -> b c h w")

        return zq

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        B, C, H, W = z.shape

        with torch.no_grad():
            zf = rearrange(z, "b c h w -> (b h w) c")

            # distance (slow) [debug-only]
            # d2 = zf.unsqueeze(1) - self.codebook.unsqueeze(0)
            # d2 = d2.pow_(2).sum(-1)

            dist = zf @ self.codebook.mul(-2).t()
            dist += zf.pow(2).sum(dim=1, keepdim=True)
            dist += self.codebook.pow(2).sum(dim=1, keepdim=True).t()

            idx = dist.argmin(dim=1)
            idx = rearrange(idx, "(b h w) -> b h w", b=B, h=H, w=W)

        zq = self.encode(idx)

        # copy gradients
        if self.training:
            alpha = torch.rand((B, 1, H, W), device=z.device) > 0.5
            zq = z * alpha + zq * (~alpha)

        return zq, idx
