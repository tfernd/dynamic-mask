from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from ..utils import auto_grad


# TODO models with this don't converge well...
class QuantizeLatent(nn.Module):
    def __init__(
        self,
        features: int,
        /,
        code_size: int,
        *,
        beta: float = 0.1,
    ):
        super().__init__()

        # parameters
        self.features = features
        self.code_size = code_size
        self.beta = beta

        # layers
        emb = torch.randn(code_size, features)
        self.emb = Parameter(emb)

    @auto_grad
    def forward(self, z: Tensor,/) -> tuple[Tensor, Tensor]:
        *shape, C = z.shape

        with torch.no_grad():
            zf = z.flatten(0, -2)

            # (z - emb)**2
            dist = 2 * zf @ self.emb.t()
            dist += zf.pow(2).sum(dim=1, keepdim=True)
            dist += self.emb.pow(2).sum(dim=1).unsqueeze(0)  # TODO keepdim and .t()?

            idx = dist.argmin(dim=1)
            idx = idx.view(*shape)

        zq = self.emb[idx]

        embedding_loss = F.mse_loss(zq, z.detach())
        commitment_loss = F.mse_loss(zq.detach(), z)

        loss = embedding_loss + commitment_loss * self.beta

        out = z + (zq - z).detach()

        return out, loss

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        args = ", ".join(
            [
                f"features={self.features}",
                f"code_size={self.code_size}",
                f"beta={self.beta}",
            ]
        )

        return f"{name}({args})"
