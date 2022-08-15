from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

from einops.layers.torch import Rearrange

from .block import Block
from .pos_encoding import PositionalEncoding


class PatchEncoder(nn.Module):
    """Encode a patch into a latent vector."""

    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        ratio: float | tuple[float, float] = 1,
        num_layers: int = 1,
    ):
        super().__init__()

        # parameters
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.ratio = ratio
        self.num_layers = num_layers

        self.emb_size = emb_size = num_channels * patch_size**2

        # layers
        self.mix_color = Block(num_channels)
        self.to_patches = Rearrange(
            "b (h hp) (w wp) c -> b h w (c hp wp)",
            hp=patch_size,
            wp=patch_size,
        )
        self.pre_mix = Block(emb_size)
        self.pos_enc = PositionalEncoding(emb_size, ndim=2)
        self.patches_mix = nn.Sequential(
            *[Block(emb_size, ratio=ratio) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mix_color(x)
        x = self.to_patches(x)
        x = self.pre_mix(x) + self.pos_enc(x)
        x = self.patches_mix(x)

        return x


class PatchDecoder(nn.Module):
    """Decode a latent vector into a patch."""

    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        ratio: float | tuple[float, float] = 1,
        num_layers: int = 1,
    ):
        super().__init__()

        # parameters
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.ratio = ratio
        self.num_layers = num_layers

        self.emb_size = emb_size = num_channels * patch_size**2

        # layers
        self.patches_unmix = nn.Sequential(
            *[Block(emb_size, ratio=ratio) for _ in range(num_layers)]
        )
        self.from_patches = Rearrange(
            "b h w (c hp wp) -> b (h hp) (w wp) c",
            hp=patch_size,
            wp=patch_size,
        )
        self.unmix_color = Block(num_channels)

        scale = torch.linspace(0, -5, emb_size).exp().view(1, 1, 1, -1)
        self.scale = Parameter(scale)

    def forward(self, z: Tensor) -> Tensor:
        z = self.patches_unmix(z * self.scale)
        z = self.from_patches(z)
        z = self.unmix_color(z)

        return z
