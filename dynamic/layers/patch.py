from __future__ import annotations

from functools import partial

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from .conv_block import ConvBlock


class PatchBase(nn.Module):
    def __init__(
        self,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        num_layers: int = 1,
    ):
        super().__init__()

        # parameters
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.ratio = ratio
        self.num_layers = num_layers

        self.emb_size = num_channels * patch_size**2

        # function helper
        self.block = partial(ConvBlock, kernel_size=kernel_size, ratio=ratio)


class PatchEncoder(PatchBase):
    """Encode a patch into a latent vector."""

    def __init__(
        self,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        num_layers: int = 1,
    ):
        super().__init__(patch_size, kernel_size, num_channels, ratio, num_layers)

        # layers
        self.mix_color = self.block(num_channels, ratio=1)
        self.to_patches = Rearrange(
            "b c (h hp) (w wp) -> b (c hp wp) h w",
            hp=patch_size,
            wp=patch_size,
        )
        self.patches_mix = nn.Sequential(
            *[self.block(self.emb_size) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mix_color(x)
        x = self.to_patches(x)
        x = self.patches_mix(x)

        return x


class PatchDecoder(PatchBase):
    """Decode a latent vector into a patch."""

    def __init__(
        self,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        num_layers: int = 1,
    ):
        super().__init__(patch_size, kernel_size, num_channels, ratio, num_layers)

        # layers
        self.patches_unmix = nn.Sequential(
            *[self.block(self.emb_size) for _ in range(num_layers)]
        )
        self.from_patches = Rearrange(
            "b (c hp wp) h w -> b c (h hp) (w wp)",
            hp=patch_size,
            wp=patch_size,
        )
        self.unmix_color = self.block(num_channels, ratio=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.patches_unmix(z)
        z = self.from_patches(z)
        z = self.unmix_color(z)

        return z
