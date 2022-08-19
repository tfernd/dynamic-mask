from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from ..utils import normalize, denormalize, auto_grad, auto_device
from .conv_block import ConvBlock
from .mask_latent import MaskLatent


class PatchBase(nn.Module):
    def __init__(
        self,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        groups: int = 1,
        num_layers: int = 1,
    ):
        super().__init__()

        # parameters
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.ratio = ratio
        self.groups = groups
        self.num_layers = num_layers

        self.emb_size = num_channels * patch_size**2

        self.kwargs = dict(
            patch_size=patch_size,
            kernel_size=kernel_size,
            num_channels=num_channels,
            ratio=ratio,
            groups=groups,
            num_layers=num_layers,
        )

        # function helper
        self.block = partial(
            ConvBlock,
            kernel_size=kernel_size,
            ratio=ratio,
            groups=groups,
        )


class PatchEncoder(PatchBase):
    """Encode a patch into a latent vector."""

    def __init__(
        self,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        groups: int = 1,
        num_layers: int = 1,
    ):
        super().__init__(
            patch_size=patch_size,
            kernel_size=kernel_size,
            num_channels=num_channels,
            ratio=ratio,
            groups=groups,
            num_layers=num_layers,
        )

        # layers
        self.mix_color = self.block(num_channels, ratio=1, groups=1)
        self.to_patches = Rearrange(
            "b c (h hp) (w wp) -> b (c hp wp) h w",
            hp=patch_size,
            wp=patch_size,
        )
        # ! This is needed to avoid groups > 1
        self.intra_patch_mix = nn.Conv2d(self.emb_size, self.emb_size, kernel_size=1)
        self.patches_mix = nn.Sequential(
            *[self.block(self.emb_size) for _ in range(num_layers)]
        )
        self.mask = MaskLatent(self.emb_size, groups)

        self.intra_patch_mix.weight.data.zero_()

    @auto_grad
    @auto_device
    def forward(
        self,
        x: Tensor,
        /,
        mask: Optional[Tensor] = None,
        n: Optional[int] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = normalize(x)
        x = self.mix_color(x)
        x = self.to_patches(x)
        x = x + self.intra_patch_mix(x)
        x = self.patches_mix(x)
        x, mask = self.mask(x, mask, n)
        x = denormalize(x)

        return x, mask


class PatchDecoder(PatchBase):
    """Decode a latent vector into a patch."""

    def __init__(
        self,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        groups: int = 1,
        num_layers: int = 1,
    ):
        super().__init__(
            patch_size=patch_size,
            kernel_size=kernel_size,
            num_channels=num_channels,
            ratio=ratio,
            groups=groups,
            num_layers=num_layers,
        )

        # layers
        self.mask = MaskLatent(self.emb_size, groups)
        self.patches_unmix = nn.Sequential(
            *[self.block(self.emb_size) for _ in range(num_layers)]
        )
        self.from_patches = Rearrange(
            "b (c hp wp) h w -> b c (h hp) (w wp)",
            hp=patch_size,
            wp=patch_size,
        )
        self.unmix_color = self.block(num_channels, ratio=1, groups=1)

    @auto_grad
    @auto_device
    def forward(self, z: Tensor, /) -> Tensor:
        z = normalize(z)
        z = self.mask.expand(z)
        z = self.patches_unmix(z)
        z = self.from_patches(z)
        z = self.unmix_color(z)
        z = denormalize(z)

        return z
