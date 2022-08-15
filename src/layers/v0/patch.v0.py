from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

from .block import Block
from .pos_encoding import PositionalEncoding


class PatchBase(nn.Module):
    """Base class for Patch-Encoder/Decoder and PatchMixer."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        patch_size: int,
        channel_ratio: float | tuple[float, float] = 1,
        spatial_ratio: float | tuple[float, float] = 1,
    ):
        super().__init__()

        # parameters
        self.shape = (self.height, self.width, self.num_channels) = shape
        self.patch_size = patch_size
        self.channel_ratio = channel_ratio
        self.spatial_ratio = spatial_ratio

        self.emb_size = self.num_channels * self.patch_size**2

        self.height_patches = self.height // self.patch_size
        self.width_patches = self.width // self.patch_size
        self.patch_emb_size = self.height_patches * self.width_patches


class PatchMixer(PatchBase):
    """MLP-based patch mixer."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        patch_size: int,
        channel_ratio: float = 1,
        spatial_ratio: float = 1,
    ):
        super().__init__(shape, patch_size, channel_ratio, spatial_ratio)

        self.spatial_last = Rearrange("b h w e -> b e (h w)")
        self.extra_patch_mix = Block(self.patch_emb_size, ratio=spatial_ratio)
        self.channel_last = Rearrange(
            "b e (h w) -> b h w e",
            h=self.height_patches,
            w=self.width_patches,
        )
        self.intra_patch_mix = Block(self.emb_size, ratio=channel_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = self.spatial_last(x)
        x = self.extra_patch_mix(x)
        x = self.channel_last(x)
        x = self.intra_patch_mix(x)

        return x


class PatchEncoder(PatchBase):
    """Encode a patch into a latent vector."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        patch_size: int,
        channel_ratio: float = 1,
        spatial_ratio: float = 1,
        num_layers: int = 1,
    ):
        super().__init__(shape, patch_size, channel_ratio, spatial_ratio)

        # layers
        self.mix_color = Block(self.num_channels)
        self.to_patches = Rearrange(
            "b (h hp) (w wp) c -> b h w (c hp wp)",
            hp=self.patch_size,
            wp=self.patch_size,
        )
        self.pos_enc = PositionalEncoding(self.emb_size, ndim=2)
        self.intra_patch_mix = Block(self.emb_size, ratio=channel_ratio)
        self.extra_patch_mix = nn.Sequential(
            *[
                PatchMixer(shape, patch_size, channel_ratio, spatial_ratio)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, /) -> Tensor:
        x = self.mix_color(x)
        x = self.to_patches(x)
        x = x + self.pos_enc(x)
        x = self.intra_patch_mix(x)
        x = self.extra_patch_mix(x)

        return x


class PatchDecoder(PatchBase):
    """Decode a latent vector into a patch."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        patch_size: int,
        channel_ratio: float = 1,
        spatial_ratio: float = 1,
        num_layers: int = 1,
    ):
        super().__init__(shape, patch_size, channel_ratio, spatial_ratio)

        # layers
        self.extra_patch_unmix = nn.Sequential(
            *[
                PatchMixer(shape, patch_size, channel_ratio, spatial_ratio)
                for _ in range(num_layers)
            ]
        )
        self.intra_patch_unmix = Block(self.emb_size, ratio=channel_ratio)
        self.from_patches = Rearrange(
            "b h w (c hp wp) -> b (h hp) (w wp) c",
            hp=patch_size,
            wp=patch_size,
        )
        self.unmix_color = Block(self.num_channels)

    def forward(self, z: Tensor, /) -> Tensor:
        z = self.extra_patch_unmix(z)
        z = self.intra_patch_unmix(z)
        z = self.from_patches(z)
        z = self.unmix_color(z)

        return z
