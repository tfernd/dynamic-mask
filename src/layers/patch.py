from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange

from ..utils import normalize, denormalize
from .block import Block
from .pos_encoding import PositionalEncoding


class PatchBase(nn.Module):
    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        ratio: float = 1,
        num_heads: int = 1,
        head_size: Optional[int] = None,
        num_layers: int = 1,
    ):
        super().__init__()

        # parameters
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.ratio = ratio
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size or num_channels // num_heads

        self.emb_size = num_channels * patch_size**2


class TransformerBlock(nn.Module):
    """A multi-head attention transformer block with dot-product attention."""

    def __init__(
        self,
        emb_size: int,
        num_heads: int = 1,
        head_size: Optional[int] = None,
        ratio: float = 1,
    ) -> None:
        super().__init__()

        # parameters
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size or emb_size // num_heads
        self.ratio = ratio

        # layers
        self.qkv = nn.Sequential(
            Block(self.emb_size, 3 * self.num_heads * self.head_size, ratio=ratio),
            Rearrange(
                "b h w (k n s) -> k (b n) (h w) s",
                k=3,
                n=self.num_heads,
                s=self.head_size,
            ),
        )
        self.scale = 1 / (self.head_size**0.5)

        self.proj = Block(self.num_heads * self.head_size, self.emb_size, ratio=ratio)
        self.mlp = Block(self.emb_size, ratio=ratio)

    def forward(self, x: Tensor, /) -> Tensor:
        B, h, w, c = x.shape

        q, k, v = self.qkv(x)

        score = torch.einsum("btc, blc -> btl", [q, k])
        attn = torch.softmax(score * self.scale, dim=2)  # TODO check dim
        out = torch.einsum("btc, btl -> blc", [v, attn])

        out = rearrange(
            out,
            "(b n) (h w) s -> b h w (n s)",
            n=self.num_heads,
            s=self.head_size,
            h=h,
            w=w,
        )
        x = self.proj(out)
        x = self.mlp(x)

        return x


class PatchEncoder(PatchBase):
    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        ratio: float = 1,
        num_heads: int = 1,
        head_size: Optional[int] = None,
        num_layers: int = 1,
    ):
        super().__init__(
            num_channels, patch_size, ratio, num_heads, head_size, num_layers
        )

        # layers
        self.mix_color = Block(num_channels)
        self.to_patches = Rearrange(
            "b (h hp) (w wp) c -> b h w (c hp wp)",
            hp=patch_size,
            wp=patch_size,
        )
        self.pos_enc = PositionalEncoding(self.emb_size, ndim=2)

        self.intra_patch_mix = Block(self.emb_size, ratio=ratio)
        self.extra_patch_mix = nn.Sequential(
            *[
                TransformerBlock(self.emb_size, num_heads, head_size, ratio)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, /) -> Tensor:
        x = normalize(x)
        x = self.mix_color(x)
        x = self.to_patches(x)
        x = x + self.pos_enc(x)
        x = self.intra_patch_mix(x)
        x = self.extra_patch_mix(x)

        return x


class PatchDecoder(PatchBase):
    def __init__(
        self,
        num_channels: int,
        patch_size: int,
        ratio: float = 1,
        num_heads: int = 1,
        head_size: Optional[int] = None,
        num_layers: int = 1,
    ):
        super().__init__(
            num_channels, patch_size, ratio, num_heads, head_size, num_layers
        )

        # layers
        self.extra_patch_unmix = nn.Sequential(
            *[
                TransformerBlock(self.emb_size, num_heads, head_size, ratio)
                for _ in range(num_layers)
            ]
        )
        self.intra_patch_unmix = Block(self.emb_size, ratio=ratio)
        self.from_patches = Rearrange(
            "b h w (c hp wp) -> b (h hp) (w wp) c",
            hp=patch_size,
            wp=patch_size,
        )
        self.unmix_color = Block(num_channels)

    def forward(self, z: Tensor) -> Tensor:
        z = self.extra_patch_unmix(z)
        z = self.intra_patch_unmix(z)
        z = self.from_patches(z)
        z = self.unmix_color(z)
        z = denormalize(z)

        return z
