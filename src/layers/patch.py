#%%
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from einops.layers.torch import Rearrange

# from .block import Block
# from ..utils import normalize, denormalize, auto_grad


# def PatchMixer(
#     shape: tuple[int, int, int],
#     patch: int,
#     channel_ratio: float = 1,
#     spatial_ratio: float = 1,
# ):
#     H, W, C = shape
#     P = patch
#     E = C * P**2

#     h, w = H // P, W // P
#     hw = h * w

#     return nn.Sequential(
#         # transpose channels/patches
#         Rearrange("b h w e -> b e (h w)", h=h, w=w, e=E),
#         # extra-patch mixing
#         Block(hw, ratio=spatial_ratio),
#         # transpose patches/channels
#         Rearrange("b e (h w) -> b h w e", h=h, w=w, e=E),
#         # intra-patch mixing
#         Block(E, ratio=channel_ratio),
#     )


class PatchBase(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int, int],
        patch: int,
        channel_ratio: float = 1,
        spatial_ratio: float = 1,
        num_layers: int = 1,
    ):
        super().__init__()

        # parameters
        self.shape = H, W, C = shape
        self.patch = P = patch
        self.channel_ratio = channel_ratio
        self.spatial_ratio = spatial_ratio
        self.num_layers = num_layers

        self.emb_size = E = C * patch**2


# class PatchEncoder(PatchBase):
#     def __init__(
#         self,
#         shape: tuple[int, int, int],
#         patch: int,
#         channel_ratio: float = 1,
#         spatial_ratio: float = 1,
#         num_layers: int = 1,
#     ):
#         super().__init__(shape, patch, channel_ratio, spatial_ratio, num_layers)

#         H, W, C = shape
#         P = patch

#         # layers
#         self.mix_color = Block(C)
#         self.to_patches = Rearrange("b (h hp) (w wp) c -> b h w (c hp wp)", hp=P, wp=P)
#         self.intra_patch_mix = Block(self.emb_size, ratio=channel_ratio)
#         self.extra_patch_mix = nn.Sequential(
#             *[
#                 PatchMixer(shape, patch, channel_ratio, spatial_ratio)
#                 for _ in range(num_layers)
#             ]
#         )

#     @auto_grad
#     def forward(self, x: Tensor) -> Tensor:
#         with torch.set_grad_enabled(self.training):
#             xn = normalize(x)
#             xn = self.mix_color(xn)
#             z = self.to_patches(xn)
#             z = self.intra_patch_mix(z)
#             z = self.extra_patch_mix(z)

#         return z


# class PatchDecoder(PatchBase):
#     def __init__(
#         self,
#         shape: tuple[int, int, int],
#         patch: int,
#         channel_ratio: float = 1,
#         spatial_ratio: float = 1,
#         num_layers: int = 1,
#     ):
#         super().__init__(shape, patch, channel_ratio, spatial_ratio, num_layers)

#         H, W, C = shape
#         P = patch

#         # layers
#         self.extra_patch_unmix = nn.Sequential(
#             *[
#                 PatchMixer(shape, patch, channel_ratio, spatial_ratio)
#                 for _ in range(num_layers)
#             ]
#         )
#         self.intra_patch_unmix = Block(self.emb_size, ratio=channel_ratio)
#         self.from_patches = Rearrange(
#             "b h w (c hp wp) -> b (h hp) (w wp) c", hp=P, wp=P
#         )
#         self.unmix_color = Block(C)

#     @auto_grad
#     def forward(self, z: Tensor) -> Tensor:
#         with torch.set_grad_enabled(self.training):
#             z = self.extra_patch_unmix(z)
#             z = self.intra_patch_unmix(z)
#             out = self.from_patches(z)
#             out = self.unmix_color(out)
#             out = denormalize(out)

#         return out
