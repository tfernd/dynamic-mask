from __future__ import annotations
from re import T
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

from einops.layers.torch import Rearrange

import pytorch_lightning as pl

from tqdm.autonotebook import tqdm, trange

from ...layers import PatchEncoder, PatchDecoder, MaskLatent, Block
from ...utils import auto_grad, auto_device


class Diffusion(pl.LightningModule):
    lr: float = 1e-4
    batch_size: int = 64

    dataset: Dataset[tuple[Tensor, ...]]

    def __init__(
        self,
        *,
        shape: tuple[int, int, int],
        patch_size: int,
        channel_ratio: float = 1,
        spatial_ratio: float = 1,
        num_layers: int = 1,
        mean: tuple[float, float, float] = (255 / 2, 255 / 2, 255 / 2),
        std: tuple[float, float, float] = (255 / 2, 255 / 2, 255 / 2),
    ):
        super().__init__()

        # parameters
        self.save_hyperparameters()

        self.shape = shape

        H, W, C = shape
        self.name = f"Diffusion_({H}x{W}x{C}-{patch_size})_r({channel_ratio}-{spatial_ratio})_n({num_layers})"

        self.mean = Parameter(
            torch.tensor(mean).float().view(1, 1, 1, 3), requires_grad=False
        )
        self.std = Parameter(
            torch.tensor(std).float().view(1, 1, 1, 3), requires_grad=False
        )

        # layers
        self.patch_encoder = PatchEncoder(
            shape, patch_size, channel_ratio, spatial_ratio, num_layers=0
        )
        self.patch_decoder = PatchDecoder(
            shape, patch_size, channel_ratio, spatial_ratio, num_layers
        )

        self.emb_size = self.patch_encoder.emb_size

        self.std_emb = Block(1, self.emb_size)

    @auto_grad
    @auto_device
    def forward(self, noisy: Tensor, /, std: Tensor) -> Tensor:
        std = std.view(-1, 1, 1, 1)

        z = self.patch_encoder(noisy)
        z = z + self.std_emb(std)
        out = self.patch_decoder(z)
        out = out.mul(std)  # scale

        return out

    @auto_device
    def training_step(self, batch: tuple[Tensor, ...], idx: int, /) -> Tensor:
        x, *_ = batch

        B, H, W, C = x.shape

        # zero-mean unit-variance normalization
        xn = x.float().sub(self.mean).div(self.std)

        std = torch.empty(B, 1, 1, 1, device=self.device).uniform_(0, 1)
        eps = torch.randn_like(xn)

        noisy = xn + eps.mul(std)

        eps_hat = self.forward(noisy, std)

        loss = eps.sub(eps_hat).abs().mean()

        self.log("loss", loss)

        return loss

    @torch.no_grad()
    def sample(self, size: int = 1, steps: int = 64) -> Tensor:

        t = torch.linspace(0, 1, steps + 1, device=self.device)
        ᾱ = torch.cos(t * torch.pi / 2).pow(2)
        β = 1 - ᾱ[1:] / ᾱ[:-1]
        std = torch.sqrt(β).view(steps, 1, 1, 1, 1)
        std = std.repeat(1, size, 1, 1, 1)
        std.clamp_(1e-4, 1 - 1e-4)

        x = torch.randn(size, *self.shape, device=self.device)
        for i in trange(steps):
            eps = self.forward(x, std[i])

            x = x / (1 - std[i]) + std[i] / (std[i] - 1) * eps

        return x.min(), x.max()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def add_dataset(
        self,
        dataset: Dataset[tuple[Tensor, ...]],
    ) -> None:
        self.dataset = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


# def scheaduler(
#     steps: int,
#     *,
#     tol: float = 1e-4,
# ):
#     t = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
#     ᾱ = torch.cos(t * torch.pi / 2).pow(2)

#     β = 1 - ᾱ[1:] / ᾱ[:-1]
#     β.clamp_(tol, 1 - tol)

#     # recomputing ᾱ
#     α = 1 - β
#     ᾱ = α.cumprod(0)

#     β, α, ᾱ = β.float(), α.float(), ᾱ.float()

#     return β, α, ᾱ


# scheaduler(100)
