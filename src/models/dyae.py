from __future__ import annotations
from typing import Optional

import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from ..layers import PatchEncoder, PatchDecoder, MaskLatent
from ..utils import auto_grad, auto_device, normalize, denormalize


class DynamicAutoEncoder(pl.LightningModule):
    lr: float = 1e-4
    batch_size: int = 64

    dataset: Dataset[tuple[Tensor, ...]]

    def __init__(
        self,
        *,
        num_channels: int,
        patch_size: int,
        ratio: float | tuple[float, float] = 1,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
    ) -> None:
        super().__init__()

        # parameters
        self.save_hyperparameters()

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.ratio = ratio
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.emb_size = emb_size = num_channels * patch_size**2

        self.name = f"DyAE_c{num_channels}_p{patch_size}_r{ratio}_n(e{encoder_layers}-d{decoder_layers})"

        # layers
        args = (num_channels, patch_size, ratio)
        self.patch_encoder = PatchEncoder(*args, encoder_layers)
        self.patch_decoder = PatchDecoder(*args, decoder_layers)
        self.mask_latent = MaskLatent(emb_size)

    @auto_grad
    @auto_device
    def encode(
        self,
        x: Tensor,
        *,
        n: int | float = 1.0,
    ) -> tuple[Tensor, Optional[Tensor]]:
        xn = normalize(x)
        z = self.patch_encoder(xn)

        z, mask = self.mask_latent.mask(z)
        z = self.mask_latent.crop(z, n)

        return z, mask

    @auto_grad
    @auto_device
    def decode(self, z: Tensor) -> Tensor:
        z = self.mask_latent.expand(z)

        out = self.patch_decoder(z)
        out = denormalize(out)

        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @auto_grad
    @auto_device
    def loss(self, data: Tensor, out: Tensor) -> Tensor:
        return F.l1_loss(data.float(), out.float())

    @auto_device
    @torch.no_grad()
    def psnr(self, data: Tensor, out: Tensor) -> Tensor:
        return self.psnr_from_loss(self.loss(data, out))

    @auto_device
    @torch.no_grad()
    def psnr_from_loss(self, loss: Tensor) -> Tensor:
        return 10 * torch.log10(255**2 / loss)

    @auto_device
    def training_step(
        self,
        batch: tuple[Tensor, ...],
        idx: int,
    ) -> Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for data in batch:
            z, mask = self.encode(data)
            out = self.decode(z)

            loss += self.loss(data, out)
        loss /= len(batch)

        self.log("training/loss", loss)

        psnr = self.psnr_from_loss(loss)
        self.log("training/psnr", psnr)

        return loss

    def add_dataset(
        self,
        dataset: Dataset[tuple[Tensor, ...]],
    ) -> None:
        self.dataset = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
