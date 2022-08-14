from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from ..layers import PatchEncoder, PatchDecoder, MaskLatent
from ..utils import  auto_grad


class DynamicAutoEncoder(pl.LightningModule):
    lr: float = 1e-4
    batch_size: int = 64

    train_dataset: Dataset[tuple[Tensor, ...]]
    validation_dataset: Dataset[tuple[Tensor, ...]]

    def __init__(
        self,
        *,
        shape: tuple[int, int, int],
        patch_size: int,
        channel_ratio: float = 1,
        spatial_ratio: float = 1,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
    ) -> None:
        super().__init__()

        # parameters
        self.save_hyperparameters()

        H, W, C = shape
        self.name = f"DyAE_s({H}x{W}x{C}-{patch_size})_r({channel_ratio}-{spatial_ratio})_n({encoder_layers}-{decoder_layers})"

        # layers
        self.patch_encoder = PatchEncoder(
            shape, patch_size, channel_ratio, spatial_ratio, encoder_layers
        )
        self.patch_decoder = PatchDecoder(
            shape, patch_size, channel_ratio, spatial_ratio, decoder_layers
        )

        self.emb_size = self.patch_encoder.emb_size
        self.patch_emb_size = self.patch_encoder.patch_emb_size

        self.mask_latent = MaskLatent(self.emb_size)

    @auto_grad
    def encode(
        self,
        x: Tensor,
        *,
        n: int | float = 1.0,
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = x.to(self.device)

        z = self.patch_encoder(x)

        z, mask = self.mask_latent.mask(z)
        z = self.mask_latent.crop(z, n)

        return z, mask

    @auto_grad
    def decode(self, z: Tensor) -> Tensor:
        z = z.to(self.device)

        z = self.mask_latent.expand(z)

        out = self.patch_decoder(z)

        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @auto_grad
    def error(self, data: Tensor, out: Tensor) -> Tensor:
        data = data.to(self.device)
        out = out.to(self.device)

        return out.float().sub(data.float()).abs()

    @auto_grad
    def loss(self, data: Tensor, out: Tensor) -> Tensor:
        """L1-loss between input and output in pixel space."""

        return self.error(data, out).mean()

    def training_step(
        self,
        batch: tuple[Tensor, ...],
        idx: int,
    ) -> Tensor:
        data, *_ = batch

        z, mask = self.encode(data)
        out = self.decode(z)

        loss = self.loss(data, out)
        self.log("training/loss", loss.item())

        return loss

    # TODO this function is a mess...
    @torch.no_grad()
    def validation_step(
        self,
        batch: tuple[Tensor, ...],
        idx: int,
    ) -> Tensor:
        self.eval()  # TODO is this necessary?

        data, *_ = batch

        ns = [self.emb_size]
        while 1 not in ns:
            ns.append(ns[-1] // 2)

        z, mask = self.encode(data)
        assert mask is None

        loss: list[Tensor] = []
        std: list[Tensor] = []
        for n in ns:
            out = self.decode(z[..., :n])
            error = self.error(data, out)

            loss.append(error.mean())
            std.append(error.std())

            self.log(f'validation/loss/{n}', loss[-1].item())
            self.log(f'validation/std/{n}', std[-1].item())

        return loss[0]

    def add_dataset(
        self,
        train: Dataset[tuple[Tensor, ...]],
        validation: Dataset[tuple[Tensor, ...]],
    ) -> None:
        self.train_dataset = train
        self.validation_dataset = validation

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)
