from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from ..layers import PatchEncoder, PatchDecoder, MaskLatent, Block, PatchMixer
from ..utils import normalize, auto_grad


class DynamicAutoEncoderSuperResolution(pl.LightningModule):
    lr: float = 1e-4
    batch_size: int = 64

    train_dataset: Dataset[tuple[Tensor, Tensor]]
    validation_dataset: Dataset[tuple[Tensor, Tensor]]

    def __init__(
        self,
        shape: tuple[int, int, int],
        patch: int,
        scale: int,
        channel_ratio: float = 1,
        spatial_ratio: float = 1,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        upsample_layers: int = 1,
    ) -> None:
        super().__init__()

        # parameters
        self.save_hyperparameters()

        H, W, C = shape
        P = patch
        S = scale

        shape_small = (H // S, W // S, C)

        self.emb_size_small = e = C * (P//S)**2
        self.emb_size = E = C * P**2
        self.spatial_size = H // P * W // P

        ratios = (channel_ratio, spatial_ratio)
        layers = (encoder_layers, decoder_layers)
        self.name = f"DyAE({shape})_p{patch}_r{ratios}_n{layers}"

        # layers
        self.patch_encoder_small = PatchEncoder(
            shape_small, patch // scale, channel_ratio, spatial_ratio, encoder_layers
        )
        self.patch_encoder = PatchEncoder(
            shape, patch, channel_ratio, spatial_ratio, encoder_layers
        )
        self.patch_decoder = PatchDecoder(
            shape, patch, channel_ratio, spatial_ratio, decoder_layers
        )
        self.mask_latent = MaskLatent(E)

        self.upsample = nn.Sequential(
            Block(E //S**2, E),
            *[
                PatchMixer(shape, patch, channel_ratio, spatial_ratio)
                for _ in range(upsample_layers)
            ],
        )

    @auto_grad
    def encode(
        self,
        small: Tensor,
        big: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        small = small.to(self.device)

        if big is not None:
            big = big.to(self.device)

            zb = self.patch_encoder(big)
            zb, mask = self.mask_latent.mask(zb)
        else:
            zb, mask = None, None

        zs = self.patch_encoder_small(small)
        zsb = self.upsample(zs)

        return zsb, zb, mask

    @auto_grad
    def decode(
        self,
        zsb: Tensor,
        zb: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        zsb = zsb.to(self.device)

        xsb = self.patch_decoder(zsb)

        if zb is not None:
            zb = zb.to(self.device)
        xb = self.patch_decoder(zb) if zb is not None else None

        return xsb, xb

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @auto_grad
    def loss(self, data: Tensor, out: Tensor) -> Tensor:
        data = data.to(self.device)
        out = out.to(self.device)

        return F.l1_loss(data.float(), out.float())

    @auto_grad
    def fft_loss(self, data: Tensor, out: Tensor) -> Tensor:
        data = data.to(self.device)
        out = out.to(self.device)

        fd = torch.fft.rfftn(normalize(data), dim=(1, 2))
        fo = torch.fft.rfftn(normalize(out), dim=(1, 2))

        loss = fd.sub(fo).abs().mean()

        return loss

    def training_step(
        self,
        batch: tuple[Tensor, Tensor],
        idx: int,
        name: str = "training",
    ) -> Tensor:
        small, big = batch

        zsb, zb, mask = self.encode(small, big)
        xsb, xb = self.decode(zsb, zb)

        if xb is not None:
            loss_b = self.fft_loss(big, xb)
            self.log(f"loss/{name}/big", loss_b.item())
        else:
            loss_b = 0

        loss_sb = self.loss(big, xsb)
        self.log(f"loss/{name}/sr", loss_sb.item())

        return loss_b + loss_sb

    # TODO this function is a mess...
    @torch.no_grad()
    def validation_step(
        self,
        batch: tuple[Tensor, Tensor],
        idx: int,
    ) -> Tensor:
        self.eval()

        return self.training_step(batch, idx, "validation")

    def add_dataset(
        self,
        train: Dataset[tuple[Tensor, Tensor]],
        validation: Dataset[tuple[Tensor, Tensor]],
    ) -> None:
        self.train_dataset = train
        self.validation_dataset = validation

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_dataset, batch_size=self.batch_size)
