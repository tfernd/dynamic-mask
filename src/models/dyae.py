from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from ..layers import PatchEncoder, PatchDecoder, MaskLatent
from ..utils import normalize, auto_grad


class DynamicAutoEncoder(pl.LightningModule):
    lr: float = 1e-4
    batch_size: int = 64

    train_dataset: Dataset[tuple[Tensor, ...]]
    validation_dataset: Dataset[tuple[Tensor, ...]]

    def __init__(
        self,
        *,
        num_channels: int,
        patch_size: int,
        ratio: float = 1,
        num_heads: int = 1,
        head_size: Optional[int] = None,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
    ) -> None:
        super().__init__()

        # parameters
        self.save_hyperparameters()

        self.emb_size = num_channels * patch_size**2
        self.name = f"DyAE_c{num_channels}_p{patch_size}_r{ratio}_n({num_heads}x{head_size})_N({encoder_layers}x{decoder_layers})"

        # layers
        self.patch_encoder = PatchEncoder(
            num_channels, patch_size, ratio, num_heads, head_size, encoder_layers
        )
        self.patch_decoder = PatchDecoder(
            num_channels, patch_size, ratio, num_heads, head_size, decoder_layers
        )
        self.mask_latent = MaskLatent(self.emb_size)

    @auto_grad
    def encode(
        self,
        x: Tensor,
        /,
        *,
        n: Optional[int] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = x.to(self.device)

        z = self.patch_encoder(x)

        z, mask = self.mask_latent.mask(z)
        z = self.mask_latent.crop(z, n=n)

        return z, mask

    @auto_grad
    def decode(self, z: Tensor, /) -> Tensor:
        z = z.to(self.device)

        z = self.mask_latent.expand(z)

        out = self.patch_decoder(z)

        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @auto_grad
    def loss(self, data: Tensor, out: Tensor, /) -> Tensor:
        data = data.to(self.device)
        out = out.to(self.device)

        return F.l1_loss(data.float(), out.float())

    @auto_grad
    def fft_loss(self, data: Tensor, out: Tensor, /) -> Tensor:
        data = data.to(self.device)
        out = out.to(self.device)

        fd = torch.fft.rfftn(normalize(data), dim=(1, 2))
        fo = torch.fft.rfftn(normalize(out), dim=(1, 2))

        loss = fd.sub(fo).abs().mean()

        return loss

    def training_step(
        self,
        batch: tuple[Tensor, ...],
        idx: int,
        /,
    ) -> Tensor:
        data, *_ = batch

        z, mask = self.encode(data)
        out = self.decode(z)

        loss = self.fft_loss(data, out)
        self.log("loss/training", loss.item())

        return loss

    # TODO this function is a mess...
    @torch.no_grad()
    def validation_step(
        self,
        batch: tuple[Tensor, ...],
        idx: int,
    ) -> Tensor:
        self.eval()

        data, *_ = batch

        ns = [self.emb_size]
        while 1 not in ns:
            ns.append(ns[-1] // 2)

        z, mask = self.encode(data)
        assert mask is None

        loss: list[Tensor] = []
        for n in ns:
            out = self.decode(z[..., :n])
            loss.append(self.loss(data, out))

        metric = {f"n={n}": l.item() for n, l in zip(ns, loss)}

        # TODO fix type hinting
        self.logger.experiment.add_scalars(  # type: ignore
            "loss/validation",
            metric,
            self.current_epoch,
        )

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
