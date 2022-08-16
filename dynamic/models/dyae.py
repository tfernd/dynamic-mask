from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from ..layers import PatchEncoder, PatchDecoder, MaskLatent
from ..utils import auto_grad, auto_device, normalize, denormalize


class DynamicAutoEncoder(pl.LightningModule):
    lr: float = 1e-4

    def __init__(
        self,
        *,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
    ) -> None:
        super().__init__()

        # parameters
        self.save_hyperparameters()

        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.ratio = ratio
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.emb_size = emb_size = num_channels * patch_size**2

        self.name = f"DyAE_p{patch_size}_k{kernel_size}_c{num_channels}_r{ratio}_e{encoder_layers}_d{decoder_layers}"

        # layers
        args = (patch_size, kernel_size, num_channels, ratio)

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
        # TODO add normalizer layer

        z, mask = self.mask_latent(z)
        z = self.mask_latent.crop(z, n)

        return z, mask

    @auto_grad
    @auto_device
    def decode(self, z: Tensor) -> Tensor:
        z = self.mask_latent.expand(z)
        # TODO add exponential scale layer

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
        # TODO check if this is correct
        return 10 * torch.log10(255**2 / loss)

    @auto_device
    def training_step(
        self,
        data: Tensor,
        idx: int,
    ) -> Tensor:
        z, mask = self.encode(data)
        out = self.decode(z)

        loss = self.loss(data, out)
        self.log("training/loss", loss)

        psnr = self.psnr_from_loss(loss.detach())
        self.log("training/psnr", psnr)

        return loss
