from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch import Tensor

import pytorch_lightning as pl

from ..layers import (
    PatchEncoder,
    PatchDecoder,
    MaskLatent,
    Normalizer,
    VGGPerceptualLoss,
)
from ..utils import auto_grad, auto_device, normalize, denormalize, num_parameters


class DynamicAutoEncoder(pl.LightningModule):
    lr: float = 1e-4

    idx: Tensor

    def __init__(
        self,
        *,
        patch_size: int,
        kernel_size: int = 1,
        num_channels: int = 3,
        ratio: float = 1,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        beta: float = 1,  # TODO make it optional to add VGG-loss
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
        self.beta = beta

        self.emb_size = emb_size = num_channels * patch_size**2


        # layers
        args = (patch_size, kernel_size, num_channels, ratio)

        self.patch_encoder = PatchEncoder(*args, encoder_layers)
        self.patch_decoder = PatchDecoder(*args, decoder_layers)
        self.mask_latent = MaskLatent(emb_size)
        self.normalizer = Normalizer(emb_size)

        # TODO make into a layer
        self.rate = Parameter(torch.tensor(3.0))
        idx = torch.linspace(0, -1, emb_size).view(1, -1, 1, 1)
        self.register_buffer("idx", idx)

        params = num_parameters(self)
        self.name = f"DyAE(p{patch_size}_k{kernel_size}_c{num_channels}_r{ratio}_e{encoder_layers}_d{decoder_layers})-{params:,}"

        self.vgg_loss = VGGPerceptualLoss()

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
        z = self.normalizer(z)

        z, mask = self.mask_latent(z)
        z = self.mask_latent.crop(z, n)

        return z, mask

    @auto_grad
    @auto_device
    def decode(self, z: Tensor) -> Tensor:
        z = self.mask_latent.expand(z)

        scale = self.idx.mul(self.rate.clamp(1, 8)).exp()
        z = z * scale

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

        vgg_loss = self.vgg_loss(out, data)
        self.log("training/vgg_loss", vgg_loss)

        psnr = self.psnr_from_loss(loss.detach())
        self.log("training/psnr", psnr, prog_bar=True)

        return loss + vgg_loss * self.beta
