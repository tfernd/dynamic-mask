from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from tqdm.autonotebook import tqdm, trange

from ..layers import PatchEncoder, PatchDecoder, MaskLatent, Block, PositionalEncoding
from ..utils import auto_grad, auto_device, normalize, denormalize


class Diffusion(pl.LightningModule):
    lr: float = 1e-4
    batch_size: int = 64

    dataset: Dataset[tuple[Tensor, ...]]

    def __init__(
        self,
        *,
        features: int,
        encoder: PatchEncoder,
        decoder: PatchDecoder,
        ratio: float | tuple[float, float] = 1,
        num_layers: int = 1,
    ):
        super().__init__()

        # parameters
        self.save_hyperparameters()

        self.name = f"Diffusion_f{features}_r{ratio}_n{num_layers}"

        self.features = features
        self.ratio = ratio
        self.num_layers = num_layers

        # TODO create asserts!
        self.emb_size = encoder.emb_size
        self.patch_size = encoder.patch_size

        # layers
        encoder.eval()
        decoder.eval()

        self.encode = encoder
        self.decode = decoder

        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.decode.parameters():
            param.requires_grad = False

        # layers
        self.mix = nn.Sequential(
            *[Block(features, ratio=ratio) for _ in range(num_layers)]
        )
        self.mask_latent = MaskLatent(self.emb_size)

        self.std_emb = Block(1, features)
        self.pos_enc = PositionalEncoding(features, ndim=2)

    @auto_grad
    @auto_device
    def forward(self, noisy: Tensor,  std: Tensor) -> Tensor:
        return self.mix(noisy + self.std_emb(std) + self.pos_enc(noisy))

    @auto_device
    def training_step(self, batch: tuple[Tensor, ...], idx: int) -> Tensor:
        x, *_ = batch

        B, H, W, C = x.shape
        P = self.patch_size
        E = self.features

        # encode into latent space
        with torch.no_grad():
            xn = normalize(x)
            z = self.encode(xn)
            z = self.mask_latent.crop(z, self.features)

        std = torch.empty(B, 1, 1, 1, device=self.device).uniform_(0, 1)
        eps = torch.randn(B, H // P, W // P, E, device=self.device)

        noisy = z * (1 - std) + eps * std

        eps_hat = self.forward(noisy, std)

        loss = eps.sub(eps_hat).abs().mean()
        self.log("loss", loss)

        return loss

    @torch.no_grad()
    def sample(self, batch_size: int = 1, steps: int = 64, size: int = 256) -> Tensor:
        self.eval()

        t = torch.linspace(0, 1, steps + 1, device=self.device)

        ᾱ = (
            torch.cos(t * torch.pi / 2)
            .pow(2)
            .view(-1, 1, 1, 1, 1)
            .repeat(1, batch_size, 1, 1, 1)
        )

        β = 1 - ᾱ[1:] / ᾱ[:-1]
        β.clamp_(1e-4, 1 - 1e-4)

        # recompute
        α = 1 - β
        ᾱ = α.cumprod(dim=0)

        shape = (batch_size, size // self.patch_size, size // self.patch_size)
        x = torch.randn(*shape, self.features, device=self.device)
        for i in trange(steps):
            z = torch.randn_like(x) if i != steps - 1 else torch.zeros_like(x)

            eps_hat = self.forward(x, β[-i])

            x = x / (1 - β[-i]) - eps_hat

        x = self.mask_latent.expand(x)
        out = self.decode(x)
        out = denormalize(out)

        return out

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def add_dataset(
        self,
        dataset: Dataset[tuple[Tensor, ...]],
    ) -> None:
        self.dataset = dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
