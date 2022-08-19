from __future__ import annotations
from typing import Optional

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from .patch import PatchEncoder, PatchDecoder


class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder: PatchEncoder,
        decoder: PatchDecoder,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.log: list[float] = []

    def encode(self, x: Tensor, *args, **kwargs):
        return self.encoder(x, *args, **kwargs)

    def decode(self, z: Tensor):
        return self.decoder(z)

    def forward(self, x: Tensor, *args, **kwargs):
        z, *rest = self.encode(x, *args, **kwargs)

        return self.decode(z), rest

    def save(
        self,
        root: str | Path,
        /,
        step: int,
        loss: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Path:
        root = Path(root)

        checkpoint = root / "checkpoint" / f"step={step}_loss={loss:.8f}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        db = dict(
            model=self.state_dict(),
            log=self.log,
            enc_kwargs=self.encoder.kwargs,
            dec_kwargs=self.decoder.kwargs,
            optimizer=optimizer.state_dict() if optimizer is not None else None,
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            optimizer_cls=type(optimizer) if optimizer is not None else None,
            scheduler_cls=type(scheduler) if scheduler is not None else None,
        )
        torch.save(db, str(checkpoint))

        return checkpoint

    @classmethod
    def load(
        cls,
        path: str | Path,
        /,
    ) -> tuple[
        AutoEncoder,
        Optional[torch.optim.Optimizer],
        Optional[torch.optim.lr_scheduler._LRScheduler],
    ]:
        db = torch.load(str(path), map_location=torch.device("cpu"))

        encoder = PatchEncoder(**db["enc_kwargs"])
        decoder = PatchDecoder(**db["dec_kwargs"])
        model = cls(encoder, decoder)

        model.load_state_dict(db["model"])
        model.log = db["log"]

        # TODO load optimizer and scheduler classes
        optimizer = db["optimizer"]
        scheduler = db["scheduler"]

        return model, optimizer, scheduler
