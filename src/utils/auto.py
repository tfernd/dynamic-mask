from __future__ import annotations

from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor

import pytorch_lightning as pl


# TODO add type hints for method
def auto_grad(method):
    @wraps(method)
    def wrapper(self: nn.Module, *args, **kwargs):
        with torch.set_grad_enabled(self.training):
            return method(self, *args, **kwargs)

    return wrapper


# TODO add type hints for method
def auto_device(method):
    @wraps(method)
    def wrapper(self: pl.LightningModule, *args, **kwargs):
        args = tuple(to_device(v, self.device) for v in args)
        kwargs = {k: to_device(v, self.device) for k, v in kwargs.items()}

        return method(self, *args, **kwargs)

    return wrapper


def to_device(x,  device: str | torch.device):
    if isinstance(x, Tensor):
        return x.to(device)

    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)

    return x
