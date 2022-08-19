from __future__ import annotations
from typing import Any

from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor


# TODO add type hints for method
def auto_grad(method):
    """Automatically determine whether to use autograd or not."""

    @wraps(method)
    def wrapper(self: nn.Module, *args, **kwargs):
        with torch.set_grad_enabled(self.training):
            return method(self, *args, **kwargs)

    return wrapper


# TODO add type hints for method
def auto_device(method):
    """Automatically send args and kwargs to the correct device."""

    @wraps(method)
    def wrapper(self: nn.Module, *args, **kwargs):
        param = next(self.parameters())
        device = param.device

        args = tuple(to_device(v, device) for v in args)
        kwargs = {k: to_device(v, device) for (k, v) in kwargs.items()}

        return method(self, *args, **kwargs)

    return wrapper


def to_device(
    x: Tensor | tuple[Tensor | Any, ...] | Any,
    device: str | torch.device,
):
    """Send input to the correct device."""

    if isinstance(x, Tensor):
        return x.to(device)

    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)

    return x
