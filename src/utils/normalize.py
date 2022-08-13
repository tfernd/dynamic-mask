from __future__ import annotations

import torch
from torch import Tensor


def normalize(x: Tensor) -> Tensor:
    """Normalize a tensor from [0, 255] -> [-1, 1]."""

    if x.requires_grad:
        return x.mul(2 / 255).sub(1)

    assert x.dtype == torch.uint8

    return x.mul(2 / 255).sub_(1)  # TODO how much faster?


def denormalize(x: Tensor) -> Tensor:
    """Denormalize a tensor from [-1, 1] -> [0, 255]."""

    if x.requires_grad:
        return x.add(1).mul(255 / 2)

    return x.add(1).mul_(255 / 2).clamp_(0, 255).byte()  # TODO how much faster?
