from __future__ import annotations

import torch
from torch import Tensor


def normalize(x: Tensor) -> Tensor:
    """Normalize a tensor from [0, 255] -> [-1, 1]."""

    if x.requires_grad:
        return x.mul(2 / 255).sub(1).clamp(-1, 1)

    assert x.dtype == torch.uint8

    # TODO check speed!
    return x.mul(2 / 255).sub_(1).clamp_(-1, 1)


def denormalize(x: Tensor) -> Tensor:
    """Denormalize a tensor from [-1, 1] -> [0, 255], and retrain the gradients."""

    # retain gradients but discretize
    if x.requires_grad:
        x = x.add(1).mul(255 / 2).clamp(0, 255)
        xd = x.byte()

        return x + (xd - x).detach()

    # TODO check speed!
    return x.add(1).mul_(255 / 2).clamp_(0, 255).byte()
