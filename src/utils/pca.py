from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor


def PCA(
    data: Tensor,
    /,
    features: Optional[int] = 2,
) -> Tensor:
    """Principal Component Analysis (PCA)."""

    assert data.ndim == 2, "data must be 2D"
    if data.dtype == torch.uint8:
        data = data.float()

    mean = data.mean(dim=0, keepdim=True)
    X = data - mean

    U, S, V = torch.svd(X)
    Z = X @ (V[:, :features] if features else V)

    return Z
