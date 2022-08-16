from __future__ import annotations

from PIL.Image import Image
import numpy as np

import torch
from torch import Tensor


def img2tensor(
    img: Image,
    *,
    channel_first: bool = False,
) -> Tensor:
    """Convert a PIL image to a pytorch Tensor."""

    data = np.asarray(img)
    data = torch.from_numpy(data)
    assert data.dtype == torch.uint8

    # black and white images are converted to 3 channels
    if data.ndim == 2:
        data = data.unsqueeze(2)

    if channel_first:
        data = data.permute(2, 0, 1)

    return data
