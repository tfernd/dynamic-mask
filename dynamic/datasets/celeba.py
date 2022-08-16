from __future__ import annotations
from typing import Optional

from pathlib import Path
from PIL import Image

import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from einops import rearrange

from tqdm.autonotebook import tqdm

# TODO use img2tensor from utils


class CelebA(Dataset):
    data: Tensor
    mean: Tensor
    std: Tensor

    size: int

    def __init__(
        self,
        root: str | Path,
        *,
        size: int = 256,
        cache_path: Optional[str | Path] = None,
    ) -> None:
        assert size <= 256
        self.size = size

        root = Path(root)
        assert root.exists()
        CACHE = Path(cache_path or root)
        cached_file = CACHE / f"cache.pt"

        if cached_file.exists():
            obj = torch.load(cached_file)

            self.data = obj["data"]
            self.mean = obj["mean"]
            self.std = obj["std"]
        else:
            paths = list(
                tqdm(
                    root.rglob("*.jpg"),
                    desc="Getting all files",
                    leave=False,
                )
            )
            n = len(paths)
            assert len(paths) > 0

            data_arr: list[Tensor] = []
            x, x2 = torch.zeros(2, 3).double()

            for path in tqdm(paths, desc="Opening images"):
                img = Image.open(path).convert("RGB")

                d = np.asarray(img)
                d = torch.from_numpy(d).byte()
                d = rearrange(d, "h w c -> c h w")

                data_arr.append(d)

                # statistics
                d = d.double()  # high precision
                x += d.mean(dim=(1, 2)).div(n)
                x2 += d.pow(2).mean(dim=(1, 2)).div(n)

            self.data = torch.stack(data_arr)
            self.mean = x.float()

            B, C, H, W = self.data.shape
            N = B * H * W
            # correction factor for unbiased variance
            gamma = math.sqrt(N / (N - 1))

            self.std = torch.sqrt(x2 - x.pow(2)).mul(gamma).float()

            CACHE.mkdir(exist_ok=True, parents=True)
            obj = {
                "data": self.data,
                "mean": self.mean,
                "std": self.std,
            }
            torch.save(obj, cached_file)

    def __getitem__(self, idx: int | list[int] | Tensor | slice) -> Tensor:
        # add batch dimension
        if isinstance(idx, int):
            idx = [
                idx,
            ]

        data = self.data[idx]

        if self.size != 256:
            size = (self.size, self.size)
            data = F.interpolate(data.float(), size, mode="bicubic")
            data = data.clamp_(0, 255).byte()

        return data

    def __len__(self) -> int:
        return len(self.data)

    # TODO rename?
    def batch(
        self,
        *,
        batch_size: int = 1,
        steps: int = 1,
        sizes: Optional[tuple[int, int, int]] = None,
    ):
        for _ in range(steps):
            idx = torch.randint(0, len(self), (batch_size,))

            prev_size = self.size
            if sizes is not None:
                arr = [i for i in range(sizes[0], sizes[1] + 1) if i % sizes[2] == 0]
                self.size = np.random.choice(arr)

            data = self[idx]

            self.size = prev_size

            yield data

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}(size={self.size})"


if __name__ == "__main__":
    HERE = Path(__file__).parent
    MAIN = HERE.parent.parent

    # pre-cache
    self = CelebA(MAIN / "datasets" / "CelebA")
