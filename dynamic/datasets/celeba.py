from __future__ import annotations
from typing import Optional, Generator

from pathlib import Path
from PIL import Image

from multiprocessing.pool import ThreadPool

import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm

from ..utils import img2tensor

SIZE = 256

# TODO add multiprocessing
class CelebA(Dataset):
    def __init__(
        self,
        root: str | Path,
        /,
        *,
        size: int = SIZE,
    ) -> None:
        assert size <= SIZE
        self.size = size

        root = Path(root)

        self.paths = list(root.rglob("*.jpg"))
        assert len(self.paths) > 0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int, /) -> Tensor:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.size != SIZE:
            img = img.resize((self.size, self.size), resample=Image.BICUBIC)

        return img2tensor(img, channel_first=True)

    def dataloader(
        self,
        *,
        batch_size: int = 1,
        steps: int = 1,
        processes: Optional[int] = None,
    ) -> Generator[Tensor, None, None]:
        with ThreadPool(processes) as pool:
            for _ in range(steps):
                idx: list[int] = torch.randint(0, len(self), (batch_size,)).tolist()

                arrr = pool.map(self.__getitem__, idx)

                yield torch.stack(arrr, dim=0)

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}(size={self.size})"


class CelebACached(Dataset):
    data: Tensor
    mean: Tensor
    std: Tensor

    size: int

    def __init__(
        self,
        root: str | Path,
        /,
        *,
        size: int = SIZE,
        cache_path: Optional[str | Path] = None,
    ) -> None:
        assert size <= SIZE
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
            paths = list(root.rglob("*.jpg"))
            n = len(paths)
            assert len(paths) > 0

            # pre-allocate memory
            self.data = torch.empty((n, 3, SIZE, SIZE), dtype=torch.uint8)

            # mean and std helper tensors
            x, x2 = torch.zeros(2, 3).double()

            for i, path in enumerate(tqdm(paths, desc="Opening images")):
                img = Image.open(path).convert("RGB")

                d = img2tensor(img, channel_first=True)
                self.data[i] = d

                # statistics
                d = d.double()  # high precision

                x += d.mean(dim=(1, 2)).div(n)
                x2 += d.pow(2).mean(dim=(1, 2)).div(n)

                del d

            self.mean = x.float()

            N = self.data.size(0) * SIZE**2
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

    def __getitem__(self, idx: int | list[int] | Tensor | slice, /) -> Tensor:
        # add batch dimension
        if isinstance(idx, int):
            idx = [idx]

        data = self.data[idx]

        if self.size != SIZE:
            size = (self.size, self.size)
            data = F.interpolate(data.float(), size, mode="bicubic")
            data = data.clamp_(0, 255).byte()

        return data

    def __len__(self) -> int:
        return len(self.data)

    def dataloader(
        self,
        *,
        batch_size: int = 1,
        steps: int = 1,
    ) -> Generator[Tensor, None, None]:
        for _ in range(steps):
            idx = torch.randint(0, len(self), (batch_size,))

            yield self[idx]

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}(size={self.size})"
