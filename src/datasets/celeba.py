from __future__ import annotations
from typing import Optional, Literal

from pathlib import Path

from PIL import Image
import math
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm


Size = Literal[16, 32, 64, 128, 256]


class CelebA(Dataset):
    sizes: tuple[Size, ...]

    def __init__(
        self,
        root: str | Path,
        sizes: Size | tuple[Size, ...] = 256,
        cache_path: Optional[str | Path] = None,
    ) -> None:
        sizes = (sizes,) if not isinstance(sizes, tuple) else sizes
        self.sizes = sizes

        root = Path(root)
        assert root.exists()

        if cache_path is None:
            CACHE = root / "cache"
        else:
            CACHE = Path(cache_path)

        self.data: dict[Size, Tensor] = {}
        self.mean: dict[Size, Tensor] = {}
        self.std: dict[Size, Tensor] = {}

        for size in tqdm(sizes, desc="Parsing sizes"):
            cached_file = CACHE / f"{size}.pt"

            if cached_file.exists():
                obj = torch.load(cached_file)

                self.data[size] = obj["data"]
                self.mean[size] = obj["mean"]
                self.std[size] = obj["std"]

                continue

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

                if img.size != (size, size):
                    img = img.resize((size, size), resample=Image.Resampling.BICUBIC)  # type: ignore

                d = np.asarray(img)
                d = torch.from_numpy(d).byte()

                data_arr.append(d)

                # statistics
                d = d.double()  # high precision
                x += d.mean(dim=(0, 1)).div(n)
                x2 += d.pow(2).mean(dim=(0, 1)).div(n)

            self.data[size] = torch.stack(data_arr)
            self.mean[size] = x.float()

            B, H, W, C = self.data[size].shape
            N = B * H * W
            gamma = math.sqrt(N / (N - 1))  # correction factor for unbiased variance

            self.std[size] = torch.sqrt(x2 - x.pow(2)).mul(gamma).float()

            CACHE.mkdir(exist_ok=True, parents=True)
            obj = {
                "data": self.data[size],
                "mean": self.mean[size],
                "std": self.std[size],
            }
            torch.save(obj, cached_file)

        batch_sizes = [len(self.data[size]) for size in sizes]
        assert all(batch_sizes[0] == b for b in batch_sizes)

    def __getitem__(self, idx: int | Tensor | slice) -> tuple[Tensor, ...]:
        return tuple(d[idx] for d in self.data.values())

    def __len__(self) -> int:
        return len(self.data[self.sizes[0]])

    @property
    def shape(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(tuple(d.shape) for d in self.data.values())

    def __repr__(self) -> str:
        name = self.__class__.__qualname__

        return f"{name}(sizes={self.sizes})"


if __name__ == "__main__":
    HERE = Path(__file__).parent
    MAIN = HERE.parent.parent

    # pre-cache
    ds = CelebA(MAIN / "datasets" / "CelebA", sizes=(16, 32, 64, 128, 256))
