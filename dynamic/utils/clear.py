from __future__ import annotations

import gc
import torch


def clear_cuda():
    torch.cuda.empty_cache()
    gc.collect()
