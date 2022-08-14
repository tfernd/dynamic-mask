from __future__ import annotations

from functools import wraps

import torch
import torch.nn as nn

# TODO add type hints for method
def auto_grad(method):
    @wraps(method)
    def wrapper(self: nn.Module, *args, **kwargs):
        with torch.set_grad_enabled(self.training):
            return method(self, *args, **kwargs)

    return wrapper
