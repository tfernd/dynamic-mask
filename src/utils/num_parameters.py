from __future__ import annotations

import torch.nn as nn
from torch.nn.parameter import Parameter

def num_parameters(module: nn.Module | Parameter) -> int:
    """The number of parameters in a module."""

    if isinstance(module, Parameter):
        return module.numel() if module.requires_grad else 0

    return sum(p.numel() for p in module.parameters() if p.requires_grad)
