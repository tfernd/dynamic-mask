from __future__ import annotations
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torchvision
from torchvision.models import VGG16_Weights


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(
        self,
        resize: bool = True,
    ):
        super().__init__()

        if resize:
            SIZE = 224

            self.resize = partial(
                F.interpolate,
                mode="bilinear",
                size=(SIZE, SIZE),
                align_corners=False,
            )
        else:
            self.resize = lambda x: x

        vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).eval()

        layers = [slice(0, 4), slice(4, 9), slice(9, 16), slice(16, 23)]
        self.vgg_blocks = nn.ModuleList([vgg.features[l] for l in layers])  # type: ignore

        for p in self.vgg_blocks.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]) * 255
        std = torch.tensor([0.229, 0.224, 0.225]) * 255

        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        feature_layers: list[int] = [0, 1, 2, 3],
    ) -> Tensor:
        self.eval()

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x = self.resize(x)
        y = self.resize(y)

        loss = torch.tensor(0.0, device=x.device)
        for i, block in enumerate(self.vgg_blocks):
            x = block(x)
            y = block(y)

            if i in feature_layers:
                loss += F.l1_loss(x, y)

        return loss
