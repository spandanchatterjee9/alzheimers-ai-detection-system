"""
2D DenseNet-121 Architecture for Alzheimer's Dementia Stage Classification.
Adapted for 1-channel Grayscale MRI Slices (224x224).
"""

import torch
import torch.nn as nn
from torchvision import models


class DenseNet1212D(nn.Module):
    """
    2D DenseNet-121 adapted for 1-channel Grayscale MRI slices.
    Features dense connectivity (feature reuse) across all convolutional blocks.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        dropout: float = 0.3,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.net = models.densenet121(weights=weights)

        # Adapt first conv layer from 3 RGB channels to in_channels (1 for Grayscale)
        orig_conv = self.net.features.conv0
        if in_channels != 3:
            self.net.features.conv0 = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    self.net.features.conv0.weight = nn.Parameter(
                        orig_conv.weight.sum(dim=1, keepdim=True)
                    )

        # Classifier head with Dropout
        in_features = self.net.classifier.in_features
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
