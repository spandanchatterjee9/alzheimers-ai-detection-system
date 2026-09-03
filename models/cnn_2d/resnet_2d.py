"""
Secondary 2D Deep Learning Architecture: 2D ResNet-18 for Alzheimer's-Related Dementia Stage Classification.

Input: 2D MRI Slices (224 x 224 x 1 or 224 x 224 x 3)
Output: 4-Class Softmax Probability Distribution across CDR 0.0, 0.5, 1.0, 2.0.

Provides a lightweight, alternative 2D baseline for comparison against EfficientNet-B0.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18, ResNet18_Weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

logger = logging.getLogger("models.cnn_2d.resnet_2d")


class ResNet182D(nn.Module if HAS_TORCH else object):
    """
    2D ResNet-18 Model for CDR Stage Classification on 224x224 MRI Slices.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        pretrained: bool = True
    ) -> None:
        """
        Args:
            num_classes: Number of target CDR categories (default 4: CDR 0, 0.5, 1, 2).
            in_channels: Number of input channels (1 for grayscale MRI slice, 3 for RGB).
            pretrained: Use ImageNet pretrained weights.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to instantiate ResNet182D.")

        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)

        # Adapt first conv layer if input is grayscale
        if in_channels == 1:
            old_conv = base_model.conv1
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            base_model.conv1 = new_conv

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool

        in_features = base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
