"""
Primary 2D Deep Learning Architecture: EfficientNet-B0 for Alzheimer's-Related Dementia Stage Classification.

Input: 2D MRI Slices (224 x 224 x 1 or 224 x 224 x 3)
Output: 4-Class Softmax Probability Distribution across CDR 0.0, 0.5, 1.0, 2.0.

Slice-to-Patient Aggregation:
Computes slice-level predictions and aggregates probabilities across all valid slices of a subject using
Mean Probability Aggregation to yield a single patient-level prediction.

Rationale:
Computationally practical baseline that significantly reduces GPU memory requirements, training time,
and model complexity while retaining spatial MRI structural information through quality-filtered 2D slices.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

logger = logging.getLogger("models.cnn_2d.efficientnet_2d")


class EfficientNetB02D(nn.Module if HAS_TORCH else object):
    """
    2D EfficientNet-B0 Model for CDR Stage Classification on 224x224 MRI Slices.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ) -> None:
        """
        Args:
            num_classes: Number of target CDR categories (default 4: CDR 0, 0.5, 1, 2).
            in_channels: Number of input channels (1 for grayscale MRI slice, 3 for RGB).
            pretrained: Use ImageNet pretrained weights for transfer learning.
            dropout_rate: Dropout probability in classification head.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to instantiate EfficientNetB02D.")

        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load torchvision EfficientNet-B0 backbone
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        # Adapt first conv layer if input is grayscale (in_channels=1)
        if in_channels == 1:
            old_conv = base_model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            if pretrained:
                # Average ImageNet RGB weights across input channels for grayscale initialization
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            base_model.features[0][0] = new_conv

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Replace classification head
        in_features = base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2D MRI slice tensor (Batch_Size, in_channels, 224, 224).
        Returns raw logits (Batch_Size, num_classes).
        """
        # Expand grayscale to 3 channels if in_channels=1 and input tensor has 1 channel
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes 4-class Softmax probability distribution for input 2D slices.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities


def aggregate_slice_probabilities(
    slice_probs: Union[torch.Tensor, np.ndarray],
    method: str = "mean"
) -> Union[torch.Tensor, np.ndarray]:
    """
    Aggregates slice-level probability vectors (N_slices, 4) to yield one patient-level probability vector (4,).

    Args:
        slice_probs: Slice-level probabilities array or tensor of shape (N_slices, 4).
        method: Aggregation strategy ('mean' supported as primary baseline).

    Returns:
        Patient-level probability vector of shape (4,).
    """
    if isinstance(slice_probs, torch.Tensor):
        if method == "mean":
            patient_prob = torch.mean(slice_probs, dim=0)
        else:
            patient_prob = torch.mean(slice_probs, dim=0)
        return patient_prob
    else:
        if method == "mean":
            patient_prob = np.mean(slice_probs, axis=0)
        else:
            patient_prob = np.mean(slice_probs, axis=0)
        return patient_prob
