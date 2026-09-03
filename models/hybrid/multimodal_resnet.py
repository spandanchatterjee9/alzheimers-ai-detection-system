"""
Multimodal Hybrid Architecture for Alzheimer's Stage Classification.
Combines 2D ResNet-18 MRI Vision Embeddings with Patient Tabular Clinical Biomarkers (Age, Sex, MMSE).
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


class MultimodalResNet18(nn.Module):
    """
    Multimodal Hybrid Neural Network:
    - Vision Branch: Pretrained ResNet-18 2D CNN (512-dim visual embedding)
    - Tabular Branch: 2-Layer MLP on Clinical Metadata (Age, Sex, MMSE, eTIV) (64-dim embedding)
    - Fusion Head: Concatenates visual + clinical features (576-dim) -> 3-Class CDR Classifier.
    """

    def __init__(
        self,
        num_classes: int = 3,
        clinical_features_dim: int = 4,
        vision_embedding_dim: int = 512,
        tabular_embedding_dim: int = 64,
        dropout: float = 0.3,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.clinical_features_dim = clinical_features_dim

        # 1. Vision Feature Extractor (ResNet-18)
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        # Adapt 1-channel Grayscale Conv
        orig_conv = base_resnet.conv1
        self.conv1 = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(orig_conv.weight.sum(dim=1, keepdim=True))

        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool

        # 2. Clinical Tabular Feature Extractor (MLP)
        self.tabular_net = nn.Sequential(
            nn.Linear(clinical_features_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, tabular_embedding_dim),
            nn.BatchNorm1d(tabular_embedding_dim),
            nn.ReLU()
        )

        # 3. Multimodal Fusion Classification Head
        fusion_dim = vision_embedding_dim + tabular_embedding_dim  # 512 + 64 = 576
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def extract_vision_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dim visual representation from 2D MRI slice."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(
        self,
        slices: torch.Tensor,
        clinical_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multimodal batch.
        Args:
            slices: (B, 1, 224, 224) 2D MRI slice tensor.
            clinical_data: (B, 4) Normalized tabular biomarkers [Age, Sex, MMSE, BrainVolume].
        """
        vision_feat = self.extract_vision_features(slices)  # (B, 512)

        if clinical_data is None:
            # Fallback zero clinical embedding if metadata missing
            device = slices.device
            clinical_data = torch.zeros((slices.size(0), self.clinical_features_dim), device=device)

        clinical_feat = self.tabular_net(clinical_data)  # (B, 64)

        # Fuse embeddings
        multimodal_feat = torch.cat([vision_feat, clinical_feat], dim=1)  # (B, 576)
        logits = self.fusion_classifier(multimodal_feat)  # (B, 3)
        return logits
