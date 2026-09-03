"""
2D Grad-CAM (Gradient-Weighted Class Activation Mapping) Module for EfficientNet-B0.

Responsibility:
Generates spatial attribution heatmaps for 224x224 grayscale MRI slices highlighting
anatomical regions influencing the network's CDR stage classification.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GradCAM2D:
    """
    2D Grad-CAM implementation for 2D EfficientNet-B0.
    """

    def __init__(self, model: Any, target_layer: Optional[Any] = None) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GradCAM2D.")

        self.model = model
        self.model.eval()

        # Default to final convolutional block in EfficientNet-B0 (features[8])
        if target_layer is None:
            self.target_layer = self.model.features[-1]
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates 2D Grad-CAM heatmap normalized to [0, 1] for a single 2D slice.

        Args:
            input_tensor: Tensor of shape (1, 1, 224, 224).
            target_class: Target class index (0, 1, 2, 3). If None, uses argmax prediction.

        Returns:
            Normalized 2D heatmap NumPy array of shape (224, 224) with values in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients over spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Interpolate to input slice resolution (224, 224)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam
