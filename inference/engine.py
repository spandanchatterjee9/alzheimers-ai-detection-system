"""
Inference Engine for 2D EfficientNet-B0 Dementia Stage Prediction.

Responsibility:
- Loads trained model checkpoint (efficientnet_b0_best.pt).
- Runs 2D slice forward inference and aggregates slice probabilities to patient level.
- Computes CDR stage prediction, class probability distribution, and overall dementia score.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from models.cnn_2d import EfficientNetB02D
from training.patient_aggregator import aggregate_slice_probabilities

logger = logging.getLogger("inference.engine")


class DementiaInferenceEngine:
    """
    Inference Engine for patient-level 4-class dementia stage prediction.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/checkpoints/efficientnet_b0_best.pt",
        device: str = "auto"
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for DementiaInferenceEngine.")

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_path = Path(checkpoint_path)
        self.model = EfficientNetB02D(num_classes=4, in_channels=1, pretrained=False)

        if self.checkpoint_path.exists():
            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from: {self.checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}. Using initialized weights.")

        self.model.to(self.device)
        self.model.eval()

        self.class_names = {
            0: "Cognitively Normal (CDR 0.0)",
            1: "Very Mild Dementia (CDR 0.5)",
            2: "Mild Dementia (CDR 1.0)",
            3: "Moderate Dementia (CDR 2.0)"
        }

    def predict_slices(self, slice_tensors: Union[torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Runs inference on a collection of 2D MRI slices for a single patient.

        Args:
            slice_tensors: Tensor or NumPy array of shape (N_slices, 1, 224, 224) or (N_slices, 224, 224).

        Returns:
            Dictionary with patient-level prediction, class probabilities, and slice-level predictions.
        """
        if isinstance(slice_tensors, np.ndarray):
            if slice_tensors.ndim == 3:
                slice_tensors = np.expand_dims(slice_tensors, axis=1)
            tensor = torch.from_numpy(slice_tensors).float().to(self.device)
        else:
            if slice_tensors.ndim == 3:
                slice_tensors = slice_tensors.unsqueeze(1)
            tensor = slice_tensors.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            slice_probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Patient-Level Mean Probability Aggregation
        patient_prob_vec = aggregate_slice_probabilities(slice_probs, method="mean")
        pred_class = int(np.argmax(patient_prob_vec))

        # Overall non-zero CDR dementia classification probability
        overall_dementia_prob = float(patient_prob_vec[1] + patient_prob_vec[2] + patient_prob_vec[3])

        return {
            "predicted_class_idx": pred_class,
            "predicted_stage_name": self.class_names.get(pred_class, "Unknown"),
            "patient_probabilities": {
                "CDR_0.0_Normal": float(patient_prob_vec[0]),
                "CDR_0.5_Very_Mild": float(patient_prob_vec[1]),
                "CDR_1.0_Mild": float(patient_prob_vec[2]),
                "CDR_2.0_Moderate": float(patient_prob_vec[3])
            },
            "overall_dementia_probability": overall_dementia_prob,
            "num_slices_evaluated": len(slice_probs),
            "slice_probabilities": slice_probs.tolist()
        }
