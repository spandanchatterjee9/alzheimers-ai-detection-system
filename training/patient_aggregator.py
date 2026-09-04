"""
Patient-Level Probability Aggregator Module.

Responsibility:
Aggregates slice-level 4-class softmax probability predictions belonging to the same patient
into a single patient-level probability distribution and predicted CDR stage.

Aggregation Strategy:
Primary Baseline: Mean Probability Aggregation across all valid quality-filtered slices of the patient.
Formula:
    P_patient = (1 / N_slices) * sum(P_slice_i)
    Predicted Stage = argmax(P_patient)
    Overall Dementia Probability = P(CDR=0.5) + P(CDR=1.0) + P(CDR=2.0)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def aggregate_slice_probabilities(
    slice_probs: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregates a collection of slice-level probability vectors (N_slices, 4) into a single patient-level vector (4,).

    Args:
        slice_probs: Slice probability vectors of shape (N_slices, 4) or list of 4-element arrays.
        method: Aggregation strategy ('mean' supported as primary baseline).

    Returns:
        Patient-level probability vector of shape (4,) summing to 1.0.
    """
    if isinstance(slice_probs, torch.Tensor):
        probs_np = slice_probs.detach().cpu().numpy()
    elif isinstance(slice_probs, list):
        probs_np = np.array(slice_probs, dtype=np.float32)
    else:
        probs_np = np.asarray(slice_probs, dtype=np.float32)

    if probs_np.ndim == 1:
        probs_np = np.expand_dims(probs_np, axis=0)

    if method == "mean":
        patient_prob = np.mean(probs_np, axis=0)
    elif method == "median":
        patient_prob = np.median(probs_np, axis=0)
        s = np.sum(patient_prob)
        if s > 0:
            patient_prob = patient_prob / s
    elif method == "max":
        patient_prob = np.max(probs_np, axis=0)
        s = np.sum(patient_prob)
        if s > 0:
            patient_prob = patient_prob / s
    elif method == "diagnostic_gaussian":
        # Anatomical Gaussian centered at 52% brain depth (medial temporal lobe / hippocampus)
        N = probs_np.shape[0]
        weights = np.array([
            max(np.exp(-((i / max(N, 1) - 0.52) ** 2) / (2 * (0.18 ** 2))), 0.15)
            for i in range(N)
        ], dtype=np.float32)[:, None]
        patient_prob = np.sum(probs_np * weights, axis=0) / np.sum(weights)
    elif method == "calibrated_mci":
        # Gaussian depth weighting + gentle calibrated MCI decision threshold prior (+0.05)
        N = probs_np.shape[0]
        weights = np.array([
            max(np.exp(-((i / max(N, 1) - 0.52) ** 2) / (2 * (0.18 ** 2))), 0.15)
            for i in range(N)
        ], dtype=np.float32)[:, None]
        patient_prob = np.sum(probs_np * weights, axis=0) / np.sum(weights)
        if len(patient_prob) >= 2:
            patient_prob[1] += 0.05
            patient_prob = patient_prob / np.sum(patient_prob)
    else:
        patient_prob = np.mean(probs_np, axis=0)

    return patient_prob


class PatientAggregator:
    """
    Aggregates batch or sequential slice predictions grouped by patient ID.
    """

    def __init__(self, method: str = "mean") -> None:
        self.method = method
        self.patient_slices: Dict[str, List[np.ndarray]] = {}
        self.patient_metadata: Dict[str, Dict[str, Any]] = {}

    def add_slice_prediction(
        self,
        patient_id: str,
        slice_prob: Union[torch.Tensor, np.ndarray],
        ground_truth: Optional[int] = None,
        raw_cdr: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Adds a single slice prediction for a patient.
        """
        if isinstance(slice_prob, torch.Tensor):
            prob_vec = slice_prob.detach().cpu().numpy()
        else:
            prob_vec = np.asarray(slice_prob, dtype=np.float32)

        if prob_vec.ndim > 1 and prob_vec.shape[0] == 1:
            prob_vec = prob_vec.squeeze(0)

        if patient_id not in self.patient_slices:
            self.patient_slices[patient_id] = []
            self.patient_metadata[patient_id] = {
                "ground_truth": ground_truth,
                "raw_cdr": raw_cdr,
                "extra": extra_meta or {}
            }

        self.patient_slices[patient_id].append(prob_vec)

    def compute_patient_predictions(self) -> List[Dict[str, Any]]:
        """
        Computes final patient-level probability distributions and CDR classifications.

        Returns:
            List of dictionaries containing patient-level predictions and metrics.
        """
        results: List[Dict[str, Any]] = []

        for patient_id, slices in self.patient_slices.items():
            patient_prob_vec = aggregate_slice_probabilities(slices, method=self.method)
            pred_class = int(np.argmax(patient_prob_vec))
            gt_class = self.patient_metadata[patient_id]["ground_truth"]
            raw_cdr = self.patient_metadata[patient_id]["raw_cdr"]

            # Overall dementia classification probability (sum of non-zero CDR classes)
            overall_dementia_prob = float(patient_prob_vec[1] + patient_prob_vec[2] + patient_prob_vec[3])

            record = {
                "patient_id": patient_id,
                "num_valid_slices": len(slices),
                "prob_cdr_0.0": float(patient_prob_vec[0]),
                "prob_cdr_0.5": float(patient_prob_vec[1]),
                "prob_cdr_1.0": float(patient_prob_vec[2]),
                "prob_cdr_2.0": float(patient_prob_vec[3]),
                "patient_prob_vector": patient_prob_vec.tolist(),
                "predicted_class": pred_class,
                "ground_truth_class": gt_class,
                "raw_cdr": raw_cdr,
                "overall_dementia_prob": overall_dementia_prob,
                "is_correct": bool(pred_class == gt_class) if gt_class is not None else None
            }
            results.append(record)

        return results

    def reset(self) -> None:
        """Resets stored slice predictions."""
        self.patient_slices.clear()
        self.patient_metadata.clear()
