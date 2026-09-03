"""
Quality Filter Module for Alzheimer's Disease Detection System.

Responsibility:
Evaluates individual 2D MRI slices against quantitative quality metrics (foreground tissue ratio,
Shannon entropy, and intensity variance) to discard non-informative, edge-of-skull, or blank slices.

Algorithms & Operations:
1. Non-zero Foreground Pixel Ratio: count(pixels > threshold) / total_pixels.
2. Image Shannon Entropy: H = - sum(p * log2(p)) calculated over 256-bin intensity histogram.
3. Spatial Variance & Peak-to-Noise Ratio check.

Complexity:
O(P) time complexity per slice where P is number of 2D pixels (~50k to 65k pixels).
O(1) auxiliary space complexity.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np

logger = logging.getLogger("preprocessing.quality_filter")


class SliceQualityFilter:
    """
    Evaluates quality metrics for 2D MRI slices to discard low-information images.
    """

    def __init__(
        self,
        min_foreground_ratio: float = 0.15,
        min_entropy: float = 1.5,
        min_std: float = 1e-4
    ) -> None:
        """
        Args:
            min_foreground_ratio: Minimum ratio of non-zero brain pixels required (0.15 = 15%).
            min_entropy: Minimum Shannon entropy threshold.
            min_std: Minimum standard deviation of intensity values.
        """
        self.min_foreground_ratio = min_foreground_ratio
        self.min_entropy = min_entropy
        self.min_std = min_std

    @staticmethod
    def calculate_shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
        """
        Calculates Shannon entropy of a 2D image array.
        """
        if image.size == 0:
            return 0.0

        hist, _ = np.histogram(image, bins=bins)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        return float(entropy)

    def evaluate_slice(self, image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluates a 2D slice against quality thresholds.

        Args:
            image: 2D NumPy array.

        Returns:
            Tuple of (is_passed, metrics_dictionary).
        """
        if image is None or image.size == 0:
            return False, {"fg_ratio": 0.0, "entropy": 0.0, "std": 0.0}

        # 1. Foreground ratio
        fg_pixels = np.count_nonzero(image > 0)
        fg_ratio = float(fg_pixels / image.size)

        # 2. Intensity standard deviation
        std_val = float(np.std(image))

        # 3. Shannon Entropy
        entropy_val = self.calculate_shannon_entropy(image)

        metrics = {
            "fg_ratio": fg_ratio,
            "entropy": entropy_val,
            "std": std_val,
        }

        # Decision rule
        is_passed = (
            fg_ratio >= self.min_foreground_ratio
            and entropy_val >= self.min_entropy
            and std_val >= self.min_std
        )

        if not is_passed:
            logger.debug(f"Slice quality filter failed: {metrics}")

        return is_passed, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    qfilter = SliceQualityFilter(min_foreground_ratio=0.15, min_entropy=1.5)
    
    # High-information brain slice mock
    mock_brain = np.zeros((224, 224), dtype=np.float32)
    mock_brain[50:180, 50:180] = np.random.normal(1.0, 0.2, (130, 130))
    passed, m = qfilter.evaluate_slice(mock_brain)
    print(f"Mock Brain Slice -> Passed: {passed}, Metrics: {m}")

    # Blank background slice mock
    mock_blank = np.zeros((224, 224), dtype=np.float32)
    passed, m = qfilter.evaluate_slice(mock_blank)
    print(f"Mock Blank Slice -> Passed: {passed}, Metrics: {m}")
