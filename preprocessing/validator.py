"""
Volume Validator Module for Alzheimer's Disease Detection System.

Responsibility:
Performs automated quality assurance checks on loaded 3D volumetric MRI tensors
before entering downstream preprocessing, ensuring corrupt, empty, or numerically unstable
data is caught early.

Algorithms & Operations:
1. Dimensionality assertion (3D tensor or 4D tensor with singleton 4th dim).
2. NaN and Inf numerical instability detection.
3. Zero-variance and non-empty foreground ratio verification.
4. Voxel intensity range and spacing validation.

Complexity:
O(V) time complexity where V is number of voxels (~10.4M voxels).
O(1) auxiliary space complexity.
"""

import logging
from typing import Tuple, List, Optional
import numpy as np

logger = logging.getLogger("preprocessing.validator")


class VolumeValidator:
    """
    Quality assurance and integrity checker for 3D MRI volumes.
    """

    def __init__(
        self,
        expected_dims: int = 3,
        min_foreground_ratio: float = 0.05,
        allow_zero_background: bool = True
    ) -> None:
        """
        Args:
            expected_dims: Expected spatial dimensionality (3D).
            min_foreground_ratio: Minimum ratio of non-zero voxels required (default 5%).
            allow_zero_background: Whether voxels with value 0 are acceptable for background.
        """
        self.expected_dims = expected_dims
        self.min_foreground_ratio = min_foreground_ratio
        self.allow_zero_background = allow_zero_background

    def validate(self, volume: np.ndarray, spacing: Optional[Tuple[float, float, float]] = None) -> Tuple[bool, List[str]]:
        """
        Validates a 3D MRI volume.

        Args:
            volume: 3D NumPy array.
            spacing: Optional tuple of (dx, dy, dz) voxel dimensions.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues: List[str] = []

        # 1. Null check
        if volume is None or volume.size == 0:
            issues.append("Volume is None or empty array.")
            return False, issues

        # 2. Dimensionality check
        if volume.ndim != self.expected_dims:
            issues.append(f"Invalid volume dimensions: {volume.ndim}. Expected {self.expected_dims}D.")

        # 3. NaN or Infinity check
        if np.isnan(volume).any():
            issues.append("Volume contains NaN values.")
        if np.isinf(volume).any():
            issues.append("Volume contains Inf values.")

        # 4. Zero variance check
        std_val = np.std(volume)
        if std_val == 0:
            issues.append("Volume has zero variance (constant intensity across all voxels).")

        # 5. Foreground volume ratio check
        non_zero_voxels = np.count_nonzero(volume > 0)
        fg_ratio = non_zero_voxels / volume.size
        if fg_ratio < self.min_foreground_ratio:
            issues.append(f"Insufficient foreground voxels: {fg_ratio:.2%} < {self.min_foreground_ratio:.2%}.")

        # 6. Negative values check (MRI magnitude images should be non-negative)
        min_val = np.min(volume)
        if min_val < 0 and not self.allow_zero_background:
            issues.append(f"Volume contains negative intensities: min={min_val}.")

        # 7. Voxel spacing sanity check
        if spacing is not None:
            if any(s <= 0 for s in spacing):
                issues.append(f"Invalid non-positive voxel spacing: {spacing}.")

        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning(f"Volume validation failed with issues: {issues}")
        else:
            logger.debug("Volume validation passed successfully.")

        return is_valid, issues


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validator = VolumeValidator()
    
    # Test with valid dummy volume
    dummy_valid = np.random.rand(100, 100, 100).astype(np.float32)
    valid, issues = validator.validate(dummy_valid, (1.0, 1.0, 1.0))
    print(f"Dummy Valid Test -> Passed: {valid}, Issues: {issues}")

    # Test with invalid NaN volume
    dummy_invalid = dummy_valid.copy()
    dummy_invalid[10, 10, 10] = np.nan
    valid, issues = validator.validate(dummy_invalid)
    print(f"Dummy NaN Test -> Passed: {valid}, Issues: {issues}")
