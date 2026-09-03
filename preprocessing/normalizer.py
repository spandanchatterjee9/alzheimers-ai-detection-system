"""
Intensity Normalization Module for Alzheimer's Disease Detection System.

Responsibility:
Normalizes MRI voxel intensities into a numerically standardized range, eliminating
scanner gain variances, coil sensitivity profiles, and inter-subject intensity scaling differences.

Algorithms & Operations:
1. Percentile Intensity Clipping (0.5th to 99.5th percentile) to remove scanner spikes.
2. Z-Score Normalization on Non-Zero Brain Voxels: (x - mean_fg) / std_fg.
3. Min-Max Rescaling to [0.0, 1.0] range.
4. Peak Intensity Normalization.

Complexity:
O(V) time complexity where V is number of voxels (~10.4M voxels).
O(V) space complexity for normalized volume tensor.
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger("preprocessing.normalizer")


class IntensityNormalizer:
    """
    Normalizes MRI voxel intensity distributions.
    """

    def __init__(
        self,
        method: str = "z_score",
        clip_percentiles: Optional[Tuple[float, float]] = (0.5, 99.5),
        min_max_range: Tuple[float, float] = (0.0, 1.0)
    ) -> None:
        """
        Args:
            method: Normalization method ('z_score', 'min_max', 'peak').
            clip_percentiles: Tuple of (min_p, max_p) percentiles for intensity clipping.
            min_max_range: Output min and max bounds for min_max scaling.
        """
        self.method = method.lower()
        self.clip_percentiles = clip_percentiles
        self.min_max_range = min_max_range

    def clip_outliers(self, volume: np.ndarray) -> np.ndarray:
        """Clips extreme intensity outliers based on foreground percentiles."""
        if self.clip_percentiles is None:
            return volume

        fg_mask = volume > 0
        if not np.any(fg_mask):
            return volume

        lower_p, upper_p = self.clip_percentiles
        min_val = np.percentile(volume[fg_mask], lower_p)
        max_val = np.percentile(volume[fg_mask], upper_p)

        clipped = np.clip(volume, min_val, max_val)
        return clipped

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """
        Applies intensity normalization to the 3D volume.

        Args:
            volume: 3D NumPy array float32.

        Returns:
            Normalized 3D NumPy array float32.
        """
        vol_clipped = self.clip_outliers(volume.astype(np.float32))
        fg_mask = vol_clipped > 0

        if not np.any(fg_mask):
            logger.warning("Volume has no positive foreground voxels during normalization.")
            return vol_clipped

        if self.method == "z_score":
            mean_val = np.mean(vol_clipped[fg_mask])
            std_val = np.std(vol_clipped[fg_mask])
            if std_val < 1e-8:
                std_val = 1e-8

            normalized = np.zeros_like(vol_clipped)
            normalized[fg_mask] = (vol_clipped[fg_mask] - mean_val) / std_val
            return normalized.astype(np.float32)

        elif self.method == "min_max":
            min_val = np.min(vol_clipped[fg_mask])
            max_val = np.max(vol_clipped[fg_mask])
            range_val = max_val - min_val
            if range_val < 1e-8:
                range_val = 1e-8

            target_min, target_max = self.min_max_range
            normalized = np.zeros_like(vol_clipped)
            normalized[fg_mask] = target_min + (vol_clipped[fg_mask] - min_val) * (target_max - target_min) / range_val
            return normalized.astype(np.float32)

        elif self.method == "peak":
            p99 = np.percentile(vol_clipped[fg_mask], 99.0)
            if p99 < 1e-8:
                p99 = 1e-8
            normalized = vol_clipped / p99
            return np.clip(normalized, 0.0, 1.0).astype(np.float32)

        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    normalizer = IntensityNormalizer(method="z_score")
    
    dummy_vol = np.random.uniform(0, 4000, (100, 100, 100)).astype(np.float32)
    norm_vol = normalizer.normalize(dummy_vol)
    print(f"Z-Score Test -> Mean FG: {np.mean(norm_vol[dummy_vol > 0]):.4f}, Std FG: {np.std(norm_vol[dummy_vol > 0]):.4f}")
