"""
Brain Bounding Box Cropper Module for Alzheimer's Disease Detection System.

Responsibility:
Detects non-zero brain tissue bounding boxes in 3D volumes and crops away redundant
empty background voxels, significantly reducing tensor size and memory consumption.

Algorithms & Operations:
1. Identifies non-zero voxel coordinates across all 3 spatial axes.
2. Computes tight spatial bounding box [xmin:xmax, ymin:ymax, zmin:zmax].
3. Applies safety padding margin.
4. Returns cropped 3D volume tensor and bounding box coordinate metadata.

Complexity:
O(V) time complexity where V is total voxels (~10.4M voxels).
O(V_cropped) space complexity for output cropped tensor.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np

logger = logging.getLogger("preprocessing.cropper")


class VolumeCropper:
    """
    Crops empty background surrounding brain tissue in 3D volumes.
    """

    def __init__(self, background_threshold: float = 0.0, padding_voxels: int = 4) -> None:
        """
        Args:
            background_threshold: Voxel intensity below which is considered background.
            padding_voxels: Extra padding voxels to retain around brain edges.
        """
        self.background_threshold = background_threshold
        self.padding_voxels = padding_voxels

    def compute_bounding_box(self, volume: np.ndarray) -> Tuple[int, int, int, int, int, int]:
        """
        Computes 3D bounding box coordinates of non-background voxels.

        Args:
            volume: 3D NumPy array.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        non_zero_coords = np.argwhere(volume > self.background_threshold)
        if non_zero_coords.size == 0:
            logger.warning("No foreground voxels found above threshold. Returning full volume bounds.")
            return 0, volume.shape[0], 0, volume.shape[1], 0, volume.shape[2]

        xmin, ymin, zmin = non_zero_coords.min(axis=0)
        xmax, ymax, zmax = non_zero_coords.max(axis=0) + 1

        # Apply padding
        xmin = max(0, xmin - self.padding_voxels)
        ymin = max(0, ymin - self.padding_voxels)
        zmin = max(0, zmin - self.padding_voxels)

        xmax = min(volume.shape[0], xmax + self.padding_voxels)
        ymax = min(volume.shape[1], ymax + self.padding_voxels)
        zmax = min(volume.shape[2], zmax + self.padding_voxels)

        return int(xmin), int(xmax), int(ymin), int(ymax), int(zmin), int(zmax)

    def crop(self, volume: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Crops volume to non-background bounding box.

        Args:
            volume: 3D NumPy array.

        Returns:
            Tuple of (cropped_volume, bounding_box_dict).
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.compute_bounding_box(volume)
        cropped_vol = volume[xmin:xmax, ymin:ymax, zmin:zmax]

        bbox_dict = {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax,
            "orig_shape": volume.shape,
            "cropped_shape": cropped_vol.shape,
        }

        logger.debug(f"Cropped volume from {volume.shape} to {cropped_vol.shape}")
        return cropped_vol, bbox_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cropper = VolumeCropper(padding_voxels=2)
    
    # Test dummy volume with padding
    dummy_vol = np.zeros((100, 100, 100), dtype=np.float32)
    dummy_vol[20:70, 30:80, 15:65] = 1.0
    cropped, bbox = cropper.crop(dummy_vol)
    print(f"Crop Test -> Original: {dummy_vol.shape}, Cropped: {cropped.shape}")
    print(f"Bounding Box: {bbox}")
