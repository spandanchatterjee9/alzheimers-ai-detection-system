"""
Spatial Resizer Module for Alzheimer's Disease Detection System.

Responsibility:
Resizes 3D volumetric MRI tensors and 2D slices to standardized input resolutions
required by deep learning model backbones (e.g. 128x128x128 for 3D CNNs, 224x224 for 2D CNNs).

Algorithms & Operations:
1. 3D Spline Interpolation (order 3 cubic for volumetric, order 1 linear for slices).
2. Anti-aliasing filtering via skimage.transform.resize / scipy.ndimage.zoom.
3. Aspect-ratio preserving padding or direct shape transformation.

Complexity:
O(V_target) time complexity where V_target is target voxel count.
O(V_target) space complexity for output tensor.
"""

import logging
from typing import Tuple, Sequence
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize

logger = logging.getLogger("preprocessing.resizer")


class VolumeResizer:
    """
    Handles spatial resizing of 3D volumes and 2D slices.
    """

    def __init__(self, target_shape: Sequence[int] = (128, 128, 128), interpolation_order: int = 3) -> None:
        """
        Args:
            target_shape: Target dimensions tuple (e.g. (128, 128, 128) or (224, 224)).
            interpolation_order: Order of spline interpolation (0=nearest, 1=linear, 3=cubic).
        """
        self.target_shape = tuple(target_shape)
        self.interpolation_order = interpolation_order

    def resize_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resizes a 3D volume to target_shape (depth, height, width).

        Args:
            volume: 3D NumPy array.

        Returns:
            Resized 3D NumPy array float32.
        """
        if volume.shape == self.target_shape:
            return volume.astype(np.float32)

        resized = resize(
            volume,
            self.target_shape,
            order=self.interpolation_order,
            mode="constant",
            cval=0.0,
            anti_aliasing=True,
            preserve_range=True,
        )

        logger.debug(f"Resized 3D volume from {volume.shape} to {resized.shape}")
        return resized.astype(np.float32)

    def resize_2d(self, image: np.ndarray, target_2d_shape: Sequence[int] = (224, 224)) -> np.ndarray:
        """
        Resizes a 2D slice to target 2D shape.

        Args:
            image: 2D NumPy array.
            target_2d_shape: Tuple of (height, width).

        Returns:
            Resized 2D NumPy array float32.
        """
        target_tuple = tuple(target_2d_shape)
        if image.shape == target_tuple:
            return image.astype(np.float32)

        resized = resize(
            image,
            target_tuple,
            order=1,  # Bilinear for 2D
            mode="constant",
            cval=0.0,
            anti_aliasing=True,
            preserve_range=True,
        )
        return resized.astype(np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    resizer = VolumeResizer(target_shape=(128, 128, 128))
    
    dummy_3d = np.random.rand(256, 256, 160).astype(np.float32)
    res_3d = resizer.resize_3d(dummy_3d)
    print(f"3D Resize Test -> Input: {dummy_3d.shape}, Output: {res_3d.shape}")

    dummy_2d = np.random.rand(256, 256).astype(np.float32)
    res_2d = resizer.resize_2d(dummy_2d, (224, 224))
    print(f"2D Resize Test -> Input: {dummy_2d.shape}, Output: {res_2d.shape}")
