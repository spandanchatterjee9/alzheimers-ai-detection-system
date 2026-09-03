"""
Spatial Orientation Module for Alzheimer's Disease Detection System.

Responsibility:
Standardizes all incoming 3D volumetric MRI tensors to canonical RAS (Right-Anterior-Superior)
spatial orientation, eliminating orientation discrepancies across scanner platforms.

Algorithms & Operations:
1. Computes orientation matrix from affine transform via nibabel.orientations.io_orientation.
2. Applies axis permutations and spatial flips via nibabel.orientations.apply_orientation.
3. Updates the affine transform matrix to reflect RAS coordinate alignment.

Complexity:
O(V) time complexity where V is total voxels (~10.4M voxels).
O(V) space complexity for reoriented output array.
"""

import logging
from typing import Tuple
import numpy as np

try:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logger = logging.getLogger("preprocessing.orientation")


class VolumeOrientation:
    """
    Standardizes 3D MRI volume orientation to canonical RAS.
    """

    def __init__(self, target_orientation: Tuple[str, str, str] = ("R", "A", "S")) -> None:
        """
        Args:
            target_orientation: Target orientation codes tuple (e.g. ('R', 'A', 'S')).
        """
        self.target_orientation = target_orientation

    def reorient_to_ras(self, volume: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reorients volume and affine matrix to RAS canonical orientation.

        Args:
            volume: 3D NumPy array of shape (X, Y, Z).
            affine: 4x4 affine transform matrix.

        Returns:
            Tuple of (reoriented_volume, ras_affine).
        """
        if not HAS_NIBABEL:
            logger.warning("NiBabel not installed; returning original orientation.")
            return volume, affine

        try:
            # Determine current orientation code from affine
            curr_ornt = io_orientation(affine)
            target_ornt = axcodes2ornt(self.target_orientation)

            # Compute transformation needed from current to target
            transform = ornt_transform(curr_ornt, target_ornt)

            # Apply orientation transformation to 3D volume array
            reoriented_vol = apply_orientation(volume, transform)

            # Compute updated affine matrix
            rotation_transform = nib.orientations.inv_ornt_aff(transform, volume.shape)
            ras_affine = np.dot(affine, rotation_transform)

            logger.debug(f"Reoriented volume from {curr_ornt} to {target_ornt}. Shape: {reoriented_vol.shape}")
            return reoriented_vol.astype(np.float32), ras_affine
        except Exception as e:
            logger.warning(f"Orientation reorient failed: {e}. Returning original volume.")
            return volume, affine


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orienter = VolumeOrientation()
    
    # Test dummy volume and affine
    dummy_vol = np.random.rand(256, 256, 160).astype(np.float32)
    dummy_aff = np.eye(4)
    res_vol, res_aff = orienter.reorient_to_ras(dummy_vol, dummy_aff)
    print(f"Orientation Test -> Input Shape: {dummy_vol.shape}, Output Shape: {res_vol.shape}")
