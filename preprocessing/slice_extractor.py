"""
Slice Extractor Module for Alzheimer's Disease Detection System.

Responsibility:
Decomposes 3D volumetric MRI tensors into 2D planar images along specified anatomical axes
(Axial, Coronal, Sagittal) for 2D Deep Learning architectures (ResNet, EfficientNet, Vision Transformers).

Algorithms & Operations:
1. Plane slicing along primary anatomical axes (RAS orientation):
   - Axial (Transverse): XY plane along Z-axis (Inferior -> Superior).
   - Coronal: XZ plane along Y-axis (Posterior -> Anterior).
   - Sagittal: YZ plane along X-axis (Left -> Right).
2. Index selection: All slices, middle N percentage, or specific slice indices.

Complexity:
O(V) time complexity where V is total voxels (~10.4M voxels).
O(V) space complexity for extracted slice arrays.
"""

import logging
from typing import Dict, List, Tuple, Sequence, Optional
import numpy as np

logger = logging.getLogger("preprocessing.slice_extractor")


class SliceExtractor:
    """
    Extracts 2D planar images from 3D MRI volumes across specified axes.
    """

    def __init__(
        self,
        axes: Sequence[str] = ("axial", "coronal", "sagittal"),
        slice_range_pct: Tuple[float, float] = (0.15, 0.85)
    ) -> None:
        """
        Args:
            axes: Sequence of anatomical axes to extract ('axial', 'coronal', 'sagittal').
            slice_range_pct: Range percentage (min_pct, max_pct) of slices to extract,
                             filtering out extreme peripheral non-brain slices.
        """
        self.axes = [a.lower() for a in axes]
        self.slice_range_pct = slice_range_pct

    def extract_slices(self, volume: np.ndarray) -> Dict[str, List[Tuple[int, np.ndarray]]]:
        """
        Extracts 2D planar slices from 3D volume.

        Args:
            volume: 3D NumPy array of shape (X, Y, Z) in RAS orientation.

        Returns:
            Dictionary mapping axis name -> List of (slice_index, 2D_slice_array).
        """
        nx, ny, nz = volume.shape
        extracted: Dict[str, List[Tuple[int, np.ndarray]]] = {}

        min_pct, max_pct = self.slice_range_pct

        for axis in self.axes:
            extracted[axis] = []

            if axis == "axial":
                # Z-axis (0 to nz-1)
                start_idx = int(nz * min_pct)
                end_idx = int(nz * max_pct)
                for z in range(start_idx, end_idx):
                    slice_2d = volume[:, :, z]
                    extracted[axis].append((z, slice_2d.astype(np.float32)))

            elif axis == "coronal":
                # Y-axis (0 to ny-1)
                start_idx = int(ny * min_pct)
                end_idx = int(ny * max_pct)
                for y in range(start_idx, end_idx):
                    slice_2d = volume[:, y, :]
                    extracted[axis].append((y, slice_2d.astype(np.float32)))

            elif axis == "sagittal":
                # X-axis (0 to nx-1)
                start_idx = int(nx * min_pct)
                end_idx = int(nx * max_pct)
                for x in range(start_idx, end_idx):
                    slice_2d = volume[x, :, :]
                    extracted[axis].append((x, slice_2d.astype(np.float32)))

            else:
                logger.warning(f"Unrecognized axis requested: {axis}")

            logger.debug(f"Extracted {len(extracted[axis])} slices along {axis} axis.")

        return extracted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = SliceExtractor()
    
    dummy_vol = np.random.rand(256, 256, 160).astype(np.float32)
    slices_dict = extractor.extract_slices(dummy_vol)
    for axis, slices in slices_dict.items():
        print(f"Axis '{axis}': {len(slices)} slices extracted. First slice shape: {slices[0][1].shape}")
