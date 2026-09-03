"""
Volumetric MRI Loader Module for Alzheimer's Disease Detection System.

Responsibility:
Loads 3D volumetric MRI scans from Analyze 7.5 (.hdr/.img) or NIfTI (.nii/.nii.gz) files
into unified 3D NumPy float32 arrays along with spatial voxel dimensions and affine transformation matrices.

Algorithms & Operations:
1. Primary loading via NiBabel (nib.load).
2. Fallback pure-Python Analyze 7.5 header & binary parser supporting Big-Endian (>i2)
   and Little-Endian (<i2) 16-bit integer conversions.
3. Affine matrix extraction and voxel spacing resolution extraction.

Complexity:
O(V) time complexity where V is total voxels (~10.4M voxels for 256x256x160).
O(V) space complexity to hold the 3D float32 volume in RAM (~41.9 MB float32 tensor).
"""

import os
import struct
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logger = logging.getLogger("preprocessing.loader")


class MriLoader:
    """
    Handles reading Analyze 7.5 (.hdr/.img) and NIfTI (.nii/.nii.gz) volumes.
    """

    @staticmethod
    def _read_analyze_hdr(hdr_path: str) -> Dict[str, Any]:
        """
        Fallback parser for 348-byte Analyze 7.5 header files.
        """
        with open(hdr_path, "rb") as f:
            data = f.read(348)

        if len(data) < 348:
            raise ValueError(f"Header file is corrupt or less than 348 bytes: {hdr_path}")

        # Check Big-Endian vs Little-Endian header size
        sizeof_hdr_be = struct.unpack(">i", data[0:4])[0]
        sizeof_hdr_le = struct.unpack("<i", data[0:4])[0]

        if sizeof_hdr_be == 348:
            endian = ">"
        elif sizeof_hdr_le == 348:
            endian = "<"
        else:
            raise ValueError(f"Unrecognized Analyze 7.5 header size: {hdr_path}")

        dim = struct.unpack(endian + "8h", data[40:56])
        datatype = struct.unpack(endian + "h", data[70:72])[0]
        bitpix = struct.unpack(endian + "h", data[72:74])[0]
        pixdim = struct.unpack(endian + "8f", data[76:108])

        # dim[1], dim[2], dim[3] are spatial dimensions (e.g. 256, 256, 160)
        nx, ny, nz = dim[1], dim[2], dim[3]
        dx, dy, dz = pixdim[1], pixdim[2], pixdim[3]

        return {
            "endian": endian,
            "shape": (nx, ny, nz),
            "spacing": (dx, dy, dz),
            "datatype": datatype,
            "bitpix": bitpix,
        }

    @classmethod
    def load_volume(cls, file_path: str) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
        """
        Loads a 3D MRI volume from disk.

        Args:
            file_path: Path to .hdr, .img, .nii, or .nii.gz file.

        Returns:
            Tuple of:
            - volume: 3D NumPy array of shape (X, Y, Z) and float32 dtype.
            - voxel_spacing: Tuple of (dx, dy, dz) in mm.
            - affine: 4x4 affine transform matrix.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MRI file does not exist: {file_path}")

        # Primary route: Try NiBabel
        if HAS_NIBABEL:
            try:
                img = nib.load(file_path)
                data = img.get_fdata(dtype=np.float32)
                
                # If data has 4 dimensions (e.g. 256x256x160x1), squeeze 4th dim
                if data.ndim == 4 and data.shape[3] == 1:
                    data = np.squeeze(data, axis=3)

                zooms = img.header.get_zooms()[:3]
                affine = img.affine
                logger.debug(f"Loaded volume via NiBabel. Shape: {data.shape}, Spacing: {zooms}")
                return data.astype(np.float32), (float(zooms[0]), float(zooms[1]), float(zooms[2])), affine
            except Exception as e:
                logger.warning(f"NiBabel load failed for {file_path}: {e}. Retrying fallback loader.")

        # Fallback route for Analyze 7.5 .hdr / .img pair
        base_path, ext = os.path.splitext(file_path)
        hdr_path = base_path + ".hdr" if ext.lower() in [".hdr", ".img"] else file_path
        img_path = base_path + ".img" if ext.lower() in [".hdr", ".img"] else file_path

        if not os.path.exists(hdr_path) or not os.path.exists(img_path):
            raise FileNotFoundError(f"Both HDR and IMG files are required for Analyze 7.5: {hdr_path}, {img_path}")

        hdr_info = cls._read_analyze_hdr(hdr_path)
        nx, ny, nz = hdr_info["shape"]
        total_voxels = nx * ny * nz
        endian = hdr_info["endian"]

        dtype_map = {
            2: np.uint8,
            4: np.int16,
            8: np.int32,
            16: np.float32,
            64: np.float64,
        }

        elem_dtype = dtype_map.get(hdr_info["datatype"], np.int16)
        dt = np.dtype(elem_dtype).newbyteorder(endian)

        with open(img_path, "rb") as f:
            raw_bytes = f.read(total_voxels * dt.itemsize)

        volume = np.frombuffer(raw_bytes, dtype=dt).reshape((nx, ny, nz)).astype(np.float32)
        spacing = hdr_info["spacing"]

        # Default diagonal affine matrix
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

        logger.debug(f"Loaded volume via Fallback. Shape: {volume.shape}, Spacing: {spacing}")
        return volume, spacing, affine


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_hdr = r"oasis_cross-sectional_disc1/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr"
    if os.path.exists(test_hdr):
        vol, sp, aff = MriLoader.load_volume(test_hdr)
        print(f"Volume loaded successfully!")
        print(f"Shape: {vol.shape}, Dtype: {vol.dtype}")
        print(f"Min: {vol.min()}, Max: {vol.max()}, Mean: {vol.mean():.2f}")
        print(f"Voxel Spacing: {sp}")
        print(f"Affine:\n{aff}")
    else:
        print(f"Test file not found: {test_hdr}")
