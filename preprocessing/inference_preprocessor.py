"""
Inference Preprocessor Module for Alzheimer's Disease Detection System.

Responsibility:
Provides a deterministic, lightweight single-volume preprocessing engine designed for
the FastAPI Backend API / Web Application inference pipeline. Guarantees that any uploaded
MRI scan undergoes the EXACT SAME mathematical transformations as training samples.

Algorithms & Operations:
1. Load single volume from uploaded file path (.nii, .nii.gz, .hdr/.img).
2. Volume validation and quality sanity check.
3. Spatial re-orientation to canonical RAS coordinate space.
4. Intensity normalization (Z-score / Min-Max based on config).
5. Brain bounding box crop.
6. Volumetric resizing to model input tensor (e.g. 1x1x128x128x128).
7. Diagnostic key-slice extraction and 2D slice resizing (1x1x224x224).

Complexity:
O(V) time complexity per uploaded volume (~2 to 3 seconds execution time).
Outputs ready PyTorch-compatible batch tensors (N=1).
"""

import os
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import yaml

from preprocessing.loader import MriLoader
from preprocessing.validator import VolumeValidator
from preprocessing.orientation import VolumeOrientation
from preprocessing.normalizer import IntensityNormalizer
from preprocessing.cropper import VolumeCropper
from preprocessing.resizer import VolumeResizer
from preprocessing.slice_extractor import SliceExtractor
from preprocessing.quality_filter import SliceQualityFilter

logger = logging.getLogger("preprocessing.inference")


class InferencePreprocessor:
    """
    Deterministic inference preprocessor for single volume web uploads.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.loader = MriLoader()
        self.validator = VolumeValidator(expected_dims=3, min_foreground_ratio=0.05)
        self.orienter = VolumeOrientation(
            target_orientation=tuple(self.config["volume_specs"].get("orientation", "RAS"))
        )
        self.normalizer = IntensityNormalizer(
            method=self.config["normalization"]["method"],
            clip_percentiles=tuple(self.config["normalization"]["clip_percentiles"]),
            min_max_range=tuple(self.config["normalization"]["min_max_range"])
        )
        self.cropper = VolumeCropper(
            background_threshold=self.config["cropping"]["background_threshold"],
            padding_voxels=self.config["cropping"]["padding_voxels"]
        )
        self.resizer = VolumeResizer(
            target_shape=tuple(self.config["volume_specs"]["target_shape"]),
            interpolation_order=self.config["resizing"]["interpolation_order"]
        )
        self.slice_extractor = SliceExtractor(
            axes=self.config["slice_extraction"]["axes"]
        )
        self.quality_filter = SliceQualityFilter(
            min_foreground_ratio=self.config["slice_extraction"]["quality_filter"]["min_foreground_ratio"],
            min_entropy=self.config["slice_extraction"]["quality_filter"]["min_entropy"]
        )

    def preprocess_file(self, file_path: str) -> Dict[str, Any]:
        """
        Preprocesses a single uploaded MRI file for inference.

        Args:
            file_path: Path to uploaded .nii, .nii.gz, or .hdr file.

        Returns:
            Dictionary containing:
            - "volume_tensor": 5D NumPy array (1, 1, D, H, W) float32.
            - "slices_tensors": Dict mapping axis -> 4D NumPy array (N_slices, 1, 224, 224).
            - "metadata": Preprocessing metadata and validation info.
        """
        logger.info(f"Starting inference preprocessing for file: {file_path}")

        # 1. Load volume
        raw_volume, spacing, affine = self.loader.load_volume(file_path)

        # 2. Validate
        is_valid, issues = self.validator.validate(raw_volume, spacing)
        if not is_valid:
            raise ValueError(f"Uploaded volume failed quality check: {issues}")

        # 3. Reorient to RAS
        ras_volume, ras_affine = self.orienter.reorient_to_ras(raw_volume, affine)

        # 4. Intensity Normalization
        norm_volume = self.normalizer.normalize(ras_volume)

        # 5. Crop background
        cropped_volume, bbox_info = self.cropper.crop(norm_volume)

        # 6. Resize 3D volume
        resized_3d = self.resizer.resize_3d(cropped_volume)

        # Shape formatting for PyTorch 3D model input: (Batch=1, Channel=1, Depth, Height, Width)
        volume_tensor = np.expand_dims(np.expand_dims(resized_3d, axis=0), axis=0).astype(np.float32)

        # 7. Extract & Filter 2D slices
        slices_dict = self.slice_extractor.extract_slices(norm_volume)
        slices_tensors: Dict[str, np.ndarray] = {}
        target_2d_shape = tuple(self.config["slice_extraction"]["target_slice_shape"])

        for axis, slice_list in slices_dict.items():
            valid_slices = []
            for idx, slice_img in slice_list:
                passed, _ = self.quality_filter.evaluate_slice(slice_img)
                if passed:
                    slice_resized = self.resizer.resize_2d(slice_img, target_2d_shape)
                    valid_slices.append(slice_resized)

            if valid_slices:
                # Shape formatting for PyTorch 2D batch: (N_slices, Channel=1, Height, Width)
                axis_array = np.array(valid_slices, dtype=np.float32)
                axis_tensor = np.expand_dims(axis_array, axis=1)
                slices_tensors[axis] = axis_tensor

        metadata = {
            "source_file": os.path.basename(file_path),
            "original_shape": list(raw_volume.shape),
            "voxel_spacing": list(spacing),
            "cropped_shape": list(cropped_volume.shape),
            "volume_tensor_shape": list(volume_tensor.shape),
            "bounding_box": bbox_info,
            "extracted_slice_counts": {axis: arr.shape[0] for axis, arr in slices_tensors.items()},
        }

        logger.info(f"Inference preprocessing completed successfully for {file_path}.")
        return {
            "volume_tensor": volume_tensor,
            "slices_tensors": slices_tensors,
            "metadata": metadata,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = InferencePreprocessor()
    test_hdr = r"oasis_cross-sectional_disc1/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr"
    if os.path.exists(test_hdr):
        res = preprocessor.preprocess_file(test_hdr)
        print("Inference Preprocessor Verification:")
        print(f"3D Volume Tensor Shape: {res['volume_tensor'].shape}")
        for axis, tensor in res['slices_tensors'].items():
            print(f"Axis '{axis}' Tensor Shape: {tensor.shape}")
        print("Metadata:", json.dumps(res["metadata"], indent=2))
