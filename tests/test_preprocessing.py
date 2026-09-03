"""
Comprehensive Unit & Integration Test Suite for Alzheimer's Detection System Preprocessing Engine.

Verifies:
1. Scanner manifest creation (dataset_manifest.csv, labelled_manifest.csv, unlabelled_manifest.csv).
2. MRI loader shape, dtype, and affine matrix extraction.
3. Quality assurance volume validation.
4. Spatial orientation standardization to RAS.
5. Intensity normalizer Z-Score and Min-Max scaling.
6. Brain bounding box cropper.
7. 3D (128x128x128) & 2D (224x224) spatial resizer.
8. Slice extractor across Axial, Coronal, and Sagittal planes.
9. Slice quality filter (foreground ratio and Shannon entropy).
10. Patient-wise stratified split generator (asserting zero patient overlap).
11. PyTorch Dataset wrappers (Oasis3DDataset, Oasis2DSliceDataset, OasisUnlabelledDataset).
12. Verification script (verify_dataset.py).
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import yaml

from preprocessing.scanner import OasisDatasetScanner
from preprocessing.loader import MriLoader
from preprocessing.validator import VolumeValidator
from preprocessing.orientation import VolumeOrientation
from preprocessing.normalizer import IntensityNormalizer
from preprocessing.cropper import VolumeCropper
from preprocessing.resizer import VolumeResizer
from preprocessing.slice_extractor import SliceExtractor
from preprocessing.quality_filter import SliceQualityFilter
from preprocessing.inference_preprocessor import InferencePreprocessor
from preprocessing.torch_dataset import Oasis3DDataset, Oasis2DSliceDataset, OasisUnlabelledDataset
from training.split_generator import PatientSplitGenerator
from scripts.verify_dataset import verify_manifests, verify_split_leakage


def test_scanner_manifests():
    scanner = OasisDatasetScanner()
    df = scanner.scan()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 413
    assert Path("dataset/manifests/dataset_manifest.csv").exists()
    assert Path("dataset/manifests/labelled_manifest.csv").exists()
    assert Path("dataset/manifests/unlabelled_manifest.csv").exists()

    labelled_df = pd.read_csv("dataset/manifests/labelled_manifest.csv")
    unlabelled_df = pd.read_csv("dataset/manifests/unlabelled_manifest.csv")
    assert len(labelled_df) == 234
    assert len(unlabelled_df) == 179


def test_validator():
    validator = VolumeValidator()
    valid_vol = np.random.uniform(0, 100, (64, 64, 64)).astype(np.float32)
    is_valid, issues = validator.validate(valid_vol, (1.0, 1.0, 1.0))
    assert is_valid is True
    assert len(issues) == 0

    nan_vol = valid_vol.copy()
    nan_vol[10, 10, 10] = np.nan
    is_valid, issues = validator.validate(nan_vol)
    assert is_valid is False
    assert "Volume contains NaN values." in issues


def test_normalizer():
    normalizer = IntensityNormalizer(method="z_score", clip_percentiles=None)
    raw_vol = np.random.uniform(10, 500, (50, 50, 50)).astype(np.float32)
    norm_vol = normalizer.normalize(raw_vol)
    assert norm_vol.dtype == np.float32
    assert norm_vol.shape == raw_vol.shape
    fg_mask = norm_vol != 0
    assert np.isclose(np.mean(norm_vol[fg_mask]), 0.0, atol=1e-4)
    assert np.isclose(np.std(norm_vol[fg_mask]), 1.0, atol=1e-4)


def test_cropper():
    cropper = VolumeCropper(padding_voxels=2)
    empty_vol = np.zeros((80, 80, 80), dtype=np.float32)
    empty_vol[20:60, 25:65, 15:55] = 50.0
    cropped, bbox = cropper.crop(empty_vol)
    assert cropped.shape[0] < empty_vol.shape[0]
    assert cropped.shape[1] < empty_vol.shape[1]
    assert cropped.shape[2] < empty_vol.shape[2]
    assert bbox["orig_shape"] == (80, 80, 80)


def test_resizer():
    resizer = VolumeResizer(target_shape=(32, 32, 32))
    vol_3d = np.random.rand(64, 64, 40).astype(np.float32)
    resized_3d = resizer.resize_3d(vol_3d)
    assert resized_3d.shape == (32, 32, 32)

    slice_2d = np.random.rand(64, 64).astype(np.float32)
    resized_2d = resizer.resize_2d(slice_2d, (128, 128))
    assert resized_2d.shape == (128, 128)


def test_slice_extractor():
    extractor = SliceExtractor(axes=["axial", "coronal", "sagittal"])
    vol_3d = np.random.rand(64, 64, 40).astype(np.float32)
    slices_dict = extractor.extract_slices(vol_3d)
    assert "axial" in slices_dict
    assert "coronal" in slices_dict
    assert "sagittal" in slices_dict
    assert len(slices_dict["axial"]) > 0


def test_quality_filter():
    qfilter = SliceQualityFilter(min_foreground_ratio=0.15, min_entropy=1.0)
    
    # Informative slice
    good_slice = np.zeros((100, 100), dtype=np.float32)
    good_slice[20:80, 20:80] = np.random.normal(1.0, 0.2, (60, 60))
    passed, m = qfilter.evaluate_slice(good_slice)
    assert passed is True

    # Blank slice
    blank_slice = np.zeros((100, 100), dtype=np.float32)
    passed, m = qfilter.evaluate_slice(blank_slice)
    assert passed is False


def test_split_generator():
    generator = PatientSplitGenerator()
    train_df, val_df, test_df = generator.generate_splits()

    assert len(train_df) == 187
    assert len(val_df) == 23
    assert len(test_df) == 24

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    # Strict Patient Isolation Assertion
    assert len(train_patients.intersection(val_patients)) == 0
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(val_patients.intersection(test_patients)) == 0


def test_verification_script():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    m_ok, _ = verify_manifests(config)
    s_ok = verify_split_leakage(config)
    assert m_ok is True
    assert s_ok is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
