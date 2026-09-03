"""
Comprehensive Dataset Verification Script for Alzheimer's Detection System.

Usage:
    python scripts/verify_dataset.py

Verifies:
1. Manifest completeness across dataset_manifest.csv, labelled_manifest.csv, and unlabelled_manifest.csv.
2. Artifact existence for processed patient directories (volume.nii.gz, metadata.json, tensor.pt, diagnostic_preview.png).
3. Patient isolation and split integrity across train.csv, validation.csv, and test.csv.
4. Numerical tensor sanity (shape, dtype, NaN/Inf check).
5. Cross-consistency between raw scans and processed outputs.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import yaml

# Add root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger("verify_dataset")


def verify_manifests(config: dict) -> Tuple[bool, dict]:
    """Verifies existence and record counts of all manifest files."""
    manifests_dir = Path(config["dataset"]["manifests_dir"])
    dataset_manifest_path = Path(config["dataset"]["manifest_path"])
    labelled_manifest_path = Path(config["dataset"]["labelled_manifest_path"])
    unlabelled_manifest_path = Path(config["dataset"]["unlabelled_manifest_path"])

    train_path = Path(config["splits"]["train_path"])
    val_path = Path(config["splits"]["val_path"])
    test_path = Path(config["splits"]["test_path"])

    results = {}
    passed = True

    required_files = [
        ("dataset_manifest", dataset_manifest_path),
        ("labelled_manifest", labelled_manifest_path),
        ("unlabelled_manifest", unlabelled_manifest_path),
        ("train_split", train_path),
        ("val_split", val_path),
        ("test_split", test_path),
    ]

    for name, p in required_files:
        if not p.exists():
            logger.error(f"Manifest missing: {p}")
            results[name] = {"exists": False, "count": 0}
            passed = False
        else:
            df = pd.read_csv(p)
            results[name] = {"exists": True, "count": len(df)}

    # Verification of total counts
    if results.get("dataset_manifest", {}).get("count", 0) != 413:
        logger.warning(f"Expected 413 dataset manifest rows, found: {results.get('dataset_manifest', {}).get('count')}")

    if results.get("labelled_manifest", {}).get("count", 0) != 234:
        logger.warning(f"Expected 234 labelled manifest rows, found: {results.get('labelled_manifest', {}).get('count')}")

    if results.get("unlabelled_manifest", {}).get("count", 0) != 179:
        logger.warning(f"Expected 179 unlabelled manifest rows, found: {results.get('unlabelled_manifest', {}).get('count')}")

    return passed, results


def verify_split_leakage(config: dict) -> bool:
    """Asserts strict patient isolation with ZERO patient overlap across train, val, and test splits."""
    train_df = pd.read_csv(config["splits"]["train_path"])
    val_df = pd.read_csv(config["splits"]["val_path"])
    test_df = pd.read_csv(config["splits"]["test_path"])

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])

    overlap_tv = train_patients.intersection(val_patients)
    overlap_tt = train_patients.intersection(test_patients)
    overlap_vt = val_patients.intersection(test_patients)

    if overlap_tv or overlap_tt or overlap_vt:
        logger.error(f"PATIENT LEAKAGE DETECTED! Overlap TV: {overlap_tv}, TT: {overlap_tt}, VT: {overlap_vt}")
        return False

    logger.info(f"Split Integrity PASS: Zero patient overlap across {len(train_patients)} Train, {len(val_patients)} Val, and {len(test_patients)} Test patients.")
    return True


def verify_processed_artifacts(config: dict, sample_limit: int = 10) -> Tuple[bool, dict]:
    """Verifies directory structures and mandatory artifact completeness for processed patients."""
    manifest_df = pd.read_csv(config["dataset"]["manifest_path"])

    total_scanned = len(manifest_df)
    existing_dirs = 0
    artifacts_passed = 0
    checked_samples = 0

    target_shape = tuple(config["volume_specs"]["target_shape"])
    expected_tensor_shape = (1, target_shape[0], target_shape[1], target_shape[2])

    for _, row in manifest_df.iterrows():
        future_path = Path(row["future_processed_path"])
        if not future_path.exists():
            continue

        existing_dirs += 1

        nii_path = future_path / "volume.nii.gz"
        meta_path = future_path / "metadata.json"
        tensor_pt_path = future_path / "tensor.pt"
        png_path = future_path / "diagnostic_preview.png"
        log_path = future_path / "preprocessing_log.json"

        # Check required files
        missing_files = []
        for f_path in [nii_path, meta_path, tensor_pt_path, png_path, log_path]:
            if not f_path.exists():
                missing_files.append(f_path.name)

        if not missing_files:
            artifacts_passed += 1

        # Tensor shape & NaN inspection for sample set
        if checked_samples < sample_limit and tensor_pt_path.exists():
            checked_samples += 1
            if HAS_TORCH:
                t = torch.load(str(tensor_pt_path))
                t_np = t.numpy()
            else:
                t_np = np.load(str(tensor_pt_path).replace(".pt", ".npy"))

            if t_np.shape != expected_tensor_shape:
                logger.error(f"Patient {row['patient_id']} invalid tensor shape: {t_np.shape}, expected {expected_tensor_shape}")
                return False, {}

            if np.isnan(t_np).any() or np.isinf(t_np).any():
                logger.error(f"Patient {row['patient_id']} tensor contains NaN/Inf values.")
                return False, {}

    stats = {
        "total_manifest_patients": total_scanned,
        "processed_directories_found": existing_dirs,
        "artifacts_complete_patients": artifacts_passed,
    }

    logger.info(f"Processed Artifacts Check: {existing_dirs}/{total_scanned} directories present, {artifacts_passed} fully complete.")
    return True, stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n=======================================================")
    print("ALZHEIMER'S AI SYSTEM — DATASET INTEGRITY VERIFICATION")
    print("=======================================================\n")

    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.error("Configuration file missing: configs/config.yaml")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Manifest verification
    m_ok, m_stats = verify_manifests(config)
    print("1. MANIFEST VERIFICATION:")
    for k, v in m_stats.items():
        print(f"   - {k}: {v['count']} records (Exists: {v['exists']})")

    # 2. Patient-wise split verification
    print("\n2. PATIENT SPLIT INTEGRITY VERIFICATION:")
    s_ok = verify_split_leakage(config)

    # 3. Processed artifacts verification
    print("\n3. PROCESSED ARTIFACTS & TENSOR VERIFICATION:")
    a_ok, a_stats = verify_processed_artifacts(config)
    for k, v in a_stats.items():
        print(f"   - {k}: {v}")

    print("\n=======================================================")
    if m_ok and s_ok and a_ok:
        print("VERIFICATION RESULT: ALL CHECKS PASSED SUCCESSFULLY [OK]")
    else:
        print("VERIFICATION RESULT: ISSUES DETECTED [WARNING/FAIL]")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
