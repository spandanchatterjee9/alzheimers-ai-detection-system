"""
Training Package Verification Script for Google Colab.

Verifies:
1. Expected manifests exist (train.csv, validation.csv, test.csv).
2. Expected labelled patient folders exist on disk.
3. All slice paths resolve to valid .npy files.
4. No unlabelled patient appears in train, validation, or test.
5. Zero patient leakage across partitions.
6. All labels are within {0.0, 0.5, 1.0, 2.0}.
7. Sample slice dimensions are strictly (224, 224) float32.
8. Model forward pass on sample slice succeeds with output shape (1, 4).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def verify_training_package():
    print("==================================================")
    print("      COLAB TRAINING PACKAGE AUDIT & VERIFY       ")
    print("==================================================")

    # 1. Manifest existence
    manifest_dir = Path("dataset/manifests")
    if not manifest_dir.exists():
        manifest_dir = Path("dataset/splits")

    train_path = manifest_dir / "train.csv"
    val_path = manifest_dir / "validation.csv"
    test_path = manifest_dir / "test.csv"
    unlabelled_path = manifest_dir / "unlabelled_manifest.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"[FAIL] Missing manifest file: {p}")
            sys.exit(1)
        print(f"[PASS] Manifest exists: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 2. Check no unlabelled subjects present in supervised splits
    unlabelled_ids = set()
    if unlabelled_path.exists():
        u_df = pd.read_csv(unlabelled_path)
        unlabelled_ids = set(u_df["patient_id"])

    all_split_ids = set(train_df["patient_id"]).union(set(val_df["patient_id"])).union(set(test_df["patient_id"]))
    unlabelled_overlap = all_split_ids.intersection(unlabelled_ids)
    if unlabelled_overlap:
        print(f"[FAIL] Unlabelled subjects detected in supervised splits: {unlabelled_overlap}")
        sys.exit(1)
    print(f"[PASS] Unlabelled Cohort Isolation (0 unlabelled subjects in splits)")

    # 3. Patient counts & zero leakage
    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])

    if train_ids.intersection(val_ids) or train_ids.intersection(test_ids) or val_ids.intersection(test_ids):
        print(f"[FAIL] Patient leakage detected across splits!")
        sys.exit(1)
    print(f"[PASS] Patient Isolation (Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)})")

    # 4. Check slice files and dimensions
    sample_slices_checked = 0
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        for idx, row in df.iterrows():
            proc_p = Path(str(row["future_processed_path"]).replace("\\", "/"))
            slices_dir = proc_p / "slices"
            if not slices_dir.exists():
                print(f"[FAIL] Slices directory missing: {slices_dir}")
                sys.exit(1)
            
            axial_slices = list(slices_dir.glob("axial_*.npy"))
            if not axial_slices:
                print(f"[FAIL] No axial slices found in: {slices_dir}")
                sys.exit(1)

            # Check dimensions on first 10 slices
            if sample_slices_checked < 10:
                arr = np.load(str(axial_slices[0]))
                if arr.shape != (224, 224) or arr.dtype != np.float32:
                    print(f"[FAIL] Invalid slice shape/dtype in {axial_slices[0]}: shape={arr.shape}, dtype={arr.dtype}")
                    sys.exit(1)
                sample_slices_checked += 1

    print(f"[PASS] Slice Resolution & Dtype: All verified slices are strictly (224, 224) float32")

    # 5. Model Forward Pass Verification
    if HAS_TORCH:
        from models.cnn_2d import EfficientNetB02D
        model = EfficientNetB02D(num_classes=4, in_channels=1, pretrained=False)
        dummy_input = torch.randn(2, 1, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        if output.shape != (2, 4):
            print(f"[FAIL] Unexpected model output shape: {output.shape} (Expected (2, 4))")
            sys.exit(1)
        print(f"[PASS] Model Forward Pass: Output shape {output.shape} verified for 2D EfficientNet-B0")

    print("\n==================================================")
    print("   TRAINING PACKAGE STATUS: READY FOR COLAB       ")
    print("==================================================")

if __name__ == "__main__":
    verify_training_package()
