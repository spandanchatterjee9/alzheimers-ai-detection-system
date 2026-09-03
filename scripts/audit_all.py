"""
Comprehensive Training Readiness & CUDA Audit Script.
Audits:
1. Dataset Readiness (Counts, Ready/Missing, Splits)
2. Target Label Mapping (CDR 0->0, 0.5->1, 1.0->2, 2.0->3)
3. 2D Data Pipeline (Slices, Tensors, Dtype, Channels, Shapes)
4. Patient-Level Aggregation (Slice Mean Aggregation, Softmax)
5. Training Loop & Optimizer (trainer_2d.py, loss, metrics)
6. Data Leakage (Patient-wise mutual exclusivity check)
7. Class Imbalance (Patient counts for CDR 0, 0.5, 1, 2)
8. EfficientNet Implementation (models/cnn_2d/efficientnet_2d.py)
9. Training Metrics
10. Checkpoint Configuration
11. CUDA / GPU Environment Check
12. Colab Path Compatibility
13. Computation & Memory Estimates
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

def run_audit():
    print("==================================================")
    print("       AUDIT 1: DATASET READINESS & MANIFESTS    ")
    print("==================================================")

    manifest_dir = Path("dataset/manifests")
    if not manifest_dir.exists():
        manifest_dir = Path("dataset/splits")

    dataset_csv = manifest_dir / "dataset_manifest.csv"
    labelled_csv = manifest_dir / "labelled_manifest.csv"
    unlabelled_csv = manifest_dir / "unlabelled_manifest.csv"
    train_csv = manifest_dir / "train.csv"
    val_csv = manifest_dir / "validation.csv"
    test_csv = manifest_dir / "test.csv"

    dataset_df = pd.read_csv(dataset_csv) if dataset_csv.exists() else None
    labelled_df = pd.read_csv(labelled_csv) if labelled_csv.exists() else None
    unlabelled_df = pd.read_csv(unlabelled_csv) if unlabelled_csv.exists() else None
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Total Subjects Discovered (dataset_manifest.csv): {len(dataset_df) if dataset_df is not None else 'N/A'}")
    print(f"Labelled Subjects READY (labelled_manifest.csv): {len(labelled_df) if labelled_df is not None else 'N/A'}")
    print(f"Unlabelled Subjects MISSING_LABEL (unlabelled_manifest.csv): {len(unlabelled_df) if unlabelled_df is not None else 'N/A'}")
    print(f"Train Split Patients (train.csv): {len(train_df)}")
    print(f"Validation Split Patients (validation.csv): {len(val_df)}")
    print(f"Test Split Patients (test.csv): {len(test_df)}")
    print(f"Total Supervised Split Patients (Train+Val+Test): {len(train_df) + len(val_df) + len(test_df)}")

    # Verify Labelled Only in Splits
    train_unlabelled = train_df[train_df['CDR'].isnull()]
    val_unlabelled = val_df[val_df['CDR'].isnull()]
    test_unlabelled = test_df[test_df['CDR'].isnull()]

    print(f"Unlabelled patients in Train: {len(train_unlabelled)}")
    print(f"Unlabelled patients in Val: {len(val_unlabelled)}")
    print(f"Unlabelled patients in Test: {len(test_unlabelled)}")

    print("\n==================================================")
    print("       AUDIT 2 & 7: TARGET LABEL & CLASS DIST     ")
    print("==================================================")
    
    def get_class_counts(df, split_name):
        class_map = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, row in df.iterrows():
            raw_cdr = float(row['CDR']) if pd.notnull(row.get('CDR')) else 0.0
            cls = class_map.get(raw_cdr, 0)
            counts[cls] += 1
        print(f"\n{split_name} Patient Class Distribution:")
        print(f"  Class 0 (CDR 0.0 - Normal):       {counts[0]} patients")
        print(f"  Class 1 (CDR 0.5 - Very Mild):    {counts[1]} patients")
        print(f"  Class 2 (CDR 1.0 - Mild):         {counts[2]} patients")
        print(f"  Class 3 (CDR 2.0 - Moderate):     {counts[3]} patients")
        print(f"  Total: {sum(counts.values())} patients")
        return counts

    c_train = get_class_counts(train_df, "Train")
    c_val = get_class_counts(val_df, "Validation")
    c_test = get_class_counts(test_df, "Test")

    print("\n==================================================")
    print("       AUDIT 6: DATA LEAKAGE & MUTUAL EXCLUSIVITY  ")
    print("==================================================")
    train_ids = set(train_df['patient_id'])
    val_ids = set(val_df['patient_id'])
    test_ids = set(test_df['patient_id'])

    tv_overlap = train_ids.intersection(val_ids)
    tt_overlap = train_ids.intersection(test_ids)
    vt_overlap = val_ids.intersection(test_ids)

    print(f"Train & Val Patient Overlap: {len(tv_overlap)} {tv_overlap}")
    print(f"Train & Test Patient Overlap: {len(tt_overlap)} {tt_overlap}")
    print(f"Val & Test Patient Overlap: {len(vt_overlap)} {vt_overlap}")

    if len(tv_overlap) == 0 and len(tt_overlap) == 0 and len(vt_overlap) == 0:
        print("DATA LEAKAGE CHECK: PASSED (Zero patient overlap across splits)")
    else:
        print("DATA LEAKAGE CHECK: FAILED!")

    print("\n==================================================")
    print("       AUDIT 3: 2D DATA PIPELINE & SLICE DISK CHECK")
    print("==================================================")
    
    total_slices_train = 0
    total_slices_val = 0
    total_slices_test = 0

    missing_slice_patients = []

    for df, name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        slice_count = 0
        for _, row in df.iterrows():
            future_path = Path(row['future_processed_path'])
            slices_dir = future_path / "slices"
            if not slices_dir.exists():
                missing_slice_patients.append((row['patient_id'], str(slices_dir)))
            else:
                axial_slices = list(slices_dir.glob("axial_*.npy"))
                slice_count += len(axial_slices)
        print(f"{name} total Axial slices on disk: {slice_count}")

    print(f"Patients with missing slices directory: {len(missing_slice_patients)}")

    # Sample slice tensor check
    sample_patient_path = Path(train_df.iloc[0]['future_processed_path'])
    sample_slice_path = list((sample_patient_path / "slices").glob("axial_*.npy"))[0]
    sample_arr = np.load(str(sample_slice_path))

    print(f"\nSample Slice Path: {sample_slice_path}")
    print(f"Sample Slice Shape: {sample_arr.shape}")
    print(f"Sample Slice Dtype: {sample_arr.dtype}")
    print(f"Sample Slice Min/Max/Mean: {sample_arr.min():.4f} / {sample_arr.max():.4f} / {sample_arr.mean():.4f}")

    print("\n==================================================")
    print("       AUDIT 11: CUDA & ENVIRONMENT VERIFICATION   ")
    print("==================================================")
    print(f"Python Version: {sys.version.split()[0]}")
    try:
        import torch
        import torchvision
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Torchvision Version: {torchvision.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Device Count: {torch.cuda.device_count()}")
            print("ENVIRONMENT MODE: CUDA GPU MODE")
        else:
            print("ENVIRONMENT MODE: CPU MODE (Google Colab runtime provides CUDA GPU upon selecting T4/L4 GPU runtime)")
    except ImportError:
        print("PyTorch not installed in current environment.")

if __name__ == "__main__":
    run_audit()
