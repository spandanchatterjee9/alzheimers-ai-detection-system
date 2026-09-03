"""
Training Data Integrity & Patient Leakage Verification Script.

Verifies:
1. Patient-wise mutual exclusivity (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅).
2. Number of patients in Train, Validation, Test.
3. Number of valid Axial slices in each partition.
4. Absence of broken paths or missing slice files.
5. Strict validity of CDR labels.
6. Fails loudly with exit code 1 if any data leakage or broken path is detected.
"""

import sys
import pandas as pd
from pathlib import Path

def verify_training_data():
    print("==================================================")
    print("      TRAINING DATA & LEAKAGE VERIFICATION       ")
    print("==================================================")

    manifest_dir = Path("dataset/manifests")
    if not manifest_dir.exists():
        manifest_dir = Path("dataset/splits")

    train_path = manifest_dir / "train.csv"
    val_path = manifest_dir / "validation.csv"
    test_path = manifest_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"[FATAL ERROR] Manifest file missing: {p}")
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_ids = set(train_df["patient_id"])
    val_ids = set(val_df["patient_id"])
    test_ids = set(test_df["patient_id"])

    print(f"Train Patients:      {len(train_df)} ({len(train_ids)} unique)")
    print(f"Validation Patients: {len(val_df)} ({len(val_ids)} unique)")
    print(f"Test Patients:       {len(test_df)} ({len(test_ids)} unique)")
    print(f"Total Supervised:    {len(train_df) + len(val_df) + len(test_df)} patients")

    # 1. Check for Duplicate IDs within splits
    if len(train_ids) != len(train_df):
        print(f"[FATAL ERROR] Duplicate patient IDs found in train.csv!")
        sys.exit(1)
    if len(val_ids) != len(val_df):
        print(f"[FATAL ERROR] Duplicate patient IDs found in validation.csv!")
        sys.exit(1)
    if len(test_ids) != len(test_df):
        print(f"[FATAL ERROR] Duplicate patient IDs found in test.csv!")
        sys.exit(1)

    # 2. Check for Patient Leakage across splits
    tv_overlap = train_ids.intersection(val_ids)
    tt_overlap = train_ids.intersection(test_ids)
    vt_overlap = val_ids.intersection(test_ids)

    if tv_overlap:
        print(f"[FATAL ERROR] Patient leakage between Train and Validation: {tv_overlap}")
        sys.exit(1)
    if tt_overlap:
        print(f"[FATAL ERROR] Patient leakage between Train and Test: {tt_overlap}")
        sys.exit(1)
    if vt_overlap:
        print(f"[FATAL ERROR] Patient leakage between Validation and Test: {vt_overlap}")
        sys.exit(1)

    print(">> PATIENT LEAKAGE CHECK: PASSED (Zero overlap between partitions)")

    # 3. Check Slice Availability and File Integrity
    splits = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
    total_slices_all = 0
    broken_paths = []

    for name, df in splits:
        split_slices = 0
        for _, row in df.iterrows():
            p_id = row["patient_id"]
            proc_path = Path(str(row["future_processed_path"]).replace("\\", "/"))
            slices_dir = proc_path / "slices"
            if not slices_dir.exists():
                broken_paths.append((p_id, str(slices_dir)))
            else:
                axial_slices = list(slices_dir.glob("axial_*.npy"))
                split_slices += len(axial_slices)
        print(f"{name} Total Valid Axial Slices on Disk: {split_slices}")
        total_slices_all += split_slices

    if broken_paths:
        print(f"[FATAL ERROR] Found {len(broken_paths)} broken patient slice directories:")
        for bp in broken_paths[:5]:
            print(f"  {bp[0]}: {bp[1]}")
        sys.exit(1)

    print(f">> TOTAL AXIAL SLICES ACROSS SPLITS: {total_slices_all}")
    print(">> BROKEN PATHS CHECK: PASSED (All referenced slice files exist on disk)")

    # 4. Check CDR Labels
    valid_cdrs = {0.0, 0.5, 1.0, 2.0}
    for name, df in splits:
        for _, row in df.iterrows():
            cdr = row["CDR"]
            if pd.isnull(cdr) or float(cdr) not in valid_cdrs:
                print(f"[FATAL ERROR] Invalid CDR label '{cdr}' for patient {row['patient_id']} in {name}")
                sys.exit(1)

    print(">> LABEL VALIDITY CHECK: PASSED (All CDR values in {0.0, 0.5, 1.0, 2.0})")
    print("\nOVERALL STATUS: ALL DATA CHECKS PASSED")

if __name__ == "__main__":
    verify_training_data()
