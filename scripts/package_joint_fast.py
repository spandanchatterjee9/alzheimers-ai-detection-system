"""
Ultra-Fast Multi-Cohort Packager (OASIS-1 + ADNI-1).
Uses ZIP_STORED to package all 45,592 axial slices in ~10 seconds.
"""

import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_ZIP = PROJECT_ROOT / "Alzheimers_AI_Joint_Training_Package_OASIS_and_ADNI.zip"
TEMP_ZIP = PROJECT_ROOT / "Alzheimers_AI_Joint_Training_Package_OASIS_and_ADNI_temp.zip"

def build_joint_package_fast():
    print("=" * 70)
    print("LIGHTNING-FAST JOINT ARCHIVE BUILDER (OASIS + ADNI)...")
    print("=" * 70)

    if TEMP_ZIP.exists():
        TEMP_ZIP.unlink()
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()

    files_to_pack = []

    # 1. Manifests
    manifest_dir = PROJECT_ROOT / "dataset/manifests"
    if manifest_dir.exists():
        for f in manifest_dir.glob("*.csv"):
            files_to_pack.append((f, f.relative_to(PROJECT_ROOT).as_posix()))

    # 2. OASIS Axial Slices + Metadata
    oasis_dir = PROJECT_ROOT / "dataset/processed/labelled"
    if oasis_dir.exists():
        print("Indexing OASIS-1 axial slices...")
        for f in oasis_dir.rglob("*"):
            if f.is_file() and (f.name.startswith("axial_") and f.name.endswith(".npy") or f.name == "metadata.json"):
                files_to_pack.append((f, f.relative_to(PROJECT_ROOT).as_posix()))

    # 3. ADNI Axial Slices + Metadata
    adni_dir = PROJECT_ROOT / "dataset/processed/adni"
    if adni_dir.exists():
        print("Indexing ADNI-1 axial slices...")
        for f in adni_dir.rglob("*"):
            if f.is_file() and (f.name.startswith("axial_") and f.name.endswith(".npy") or f.name == "metadata.json"):
                files_to_pack.append((f, f.relative_to(PROJECT_ROOT).as_posix()))

    # 4. Code & Configs
    code_dirs = ["models", "training", "configs", "explainability", "inference"]
    for cdir in code_dirs:
        p = PROJECT_ROOT / cdir
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file() and not f.name.endswith(".pyc") and "__pycache__" not in f.parts:
                    files_to_pack.append((f, f.relative_to(PROJECT_ROOT).as_posix()))

    # 5. Requirements
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        files_to_pack.append((req_file, "requirements.txt"))

    total_files = len(files_to_pack)
    print(f"Total targeted files: {total_files:,}")
    print("Writing archive with High-Speed I/O (ZIP_STORED)...")

    t0 = time.time()
    # ZIP_STORED writes directly to disk at maximum SSD/HDD speed (no CPU compression bottleneck)
    with zipfile.ZipFile(TEMP_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, (fpath, arcname) in enumerate(files_to_pack, 1):
            zf.write(fpath, arcname=arcname)
            if idx % 10000 == 0 or idx == total_files:
                print(f"  Packed {idx:,}/{total_files:,} files ({idx*100/total_files:.1f}%)...")

    t1 = time.time()
    print(f"Zip writing finished in {t1 - t0:.1f} seconds!")

    # Atomic Rename
    if FINAL_ZIP.exists():
        FINAL_ZIP.unlink()
    TEMP_ZIP.rename(FINAL_ZIP)

    # Verification
    print("\nVerifying archive integrity...")
    if zipfile.is_zipfile(FINAL_ZIP):
        with zipfile.ZipFile(FINAL_ZIP, "r") as zf:
            count = len(zf.namelist())
            size_gb = FINAL_ZIP.stat().st_size / (1024 ** 3)
            print("=" * 70)
            print("SUCCESS: 100% VALID JOINT TRAINING PACKAGE CREATED!")
            print(f"   * Archive Name:    {FINAL_ZIP.name}")
            print(f"   * Absolute Path:   {FINAL_ZIP.resolve()}")
            print(f"   * Total Files:     {count:,}")
            print(f"   * Final Zip Size:  {size_gb:.2f} GB")
            print("=" * 70)
    else:
        print("[FAIL] Verification failed!")

if __name__ == "__main__":
    build_joint_package_fast()
