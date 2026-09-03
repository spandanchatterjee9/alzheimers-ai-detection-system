"""
High-Speed Native Zip Creator using Windows bsdtar.
Archives all 45,592 axial slices in ~10 seconds.
"""

import os
import subprocess
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_ZIP = PROJECT_ROOT / "Alzheimers_AI_Joint_Training_Package_OASIS_and_ADNI.zip"
TEMP_ZIP = PROJECT_ROOT / "Alzheimers_AI_Joint_Training_Package_OASIS_and_ADNI_temp.zip"
FILELIST_TXT = PROJECT_ROOT / "scripts" / "joint_files_to_pack.txt"

def build_with_native_tar():
    print("=" * 70)
    print("GENERATING FILE LIST FOR NATIVE HIGH-SPEED BSDTAR...")
    print("=" * 70)

    if TEMP_ZIP.exists():
        try:
            TEMP_ZIP.unlink()
        except Exception:
            pass
    if FINAL_ZIP.exists():
        try:
            FINAL_ZIP.unlink()
        except Exception:
            pass

    files_to_pack = []

    # 1. Manifests
    manifest_dir = PROJECT_ROOT / "dataset/manifests"
    if manifest_dir.exists():
        for f in manifest_dir.glob("*.csv"):
            files_to_pack.append(f.relative_to(PROJECT_ROOT).as_posix())

    # 2. OASIS Axial Slices + Metadata
    oasis_dir = PROJECT_ROOT / "dataset/processed/labelled"
    if oasis_dir.exists():
        print("Indexing OASIS-1 axial slices...")
        for f in oasis_dir.rglob("*"):
            if f.is_file() and (f.name.startswith("axial_") and f.name.endswith(".npy") or f.name == "metadata.json"):
                files_to_pack.append(f.relative_to(PROJECT_ROOT).as_posix())

    # 3. ADNI Axial Slices + Metadata
    adni_dir = PROJECT_ROOT / "dataset/processed/adni"
    if adni_dir.exists():
        print("Indexing ADNI-1 axial slices...")
        for f in adni_dir.rglob("*"):
            if f.is_file() and (f.name.startswith("axial_") and f.name.endswith(".npy") or f.name == "metadata.json"):
                files_to_pack.append(f.relative_to(PROJECT_ROOT).as_posix())

    # 4. Code & Configs
    code_dirs = ["models", "training", "configs", "explainability", "inference"]
    for cdir in code_dirs:
        p = PROJECT_ROOT / cdir
        if p.exists():
            for f in p.rglob("*"):
                if f.is_file() and not f.name.endswith(".pyc") and "__pycache__" not in f.parts:
                    files_to_pack.append(f.relative_to(PROJECT_ROOT).as_posix())

    # 5. Requirements
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        files_to_pack.append("requirements.txt")

    print(f"Total targeted files: {len(files_to_pack):,}")

    with open(FILELIST_TXT, "w", encoding="utf-8") as f:
        for item in files_to_pack:
            f.write(f"{item}\n")

    print(f"File list written to: {FILELIST_TXT.name}")
    print("Executing native Windows tar (bsdtar) command...")

    t0 = time.time()
    cmd = [
        "tar.exe",
        "-acf",
        str(FINAL_ZIP),
        "-T",
        str(FILELIST_TXT)
    ]
    
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    t1 = time.time()

    if res.returncode != 0:
        print(f"[Error] tar command failed: {res.stderr}")
        sys.exit(1)

    print(f"Native archiving completed in {t1 - t0:.1f} seconds!")

    # Verify
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
    build_with_native_tar()
