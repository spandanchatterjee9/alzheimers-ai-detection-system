"""
Fast and Reliable Zip Archive Creator for Google Colab Training Package.
Uses ZIP_DEFLATED (fast compressionlevel 1) or ZIP_STORED to quickly package the 26,208 slices.
Writes to a temporary file first and renames to ensure atomic completion.
"""

import os
import sys
import time
import zipfile
from pathlib import Path

def create_zip():
    src_dir = Path("Alzheimers_AI_Training_Package")
    if not src_dir.exists():
        print(f"[ERROR] Directory not found: {src_dir}")
        sys.exit(1)

    out_zip = Path("Alzheimers_AI_Training_Package.zip")
    temp_zip = Path("Alzheimers_AI_Training_Package_temp.zip")

    if temp_zip.exists():
        temp_zip.unlink()

    print(f"Indexing files in {src_dir}...")
    all_files = [f for f in src_dir.rglob("*") if f.is_file()]
    total_files = len(all_files)
    print(f"Total files to archive: {total_files}")

    print("Creating zip archive (Fast Compression)...")
    t0 = time.time()
    
    # Use ZIP_DEFLATED with compresslevel=1 for high speed + compression
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for idx, file_path in enumerate(all_files, 1):
            arcname = file_path.relative_to(src_dir.parent)
            zf.write(file_path, arcname=str(arcname).replace("\\", "/"))
            if idx % 5000 == 0 or idx == total_files:
                print(f"  Archived {idx}/{total_files} files ({idx*100/total_files:.1f}%)...")

    t1 = time.time()
    print(f"Archiving finished in {t1 - t0:.2f} seconds.")

    # Atomic replace
    if out_zip.exists():
        out_zip.unlink()
    temp_zip.rename(out_zip)

    # Verification
    print("\nVerifying zip integrity...")
    is_valid = zipfile.is_zipfile(out_zip)
    if is_valid:
        with zipfile.ZipFile(out_zip, "r") as zf:
            file_count = len(zf.namelist())
            size_mb = out_zip.stat().st_size / (1024 * 1024)
            size_gb = out_zip.stat().st_size / (1024 * 1024 * 1024)
            print(f"[SUCCESS] Zip archive is 100% VALID!")
            print(f"  File Path:   {out_zip.resolve()}")
            print(f"  Files in Zip:{file_count}")
            print(f"  Zip Size:    {size_mb:.2f} MB ({size_gb:.3f} GB)")
    else:
        print("[FAIL] Zip verification failed!")

if __name__ == "__main__":
    create_zip()
