"""
Audit script for ADNI2, ADNI3, and ADNI4 folders.
"""

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def scan_adni_folder(folder_name):
    folder_path = PROJECT_ROOT / folder_name
    if not folder_path.exists():
        return {}, set(), 0

    zip_files = [f for f in folder_path.rglob("*.zip") if not f.name.startswith("._")]
    patients = set()
    total_bytes = 0
    zip_info = {}

    for zf_path in zip_files:
        try:
            total_bytes += zf_path.stat().st_size
            with zipfile.ZipFile(zf_path, "r") as zf:
                pats_in_zip = set()
                names = zf.namelist()
                for n in names:
                    parts = Path(n).parts
                    for p in parts:
                        if len(p.split("_")) >= 3 and (p.startswith("0") or p.startswith("1") or p.startswith("9")):
                            pats_in_zip.add(p)
                            patients.add(p)
                zip_info[zf_path.name] = {
                    "size_mb": zf_path.stat().st_size / (1024**2),
                    "files": len(names),
                    "patients": sorted(list(pats_in_zip))
                }
        except Exception as e:
            zip_info[zf_path.name] = {"error": str(e)}

    return zip_info, patients, total_bytes

def main():
    info2, pats2, bytes2 = scan_adni_folder("ADNI2")
    info3, pats3, bytes3 = scan_adni_folder("ADNI3")
    info4, pats4, bytes4 = scan_adni_folder("ADNI4")

    print("=" * 70)
    print("MULTI-ADNI DATASET AUDIT (ADNI2 vs ADNI3 vs ADNI4)")
    print("=" * 70)
    print(f"ADNI2: {len(info2)} zip archives | {bytes2/(1024**3):.2f} GB | {len(pats2)} unique patients")
    print(f"ADNI3: {len(info3)} zip archives | {bytes3/(1024**3):.2f} GB | {len(pats3)} unique patients")
    print(f"ADNI4: {len(info4)} zip archives | {bytes4/(1024**3):.2f} GB | {len(pats4)} unique patients")

    all_adni_patients = pats2.union(pats3).union(pats4)
    print("\n" + "-" * 70)
    print(f"TOTAL DISTINCT INDEPENDENT PATIENTS (ADNI): {len(all_adni_patients)}")
    print("-" * 70)

    # OASIS-1
    oasis_dir = PROJECT_ROOT / "dataset/processed/labelled"
    oasis_patients = set([d.name for d in oasis_dir.iterdir() if d.is_dir()]) if oasis_dir.exists() else set()
    print(f"OASIS-1 Independent Patients:               {len(oasis_patients)}")
    print(f"TOTAL COMBINED INDEPENDENT PATIENTS:        {len(oasis_patients) + len(all_adni_patients)}")

    print("\nOVERLAP ANALYSIS BETWEEN FOLDERS:")
    pats_2_3 = pats2.union(pats3)
    new_in_4 = pats4 - pats_2_3
    print(f"  - ADNI4 Total Patients:                   {len(pats4)}")
    print(f"  - ADNI4 Patients matching ADNI2/ADNI3:    {len(pats4.intersection(pats_2_3))}")
    print(f"  - ADNI4 Brand-New Unique Patients:        {len(new_in_4)}")

    print("\nADNI4 Archive Breakdown:")
    for name, z_info in info4.items():
        mb = z_info.get("size_mb", 0)
        fc = z_info.get("files", 0)
        pts = z_info.get("patients", [])
        print(f"  * {name:35s} | {mb:6.1f} MB | {fc:4d} files | Patients ({len(pts)}): {pts[:2]}...")

if __name__ == "__main__":
    main()
