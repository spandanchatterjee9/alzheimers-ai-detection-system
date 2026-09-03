"""
Comprehensive Redundancy and Subject Overlap Audit for ADNI3 and ADNI2.
"""

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def audit():
    adni3_zips = sorted([f for f in (PROJECT_ROOT / "ADNI3/ADNI3").glob("*.zip") if not f.name.startswith("._")])
    adni2_zips = sorted([f for f in (PROJECT_ROOT / "ADNI2/ADNI2").glob("*.zip") if not f.name.startswith("._")])

    print("=" * 70)
    print("AUDITING ADNI3 AND ADNI2 ARCHIVES...")
    print("=" * 70)
    print(f"ADNI3 Zip Count: {len(adni3_zips)}")
    print(f"ADNI2 Zip Count: {len(adni2_zips)}")

    adni3_patients = set()
    adni3_scans = {}
    adni3_zip_info = {}

    for zf_path in adni3_zips:
        try:
            with zipfile.ZipFile(zf_path, "r") as zf:
                infolist = zf.infolist()
                patients_in_this_zip = set()
                for info in infolist:
                    parts = Path(info.filename).parts
                    for p in parts:
                        if len(p.split("_")) >= 3 and (p.startswith("0") or p.startswith("1") or p.startswith("9")):
                            patients_in_this_zip.add(p)
                            adni3_patients.add(p)
                            if p not in adni3_scans:
                                adni3_scans[p] = set()
                            # capture scan/sequence
                            if len(parts) >= 3:
                                adni3_scans[p].add("/".join(parts[:3]))
                adni3_zip_info[zf_path.name] = {
                    "size_mb": zf_path.stat().st_size / (1024**2),
                    "files": len(infolist),
                    "patients": sorted(list(patients_in_this_zip))
                }
        except Exception as e:
            adni3_zip_info[zf_path.name] = {"error": str(e)}

    print("\nADNI3 ZIP BREAKDOWN:")
    for name, info in adni3_zip_info.items():
        mb = info.get("size_mb", 0)
        fc = info.get("files", 0)
        pts = info.get("patients", [])
        print(f"  * {name:30s} | {mb:6.1f} MB | {fc:5d} DICOMs | {len(pts)} Patients: {pts[:2]}...")

    print(f"\nTotal Distinct Patients in ADNI3: {len(adni3_patients)}")
    print("Patient IDs in ADNI3:", sorted(list(adni3_patients)))

    # Already preprocessed dataset
    processed_dir = PROJECT_ROOT / "dataset/processed/adni"
    processed_patients = set([d.name for d in processed_dir.iterdir() if d.is_dir()]) if processed_dir.exists() else set()
    print(f"\nTotal Patients already in dataset/processed/adni: {len(processed_patients)}")
    print("Patient IDs already processed:", sorted(list(processed_patients)))

    overlap = adni3_patients.intersection(processed_patients)
    new_patients = adni3_patients - processed_patients

    print("\n" + "=" * 70)
    print("REDUNDANCY AUDIT VERDICT:")
    print("=" * 70)
    print(f"Total Patients in ADNI3:                        {len(adni3_patients)}")
    print(f"Redundant Patients (Already in Training Data):  {len(overlap)} ({(len(overlap)/max(len(adni3_patients),1))*100:.1f}%)")
    print(f"Brand-New Distinct Patients in ADNI3:           {len(new_patients)}")
    
    if new_patients:
        print(f"\n[NEW PATIENTS FOUND]: {sorted(list(new_patients))}")
    else:
        print("\n[VERDICT]: 100% of the patient subjects in ADNI3 are already present in the existing ADNI dataset!")

    # Check if there are new scan dates/visits for existing patients
    print("\nChecking for new scan visits/dates...")
    total_scans = sum(len(scans) for scans in adni3_scans.values())
    print(f"Total 3D Scan Sequences in ADNI3: {total_scans}")

if __name__ == "__main__":
    audit()
