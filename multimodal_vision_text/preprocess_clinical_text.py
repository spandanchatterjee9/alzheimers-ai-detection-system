"""
Clinical Text Preprocessor & Dataset Builder for Alzheimer's Multi-Modal Pipeline.
Extracts only necessary clinical/demographic biomarkers from OASIS and ADNI,
synthesizes standardized clinical reports, and structures clean splits (train/val/test)
strictly synchronized with the vision model splits to prevent patient leakage.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from multimodal_vision_text.report_synthesizer import (
    synthesize_radiology_mri_report
)

MANIFESTS_DIR = PROJECT_ROOT / "dataset" / "manifests"
OUTPUT_DATA_DIR = PROJECT_ROOT / "multimodal_vision_text" / "data"


def cdr_to_3class(cdr: float) -> int:
    """Maps CDR float score to 3-class target: 0=CN, 1=MCI, 2=AD."""
    if cdr == 0.0:
        return 0  # CN
    elif cdr == 0.5:
        return 1  # MCI
    else:
        return 2  # AD (CDR 1.0, 2.0)


def build_clinical_split(manifest_csv: Path, split_name: str) -> pd.DataFrame:
    """Reads a split manifest, extracts biomarkers, and synthesizes clinical text."""
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    print(f"Processing {split_name}: {len(df)} patients loaded from {manifest_csv.name}")

    records = []
    for idx, row in df.iterrows():
        patient_id = str(row.get("patient_id", f"PATIENT_{idx}"))
        age = float(row["age"]) if pd.notnull(row.get("age")) else 75.0
        gender = str(row.get("gender", "Unknown"))
        cdr = float(row["CDR"]) if pd.notnull(row.get("CDR")) else 0.0
        mmse = float(row["MMSE"]) if pd.notnull(row.get("MMSE")) else 28.0
        
        nwbv = float(row["nwbv"]) if pd.notnull(row.get("nwbv")) else None
        etiv = float(row["etiv"]) if pd.notnull(row.get("etiv")) else None
        education = float(row["education"]) if pd.notnull(row.get("education")) else None
        
        raw_src = row.get("dataset_source")
        if pd.notnull(raw_src) and str(raw_src).strip() and str(raw_src) != "nan":
            dataset_src = str(raw_src).strip()
        else:
            dataset_src = "OASIS-1" if "OAS" in patient_id else "ADNI"
        label_3class = cdr_to_3class(cdr)
        label_name = "CN" if label_3class == 0 else "MCI" if label_3class == 1 else "AD"

        # Synthesize pure neuroradiology MRI anatomical description with referral indication (NO CDR, ZERO LEAK)
        radiology_report = synthesize_radiology_mri_report(
            patient_id=patient_id,
            age=age,
            gender=gender,
            mmse=mmse,
            nwbv=nwbv,
            etiv=etiv,
            dataset_source=dataset_src
        )

        records.append({
            "patient_id": patient_id,
            "short_id": str(row.get("short_id", patient_id)),
            "age": age,
            "gender": gender,
            "CDR": cdr,
            "label_3class": label_3class,
            "label_name": label_name,
            "MMSE": mmse,
            "nwbv": nwbv if nwbv is not None else np.nan,
            "etiv": etiv if etiv is not None else np.nan,
            "dataset_source": dataset_src,
            "clinical_report": radiology_report
        })

    out_df = pd.DataFrame(records)
    print(f"  -> {split_name} class distribution:")
    print(out_df["label_name"].value_counts().to_string())
    return out_df


def main():
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory initialized: {OUTPUT_DATA_DIR}")

    splits = [
        ("train", MANIFESTS_DIR / "joint_train.csv"),
        ("val", MANIFESTS_DIR / "joint_validation.csv"),
        ("test", MANIFESTS_DIR / "joint_test.csv"),
    ]

    all_dfs = {}
    for split_name, manifest_path in splits:
        split_df = build_clinical_split(manifest_path, split_name)
        out_csv = OUTPUT_DATA_DIR / f"{split_name}_clinical_text.csv"
        split_df.to_csv(out_csv, index=False)
        print(f"  Saved clean {split_name} text dataset -> {out_csv} ({len(split_df)} rows, {out_csv.stat().st_size / 1024:.1f} KB)\n")
        all_dfs[split_name] = split_df

    # Combine into unified master clinical text catalog
    master_df = pd.concat([
        all_dfs["train"].assign(split="train"),
        all_dfs["val"].assign(split="val"),
        all_dfs["test"].assign(split="test")
    ], ignore_index=True)
    
    master_csv = OUTPUT_DATA_DIR / "master_clinical_text.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"Master clinical text dataset saved -> {master_csv} ({len(master_df)} total patients)")
    print("\nSummary of Clinical Dataset Preprocessing:")
    print(f"  Total Patients: {len(master_df)}")
    print(f"  Disk Size: {master_csv.stat().st_size / 1024:.1f} KB (ultra-compact, no image bulk!)")
    print(f"  Classes: CN={sum(master_df['label_name']=='CN')}, MCI={sum(master_df['label_name']=='MCI')}, AD={sum(master_df['label_name']=='AD')}")


if __name__ == "__main__":
    main()
