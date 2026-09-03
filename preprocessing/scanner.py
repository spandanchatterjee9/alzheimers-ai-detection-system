"""
Dataset Scanner Module for Alzheimer's Disease Detection System.

Responsibility:
Scans the dataset directory for OASIS-1 patient folders across all 12 discs,
parses clinical metadata from OAS1_XXXX_MR1.txt files, classifies subjects into
READY (Labelled) vs MISSING_LABEL (Unlabelled), locates native and T88 volume pairs,
and exports dataset manifests to dataset/manifests/.

Outputs generated:
- dataset/manifests/dataset_manifest.csv (All 413 patients)
- dataset/manifests/labelled_manifest.csv (234 READY patients)
- dataset/manifests/unlabelled_manifest.csv (179 MISSING_LABEL patients)

Complexity:
O(N) time complexity where N is the total number of subjects (~413).
O(N) space complexity to store manifest DataFrames.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml

logger = logging.getLogger("preprocessing.scanner")


class OasisDatasetScanner:
    """
    Scans the OASIS-1 dataset structure across all discs, parses metadata,
    and creates dataset_manifest.csv, labelled_manifest.csv, and unlabelled_manifest.csv.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """
        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.raw_dirs = [Path(p) for p in self.config["dataset"]["raw_dirs"]]
        self.disc_prefix = self.config["dataset"]["disc_prefix"]
        self.manifests_dir = Path(self.config["dataset"]["manifests_dir"])
        self.manifest_path = Path(self.config["dataset"]["manifest_path"])
        self.labelled_manifest_path = Path(self.config["dataset"]["labelled_manifest_path"])
        self.unlabelled_manifest_path = Path(self.config["dataset"]["unlabelled_manifest_path"])
        self.processed_labelled_dir = Path(self.config["dataset"]["processed_labelled_dir"])
        self.processed_unlabelled_dir = Path(self.config["dataset"]["processed_unlabelled_dir"])

        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.processed_labelled_dir.mkdir(parents=True, exist_ok=True)
        self.processed_unlabelled_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads YAML configuration."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def parse_metadata_txt(self, txt_path: Path) -> Dict[str, Any]:
        """
        Parses clinical metadata from an OASIS .txt file.

        Args:
            txt_path: Path object pointing to OAS1_XXXX_MR1.txt file.

        Returns:
            Dictionary of extracted clinical parameters.
        """
        metadata: Dict[str, Any] = {
            "age": None,
            "gender": None,
            "hand": None,
            "education": None,
            "ses": None,
            "CDR": None,
            "MMSE": None,
            "etiv": None,
            "asf": None,
            "nwbv": None,
        }

        if not txt_path.exists():
            logger.warning(f"Metadata file missing: {txt_path}")
            return metadata

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            line_str = line.strip()
            if ":" not in line_str:
                continue

            parts = [p.strip() for p in line_str.split(":", 1)]
            key, val = parts[0].upper(), parts[1]

            if key == "AGE":
                metadata["age"] = float(val) if val and val != "N/A" else None
            elif key in ["M/F", "GENDER"]:
                metadata["gender"] = val
            elif key == "HAND":
                metadata["hand"] = val
            elif key in ["EDUC", "EDUCATION"]:
                metadata["education"] = float(val) if val and val != "N/A" else None
            elif key == "SES":
                metadata["ses"] = float(val) if val and val != "N/A" else None
            elif key == "CDR":
                metadata["CDR"] = float(val) if val and val != "N/A" else None
            elif key == "MMSE":
                metadata["MMSE"] = float(val) if val and val != "N/A" else None
            elif key == "ETIV":
                metadata["etiv"] = float(val) if val and val != "N/A" else None
            elif key == "ASF":
                metadata["asf"] = float(val) if val and val != "N/A" else None
            elif key == "NWBV":
                metadata["nwbv"] = float(val) if val and val != "N/A" else None

        return metadata

    def scan(self) -> pd.DataFrame:
        """
        Scans all 12 disc folders for patient directories and extracts manifests.

        Returns:
            Pandas DataFrame of total scanned manifest.
        """
        disc_dirs: List[Path] = []
        for raw_dir in self.raw_dirs:
            if not raw_dir.exists():
                continue
            matches = sorted(list(raw_dir.glob(f"{self.disc_prefix}*")))
            for m in matches:
                if m.is_dir() and m not in disc_dirs:
                    disc_dirs.append(m)

        if not disc_dirs:
            logger.error("No disc directories found matching pattern across raw search locations.")
            return pd.DataFrame()

        manifest_rows: List[Dict[str, Any]] = []
        processed_subject_ids = set()

        for disc in disc_dirs:
            disc_name = disc.name
            # Disc number extraction (e.g. 1 to 12)
            disc_num_str = disc_name.replace(self.disc_prefix, "").strip("_")
            disc_number = int(disc_num_str) if disc_num_str.isdigit() else None

            # Look for inner disc folder or direct subject directories
            inner_disc = disc / f"disc{disc_num_str}"
            search_dir = inner_disc if inner_disc.exists() else disc

            subject_dirs = sorted([d for d in search_dir.glob("OAS1_*_MR1") if d.is_dir()])

            for subj_dir in subject_dirs:
                patient_id = subj_dir.name
                if patient_id in processed_subject_ids:
                    continue  # Deduplicate if found in multiple locations

                processed_subject_ids.add(patient_id)
                txt_path = subj_dir / f"{patient_id}.txt"

                # Parse metadata
                meta = self.parse_metadata_txt(txt_path)

                # Native scan path: SUBJ_111
                subj_111_dir = subj_dir / "PROCESSED" / "MPRAGE" / "SUBJ_111"
                hdr_files = list(subj_111_dir.glob("*.hdr"))
                img_files = list(subj_111_dir.glob("*.img"))

                hdr_path = hdr_files[0] if hdr_files else None
                img_path = img_files[0] if img_files else None

                # T88 scan path
                t88_dir = subj_dir / "PROCESSED" / "MPRAGE" / "T88_111"
                t88_hdr_files = list(t88_dir.glob("*_t88_masked_gfc.hdr"))
                t88_hdr_path = t88_hdr_files[0] if t88_hdr_files else None

                # Status determination
                status = "READY" if meta["CDR"] is not None else "MISSING_LABEL"
                if not hdr_path or not hdr_path.exists():
                    status = "MISSING_HDR"
                elif not img_path or not img_path.exists():
                    status = "MISSING_IMG"

                # Standardized output patient directory ID format (e.g. OAS1_0001)
                short_id = patient_id.replace("_MR1", "")
                if status == "READY":
                    future_processed_dir = self.processed_labelled_dir / short_id
                else:
                    future_processed_dir = self.processed_unlabelled_dir / short_id

                row = {
                    "patient_id": patient_id,
                    "short_id": short_id,
                    "disc_number": disc_number,
                    "age": meta["age"],
                    "gender": meta["gender"],
                    "hand": meta["hand"],
                    "education": meta["education"],
                    "ses": meta["ses"],
                    "CDR": meta["CDR"],
                    "MMSE": meta["MMSE"],
                    "etiv": meta["etiv"],
                    "asf": meta["asf"],
                    "nwbv": meta["nwbv"],
                    "native_scan_path": str(hdr_path.resolve()) if hdr_path else "",
                    "img_path": str(img_path.resolve()) if img_path else "",
                    "t88_scan_path": str(t88_hdr_path.resolve()) if t88_hdr_path else "",
                    "txt_path": str(txt_path.resolve()) if txt_path.exists() else "",
                    "future_processed_path": str(future_processed_dir.resolve()),
                    "status": status,
                }
                manifest_rows.append(row)

        df = pd.DataFrame(manifest_rows)
        df.to_csv(self.manifest_path, index=False)

        # Generate Labelled and Unlabelled Manifests
        labelled_df = df[df["status"] == "READY"].copy()
        unlabelled_df = df[df["status"] == "MISSING_LABEL"].copy()

        labelled_df.to_csv(self.labelled_manifest_path, index=False)
        unlabelled_df.to_csv(self.unlabelled_manifest_path, index=False)

        logger.info(
            f"Scanner completed. Total: {len(df)}, Labelled (READY): {len(labelled_df)}, "
            f"Unlabelled (MISSING_LABEL): {len(unlabelled_df)}."
        )
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scanner = OasisDatasetScanner()
    df = scanner.scan()
    print(f"Total Scanned: {len(df)}")
    print(df["status"].value_counts())
