"""
Patient-Wise Stratified Split Generator Module for Alzheimer's Disease Detection System.

Responsibility:
Partition labelled dataset manifest into Train (80%), Validation (10%), and Test (10%) splits
STRICTLY at the Patient ID (Subject Level) with stratification on Clinical Dementia Rating (CDR).

CRITICAL MEDICAL IMAGING RULE:
Never split data by slice or image. Doing so causes severe data leakage, false high accuracy,
and evaluation failure on unseen clinical patients. Unlabelled patients are stored separately in
unlabelled_manifest.csv and are NEVER included in supervised train/val/test splits.

Outputs generated:
- dataset/manifests/train.csv (80%)
- dataset/manifests/validation.csv (10%)
- dataset/manifests/test.csv (10%)

Complexity:
O(N) time complexity where N is ready patients (~234).
O(N) space complexity.
"""

import os
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger("training.split_generator")


class PatientSplitGenerator:
    """
    Generates stratified patient-wise train, validation, and test splits.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        """
        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.manifest_path = Path(self.config["dataset"]["manifest_path"])
        self.labelled_manifest_path = Path(self.config["dataset"]["labelled_manifest_path"])
        self.train_ratio = self.config["splits"]["train_ratio"]
        self.val_ratio = self.config["splits"]["val_ratio"]
        self.test_ratio = self.config["splits"]["test_ratio"]
        self.stratify_col = self.config["splits"]["stratify_col"]
        self.random_seed = self.config["splits"]["random_seed"]

        self.train_path = Path(self.config["splits"]["train_path"])
        self.val_path = Path(self.config["splits"]["val_path"])
        self.test_path = Path(self.config["splits"]["test_path"])

        self.train_path.parent.mkdir(parents=True, exist_ok=True)

    def generate_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates and saves stratified patient-wise train, validation, and test split CSVs.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        source_path = self.labelled_manifest_path if self.labelled_manifest_path.exists() else self.manifest_path
        if not source_path.exists():
            raise FileNotFoundError(f"Labelled manifest not found at {source_path}. Run scanner first.")

        df = pd.read_csv(source_path)
        ready_df = df[df["status"] == "READY"].copy() if "status" in df.columns else df.copy()

        if len(ready_df) == 0:
            raise ValueError("No patients with status 'READY' found in manifest.")

        logger.info(f"Generating patient-wise splits for {len(ready_df)} ready subjects...")

        # Binary/Multi-class CDR stratification label
        ready_df["strat_label"] = ready_df[self.stratify_col].apply(lambda x: str(x))

        # First stage: Split into Train (80%) and Temp (20%)
        temp_ratio = self.val_ratio + self.test_ratio
        train_df, temp_df = train_test_split(
            ready_df,
            test_size=temp_ratio,
            random_state=self.random_seed,
            stratify=ready_df["strat_label"]
        )

        # Second stage: Split Temp (20%) equally into Val (10%) and Test (10%)
        relative_val_ratio = self.val_ratio / temp_ratio  # 0.10 / 0.20 = 0.50
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1.0 - relative_val_ratio,
            random_state=self.random_seed,
            stratify=temp_df["strat_label"]
        )

        # Clean up temporary column
        for d in [train_df, val_df, test_df]:
            if "strat_label" in d.columns:
                d.drop(columns=["strat_label"], inplace=True)

        # Integrity Assertion: ZERO Patient Overlap
        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        test_patients = set(test_df["patient_id"])

        overlap_tv = train_patients.intersection(val_patients)
        overlap_tt = train_patients.intersection(test_patients)
        overlap_vt = val_patients.intersection(test_patients)

        if overlap_tv or overlap_tt or overlap_vt:
            raise RuntimeError(
                f"CRITICAL ERROR: Patient overlap detected! TV: {overlap_tv}, TT: {overlap_tt}, VT: {overlap_vt}"
            )

        logger.info("VALIDATION SUCCESSFUL: Zero patient overlap across splits.")

        # Save to disk
        train_df.to_csv(self.train_path, index=False)
        val_df.to_csv(self.val_path, index=False)
        test_df.to_csv(self.test_path, index=False)

        logger.info(f"Saved Train Split ({len(train_df)} patients) -> {self.train_path}")
        logger.info(f"Saved Val Split ({len(val_df)} patients) -> {self.val_path}")
        logger.info(f"Saved Test Split ({len(test_df)} patients) -> {self.test_path}")

        return train_df, val_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = PatientSplitGenerator()
    train_df, val_df, test_df = generator.generate_splits()
    print(f"Splits generated: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
