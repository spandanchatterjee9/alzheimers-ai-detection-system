"""
PyTorch 2D Slice Dataset & DataLoader Module for Alzheimer's Stage Classification.

Responsibility:
- Indexes and loads preprocessed 224x224 2D MRI slices (.npy float32).
- Resolves project-relative paths cleanly across Windows and Linux / Google Colab environments.
- Provides Oasis2DSliceDataset and DataLoader factory with training data augmentations.
- Calculates training class distribution weights for Weighted Cross-Entropy Loss.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

logger = logging.getLogger("training.dataset")

# Default 3-Class Mapping (CN=0, Very Mild/MCI=1, Mild/Moderate AD=2)
DEFAULT_3CLASS_MAP = {
    0.0: 0,  # Cognitively Normal
    0.5: 1,  # Very Mild Dementia / MCI
    1.0: 2,  # Mild Dementia
    2.0: 2   # Moderate Dementia (merged with Mild)
}

# Standard 4-Class Mapping
DEFAULT_4CLASS_MAP = {
    0.0: 0,  # Normal
    0.5: 1,  # Very Mild Dementia
    1.0: 2,  # Mild Dementia
    2.0: 3   # Moderate Dementia
}


class Oasis2DSliceDataset(Dataset):
    """
    2D MRI Slice PyTorch Dataset.
    Loads standardized 224x224 grayscale MRI slices (.npy) across Axial (or specified) orientation.
    """

    def __init__(
        self,
        manifest_csv: Union[str, Path],
        axis: str = "axial",
        transform: Optional[Any] = None,
        base_dir: Optional[Union[str, Path]] = None,
        class_mapping: Optional[Dict[float, int]] = None
    ) -> None:
        """
        Args:
            manifest_csv: Path to split CSV (train.csv, validation.csv, test.csv).
            axis: Anatomical plane ('axial', 'coronal', 'sagittal').
            transform: Optional data augmentation transform.
            base_dir: Project root directory for resolving relative paths (default: current working dir).
            class_mapping: Mapping from CDR float to integer class label (default: 3-class mapping).
        """
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found at: {self.manifest_csv}")

        self.df = pd.read_csv(self.manifest_csv)
        self.axis = axis.lower()
        self.transform = transform
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self.class_mapping = class_mapping or DEFAULT_3CLASS_MAP


        self.slice_items: List[Dict[str, Any]] = []
        self._index_slices()

    def _index_slices(self) -> None:
        """Indexes all valid 2D slice files for subjects in the manifest."""
        for _, row in self.df.iterrows():
            patient_id = row["patient_id"]
            short_id = row.get("short_id", patient_id.replace("_MR1", ""))
            
            # Resolve relative or absolute future_processed_path
            raw_path_str = str(row["future_processed_path"]).replace("\\", "/")
            proc_path = Path(raw_path_str)
            if not proc_path.is_absolute():
                proc_path = self.base_dir / proc_path

            slices_dir = proc_path / "slices"
            if not slices_dir.exists():
                logger.warning(f"Slices directory missing for patient {patient_id} at {slices_dir}")
                continue

            axis_slices = sorted(list(slices_dir.glob(f"{self.axis}_*.npy")))
            raw_cdr = float(row["CDR"]) if pd.notnull(row.get("CDR")) else 0.0
            class_idx = self.class_mapping.get(raw_cdr, 0)

            for s_path in axis_slices:
                self.slice_items.append({
                    "patient_id": patient_id,
                    "short_id": short_id,
                    "slice_path": s_path,
                    "slice_name": s_path.stem,
                    "raw_cdr": raw_cdr,
                    "class_idx": class_idx,
                    "age": float(row.get("age", 0.0))
                })

        logger.info(f"Indexed {len(self.slice_items)} {self.axis} slices across {len(self.df)} subjects from {self.manifest_csv.name}")

    def __len__(self) -> int:
        return len(self.slice_items)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Dict[str, Any]]:
        item = self.slice_items[idx]
        
        # Load (224, 224) float32 array
        slice_arr = np.load(str(item["slice_path"])).astype(np.float32)
        
        # Format as (1, 224, 224) channel-first tensor
        slice_tensor = np.expand_dims(slice_arr, axis=0)

        if HAS_TORCH:
            slice_tensor = torch.from_numpy(slice_tensor)
            label_tensor = torch.tensor(item["class_idx"], dtype=torch.long)
        else:
            label_tensor = item["class_idx"]

        if self.transform is not None:
            slice_tensor = self.transform(slice_tensor)

        meta = {
            "patient_id": item["patient_id"],
            "short_id": item["short_id"],
            "slice_name": item["slice_name"],
            "raw_cdr": item["raw_cdr"],
            "class_idx": item["class_idx"]
        }

        return slice_tensor, label_tensor, meta


def compute_training_class_weights(
    train_csv: Union[str, Path],
    num_classes: int = 4
) -> torch.Tensor:
    """
    Computes inverse frequency class weights from the TRAINING patient distribution only.
    Formula: weight_c = total_samples / (num_classes * count_c)
    """
    df = pd.read_csv(train_csv)
    counts = np.zeros(num_classes, dtype=np.float32)

    for _, row in df.iterrows():
        raw_cdr = float(row["CDR"]) if pd.notnull(row.get("CDR")) else 0.0
        c = CDR_CLASS_MAP.get(raw_cdr, 0)
        counts[c] += 1

    total = len(df)
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total / (num_classes * counts[c])
        else:
            weights[c] = 1.0

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32) if HAS_TORCH else weights


def create_2d_dataloaders(
    train_csv: Union[str, Path] = "dataset/manifests/train.csv",
    val_csv: Union[str, Path] = "dataset/manifests/validation.csv",
    test_csv: Optional[Union[str, Path]] = "dataset/manifests/test.csv",
    axis: str = "axial",
    batch_size: int = 32,
    num_workers: int = 2,
    base_dir: Optional[Union[str, Path]] = None
) -> Tuple[Any, Any, Optional[Any]]:
    """
    DataLoader factory for 2D Slice Training, Validation, and Testing.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to instantiate DataLoaders.")

    train_ds = Oasis2DSliceDataset(train_csv, axis=axis, base_dir=base_dir)
    val_ds = Oasis2DSliceDataset(val_csv, axis=axis, base_dir=base_dir)
    test_ds = Oasis2DSliceDataset(test_csv, axis=axis, base_dir=base_dir) if test_csv else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    ) if test_ds else None

    return train_loader, val_loader, test_loader
