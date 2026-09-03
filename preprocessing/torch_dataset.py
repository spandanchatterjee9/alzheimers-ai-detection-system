"""
PyTorch Dataset & DataLoader Module for Alzheimer's Disease Detection System.

Responsibility:
Provides reusable, model-agnostic PyTorch Dataset and DataLoader wrappers for training
2D CNNs, 3D CNNs, Vision Transformers (ViT, Swin3D), MONAI models, and Self-Supervised Learning (SSL).

Dataset Classes Provided:
1. Oasis3DDataset: Volumetric 3D PyTorch Dataset ([1, D, H, W]).
2. Oasis2DSliceDataset: 2D Planar Slice PyTorch Dataset ([1, H, W]).
3. OasisUnlabelledDataset: Unlabelled 3D PyTorch Dataset for SSL & Autoencoders.

Complexity:
O(1) memory load per sample during DataLoader iteration via disk caching.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object

logger = logging.getLogger("preprocessing.torch_dataset")


class Oasis3DDataset(Dataset):
    """
    Model-agnostic PyTorch Dataset for 3D Volumetric Models (3D ResNet, 3D ConvNeXt, Swin3D).
    Loads 3D PyTorch tensors (1, 128, 128, 128) directly from processed patient directories.
    """

    def __init__(
        self,
        manifest_csv: Union[str, Path],
        target_col: str = "CDR",
        use_binary_target: bool = False,
        transform: Optional[Any] = None
    ) -> None:
        """
        Args:
            manifest_csv: Path to split CSV (train.csv, validation.csv, test.csv).
            target_col: Column name containing ground truth target (default 'CDR').
            use_binary_target: Convert CDR into binary classification (CDR=0 vs CDR>0).
            transform: Optional PyTorch / MONAI online data augmentation transforms.
        """
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv}")

        self.df = pd.read_csv(self.manifest_csv)
        self.target_col = target_col
        self.use_binary_target = use_binary_target
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Dict[str, Any]]:
        row = self.df.iloc[idx]
        patient_id = row["patient_id"]
        short_id = row.get("short_id", patient_id.replace("_MR1", ""))
        future_path = Path(row["future_processed_path"])

        tensor_pt_path = future_path / "tensor.pt"
        if not tensor_pt_path.exists():
            # Fallback check
            tensor_pt_path = future_path / "tensors" / "volume_3d.pt"

        if not tensor_pt_path.exists():
            raise FileNotFoundError(f"Processed 3D tensor missing for patient {short_id} at {tensor_pt_path}")

        # Load PyTorch 3D Tensor float32 (1, 128, 128, 128)
        if HAS_TORCH:
            volume_tensor = torch.load(str(tensor_pt_path))
        else:
            volume_tensor = np.load(str(tensor_pt_path).replace(".pt", ".npy"))

        if self.transform is not None:
            volume_tensor = self.transform(volume_tensor)

        # Ground Truth Label formatting
        raw_label = float(row[self.target_col]) if pd.notnull(row.get(self.target_col)) else 0.0
        if self.use_binary_target:
            target = 1.0 if raw_label > 0.0 else 0.0
            label_tensor = torch.tensor(target, dtype=torch.float32) if HAS_TORCH else target
        else:
            # Map CDR (0.0, 0.5, 1.0, 2.0) to class indices (0, 1, 2, 3)
            class_map = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
            class_idx = class_map.get(raw_label, 0)
            label_tensor = torch.tensor(class_idx, dtype=torch.long) if HAS_TORCH else class_idx

        meta = {
            "patient_id": patient_id,
            "short_id": short_id,
            "raw_cdr": raw_label,
            "age": float(row.get("age", 0.0)),
        }

        return volume_tensor, label_tensor, meta


class Oasis2DSliceDataset(Dataset):
    """
    Model-agnostic PyTorch Dataset for 2D Planar Slices (ResNet, EfficientNet, Vision Transformer).
    """

    def __init__(
        self,
        manifest_csv: Union[str, Path],
        axis: str = "axial",
        target_col: str = "CDR",
        use_binary_target: bool = False,
        transform: Optional[Any] = None
    ) -> None:
        """
        Args:
            manifest_csv: Path to split CSV.
            axis: Slice orientation axis ('axial', 'coronal', 'sagittal').
            target_col: Target column name.
            use_binary_target: Binary classification flag.
            transform: Optional 2D data augmentation transforms.
        """
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)
        self.axis = axis.lower()
        self.target_col = target_col
        self.use_binary_target = use_binary_target
        self.transform = transform

        # Index all available 2D slice files for patients in manifest
        self.slice_items: List[Dict[str, Any]] = []
        self._index_slices()

    def _index_slices(self) -> None:
        for _, row in self.df.iterrows():
            patient_id = row["patient_id"]
            short_id = row.get("short_id", patient_id.replace("_MR1", ""))
            future_path = Path(row["future_processed_path"])
            slices_dir = future_path / "slices"

            if not slices_dir.exists():
                continue

            axis_slices = sorted(list(slices_dir.glob(f"{self.axis}_*.npy")))
            raw_label = float(row[self.target_col]) if pd.notnull(row.get(self.target_col)) else 0.0

            for s_path in axis_slices:
                self.slice_items.append({
                    "patient_id": patient_id,
                    "short_id": short_id,
                    "slice_path": s_path,
                    "raw_cdr": raw_label,
                    "age": float(row.get("age", 0.0)),
                })

    def __len__(self) -> int:
        return len(self.slice_items)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Dict[str, Any]]:
        item = self.slice_items[idx]
        slice_np = np.load(str(item["slice_path"])).astype(np.float32)
        # Format as (1, H, W)
        slice_tensor = np.expand_dims(slice_np, axis=0)

        if HAS_TORCH:
            slice_tensor = torch.from_numpy(slice_tensor)

        if self.transform is not None:
            slice_tensor = self.transform(slice_tensor)

        raw_label = item["raw_cdr"]
        if self.use_binary_target:
            target = 1.0 if raw_label > 0.0 else 0.0
            label_tensor = torch.tensor(target, dtype=torch.float32) if HAS_TORCH else target
        else:
            class_map = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
            class_idx = class_map.get(raw_label, 0)
            label_tensor = torch.tensor(class_idx, dtype=torch.long) if HAS_TORCH else class_idx

        return slice_tensor, label_tensor, item


class OasisUnlabelledDataset(Dataset):
    """
    PyTorch Dataset for Unlabelled Patients (179 subjects) designed for
    Self-Supervised Learning (SSL), Autoencoders, and Contrastive Learning.
    """

    def __init__(self, unlabelled_manifest_csv: Union[str, Path], transform: Optional[Any] = None) -> None:
        self.manifest_path = Path(unlabelled_manifest_csv)
        self.df = pd.read_csv(self.manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        row = self.df.iloc[idx]
        patient_id = row["patient_id"]
        short_id = row.get("short_id", patient_id.replace("_MR1", ""))
        future_path = Path(row["future_processed_path"])

        tensor_pt_path = future_path / "tensor.pt"
        if HAS_TORCH:
            volume_tensor = torch.load(str(tensor_pt_path))
        else:
            volume_tensor = np.load(str(tensor_pt_path).replace(".pt", ".npy"))

        if self.transform is not None:
            volume_tensor = self.transform(volume_tensor)

        meta = {"patient_id": patient_id, "short_id": short_id}
        return volume_tensor, meta


def create_dataloaders(
    train_csv: str = "dataset/manifests/train.csv",
    val_csv: str = "dataset/manifests/validation.csv",
    test_csv: str = "dataset/manifests/test.csv",
    batch_size: int = 4,
    num_workers: int = 2,
    use_3d: bool = True,
    use_binary_target: bool = False
) -> Dict[str, Any]:
    """
    Factory function creating PyTorch DataLoaders for Train, Val, and Test splits.

    Returns:
        Dict containing 'train', 'val', and 'test' DataLoaders.
    """
    if not HAS_TORCH:
        logger.warning("PyTorch not installed; returning None for dataloaders.")
        return {}

    DatasetClass = Oasis3DDataset if use_3d else Oasis2DSliceDataset

    train_ds = DatasetClass(train_csv, use_binary_target=use_binary_target)
    val_ds = DatasetClass(val_csv, use_binary_target=use_binary_target)
    test_ds = DatasetClass(test_csv, use_binary_target=use_binary_target)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_csv = "dataset/manifests/train.csv"
    if os.path.exists(train_csv):
        ds_3d = Oasis3DDataset(train_csv)
        print(f"Oasis3DDataset loaded successfully. Total train samples: {len(ds_3d)}")
