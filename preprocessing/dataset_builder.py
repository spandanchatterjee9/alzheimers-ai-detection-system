"""
Dataset Builder Module for Alzheimer's Disease Detection System.

Responsibility:
Coordinates end-to-end parallel preprocessing across ALL 413 patients (both labelled and unlabelled).
Processes each patient volume through loading, validation, orientation, normalization, cropping,
resizing, slice extraction, quality filtering, and saves rich persistent artifacts.

Artifacts generated per patient:
- dataset/processed/labelled/OAS1_XXXX/ (or unlabelled/OAS1_XXXX/)
  ├── volume.nii.gz              (NIfTI 1mm isotropic volume)
  ├── metadata.json              (Demographic and geometric metadata)
  ├── tensor.pt                  (PyTorch 3D tensor: [1, 128, 128, 128])
  ├── diagnostic_preview.png     (3-plane Axial/Coronal/Sagittal visualization grid)
  ├── preprocessing_log.json     (Step-by-step execution log and voxel statistics)
  ├── slices/                    (2D planar slices)
  └── tensors/                   (2D and 3D PyTorch tensors)

Features:
- Multiprocessing / Parallel processing support with tqdm progress bars.
- Skips already processed patients unless force_reprocess is True.
- Handles both labelled (READY) and unlabelled (MISSING_LABEL) subjects.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from preprocessing.loader import MriLoader
from preprocessing.validator import VolumeValidator
from preprocessing.orientation import VolumeOrientation
from preprocessing.normalizer import IntensityNormalizer
from preprocessing.cropper import VolumeCropper
from preprocessing.resizer import VolumeResizer
from preprocessing.slice_extractor import SliceExtractor
from preprocessing.quality_filter import SliceQualityFilter

logger = logging.getLogger("preprocessing.dataset_builder")


def _process_patient_worker(args_tuple: Tuple[Dict[str, Any], str, bool]) -> Tuple[str, bool, str]:
    """
    Multiprocessing worker function to process a single patient record.
    """
    row_dict, config_path, force_reprocess = args_tuple

    patient_id = row_dict["patient_id"]
    short_id = row_dict.get("short_id", patient_id.replace("_MR1", ""))
    status = row_dict.get("status", "READY")
    hdr_path = row_dict.get("native_scan_path") or row_dict.get("hdr_path")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Determine destination directory: labelled vs unlabelled
    base_processed_dir = Path(config["dataset"]["processed_dir"])
    if status == "READY":
        patient_dir = base_processed_dir / "labelled" / short_id
    else:
        patient_dir = base_processed_dir / "unlabelled" / short_id

    volume_nii_path = patient_dir / "volume.nii.gz"
    metadata_json_path = patient_dir / "metadata.json"
    tensor_pt_path = patient_dir / "tensor.pt"
    diagnostic_png_path = patient_dir / "diagnostic_preview.png"
    log_json_path = patient_dir / "preprocessing_log.json"
    slices_dir = patient_dir / "slices"
    tensors_dir = patient_dir / "tensors"

    # Resume / Cache check
    if not force_reprocess and volume_nii_path.exists() and metadata_json_path.exists() and tensor_pt_path.exists():
        return short_id, True, "SKIPPED_CACHED"

    if not hdr_path or not os.path.exists(hdr_path):
        return short_id, False, f"HDR file missing: {hdr_path}"

    start_time = time.time()
    execution_steps: List[Dict[str, Any]] = []

    try:
        # Initialize modules
        loader = MriLoader()
        validator = VolumeValidator(expected_dims=3, min_foreground_ratio=0.05)
        orienter = VolumeOrientation(target_orientation=tuple(config["volume_specs"].get("orientation", "RAS")))
        normalizer = IntensityNormalizer(
            method=config["normalization"]["method"],
            clip_percentiles=tuple(config["normalization"]["clip_percentiles"]),
            min_max_range=tuple(config["normalization"]["min_max_range"])
        )
        cropper = VolumeCropper(
            background_threshold=config["cropping"]["background_threshold"],
            padding_voxels=config["cropping"]["padding_voxels"]
        )
        resizer = VolumeResizer(
            target_shape=tuple(config["volume_specs"]["target_shape"]),
            interpolation_order=config["resizing"]["interpolation_order"]
        )
        slice_extractor = SliceExtractor(axes=config["slice_extraction"]["axes"])
        quality_filter = SliceQualityFilter(
            min_foreground_ratio=config["slice_extraction"]["quality_filter"]["min_foreground_ratio"],
            min_entropy=config["slice_extraction"]["quality_filter"]["min_entropy"]
        )

        patient_dir.mkdir(parents=True, exist_ok=True)
        slices_dir.mkdir(parents=True, exist_ok=True)
        tensors_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Load volume
        t0 = time.time()
        volume, spacing, affine = loader.load_volume(hdr_path)
        execution_steps.append({"step": "load", "duration_sec": round(time.time() - t0, 4), "orig_shape": list(volume.shape)})

        # Step 2: Validate
        t0 = time.time()
        is_valid, issues = validator.validate(volume, spacing)
        execution_steps.append({"step": "validate", "duration_sec": round(time.time() - t0, 4), "is_valid": is_valid, "issues": issues})
        if not is_valid:
            return short_id, False, f"Validation failed: {issues}"

        # Step 3: Reorient to RAS
        t0 = time.time()
        volume, ras_affine = orienter.reorient_to_ras(volume, affine)
        execution_steps.append({"step": "orientation", "duration_sec": round(time.time() - t0, 4), "target_ornt": "RAS"})

        # Step 4: Normalize
        t0 = time.time()
        vol_normalized = normalizer.normalize(volume)
        execution_steps.append({"step": "normalize", "duration_sec": round(time.time() - t0, 4), "method": config["normalization"]["method"]})

        # Step 5: Crop background
        t0 = time.time()
        vol_cropped, bbox_info = cropper.crop(vol_normalized)
        execution_steps.append({"step": "crop", "duration_sec": round(time.time() - t0, 4), "cropped_shape": list(vol_cropped.shape)})

        # Step 6: Resize 3D volume
        t0 = time.time()
        vol_resized = resizer.resize_3d(vol_cropped)
        execution_steps.append({"step": "resize_3d", "duration_sec": round(time.time() - t0, 4), "target_shape": list(vol_resized.shape)})

        # Artifact 1: Save NIfTI volume (.nii.gz)
        if config["performance"].get("save_nifti", True) and HAS_NIBABEL:
            nifti_img = nib.Nifti1Image(vol_resized.astype(np.float32), ras_affine)
            nib.save(nifti_img, str(volume_nii_path))

        # Artifact 2: Save 3D PyTorch Tensor (tensor.pt)
        vol_tensor_4d = np.expand_dims(vol_resized.astype(np.float32), axis=0) # (1, D, H, W)
        if HAS_TORCH:
            torch.save(torch.from_numpy(vol_tensor_4d), str(tensor_pt_path))
            torch.save(torch.from_numpy(vol_tensor_4d), str(tensors_dir / "volume_3d.pt"))
        else:
            np.save(str(tensor_pt_path).replace(".pt", ".npy"), vol_tensor_4d)

        # Step 7: Slice Extraction & Quality Filtering
        t0 = time.time()
        slices_dict = slice_extractor.extract_slices(vol_normalized)
        target_2d_shape = tuple(config["slice_extraction"]["target_slice_shape"])
        saved_slices_meta = []

        for axis, slice_list in slices_dict.items():
            for idx, slice_img in slice_list:
                passed, metrics = quality_filter.evaluate_slice(slice_img)
                if passed:
                    slice_resized = resizer.resize_2d(slice_img, target_2d_shape)
                    slice_filename = f"{axis}_{idx:03d}.npy"
                    slice_npy_path = slices_dir / slice_filename
                    np.save(str(slice_npy_path), slice_resized.astype(np.float32))

                    # Save 2D PyTorch slice tensor
                    if HAS_TORCH:
                        slice_tensor_3d = np.expand_dims(slice_resized.astype(np.float32), axis=0) # (1, H, W)
                        torch.save(torch.from_numpy(slice_tensor_3d), str(tensors_dir / f"{axis}_{idx:03d}.pt"))

                    saved_slices_meta.append({
                        "axis": axis,
                        "index": idx,
                        "file": slice_filename,
                        "fg_ratio": round(metrics["fg_ratio"], 4),
                        "entropy": round(metrics["entropy"], 4),
                    })

        execution_steps.append({"step": "slice_extraction", "duration_sec": round(time.time() - t0, 4), "saved_slices_count": len(saved_slices_meta)})

        # Artifact 3: Generate 3-Plane Diagnostic Preview Image (diagnostic_preview.png)
        if config["performance"].get("save_diagnostic_preview", True) and HAS_MATPLOTLIB:
            _generate_diagnostic_preview(
                volume=vol_resized,
                short_id=short_id,
                cdr=row_dict.get("CDR"),
                age=row_dict.get("age"),
                save_path=diagnostic_png_path
            )

        # Artifact 4: Metadata JSON
        cdr_val = float(row_dict["CDR"]) if pd.notnull(row_dict.get("CDR")) else None
        mmse_val = float(row_dict["MMSE"]) if pd.notnull(row_dict.get("MMSE")) else None
        age_val = float(row_dict["age"]) if pd.notnull(row_dict.get("age")) else None

        metadata = {
            "patient_id": patient_id,
            "short_id": short_id,
            "status": status,
            "age": age_val,
            "gender": str(row_dict.get("gender")) if pd.notnull(row_dict.get("gender")) else None,
            "CDR": cdr_val,
            "MMSE": mmse_val,
            "orig_shape": list(volume.shape),
            "cropped_shape": list(vol_cropped.shape),
            "final_3d_shape": list(vol_resized.shape),
            "bounding_box": bbox_info,
            "selected_slices_count": len(saved_slices_meta),
            "voxel_stats": {
                "min": float(np.min(vol_resized)),
                "max": float(np.max(vol_resized)),
                "mean": float(np.mean(vol_resized)),
                "std": float(np.std(vol_resized)),
            },
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Artifact 5: Preprocessing Log JSON
        total_duration = round(time.time() - start_time, 4)
        log_data = {
            "patient_id": patient_id,
            "short_id": short_id,
            "total_duration_sec": total_duration,
            "execution_steps": execution_steps,
        }

        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        return short_id, True, "SUCCESS"

    except Exception as e:
        logger.error(f"Worker error processing patient {short_id}: {e}", exc_info=True)
        return short_id, False, str(e)


def _generate_diagnostic_preview(
    volume: np.ndarray,
    short_id: str,
    cdr: Optional[float],
    age: Optional[float],
    save_path: Path
) -> None:
    """Generates a 3-plane (Axial, Coronal, Sagittal) diagnostic visualization grid."""
    nx, ny, nz = volume.shape
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

    # Slices
    sag_slice = np.rot90(volume[mid_x, :, :])
    cor_slice = np.rot90(volume[:, mid_y, :])
    axi_slice = np.rot90(volume[:, :, mid_z])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0f172a")

    cdr_str = f"CDR: {cdr}" if cdr is not None else "CDR: Unlabelled"
    age_str = f"Age: {age:.0f}" if age is not None else "Age: N/A"

    titles = [f"Axial (Z={mid_z})", f"Coronal (Y={mid_y})", f"Sagittal (X={mid_x})"]
    slices = [axi_slice, cor_slice, sag_slice]

    for ax, slc, title in zip(axes, slices, titles):
        ax.imshow(slc, cmap="gray")
        ax.set_title(title, color="#e2e8f0", fontsize=11, pad=8)
        ax.axis("off")

    fig.suptitle(
        f"Diagnostic Preview | Patient ID: {short_id} | {age_str} | {cdr_str}",
        color="#38bdf8",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


class DatasetBuilder:
    """
    Master dataset builder managing parallel processing for all 413 patients.
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
        self.num_workers = self.config["performance"].get("num_workers", 4)
        self.force_reprocess = self.config["performance"].get("force_reprocess", False)

    def build_dataset(self, max_patients: Optional[int] = None, force: Optional[bool] = None) -> Dict[str, Any]:
        """
        Executes parallel preprocessing across all patients in manifest.

        Args:
            max_patients: Optional limit on number of patients to process.
            force: Force reprocessing existing cached files.

        Returns:
            Dictionary summary of preprocessing statistics.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {self.manifest_path}. Run scanner first.")

        df = pd.read_csv(self.manifest_path)
        if max_patients is not None:
            df = df.head(max_patients)

        force_flag = force if force is not None else self.force_reprocess

        total_patients = len(df)
        logger.info(f"Starting Dataset Builder for {total_patients} total subjects (Labelled & Unlabelled)...")

        tasks = [(row.to_dict(), self.config_path, force_flag) for _, row in df.iterrows()]

        start_time = time.time()
        success_count = 0
        skipped_count = 0
        failed_count = 0
        failed_details = []

        # Single or Multi-process execution
        if self.num_workers <= 1:
            for task in tasks:
                short_id, ok, msg = _process_patient_worker(task)
                if ok:
                    if msg == "SKIPPED_CACHED":
                        skipped_count += 1
                    else:
                        success_count += 1
                else:
                    failed_count += 1
                    failed_details.append((short_id, msg))
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(_process_patient_worker, task): task[0]["patient_id"] for task in tasks}

                iterator = as_completed(futures)
                if HAS_TQDM:
                    iterator = tqdm(iterator, total=total_patients, desc="Processing Patients")

                for future in iterator:
                    try:
                        short_id, ok, msg = future.result()
                        if ok:
                            if msg == "SKIPPED_CACHED":
                                skipped_count += 1
                            else:
                                success_count += 1
                        else:
                            failed_count += 1
                            failed_details.append((short_id, msg))
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Worker execution exception: {e}")

        total_time = round(time.time() - start_time, 2)
        avg_time = round(total_time / total_patients, 2) if total_patients > 0 else 0.0

        summary = {
            "total_patients": total_patients,
            "processed_success": success_count,
            "skipped_cached": skipped_count,
            "failed_count": failed_count,
            "failed_details": failed_details,
            "total_time_sec": total_time,
            "avg_time_per_patient_sec": avg_time,
        }

        logger.info(
            f"Dataset Build Finished! Success: {success_count}, Cached: {skipped_count}, "
            f"Failed: {failed_count}, Total Time: {total_time}s (Avg {avg_time}s/patient)."
        )
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = DatasetBuilder()
    summary = builder.build_dataset(max_patients=2)
    print("Build Summary:", json.dumps(summary, indent=2))
