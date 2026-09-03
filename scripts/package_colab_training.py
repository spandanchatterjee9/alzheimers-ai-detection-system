"""
Google Colab Training Package Builder.

Creates a clean, self-contained, minimal training package in 'Alzheimers_AI_Training_Package/'.
Copies only required 2D axial slices (.npy), manifests, model architectures, training loop,
inference engine, explainability module, verification scripts, and requirements.txt.
"""

import os
import shutil
from pathlib import Path
import pandas as pd

def build_training_package():
    print("==================================================")
    print("   BUILDING GOOGLE COLAB TRAINING PACKAGE         ")
    print("==================================================")

    root_dir = Path(".")
    dest_dir = root_dir / "Alzheimers_AI_Training_Package"

    if dest_dir.exists():
        print(f"Clearing existing package directory: {dest_dir}")
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Configs
    print(">> Copying configuration files...")
    (dest_dir / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_dir / "configs" / "config.yaml", dest_dir / "configs" / "config.yaml")
    shutil.copy2(root_dir / "configs" / "training.yaml", dest_dir / "configs" / "training.yaml")

    # 2. Manifests
    print(">> Copying manifests (train.csv, validation.csv, test.csv)...")
    (dest_dir / "dataset" / "manifests").mkdir(parents=True, exist_ok=True)
    manifest_src = root_dir / "dataset" / "manifests"
    shutil.copy2(manifest_src / "train.csv", dest_dir / "dataset" / "manifests" / "train.csv")
    shutil.copy2(manifest_src / "validation.csv", dest_dir / "dataset" / "manifests" / "validation.csv")
    shutil.copy2(manifest_src / "test.csv", dest_dir / "dataset" / "manifests" / "test.csv")

    # Also copy labelled and unlabelled manifest for reference integrity
    if (manifest_src / "labelled_manifest.csv").exists():
        shutil.copy2(manifest_src / "labelled_manifest.csv", dest_dir / "dataset" / "manifests" / "labelled_manifest.csv")

    # 3. Models
    print(">> Copying 2D model definitions...")
    (dest_dir / "models" / "cnn_2d").mkdir(parents=True, exist_ok=True)
    (dest_dir / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_dir / "models" / "__init__.py", dest_dir / "models" / "__init__.py")
    shutil.copy2(root_dir / "models" / "cnn_2d" / "__init__.py", dest_dir / "models" / "cnn_2d" / "__init__.py")
    shutil.copy2(root_dir / "models" / "cnn_2d" / "efficientnet_2d.py", dest_dir / "models" / "cnn_2d" / "efficientnet_2d.py")
    shutil.copy2(root_dir / "models" / "cnn_2d" / "resnet_2d.py", dest_dir / "models" / "cnn_2d" / "resnet_2d.py")

    # 4. Training
    print(">> Copying training engine and data loader...")
    (dest_dir / "training").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_dir / "training" / "__init__.py", dest_dir / "training" / "__init__.py")
    shutil.copy2(root_dir / "training" / "dataset.py", dest_dir / "training" / "dataset.py")
    shutil.copy2(root_dir / "training" / "evaluator.py", dest_dir / "training" / "evaluator.py")
    shutil.copy2(root_dir / "training" / "patient_aggregator.py", dest_dir / "training" / "patient_aggregator.py")
    shutil.copy2(root_dir / "training" / "trainer_2d.py", dest_dir / "training" / "trainer_2d.py")

    # 5. Scripts
    print(">> Copying verification and sanity scripts...")
    (dest_dir / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_dir / "scripts" / "verify_cuda.py", dest_dir / "scripts" / "verify_cuda.py")
    shutil.copy2(root_dir / "scripts" / "verify_training_data.py", dest_dir / "scripts" / "verify_training_data.py")
    shutil.copy2(root_dir / "scripts" / "verify_training_package.py", dest_dir / "scripts" / "verify_training_package.py")
    shutil.copy2(root_dir / "scripts" / "sanity_train.py", dest_dir / "scripts" / "sanity_train.py")

    # 6. Inference
    print(">> Copying inference module...")
    (dest_dir / "inference").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_dir / "inference" / "__init__.py", dest_dir / "inference" / "__init__.py")
    shutil.copy2(root_dir / "inference" / "engine.py", dest_dir / "inference" / "engine.py")

    # 7. Explainability
    print(">> Copying explainability module (Grad-CAM)...")
    (dest_dir / "explainability").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root_dir / "explainability" / "__init__.py", dest_dir / "explainability" / "__init__.py")
    shutil.copy2(root_dir / "explainability" / "gradcam.py", dest_dir / "explainability" / "gradcam.py")

    # 8. Requirements
    print(">> Creating requirements.txt...")
    req_content = """# Minimal requirements for 2D EfficientNet-B0 Training on Google Colab
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.23.0
pyyaml>=6.0
scikit-learn>=1.2.0
"""
    with open(dest_dir / "requirements.txt", "w") as f:
        f.write(req_content)

    # 9. Copy ONLY 2D Axial Slices for all 234 labelled patients
    print(">> Copying 2D Axial slices for 234 labelled patients (Excluding volumes, logs, previews)...")
    labelled_src_dir = root_dir / "dataset" / "processed" / "labelled"
    labelled_dst_dir = dest_dir / "dataset" / "processed" / "labelled"
    labelled_dst_dir.mkdir(parents=True, exist_ok=True)

    patient_folders = sorted([p for p in labelled_src_dir.iterdir() if p.is_dir()])
    print(f"Found {len(patient_folders)} labelled patient directories in source.")

    total_slices_copied = 0
    for p_folder in patient_folders:
        p_id = p_folder.name
        dst_slices_dir = labelled_dst_dir / p_id / "slices"
        dst_slices_dir.mkdir(parents=True, exist_ok=True)

        src_slices_dir = p_folder / "slices"
        if src_slices_dir.exists():
            axial_files = list(src_slices_dir.glob("axial_*.npy"))
            for ax_file in axial_files:
                shutil.copy2(ax_file, dst_slices_dir / ax_file.name)
                total_slices_copied += 1

    print(f">> Copied {total_slices_copied} Axial slices across {len(patient_folders)} patient folders.")

    # 10. Audit File Count and Size
    print("\n==================================================")
    print("   PACKAGE INVENTORY & SIZE AUDIT                 ")
    print("==================================================")

    all_files = list(dest_dir.rglob("*"))
    file_count = sum(1 for f in all_files if f.is_file())
    dir_count = sum(1 for f in all_files if f.is_dir())
    total_bytes = sum(f.stat().st_size for f in all_files if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_bytes / (1024 * 1024 * 1024)

    print(f"Destination Package: {dest_dir.resolve()}")
    print(f"Total Files:         {file_count}")
    print(f"Total Directories:   {dir_count}")
    print(f"Total Package Size:  {total_mb:.2f} MB ({total_gb:.3f} GB)")

    print("\nPackage creation complete.")

if __name__ == "__main__":
    build_training_package()
