"""
Sanity Training Verification Script for Google Colab / GPU Environment.

Purpose:
Executes a fast 1-epoch smoke test on a small subset of training and validation data to verify:
1. Dataset & DataLoader loading
2. Tensor shape compatibility [B, 1, 224, 224]
3. Model forward pass, loss calculation, backpropagation, and optimizer step
4. Patient-level mean probability aggregation
5. Validation evaluation and metric computation
6. Model checkpoint serialization to models/checkpoints/sanity_checkpoint.pt

NOTE: This is a smoke/sanity test only, not the full research experiment.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd

from models.cnn_2d import EfficientNetB02D
from training.dataset import Oasis2DSliceDataset
from training.patient_aggregator import PatientAggregator
from training.evaluator import ModelEvaluator

def run_sanity_train():
    print("==================================================")
    print("      STARTING SANITY TRAINING SMOKE TEST         ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Execution Device: {device}")

    train_csv = "dataset/manifests/train.csv"
    val_csv = "dataset/manifests/validation.csv"

    # Load Full Datasets
    full_train_ds = Oasis2DSliceDataset(train_csv, axis="axial")
    full_val_ds = Oasis2DSliceDataset(val_csv, axis="axial")

    # Select a subset of slices (e.g. first 64 slices) for quick smoke testing
    train_subset = Subset(full_train_ds, indices=list(range(min(64, len(full_train_ds)))))
    val_subset = Subset(full_val_ds, indices=list(range(min(32, len(full_val_ds)))))

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    print(f">> Train Subset Slices: {len(train_subset)} | Val Subset Slices: {len(val_subset)}")

    # Initialize 2D EfficientNet-B0
    model = EfficientNetB02D(num_classes=4, in_channels=1, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 1. Train Step
    model.train()
    running_loss = 0.0
    total_samples = 0

    print(">> Running 1 Sanity Training Epoch...")
    for batch_idx, (slices, labels, meta) in enumerate(train_loader):
        slices = slices.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(slices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * slices.size(0)
        total_samples += slices.size(0)

    train_loss = running_loss / max(total_samples, 1)
    print(f">> Sanity Train Loss: {train_loss:.4f} across {total_samples} slices")

    # 2. Validation & Patient Aggregation Step
    model.eval()
    aggregator = PatientAggregator(method="mean")
    val_loss = 0.0
    val_samples = 0

    print(">> Running Validation & Patient Aggregation...")
    with torch.no_grad():
        for slices, labels, meta in val_loader:
            slices = slices.to(device)
            labels = labels.to(device)

            outputs = model(slices)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * slices.size(0)
            val_samples += slices.size(0)

            probs = torch.softmax(outputs, dim=-1)
            for i in range(slices.size(0)):
                p_id = meta["patient_id"][i]
                gt = int(labels[i].item())
                raw_cdr = float(meta["raw_cdr"][i].item()) if isinstance(meta["raw_cdr"][i], torch.Tensor) else float(meta["raw_cdr"][i])
                aggregator.add_slice_prediction(
                    patient_id=p_id,
                    slice_prob=probs[i],
                    ground_truth=gt,
                    raw_cdr=raw_cdr
                )

    patient_records = aggregator.compute_patient_predictions()
    evaluator = ModelEvaluator(num_classes=4)
    metrics = evaluator.evaluate_patient_predictions(patient_records)

    print(f">> Validation Slices: {val_samples} | Slice Loss: {val_loss / max(val_samples, 1):.4f}")
    print(f">> Patients Evaluated: {len(patient_records)} | Patient Accuracy: {metrics['patient_accuracy'] * 100:.2f}%")

    # 3. Checkpoint Writing Step
    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sanity_ckpt_path = ckpt_dir / "sanity_checkpoint.pt"

    torch.save({
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sanity_metrics": metrics
    }, str(sanity_ckpt_path))

    print(f">> Checkpoint serialization verified: {sanity_ckpt_path} ({sanity_ckpt_path.stat().st_size} bytes)")
    print("\n==================================================")
    print("      SANITY SMOKE TEST: ALL CHECKS PASSED        ")
    print("==================================================")

if __name__ == "__main__":
    run_sanity_train()
