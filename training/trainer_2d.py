"""
2D Deep Learning Training Engine for Alzheimer's-Related Dementia Stage Classification.

Protocol:
- Trains 2D EfficientNet-B0 (or 2D ResNet-18) on 224x224 grayscale MRI slices (Axial plane).
- Evaluates model at both Slice Level and Patient Level.
- Applies Mean Probability Aggregation across all valid slices of a patient to yield patient-level CDR predictions.
- Strict patient-wise splitting is enforced (zero patient leakage between train, val, and test splits).
- Supports Mixed Precision (AMP), Class-Weighted Loss, and Checkpoint Serialization.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler, autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    GradScaler = object

from models.cnn_2d import EfficientNetB02D, ResNet182D
from training.dataset import Oasis2DSliceDataset, create_2d_dataloaders, compute_training_class_weights
from training.patient_aggregator import PatientAggregator
from training.evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("training.trainer_2d")


class Oasis2DTrainer:
    """
    Complete Trainer Engine for 2D Slice-Based Dementia Stage Classification.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = "configs/training.yaml"
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Oasis2DTrainer.")

        # Load YAML config if provided
        if config is None and config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config or {}

        # Core Hyperparameters
        self.num_classes = self.config.get("num_classes", 4)
        self.in_channels = self.config.get("in_channels", 1)
        self.image_size = self.config.get("image_size", 224)
        self.axis = self.config.get("axis", "axial").lower()
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = float(self.config.get("learning_rate", 1e-4))
        self.weight_decay = float(self.config.get("weight_decay", 1e-2))
        self.epochs = self.config.get("epochs", 20)
        self.num_workers = self.config.get("num_workers", 2)
        self.use_amp = self.config.get("mixed_precision", True) and torch.cuda.is_available()
        self.use_class_weights = self.config.get("use_class_weights", True)

        # Device Setup
        device_req = self.config.get("device", "auto")
        if device_req == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_req)

        logger.info(f"Training initialized on Device: {self.device} (AMP: {self.use_amp})")

        # Dataset Paths
        self.train_csv = Path(self.config.get("train_csv", "dataset/manifests/train.csv"))
        self.val_csv = Path(self.config.get("val_csv", "dataset/manifests/validation.csv"))
        self.test_csv = Path(self.config.get("test_csv", "dataset/manifests/test.csv"))
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "models/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate Model
        model_name = self.config.get("model", "efficientnet_b0").lower()
        pretrained = self.config.get("pretrained", True)
        if "resnet" in model_name:
            self.model = ResNet182D(num_classes=self.num_classes, in_channels=self.in_channels, pretrained=pretrained).to(self.device)
        else:
            self.model = EfficientNetB02D(num_classes=self.num_classes, in_channels=self.in_channels, pretrained=pretrained).to(self.device)

        # Loss Function & Class Weights
        if self.use_class_weights and self.train_csv.exists():
            class_weights = compute_training_class_weights(self.train_csv, self.num_classes).to(self.device)
            logger.info(f"Applying Training Class Weights to CrossEntropyLoss: {class_weights.tolist()}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer & Scaler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        self.scaler = GradScaler(enabled=self.use_amp)

        # DataLoaders
        self.train_loader, self.val_loader, self.test_loader = create_2d_dataloaders(
            train_csv=self.train_csv,
            val_csv=self.val_csv,
            test_csv=self.test_csv if self.test_csv.exists() else None,
            axis=self.axis,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

        self.evaluator = ModelEvaluator(num_classes=self.num_classes)
        self.best_patient_accuracy = 0.0
        self.best_macro_f1 = 0.0

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs one epoch of slice-level training."""
        self.model.train()
        running_loss = 0.0
        correct_slices = 0
        total_slices = 0

        for slices, labels, _ in self.train_loader:
            slices = slices.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(slices)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(slices)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * slices.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_slices += (preds == labels).sum().item()
            total_slices += slices.size(0)

        epoch_loss = running_loss / max(total_slices, 1)
        epoch_acc = correct_slices / max(total_slices, 1)
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader, split_name: str = "Validation") -> Dict[str, Any]:
        """
        Evaluates model with Patient-Level Mean Probability Aggregation.
        """
        self.model.eval()
        running_loss = 0.0
        total_slices = 0
        aggregator = PatientAggregator(method="mean")

        with torch.no_grad():
            for slices, labels, meta in dataloader:
                slices = slices.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(slices)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(slices)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * slices.size(0)
                total_slices += slices.size(0)

                # Softmax probabilities
                probs = torch.softmax(outputs, dim=-1)

                # Add each slice to patient aggregator
                for i in range(slices.size(0)):
                    p_id = meta["patient_id"][i]
                    p_prob = probs[i]
                    gt = int(labels[i].item())
                    raw_cdr = float(meta["raw_cdr"][i].item()) if isinstance(meta["raw_cdr"][i], torch.Tensor) else float(meta["raw_cdr"][i])
                    aggregator.add_slice_prediction(
                        patient_id=p_id,
                        slice_prob=p_prob,
                        ground_truth=gt,
                        raw_cdr=raw_cdr
                    )

        patient_records = aggregator.compute_patient_predictions()
        metrics = self.evaluator.evaluate_patient_predictions(patient_records)
        metrics["slice_loss"] = float(running_loss / max(total_slices, 1))
        metrics["patient_records"] = patient_records

        logger.info(
            f"[{split_name}] Slices: {total_slices} | Slice Loss: {metrics['slice_loss']:.4f} | "
            f"Patients: {metrics['total_patients']} | Patient Accuracy: {metrics['patient_accuracy'] * 100:.2f}% | "
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )
        return metrics

    def save_checkpoint(self, path: Union[str, Path], epoch: int, metrics: Dict[str, Any]) -> None:
        """Saves reproduction-ready model checkpoint."""
        checkpoint_payload = {
            "epoch": epoch,
            "model_architecture": self.config.get("model", "efficientnet_b0"),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "patient_accuracy": metrics.get("patient_accuracy", 0.0),
            "macro_f1": metrics.get("macro_f1", 0.0),
            "class_map": {"0.0": 0, "0.5": 1, "1.0": 2, "2.0": 3},
            "training_config": self.config
        }
        torch.save(checkpoint_payload, str(path))
        logger.info(f"Saved best model checkpoint to: {path}")

    def train(self) -> Dict[str, Any]:
        """Full training loop execution."""
        logger.info(f"Starting 2D Training for {self.epochs} epochs...")
        history: List[Dict[str, Any]] = []
        best_ckpt_path = self.checkpoint_dir / "efficientnet_b0_best.pt"

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, split_name="Validation")
            self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_slice_accuracy": train_acc,
                "val_slice_loss": val_metrics["slice_loss"],
                "val_patient_accuracy": val_metrics["patient_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"]
            }
            history.append(epoch_record)

            logger.info(
                f"Epoch [{epoch}/{self.epochs}] Train Loss: {train_loss:.4f} (Acc: {train_acc*100:.2f}%) | "
                f"Val Patient Acc: {val_metrics['patient_accuracy']*100:.2f}% | Val Macro F1: {val_metrics['macro_f1']:.4f}"
            )

            # Checkpoint on best validation patient accuracy
            if val_metrics["patient_accuracy"] >= self.best_patient_accuracy:
                self.best_patient_accuracy = val_metrics["patient_accuracy"]
                self.best_macro_f1 = val_metrics["macro_f1"]
                self.save_checkpoint(best_ckpt_path, epoch, val_metrics)

        logger.info(f"Training Complete. Best Validation Patient Accuracy: {self.best_patient_accuracy * 100:.2f}%")
        return {
            "best_patient_accuracy": self.best_patient_accuracy,
            "best_macro_f1": self.best_macro_f1,
            "checkpoint_path": str(best_ckpt_path),
            "history": history
        }


def main():
    parser = argparse.ArgumentParser(description="2D CNN Training for Alzheimer's Stage Classification")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training YAML configuration")
    parser.add_argument("--model", type=str, default=None, help="Model architecture: 'resnet18' or 'efficientnet_b0'")
    parser.add_argument("--train_manifest", "--train_csv", type=str, default=None, help="Path to training manifest CSV")
    parser.add_argument("--val_manifest", "--val_csv", type=str, default=None, help="Path to validation manifest CSV")
    parser.add_argument("--test_manifest", "--test_csv", type=str, default=None, help="Path to test manifest CSV")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", "--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    # Load base config if exists
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # Override with CLI args
    if args.model:
        config["model"] = args.model
    if args.train_manifest:
        config["train_csv"] = args.train_manifest
    if args.val_manifest:
        config["val_csv"] = args.val_manifest
    if args.test_manifest:
        config["test_csv"] = args.test_manifest
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device

    trainer = Oasis2DTrainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()

