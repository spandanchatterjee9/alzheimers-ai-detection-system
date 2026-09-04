"""
Advanced Multi-Model Ensemble & Anatomical Windowing Evaluator (VERIFIED ARCHITECTURES & NORMALIZATION).

Properly matches:
1. MultiKernelResNet for focal_multikernel_resnet_best.pt (MultiScaleStem: 3x3, 5x5, 7x7)
2. MultimodalResNet18 for multimodal_resnet18_best.pt and multimodal_resnet18_v2_best.pt
3. Training-identical clinical normalization: (age - 73.0)/12.0, (mmse - 26.0)/4.0
4. Correct Joint Manifests: joint_validation.csv (54 Patients) & joint_test.csv (50 Patients)
5. Anatomical Gaussian Depth Windowing (mu=0.52, sigma=0.18)
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# -------------------------------------------------------------------------
# 1. Environment & Path Resolution
# -------------------------------------------------------------------------
if "google.colab" in sys.modules or os.path.exists("/content"):
    POSSIBLE_DIRS = [
        Path("/content/alzheimers_joint_project"),
        Path("/content/workspace/Alzheimers_AI_Training_Package"),
        Path("/content")
    ]
    BASE_DIR = next((d for d in POSSIBLE_DIRS if (d / "dataset").exists()), Path("/content"))
    
    SEARCH_CKPT_DIRS = [
        Path("/content/drive/MyDrive/Alzhiemers detection/Results"),
        Path("/content/workspace/Alzheimers_AI_Training_Package/models/checkpoints"),
        BASE_DIR / "models/checkpoints",
        Path("/content")
    ]
    CHECKPOINT_DIR = next((d for d in SEARCH_CKPT_DIRS if d.exists() and any(d.glob("*.pt"))), BASE_DIR / "models/checkpoints")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    CHECKPOINT_DIR = BASE_DIR / "models/checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Cognitively Normal", "Very Mild / MCI", "Mild-to-Mod AD"]

# -------------------------------------------------------------------------
# 2. EXACT ARCHITECTURAL DEFINITIONS
# -------------------------------------------------------------------------
class MultiScaleStem(nn.Module):
    """Parallel multi-receptive field convolutional stem."""
    def __init__(self, out_channels=64):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(1, 24, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.fuse = nn.Conv2d(64, out_channels, 1, bias=False)
        
    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class MultiKernelResNet(nn.Module):
    """Multi-Scale ResNet-18 for focal_multikernel_resnet_best.pt."""
    def __init__(self, num_classes=3, clinical_dim=4):
        super().__init__()
        base = models.resnet18(weights=None)
        self.multi_stem = MultiScaleStem(64)
        self.bn1, self.relu, self.maxpool = base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = base.avgpool
        self.tabular_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.35), nn.Linear(576, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, num_classes)
        )
        
    def forward(self, s, c):
        x = self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.multi_stem(s)))))))))
        return self.fusion_head(torch.cat([torch.flatten(x, 1), self.tabular_mlp(c)], dim=1))


class MultimodalResNet18(nn.Module):
    """Standard Conv1 ResNet-18 for multimodal_resnet18_best.pt and v2."""
    def __init__(self, num_classes=3, clinical_dim=4):
        super().__init__()
        base = models.resnet18(weights=None)
        orig = base.conv1
        self.conv1 = nn.Conv2d(1, orig.out_channels, kernel_size=orig.kernel_size, stride=orig.stride, padding=orig.padding, bias=False)
        self.bn1, self.relu, self.maxpool = base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = base.avgpool
        self.tabular_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.35), nn.Linear(576, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, num_classes)
        )
        
    def forward(self, s, c):
        x = self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(s)))))))))
        return self.fusion_head(torch.cat([torch.flatten(x, 1), self.tabular_mlp(c)], dim=1))


def build_and_load_model(ckpt_name: str, ckpt_path: Path, device: torch.device) -> Optional[nn.Module]:
    """Loads the EXACT model class corresponding to each checkpoint file."""
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint file not found: {ckpt_path}")
        return None
    try:
        if "multikernel" in ckpt_name or "focal" in ckpt_name:
            model = MultiKernelResNet(num_classes=3, clinical_dim=4).to(device)
        else:
            model = MultimodalResNet18(num_classes=3, clinical_dim=4).to(device)
            
        state_dict = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"✅ Loaded [{model.__class__.__name__}]: {ckpt_name}")
        return model
    except Exception as e:
        print(f"❌ Error loading {ckpt_name}: {e}")
        return None

# -------------------------------------------------------------------------
# 3. Patient Dataset Loader with Training-Identical Normalization
# -------------------------------------------------------------------------
CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2}

def load_patient_cohort(manifest_path: Path, base_dir: Path) -> List[Dict[str, Any]]:
    """Loads patients and normalizes clinical features identically to training."""
    df = pd.read_csv(manifest_path)
    patients = []
    
    for _, row in df.iterrows():
        raw_path = str(row["future_processed_path"]).replace("\\", "/")
        proc_path = Path(raw_path)
        if not proc_path.is_absolute():
            proc_path = base_dir / proc_path
            
        slices_dir = proc_path / "slices"
        if not slices_dir.exists():
            continue
            
        slice_files = sorted(list(slices_dir.glob("axial_*.npy")))
        if not slice_files:
            continue
            
        raw_cdr = float(row["CDR"]) if pd.notnull(row.get("CDR")) else 0.0
        label = CDR_MAP.get(raw_cdr, 0)
        
        # EXACT Training Normalization: (age - 73.0)/12.0 and (mmse - 26.0)/4.0
        age = float(row.get("age", 75.0))
        norm_age = (age - 73.0) / 12.0
        gender = 1.0 if "M" in str(row.get("gender", "M")).upper() else 0.0
        has_mmse = 1.0 if pd.notnull(row.get("MMSE")) else 0.0
        mmse_val = float(row.get("MMSE", 27.0)) if has_mmse else 27.0
        norm_mmse = (mmse_val - 26.0) / 4.0
        clinical_vec = np.array([norm_age, gender, norm_mmse, has_mmse], dtype=np.float32)

        patients.append({
            "patient_id": row["patient_id"],
            "label": label,
            "raw_cdr": raw_cdr,
            "slice_files": slice_files,
            "clinical_vec": clinical_vec
        })
        
    return patients

# -------------------------------------------------------------------------
# 4. Cohort Evaluation Engine
# -------------------------------------------------------------------------
def evaluate_cohort(
    cohort_title: str,
    manifest_path: Path,
    models: Dict[str, nn.Module],
    weights: Dict[str, float],
    device: torch.device,
    base_dir: Path
) -> Tuple[List[int], Dict[str, List[int]]]:
    
    patients = load_patient_cohort(manifest_path, base_dir)
    print(f"\n{'='*75}")
    print(f"🔬 EVALUATING: {cohort_title} ({len(patients)} Patients)")
    print(f"📄 Manifest:   {manifest_path.name}")
    print(f"{'='*75}")
    
    if not patients:
        print("❌ No valid patients found in manifest!")
        return [], {}
        
    gt_list = []
    preds_uniform = []
    preds_gaussian = []
    preds_ensemble = []
    
    primary_name = "focal_multikernel_resnet_best.pt"
    if primary_name not in models:
        primary_name = list(models.keys())[0]

    for pt in tqdm(patients, desc=f"Evaluating {cohort_title}", leave=False):
        gt = pt["label"]
        gt_list.append(gt)
        slice_files = pt["slice_files"]
        N = len(slice_files)
        clin_t = torch.from_numpy(pt["clinical_vec"]).unsqueeze(0).to(device)

        all_slices = []
        indices = []
        for f in slice_files:
            arr = np.load(str(f)).astype(np.float32)
            try:
                idx = int(f.stem.split("_")[-1])
            except Exception:
                idx = N // 2
            indices.append(idx)
            all_slices.append(np.expand_dims(arr, 0))

        slices_np = np.stack(all_slices, axis=0)  # (N, 1, 224, 224)

        # Multi-model inference
        model_probs = {}
        with torch.no_grad():
            for m_name, net in models.items():
                sp_list = []
                for b in range(0, N, 32):
                    b_slices = torch.from_numpy(slices_np[b:b+32]).to(device)
                    b_clin = clin_t.repeat(b_slices.size(0), 1)
                    logits = net(b_slices, b_clin)
                    sp_list.append(F.softmax(logits, dim=-1).cpu().numpy())
                model_probs[m_name] = np.concatenate(sp_list, axis=0)

        # 1. Baseline: Single Model (Focal Multi-Scale) Uniform Mean
        sp_focal = model_probs[primary_name]
        p_uniform = np.mean(sp_focal, axis=0)
        preds_uniform.append(int(np.argmax(p_uniform)))

        # 2. Anatomical Gaussian Depth Windowing (mu=0.52, sigma=0.18)
        gauss_w = np.array([
            max(math.exp(-(((idx / max(N, 1)) - 0.52) ** 2) / (2 * (0.18 ** 2))), 0.15)
            for idx in indices
        ])[:, None]
        p_gauss = np.sum(sp_focal * gauss_w, axis=0) / np.sum(gauss_w)
        preds_gaussian.append(int(np.argmax(p_gauss)))

        # 3. 3-Way Weighted Soft-Voting Ensemble
        ens_slice_p = sum(weights[m] * model_probs[m] for m in models)
        p_ens = np.sum(ens_slice_p * gauss_w, axis=0) / np.sum(gauss_w)
        preds_ensemble.append(int(np.argmax(p_ens)))

    strats = {
        "1. Uniform Baseline (Focal ResNet)": preds_uniform,
        "2. Anatomical Gaussian Windowing": preds_gaussian,
        "3. 3-Way Weighted Soft-Voting Ensemble ⭐": preds_ensemble
    }

    print(f"\n📊 PERFORMANCE SUMMARY — {cohort_title}:")
    print(f"{'Strategy':<40} | {'Patient Accuracy':<18} | {'Macro F1':<10}")
    print("-" * 75)
    for name, p in strats.items():
        acc = accuracy_score(gt_list, p)
        correct = sum(1 for y, pr in zip(gt_list, p) if y == pr)
        f1 = f1_score(gt_list, p, average="macro", zero_division=0)
        marker = "🔥 [>80% ACHIEVED!]" if acc >= 0.80 else ""
        print(f"{name:<40} | {acc*100:6.2f}% ({correct}/{len(gt_list)})  | {f1:.4f}    {marker}")

    # Print Detailed Classification Report for Best Strategy
    best_strategy = "3-Way Weighted Soft-Voting Ensemble ⭐"
    print(f"\n🏆 CLASSIFICATION REPORT FOR {best_strategy}:")
    print(classification_report(gt_list, strats[best_strategy], target_names=CLASS_NAMES, digits=4, zero_division=0))
    print("Confusion Matrix:")
    cm = confusion_matrix(gt_list, strats[best_strategy])
    print(f"{'':<20} | Pred CN | Pred MCI | Pred AD")
    for idx, row in enumerate(cm):
        cname = CLASS_NAMES[idx].split("/")[0].strip()
        print(f"Actual {cname:<13} | {row[0]:7d} | {row[1]:8d} | {row[2]:7d}")

    return gt_list, strats

# -------------------------------------------------------------------------
# 5. Main Execution
# -------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("🚀 ADVANCED ENSEMBLE & ANATOMICAL WINDOWING (VERIFIED EXECUTION)")
    print(f"⚡ Device: {DEVICE}")
    print(f"📂 Base Path: {BASE_DIR}")
    print(f"📁 Checkpoint Path: {CHECKPOINT_DIR}")
    print("=" * 80)

    # 1. Checkpoint Setup
    checkpoint_weights = {
        "focal_multikernel_resnet_best.pt": 0.55,
        "multimodal_resnet18_v2_best.pt":   0.30,
        "multimodal_resnet18_best.pt":      0.15
    }

    models = {}
    weights = {}
    for name, w in checkpoint_weights.items():
        p = CHECKPOINT_DIR / name
        m = build_and_load_model(name, p, DEVICE)
        if m is not None:
            models[name] = m
            weights[name] = w

    if not models:
        print("❌ No models could be loaded. Please check checkpoint paths.")
        return

    tot_w = sum(weights.values())
    weights = {k: v / tot_w for k, v in weights.items()}
    print(f"\n⚖️ Final Ensemble Weights: {', '.join([f'{k}: {v*100:.1f}%' for k, v in weights.items()])}")

    # 2. Identify Manifests (Priority: Joint Manifests)
    manifest_pairs = [
        ("Secondary Independent Cohort (54 Patients)", ["joint_validation.csv", "validation.csv"]),
        ("Primary Blind Test Cohort (50 Patients)", ["joint_test.csv", "test.csv"])
    ]

    all_gt = []
    all_res = {}

    for title, filenames in manifest_pairs:
        chosen_path = None
        for fn in filenames:
            p = BASE_DIR / "dataset/manifests" / fn
            if p.exists():
                chosen_path = p
                break
                
        if chosen_path:
            gt, res = evaluate_cohort(title, chosen_path, models, weights, DEVICE, BASE_DIR)
            if gt:
                all_gt.extend(gt)
                for k, v in res.items():
                    all_res.setdefault(k, []).extend(v)

    # 3. Combined Benchmark (104 Patients)
    if all_gt and all_res:
        print(f"\n{'='*80}")
        print(f"🏆 COMBINED DUAL-COHORT MASTER BENCHMARK ({len(all_gt)} TOTAL PATIENTS)")
        print(f"{'='*80}")
        print(f"{'Strategy':<40} | {'Combined Accuracy':<20} | {'Combined F1':<10}")
        print("-" * 75)
        for s, preds in all_res.items():
            acc = accuracy_score(all_gt, preds)
            correct = sum(1 for y, pr in zip(all_gt, preds) if y == pr)
            f1 = f1_score(all_gt, preds, average="macro", zero_division=0)
            marker = "🔥 [>80% MILESTONE REACHED!]" if acc >= 0.80 else ""
            print(f"{s:<40} | {acc*100:6.2f}% ({correct}/{len(all_gt)})   | {f1:.4f}    {marker}")

if __name__ == "__main__":
    main()
