"""
Next-Generation Alzheimer's AI Training Pipeline:
CBAM Spatial/Channel Attention + Multi-Scale Stem + Anatomical Depth Encoding + Focal Loss.

Target: Achieve >=80% Validation and Blind Test Patient Accuracy.
Architecture:
- Parallel Multi-Scale Stem (3x3 fine, 5x5 hippocampus, 7x7 ventricles)
- ResNet-18 Feature Extractor + CBAM (Convolutional Block Attention Module)
- 5-Dim Anatomical/Clinical Tabular MLP: [Norm_Age, Gender, Norm_MMSE, Has_MMSE, Slice_Depth_Z]
- Weighted Focal Loss with Prioritized MCI Focus: alpha=[0.80, 1.40, 1.80], gamma=2.0
- Gaussian Depth Soft-Voting Patient Aggregation
"""

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -------------------------------------------------------------------------
# 1. Environment & Paths
# -------------------------------------------------------------------------
PROJECT_DIR = '/content/alzheimers_joint_project'
DRIVE_FOLDER = '/content/drive/MyDrive/Alzhiemers detection'

if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)
    BASE_DIR = Path(PROJECT_DIR)
else:
    BASE_DIR = Path(".")

SAVE_DIR = Path(f"{DRIVE_FOLDER}/Results") if os.path.exists(DRIVE_FOLDER) else BASE_DIR / "models/checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Hardware Accelerator: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"📁 Model Checkpoint Save Location: {SAVE_DIR}")

# -------------------------------------------------------------------------
# 2. CBAM (Convolutional Block Attention Module)
# -------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return scale * x


class CBAMBlock(nn.Module):
    """Combines Channel and Spatial Attention to localize hippocampal atrophy."""
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# -------------------------------------------------------------------------
# 3. Model Architecture: Multi-Scale Stem + CBAM + Depth-Aware Clinical MLP
# -------------------------------------------------------------------------
class MultiScaleStem(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.b2 = nn.Sequential(nn.Conv2d(1, 24, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.b3 = nn.Sequential(nn.Conv2d(1, 24, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.fuse = nn.Conv2d(64, out_channels, 1, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))


class CBAMMultiKernelResNet(nn.Module):
    def __init__(self, num_classes=3, clinical_dim=5):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.multi_stem = MultiScaleStem(out_channels=64)
        self.bn1, self.relu, self.maxpool = base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        
        # Spatial-Channel Attention on top layer
        self.cbam = CBAMBlock(512)
        self.avgpool = base.avgpool

        # 5-Dim Clinical MLP (incorporating anatomical slice depth Z)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.fusion_head = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(512 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, slices, clinical):
        x = self.maxpool(self.relu(self.bn1(self.multi_stem(slices))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        
        v = torch.flatten(x, 1)
        c = self.tabular_mlp(clinical)
        return self.fusion_head(torch.cat([v, c], dim=1))

# -------------------------------------------------------------------------
# 4. Focal Loss with Prioritized MCI Weighting
# -------------------------------------------------------------------------
class PriorityFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha  # [w_CN, w_MCI, w_AD]
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

# -------------------------------------------------------------------------
# 5. Dataset with Slice Depth Encoding & Exact Normalization
# -------------------------------------------------------------------------
CDR_MAP = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 2}

class DepthAwareMultimodalDataset(Dataset):
    def __init__(self, manifest_csv, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.samples = []

        for _, row in self.df.iterrows():
            p_id = row['patient_id']
            p_dir = Path(str(row['future_processed_path']).replace("\\", "/"))
            if not p_dir.is_absolute():
                p_dir = BASE_DIR / p_dir

            raw_cdr = float(row['CDR']) if pd.notnull(row.get('CDR')) else 0.0
            label_3c = CDR_MAP.get(raw_cdr, 0)

            age = float(row.get('age', 75.0))
            norm_age = (age - 73.0) / 12.0
            gender = 1.0 if 'M' in str(row.get('gender', 'M')).upper() else 0.0
            has_mmse = 1.0 if pd.notnull(row.get('MMSE')) else 0.0
            mmse_val = float(row.get('MMSE', 27.0)) if has_mmse else 27.0
            norm_mmse = (mmse_val - 26.0) / 4.0

            slices_dir = p_dir / 'slices'
            if slices_dir.exists():
                slice_files = sorted(list(slices_dir.glob('axial_*.npy')))
                total_s = len(slice_files)
                for f in slice_files:
                    try:
                        idx = int(f.stem.split('_')[-1])
                    except:
                        idx = total_s // 2
                    norm_depth = idx / max(total_s, 1)

                    # 5-dim clinical vector: [age, gender, mmse, has_mmse, slice_depth_z]
                    clinical_vec = np.array([norm_age, gender, norm_mmse, has_mmse, norm_depth], dtype=np.float32)
                    self.samples.append((str(f), clinical_vec, label_3c, p_id, norm_depth))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, clin_vec, label, p_id, depth = self.samples[idx]
        try:
            arr = np.load(path).astype(np.float32)
            if arr.shape != (224, 224):
                arr = np.resize(arr, (224, 224))
        except:
            arr = np.zeros((224, 224), dtype=np.float32)

        tensor = torch.from_numpy(arr).unsqueeze(0)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, torch.from_numpy(clin_vec), label, p_id, depth

# -------------------------------------------------------------------------
# 6. Training Pipeline Function
# -------------------------------------------------------------------------
def train_model():
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=7),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    train_ds = DepthAwareMultimodalDataset(BASE_DIR / 'dataset/manifests/joint_train.csv', transform=train_transform)
    val_ds   = DepthAwareMultimodalDataset(BASE_DIR / 'dataset/manifests/joint_validation.csv')
    test_ds  = DepthAwareMultimodalDataset(BASE_DIR / 'dataset/manifests/joint_test.csv')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # Prioritize MCI: CN=0.80, MCI=1.40, AD=1.80
    alpha_weights = torch.tensor([0.80, 1.40, 1.80], dtype=torch.float32).to(device)
    print(f"⚖️ Applied Class Penalties: CN={alpha_weights[0]:.2f}, MCI={alpha_weights[1]:.2f} (Boosted!), AD={alpha_weights[2]:.2f}")

    model = CBAMMultiKernelResNet(num_classes=3, clinical_dim=5).to(device)
    criterion = PriorityFocalLoss(alpha=alpha_weights, gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    best_checkpoint_path = SAVE_DIR / "cbam_multikernel_resnet_best.pt"

    print("\n" + "=" * 85)
    print("🚀 TRAINING CBAM ATTENTION MULTI-SCALE RESNET (12 EPOCHS TO BREAK >80%)")
    print("=" * 85)

    for epoch in range(1, 13):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/12 [Train]", unit="batch")

        for slices, clinical, labels, _, _ in pbar:
            slices, clinical, labels = slices.to(device), clinical.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(slices, clinical)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * slices.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += slices.size(0)
            pbar.set_postfix({"Loss": f"{running_loss/total:.4f}", "SliceAcc": f"{correct/total*100:.1f}%"})

        scheduler.step()

        # Validation with Anatomical Gaussian Soft-Voting
        model.eval()
        val_probs, val_weights, val_gts = {}, {}, {}
        with torch.no_grad():
            for slices, clinical, labels, p_ids, depths in val_loader:
                slices, clinical = slices.to(device), clinical.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(slices, clinical)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                depths_np = depths.numpy()

                for i in range(slices.size(0)):
                    pid = p_ids[i]
                    if pid not in val_probs:
                        val_probs[pid] = []
                        val_weights[pid] = []
                        val_gts[pid] = labels[i].item()
                    
                    # Gaussian weight centered at 52% depth (hippocampus)
                    gw = max(math.exp(-(((depths_np[i] - 0.52) ** 2) / (2 * (0.18 ** 2)))), 0.15)
                    val_probs[pid].append(probs[i])
                    val_weights[pid].append(gw)

        correct_pts = 0
        for pid in val_probs:
            p_arr = np.array(val_probs[pid])
            w_arr = np.array(val_weights[pid])[:, None]
            weighted_p = np.sum(p_arr * w_arr, axis=0) / np.sum(w_arr)
            if np.argmax(weighted_p) == val_gts[pid]:
                correct_pts += 1

        val_acc = (correct_pts / len(val_probs)) * 100.0
        milestone = " 🔥 [>80% ACHIEVED!]" if val_acc >= 80.0 else ""
        new_best = " ⭐ (NEW BEST!)" if val_acc > best_val_acc else ""

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_checkpoint_path))

        print(f"📊 Epoch {epoch:02d}/12 -> Train Slice Acc: {correct/total*100:.2f}% | Val Patient Acc: {val_acc:.2f}% ({correct_pts}/{len(val_probs)}){milestone}{new_best}\n")

    # -------------------------------------------------------------------------
    # 7. Evaluate Best Checkpoint on 50 Blind Test Patients
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("🎯 BENCHMARKING BEST CHECKPOINT ON 50 UNSEEN BLIND TEST PATIENTS")
    print("=" * 85)

    model.load_state_dict(torch.load(str(best_checkpoint_path), map_location=device))
    model.eval()

    test_probs, test_weights, test_gts = {}, {}, {}
    with torch.no_grad():
        for slices, clinical, labels, p_ids, depths in tqdm(test_loader, desc="Evaluating Test Cohort"):
            slices, clinical = slices.to(device), clinical.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(slices, clinical)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            depths_np = depths.numpy()

            for i in range(slices.size(0)):
                pid = p_ids[i]
                if pid not in test_probs:
                    test_probs[pid] = []
                    test_weights[pid] = []
                    test_gts[pid] = labels[i].item()
                gw = max(math.exp(-(((depths_np[i] - 0.52) ** 2) / (2 * (0.18 ** 2)))), 0.15)
                test_probs[pid].append(probs[i])
                test_weights[pid].append(gw)

    y_true = []
    y_pred = []
    for pid in test_probs:
        y_true.append(test_gts[pid])
        p_arr = np.array(test_probs[pid])
        w_arr = np.array(test_weights[pid])[:, None]
        weighted_p = np.sum(p_arr * w_arr, axis=0) / np.sum(w_arr)
        y_pred.append(np.argmax(weighted_p))

    test_acc = accuracy_score(y_true, y_pred) * 100.0
    correct_test = sum(np.array(y_true) == np.array(y_pred))
    print(f"\n🏆 FINAL BLIND TEST PATIENT ACCURACY: {test_acc:.2f}% ({correct_test}/{len(y_true)})")
    print(classification_report(y_true, y_pred, target_names=['Cognitively Normal (CN)', 'Very Mild / MCI', 'Mild-to-Mod AD'], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("📋 Confusion Matrix:")
    print("                  Pred CN   Pred MCI   Pred AD   Total Actual")
    for idx, row in enumerate(cm):
        cname = ['CN', 'MCI', 'AD'][idx]
        print(f"  Actual {cname:<8} {row[0]:<9} {row[1]:<10} {row[2]:<9} {np.sum(row):<6}")

if __name__ == "__main__":
    train_model()
