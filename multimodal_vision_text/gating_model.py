"""
Dynamic Gating Multimodal Fusion Network.
Supports:
  1. Dual-Modal Gating: Vision (512-dim) + Clinical Text (768-dim)
  2. Tri-Modal Gating (Strategy 1 & 2):
     - Vision: 512-dim ResNet visual features (hippocampal & ventricular geometry)
     - Tabular: 6-dim standardized clinical biomarkers (Age, Sex, MMSE, nWBV, eTIV, has_mmse)
     - Text: 768-dim PubMedBERT embeddings
     - Temperature Scaling (Strategy 2): Learnable temperature calibration per modality
     - Entropy-Aware Gating: Automatically downweights uncertain/ambiguous modalities
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class DynamicGatingFusion(nn.Module):
    """Dual-Modal Dynamic Gating: Vision (512-dim) + Clinical Text (768-dim)."""
    def __init__(
        self,
        vision_dim: int = 512,
        text_dim: int = 768,
        num_classes: int = 3,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.num_classes = num_classes

        # Individual expert heads
        self.vision_head = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.text_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        # Dynamic Gating Network: Unconstrained sigmoid with wide patient-specific dynamic range
        self.gate_net = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        vision_feats: torch.Tensor,
        text_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_logits = self.vision_head(vision_feats)
        t_logits = self.text_head(text_feats)

        # Dynamic gate g in [0, 1]
        concat_feats = torch.cat([vision_feats, text_feats], dim=-1)
        g = self.gate_net(concat_feats)

        # Convex gated combination: preserves mathematical stability
        fused_logits = g * v_logits + (1.0 - g) * t_logits
        return fused_logits, g


class TriModalDynamicGatingFusion(nn.Module):
    """
    🚀 Strategy 1 & 2: Tri-Modal Dynamic Gating with Temperature & Entropy Calibration.
    Fuses:
      1. Vision (512-dim): 3D/2D MRI structural features
      2. Tabular (6-dim): Age, Sex, MMSE, nWBV, eTIV, has_mmse
      3. Text (768-dim): PubMedBERT clinical consultation embeddings
    Features:
      - Learnable Temperature Scaling (T_v, T_tab, T_t) for logit calibration
      - Dynamic 3-Way Softmax Gating (g_v + g_tab + g_t = 1.0)
      - Entropy Penalty: Dampens predictions from modalities with high diagnostic ambiguity
    """
    def __init__(
        self,
        vision_dim: int = 512,
        tabular_dim: int = 6,
        text_dim: int = 768,
        num_classes: int = 3,
        hidden_dim: int = 128,
        use_entropy_penalty: bool = True
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.tabular_dim = tabular_dim
        self.text_dim = text_dim
        self.num_classes = num_classes
        self.use_entropy_penalty = use_entropy_penalty

        # 1. Specialized Expert Branches
        self.vision_head = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.text_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

        # Tabular representation projector for gate network
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.ReLU()
        )

        # 2. Tri-Modal Dynamic Gating Network (Outputs 3 simplex weights [g_v, g_tab, g_t])
        gate_input_dim = vision_dim + 32 + text_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 3)  # 3-way logits for [vision, tabular, text]
        )

        # 3. Learnable Modality Temperatures (initialized to 1.0)
        # Scaled via softplus to ensure strictly positive temperatures
        self.log_temp_v = nn.Parameter(torch.zeros(1))
        self.log_temp_tab = nn.Parameter(torch.zeros(1))
        self.log_temp_t = nn.Parameter(torch.zeros(1))

    @property
    def temp_v(self) -> torch.Tensor:
        return F.softplus(self.log_temp_v) + 0.5

    @property
    def temp_tab(self) -> torch.Tensor:
        return F.softplus(self.log_temp_tab) + 0.5

    @property
    def temp_t(self) -> torch.Tensor:
        return F.softplus(self.log_temp_t) + 0.5

    def forward(
        self,
        vision_feats: torch.Tensor,
        tabular_feats: torch.Tensor,
        text_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Forward through individual expert branches
        z_v = self.vision_head(vision_feats)
        z_tab = self.tabular_mlp(tabular_feats)
        z_t = self.text_head(text_feats)

        # Temperature-calibrated probabilities
        p_v = F.softmax(z_v / self.temp_v, dim=-1)
        p_tab = F.softmax(z_tab / self.temp_tab, dim=-1)
        p_t = F.softmax(z_t / self.temp_t, dim=-1)

        # Tri-Modal Gate computation
        tab_proj = self.tabular_proj(tabular_feats)
        concat_feats = torch.cat([vision_feats, tab_proj, text_feats], dim=-1)
        gate_logits = self.gate_net(concat_feats)
        raw_gates = F.softmax(gate_logits, dim=-1)  # (B, 3): [g_v, g_tab, g_t]

        g_v = raw_gates[:, 0:1]
        g_tab = raw_gates[:, 1:2]
        g_t = raw_gates[:, 2:3]

        # Entropy-Weighted Calibration (Strategy 2)
        if self.use_entropy_penalty:
            eps = 1e-7
            h_v = -torch.sum(p_v * torch.log(p_v + eps), dim=-1, keepdim=True)
            h_tab = -torch.sum(p_tab * torch.log(p_tab + eps), dim=-1, keepdim=True)
            h_t = -torch.sum(p_t * torch.log(p_t + eps), dim=-1, keepdim=True)

            # Max possible entropy for 3 classes is ln(3) ~ 1.0986
            max_h = math.log(3.0)
            conf_v = torch.clamp(1.0 - (h_v / max_h), min=0.1, max=1.0)
            conf_tab = torch.clamp(1.0 - (h_tab / max_h), min=0.1, max=1.0)
            conf_t = torch.clamp(1.0 - (h_t / max_h), min=0.1, max=1.0)

            # Modulate gates by certainty
            adj_v = g_v * conf_v
            adj_tab = g_tab * conf_tab
            adj_t = g_t * conf_t

            total_weight = adj_v + adj_tab + adj_t + eps
            w_v = adj_v / total_weight
            w_tab = adj_tab / total_weight
            w_t = adj_t / total_weight
        else:
            w_v, w_tab, w_t = g_v, g_tab, g_t

        # Final Fused Probabilities
        fused_probs = w_v * p_v + w_tab * p_tab + w_t * p_t
        fused_logits = torch.log(fused_probs + 1e-7)

        gate_dict = {
            "weight_vision": w_v,
            "weight_tabular": w_tab,
            "weight_text": w_t,
            "temp_v": self.temp_v.detach(),
            "temp_tab": self.temp_tab.detach(),
            "temp_t": self.temp_t.detach()
        }
        return fused_logits, raw_gates, gate_dict


def test_gating_network():
    batch_size = 4
    v = torch.randn(batch_size, 512)
    tab = torch.tensor([
        [74.0, 1.0, 26.0 / 30.0, 0.72, 1450.0 / 2000.0, 1.0],
        [82.0, 0.0, 18.0 / 30.0, 0.65, 1380.0 / 2000.0, 1.0],
        [68.0, 1.0, 29.0 / 30.0, 0.79, 1520.0 / 2000.0, 1.0],
        [75.0, 0.0, 24.0 / 30.0, 0.70, 1410.0 / 2000.0, 1.0]
    ], dtype=torch.float32)
    t = torch.randn(batch_size, 768)

    print("1. Testing Dual-Modal Gating...")
    dual_model = DynamicGatingFusion()
    logits, gate = dual_model(v, t)
    assert logits.shape == (batch_size, 3)
    assert gate.shape == (batch_size, 1)
    print("   Dual-Modal Gating OK!")

    print("2. Testing Tri-Modal Gating with Temperature & Entropy Scaling...")
    tri_model = TriModalDynamicGatingFusion(use_entropy_penalty=True)
    fused_logits, raw_gates, gate_info = tri_model(v, tab, t)
    assert fused_logits.shape == (batch_size, 3)
    assert raw_gates.shape == (batch_size, 3)
    print(f"   Vision Weights:  {gate_info['weight_vision'].squeeze().tolist()}")
    print(f"   Tabular Weights: {gate_info['weight_tabular'].squeeze().tolist()}")
    print(f"   Text Weights:    {gate_info['weight_text'].squeeze().tolist()}")
    print(f"   Temperatures: V={gate_info['temp_v'].item():.2f}, Tab={gate_info['temp_tab'].item():.2f}, T={gate_info['temp_t'].item():.2f}")
    print("[OK] Tri-Modal Dynamic Gating with Calibration verified successfully!")


if __name__ == "__main__":
    test_gating_network()

