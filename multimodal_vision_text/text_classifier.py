"""
Clinical Text Classification and Embedding Extractor using Bio_ClinicalBERT.
Trains a Transformer on patient clinical consultation reports (CN vs MCI vs AD)
and extracts 768-dimensional text feature vectors for Multimodal Fusion.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModel


DEFAULT_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


class ClinicalTextDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 384,
        text_column: str = "clinical_report"
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[self.text_column])
        label = int(row["label_3class"])
        patient_id = str(row["patient_id"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id
        }


class PubMedBERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = 3,
        dropout_p: float = 0.3,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )

    def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts attention-weighted mean pooled 768-dimensional text embedding
        across all non-padding tokens.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.get_embedding(input_ids, attention_mask)
        logits = self.classifier(embedding)
        return logits, embedding


# Backward compatibility alias
ClinicalBERTClassifier = PubMedBERTClassifier
