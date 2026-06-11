#!/usr/bin/env python3
"""Train a compact Transformer multi-label head from Feather + embedding NPZ files.

This reuses the same input construction as train_multilabel_from_feather_embeddings.py:
six row-level base features concatenated with flattened Aurora encoder embeddings.
Only the classifier changes from an MLP to a small Transformer over embedding chunks.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import train_multilabel_from_feather_embeddings as base


OUTPUT_DIR = str(base.PROJECT_ROOT / "model_outputs_transformer")
LOG_DIR = str(base.PROJECT_ROOT / "logs")
DEFAULT_TRANSFORMER_CONFIG = [256, 2, 4, 512, 512]


def _resolve_transformer_config(
    input_dim: int,
    hidden_dims: Optional[Sequence[int]] = None,
) -> List[int]:
    """Return [d_model, layers, heads, feedforward_dim, token_feature_dim]."""
    if hidden_dims:
        dims = [int(d) for d in hidden_dims if int(d) > 0]
        if len(dims) == 4:
            return dims + [DEFAULT_TRANSFORMER_CONFIG[-1]]
        if len(dims) >= 5:
            return dims[:5]
        raise ValueError(
            "--hidden-dims for the transformer must be d_model,layers,heads,feedforward_dim "
            "or d_model,layers,heads,feedforward_dim,token_feature_dim"
        )
    return list(DEFAULT_TRANSFORMER_CONFIG)


class EmbeddingTransformerClassifier(nn.Module):
    """Small Transformer classifier for flattened Aurora embedding features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 40,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        config = _resolve_transformer_config(input_dim=input_dim, hidden_dims=hidden_dims)
        d_model, num_layers, num_heads, feedforward_dim, token_feature_dim = config
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.base_feature_dim = len(base.BASE_FEATURE_COLUMNS)
        self.embedding_dim = self.input_dim - self.base_feature_dim
        if self.embedding_dim <= 0:
            raise ValueError(
                f"input_dim={input_dim} must include {self.base_feature_dim} base features "
                "plus at least one embedding feature."
            )

        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.feedforward_dim = int(feedforward_dim)
        self.token_feature_dim = int(token_feature_dim)
        self.hidden_dims = tuple(config)

        self.embedding_pad = (-self.embedding_dim) % self.token_feature_dim
        self.embedding_token_count = (self.embedding_dim + self.embedding_pad) // self.token_feature_dim
        token_count = 1 + 1 + self.embedding_token_count  # cls + base + embedding tokens

        self.embedding_token_proj = nn.Linear(self.token_feature_dim, self.d_model)
        self.base_token_proj = nn.Sequential(
            nn.Linear(self.base_feature_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_embed = nn.Parameter(torch.zeros(1, token_count, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, output_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_features = x[:, : self.base_feature_dim]
        embedding_features = x[:, self.base_feature_dim :]
        if self.embedding_pad:
            embedding_features = F.pad(embedding_features, (0, self.embedding_pad))

        embedding_tokens = embedding_features.reshape(
            x.shape[0],
            self.embedding_token_count,
            self.token_feature_dim,
        )
        embedding_tokens = self.embedding_token_proj(embedding_tokens)
        base_token = self.base_token_proj(base_features).unsqueeze(1)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        tokens = torch.cat([cls_token, base_token, embedding_tokens], dim=1)
        tokens = tokens + self.position_embed[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0])


def _save_transformer_artifacts(
    out_dir: Path,
    model: nn.Module,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    train_files: Sequence[base.FileMeta],
    validation_files: Sequence[base.FileMeta],
    test_files: Sequence[base.FileMeta],
    train_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    best_epoch: int,
    best_score: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "multilabel_transformer.pt"
    stats_path = out_dir / "feature_stats.npz"
    split_path = out_dir / "file_split.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "embedding_transformer_classifier",
            "input_dim": int(input_dim),
            "output_dim": 40,
            "transformer_config": list(hidden_dims),
            "dropout": float(dropout),
            "base_features": base.BASE_FEATURE_COLUMNS,
            "target_columns": base.TARGET_COLUMNS,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
        },
        model_path,
    )
    np.savez_compressed(stats_path, x_mean=x_mean, x_std=x_std)

    rows = []
    for m in train_files:
        rows.append({"split": "train", "file": str(m.feather_path), "file_time_utc": str(m.file_time)})
    for m in validation_files:
        rows.append({"split": "validation", "file": str(m.feather_path), "file_time_utc": str(m.file_time)})
    for m in test_files:
        rows.append({"split": "test", "file": str(m.feather_path), "file_time_utc": str(m.file_time)})
    pd.DataFrame(rows).to_csv(split_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved feature stats: {stats_path}")
    print(f"Saved split manifest: {split_path}")


def main() -> int:
    base.MultiLabelMLP = EmbeddingTransformerClassifier
    base._resolve_hidden_dims = _resolve_transformer_config
    base._save_artifacts = _save_transformer_artifacts
    base.OUTPUT_DIR = OUTPUT_DIR
    base.LOG_DIR = LOG_DIR
    print(
        "Transformer classifier config format: "
        "d_model,layers,heads,feedforward_dim,token_feature_dim. "
        f"Default={DEFAULT_TRANSFORMER_CONFIG}"
    )
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
