#!/usr/bin/env python3
"""Train the embedding Transformer without cyclic time features.

This isolated ablation keeps latitude and longitude base features, but removes
UTC day and year cyclic features. It uses the same Feather targets and Aurora
embedding NPZ files as the standard embedding Transformer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"
if str(FINE_TUNED_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNED_DIR))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import train_multilabel_from_feather_embeddings as base  # noqa: E402
from train_multilabel_from_feather_embeddings_transformer import (  # noqa: E402
    EmbeddingTransformerClassifier,
    _resolve_transformer_config,
    _save_transformer_artifacts,
)


OUTPUT_DIR = PROJECT_ROOT / "results-v3" / "model_outputs_transformer_no_time"
LOG_DIR = PROJECT_ROOT / "logs"
BASE_FEATURE_COLUMNS = [
    "Latitude_0",
    "Longitude_0",
]


def main() -> int:
    # The shared loader reads this list when constructing each training row.
    base.BASE_FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)
    base.MultiLabelMLP = EmbeddingTransformerClassifier
    base._resolve_hidden_dims = _resolve_transformer_config
    base._save_artifacts = _save_transformer_artifacts
    base.OUTPUT_DIR = str(OUTPUT_DIR)
    base.LOG_DIR = str(LOG_DIR)
    print(f"Base features: {base.BASE_FEATURE_COLUMNS}")
    print(f"Output directory: {OUTPUT_DIR}")
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
