#!/usr/bin/env python3
"""Train from forecast-aligned Feather embedding NPZ files.

Forecast embeddings are generated with `get_embedings_from_all_feather_files_forecast.py`.
The NPZ layout is intentionally the same as the original embedding files:
`emb_all_levels` plus `row_indices`, where each row index points directly to
the target row in the Feather file. Training can therefore reuse the original
loader without any cross-file timestamp or location matching.
"""

import os
from pathlib import Path

import train_multilabel_from_feather_embeddings as base


FORECAST_EMBEDDING_DIR = os.getenv(
    "FORECAST_EMBEDDING_OUTPUT_DIR",
    os.getenv("EMBEDDING_FORECAST_OUTPUT_DIR", str(base.PROJECT_ROOT / "embeddings_forecast")),
)


def main() -> int:
    base.EMBEDDING_DIR = FORECAST_EMBEDDING_DIR
    base.OUTPUT_DIR = str(base.PROJECT_ROOT / "model_outputs_forecast")
    base.LOG_DIR = str(base.PROJECT_ROOT / "logs_forecast")
    Path(base.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
