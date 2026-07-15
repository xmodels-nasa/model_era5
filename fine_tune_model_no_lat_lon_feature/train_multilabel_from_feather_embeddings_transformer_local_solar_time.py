#!/usr/bin/env python3
"""Train the embedding Transformer with local-solar-time features.

This ablation uses latitude, longitude, local solar time, and year/season
features with the Aurora embedding. Local solar time is derived per sample from
UTC timestamp and longitude:

    local_solar_hour = (utc_hour + longitude / 15) mod 24
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


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


OUTPUT_DIR = PROJECT_ROOT / "results-v3" / "model_outputs_transformer_local_solar_time"
LOG_DIR = PROJECT_ROOT / "logs"
BASE_FEATURE_COLUMNS = [
    "Latitude_0",
    "Longitude_0",
    "local_solar_time_sin",
    "local_solar_time_cos",
    "time_year_sin",
    "time_year_cos",
]


def _local_solar_time_features(timestamp_values: np.ndarray, longitude_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if timestamp_values.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty

    first = int(timestamp_values[0])
    unit = base._timestamp_unit_from_value(first)
    ts = pd.to_datetime(timestamp_values.astype(np.int64), unit=unit, utc=True)
    seconds = (
        ts.hour.to_numpy(dtype=np.float64) * 3600.0
        + ts.minute.to_numpy(dtype=np.float64) * 60.0
        + ts.second.to_numpy(dtype=np.float64)
        + ts.microsecond.to_numpy(dtype=np.float64) / 1e6
    )
    utc_hour = seconds / 3600.0
    local_hour = np.mod(utc_hour + longitude_values.astype(np.float64) / 15.0, 24.0)
    phase = 2.0 * math.pi * local_hour / 24.0
    return np.sin(phase).astype(np.float32), np.cos(phase).astype(np.float32)


def _load_one_file_arrays_local_solar_time(
    meta: base.FileMeta,
    sample_ratio: float = 1.0,
    max_samples_per_file: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical_feature_columns = [
        "Latitude_0",
        "Longitude_0",
        "time_year_sin",
        "time_year_cos",
    ]
    required_cols = ["timestamp_0"] + physical_feature_columns + base.TARGET_COLUMNS
    df = pd.read_feather(meta.feather_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {meta.feather_path.name}: {missing}")

    with np.load(meta.npz_path) as data:
        if "emb_all_levels" not in data or "row_indices" not in data:
            raise ValueError(f"Missing arrays in {meta.npz_path.name}; expected emb_all_levels and row_indices.")
        emb = data["emb_all_levels"]
        rows = data["row_indices"]

    if emb.size == 0 or rows.size == 0:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, np.empty((0, 40), dtype=np.float32), np.empty((0,), dtype=np.int64)

    rows = rows.astype(np.int64)
    valid = (rows >= 0) & (rows < len(df))
    if not np.all(valid):
        rows = rows[valid]
        emb = emb[valid]
    if rows.size == 0:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, np.empty((0, 40), dtype=np.float32), np.empty((0,), dtype=np.int64)

    if not (0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    sample_indices = np.arange(rows.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if sample_ratio < 1.0:
        count = max(1, int(len(sample_indices) * sample_ratio))
        sample_indices = rng.choice(sample_indices, size=count, replace=False)
    if max_samples_per_file is not None and max_samples_per_file > 0 and len(sample_indices) > max_samples_per_file:
        sample_indices = rng.choice(sample_indices, size=max_samples_per_file, replace=False)
    sample_indices = np.sort(sample_indices)
    rows = rows[sample_indices]
    emb = emb[sample_indices]

    selected = df.iloc[rows]
    longitude = selected["Longitude_0"].to_numpy(dtype=np.float32, copy=True)
    local_sin, local_cos = _local_solar_time_features(
        selected["timestamp_0"].to_numpy(copy=True),
        longitude,
    )
    base_features = np.column_stack(
        [
            selected["Latitude_0"].to_numpy(dtype=np.float32, copy=True),
            longitude,
            local_sin,
            local_cos,
            selected["time_year_sin"].to_numpy(dtype=np.float32, copy=True),
            selected["time_year_cos"].to_numpy(dtype=np.float32, copy=True),
        ]
    ).astype(np.float32, copy=False)

    emb_flat = emb.reshape(emb.shape[0], -1).astype(np.float32)
    y = selected[base.TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    y = (y > 0.5).astype(np.float32)
    x = np.concatenate([base_features, emb_flat], axis=1)
    return x, y, rows


def main() -> int:
    base.BASE_FEATURE_COLUMNS = list(BASE_FEATURE_COLUMNS)
    base._load_one_file_arrays = _load_one_file_arrays_local_solar_time
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
