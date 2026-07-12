#!/usr/bin/env python3
"""Diagnose whether global inference matches the transformer training path.

For samples from the CloudSat Feather file associated with a global NetCDF
output, this script compares four predictions:

1. saved_training: saved embedding NPZ + original Feather base features.
2. live_track: newly extracted Aurora embedding + original base features.
3. live_grid: newly extracted grid-patch embedding + global grid base features.
4. saved_global: value sampled from the saved global NetCDF output.

Comparisons 1 vs 2 isolate encoder/token consistency. Comparisons 3 vs 4
isolate global input construction and NetCDF placement. Comparison 2 vs 3
quantifies the intentional difference between track and global-grid features.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"
for path in (FINE_TUNED_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_aurora_batches as aurora_batches  # noqa: E402
import train_multilabel_from_feather_embeddings as emb_train  # noqa: E402
from build_global_column_cloud_probability import (  # noqa: E402
    base_longitudes,
    cyclic_time_features,
    load_transformer,
    resolve_device,
    target_string,
)


DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v2" / "model_outputs_transformer"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "track_global_consistency"


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def _timestamp_series_to_target_hours(values: pd.Series) -> pd.Series:
    first = int(values.iloc[0])
    unit = "s" if len(str(abs(first))) <= 10 else "ms"
    timestamps = pd.to_datetime(values, unit=unit, utc=True)
    floor = timestamps.dt.floor("h")
    return floor + pd.to_timedelta((timestamps - floor > pd.Timedelta(minutes=30)).astype(int), unit="h")


def _resolve_feather_path(value: str, feather_root: Path | None) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if feather_root is not None:
        candidate = feather_root / path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find Feather file {path}. Pass --feather-file or --feather-root."
    )


def _nearest_grid_indices(
    query_lat: np.ndarray,
    query_lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lat_idx = np.abs(grid_lat[None, :] - query_lat[:, None]).argmin(axis=1)
    lon_norm = np.mod(query_lon, 360.0)
    lon_distance = np.abs(((grid_lon[None, :] - lon_norm[:, None] + 180.0) % 360.0) - 180.0)
    lon_idx = lon_distance.argmin(axis=1)
    return lat_idx.astype(np.int64), lon_idx.astype(np.int64)


def _predict(model: torch.nn.Module, x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        probs = torch.sigmoid(model((x - x_mean) / x_std))
    return probs.detach().cpu().numpy().astype(np.float32, copy=False)


def _prediction_metrics(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    delta = left.astype(np.float64) - right.astype(np.float64)
    left_column = left.max(axis=1)
    right_column = right.max(axis=1)
    return {
        "mask_mae": float(np.abs(delta).mean()),
        "mask_max_abs_error": float(np.abs(delta).max()),
        "column_mae": float(np.abs(left_column - right_column).mean()),
        "column_max_abs_error": float(np.abs(left_column - right_column).max()),
        "column_correlation": float(np.corrcoef(left_column, right_column)[0, 1])
        if len(left_column) > 1 and np.std(left_column) > 0 and np.std(right_column) > 0
        else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--feather-file", type=Path, default=None)
    parser.add_argument("--feather-root", type=Path, default=os.getenv("FEATHER_ROOT") or None)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=os.getenv("EMBEDDING_OUTUT_DIR", os.getenv("EMBEDDING_OUTPUT_DIR", "embeddings")),
    )
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data_era5")))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aurora-device", default=None)
    parser.add_argument("--aurora-backbone", choices=["full", "small"], default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")
    if not args.global_file.is_file():
        raise FileNotFoundError(f"Missing global file: {args.global_file}")

    with xr.open_dataset(args.global_file) as global_ds:
        required = {"cloud_mask_prob", "column_cloud_prob"}
        missing = required - set(global_ds.data_vars)
        if missing:
            raise ValueError(f"Global file is missing variables: {sorted(missing)}")
        target_attr = str(global_ds.attrs.get("target_time_utc", ""))
        selected_file = str(global_ds.attrs.get("selected_split_file", ""))
        if not target_attr:
            raise ValueError("Global file has no target_time_utc attribute.")
        if not selected_file and args.feather_file is None:
            raise ValueError("Global file has no selected_split_file; pass --feather-file.")
        target_dt = pd.Timestamp(target_attr).tz_convert("UTC").to_pydatetime().replace(tzinfo=None)
        global_lat = global_ds.latitude.values.astype(np.float32)
        global_lon = global_ds.longitude.values.astype(np.float32)
        global_masks = global_ds.cloud_mask_prob.values.astype(np.float32)
        base_lon_convention = str(global_ds.attrs.get("base_longitude_convention", "minus180_180"))

    feather_path = args.feather_file or _resolve_feather_path(selected_file, args.feather_root)
    embedding_path = args.embedding_dir / f"{_safe_stem(feather_path)}.npz"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {embedding_path}")

    meta = emb_train.FileMeta(
        feather_path=feather_path,
        npz_path=embedding_path,
        file_time=pd.Timestamp(target_dt, tz="UTC"),
    )
    x_saved, _, row_indices = emb_train._load_one_file_arrays(meta)
    if not len(row_indices):
        raise ValueError("No valid rows found in the Feather/embedding pair.")

    df = pd.read_feather(feather_path, columns=["timestamp_0", *emb_train.BASE_FEATURE_COLUMNS])
    row_targets = _timestamp_series_to_target_hours(df["timestamp_0"])
    target_utc = pd.Timestamp(target_dt, tz="UTC")
    selected = row_targets.iloc[row_indices].to_numpy() == target_utc
    if not selected.any():
        raise ValueError(f"No embedding rows in {feather_path.name} belong to target {target_string(target_dt)}.")
    available = np.flatnonzero(selected)
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(available, size=min(args.samples, len(available)), replace=False))

    x_saved = x_saved[chosen]
    source_rows = row_indices[chosen]
    base_track = x_saved[:, : len(emb_train.BASE_FEATURE_COLUMNS)]
    query_lat = base_track[:, 0]
    query_lon = base_track[:, 1]

    classifier_device = resolve_device(args.device)
    aurora_device = resolve_device(args.aurora_device or args.device)
    model, x_mean, x_std, checkpoint = load_transformer(args.model_dir, classifier_device)
    x_mean_t = torch.from_numpy(x_mean).to(classifier_device)
    x_std_t = torch.from_numpy(x_std).to(classifier_device)
    saved_training = _predict(
        model,
        torch.from_numpy(x_saved).to(classifier_device),
        x_mean_t,
        x_std_t,
    )

    aurora_batches.DEBUG = args.aurora_backbone == "small"
    aurora_model = aurora_batches.load_aurora_model(device=aurora_device)
    context = aurora_batches.get_encoder_context_for_target(
        data_root=str(args.data_root),
        target=target_string(target_dt),
        model=aurora_model,
    )
    enc_out = context["enc_out"]
    enc_batch = context["enc_batch"]
    local_model = context["model"]
    grid_lat = enc_batch.metadata.lat.detach().cpu().numpy().astype(np.float32)
    grid_lon = enc_batch.metadata.lon.detach().cpu().numpy().astype(np.float32)
    patch_size = int(local_model.encoder.patch_size)
    h_grid, w_grid = enc_batch.spatial_shape
    h_patches, w_patches = h_grid // patch_size, w_grid // patch_size
    latent_levels = int(local_model.encoder.latent_levels)
    leading_dim = int(enc_out.shape[0])
    embed_dim = int(enc_out.shape[-1])
    expected_embedding_dim = int(checkpoint["input_dim"]) - len(emb_train.BASE_FEATURE_COLUMNS)
    if leading_dim * latent_levels * embed_dim != expected_embedding_dim:
        raise ValueError("Live Aurora encoder output does not match checkpoint embedding dimension.")
    level_tokens = enc_out[:, : latent_levels * h_patches * w_patches, :].reshape(
        leading_dim, latent_levels, h_patches * w_patches, embed_dim
    )

    track_lat_idx, track_lon_idx = _nearest_grid_indices(query_lat, query_lon, grid_lat, grid_lon)
    patch_idx = (track_lat_idx // patch_size) * w_patches + (track_lon_idx // patch_size)
    patch_idx_t = torch.from_numpy(patch_idx).to(level_tokens.device)
    live_embeddings = level_tokens[:, :, patch_idx_t, :].permute(2, 0, 1, 3).reshape(len(chosen), -1)
    live_track_x = torch.cat(
        [torch.from_numpy(base_track).to(classifier_device), live_embeddings.to(classifier_device, dtype=torch.float32)], dim=1
    )
    live_track = _predict(model, live_track_x, x_mean_t, x_std_t)

    global_lat_idx, global_lon_idx = _nearest_grid_indices(query_lat, query_lon, global_lat, global_lon)
    global_grid_lat = global_lat[global_lat_idx]
    global_grid_lon = global_lon[global_lon_idx]
    global_patch_idx = (global_lat_idx // patch_size) * w_patches + (global_lon_idx // patch_size)
    global_patch_idx_t = torch.from_numpy(global_patch_idx).to(level_tokens.device)
    global_embeddings = level_tokens[:, :, global_patch_idx_t, :].permute(2, 0, 1, 3).reshape(len(chosen), -1)
    day_sin, day_cos, year_sin, year_cos = cyclic_time_features(target_dt)
    global_base = np.column_stack(
        [
            global_grid_lat,
            base_longitudes(global_grid_lon, base_lon_convention),
            np.full(len(chosen), day_sin, dtype=np.float32),
            np.full(len(chosen), day_cos, dtype=np.float32),
            np.full(len(chosen), year_sin, dtype=np.float32),
            np.full(len(chosen), year_cos, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    live_grid_x = torch.cat(
        [torch.from_numpy(global_base).to(classifier_device), global_embeddings.to(classifier_device, dtype=torch.float32)], dim=1
    )
    live_grid = _predict(model, live_grid_x, x_mean_t, x_std_t)
    saved_global = global_masks[:, global_lat_idx, global_lon_idx].T.copy()

    output_prefix = args.output_dir / f"track_global_consistency_{target_string(target_dt)}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns: Dict[str, np.ndarray] = {
        "source_row": source_rows,
        "track_latitude": query_lat,
        "track_longitude": query_lon,
        "grid_latitude": global_grid_lat,
        "grid_longitude_era5": global_grid_lon,
        "saved_training_column_prob": saved_training.max(axis=1),
        "live_track_column_prob": live_track.max(axis=1),
        "live_grid_column_prob": live_grid.max(axis=1),
        "saved_global_column_prob": saved_global.max(axis=1),
    }
    for level in range(saved_training.shape[1]):
        columns[f"saved_training_level_{level:02d}"] = saved_training[:, level]
        columns[f"live_track_level_{level:02d}"] = live_track[:, level]
        columns[f"live_grid_level_{level:02d}"] = live_grid[:, level]
        columns[f"saved_global_level_{level:02d}"] = saved_global[:, level]
    csv_path = output_prefix.with_suffix(".csv")
    pd.DataFrame(columns).to_csv(csv_path, index=False)

    summary = {
        "target_hour_utc": target_dt.replace(tzinfo=timezone.utc).isoformat(),
        "global_file": str(args.global_file),
        "feather_file": str(feather_path),
        "embedding_file": str(embedding_path),
        "model_dir": str(args.model_dir),
        "available_rows_for_target": int(len(available)),
        "sampled_rows": int(len(chosen)),
        "global_base_longitude_convention": base_lon_convention,
        "saved_training_vs_live_track": _prediction_metrics(saved_training, live_track),
        "live_track_vs_live_grid": _prediction_metrics(live_track, live_grid),
        "live_grid_vs_saved_global": _prediction_metrics(live_grid, saved_global),
    }
    summary_path = output_prefix.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"Saved row-level comparison: {csv_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
