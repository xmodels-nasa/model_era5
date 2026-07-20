#!/usr/bin/env python3
"""Predict 40-level cloud probabilities along July CloudSat tracks.

This script uses saved track embedding NPZ files, not global-grid Aurora
embeddings. It reads the model split manifest, selects July files, predicts
all valid embedding rows along those tracks, and writes labels/probabilities to
a NetCDF file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"
INFERENCE_DIR = PROJECT_ROOT / "inference"
for path in (FINE_TUNED_DIR, INFERENCE_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import train_multilabel_from_feather_embeddings as emb_train  # noqa: E402
from build_global_column_cloud_probability import load_transformer, resolve_device  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v2" / "model_outputs_transformer"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "data"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "transformer-v2_july_track_cloud_probabilities.nc"
TARGET_COLUMNS = [f"y_40dim_{i}" for i in range(40)]


def month_start_end(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(datetime(year, month, 1, 0), tz="UTC")
    if month == 12:
        end = pd.Timestamp(datetime(year, 12, 31, 23, 59, 59, 999999), tz="UTC")
    else:
        end = pd.Timestamp(datetime(year, month + 1, 1, 0), tz="UTC") - pd.Timedelta(nanoseconds=1)
    return start, end


def parse_timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    normalized = value.strip().replace("T", "_").replace("-", "_").replace(":", "_")
    parts = [part for part in normalized.split("_") if part]
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        ts = pd.Timestamp(datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])), tz="UTC")
        return ts
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def timestamp_unit_from_values(values: np.ndarray) -> str:
    if values.size == 0:
        return "s"
    return "s" if len(str(abs(int(values[0])))) <= 10 else "ms"


def timestamps_from_series(values: np.ndarray) -> pd.DatetimeIndex:
    unit = timestamp_unit_from_values(values)
    return pd.to_datetime(values.astype(np.int64), unit=unit, utc=True)


def local_solar_time_features(timestamps: pd.DatetimeIndex, longitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seconds = (
        timestamps.hour.to_numpy(dtype=np.float64) * 3600.0
        + timestamps.minute.to_numpy(dtype=np.float64) * 60.0
        + timestamps.second.to_numpy(dtype=np.float64)
        + timestamps.microsecond.to_numpy(dtype=np.float64) / 1e6
    )
    utc_hour = seconds / 3600.0
    local_hour = np.mod(utc_hour + longitude.astype(np.float64) / 15.0, 24.0)
    phase = 2.0 * math.pi * local_hour / 24.0
    return np.sin(phase).astype(np.float32), np.cos(phase).astype(np.float32)


def resolve_feather_path(value: str, feather_root: Optional[Path]) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if feather_root is not None:
        candidate = feather_root / path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Feather file: {value}")


def selected_split_rows(
    split_path: Path,
    split: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Dict[str, str]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    rows: List[Dict[str, str]] = []
    split_filter = split.strip().lower()
    include_all_splits = split_filter in {"", "all", "*"}
    with split_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not include_all_splits and row.get("split") != split:
                continue
            file_time = pd.Timestamp(row["file_time_utc"])
            if file_time.tzinfo is None:
                file_time = file_time.tz_localize("UTC")
            else:
                file_time = file_time.tz_convert("UTC")
            if start <= file_time <= end:
                rows.append(row)
    if not rows:
        split_note = "any split" if include_all_splits else f"split={split!r}"
        raise ValueError(f"No {split_note} files found in {split_path} for {start} through {end}.")
    return rows


def load_track_file_inputs(
    feather_path: Path,
    embedding_path: Path,
    base_features: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {embedding_path}")

    required_physical = {"timestamp_0", "Latitude_0", "Longitude_0", *TARGET_COLUMNS}
    for name in base_features:
        if name in {"local_solar_time_sin", "local_solar_time_cos"}:
            required_physical.update({"timestamp_0", "Longitude_0"})
        elif name in {"Latitude_0", "Longitude_0", "time_day_sin", "time_day_cos", "time_year_sin", "time_year_cos"}:
            required_physical.add(name)
        else:
            raise ValueError(f"Unsupported checkpoint base feature for track inference: {name}")

    df = pd.read_feather(feather_path)
    missing = [name for name in sorted(required_physical) if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {feather_path.name}: {missing}")

    with np.load(embedding_path) as data:
        if "emb_all_levels" not in data or "row_indices" not in data:
            raise ValueError(f"Missing arrays in {embedding_path.name}; expected emb_all_levels and row_indices.")
        emb = data["emb_all_levels"]
        rows = data["row_indices"].astype(np.int64)

    valid = (rows >= 0) & (rows < len(df))
    if not np.all(valid):
        rows = rows[valid]
        emb = emb[valid]
    if rows.size == 0:
        empty_x = np.empty((0, 0), dtype=np.float32)
        empty_y = np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32)
        empty_rows = np.empty((0,), dtype=np.int64)
        empty_time = pd.DatetimeIndex([], tz="UTC")
        return empty_x, empty_y, empty_rows, empty_rows.astype(np.float32), empty_rows.astype(np.float32), empty_time

    selected = df.iloc[rows]
    timestamps = timestamps_from_series(selected["timestamp_0"].to_numpy(copy=True))
    in_window = (timestamps >= start) & (timestamps <= end)
    if not np.all(in_window):
        rows = rows[in_window]
        emb = emb[in_window]
        selected = df.iloc[rows]
        timestamps = timestamps[in_window]
    if rows.size == 0:
        empty_x = np.empty((0, 0), dtype=np.float32)
        empty_y = np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32)
        empty_rows = np.empty((0,), dtype=np.int64)
        empty_time = pd.DatetimeIndex([], tz="UTC")
        return empty_x, empty_y, empty_rows, empty_rows.astype(np.float32), empty_rows.astype(np.float32), empty_time

    longitude = selected["Longitude_0"].to_numpy(dtype=np.float32, copy=True) if "Longitude_0" in selected else None
    local_sin: Optional[np.ndarray] = None
    local_cos: Optional[np.ndarray] = None
    if "local_solar_time_sin" in base_features or "local_solar_time_cos" in base_features:
        if longitude is None:
            raise ValueError("local_solar_time features require Longitude_0.")
        local_sin, local_cos = local_solar_time_features(timestamps, longitude)

    feature_values: Dict[str, np.ndarray] = {}
    for name in base_features:
        if name == "local_solar_time_sin":
            feature_values[name] = local_sin  # type: ignore[assignment]
        elif name == "local_solar_time_cos":
            feature_values[name] = local_cos  # type: ignore[assignment]
        else:
            feature_values[name] = selected[name].to_numpy(dtype=np.float32, copy=True)

    base_x = np.column_stack([feature_values[name] for name in base_features]).astype(np.float32, copy=False)
    emb_flat = emb.reshape(emb.shape[0], -1).astype(np.float32)
    x = np.concatenate([base_x, emb_flat], axis=1)
    y = selected[TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    y = (y > 0.5).astype(np.float32)
    lat = selected["Latitude_0"].to_numpy(dtype=np.float32, copy=True)
    lon = selected["Longitude_0"].to_numpy(dtype=np.float32, copy=True)
    return x, y, rows, lat, lon, timestamps


def predict_in_batches(
    model: torch.nn.Module,
    x: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)
    probs = np.empty((x.shape[0], len(TARGET_COLUMNS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:stop]).to(device)
            pb = torch.sigmoid(model((xb - x_mean_t) / x_std_t))
            probs[start:stop] = pb.detach().cpu().numpy().astype(np.float32)
    return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--split", default="all", help="Split to include. Use 'all' to include every July row.")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path(os.getenv("EMBEDDING_OUTUT_DIR", os.getenv("EMBEDDING_OUTPUT_DIR", PROJECT_ROOT / "embeddings"))),
    )
    parser.add_argument("--feather-root", type=Path, default=os.getenv("FEATHER_ROOT") or None)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp, e.g. 2019_07_01_00.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp, e.g. 2019_07_10_23.")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all selected July files.")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_files < 0:
        raise ValueError("--max-files must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    start_default, end_default = month_start_end(args.year, args.month)
    start = parse_timestamp(args.start) or start_default
    end = parse_timestamp(args.end) or end_default
    if end < start:
        raise ValueError(f"End {end} is before start {start}.")

    device = resolve_device(args.device)
    model, x_mean, x_std, checkpoint = load_transformer(args.model_dir, device)
    base_features = list(checkpoint.get("base_features", emb_train.BASE_FEATURE_COLUMNS))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("test_metrics", {}).get("iou_threshold", 0.5))
    split_path = args.split_path or args.model_dir / "file_split.csv"
    split_rows = selected_split_rows(split_path, args.split, start, end)
    if args.max_files:
        split_rows = split_rows[: args.max_files]

    all_probs: List[np.ndarray] = []
    all_truth: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_source_rows: List[np.ndarray] = []
    all_lat: List[np.ndarray] = []
    all_lon: List[np.ndarray] = []
    all_times: List[np.ndarray] = []
    all_file_index: List[np.ndarray] = []
    source_files: List[str] = []
    rows_per_file: List[int] = []

    for file_index, row in enumerate(split_rows):
        feather_path = resolve_feather_path(row["file"], args.feather_root)
        embedding_path = args.embedding_dir / f"{emb_train._safe_stem(feather_path)}.npz"
        x, y, source_rows, lat, lon, timestamps = load_track_file_inputs(
            feather_path=feather_path,
            embedding_path=embedding_path,
            base_features=base_features,
            start=start,
            end=end,
        )
        if x.size == 0:
            print(f"Skipped empty file after filtering: {feather_path.name}")
            continue
        if x.shape[1] != int(checkpoint["input_dim"]):
            raise ValueError(f"Built input_dim={x.shape[1]} for {feather_path.name}, expected {checkpoint['input_dim']}")

        probs = predict_in_batches(model, x, x_mean, x_std, args.batch_size, device)
        pred = (probs >= threshold).astype(np.int8)

        all_probs.append(probs)
        all_truth.append(y.astype(np.int8))
        all_pred.append(pred)
        all_source_rows.append(source_rows.astype(np.int64))
        all_lat.append(lat.astype(np.float32))
        all_lon.append(lon.astype(np.float32))
        all_times.append(timestamps.to_numpy(dtype="datetime64[ns]"))
        all_file_index.append(np.full(len(source_rows), len(source_files), dtype=np.int32))
        source_files.append(str(feather_path))
        rows_per_file.append(int(len(source_rows)))
        print(f"Processed {len(source_files)}/{len(split_rows)}: {feather_path.name} ({len(source_rows):,} samples)")

    if not all_probs:
        raise ValueError("No track samples were loaded for prediction.")

    cloud_prob = np.concatenate(all_probs, axis=0).astype(np.float32)
    cloud_label = np.concatenate(all_truth, axis=0).astype(np.int8)
    predicted_label = np.concatenate(all_pred, axis=0).astype(np.int8)
    source_row = np.concatenate(all_source_rows, axis=0).astype(np.int64)
    lat = np.concatenate(all_lat, axis=0).astype(np.float32)
    lon = np.concatenate(all_lon, axis=0).astype(np.float32)
    timestamps = np.concatenate(all_times, axis=0).astype("datetime64[ns]")
    file_index = np.concatenate(all_file_index, axis=0).astype(np.int32)

    sample_order = np.lexsort((source_row, file_index))
    cloud_prob = cloud_prob[sample_order]
    cloud_label = cloud_label[sample_order]
    predicted_label = predicted_label[sample_order]
    source_row = source_row[sample_order]
    lat = lat[sample_order]
    lon = lon[sample_order]
    timestamps = timestamps[sample_order]
    file_index = file_index[sample_order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "cloud_mask_label": (
                ("sample", "level"),
                cloud_label,
                {"description": "Ground-truth CloudSat/CALIPSO cloud mask label, thresholded to 0/1."},
            ),
            "cloud_mask_prob": (
                ("sample", "level"),
                cloud_prob,
                {"description": "Predicted cloud probability for each of the 40 mask levels."},
            ),
            "predicted_label": (
                ("sample", "level"),
                predicted_label,
                {"description": "Predicted 0/1 cloud label using the saved/test threshold."},
            ),
            "column_label": ("sample", cloud_label.max(axis=1).astype(np.int8)),
            "column_cloud_prob": ("sample", cloud_prob.max(axis=1).astype(np.float32)),
            "predicted_column_label": ("sample", predicted_label.max(axis=1).astype(np.int8)),
            "latitude": ("sample", lat),
            "longitude": ("sample", lon),
            "timestamp_utc": ("sample", timestamps),
            "source_row": ("sample", source_row),
            "source_file_index": ("sample", file_index),
            "source_file_name": ("source_file", np.asarray(source_files, dtype=str)),
            "source_file_sample_count": ("source_file", np.asarray(rows_per_file, dtype=np.int32)),
        },
        coords={
            "sample": np.arange(cloud_prob.shape[0], dtype=np.int64),
            "level": np.arange(cloud_prob.shape[1], dtype=np.int16),
            "source_file": np.arange(len(source_files), dtype=np.int32),
        },
        attrs={
            "model_dir": str(args.model_dir),
            "split_path": str(split_path),
            "split_selection": args.split,
            "embedding_dir": str(args.embedding_dir),
            "feather_root": str(args.feather_root or ""),
            "start_time_utc": str(start),
            "end_time_utc": str(end),
            "threshold": threshold,
            "base_features": json.dumps(base_features),
            "total_samples": int(cloud_prob.shape[0]),
            "source_file_count": int(len(source_files)),
            "created_by": Path(__file__).name,
        },
    )
    encoding = {
        "cloud_mask_label": {"zlib": True, "complevel": 4, "dtype": "int8"},
        "cloud_mask_prob": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "predicted_label": {"zlib": True, "complevel": 4, "dtype": "int8"},
    }
    ds.to_netcdf(args.output, encoding=encoding)
    print(f"Saved: {args.output}")
    print(f"Samples: {cloud_prob.shape[0]:,}; source files: {len(source_files)}; threshold: {threshold:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
