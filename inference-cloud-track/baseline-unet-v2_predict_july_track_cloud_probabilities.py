#!/usr/bin/env python3
"""Predict 40-level cloud probabilities along July CloudSat test tracks with the baseline U-Net v2 model.

This script reads the raw-chip NPZ files listed by the baseline split manifest,
predicts all valid rows in the requested July window, and writes labels,
probabilities, and predicted labels to a NetCDF file.
"""

from __future__ import annotations

import argparse
import csv
import json
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
BASELINE_DIR = PROJECT_ROOT / "baseline_model"
INFERENCE_DIR = PROJECT_ROOT / "inference"
for path in (BASELINE_DIR, INFERENCE_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import train_multilabel_from_raw_chips as raw_train  # noqa: E402
from build_global_column_cloud_probability import resolve_device  # noqa: E402
from build_global_column_cloud_probability_baseline_raw_chips import load_baseline_model  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v2" / "baseline_model_outputs"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "data"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "baseline-unet-v2_july_track_cloud_probabilities.nc"
DEFAULT_RAW_CHIPS_DIR = Path(os.getenv("RAW_CHIPS_DIR", str(PROJECT_ROOT / "raw_chips")))
DEFAULT_FEATHER_ROOT = Path(os.environ["FEATHER_ROOT"]) if os.getenv("FEATHER_ROOT") else None
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
        return pd.Timestamp(datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])), tz="UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def timestamp_unit_from_values(values: np.ndarray) -> str:
    if values.size == 0:
        return "s"
    return "s" if len(str(abs(int(values[0])))) <= 10 else "ms"


def timestamps_from_values(values: np.ndarray) -> pd.DatetimeIndex:
    unit = timestamp_unit_from_values(values)
    return pd.to_datetime(values.astype(np.float64), unit=unit, utc=True)


def resolve_file_path(value: str, fallback_root: Optional[Path]) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if fallback_root is not None:
        candidate = fallback_root / path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find file: {value}")


def selected_split_rows(
    split_path: Path,
    split: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Dict[str, str]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")

    rows: List[Dict[str, str]] = []
    with split_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            file_time = pd.Timestamp(row["file_time_utc"])
            if file_time.tzinfo is None:
                file_time = file_time.tz_localize("UTC")
            else:
                file_time = file_time.tz_convert("UTC")
            if start <= file_time <= end:
                rows.append(row)
    if not rows:
        raise ValueError(f"No split={split!r} files found in {split_path} for {start} through {end}.")
    return rows


def validate_npz_against_checkpoint(
    *,
    npz_path: Path,
    dynamic_channel_names: np.ndarray,
    static_channel_names: np.ndarray,
    chip_size: int,
    checkpoint: Dict[str, object],
) -> None:
    expected_channel_names = np.asarray(checkpoint["channel_names"])
    actual_channel_names = raw_train._time_expanded_channel_names(
        dynamic_channel_names=dynamic_channel_names,
        static_channel_names=static_channel_names,
    )
    if not np.array_equal(expected_channel_names, actual_channel_names):
        raise ValueError(f"Raw-chip channel order does not match checkpoint for {npz_path.name}.")
    if chip_size != int(checkpoint["chip_size"]):
        raise ValueError(f"Raw-chip size {chip_size} does not match checkpoint chip_size={checkpoint['chip_size']}.")


def load_raw_chip_track_file(
    npz_path: Path,
    base_features: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    checkpoint: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing raw-chip NPZ: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "dynamic_chips",
            "static_chips",
            "labels",
            "row_indices",
            "latitudes",
            "longitudes",
            "timestamps",
            "dynamic_channel_names",
            "static_channel_names",
            "chip_size",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Missing arrays in {npz_path.name}: {missing}")

        dynamic = data["dynamic_chips"].astype(np.float32, copy=False)
        static = data["static_chips"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.float32, copy=False)
        source_rows = data["row_indices"].astype(np.int64, copy=False)
        latitudes = data["latitudes"].astype(np.float32, copy=False)
        longitudes = data["longitudes"].astype(np.float32, copy=False)
        timestamps_raw = data["timestamps"].astype(np.float64, copy=False)
        dynamic_channel_names = data["dynamic_channel_names"]
        static_channel_names = data["static_channel_names"]
        chip_size = int(data["chip_size"])

    if labels.shape[0] == 0:
        empty_chips = np.empty((0, int(checkpoint["input_channels"]), chip_size, chip_size), dtype=np.float32)
        empty_base = np.empty((0, len(base_features)), dtype=np.float32)
        empty_rows = np.empty((0,), dtype=np.int64)
        empty_time = pd.DatetimeIndex([], tz="UTC")
        return empty_chips, empty_base, np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32), empty_rows, empty_rows.astype(np.float32), empty_rows.astype(np.float32), empty_time

    validate_npz_against_checkpoint(
        npz_path=npz_path,
        dynamic_channel_names=dynamic_channel_names,
        static_channel_names=static_channel_names,
        chip_size=chip_size,
        checkpoint=checkpoint,
    )

    timestamps = timestamps_from_values(timestamps_raw)
    in_window = (timestamps >= start) & (timestamps <= end)
    if not np.all(in_window):
        dynamic = dynamic[in_window]
        static = static[in_window]
        labels = labels[in_window]
        source_rows = source_rows[in_window]
        latitudes = latitudes[in_window]
        longitudes = longitudes[in_window]
        timestamps_raw = timestamps_raw[in_window]
        timestamps = timestamps[in_window]

    if labels.shape[0] == 0:
        empty_chips = np.empty((0, int(checkpoint["input_channels"]), chip_size, chip_size), dtype=np.float32)
        empty_base = np.empty((0, len(base_features)), dtype=np.float32)
        empty_rows = np.empty((0,), dtype=np.int64)
        empty_time = pd.DatetimeIndex([], tz="UTC")
        return empty_chips, empty_base, np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32), empty_rows, empty_rows.astype(np.float32), empty_rows.astype(np.float32), empty_time

    unsupported = [name for name in base_features if name not in raw_train.BASE_FEATURE_COLUMNS]
    if unsupported:
        raise ValueError(f"Unsupported checkpoint base features for baseline track inference: {unsupported}")

    chips = raw_train._flatten_chip_channels(dynamic, static)
    if chips.shape[1] != int(checkpoint["input_channels"]):
        raise ValueError(f"Built input_channels={chips.shape[1]} for {npz_path.name}, expected {checkpoint['input_channels']}.")

    full_base = raw_train._timestamps_to_base_features(
        latitudes=latitudes,
        longitudes=longitudes,
        timestamps=timestamps_raw,
    )
    if base_features:
        feature_positions = [raw_train.BASE_FEATURE_COLUMNS.index(name) for name in base_features]
        base = full_base[:, feature_positions].astype(np.float32, copy=False)
    else:
        base = np.empty((labels.shape[0], 0), dtype=np.float32)

    labels = (labels > 0.5).astype(np.float32, copy=False)
    return chips, base, labels, source_rows, latitudes, longitudes, timestamps


def predict_in_batches(
    model: torch.nn.Module,
    chips: np.ndarray,
    base_features: np.ndarray,
    stats: raw_train.NormalizationStats,
    use_base_features: bool,
    batch_size: int,
    device: str,
) -> np.ndarray:
    probs = np.empty((chips.shape[0], len(TARGET_COLUMNS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, chips.shape[0], batch_size):
            stop = min(start + batch_size, chips.shape[0])
            chips_t, base_t = raw_train._normalize_batch(
                chips=chips[start:stop],
                base_features=base_features[start:stop],
                stats=stats,
                use_base_features=use_base_features,
            )
            logits = model(chips_t.to(device), base_t.to(device))
            probs[start:stop] = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-chips-dir", type=Path, default=DEFAULT_RAW_CHIPS_DIR)
    parser.add_argument("--feather-root", type=Path, default=DEFAULT_FEATHER_ROOT)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp, e.g. 2019_07_01_00.")
    parser.add_argument("--end", default=None, help="Optional UTC end timestamp, e.g. 2019_07_10_23.")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all selected split files.")
    parser.add_argument("--batch-size", type=int, default=1024)
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
    model, stats, checkpoint = load_baseline_model(args.model_dir, device)
    base_features = list(checkpoint.get("base_features", raw_train.BASE_FEATURE_COLUMNS))
    use_base_features = bool(base_features)
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
    source_npz_files: List[str] = []
    rows_per_file: List[int] = []

    print(f"Model dir: {args.model_dir}")
    print(f"Split rows: {len(split_rows)} from {split_path}")
    print(f"Device: {device}; threshold={threshold:g}; base_features={base_features}")

    for row in split_rows:
        npz_path = resolve_file_path(row["npz"], args.raw_chips_dir)
        try:
            feather_path = resolve_file_path(row["file"], args.feather_root)
            source_file = str(feather_path)
        except FileNotFoundError:
            source_file = row["file"]

        chips, base, y, source_rows, lat, lon, timestamps = load_raw_chip_track_file(
            npz_path=npz_path,
            base_features=base_features,
            start=start,
            end=end,
            checkpoint=checkpoint,
        )
        if chips.shape[0] == 0:
            print(f"Skipped empty file after filtering: {npz_path.name}")
            continue

        probs = predict_in_batches(
            model=model,
            chips=chips,
            base_features=base,
            stats=stats,
            use_base_features=use_base_features,
            batch_size=args.batch_size,
            device=device,
        )
        pred = (probs >= threshold).astype(np.int8)

        all_probs.append(probs)
        all_truth.append(y.astype(np.int8))
        all_pred.append(pred)
        all_source_rows.append(source_rows.astype(np.int64))
        all_lat.append(lat.astype(np.float32))
        all_lon.append(lon.astype(np.float32))
        all_times.append(timestamps.to_numpy(dtype="datetime64[ns]"))
        all_file_index.append(np.full(len(source_rows), len(source_files), dtype=np.int32))
        source_files.append(source_file)
        source_npz_files.append(str(npz_path))
        rows_per_file.append(int(len(source_rows)))
        print(f"Processed {len(source_files)}/{len(split_rows)}: {npz_path.name} ({len(source_rows):,} samples)")

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
                {"description": "Baseline U-Net predicted cloud probability for each of the 40 mask levels."},
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
            "source_npz_name": ("source_file", np.asarray(source_npz_files, dtype=str)),
            "source_file_sample_count": ("source_file", np.asarray(rows_per_file, dtype=np.int32)),
        },
        coords={
            "sample": np.arange(cloud_prob.shape[0], dtype=np.int64),
            "level": np.arange(cloud_prob.shape[1], dtype=np.int16),
            "source_file": np.arange(len(source_files), dtype=np.int32),
        },
        attrs={
            "model_dir": str(args.model_dir),
            "model_architecture": "baseline_raw_chip_unet_v2",
            "split_path": str(split_path),
            "split": args.split,
            "raw_chips_dir": str(args.raw_chips_dir or ""),
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
