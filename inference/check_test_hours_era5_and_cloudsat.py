#!/usr/bin/env python3
"""Check ERA5 inputs and CloudSat labels for random held-out test hours.

For distinct hours sampled from a saved test split, this script:

* checks every raw-chip ERA5 channel for NaN/Inf values and band statistics;
* computes CloudSat column-cloud frequency in the matching Feather rows;
* writes CSV outputs for latitude-band comparison and a compact JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
BASELINE_DIR = PROJECT_ROOT / "baseline_model"
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"
for path in (BASELINE_DIR, FINE_TUNED_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_raw_chips_from_feather_file import load_era5_pair_as_tensors  # noqa: E402
from build_global_column_cloud_probability import (  # noqa: E402
    has_era5_pair,
    target_string,
    timestamp_to_rounded_hour,
)
import train_multilabel_from_raw_chips as raw_train  # noqa: E402


DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results-v2" / "baseline_model_outputs" / "file_split.csv"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "era5_cloudsat_test_checks"
LATITUDE_BANDS = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]


def _timestamp_series_to_target_hours(values: pd.Series) -> pd.Series:
    first = int(values.iloc[0])
    unit = "s" if len(str(abs(first))) <= 10 else "ms"
    timestamps = pd.to_datetime(values, unit=unit, utc=True)
    floor = timestamps.dt.floor("h")
    return floor + pd.to_timedelta((timestamps - floor > pd.Timedelta(minutes=30)).astype(int), unit="h")


def _resolve_feather_path(value: str, feather_root: Path | None) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if feather_root is not None:
        candidate = feather_root / candidate.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find test Feather file: {value}")


def _finite_stats(values: np.ndarray) -> Dict[str, float | int | None]:
    finite = np.isfinite(values)
    valid = values[finite]
    if not valid.size:
        return {
            "count": int(values.size),
            "invalid_count": int(values.size),
            "finite_fraction": 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "invalid_count": int((~finite).sum()),
        "finite_fraction": float(finite.mean()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def _era5_band_rows(target_dt: datetime, context: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    lat = context["lat"]
    dynamic = context["dynamic"]
    static = context["static"]
    dynamic_names = context["dynamic_channel_names"].tolist()
    static_names = context["static_channel_names"].tolist()
    channels: List[Tuple[str, np.ndarray]] = []
    for time_name, time_index in (("t_minus_6", 0), ("t", 1)):
        channels.extend((f"{time_name}_{name}", dynamic[time_index, idx]) for idx, name in enumerate(dynamic_names))
    channels.extend((name, static[idx]) for idx, name in enumerate(static_names))

    rows: List[Dict[str, object]] = []
    for low, high in LATITUDE_BANDS:
        latitude_mask = (lat >= low) & (lat < high)
        for channel_name, field in channels:
            stats = _finite_stats(field[latitude_mask])
            rows.append(
                {
                    "target_hour_utc": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                    "lat_min": low,
                    "lat_max": high,
                    "channel": channel_name,
                    **stats,
                }
            )
    return rows


def _cloudsat_band_rows(target_dt: datetime, feather_path: Path) -> List[Dict[str, object]]:
    columns = ["timestamp_0", "Latitude_0", *raw_train.TARGET_COLUMNS]
    df = pd.read_feather(feather_path, columns=columns)
    assigned_hours = _timestamp_series_to_target_hours(df["timestamp_0"])
    target_utc = pd.Timestamp(target_dt, tz="UTC")
    df = df.loc[assigned_hours == target_utc]
    column_cloud = (df[raw_train.TARGET_COLUMNS].to_numpy(dtype=np.float32) > 0.5).any(axis=1)
    lat = df["Latitude_0"].to_numpy(dtype=np.float32)

    rows: List[Dict[str, object]] = []
    for low, high in LATITUDE_BANDS:
        mask = (lat >= low) & (lat < high)
        count = int(mask.sum())
        rows.append(
            {
                "target_hour_utc": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                "source_feather_file": str(feather_path),
                "lat_min": low,
                "lat_max": high,
                "sample_count": count,
                "column_cloud_fraction": float(column_cloud[mask].mean()) if count else None,
            }
        )
    return rows


def _select_test_hours(split_path: Path, data_root: Path, count: int, seed: int) -> List[Tuple[datetime, Dict[str, str]]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    candidates: Dict[datetime, Dict[str, str]] = {}
    with split_path.open("r", encoding="utf-8", newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            if row.get("split") != "test":
                continue
            target_dt = timestamp_to_rounded_hour(row["file_time_utc"])
            if has_era5_pair(data_root, target_dt):
                candidates.setdefault(target_dt, row)
    if not candidates:
        raise FileNotFoundError(f"No test-split hours with an ERA5 input pair under {data_root}")
    choices = sorted(candidates.items(), key=lambda item: item[0])
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(choices), size=min(count, len(choices)), replace=False)
    return [choices[int(index)] for index in sorted(picked)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data_era5")))
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--feather-root", type=Path, default=os.getenv("FEATHER_ROOT") or None)
    parser.add_argument("--sample-hours", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_hours <= 0:
        raise ValueError("--sample-hours must be positive.")
    selected = _select_test_hours(args.split_path, args.data_root, args.sample_hours, args.seed)
    era5_rows: List[Dict[str, object]] = []
    cloudsat_rows: List[Dict[str, object]] = []
    hours_summary: List[Dict[str, object]] = []

    for target_dt, split_row in selected:
        feather_path = _resolve_feather_path(split_row["file"], args.feather_root)
        print(f"Checking {target_string(target_dt)} from {feather_path.name}")
        context = load_era5_pair_as_tensors(data_root=str(args.data_root), target=target_string(target_dt))
        hour_era5_rows = _era5_band_rows(target_dt, context)
        hour_cloudsat_rows = _cloudsat_band_rows(target_dt, feather_path)
        era5_rows.extend(hour_era5_rows)
        cloudsat_rows.extend(hour_cloudsat_rows)
        invalid_values = sum(int(row["invalid_count"]) for row in hour_era5_rows)
        cloudsat_count = sum(int(row["sample_count"]) for row in hour_cloudsat_rows)
        hours_summary.append(
            {
                "target_hour_utc": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
                "source_feather_file": str(feather_path),
                "era5_invalid_value_count": invalid_values,
                "cloudsat_rows_for_target": cloudsat_count,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    era5_path = args.output_dir / "era5_latitude_band_statistics.csv"
    cloudsat_path = args.output_dir / "cloudsat_latitude_band_statistics.csv"
    summary_path = args.output_dir / "summary.json"
    pd.DataFrame(era5_rows).to_csv(era5_path, index=False)
    pd.DataFrame(cloudsat_rows).to_csv(cloudsat_path, index=False)
    summary = {
        "split_path": str(args.split_path),
        "data_root": str(args.data_root),
        "sample_hours_requested": args.sample_hours,
        "hours_checked": hours_summary,
        "era5_statistics_csv": str(era5_path),
        "cloudsat_statistics_csv": str(cloudsat_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
