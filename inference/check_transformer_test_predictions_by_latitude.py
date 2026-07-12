#!/usr/bin/env python3
"""Measure embedding-Transformer calibration by latitude/longitude on test tracks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"
for path in (FINE_TUNED_DIR, THIS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multilabel_from_feather_embeddings as emb_train  # noqa: E402
from build_global_column_cloud_probability import load_transformer, resolve_device  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v2" / "model_outputs_transformer"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "transformer_test_latitude_check"
LATITUDE_BANDS = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
LONGITUDE_BANDS = [(-180, 0), (0, 180)]


def _new_metrics() -> Dict[str, float]:
    return {
        "sample_count": 0.0,
        "truth_sum": 0.0,
        "prediction_sum": 0.0,
        "brier_sum": 0.0,
        "positive_prediction_count": 0.0,
    }


def _summary_row(
    lat_low: int,
    lat_high: int,
    lon_low: int,
    lon_high: int,
    values: Dict[str, float],
    threshold: float,
    day_utc: str | None = None,
) -> Dict[str, object]:
    count = int(values["sample_count"])
    row: Dict[str, object] = {
        "lat_min": lat_low,
        "lat_max": lat_high,
        "lon_min": lon_low,
        "lon_max": lon_high,
        "sample_count": count,
        "observed_column_cloud_fraction": values["truth_sum"] / count if count else None,
        "mean_predicted_column_probability": values["prediction_sum"] / count if count else None,
        "predicted_cloud_fraction_at_threshold": values["positive_prediction_count"] / count if count else None,
        "column_brier_score": values["brier_sum"] / count if count else None,
        "threshold": threshold,
    }
    if day_utc is not None:
        row["day_utc"] = day_utc
    return row


def _resolve_feather_path(value: str, feather_root: Path | None) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    if feather_root is not None:
        candidate = feather_root / path.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find Feather file: {value}")


def _load_test_metas(
    split_path: Path,
    embedding_dir: Path,
    feather_root: Path | None,
    max_files: int,
) -> List[emb_train.FileMeta]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    metas: List[emb_train.FileMeta] = []
    with split_path.open("r", encoding="utf-8", newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            if row.get("split") != "test":
                continue
            feather_path = _resolve_feather_path(row["file"], feather_root)
            npz_path = embedding_dir / f"{emb_train._safe_stem(feather_path)}.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing embedding file: {npz_path}")
            metas.append(
                emb_train.FileMeta(
                    feather_path=feather_path,
                    npz_path=npz_path,
                    file_time=pd.Timestamp(row["file_time_utc"]),
                )
            )
    if max_files > 0:
        metas = metas[:max_files]
    if not metas:
        raise ValueError("No test embedding files selected.")
    return metas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_MODEL_DIR / "file_split.csv")
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path(os.getenv("EMBEDDING_OUTUT_DIR", os.getenv("EMBEDDING_OUTPUT_DIR", PROJECT_ROOT / "embeddings"))),
    )
    parser.add_argument("--feather-root", type=Path, default=os.getenv("FEATHER_ROOT") or None)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all held-out test files.")
    parser.add_argument("--max-samples-per-file", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples_per_file < 0:
        raise ValueError("--max-samples-per-file must be 0 or positive.")
    device = resolve_device(args.device)
    model, x_mean, x_std, checkpoint = load_transformer(args.model_dir, device)
    base_features = list(checkpoint.get("base_features", emb_train.BASE_FEATURE_COLUMNS))
    if "Latitude_0" not in base_features or "Longitude_0" not in base_features:
        raise ValueError("This check requires a model checkpoint with Latitude_0 and Longitude_0 base features.")
    latitude_index = base_features.index("Latitude_0")
    longitude_index = base_features.index("Longitude_0")
    threshold = float(args.threshold if args.threshold is not None else checkpoint["test_metrics"]["iou_threshold"])
    metas = _load_test_metas(args.split_path, args.embedding_dir, args.feather_root, args.max_files)
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)

    totals: Dict[tuple[int, int, int, int], Dict[str, float]] = {
        (*lat_band, *lon_band): _new_metrics()
        for lat_band in LATITUDE_BANDS
        for lon_band in LONGITUDE_BANDS
    }
    daily_totals: Dict[str, Dict[tuple[int, int, int, int], Dict[str, float]]] = {}
    files_processed: List[Dict[str, object]] = []
    for file_index, meta in enumerate(metas):
        x, y, _ = emb_train._load_one_file_arrays(
            meta=meta,
            sample_ratio=1.0,
            max_samples_per_file=(None if args.max_samples_per_file == 0 else args.max_samples_per_file),
            seed=args.seed + file_index,
        )
        if not len(y):
            continue
        truth = (y > 0.5).any(axis=1).astype(np.float32)
        column_prob = np.empty(len(truth), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(truth), args.batch_size):
                stop = min(start + args.batch_size, len(truth))
                inputs = torch.from_numpy(x[start:stop]).to(device)
                probs = torch.sigmoid(model((inputs - x_mean_t) / x_std_t))
                column_prob[start:stop] = probs.max(dim=1).values.detach().cpu().numpy()

        lat = x[:, latitude_index]
        lon = x[:, longitude_index]
        file_time = pd.Timestamp(meta.file_time)
        if file_time.tzinfo is None:
            file_time = file_time.tz_localize("UTC")
        else:
            file_time = file_time.tz_convert("UTC")
        day_utc = file_time.strftime("%Y-%m-%d")
        day_metrics = daily_totals.setdefault(day_utc, {})
        for (lat_low, lat_high, lon_low, lon_high), values in totals.items():
            lon_upper = lon <= lon_high if lon_high == 180 else lon < lon_high
            mask = (lat >= lat_low) & (lat < lat_high) & (lon >= lon_low) & lon_upper
            if not mask.any():
                continue
            selected_truth = truth[mask]
            selected_prob = column_prob[mask]
            region = (lat_low, lat_high, lon_low, lon_high)
            daily_values = day_metrics.setdefault(region, _new_metrics())
            for metrics in (values, daily_values):
                metrics["sample_count"] += int(mask.sum())
                metrics["truth_sum"] += float(selected_truth.sum())
                metrics["prediction_sum"] += float(selected_prob.sum())
                metrics["brier_sum"] += float(np.square(selected_prob - selected_truth).sum())
                metrics["positive_prediction_count"] += int((selected_prob >= threshold).sum())
        files_processed.append({"embedding_file": str(meta.npz_path), "sample_count": int(len(truth))})
        print(f"Processed {file_index + 1}/{len(metas)}: {meta.npz_path.name} ({len(truth):,} samples)")

    rows: List[Dict[str, object]] = []
    for (lat_low, lat_high, lon_low, lon_high), values in totals.items():
        rows.append(_summary_row(lat_low, lat_high, lon_low, lon_high, values, threshold))
    daily_rows: List[Dict[str, object]] = []
    for day_utc, regions in sorted(daily_totals.items()):
        for (lat_low, lat_high, lon_low, lon_high), values in regions.items():
            daily_rows.append(_summary_row(lat_low, lat_high, lon_low, lon_high, values, threshold, day_utc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "test_prediction_by_latitude_longitude.csv"
    daily_csv_path = args.output_dir / "daily_prediction_by_latitude_longitude.csv"
    summary_path = args.output_dir / "summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(daily_rows).to_csv(daily_csv_path, index=False)
    summary = {
        "model_dir": str(args.model_dir),
        "split_path": str(args.split_path),
        "embedding_dir": str(args.embedding_dir),
        "device": device,
        "threshold": threshold,
        "test_files_selected": len(metas),
        "files_processed": files_processed,
        "latitude_longitude_summary_csv": str(csv_path),
        "daily_latitude_longitude_summary_csv": str(daily_csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {csv_path}")
    print(pd.DataFrame(daily_rows).to_string(index=False))
    print(f"Saved daily regional summary: {daily_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
