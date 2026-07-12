#!/usr/bin/env python3
"""Measure baseline raw-chip calibration by latitude on held-out test tracks."""

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
BASELINE_DIR = PROJECT_ROOT / "baseline_model"
for path in (BASELINE_DIR, THIS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multilabel_from_raw_chips as raw_train  # noqa: E402
from build_global_column_cloud_probability_baseline_raw_chips import load_baseline_model  # noqa: E402
from build_global_column_cloud_probability import resolve_device  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v2" / "baseline_model_outputs"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "baseline_test_latitude_check"
LATITUDE_BANDS = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]


def _resolve_npz_path(value: str, raw_chip_dir: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidate = raw_chip_dir / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find raw-chip file: {value}")


def _load_test_metas(split_path: Path, raw_chip_dir: Path, max_files: int) -> List[raw_train.FileMeta]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    metas: List[raw_train.FileMeta] = []
    with split_path.open("r", encoding="utf-8", newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            if row.get("split") != "test":
                continue
            source_file = Path(row["file"])
            npz_path = _resolve_npz_path(row["npz"], raw_chip_dir)
            metas.append(
                raw_train.FileMeta(
                    source_file=source_file,
                    npz_path=npz_path,
                    file_time=pd.Timestamp(row["file_time_utc"]),
                )
            )
    if max_files > 0:
        metas = metas[:max_files]
    if not metas:
        raise ValueError("No test raw-chip files selected.")
    return metas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_MODEL_DIR / "file_split.csv")
    parser.add_argument("--raw-chip-dir", type=Path, default=Path(os.getenv("RAW_CHIPS_DIR", PROJECT_ROOT / "raw_chips")))
    parser.add_argument("--max-files", type=int, default=0, help="0 means all held-out test files.")
    parser.add_argument("--max-samples-per-file", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
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
    model, stats, checkpoint = load_baseline_model(args.model_dir, device)
    threshold = float(args.threshold if args.threshold is not None else checkpoint["test_metrics"]["iou_threshold"])
    metas = _load_test_metas(args.split_path, args.raw_chip_dir, args.max_files)
    use_base_features = bool(checkpoint.get("base_features", raw_train.BASE_FEATURE_COLUMNS))

    totals: Dict[tuple[int, int], Dict[str, float]] = {
        band: {"sample_count": 0.0, "truth_sum": 0.0, "prediction_sum": 0.0, "brier_sum": 0.0, "positive_prediction_count": 0.0}
        for band in LATITUDE_BANDS
    }
    files_processed: List[Dict[str, object]] = []
    for file_index, meta in enumerate(metas):
        payload = raw_train._load_npz_payload(
            meta=meta,
            sample_ratio=1.0,
            max_samples_per_file=(None if args.max_samples_per_file == 0 else args.max_samples_per_file),
            seed=args.seed + file_index,
        )
        if not len(payload["labels"]):
            continue
        chips = raw_train._flatten_chip_channels(payload["dynamic"], payload["static"])
        base = payload["base_features"]
        truth = (payload["labels"] > 0.5).any(axis=1).astype(np.float32)
        column_prob = np.empty(len(truth), dtype=np.float32)
        for start in range(0, len(truth), args.batch_size):
            stop = min(start + args.batch_size, len(truth))
            chips_t, base_t = raw_train._normalize_batch(
                chips=chips[start:stop],
                base_features=base[start:stop],
                stats=stats,
                use_base_features=use_base_features,
            )
            with torch.inference_mode():
                probs = torch.sigmoid(model(chips_t.to(device), base_t.to(device)))
            column_prob[start:stop] = probs.max(dim=1).values.detach().cpu().numpy()

        lat = payload["base_features"][:, 0]
        for band, values in totals.items():
            low, high = band
            mask = (lat >= low) & (lat < high)
            if not mask.any():
                continue
            selected_truth = truth[mask]
            selected_prob = column_prob[mask]
            values["sample_count"] += int(mask.sum())
            values["truth_sum"] += float(selected_truth.sum())
            values["prediction_sum"] += float(selected_prob.sum())
            values["brier_sum"] += float(np.square(selected_prob - selected_truth).sum())
            values["positive_prediction_count"] += int((selected_prob >= threshold).sum())
        files_processed.append({"raw_chip_file": str(meta.npz_path), "sample_count": int(len(truth))})
        print(f"Processed {file_index + 1}/{len(metas)}: {meta.npz_path.name} ({len(truth):,} samples)")

    rows: List[Dict[str, object]] = []
    for (low, high), values in totals.items():
        count = int(values["sample_count"])
        rows.append(
            {
                "lat_min": low,
                "lat_max": high,
                "sample_count": count,
                "observed_column_cloud_fraction": values["truth_sum"] / count if count else None,
                "mean_predicted_column_probability": values["prediction_sum"] / count if count else None,
                "predicted_cloud_fraction_at_threshold": values["positive_prediction_count"] / count if count else None,
                "column_brier_score": values["brier_sum"] / count if count else None,
                "threshold": threshold,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "test_prediction_by_latitude.csv"
    summary_path = args.output_dir / "summary.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    summary = {
        "model_dir": str(args.model_dir),
        "split_path": str(args.split_path),
        "raw_chip_dir": str(args.raw_chip_dir),
        "device": device,
        "threshold": threshold,
        "test_files_selected": len(metas),
        "files_processed": files_processed,
        "latitude_summary_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
