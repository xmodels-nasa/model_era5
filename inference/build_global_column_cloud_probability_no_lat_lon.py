#!/usr/bin/env python3
"""Run global cloud-mask inference using the results-v3 no-lat/lon model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from build_global_column_cloud_probability import (
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
    has_era5_pair,
    predict_global_cloud_probabilities,
    resolve_device,
    save_dataset,
    select_test_target,
    target_string,
)
import build_aurora_batches as aurora_batches


DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v3" / "model_outputs_transformer_no_lat_lon"
DEFAULT_V3_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "no_lat_lon"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data_era5")))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_MODEL_DIR / "file_split.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default=None, help="Override random split selection with YYYY_MM_DD_HH.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_V3_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aurora-device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument("--aurora-backbone", choices=["full", "small"], default="full")
    parser.add_argument("--tokens-on-device", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    aurora_device = resolve_device(args.aurora_device or args.device)
    selected_row = None
    available_candidate_count = None
    if args.target:
        target_dt = aurora_batches.parse_target(args.target)
        if not has_era5_pair(args.data_root, target_dt):
            raise FileNotFoundError(f"Target {args.target} is missing its required ERA5 input pair.")
    else:
        target_dt, selected_row, available_candidate_count = select_test_target(
            split_path=args.split_path,
            split_name=args.split,
            data_root=args.data_root,
            seed=args.seed,
        )

    output_path = args.output or args.output_dir / f"global_cloud_probabilities_no_lat_lon_{target_string(target_dt)}.nc"
    print(f"Target hour: {target_dt.strftime('%Y-%m-%d %H:00 UTC')}")
    print(f"Classifier device: {device}; Aurora device: {aurora_device}")
    cloud_mask_prob, column_cloud_prob, lat, lon, attrs = predict_global_cloud_probabilities(
        target_dt=target_dt,
        data_root=args.data_root,
        model_dir=args.model_dir,
        device=device,
        aurora_device=aurora_device,
        batch_size=args.batch_size,
        row_chunk_size=args.row_chunk_size,
        base_lon_convention="minus180_180",
        tokens_on_device=args.tokens_on_device,
        aurora_backbone=args.aurora_backbone,
    )
    save_dataset(
        output_path=output_path,
        cloud_mask_prob=cloud_mask_prob,
        column_cloud_prob=column_cloud_prob,
        lat=lat,
        lon=lon,
        attrs=attrs,
        selected_row=selected_row,
        available_candidate_count=available_candidate_count,
    )
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
