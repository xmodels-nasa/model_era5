#!/usr/bin/env python3
"""Run global cloud-mask inference with the saved raw-ERA5-chip U-Net model.

This rebuilds the same 9x9 dynamic/static ERA5 chips, six base features, and
normalization used by ``baseline_model/train_multilabel_from_raw_chips.py``.
It saves all 40 probabilities and their column-wise maximum.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
BASELINE_DIR = PROJECT_ROOT / "baseline_model"
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"
for path in (BASELINE_DIR, FINE_TUNED_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multilabel_from_raw_chips as raw_train  # noqa: E402
from build_raw_chips_from_feather_file import load_era5_pair_as_tensors  # noqa: E402
from build_global_column_cloud_probability import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    has_era5_pair,
    resolve_device,
    save_dataset,
    select_test_target,
    target_string,
)
import build_aurora_batches as aurora_batches  # noqa: E402


DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v2" / "baseline_model_outputs"
DEFAULT_BASELINE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "baseline_raw_chips"


def torch_load(path: Path, device: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _signed_longitudes(lon: np.ndarray) -> np.ndarray:
    return (((lon + 180.0) % 360.0) - 180.0).astype(np.float32, copy=False)


def _extract_global_chips(
    dynamic: np.ndarray,
    static: np.ndarray,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    chip_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(chip_size, dtype=np.int64) - chip_size // 2
    rows = np.clip(lat_indices[:, None] + offsets[None, :], 0, dynamic.shape[-2] - 1)
    cols = np.mod(lon_indices[:, None] + offsets[None, :], dynamic.shape[-1])

    dynamic_chips = dynamic[:, :, rows[:, :, None], cols[:, None, :]].transpose(2, 0, 1, 3, 4)
    static_chips = static[:, rows[:, :, None], cols[:, None, :]].transpose(1, 0, 2, 3)
    return dynamic_chips.astype(np.float32, copy=False), static_chips.astype(np.float32, copy=False)


def load_baseline_model(
    model_dir: Path, device: str
) -> Tuple[torch.nn.Module, raw_train.NormalizationStats, Dict[str, object]]:
    model_path = model_dir / "multilabel_unet_classifier.pt"
    stats_path = model_dir / "normalization_stats.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")

    checkpoint = torch_load(model_path, device)
    base_features = checkpoint.get("base_features", raw_train.BASE_FEATURE_COLUMNS)
    if not isinstance(base_features, (list, tuple)):
        raise ValueError("Checkpoint base_features must be a list.")
    model = raw_train.UNetClassifier(
        in_channels=int(checkpoint["input_channels"]),
        output_dim=int(checkpoint.get("output_dim", len(raw_train.TARGET_COLUMNS))),
        base_feature_dim=len(base_features),
        base_channels=int(checkpoint.get("base_channels", 32)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats_npz = np.load(stats_path)
    stats = raw_train.NormalizationStats(
        chip_mean=stats_npz["chip_mean"].astype(np.float32),
        chip_std=stats_npz["chip_std"].astype(np.float32),
        base_mean=stats_npz["base_mean"].astype(np.float32),
        base_std=stats_npz["base_std"].astype(np.float32),
    )
    return model, stats, checkpoint


def predict_global_cloud_probabilities(
    *,
    target_dt: datetime,
    data_root: Path,
    model_dir: Path,
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    model, stats, checkpoint = load_baseline_model(model_dir, device)
    context = load_era5_pair_as_tensors(data_root=str(data_root), target=target_string(target_dt))
    dynamic = context["dynamic"]
    static = context["static"]
    lat = context["lat"]
    lon = context["lon"]
    chip_size = int(checkpoint["chip_size"])
    base_features = list(checkpoint.get("base_features", raw_train.BASE_FEATURE_COLUMNS))
    use_base_features = bool(base_features)
    expected_channel_names = np.asarray(checkpoint["channel_names"])
    actual_channel_names = raw_train._time_expanded_channel_names(
        dynamic_channel_names=context["dynamic_channel_names"],
        static_channel_names=context["static_channel_names"],
    )
    if not np.array_equal(expected_channel_names, actual_channel_names):
        raise ValueError("ERA5 raw-chip channel order does not match the saved baseline checkpoint.")
    expected_channels = int(checkpoint["input_channels"])
    if dynamic.shape[0] * dynamic.shape[1] + static.shape[0] != expected_channels:
        raise ValueError("ERA5 raw-chip channel count does not match the saved baseline checkpoint.")

    height, width = len(lat), len(lon)
    total_points = height * width
    output_levels = np.empty((len(raw_train.TARGET_COLUMNS), height, width), dtype=np.float32)
    flat_lat_indices = np.repeat(np.arange(height, dtype=np.int64), width)
    flat_lon_indices = np.tile(np.arange(width, dtype=np.int64), height)
    target_timestamp = int(target_dt.replace(tzinfo=timezone.utc).timestamp())

    print(
        f"Global raw-chip grid: lat={height}, lon={width}, chip_size={chip_size}, "
        f"channels={expected_channels}, base_features={base_features}"
    )
    print(f"Scoring {total_points:,} grid points in batches of {batch_size}")
    with torch.inference_mode():
        for start in range(0, total_points, batch_size):
            stop = min(start + batch_size, total_points)
            lat_idx = flat_lat_indices[start:stop]
            lon_idx = flat_lon_indices[start:stop]
            dynamic_chips, static_chips = _extract_global_chips(
                dynamic=dynamic,
                static=static,
                lat_indices=lat_idx,
                lon_indices=lon_idx,
                chip_size=chip_size,
            )
            chips = raw_train._flatten_chip_channels(dynamic_chips, static_chips)
            full_base = raw_train._timestamps_to_base_features(
                latitudes=lat[lat_idx],
                longitudes=_signed_longitudes(lon[lon_idx]),
                timestamps=np.full(stop - start, target_timestamp, dtype=np.float64),
            )
            if use_base_features:
                feature_positions = [raw_train.BASE_FEATURE_COLUMNS.index(name) for name in base_features]
                base = full_base[:, feature_positions]
            else:
                base = np.empty((stop - start, 0), dtype=np.float32)
            chips_t, base_t = raw_train._normalize_batch(
                chips=chips,
                base_features=base,
                stats=stats,
                use_base_features=use_base_features,
            )
            probs = torch.sigmoid(model(chips_t.to(device), base_t.to(device)))
            output_levels[:, lat_idx, lon_idx] = probs.detach().cpu().numpy().T.astype(np.float32, copy=False)
            if stop == total_points or start // batch_size % 100 == 0:
                print(f"Completed {stop:,}/{total_points:,} grid points")

    column_cloud_prob = output_levels.max(axis=0)
    attrs: Dict[str, object] = {
        "target_time_utc": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
        "data_root": str(data_root),
        "model_dir": str(model_dir),
        "model_architecture": "raw_chip_unet_classifier",
        "model_input_channels": expected_channels,
        "chip_size": chip_size,
        "base_features": json.dumps(base_features),
        "base_longitude_convention": "minus180_180",
        "reduction": "max over 40 U-Net classifier output levels",
        "created_by": Path(__file__).name,
    }
    return output_levels, column_cloud_prob, lat, lon, attrs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data_era5")))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_MODEL_DIR / "file_split.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default=None, help="Override random split selection with YYYY_MM_DD_HH.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
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
    output_path = args.output or args.output_dir / f"global_cloud_probabilities_baseline_raw_chips_{target_string(target_dt)}.nc"
    print(f"Target hour: {target_dt.strftime('%Y-%m-%d %H:00 UTC')}")
    print(f"Classifier device: {device}")
    cloud_mask_prob, column_cloud_prob, lat, lon, attrs = predict_global_cloud_probabilities(
        target_dt=target_dt,
        data_root=args.data_root,
        model_dir=args.model_dir,
        device=device,
        batch_size=args.batch_size,
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
