#!/usr/bin/env python3
"""Run global cloud-mask inference for one held-out test hour.

This script selects one random test timestamp from the transformer split
manifest, builds the Aurora encoder context once for that hour, scores every
Aurora-cropped ERA5 grid point with the saved embedding transformer, then saves
both the full 40-level probability cube and the column probability:

    cloud_mask_prob[level, latitude, longitude] = probability per mask level

    column_cloud_prob[latitude, longitude] = max(probability over 40 levels)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

for path in (FINE_TUNED_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multilabel_from_feather_embeddings as emb_train  # noqa: E402
import build_aurora_batches as aurora_batches  # noqa: E402
from train_multilabel_from_feather_embeddings_transformer import (  # noqa: E402
    EmbeddingTransformerClassifier,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results-v2"
DEFAULT_MODEL_DIR = DEFAULT_RESULTS_DIR / "model_outputs_transformer"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs"


def torch_load(path: Path, device: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def timestamp_to_rounded_hour(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    floor = ts.floor("h")
    if ts - floor > pd.Timedelta(minutes=30):
        floor = floor + pd.Timedelta(hours=1)
    return floor.to_pydatetime().replace(tzinfo=None)


def target_string(dt: datetime) -> str:
    return dt.strftime("%Y_%m_%d_%H")


def has_era5_pair(data_root: Path, target_dt: datetime) -> bool:
    return (data_root / aurora_batches.folder_name(target_dt - timedelta(hours=6))).is_dir() and (
        data_root / aurora_batches.folder_name(target_dt)
    ).is_dir()


def select_test_target(
    split_path: Path,
    split_name: str,
    data_root: Path,
    seed: int,
) -> Tuple[datetime, Dict[str, str], int]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")

    rows: List[Dict[str, str]] = []
    with split_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == split_name:
                rows.append(row)
    if not rows:
        raise ValueError(f"No rows with split={split_name!r} in {split_path}")

    candidates: List[Tuple[datetime, Dict[str, str]]] = []
    unavailable: List[str] = []
    for row in rows:
        target_dt = timestamp_to_rounded_hour(row["file_time_utc"])
        if has_era5_pair(data_root, target_dt):
            candidates.append((target_dt, row))
        else:
            unavailable.append(target_string(target_dt))

    if not candidates:
        preview = ", ".join(sorted(set(unavailable))[:10])
        raise FileNotFoundError(
            f"No {split_name} split hours have both ERA5 input folders under {data_root}. "
            f"Checked {len(rows)} split rows. First unavailable rounded target hours: {preview}"
        )

    rng = random.Random(seed)
    return (*rng.choice(candidates), len(candidates))


def cyclic_time_features(target_dt: datetime) -> Tuple[float, float, float, float]:
    dt = target_dt.replace(tzinfo=timezone.utc)
    day_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    day_phase = 2.0 * math.pi * day_seconds / 86400.0

    year_start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    next_year = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    year_seconds = (next_year - year_start).total_seconds()
    year_phase = 2.0 * math.pi * (dt - year_start).total_seconds() / year_seconds

    return (
        math.sin(day_phase),
        math.cos(day_phase),
        math.sin(year_phase),
        math.cos(year_phase),
    )


def base_longitudes(lon: np.ndarray, convention: str) -> np.ndarray:
    if convention == "era5":
        return lon.astype(np.float32, copy=False)
    if convention == "minus180_180":
        return (((lon + 180.0) % 360.0) - 180.0).astype(np.float32, copy=False)
    raise ValueError(f"Unknown longitude convention: {convention}")


def load_transformer(model_dir: Path, device: str) -> Tuple[EmbeddingTransformerClassifier, np.ndarray, np.ndarray, Dict[str, object]]:
    model_path = model_dir / "multilabel_transformer.pt"
    stats_path = model_dir / "feature_stats.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing feature stats: {stats_path}")

    ckpt = torch_load(model_path, device)
    stats = np.load(stats_path)
    model = EmbeddingTransformerClassifier(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", len(emb_train.TARGET_COLUMNS))),
        hidden_dims=ckpt.get("transformer_config"),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, stats["x_mean"].astype(np.float32), stats["x_std"].astype(np.float32), ckpt


def predict_global_cloud_probabilities(
    *,
    target_dt: datetime,
    data_root: Path,
    model_dir: Path,
    device: str,
    aurora_device: str,
    batch_size: int,
    row_chunk_size: int,
    base_lon_convention: str,
    tokens_on_device: bool,
    aurora_backbone: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    model, x_mean, x_std, ckpt = load_transformer(model_dir, device)
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)

    if aurora_backbone == "full":
        aurora_batches.DEBUG = False
    elif aurora_backbone == "small":
        aurora_batches.DEBUG = True
    else:
        raise ValueError(f"Unknown aurora_backbone={aurora_backbone!r}")

    aurora_model = aurora_batches.load_aurora_model(device=aurora_device)
    context = aurora_batches.get_encoder_context_for_target(
        data_root=str(data_root),
        target=target_string(target_dt),
        model=aurora_model,
    )
    enc_out = context["enc_out"]
    enc_batch = context["enc_batch"]
    local_aurora_model = context["model"]

    lat = enc_batch.metadata.lat.detach().cpu().numpy().astype(np.float32)
    lon = enc_batch.metadata.lon.detach().cpu().numpy().astype(np.float32)
    h_grid, w_grid = enc_batch.spatial_shape
    patch_size = int(local_aurora_model.encoder.patch_size)
    h_patches = int(h_grid // patch_size)
    w_patches = int(w_grid // patch_size)
    patch_count = int(h_patches * w_patches)
    latent_levels = int(local_aurora_model.encoder.latent_levels)
    embed_dim = int(enc_out.shape[-1])
    expected_embedding_dim = int(ckpt["input_dim"]) - len(emb_train.BASE_FEATURE_COLUMNS)

    if latent_levels * embed_dim != expected_embedding_dim:
        raise ValueError(
            "Checkpoint input dimension does not match Aurora encoder output: "
            f"latent_levels={latent_levels}, embed_dim={embed_dim}, "
            f"latent_levels*embed_dim={latent_levels * embed_dim}, "
            f"expected_embedding_dim={expected_embedding_dim}"
        )

    token_device = device if tokens_on_device else "cpu"
    level_tokens = (
        enc_out.detach()[0, : latent_levels * patch_count, :]
        .reshape(latent_levels, patch_count, embed_dim)
        .to(token_device, dtype=torch.float32)
    )

    del enc_out
    if device == "cuda":
        torch.cuda.empty_cache()

    day_sin, day_cos, year_sin, year_cos = cyclic_time_features(target_dt)
    lon_base = base_longitudes(lon, base_lon_convention)
    output_levels = np.empty((len(emb_train.TARGET_COLUMNS), len(lat), len(lon)), dtype=np.float32)

    model_input_dim = int(ckpt["input_dim"])
    print(
        "Global inference grid: "
        f"lat={len(lat)}, lon={len(lon)}, patch_size={patch_size}, "
        f"patches={h_patches}x{w_patches}, input_dim={model_input_dim}"
    )
    print(f"Scoring {len(lat) * len(lon):,} grid points in batches of {batch_size}")

    with torch.inference_mode():
        for row_start in range(0, len(lat), row_chunk_size):
            row_stop = min(len(lat), row_start + row_chunk_size)
            lat_indices = np.arange(row_start, row_stop, dtype=np.int64)
            lon_indices = np.arange(len(lon), dtype=np.int64)

            patch_idx = (
                (lat_indices[:, None] // patch_size) * w_patches
                + (lon_indices[None, :] // patch_size)
            ).reshape(-1)
            lat_flat = np.repeat(lat[row_start:row_stop], len(lon))
            lon_base_flat = np.tile(lon_base, row_stop - row_start)
            n_rows = patch_idx.shape[0]
            chunk_probs = np.empty((n_rows, len(emb_train.TARGET_COLUMNS)), dtype=np.float32)

            for start in range(0, n_rows, batch_size):
                stop = min(n_rows, start + batch_size)
                patch_idx_t = torch.from_numpy(patch_idx[start:stop]).to(token_device)
                emb = level_tokens[:, patch_idx_t, :].permute(1, 0, 2).reshape(stop - start, -1)
                emb = emb.to(device)

                base_np = np.column_stack(
                    [
                        lat_flat[start:stop],
                        lon_base_flat[start:stop],
                        np.full(stop - start, day_sin, dtype=np.float32),
                        np.full(stop - start, day_cos, dtype=np.float32),
                        np.full(stop - start, year_sin, dtype=np.float32),
                        np.full(stop - start, year_cos, dtype=np.float32),
                    ]
                ).astype(np.float32, copy=False)
                base_t = torch.from_numpy(base_np).to(device)
                x = torch.cat([base_t, emb], dim=1)
                if x.shape[1] != model_input_dim:
                    raise ValueError(f"Built input_dim={x.shape[1]}, expected {model_input_dim}")
                x = (x - x_mean_t) / x_std_t
                probs = torch.sigmoid(model(x))
                chunk_probs[start:stop] = probs.detach().cpu().numpy().astype(np.float32)

            output_levels[:, row_start:row_stop, :] = chunk_probs.reshape(
                row_stop - row_start,
                len(lon),
                len(emb_train.TARGET_COLUMNS),
            ).transpose(2, 0, 1)
            print(f"Completed latitude rows {row_start}:{row_stop}")

    column_cloud_prob = output_levels.max(axis=0)
    attrs: Dict[str, object] = {
        "target_time_utc": target_dt.strftime("%Y-%m-%dT%H:00:00Z"),
        "data_root": str(data_root),
        "model_dir": str(model_dir),
        "model_architecture": str(ckpt.get("architecture", "embedding_transformer_classifier")),
        "model_input_dim": model_input_dim,
        "base_features": json.dumps(list(ckpt.get("base_features", emb_train.BASE_FEATURE_COLUMNS))),
        "base_longitude_convention": base_lon_convention,
        "reduction": "max over 40 transformer output levels",
        "aurora_patch_size": patch_size,
        "aurora_latent_levels": latent_levels,
        "aurora_embed_dim": embed_dim,
        "aurora_cropped_grid": "true",
        "created_by": Path(__file__).name,
    }
    return output_levels, column_cloud_prob, lat, lon, attrs


def save_dataset(
    output_path: Path,
    cloud_mask_prob: np.ndarray,
    column_cloud_prob: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    attrs: Dict[str, object],
    selected_row: Optional[Dict[str, str]],
    available_candidate_count: Optional[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            "cloud_mask_prob": (
                ("level", "latitude", "longitude"),
                cloud_mask_prob.astype(np.float32, copy=False),
                {
                    "long_name": "cloud mask probability by vertical mask level",
                    "description": "Predicted cloud probability for each of the 40 vertical mask levels.",
                    "valid_min": 0.0,
                    "valid_max": 1.0,
                },
            ),
            "column_cloud_prob": (
                ("latitude", "longitude"),
                column_cloud_prob.astype(np.float32, copy=False),
                {
                    "long_name": "column cloud probability",
                    "description": "Maximum predicted cloud probability over the 40 vertical mask levels.",
                    "valid_min": 0.0,
                    "valid_max": 1.0,
                },
            )
        },
        coords={
            "level": ("level", np.arange(cloud_mask_prob.shape[0], dtype=np.int16)),
            "latitude": ("latitude", lat.astype(np.float32, copy=False)),
            "longitude": ("longitude", lon.astype(np.float32, copy=False)),
        },
        attrs=attrs,
    )
    if selected_row is not None:
        ds.attrs["selected_split_file"] = selected_row.get("file", "")
        ds.attrs["selected_split_file_time_utc"] = selected_row.get("file_time_utc", "")
    if available_candidate_count is not None:
        ds.attrs["available_test_candidate_count"] = int(available_candidate_count)

    encoding = {
        "cloud_mask_prob": {"zlib": True, "complevel": 4, "dtype": "float32"},
        "column_cloud_prob": {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    ds.to_netcdf(output_path, encoding=encoding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data_era5")))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_MODEL_DIR / "file_split.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", default=None, help="Override random split selection with YYYY_MM_DD_HH.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aurora-device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument("--base-lon-convention", choices=["minus180_180", "era5"], default="minus180_180")
    parser.add_argument(
        "--aurora-backbone",
        choices=["full", "small"],
        default="full",
        help="Aurora backbone used to build embeddings. The saved transformer expects full Aurora by default.",
    )
    parser.add_argument(
        "--tokens-on-device",
        action="store_true",
        help="Keep the full Aurora token tensor on the classifier device. Faster, but uses much more device memory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    aurora_device = resolve_device(args.aurora_device or args.device)

    selected_row: Optional[Dict[str, str]] = None
    available_candidate_count: Optional[int] = None
    if args.target:
        target_dt = aurora_batches.parse_target(args.target)
        if not has_era5_pair(args.data_root, target_dt):
            raise FileNotFoundError(
                f"Target {args.target} is missing required ERA5 pair: "
                f"{aurora_batches.folder_name(target_dt - timedelta(hours=6))} "
                f"and/or {aurora_batches.folder_name(target_dt)}"
            )
    else:
        target_dt, selected_row, available_candidate_count = select_test_target(
            split_path=args.split_path,
            split_name=args.split,
            data_root=args.data_root,
            seed=args.seed,
        )

    output_path = args.output
    if output_path is None:
        output_path = args.output_dir / f"global_cloud_probabilities_{target_string(target_dt)}.nc"

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
        base_lon_convention=args.base_lon_convention,
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
