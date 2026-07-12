#!/usr/bin/env python3
"""Check CloudSat training-data coverage and cloud-label density.

This diagnostic reads the saved split manifest and the corresponding Feather
files, bins rows into a regular latitude/longitude grid, and saves maps for:

- sample count
- column cloud fraction, defined as any of the 40 target bins being cloudy
- mean cloudy-bin fraction across the 40 target bins

Use it to compare training coverage against global inference artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "results-v2" / "model_outputs_transformer"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "training_coverage"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


TARGET_COLUMNS = [f"y_40dim_{i}" for i in range(40)]
REQUIRED_COLUMNS = ["Latitude_0", "Longitude_0", *TARGET_COLUMNS]


def load_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, value = s.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def split_rows(split_path: Path, split_names: Sequence[str]) -> List[Dict[str, str]]:
    wanted = set(split_names)
    rows: List[Dict[str, str]] = []
    with split_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") in wanted:
                rows.append(row)
    if not rows:
        raise ValueError(f"No split rows found for {sorted(wanted)} in {split_path}")
    return rows


def path_by_stem(root: Path) -> Dict[str, Path]:
    return {p.stem: p for p in root.rglob("*.feather")}


def resolve_feather_path(row: Dict[str, str], feather_root: Optional[Path]) -> Optional[Path]:
    manifest_path = Path(row["file"])
    if manifest_path.is_file():
        return manifest_path
    if feather_root is None:
        return None
    by_stem = getattr(resolve_feather_path, "_by_stem", None)
    cache_root = getattr(resolve_feather_path, "_cache_root", None)
    if by_stem is None or cache_root != feather_root:
        by_stem = path_by_stem(feather_root)
        setattr(resolve_feather_path, "_by_stem", by_stem)
        setattr(resolve_feather_path, "_cache_root", feather_root)
    return by_stem.get(manifest_path.stem)


def make_edges(resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_edges = np.arange(-90.0, 90.0 + resolution, resolution, dtype=np.float64)
    lon_edges = np.arange(-180.0, 180.0 + resolution, resolution, dtype=np.float64)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    return lat_edges, lon_edges, lat_centers.astype(np.float32), lon_centers.astype(np.float32)


def update_histograms(
    df: pd.DataFrame,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    count: np.ndarray,
    column_cloud_sum: np.ndarray,
    cloudy_bin_fraction_sum: np.ndarray,
) -> int:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    lat = df["Latitude_0"].to_numpy(dtype=np.float64)
    lon = df["Longitude_0"].to_numpy(dtype=np.float64)
    lon = ((lon + 180.0) % 360.0) - 180.0
    targets = df[TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=False) > 0.5
    column_cloud = targets.any(axis=1).astype(np.float32)
    cloudy_bin_fraction = targets.mean(axis=1).astype(np.float32)

    valid = np.isfinite(lat) & np.isfinite(lon)
    if not np.all(valid):
        lat = lat[valid]
        lon = lon[valid]
        column_cloud = column_cloud[valid]
        cloudy_bin_fraction = cloudy_bin_fraction[valid]

    lat_idx = np.searchsorted(lat_edges, lat, side="right") - 1
    lon_idx = np.searchsorted(lon_edges, lon, side="right") - 1
    valid_idx = (
        (lat_idx >= 0)
        & (lat_idx < count.shape[0])
        & (lon_idx >= 0)
        & (lon_idx < count.shape[1])
    )
    lat_idx = lat_idx[valid_idx]
    lon_idx = lon_idx[valid_idx]
    column_cloud = column_cloud[valid_idx]
    cloudy_bin_fraction = cloudy_bin_fraction[valid_idx]

    np.add.at(count, (lat_idx, lon_idx), 1)
    np.add.at(column_cloud_sum, (lat_idx, lon_idx), column_cloud)
    np.add.at(cloudy_bin_fraction_sum, (lat_idx, lon_idx), cloudy_bin_fraction)
    return int(lat_idx.size)


def finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def save_dataset(
    output_path: Path,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    count: np.ndarray,
    column_cloud_sum: np.ndarray,
    cloudy_bin_fraction_sum: np.ndarray,
    attrs: Dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_attrs = {
        key: (json.dumps(value) if isinstance(value, (dict, list, tuple)) else value)
        for key, value in attrs.items()
    }
    column_cloud_fraction = finite_ratio(column_cloud_sum, count)
    mean_cloudy_bin_fraction = finite_ratio(cloudy_bin_fraction_sum, count)
    ds = xr.Dataset(
        data_vars={
            "sample_count": (
                ("latitude", "longitude"),
                count.astype(np.int32, copy=False),
                {"long_name": "training sample count"},
            ),
            "column_cloud_fraction": (
                ("latitude", "longitude"),
                column_cloud_fraction,
                {"long_name": "fraction of samples with any cloudy target bin"},
            ),
            "mean_cloudy_bin_fraction": (
                ("latitude", "longitude"),
                mean_cloudy_bin_fraction,
                {"long_name": "mean fraction of cloudy target bins across 40 levels"},
            ),
        },
        coords={
            "latitude": ("latitude", lat_centers),
            "longitude": ("longitude", lon_centers),
        },
        attrs=serializable_attrs,
    )
    ds.to_netcdf(
        output_path,
        encoding={
            "sample_count": {"zlib": True, "complevel": 4},
            "column_cloud_fraction": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "mean_cloudy_bin_fraction": {"zlib": True, "complevel": 4, "dtype": "float32"},
        },
    )


def plot_map(
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    output_path: Path,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_count: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = data.astype(np.float32, copy=True)
    if log_count:
        with np.errstate(divide="ignore", invalid="ignore"):
            plot_data = np.where(plot_data > 0, np.log10(plot_data), np.nan)
        cbar_label = f"log10({cbar_label})"

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    mesh = ax.pcolormesh(lon, lat, plot_data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.9)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(color="0.7", linewidth=0.4, alpha=0.5)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_region(
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    count: np.ndarray,
    column_cloud_fraction: np.ndarray,
    region: Optional[str],
) -> Optional[Dict[str, object]]:
    if not region:
        return None
    parts = [float(v.strip()) for v in region.split(",")]
    if len(parts) != 4:
        raise ValueError("--region must be 'lat_min,lat_max,lon_min,lon_max'")
    lat_min, lat_max, lon_min, lon_max = parts
    lon_min = ((lon_min + 180.0) % 360.0) - 180.0
    lon_max = ((lon_max + 180.0) % 360.0) - 180.0
    lat_mask = (lat_centers >= min(lat_min, lat_max)) & (lat_centers <= max(lat_min, lat_max))
    if lon_min <= lon_max:
        lon_mask = (lon_centers >= lon_min) & (lon_centers <= lon_max)
    else:
        lon_mask = (lon_centers >= lon_min) | (lon_centers <= lon_max)
    mask = lat_mask[:, None] & lon_mask[None, :]
    selected_count = count[mask]
    selected_frac = column_cloud_fraction[mask]
    covered = selected_count > 0
    return {
        "region": {
            "lat_min": min(lat_min, lat_max),
            "lat_max": max(lat_min, lat_max),
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "bin_count": int(mask.sum()),
        "covered_bin_count": int(covered.sum()),
        "covered_bin_fraction": float(covered.mean()) if covered.size else float("nan"),
        "sample_count": int(selected_count.sum()),
        "mean_column_cloud_fraction_over_covered_bins": (
            float(np.nanmean(selected_frac[covered])) if np.any(covered) else float("nan")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_MODEL_DIR / "file_split.csv")
    parser.add_argument("--split", default="train", help="Comma-separated split names, e.g. train or train,validation.")
    parser.add_argument("--feather-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resolution", type=float, default=2.0, help="Lat/lon bin size in degrees.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--region", default=None, help="Optional diagnostic bbox: lat_min,lat_max,lon_min,lon_max.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    feather_root = args.feather_root or Path(os.getenv("FEATHER_ROOT", ""))
    if not str(feather_root):
        feather_root = None
    elif not feather_root.exists():
        raise FileNotFoundError(f"Feather root does not exist: {feather_root}")

    split_names = [s.strip() for s in args.split.split(",") if s.strip()]
    rows = split_rows(args.split_path, split_names)
    if args.max_files is not None:
        rows = rows[: args.max_files]

    lat_edges, lon_edges, lat_centers, lon_centers = make_edges(args.resolution)
    count = np.zeros((len(lat_centers), len(lon_centers)), dtype=np.int64)
    column_cloud_sum = np.zeros_like(count, dtype=np.float64)
    cloudy_bin_fraction_sum = np.zeros_like(count, dtype=np.float64)

    files_read = 0
    files_missing = 0
    rows_read = 0
    missing_preview: List[str] = []
    for idx, row in enumerate(rows, start=1):
        path = resolve_feather_path(row, feather_root)
        if path is None:
            files_missing += 1
            if len(missing_preview) < 10:
                missing_preview.append(row["file"])
            continue
        df = pd.read_feather(path, columns=REQUIRED_COLUMNS)
        rows_read += update_histograms(
            df=df,
            lat_edges=lat_edges,
            lon_edges=lon_edges,
            count=count,
            column_cloud_sum=column_cloud_sum,
            cloudy_bin_fraction_sum=cloudy_bin_fraction_sum,
        )
        files_read += 1
        print(f"[{idx}/{len(rows)}] {path.name}: rows={len(df):,}")

    column_cloud_fraction = finite_ratio(column_cloud_sum, count)
    mean_cloudy_bin_fraction = finite_ratio(cloudy_bin_fraction_sum, count)
    covered = count > 0
    summary: Dict[str, object] = {
        "split": split_names,
        "split_path": str(args.split_path),
        "feather_root": str(feather_root) if feather_root is not None else "",
        "resolution_degrees": float(args.resolution),
        "manifest_files": int(len(rows)),
        "files_read": int(files_read),
        "files_missing": int(files_missing),
        "missing_preview": missing_preview,
        "rows_read": int(rows_read),
        "grid_bins": int(count.size),
        "covered_bins": int(covered.sum()),
        "covered_bin_fraction": float(covered.mean()),
        "mean_column_cloud_fraction_over_covered_bins": (
            float(np.nanmean(column_cloud_fraction[covered])) if np.any(covered) else float("nan")
        ),
        "mean_cloudy_bin_fraction_over_covered_bins": (
            float(np.nanmean(mean_cloudy_bin_fraction[covered])) if np.any(covered) else float("nan")
        ),
    }
    region_summary = summarize_region(lat_centers, lon_centers, count, column_cloud_fraction, args.region)
    if region_summary is not None:
        summary["region_summary"] = region_summary

    split_label = "_".join(split_names)
    stem = f"training_coverage_{split_label}_{args.resolution:g}deg"
    output_nc = args.output_dir / f"{stem}.nc"
    save_dataset(
        output_path=output_nc,
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        count=count,
        column_cloud_sum=column_cloud_sum,
        cloudy_bin_fraction_sum=cloudy_bin_fraction_sum,
        attrs=summary,
    )

    with (args.output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_map(
        lon=lon_centers,
        lat=lat_centers,
        data=count,
        output_path=args.output_dir / f"{stem}_sample_count.png",
        title=f"Training sample count | split={','.join(split_names)}",
        cbar_label="sample count",
        cmap="magma",
        log_count=True,
    )
    plot_map(
        lon=lon_centers,
        lat=lat_centers,
        data=column_cloud_fraction,
        output_path=args.output_dir / f"{stem}_column_cloud_fraction.png",
        title=f"Observed column cloud fraction | split={','.join(split_names)}",
        cbar_label="fraction with any cloudy target bin",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plot_map(
        lon=lon_centers,
        lat=lat_centers,
        data=mean_cloudy_bin_fraction,
        output_path=args.output_dir / f"{stem}_mean_cloudy_bin_fraction.png",
        title=f"Observed mean cloudy-bin fraction | split={','.join(split_names)}",
        cbar_label="mean fraction of 40 cloudy bins",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_nc}")
    print(f"Saved figures under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
