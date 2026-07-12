#!/usr/bin/env python3
"""Visualize a global column-cloud probability NetCDF file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs"
PROJECT_ROOT = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def prepare_plot_grid(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "column_cloud_prob" not in ds:
        raise KeyError("Input dataset must contain variable 'column_cloud_prob'.")

    data = ds["column_cloud_prob"].values.astype(np.float32)
    lat = ds["latitude"].values.astype(np.float32)
    lon = ds["longitude"].values.astype(np.float32)

    lon_plot = (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)
    lon_order = np.argsort(lon_plot)
    lon_plot = lon_plot[lon_order]
    data = data[:, lon_order]

    lat_order = np.argsort(lat)
    lat_plot = lat[lat_order]
    data = data[lat_order, :]

    return lon_plot, lat_plot, data


def default_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}.png"


def plot_probability_map(
    input_path: Path,
    output_path: Path,
    cmap: str,
    threshold: Optional[float],
    dpi: int,
) -> None:
    ds = xr.open_dataset(input_path)
    try:
        lon, lat, data = prepare_plot_grid(ds)
        target_time = ds.attrs.get("target_time_utc", input_path.stem)
        reduction = ds.attrs.get("reduction", "max over vertical levels")
    finally:
        ds.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    mesh = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, shrink=0.9)
    cbar.set_label("Column cloud probability")

    if threshold is not None:
        ax.contour(lon, lat, data, levels=[threshold], colors="black", linewidths=0.45, alpha=0.7)

    ax.set_title(f"Global column cloud probability | {target_time}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(color="0.7", linewidth=0.4, alpha=0.5)
    ax.text(
        0.01,
        0.015,
        str(reduction),
        transform=ax.transAxes,
        fontsize=9,
        color="0.2",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 3},
    )

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NetCDF file from build_global_column_cloud_probability.py")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--threshold", type=float, default=None, help="Optional contour level, e.g. 0.5")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or default_output_path(args.input, args.output_dir)
    plot_probability_map(
        input_path=args.input,
        output_path=output_path,
        cmap=args.cmap,
        threshold=args.threshold,
        dpi=args.dpi,
    )
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
