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


def default_globe_output_path(input_path: Path, output_dir: Path, center_lon: float, center_lat: float) -> Path:
    lon_label = f"{center_lon:g}".replace("-", "m").replace(".", "p")
    lat_label = f"{center_lat:g}".replace("-", "m").replace(".", "p")
    return output_dir / f"{input_path.stem}_globe_lon{lon_label}_lat{lat_label}.png"


def coordinate_edges(values: np.ndarray, clamp: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two coordinate values to build cell edges.")
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    edges = np.concatenate([[first], mids, [last]]).astype(np.float32)
    if clamp is not None:
        edges = np.clip(edges, clamp[0], clamp[1])
    return edges


def orthographic_project(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    center_lon: float,
    center_lat: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    lon0 = np.deg2rad(center_lon)
    lat0 = np.deg2rad(center_lat)
    dlon = lon - lon0

    cosc = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dlon)
    x = np.cos(lat) * np.sin(dlon)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    return x, y, cosc


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


def plot_probability_globe(
    input_path: Path,
    output_path: Path,
    cmap: str,
    threshold: Optional[float],
    dpi: int,
    center_lon: float,
    center_lat: float,
) -> None:
    ds = xr.open_dataset(input_path)
    try:
        lon, lat, data = prepare_plot_grid(ds)
        target_time = ds.attrs.get("target_time_utc", input_path.stem)
        reduction = ds.attrs.get("reduction", "max over vertical levels")
    finally:
        ds.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lon_edges = coordinate_edges(lon)
    lat_edges = coordinate_edges(lat, clamp=(-90.0, 90.0))
    lon_edge_grid, lat_edge_grid = np.meshgrid(lon_edges, lat_edges)
    x_edge, y_edge, _ = orthographic_project(lon_edge_grid, lat_edge_grid, center_lon, center_lat)

    lon_center_grid, lat_center_grid = np.meshgrid(lon, lat)
    _, _, visible = orthographic_project(lon_center_grid, lat_center_grid, center_lon, center_lat)
    data_visible = np.ma.masked_where(visible < 0.0, data)

    fig, ax = plt.subplots(figsize=(8.4, 8.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        x_edge,
        y_edge,
        data_visible,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.78)
    cbar.set_label("Column cloud probability")

    # Globe limb.
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    ax.plot(np.cos(theta), np.sin(theta), color="0.15", linewidth=1.0)

    # Graticule.
    for meridian in np.arange(-180, 181, 30):
        lat_line = np.linspace(-90, 90, 361)
        lon_line = np.full_like(lat_line, float(meridian))
        x, y, cosc = orthographic_project(lon_line, lat_line, center_lon, center_lat)
        x = np.ma.masked_where(cosc < 0, x)
        y = np.ma.masked_where(cosc < 0, y)
        ax.plot(x, y, color="0.55", linewidth=0.35, alpha=0.55)
    for parallel in np.arange(-60, 61, 30):
        lon_line = np.linspace(-180, 180, 721)
        lat_line = np.full_like(lon_line, float(parallel))
        x, y, cosc = orthographic_project(lon_line, lat_line, center_lon, center_lat)
        x = np.ma.masked_where(cosc < 0, x)
        y = np.ma.masked_where(cosc < 0, y)
        ax.plot(x, y, color="0.55", linewidth=0.35, alpha=0.55)

    if threshold is not None:
        x_center, y_center, _ = orthographic_project(lon_center_grid, lat_center_grid, center_lon, center_lat)
        ax.contour(
            x_center,
            y_center,
            data_visible,
            levels=[threshold],
            colors="black",
            linewidths=0.5,
            alpha=0.75,
        )

    ax.set_title(
        f"Column cloud probability | {target_time}\n"
        f"Orthographic view centered at lon={center_lon:g}, lat={center_lat:g}"
    )
    ax.text(
        -0.94,
        -0.94,
        str(reduction),
        fontsize=8.5,
        color="0.2",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 3},
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.axis("off")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NetCDF file from build_global_column_cloud_probability.py")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--projection", choices=["flat", "globe"], default="flat")
    parser.add_argument("--center-lon", type=float, default=0.0, help="Center longitude for --projection globe.")
    parser.add_argument("--center-lat", type=float, default=0.0, help="Center latitude for --projection globe.")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--threshold", type=float, default=None, help="Optional contour level, e.g. 0.5")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is not None:
        output_path = args.output
    elif args.projection == "globe":
        output_path = default_globe_output_path(args.input, args.output_dir, args.center_lon, args.center_lat)
    else:
        output_path = default_output_path(args.input, args.output_dir)

    if args.projection == "globe":
        plot_probability_globe(
            input_path=args.input,
            output_path=output_path,
            cmap=args.cmap,
            threshold=args.threshold,
            dpi=args.dpi,
            center_lon=args.center_lon,
            center_lat=args.center_lat,
        )
    else:
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
