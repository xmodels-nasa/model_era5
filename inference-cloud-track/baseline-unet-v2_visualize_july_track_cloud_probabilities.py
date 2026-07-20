#!/usr/bin/env python3
"""Visualize July track baseline U-Net v2 cloud labels/probabilities as ordered curtains."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


DEFAULT_INPUT = THIS_DIR / "outputs" / "data" / "baseline-unet-v2_july_track_cloud_probabilities.nc"
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "visualization"


def ordered_sample_indices(file_index: np.ndarray, samples_per_file: int) -> np.ndarray:
    if samples_per_file <= 0:
        return np.arange(file_index.shape[0], dtype=np.int64)

    selected_parts: List[np.ndarray] = []
    for value in np.unique(file_index):
        indices = np.flatnonzero(file_index == value)
        if indices.size <= samples_per_file:
            selected_parts.append(indices)
            continue
        chosen_positions = np.linspace(0, indices.size - 1, samples_per_file).round().astype(np.int64)
        selected_parts.append(indices[np.unique(chosen_positions)])
    if not selected_parts:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(selected_parts).astype(np.int64)


def file_boundaries(file_index: np.ndarray) -> List[int]:
    if file_index.size == 0:
        return []
    boundaries = np.flatnonzero(file_index[1:] != file_index[:-1]) + 1
    return boundaries.astype(int).tolist()


def default_output_path(input_path: Path, output_dir: Path, samples_per_file: int) -> Path:
    suffix = "all" if samples_per_file <= 0 else f"{samples_per_file}_per_file"
    return output_dir / f"{input_path.stem}_curtain_{suffix}.png"


def plot_curtains(
    input_path: Path,
    output_path: Path,
    samples_per_file: int,
    dpi: int,
    cmap_probability: str,
) -> None:
    with xr.open_dataset(input_path) as ds:
        truth = ds["cloud_mask_label"].values.astype(np.float32)
        prob = ds["cloud_mask_prob"].values.astype(np.float32)
        pred = ds["predicted_label"].values.astype(np.float32)
        file_index = ds["source_file_index"].values.astype(np.int64)
        latitude = ds["latitude"].values.astype(np.float32)
        longitude = ds["longitude"].values.astype(np.float32)
        timestamp = ds["timestamp_utc"].values
        threshold = float(ds.attrs.get("threshold", 0.5))
        source_file_count = int(ds.attrs.get("source_file_count", len(np.unique(file_index))))
        split_selection = str(ds.attrs.get("split_selection", ds.attrs.get("split", "unknown")))

    sample_idx = ordered_sample_indices(file_index, samples_per_file)
    if sample_idx.size == 0:
        raise ValueError("No samples available for visualization.")

    truth_plot = truth[sample_idx].T
    prob_plot = prob[sample_idx].T
    pred_plot = pred[sample_idx].T
    sampled_file_index = file_index[sample_idx]
    boundaries = file_boundaries(sampled_file_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = min(28.0, max(14.0, sample_idx.size / 260.0))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(width, 10.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.15, 1.0]},
    )

    panels: List[Tuple[str, np.ndarray, str, float, float]] = [
        ("Ground truth label", truth_plot, "Greys", 0.0, 1.0),
        ("Predicted probability", prob_plot, cmap_probability, 0.0, 1.0),
        (f"Predicted label (threshold={threshold:g})", pred_plot, "Greys", 0.0, 1.0),
    ]

    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylabel("Mask level")
        ax.set_yticks([0, 10, 20, 30, 39])
        ax.set_title(title)
        for boundary in boundaries:
            ax.axvline(boundary - 0.5, color="tab:red", linewidth=0.35, alpha=0.35)
        cbar = fig.colorbar(image, ax=ax, pad=0.01, shrink=0.85)
        cbar.ax.tick_params(labelsize=8)

    start_time = np.datetime_as_string(timestamp[sample_idx[0]], unit="m")
    end_time = np.datetime_as_string(timestamp[sample_idx[-1]], unit="m")
    axes[-1].set_xlabel(
        f"Ordered sampled track points ({sample_idx.size:,}; {samples_per_file if samples_per_file > 0 else 'all'} per file)"
    )
    fig.suptitle(
        "CloudSat track cloud mask curtains - baseline U-Net v2\n"
        f"{start_time} to {end_time} UTC | files={source_file_count} | "
        f"split={split_selection} | "
        f"lat {latitude[sample_idx].min():.1f}..{latitude[sample_idx].max():.1f}, "
        f"lon {longitude[sample_idx].min():.1f}..{longitude[sample_idx].max():.1f}",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=120,
        help="Uniformly sample this many ordered rows per source Feather file. 0 means all rows.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--cmap-probability", default="viridis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_file < 0:
        raise ValueError("--samples-per-file must be non-negative.")
    output_path = args.output or default_output_path(args.input, args.output_dir, args.samples_per_file)
    plot_curtains(
        input_path=args.input,
        output_path=output_path,
        samples_per_file=args.samples_per_file,
        dpi=args.dpi,
        cmap_probability=args.cmap_probability,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
