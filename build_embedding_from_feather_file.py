#!/usr/bin/env python3
"""Utility for sampling rows from a Feather feature file and
building Aurora encoder embeddings for each sample.

The input Feather files are expected to contain at least the
following columns:
    ['Latitude_0', 'Longitude_0', 'timestamp_0']

Timestamps are interpreted as UTC seconds (or milliseconds if they
have more digits).  A string in the form ``YYYY_MM_DD_HH`` is
derived from the timestamp and is used as the ``target`` argument
for :func:`model_era5.build_aurora_batches.get_embeddings_for_target`.

By default 1% of the rows are sampled (random seed 42) so that a
subset of the file is processed.  The embeddings returned by
``get_embeddings_for_target`` are collected and dumped to a parquet
file alongside the metadata.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from build_aurora_batches import get_embeddings_for_target, DATA_ROOT

# configuration defaults
FEATHER_FILE = "/path/to/your.feather"  # override as needed
DATA_ROOT_OVERRIDE: Optional[str] = None  # if None, uses build_aurora_batches.DATA_ROOT
SAMPLE_RATIO = 0.01
RANDOM_STATE = 42


def inspect_coordinate_conventions(file_path: str, data_root: str) -> None:
    """Print Feather/ERA5 latitude-longitude conventions for quick validation."""
    sample_df = pd.read_feather(file_path, columns=["Latitude_0", "Longitude_0"])
    feather_lat = sample_df["Latitude_0"].astype(float)
    feather_lon = sample_df["Longitude_0"].astype(float)

    first_era5_dir = sorted(d for d in os.listdir(data_root) if d.endswith("_data"))[0]
    era5_surface = os.path.join(data_root, first_era5_dir, "_surface.nc")
    ds = xr.open_dataset(era5_surface, engine="netcdf4")
    era5_lat = ds.latitude.values
    era5_lon = ds.longitude.values

    print(
        "Feather convention: "
        f"lat[{feather_lat.min():.4f}, {feather_lat.max():.4f}], "
        f"lon[{feather_lon.min():.4f}, {feather_lon.max():.4f}]"
    )
    print(
        "ERA5 convention: "
        f"lat[{float(era5_lat.min()):.4f}, {float(era5_lat.max()):.4f}] "
        f"(stored {'decreasing' if np.all(np.diff(era5_lat) < 0) else 'non-decreasing'}), "
        f"lon[{float(era5_lon.min()):.4f}, {float(era5_lon.max()):.4f}]"
    )
    if feather_lon.min() < 0 and era5_lon.min() >= 0:
        print("Longitude mapping: Feather uses [-180, 180], ERA5 uses [0, 360). Conversion is applied internally.")
    ds.close()


def process_feather(
    file_path: str,
    data_root: Optional[str] = None,
    sample_ratio: float = 0.01,
    random_state: int = 42,
) -> List[Dict[str, object]]:
    """Read ``file_path`` and compute embeddings for a small sample.

    ``data_root`` is passed directly to ``get_embeddings_for_target``;
    if ``None`` the default DATA_ROOT from
    ``build_aurora_batches`` is used.
    """

    df = pd.read_feather(file_path)
    required = ["Latitude_0", "Longitude_0", "timestamp_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feather file: {missing}")

    # figure out units based on digit count of first value
    first_ts = int(df["timestamp_0"].iloc[0])
    unit = "s" if len(str(abs(first_ts))) <= 10 else "ms"
    df["datetime"] = pd.to_datetime(df["timestamp_0"], unit=unit, utc=True)
    hour_floor = df["datetime"].dt.floor("h")
    round_up = (df["datetime"] - hour_floor) > pd.Timedelta(minutes=30)
    rounded_hour = hour_floor + pd.to_timedelta(round_up.astype(int), unit="h")
    df["target"] = rounded_hour.dt.strftime("%Y_%m_%d_%H")

    n = max(1, int(len(df) * sample_ratio))
    sampled = df.sample(n=n, random_state=random_state)

    data_root = data_root or DATA_ROOT
    inspect_coordinate_conventions(file_path, data_root)
    results: List[Dict[str, object]] = []
    # create model once for speed
    model = None
    for _, row in sampled.iterrows():
        print("-" * 40)
        print(f"Datetime input: {row['datetime']} -> target: {row['target']}")
        print(f"Processing target {row['target']} at lat {row['Latitude_0']}, lon {row['Longitude_0']}")
        emb = get_embeddings_for_target(
            data_root=data_root,
            target=row["target"],
            lat=float(row["Latitude_0"]),
            lon=float(row["Longitude_0"]),
            model=model,
        )
        results.append(emb)
        # keep the same model if returned; get_embeddings_for_target accepts
        # an existing model instance and will reuse it, so we don't repeatedly
        # re-load the checkpoint.
        if model is None:
            model = emb.get("_model") if "_model" in emb else None
    return results


def main(
    feather_path: Optional[str] = None,
    data_root: Optional[str] = None,
    sample_ratio: float = SAMPLE_RATIO,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, object]]:
    """Convenience entry point.

    If ``feather_path`` is not provided the module-level ``FEATHER_FILE``
    constant will be used.  ``data_root`` defaults to
    ``DATA_ROOT_OVERRIDE`` or the original ``DATA_ROOT``.

    Returns the list of embedding dictionaries so callers can further
    inspect or save them as desired.  No side effects (such as writing
    files) occur.
    """

    path = feather_path or FEATHER_FILE
    root = data_root or DATA_ROOT_OVERRIDE or DATA_ROOT
    embeddings = process_feather(path, data_root=root, sample_ratio=sample_ratio, random_state=random_state)
    print(f"Computed embeddings for {len(embeddings)} sample(s)")
    return embeddings
