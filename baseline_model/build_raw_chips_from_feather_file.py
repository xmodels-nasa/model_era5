#!/usr/bin/env python3
"""Build raw ERA5 chips aligned to one ground-truth Feather file.

This is the first baseline step for end-to-end training from raw ERA5 fields
instead of Aurora embeddings. For each Feather row, the script:

1. Rounds the row timestamp to the same target hour used in the embedding flow.
2. Loads the ERA5 pair for `(target-6h, target)`.
3. Extracts a centered spatial chip around the nearest ERA5 grid point.
4. Saves dynamic inputs, static inputs, labels, and row metadata to one `.npz`.

Important: raw chips are much larger than point embeddings. Large chip sizes
and full-row extraction can easily create extremely large output files, so a
size guard is enabled by default.

At ERA5 0.25 degree resolution:
- 9x9 covers about 2.25 x 2.25 degrees, or about 250 x 250 km in latitude.
- 17x17 covers about 4.25 x 4.25 degrees, or about 470 x 470 km in latitude.
- 33x33 covers about 8.25 x 8.25 degrees, or about 915 x 915 km in latitude.
"""

import argparse
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "baseline_model" else SCRIPT_DIR


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
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


_load_dotenv()

DATA_ROOT = os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data_era5"))
FEATHER_FILE = os.getenv("FEATHER_FILE", "")
RAW_CHIPS_DIR = os.getenv("RAW_CHIPS_DIR", str(PROJECT_ROOT / "raw_chips"))

SURFACE_VARS = ("t2m", "u10", "v10", "msl")
STATIC_VARS = ("z", "lsm", "slt")
ATMOS_VARS = ("t", "u", "v", "q", "z")
TARGET_COLUMNS = [f"y_40dim_{i}" for i in range(40)]
REQUIRED_COLUMNS = ["timestamp_0", "Latitude_0", "Longitude_0"] + TARGET_COLUMNS
ERA5_GRID_RESOLUTION_DEG = 0.25
KM_PER_DEG_LAT = 111.0


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def _timestamp_unit_from_value(value: float) -> str:
    return "s" if len(str(abs(int(value)))) <= 10 else "ms"


def _build_target_column(df: pd.DataFrame) -> pd.Series:
    first_ts = float(df["timestamp_0"].iloc[0])
    unit = _timestamp_unit_from_value(first_ts)
    df_dt = pd.to_datetime(df["timestamp_0"], unit=unit, utc=True)
    hour_floor = df_dt.dt.floor("h")
    round_up = (df_dt - hour_floor) > pd.Timedelta(minutes=30)
    rounded_hour = hour_floor + pd.to_timedelta(round_up.astype(int), unit="h")
    return rounded_hour.dt.strftime("%Y_%m_%d_%H")


def parse_target(ts: str) -> datetime:
    normalized = ts.strip().replace("T", "-").replace("_", "-")
    parts = normalized.split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid target '{ts}'. Use YYYY_MM_DD_HH.")
    y, m, d, h = [int(x) for x in parts]
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def folder_name(dt: datetime) -> str:
    return dt.strftime("%Y_%m_%d_%H_data")


def resolve_dir(data_root: str, dt: datetime) -> Path:
    d = Path(data_root) / folder_name(dt)
    if not d.is_dir():
        raise FileNotFoundError(f"Missing directory: {d}")
    return d


def _normalize_lon_to_grid(lon: float, lon_vec: np.ndarray) -> float:
    lon_min = float(lon_vec.min())
    lon_max = float(lon_vec.max())
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    if lon_min < 0.0 and lon_max <= 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _ensure_lat_desc(array: np.ndarray, lat_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if np.all(np.diff(lat_vals) < 0):
        return array, np.array(lat_vals, copy=True)
    if np.all(np.diff(lat_vals) > 0):
        return np.flip(array, axis=-2), np.array(lat_vals[::-1], copy=True)
    raise ValueError("Latitude coordinate is not monotonic.")


def _surface_stack(ds: xr.Dataset) -> np.ndarray:
    parts = [np.asarray(ds[var].values[0], dtype=np.float32) for var in SURFACE_VARS]
    return np.stack(parts, axis=0)


def _atmos_stack(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    levels = np.asarray(ds["pressure_level"].values, dtype=np.int32)
    parts = [np.asarray(ds[var].values[0], dtype=np.float32) for var in ATMOS_VARS]
    flattened = [part for var_block in parts for part in var_block]
    return np.stack(flattened, axis=0), levels


def _static_stack(ds: xr.Dataset) -> np.ndarray:
    parts = [np.asarray(ds[var].values[0], dtype=np.float32) for var in STATIC_VARS]
    return np.stack(parts, axis=0)


def _dynamic_channel_names(levels: Sequence[int]) -> np.ndarray:
    names = [f"surface_{var}" for var in SURFACE_VARS]
    for var in ATMOS_VARS:
        for level in levels:
            names.append(f"atmos_{var}_{int(level)}")
    return np.asarray(names, dtype="<U32")


def load_era5_pair_as_tensors(data_root: str, target: str) -> Dict[str, np.ndarray]:
    target_dt = parse_target(target)
    prev_dt = target_dt - timedelta(hours=6)

    dir_prev = resolve_dir(data_root, prev_dt)
    dir_curr = resolve_dir(data_root, target_dt)

    with xr.open_dataset(dir_prev / "_surface.nc", engine="netcdf4") as surf_prev, \
        xr.open_dataset(dir_curr / "_surface.nc", engine="netcdf4") as surf_curr, \
        xr.open_dataset(dir_prev / "_atmospheric.nc", engine="netcdf4") as atmos_prev, \
        xr.open_dataset(dir_curr / "_atmospheric.nc", engine="netcdf4") as atmos_curr, \
        xr.open_dataset(dir_curr / "_static.nc", engine="netcdf4") as static_curr:
        lat_vals = np.asarray(surf_curr["latitude"].values)
        lon_vals = np.asarray(surf_curr["longitude"].values)

        surf_prev_arr, lat_desc = _ensure_lat_desc(_surface_stack(surf_prev), lat_vals)
        surf_curr_arr, _ = _ensure_lat_desc(_surface_stack(surf_curr), lat_vals)

        atmos_prev_arr, levels = _atmos_stack(atmos_prev)
        atmos_prev_arr, _ = _ensure_lat_desc(atmos_prev_arr, lat_vals)
        atmos_curr_arr, _ = _ensure_lat_desc(_atmos_stack(atmos_curr)[0], lat_vals)

        static_arr, _ = _ensure_lat_desc(_static_stack(static_curr), lat_vals)

    dynamic_prev = np.concatenate([surf_prev_arr, atmos_prev_arr], axis=0)
    dynamic_curr = np.concatenate([surf_curr_arr, atmos_curr_arr], axis=0)
    dynamic = np.stack([dynamic_prev, dynamic_curr], axis=0)

    return {
        "dynamic": dynamic,
        "static": static_arr,
        "lat": lat_desc.astype(np.float32, copy=False),
        "lon": np.asarray(lon_vals, dtype=np.float32),
        "pressure_levels": levels.astype(np.int32, copy=False),
        "dynamic_channel_names": _dynamic_channel_names(levels),
        "static_channel_names": np.asarray([f"static_{var}" for var in STATIC_VARS], dtype="<U16"),
        "input_times_unix_s": np.asarray(
            [int(prev_dt.timestamp()), int(target_dt.timestamp())],
            dtype=np.int64,
        ),
    }


def _nearest_grid_indices(lat: float, lon: float, lat_vec: np.ndarray, lon_vec: np.ndarray) -> Tuple[int, int, float]:
    lat_idx = int(np.abs(lat_vec - float(lat)).argmin())
    lon_norm = _normalize_lon_to_grid(float(lon), lon_vec)
    lon_span = float(lon_vec.max() - lon_vec.min())
    if lon_span > 300.0:
        lon_dist = np.abs(((lon_vec - lon_norm + 180.0) % 360.0) - 180.0)
    else:
        lon_dist = np.abs(lon_vec - lon_norm)
    lon_idx = int(lon_dist.argmin())
    return lat_idx, lon_idx, lon_norm


def _extract_chip_3d(array: np.ndarray, lat_idx: int, lon_idx: int, chip_size: int) -> np.ndarray:
    rows = np.clip(
        np.arange(lat_idx - chip_size // 2, lat_idx - chip_size // 2 + chip_size),
        0,
        array.shape[1] - 1,
    )
    cols = np.mod(
        np.arange(lon_idx - chip_size // 2, lon_idx - chip_size // 2 + chip_size),
        array.shape[2],
    )
    return array[:, rows[:, None], cols[None, :]]


def _estimate_output_bytes(
    row_count: int,
    chip_size: int,
    dynamic_channels: int,
    static_channels: int,
    output_dtype: np.dtype,
) -> int:
    bytes_per_value = np.dtype(output_dtype).itemsize
    chip_values_per_row = ((2 * dynamic_channels) + static_channels) * chip_size * chip_size
    label_values_per_row = len(TARGET_COLUMNS)
    return row_count * ((chip_values_per_row * bytes_per_value) + label_values_per_row)


def _chip_coverage_note(chip_size: int) -> str:
    span_deg = chip_size * ERA5_GRID_RESOLUTION_DEG
    span_km = span_deg * KM_PER_DEG_LAT
    return (
        f"{chip_size}x{chip_size} cells ~= {span_deg:.2f} x {span_deg:.2f} degrees "
        f"(~{span_km:.0f} x {span_km:.0f} km in latitude)"
    )


def process_feather_to_raw_chips(
    feather_path: Path,
    output_dir: Path,
    data_root: str,
    chip_size: int,
    sample_ratio: float,
    random_state: int,
    max_rows: Optional[int],
    output_dtype: str,
    max_estimated_gb: float,
    overwrite: bool,
) -> Dict[str, object]:
    if chip_size <= 0:
        raise ValueError("chip_size must be positive.")

    output_dtype_np = np.float16 if output_dtype == "float16" else np.float32
    output_file = output_dir / f"{_safe_stem(feather_path)}.npz"
    if output_file.exists() and not overwrite:
        return {
            "file": str(feather_path),
            "output_file": str(output_file),
            "rows_total": -1,
            "rows_written": -1,
            "rows_failed": 0,
            "status": "skipped_existing",
        }

    df = pd.read_feather(feather_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {feather_path.name}: {missing}")
    if len(df) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_file,
            dynamic_chips=np.empty((0, 2, 0, chip_size, chip_size), dtype=output_dtype_np),
            static_chips=np.empty((0, 0, chip_size, chip_size), dtype=output_dtype_np),
            labels=np.empty((0, len(TARGET_COLUMNS)), dtype=np.uint8),
            row_indices=np.empty((0,), dtype=np.int64),
        )
        return {
            "file": str(feather_path),
            "output_file": str(output_file),
            "rows_total": 0,
            "rows_written": 0,
            "rows_failed": 0,
            "status": "empty",
        }

    df = df.reset_index().rename(columns={"index": "source_row"})
    df["target"] = _build_target_column(df)

    if not (0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")
    if sample_ratio < 1.0:
        n = max(1, int(len(df) * sample_ratio))
        df = df.sample(n=n, random_state=random_state).copy()
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).copy()
    df = df.sort_values("source_row").reset_index(drop=True)

    preview: Optional[Dict[str, np.ndarray]] = None
    preview_target: Optional[str] = None
    for candidate_target in df["target"].drop_duplicates().tolist():
        try:
            preview = load_era5_pair_as_tensors(data_root=data_root, target=str(candidate_target))
            preview_target = str(candidate_target)
            break
        except FileNotFoundError:
            continue
    if preview is None or preview_target is None:
        raise FileNotFoundError(
            f"No available ERA5 (t-6h, t) input pairs were found for any target hour referenced by {feather_path.name}."
        )

    estimated_bytes = _estimate_output_bytes(
        row_count=len(df),
        chip_size=chip_size,
        dynamic_channels=int(preview["dynamic"].shape[1]),
        static_channels=int(preview["static"].shape[0]),
        output_dtype=np.dtype(output_dtype_np),
    )
    estimated_gb = estimated_bytes / (1024 ** 3)
    if estimated_gb > max_estimated_gb:
        raise ValueError(
            f"Estimated output for {feather_path.name} is {estimated_gb:.2f} GiB, "
            f"which exceeds --max-estimated-gb={max_estimated_gb:.2f}. "
            f"Reduce --chip-size, reduce rows via --sample-ratio/--max-rows, or raise the guard explicitly."
        )

    dynamic_chips: List[np.ndarray] = []
    static_chips: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    row_indices: List[int] = []
    latitudes: List[float] = []
    longitudes: List[float] = []
    matched_latitudes: List[float] = []
    matched_longitudes: List[float] = []
    timestamps: List[float] = []
    target_hours: List[str] = []
    input_times_unix_s: List[np.ndarray] = []
    rows_failed = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("target", sort=True)
    context_cache: Dict[str, Dict[str, np.ndarray]] = {preview_target: preview}
    for target, group in grouped:
        if target in context_cache:
            context = context_cache[target]
        else:
            try:
                context = load_era5_pair_as_tensors(data_root=data_root, target=target)
            except FileNotFoundError as exc:
                rows_failed += int(len(group))
                print(
                    f"[WARN] Skipping target={target} for {feather_path.name}: "
                    f"{len(group)} row(s) skipped because ERA5 inputs are missing. {exc}"
                )
                continue
            context_cache[target] = context

        lat_vec = context["lat"]
        lon_vec = context["lon"]
        dynamic = context["dynamic"]
        static = context["static"]
        for _, row in group.iterrows():
            lat_idx, lon_idx, lon_norm = _nearest_grid_indices(
                lat=float(row["Latitude_0"]),
                lon=float(row["Longitude_0"]),
                lat_vec=lat_vec,
                lon_vec=lon_vec,
            )
            dynamic_chip = np.stack(
                [
                    _extract_chip_3d(dynamic[time_idx], lat_idx=lat_idx, lon_idx=lon_idx, chip_size=chip_size)
                    for time_idx in range(dynamic.shape[0])
                ],
                axis=0,
            )
            static_chip = _extract_chip_3d(static, lat_idx=lat_idx, lon_idx=lon_idx, chip_size=chip_size)

            dynamic_chips.append(dynamic_chip.astype(output_dtype_np, copy=False))
            static_chips.append(static_chip.astype(output_dtype_np, copy=False))
            labels.append((row[TARGET_COLUMNS].to_numpy(dtype=np.float32) > 0.5).astype(np.uint8))
            row_indices.append(int(row["source_row"]))
            latitudes.append(float(row["Latitude_0"]))
            longitudes.append(float(row["Longitude_0"]))
            matched_latitudes.append(float(lat_vec[lat_idx]))
            matched_longitudes.append(float(lon_vec[lon_idx]))
            timestamps.append(float(row["timestamp_0"]))
            target_hours.append(str(target))
            input_times_unix_s.append(context["input_times_unix_s"])

    if dynamic_chips:
        np.savez_compressed(
            output_file,
            dynamic_chips=np.stack(dynamic_chips, axis=0),
            static_chips=np.stack(static_chips, axis=0),
            labels=np.stack(labels, axis=0),
            row_indices=np.asarray(row_indices, dtype=np.int64),
            latitudes=np.asarray(latitudes, dtype=np.float32),
            longitudes=np.asarray(longitudes, dtype=np.float32),
            matched_latitudes=np.asarray(matched_latitudes, dtype=np.float32),
            matched_longitudes=np.asarray(matched_longitudes, dtype=np.float32),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            target_hours=np.asarray(target_hours, dtype="<U16"),
            input_times_unix_s=np.stack(input_times_unix_s, axis=0).astype(np.int64, copy=False),
            pressure_levels=preview["pressure_levels"],
            dynamic_channel_names=preview["dynamic_channel_names"],
            static_channel_names=preview["static_channel_names"],
            chip_size=np.asarray(chip_size, dtype=np.int32),
            source_file=np.asarray(str(feather_path)),
        )
    else:
        np.savez_compressed(
            output_file,
            dynamic_chips=np.empty((0, 2, preview["dynamic"].shape[1], chip_size, chip_size), dtype=output_dtype_np),
            static_chips=np.empty((0, preview["static"].shape[0], chip_size, chip_size), dtype=output_dtype_np),
            labels=np.empty((0, len(TARGET_COLUMNS)), dtype=np.uint8),
            row_indices=np.empty((0,), dtype=np.int64),
            latitudes=np.empty((0,), dtype=np.float32),
            longitudes=np.empty((0,), dtype=np.float32),
            matched_latitudes=np.empty((0,), dtype=np.float32),
            matched_longitudes=np.empty((0,), dtype=np.float32),
            timestamps=np.empty((0,), dtype=np.float64),
            target_hours=np.empty((0,), dtype="<U16"),
            input_times_unix_s=np.empty((0, 2), dtype=np.int64),
            pressure_levels=preview["pressure_levels"],
            dynamic_channel_names=preview["dynamic_channel_names"],
            static_channel_names=preview["static_channel_names"],
            chip_size=np.asarray(chip_size, dtype=np.int32),
            source_file=np.asarray(str(feather_path)),
        )

    return {
        "file": str(feather_path),
        "output_file": str(output_file),
        "rows_total": int(len(pd.read_feather(feather_path, columns=["timestamp_0"]))),
        "rows_written": int(len(dynamic_chips)),
        "rows_failed": int(rows_failed),
        "status": "ok",
        "estimated_gb": float(estimated_gb),
    }


def main() -> Dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feather-file", type=str, default=FEATHER_FILE)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=RAW_CHIPS_DIR)
    parser.add_argument(
        "--chip-size",
        type=int,
        default=9,
        help=(
            "Spatial chip size in ERA5 grid cells. Default 9 means about 2.25 x 2.25 degrees "
            "(about 250 x 250 km in latitude). Keep this small initially."
        ),
    )
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional hard cap per Feather file. 0 means no cap.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dtype",
        type=str,
        choices=["float16", "float32"],
        default="float32",
        help="Use float32 by default because some raw ERA5 variables overflow float16.",
    )
    parser.add_argument(
        "--max-estimated-gb",
        type=float,
        default=4.0,
        help="Fail early if the estimated output for one Feather file exceeds this size.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.feather_file:
        raise ValueError("FEATHER_FILE is not set (or pass --feather-file).")

    print(f"Chip coverage: {_chip_coverage_note(args.chip_size)}")

    summary = process_feather_to_raw_chips(
        feather_path=Path(args.feather_file),
        output_dir=Path(args.output_dir),
        data_root=args.data_root,
        chip_size=args.chip_size,
        sample_ratio=args.sample_ratio,
        random_state=args.random_state,
        max_rows=(None if args.max_rows <= 0 else int(args.max_rows)),
        output_dtype=args.output_dtype,
        max_estimated_gb=args.max_estimated_gb,
        overwrite=args.overwrite,
    )
    print(
        f"[DONE] {Path(summary['file']).name}: "
        f"written={summary['rows_written']}, failed={summary['rows_failed']} | "
        f"estimated={summary.get('estimated_gb', 0.0):.2f} GiB | out={summary['output_file']}"
    )
    return summary


if __name__ == "__main__":
    main()
