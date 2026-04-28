#!/usr/bin/env python3
"""Generate 3x3 Aurora embedding grids for all rows in all Feather files under FEATHER_ROOT.

For each source row, this script:
1. Builds/reuses the Aurora encoder context for the row's rounded target hour.
2. Finds the nearest ERA5 gridpoint to the row latitude/longitude.
3. Extracts embeddings on a 3x3 grid centered on that nearest gridpoint.

Outputs one `.npz` file per feather file (same stem), containing:
    - `emb_3x3_all_levels`: stacked 3x3 embeddings for successful rows
      with shape [rows, 3, 3, ...], where the trailing embedding shape is
      whatever Aurora returns for one grid cell after the leading batch axis is removed
    - `row_indices`: source row index for each embedding entry
    - `grid_lats`: matched ERA5 latitudes for each 3x3 neighborhood
    - `grid_lons`: matched ERA5 longitudes for each 3x3 neighborhood
"""

import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "fine_tuned_model" else SCRIPT_DIR


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

from build_aurora_batches import (
    DATA_ROOT,
    get_embedding_from_encoder_context,
    get_encoder_context_for_target,
    load_aurora_model,
)

FEATHER_ROOT = os.getenv("FEATHER_ROOT", "")
EMBEDDING_3X3_OUTPUT_DIR = os.getenv(
    "EMBEDDING_3X3_OUTPUT_DIR",
    str(PROJECT_ROOT / "embeddings_3x3"),
)
MAX_FILE_WORKERS = 10
GRID_SIZE = 3
DEFAULT_TEST_START_TIME = "2019-07-01T00:00:00Z"
TRAIN_FILE_SAMPLE_COUNT = 100
TEST_FILE_SAMPLE_COUNT = 10
FILE_SAMPLE_SEED = 42


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def _timestamp_unit_from_value(v: int) -> str:
    return "s" if len(str(abs(int(v)))) <= 10 else "ms"


def _file_time(feather_path: Path) -> Optional[pd.Timestamp]:
    df_ts = pd.read_feather(feather_path, columns=["timestamp_0"])
    if len(df_ts) == 0:
        return None
    first = int(df_ts["timestamp_0"].iloc[0])
    unit = _timestamp_unit_from_value(first)
    ts = pd.to_datetime(df_ts["timestamp_0"], unit=unit, utc=True)
    return pd.Timestamp(ts.max())


def _parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _build_target_column(df: pd.DataFrame) -> pd.Series:
    first_ts = int(df["timestamp_0"].iloc[0])
    unit = _timestamp_unit_from_value(first_ts)
    df_dt = pd.to_datetime(df["timestamp_0"], unit=unit, utc=True)
    hour_floor = df_dt.dt.floor("h")
    round_up = (df_dt - hour_floor) > pd.Timedelta(minutes=30)
    rounded_hour = hour_floor + pd.to_timedelta(round_up.astype(int), unit="h")
    return rounded_hour.dt.strftime("%Y_%m_%d_%H")


def _normalize_lon_to_grid(lon: float, lon_vec: np.ndarray) -> float:
    lon_min = float(lon_vec.min())
    lon_max = float(lon_vec.max())
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    if lon_min < 0.0 and lon_max <= 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _nearest_grid_indices(lat_vec: np.ndarray, lon_vec: np.ndarray, lat: float, lon: float) -> Tuple[int, int]:
    lat_idx = int(np.abs(lat_vec - float(lat)).argmin())
    lon_norm = _normalize_lon_to_grid(float(lon), lon_vec)
    lon_span = float(lon_vec.max() - lon_vec.min())
    if lon_span > 300.0:
        lon_dist = np.abs(((lon_vec - lon_norm + 180.0) % 360.0) - 180.0)
    else:
        lon_dist = np.abs(lon_vec - lon_norm)
    lon_idx = int(lon_dist.argmin())
    return lat_idx, lon_idx


def _window_indices(center: int, size: int, length: int, wrap: bool = False) -> np.ndarray:
    half = size // 2
    if wrap:
        return (np.arange(center - half, center + half + 1) % length).astype(np.int64)

    start = center - half
    end = center + half + 1
    if start < 0:
        end = min(length, end - start)
        start = 0
    if end > length:
        start = max(0, start - (end - length))
        end = length
    return np.arange(start, end, dtype=np.int64)


def _extract_3x3_embedding_grid(
    encoder_context: Dict[str, object],
    lat: float,
    lon: float,
    grid_size: int = GRID_SIZE,
) -> Dict[str, np.ndarray]:
    enc_batch = encoder_context["enc_batch"]
    lat_vec = enc_batch.metadata.lat.detach().cpu().numpy()
    lon_vec = enc_batch.metadata.lon.detach().cpu().numpy()
    center_lat_idx, center_lon_idx = _nearest_grid_indices(lat_vec=lat_vec, lon_vec=lon_vec, lat=lat, lon=lon)
    lon_wrap = float(lon_vec.max() - lon_vec.min()) > 300.0
    lat_indices = _window_indices(center=center_lat_idx, size=grid_size, length=len(lat_vec), wrap=False)
    lon_indices = _window_indices(center=center_lon_idx, size=grid_size, length=len(lon_vec), wrap=lon_wrap)
    if len(lat_indices) != grid_size or len(lon_indices) != grid_size:
        raise ValueError(
            f"Could not build a full {grid_size}x{grid_size} grid around "
            f"lat={lat}, lon={lon}."
        )

    grid_embeddings: List[List[np.ndarray]] = []
    grid_lats = np.empty((grid_size, grid_size), dtype=np.float32)
    grid_lons = np.empty((grid_size, grid_size), dtype=np.float32)
    for i, lat_idx in enumerate(lat_indices):
        row_embeddings: List[np.ndarray] = []
        grid_lat = float(lat_vec[int(lat_idx)])
        for j, lon_idx in enumerate(lon_indices):
            grid_lon = float(lon_vec[int(lon_idx)])
            emb = get_embedding_from_encoder_context(
                encoder_context=encoder_context,
                lat=grid_lat,
                lon=grid_lon,
            )["emb_all_levels"].detach().cpu().numpy()
            # Aurora may return more than one latent axis here, e.g. [1, 2, 4, 512].
            # Strip only the leading batch axis and preserve the full per-cell embedding shape.
            if emb.shape[0] == 1:
                emb = emb[0]
            row_embeddings.append(emb.astype(np.float32, copy=False))
            grid_lats[i, j] = grid_lat
            grid_lons[i, j] = grid_lon
        grid_embeddings.append(row_embeddings)

    return {
        "emb_3x3_all_levels": np.stack([np.stack(row, axis=0) for row in grid_embeddings], axis=0),
        "grid_lats": grid_lats,
        "grid_lons": grid_lons,
    }


def process_feather_batched_3x3(file_path: str, data_root: Optional[str] = None) -> List[Dict[str, object]]:
    df = pd.read_feather(file_path)
    required = ["Latitude_0", "Longitude_0", "timestamp_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feather file: {missing}")
    if len(df) == 0:
        return []

    df = df.reset_index().rename(columns={"index": "source_row"})
    df["target"] = _build_target_column(df)

    root = data_root or DATA_ROOT
    model = load_aurora_model()
    results: List[Dict[str, object]] = []

    unique_targets = list(df["target"].unique())
    print(f"Processing {len(df)} row(s) across {len(unique_targets)} unique target hour(s)")
    for target in unique_targets:
        group = df[df["target"] == target]
        print(f"Building shared encoder context for target={target} ({len(group)} row(s))")
        try:
            encoder_context = get_encoder_context_for_target(
                data_root=root,
                target=target,
                model=model,
            )
        except FileNotFoundError as e:
            print(
                f"[WARN] Skipping target={target}: missing ERA5 input for this hour "
                f"({len(group)} row(s) skipped). Details: {e}"
            )
            continue

        for _, row in group.iterrows():
            lat = float(row["Latitude_0"])
            lon = float(row["Longitude_0"])
            grid_payload = _extract_3x3_embedding_grid(
                encoder_context=encoder_context,
                lat=lat,
                lon=lon,
                grid_size=GRID_SIZE,
            )
            results.append(
                {
                    "source_row": int(row["source_row"]),
                    "source_file": file_path,
                    "lat": lat,
                    "lon": lon,
                    **grid_payload,
                }
            )

    results.sort(key=lambda x: x["source_row"])
    return results


def process_one_feather_file(feather_file: str, output_dir: str, data_root: str) -> Dict[str, object]:
    path = Path(feather_file)
    stem = _safe_stem(path)
    output_file = Path(output_dir) / f"{stem}.npz"
    rows_total = len(pd.read_feather(path))
    if rows_total == 0:
        np.savez_compressed(
            output_file,
            emb_3x3_all_levels=np.empty((0,), dtype=np.float32),
            row_indices=np.empty((0,), dtype=np.int64),
            grid_lats=np.empty((0,), dtype=np.float32),
            grid_lons=np.empty((0,), dtype=np.float32),
        )
        return {
            "file": str(path),
            "output_file": str(output_file),
            "rows_total": 0,
            "rows_written": 0,
            "rows_failed": 0,
        }

    embeddings = process_feather_batched_3x3(
        file_path=str(path),
        data_root=data_root,
    )

    rows_written = 0
    row_indices: List[int] = []
    emb_arrays: List[np.ndarray] = []
    lat_arrays: List[np.ndarray] = []
    lon_arrays: List[np.ndarray] = []
    for emb in embeddings:
        emb_arrays.append(emb["emb_3x3_all_levels"])
        lat_arrays.append(emb["grid_lats"])
        lon_arrays.append(emb["grid_lons"])
        row_indices.append(int(emb["source_row"]))
        rows_written += 1
    rows_failed = int(rows_total - rows_written)

    if emb_arrays:
        np.savez_compressed(
            output_file,
            emb_3x3_all_levels=np.stack(emb_arrays, axis=0),
            row_indices=np.asarray(row_indices, dtype=np.int64),
            grid_lats=np.stack(lat_arrays, axis=0),
            grid_lons=np.stack(lon_arrays, axis=0),
        )
    else:
        np.savez_compressed(
            output_file,
            emb_3x3_all_levels=np.empty((0,), dtype=np.float32),
            row_indices=np.empty((0,), dtype=np.int64),
            grid_lats=np.empty((0,), dtype=np.float32),
            grid_lons=np.empty((0,), dtype=np.float32),
        )

    return {
        "file": str(path),
        "output_file": str(output_file),
        "rows_total": int(rows_total),
        "rows_written": int(rows_written),
        "rows_failed": int(rows_failed),
    }


def _select_sampled_feather_files(feather_files: List[Path]) -> List[Path]:
    cutoff = _parse_utc_timestamp(DEFAULT_TEST_START_TIME)
    rng = random.Random(FILE_SAMPLE_SEED)

    before_cutoff: List[Tuple[pd.Timestamp, Path]] = []
    after_cutoff: List[Tuple[pd.Timestamp, Path]] = []
    empty_files = 0

    for feather_path in feather_files:
        file_time = _file_time(feather_path)
        if file_time is None:
            empty_files += 1
            continue
        if file_time < cutoff:
            before_cutoff.append((file_time, feather_path))
        else:
            after_cutoff.append((file_time, feather_path))

    if len(before_cutoff) < TRAIN_FILE_SAMPLE_COUNT:
        raise ValueError(
            f"Need {TRAIN_FILE_SAMPLE_COUNT} feather files before {cutoff}, "
            f"but only found {len(before_cutoff)}."
        )
    if len(after_cutoff) < TEST_FILE_SAMPLE_COUNT:
        raise ValueError(
            f"Need {TEST_FILE_SAMPLE_COUNT} feather files on or after {cutoff}, "
            f"but only found {len(after_cutoff)}."
        )

    before_selected = rng.sample(before_cutoff, k=TRAIN_FILE_SAMPLE_COUNT)
    after_selected = rng.sample(after_cutoff, k=TEST_FILE_SAMPLE_COUNT)
    combined = before_selected + after_selected
    combined.sort(key=lambda item: (item[0], str(item[1])))

    print(f"Sampling feather files with cutoff {cutoff}")
    print(
        f"Available before cutoff: {len(before_cutoff)} | selected: {TRAIN_FILE_SAMPLE_COUNT}"
    )
    print(
        f"Available on/after cutoff: {len(after_cutoff)} | selected: {TEST_FILE_SAMPLE_COUNT}"
    )
    if empty_files:
        print(f"Skipped empty feather files during sampling: {empty_files}")

    return [path for _, path in combined]


def main() -> List[Dict[str, object]]:
    if not FEATHER_ROOT:
        raise ValueError("FEATHER_ROOT is not set. Add FEATHER_ROOT to .env.")

    root = Path(FEATHER_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"FEATHER_ROOT does not exist: {root}")

    feather_paths = sorted(root.rglob("*.feather"))
    feather_files = [str(p) for p in _select_sampled_feather_files(feather_paths)]
    if not feather_files:
        raise FileNotFoundError(f"No .feather files found under FEATHER_ROOT: {root}")

    out_dir = Path(EMBEDDING_3X3_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(feather_files)} feather file(s)")
    print(f"Writing 3x3 embeddings to: {out_dir}")
    print(f"Parallel file workers: {MAX_FILE_WORKERS}")

    summaries: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=MAX_FILE_WORKERS) as pool:
        futures = [pool.submit(process_one_feather_file, fp, str(out_dir), DATA_ROOT) for fp in feather_files]
        for fut in as_completed(futures):
            summary = fut.result()
            summaries.append(summary)
            print(
                f"[DONE] {Path(summary['file']).name}: "
                f"written={summary['rows_written']}/{summary['rows_total']}, "
                f"failed={summary['rows_failed']} | out={Path(summary['output_file']).name}"
            )

    total_files = len(summaries)
    total_written = sum(int(s["rows_written"]) for s in summaries)
    total_failed = sum(int(s["rows_failed"]) for s in summaries)
    print(f"Completed {total_files} file(s) | rows written={total_written}, failed={total_failed}")
    return summaries


if __name__ == "__main__":
    main()
