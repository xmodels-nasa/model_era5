#!/usr/bin/env python3
"""Generate forecast Aurora embeddings for all Feather files under FEATHER_ROOT.

Outputs one `.npz` file per feather file (same stem), containing:
    - `emb_all_levels`: stacked forecast embeddings for successful target rows
    - `row_indices`: target feather row index for each embedding entry
    - `forecast_lead_hours`: scalar lead used to shift embedding context

For a target row at rounded hour t and default lead 6h, the embedding context
is built for t-6. Aurora therefore receives input timestamps (t-12, t-6), and
the trainer can still read labels directly from the target row t.
"""

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

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

from build_aurora_batches import DATA_ROOT  # noqa: E402
from batch_buid_forecast_emdding_from_feather_file import process_feather_batched_forecast  # noqa: E402


FEATHER_ROOT = os.getenv("FEATHER_ROOT", "")
FORECAST_EMBEDDING_OUTPUT_DIR = os.getenv(
    "FORECAST_EMBEDDING_OUTPUT_DIR",
    os.getenv("EMBEDDING_FORECAST_OUTPUT_DIR", str(PROJECT_ROOT / "embeddings_forecast")),
)
FORECAST_LEAD_HOURS = int(os.getenv("FORECAST_LEAD_HOURS", "6"))
MAX_FILE_WORKERS = int(os.getenv("FORECAST_EMBEDDING_MAX_FILE_WORKERS", "10"))


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def _write_empty(output_file: Path, forecast_lead_hours: int) -> None:
    np.savez_compressed(
        output_file,
        emb_all_levels=np.empty((0,), dtype=np.float32),
        row_indices=np.empty((0,), dtype=np.int64),
        forecast_lead_hours=np.asarray(forecast_lead_hours, dtype=np.int64),
    )


def process_one_feather_file(
    feather_file: str,
    output_dir: str,
    data_root: str,
    forecast_lead_hours: int,
) -> Dict[str, object]:
    path = Path(feather_file)
    stem = _safe_stem(path)
    output_file = Path(output_dir) / f"{stem}.npz"
    rows_total = len(pd.read_feather(path))
    if rows_total == 0:
        _write_empty(output_file, forecast_lead_hours)
        return {
            "file": str(path),
            "output_file": str(output_file),
            "rows_total": 0,
            "rows_written": 0,
            "rows_failed": 0,
        }

    embeddings = process_feather_batched_forecast(
        file_path=str(path),
        data_root=data_root,
        sample_ratio=1.0,
        forecast_lead_hours=forecast_lead_hours,
    )

    rows_written = 0
    row_indices: List[int] = []
    emb_arrays: List[np.ndarray] = []
    target_hours: List[str] = []
    forecast_source_hours: List[str] = []
    for emb in embeddings:
        arr = emb["emb_all_levels"].detach().cpu().numpy()
        emb_arrays.append(arr)
        row_indices.append(int(emb["source_row"]))
        target_hours.append(str(emb.get("target_hour", "")))
        forecast_source_hours.append(str(emb.get("forecast_source_hour", "")))
        rows_written += 1
    rows_failed = int(rows_total - rows_written)

    if emb_arrays:
        emb_stacked = np.stack(emb_arrays, axis=0)
        np.savez_compressed(
            output_file,
            emb_all_levels=emb_stacked,
            row_indices=np.asarray(row_indices, dtype=np.int64),
            target_hours=np.asarray(target_hours),
            forecast_source_hours=np.asarray(forecast_source_hours),
            forecast_lead_hours=np.asarray(forecast_lead_hours, dtype=np.int64),
        )
    else:
        _write_empty(output_file, forecast_lead_hours)

    return {
        "file": str(path),
        "output_file": str(output_file),
        "rows_total": int(rows_total),
        "rows_written": int(rows_written),
        "rows_failed": int(rows_failed),
    }


def main() -> List[Dict[str, object]]:
    if not FEATHER_ROOT:
        raise ValueError("FEATHER_ROOT is not set. Add FEATHER_ROOT to .env.")
    if FORECAST_LEAD_HOURS <= 0:
        raise ValueError("FORECAST_LEAD_HOURS must be > 0.")

    root = Path(FEATHER_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"FEATHER_ROOT does not exist: {root}")

    feather_files = sorted(str(p) for p in root.rglob("*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No .feather files found under FEATHER_ROOT: {root}")

    out_dir = Path(FORECAST_EMBEDDING_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(feather_files)} feather file(s)")
    print(f"Writing forecast embeddings to: {out_dir}")
    print(f"Forecast lead hours: {FORECAST_LEAD_HOURS}")
    print(f"Parallel file workers: {MAX_FILE_WORKERS}")

    summaries: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=MAX_FILE_WORKERS) as pool:
        futures = [
            pool.submit(process_one_feather_file, fp, str(out_dir), DATA_ROOT, FORECAST_LEAD_HOURS)
            for fp in feather_files
        ]
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
