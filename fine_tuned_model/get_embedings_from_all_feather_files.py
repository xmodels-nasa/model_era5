#!/usr/bin/env python3
"""Generate Aurora embeddings for all rows in all Feather files under FEATHER_ROOT.

Outputs one `.npz` file per feather file (same stem), containing:
    - `emb_all_levels`: stacked embeddings for successful rows
    - `row_indices`: source row index for each embedding entry
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

from build_aurora_batches import DATA_ROOT
from batch_buid_emdding_from_feather_file import process_feather_batched

FEATHER_ROOT = os.getenv("FEATHER_ROOT", "")
EMBEDDING_OUTPUT_DIR = os.getenv(
    "EMBEDDING_OUTUT_DIR", os.getenv("EMBEDDING_OUTPUT_DIR", str(PROJECT_ROOT / "embeddings"))
)
MAX_FILE_WORKERS = 10
def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def process_one_feather_file(feather_file: str, output_dir: str, data_root: str) -> Dict[str, object]:
    path = Path(feather_file)
    stem = _safe_stem(path)
    output_file = Path(output_dir) / f"{stem}.npz"
    rows_total = len(pd.read_feather(path))
    if rows_total == 0:
        np.savez_compressed(
            output_file,
            emb_all_levels=np.empty((0,), dtype=np.float32),
            row_indices=np.empty((0,), dtype=np.int64),
        )
        return {
            "file": str(path),
            "output_file": str(output_file),
            "rows_total": 0,
            "rows_written": 0,
            "rows_failed": 0,
        }

    # sample_ratio is intentionally 1.0: process every row using batched target-hour reuse.
    embeddings = process_feather_batched(
        file_path=str(path),
        data_root=data_root,
        sample_ratio=1.0,
    )

    rows_written = 0
    row_indices: List[int] = []
    emb_arrays: List[np.ndarray] = []
    for emb in embeddings:
        arr = emb["emb_all_levels"].detach().cpu().numpy()
        emb_arrays.append(arr)
        row_indices.append(int(emb["source_row"]))
        rows_written += 1
    rows_failed = int(rows_total - rows_written)

    if emb_arrays:
        # Keep row alignment via `row_indices` for lookup back to feather rows.
        emb_stacked = np.stack(emb_arrays, axis=0)
        np.savez_compressed(
            output_file,
            emb_all_levels=emb_stacked,
            row_indices=np.asarray(row_indices, dtype=np.int64),
        )
    else:
        np.savez_compressed(
            output_file,
            emb_all_levels=np.empty((0,), dtype=np.float32),
            row_indices=np.empty((0,), dtype=np.int64),
        )

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

    root = Path(FEATHER_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"FEATHER_ROOT does not exist: {root}")

    feather_files = sorted(str(p) for p in root.rglob("*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No .feather files found under FEATHER_ROOT: {root}")

    out_dir = Path(EMBEDDING_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(feather_files)} feather file(s)")
    print(f"Writing embeddings to: {out_dir}")
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
