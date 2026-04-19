#!/usr/bin/env python3
"""Generate raw ERA5 chips for every Feather file under FEATHER_ROOT."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


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

from build_raw_chips_from_feather_file import DATA_ROOT, RAW_CHIPS_DIR, _chip_coverage_note, process_feather_to_raw_chips

FEATHER_ROOT = os.getenv("FEATHER_ROOT", "")


def main() -> List[Dict[str, object]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feather-root", type=str, default=FEATHER_ROOT)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default=RAW_CHIPS_DIR)
    parser.add_argument(
        "--chip-size",
        type=int,
        default=9,
        help="Default 9 means about 2.25 x 2.25 degrees, or about 250 x 250 km in latitude.",
    )
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dtype", type=str, choices=["float16", "float32"], default="float32")
    parser.add_argument("--max-estimated-gb", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-file-workers",
        type=int,
        default=5,
        help="Parallel file workers. Keep this low because each worker loads large ERA5 tensors.",
    )
    args = parser.parse_args()

    if not args.feather_root:
        raise ValueError("FEATHER_ROOT is not set. Add FEATHER_ROOT to .env or pass --feather-root.")

    root = Path(args.feather_root)
    if not root.exists():
        raise FileNotFoundError(f"FEATHER_ROOT does not exist: {root}")

    feather_files = sorted(root.rglob("*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No .feather files found under FEATHER_ROOT: {root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(feather_files)} feather file(s)")
    print(f"Writing raw chips to: {out_dir}")
    print(f"chip_size={args.chip_size} | sample_ratio={args.sample_ratio} | max_rows={args.max_rows}")
    print(f"Chip coverage: {_chip_coverage_note(args.chip_size)}")
    print(f"Parallel file workers: {args.max_file_workers}")

    summaries: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.max_file_workers)) as pool:
        futures = [
            pool.submit(
                process_feather_to_raw_chips,
                feather_path=feather_file,
                output_dir=out_dir,
                data_root=args.data_root,
                chip_size=args.chip_size,
                sample_ratio=args.sample_ratio,
                random_state=args.random_state,
                max_rows=(None if args.max_rows <= 0 else int(args.max_rows)),
                output_dtype=args.output_dtype,
                max_estimated_gb=args.max_estimated_gb,
                overwrite=args.overwrite,
            )
            for feather_file in feather_files
        ]

        for fut in as_completed(futures):
            summary = fut.result()
            summaries.append(summary)
            print(
                f"[DONE] {Path(summary['file']).name}: "
                f"status={summary['status']} | written={summary['rows_written']} | "
                f"failed={summary['rows_failed']} | out={Path(summary['output_file']).name}"
            )

    total_written = sum(max(0, int(s["rows_written"])) for s in summaries)
    total_failed = sum(max(0, int(s["rows_failed"])) for s in summaries)
    print(
        f"Completed {len(summaries)} file(s) | "
        f"rows written={total_written}, failed={total_failed}"
    )
    return summaries


if __name__ == "__main__":
    main()
