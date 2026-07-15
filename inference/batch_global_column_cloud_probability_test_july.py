#!/usr/bin/env python3
"""Run global transformer inference for July test-split target hours.

This is the cloud-oriented batch driver. It reads the saved split manifest,
selects test rows whose rounded target hour falls in the requested July window,
checks that the ERA5 t-6 and t folders exist, then writes NetCDF and 2D flat PNG
outputs for every unique target hour.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "test_july_global_column_cloud_probability"

import build_global_column_cloud_probability as global_infer  # noqa: E402
import visualize_global_column_cloud_probability as global_vis  # noqa: E402


def read_dotenv(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def default_data_root() -> Path:
    env_value = os.getenv("DATA_ROOT")
    if env_value:
        return Path(env_value)
    dotenv_value = read_dotenv(PROJECT_ROOT / ".env").get("DATA_ROOT")
    if dotenv_value:
        return Path(dotenv_value)
    return PROJECT_ROOT / "data_era5"


def month_start_end(year: int, month: int) -> Tuple[datetime, datetime]:
    start = datetime(year, month, 1, 0)
    if month == 12:
        end = datetime(year, 12, 31, 23)
    else:
        end = datetime(year, month + 1, 1, 0) - global_infer.timedelta(hours=1)
    return start, end


def parse_optional_target(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return global_infer.aurora_batches.parse_target(value)


def split_targets(
    *,
    split_path: Path,
    split_name: str,
    data_root: Path,
    start: datetime,
    end: datetime,
) -> Tuple[List[datetime], DefaultDict[datetime, List[Dict[str, str]]], List[Tuple[datetime, Dict[str, str]]]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")

    rows_by_target: DefaultDict[datetime, List[Dict[str, str]]] = defaultdict(list)
    unavailable: List[Tuple[datetime, Dict[str, str]]] = []
    with split_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split_name:
                continue
            target_dt = global_infer.timestamp_to_rounded_hour(row["file_time_utc"])
            if target_dt < start or target_dt > end:
                continue
            if global_infer.has_era5_pair(data_root, target_dt):
                rows_by_target[target_dt].append(row)
            else:
                unavailable.append((target_dt, row))

    return sorted(rows_by_target), rows_by_target, unavailable


def write_manifest(
    path: Path,
    targets: List[datetime],
    rows_by_target: DefaultDict[datetime, List[Dict[str, str]]],
    unavailable: List[Tuple[datetime, Dict[str, str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "target_hour_utc",
                "split_row_count",
                "first_file_time_utc",
                "first_file",
            ],
        )
        writer.writeheader()
        for target_dt in targets:
            rows = rows_by_target[target_dt]
            writer.writerow(
                {
                    "status": "selected",
                    "target_hour_utc": global_infer.target_string(target_dt),
                    "split_row_count": len(rows),
                    "first_file_time_utc": rows[0].get("file_time_utc", ""),
                    "first_file": rows[0].get("file", ""),
                }
            )
        for target_dt, row in unavailable:
            writer.writerow(
                {
                    "status": "missing_era5_pair",
                    "target_hour_utc": global_infer.target_string(target_dt),
                    "split_row_count": 1,
                    "first_file_time_utc": row.get("file_time_utc", ""),
                    "first_file": row.get("file", ""),
                }
            )


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--model-dir", type=Path, default=global_infer.DEFAULT_MODEL_DIR)
    parser.add_argument("--split-path", type=Path, default=global_infer.DEFAULT_MODEL_DIR / "file_split.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--start", default=None, help="Optional first target hour, YYYY_MM_DD_HH.")
    parser.add_argument("--end", default=None, help="Optional last target hour, YYYY_MM_DD_HH.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of target hours.")
    parser.add_argument("--list-only", action="store_true", help="Print selected target hours without running inference.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inference when the NetCDF already exists.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--aurora-device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument("--base-lon-convention", choices=["minus180_180", "era5"], default="minus180_180")
    parser.add_argument("--aurora-backbone", choices=["full", "small"], default="full")
    parser.add_argument("--tokens-on-device", action="store_true")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--make-globe", action="store_true", help="Also write a globe PNG for each target.")
    parser.add_argument("--globe-center-lon", type=float, default=-90.0)
    parser.add_argument("--globe-center-lat", type=float, default=-20.0)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()
    netcdf_dir = output_dir / "netcdf"
    visualization_dir = output_dir / "visualizations"
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    default_start, default_end = month_start_end(args.year, args.month)
    start = parse_optional_target(args.start) or default_start
    end = parse_optional_target(args.end) or default_end
    if end < start:
        raise ValueError(f"End hour {end} is before start hour {start}")

    targets, rows_by_target, unavailable = split_targets(
        split_path=args.split_path,
        split_name=args.split,
        data_root=data_root,
        start=start,
        end=end,
    )
    if args.limit is not None:
        targets = targets[: args.limit]

    manifest_path = output_dir / "target_manifest.csv"
    write_manifest(manifest_path, targets, rows_by_target, unavailable)

    print(f"Data root: {data_root}")
    print(f"Model dir: {args.model_dir}")
    print(f"Split path: {args.split_path}")
    print(f"Split: {args.split}")
    print(f"Window: {global_infer.target_string(start)} through {global_infer.target_string(end)}")
    print(f"Output dir: {output_dir}")
    print(f"Selected target hours: {len(targets)}")
    print(f"Unavailable split rows in window: {len(unavailable)}")
    print(f"Manifest: {manifest_path}")

    if args.list_only:
        for target_dt in targets:
            print(global_infer.target_string(target_dt))
        return 0

    if not targets:
        raise ValueError("No available target hours found for the requested split/window.")

    device = global_infer.resolve_device(args.device)
    aurora_device = global_infer.resolve_device(args.aurora_device or args.device)
    print(f"Classifier device: {device}; Aurora device: {aurora_device}")

    for index, target_dt in enumerate(targets, start=1):
        target_label = global_infer.target_string(target_dt)
        netcdf_path = netcdf_dir / f"global_cloud_probabilities_{target_label}.nc"
        flat_png_path = visualization_dir / f"global_cloud_probabilities_{target_label}.png"
        globe_png_path = visualization_dir / (
            f"global_cloud_probabilities_{target_label}_globe_"
            f"lon{args.globe_center_lon:g}_lat{args.globe_center_lat:g}.png"
        )

        print(f"[{index}/{len(targets)}] Target hour: {target_dt:%Y-%m-%d %H:00 UTC}")
        if args.skip_existing and netcdf_path.is_file():
            print(f"Using existing NetCDF: {netcdf_path}")
        else:
            cloud_mask_prob, column_cloud_prob, lat, lon, attrs = global_infer.predict_global_cloud_probabilities(
                target_dt=target_dt,
                data_root=data_root,
                model_dir=args.model_dir,
                device=device,
                aurora_device=aurora_device,
                batch_size=args.batch_size,
                row_chunk_size=args.row_chunk_size,
                base_lon_convention=args.base_lon_convention,
                tokens_on_device=args.tokens_on_device,
                aurora_backbone=args.aurora_backbone,
            )
            split_rows = rows_by_target[target_dt]
            attrs["split_path"] = str(args.split_path)
            attrs["split"] = args.split
            attrs["split_row_count_for_target"] = len(split_rows)
            global_infer.save_dataset(
                output_path=netcdf_path,
                cloud_mask_prob=cloud_mask_prob,
                column_cloud_prob=column_cloud_prob,
                lat=lat,
                lon=lon,
                attrs=attrs,
                selected_row=split_rows[0] if split_rows else None,
                available_candidate_count=len(targets),
            )
            print(f"Saved NetCDF: {netcdf_path}")

        global_vis.plot_probability_map(
            input_path=netcdf_path,
            output_path=flat_png_path,
            cmap=args.cmap,
            threshold=args.threshold,
            dpi=args.dpi,
        )
        print(f"Saved flat PNG: {flat_png_path}")

        if args.make_globe:
            global_vis.plot_probability_globe(
                input_path=netcdf_path,
                output_path=globe_png_path,
                cmap=args.cmap,
                threshold=args.threshold,
                dpi=args.dpi,
                center_lon=args.globe_center_lon,
                center_lat=args.globe_center_lat,
            )
            print(f"Saved globe PNG: {globe_png_path}")

    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
