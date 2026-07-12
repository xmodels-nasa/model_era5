#!/usr/bin/env python3
"""Run global transformer inference for all local ERA5 target hours.

The single-hour script remains the source of truth for model inference. This
batch wrapper discovers local ERA5 folders named YYYY_MM_DD_HH_data, keeps only
hours that also have the required t-6 input folder, then saves NetCDF and flat
2D PNG outputs in one batch directory.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_OUTPUT_DIR = THIS_DIR / "outputs" / "batch_global_column_cloud_probability"

import build_global_column_cloud_probability as global_infer  # noqa: E402
import visualize_global_column_cloud_probability as global_vis  # noqa: E402


ERA5_HOUR_DIR_RE = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_data$")


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


def parse_hour_dir_name(name: str) -> Optional[datetime]:
    match = ERA5_HOUR_DIR_RE.match(name)
    if not match:
        return None
    year, month, day, hour = [int(part) for part in match.groups()]
    return datetime(year, month, day, hour)


def discover_targets(data_root: Path) -> List[datetime]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing ERA5 data root: {data_root}")

    available_hours = set()
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        dt = parse_hour_dir_name(child.name)
        if dt is not None:
            available_hours.add(dt)

    targets = [
        dt
        for dt in sorted(available_hours)
        if dt - timedelta(hours=6) in available_hours
        and global_infer.has_era5_pair(data_root, dt)
    ]
    return targets


def parse_target(value: str) -> datetime:
    return global_infer.aurora_batches.parse_target(value)


def filter_targets(
    targets: List[datetime],
    start: Optional[datetime],
    end: Optional[datetime],
    limit: Optional[int],
) -> List[datetime]:
    filtered = targets
    if start is not None:
        filtered = [dt for dt in filtered if dt >= start]
    if end is not None:
        filtered = [dt for dt in filtered if dt <= end]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--model-dir", type=Path, default=global_infer.DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="*", default=None, help="Optional explicit target hours: YYYY_MM_DD_HH ...")
    parser.add_argument("--start", default=None, help="Optional first target hour, YYYY_MM_DD_HH.")
    parser.add_argument("--end", default=None, help="Optional last target hour, YYYY_MM_DD_HH.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of target hours.")
    parser.add_argument("--list-only", action="store_true", help="Print runnable target hours without running inference.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inference when the NetCDF already exists.")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser()
    output_dir = args.output_dir.expanduser()
    netcdf_dir = output_dir / "netcdf"
    visualization_dir = output_dir / "visualizations"
    netcdf_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    if args.targets:
        targets = [parse_target(value) for value in args.targets]
        missing = [global_infer.target_string(dt) for dt in targets if not global_infer.has_era5_pair(data_root, dt)]
        if missing:
            raise FileNotFoundError(f"Targets missing required ERA5 t-6/t folders under {data_root}: {missing}")
    else:
        targets = discover_targets(data_root)
        targets = filter_targets(
            targets,
            start=parse_target(args.start) if args.start else None,
            end=parse_target(args.end) if args.end else None,
            limit=args.limit,
        )

    if not targets:
        raise ValueError(f"No runnable target hours found under {data_root}")

    device = global_infer.resolve_device(args.device)
    aurora_device = global_infer.resolve_device(args.aurora_device or args.device)

    print(f"Data root: {data_root}")
    print(f"Model dir: {args.model_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Targets: {len(targets)}")
    print(f"Classifier device: {device}; Aurora device: {aurora_device}")
    if args.list_only:
        for target_dt in targets:
            print(global_infer.target_string(target_dt))
        return 0

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
            global_infer.save_dataset(
                output_path=netcdf_path,
                cloud_mask_prob=cloud_mask_prob,
                column_cloud_prob=column_cloud_prob,
                lat=lat,
                lon=lon,
                attrs=attrs,
                selected_row=None,
                available_candidate_count=None,
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


if __name__ == "__main__":
    raise SystemExit(main())
