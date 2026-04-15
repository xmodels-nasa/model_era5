#!/usr/bin/env python3
"""Report missing ERA5 target-hour inputs referenced by Feather files."""

import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Set

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


def _build_targets_from_timestamp(timestamp_col: pd.Series) -> pd.Series:
    first_ts = int(timestamp_col.iloc[0])
    unit = "s" if len(str(abs(first_ts))) <= 10 else "ms"
    dt = pd.to_datetime(timestamp_col, unit=unit, utc=True)
    hour_floor = dt.dt.floor("h")
    round_up = (dt - hour_floor) > pd.Timedelta(minutes=30)
    rounded_hour = hour_floor + pd.to_timedelta(round_up.astype(int), unit="h")
    return rounded_hour.dt.strftime("%Y_%m_%d_%H")


def _required_paths(data_root: Path, target: str) -> Dict[str, List[Path]]:
    dt = pd.to_datetime(target, format="%Y_%m_%d_%H", utc=True)
    t_minus_6 = dt - timedelta(hours=6)

    d_target = data_root / f"{dt.strftime('%Y_%m_%d_%H')}_data"
    d_prev = data_root / f"{t_minus_6.strftime('%Y_%m_%d_%H')}_data"
    required_files = ["_surface.nc", "_atmospheric.nc", "_static.nc"]

    return {
        "target_dir": [d_target],
        "prev_dir": [d_prev],
        "target_files": [d_target / name for name in required_files],
        "prev_files": [d_prev / name for name in required_files],
    }


def main() -> int:
    _load_dotenv()
    feather_root = os.getenv("FEATHER_ROOT", "").strip()
    data_root = os.getenv("DATA_ROOT", "").strip()

    if not feather_root:
        raise ValueError("FEATHER_ROOT is not set.")
    if not data_root:
        raise ValueError("DATA_ROOT is not set.")

    feather_root_path = Path(feather_root)
    data_root_path = Path(data_root)
    if not feather_root_path.exists():
        raise FileNotFoundError(f"FEATHER_ROOT does not exist: {feather_root_path}")
    if not data_root_path.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root_path}")

    feather_files = sorted(feather_root_path.rglob("*.feather"))
    if not feather_files:
        raise FileNotFoundError(f"No .feather files found under: {feather_root_path}")

    targets: Set[str] = set()
    for fp in feather_files:
        df = pd.read_feather(fp, columns=["timestamp_0"])
        if len(df) == 0:
            continue
        targets.update(_build_targets_from_timestamp(df["timestamp_0"]).unique().tolist())

    missing: Dict[str, List[str]] = {}
    missing_hour_dirs: Set[str] = set()
    for target in sorted(targets):
        paths = _required_paths(data_root_path, target)
        missing_items: List[str] = []
        for group_paths in paths.values():
            for p in group_paths:
                if not p.exists():
                    missing_items.append(str(p))
                    # Track the actual missing ERA5 hour folder name.
                    if p.name.endswith("_data"):
                        missing_hour_dirs.add(p.name)
                    elif p.parent.name.endswith("_data"):
                        missing_hour_dirs.add(p.parent.name)
        if missing_items:
            missing[target] = missing_items

    print(f"Feather files scanned: {len(feather_files)}")
    print(f"Unique target hours found: {len(targets)}")
    print(f"Target hours with missing inputs: {len(missing)}")
    print(f"Missing ERA5 hour folders: {len(missing_hour_dirs)}")

    if not missing:
        print("No missing target-hour inputs found.")
        return 0

    print("\nActual missing ERA5 hour folders:")
    for folder in sorted(missing_hour_dirs):
        print(f"- {folder}")

    #print("\nDetails:")
    #for target in sorted(missing):
    #    print(f"[{target}]")
    #    for p in missing[target]:
    #        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
