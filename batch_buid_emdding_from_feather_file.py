#!/usr/bin/env python3
"""Batch embedding builder for one Feather file.

Compared with `build_embedding_from_feather_file.py`, this groups rows by target hour,
computes Aurora encoder output once per hour, and reuses it for all row lat/lon queries.
"""

import os
from typing import Dict, List, Optional

import pandas as pd


def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
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

FEATHER_FILE = os.getenv("FEATHER_FILE", "/path/to/your.feather")
DATA_ROOT_OVERRIDE: Optional[str] = None
SAMPLE_RATIO = 1.0
RANDOM_STATE = 42


def _build_target_column(df: pd.DataFrame) -> pd.Series:
    first_ts = int(df["timestamp_0"].iloc[0])
    unit = "s" if len(str(abs(first_ts))) <= 10 else "ms"
    df_dt = pd.to_datetime(df["timestamp_0"], unit=unit, utc=True)
    hour_floor = df_dt.dt.floor("h")
    round_up = (df_dt - hour_floor) > pd.Timedelta(minutes=30)
    rounded_hour = hour_floor + pd.to_timedelta(round_up.astype(int), unit="h")
    return rounded_hour.dt.strftime("%Y_%m_%d_%H")


def process_feather_batched(
    file_path: str,
    data_root: Optional[str] = None,
    sample_ratio: float = SAMPLE_RATIO,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, object]]:
    """Read one feather file and compute embeddings with per-target batching."""
    df = pd.read_feather(file_path)
    required = ["Latitude_0", "Longitude_0", "timestamp_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feather file: {missing}")
    if len(df) == 0:
        return []

    df = df.reset_index().rename(columns={"index": "source_row"})
    df["target"] = _build_target_column(df)

    if not (0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")
    if sample_ratio < 1.0:
        n = max(1, int(len(df) * sample_ratio))
        sampled = df.sample(n=n, random_state=random_state).copy()
    else:
        sampled = df.copy()

    root = data_root or DATA_ROOT_OVERRIDE or DATA_ROOT
    model = load_aurora_model()
    results: List[Dict[str, object]] = []

    unique_targets = list(sampled["target"].unique())
    print(f"Processing {len(sampled)} row(s) across {len(unique_targets)} unique target hour(s)")
    for target in unique_targets:
        group = sampled[sampled["target"] == target]
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
            emb = get_embedding_from_encoder_context(
                encoder_context=encoder_context,
                lat=float(row["Latitude_0"]),
                lon=float(row["Longitude_0"]),
            )
            emb["source_row"] = int(row["source_row"])
            emb["source_file"] = file_path
            results.append(emb)

    results.sort(key=lambda x: x["source_row"])
    return results


def main(
    feather_path: Optional[str] = None,
    data_root: Optional[str] = None,
    sample_ratio: float = SAMPLE_RATIO,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, object]]:
    path = feather_path or FEATHER_FILE
    root = data_root or DATA_ROOT_OVERRIDE or DATA_ROOT
    embeddings = process_feather_batched(
        file_path=path,
        data_root=root,
        sample_ratio=sample_ratio,
        random_state=random_state,
    )
    print(f"Computed embeddings for {len(embeddings)} row(s)")
    return embeddings


if __name__ == "__main__":
    main()
