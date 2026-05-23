#!/usr/bin/env python3
"""Batch forecast embedding builder for one Feather file.

For each target row at rounded hour t, compute the encoder context for
forecast source hour t-6. Aurora then receives input timestamps (t-12, t-6),
while `source_row` still points back to the target row at t.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

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

from build_aurora_batches import (  # noqa: E402
    DATA_ROOT,
    get_embedding_from_encoder_context,
    get_encoder_context_for_target,
    load_aurora_model,
)


FEATHER_FILE = os.getenv("FEATHER_FILE", "/path/to/your.feather")
DATA_ROOT_OVERRIDE: Optional[str] = None
SAMPLE_RATIO = 1.0
RANDOM_STATE = 42
FORECAST_LEAD_HOURS = int(os.getenv("FORECAST_LEAD_HOURS", "6"))


def _rounded_hours(df: pd.DataFrame) -> pd.Series:
    first_ts = int(df["timestamp_0"].iloc[0])
    unit = "s" if len(str(abs(first_ts))) <= 10 else "ms"
    df_dt = pd.to_datetime(df["timestamp_0"], unit=unit, utc=True)
    hour_floor = df_dt.dt.floor("h")
    round_up = (df_dt - hour_floor) > pd.Timedelta(minutes=30)
    return hour_floor + pd.to_timedelta(round_up.astype(int), unit="h")


def _format_target_hour(hours: pd.Series) -> pd.Series:
    return hours.dt.strftime("%Y_%m_%d_%H")


def process_feather_batched_forecast(
    file_path: str,
    data_root: Optional[str] = None,
    sample_ratio: float = SAMPLE_RATIO,
    random_state: int = RANDOM_STATE,
    forecast_lead_hours: int = FORECAST_LEAD_HOURS,
) -> List[Dict[str, object]]:
    """Read one feather file and compute forecast embeddings with per-hour reuse."""
    if forecast_lead_hours <= 0:
        raise ValueError("forecast_lead_hours must be > 0.")

    df = pd.read_feather(file_path)
    required = ["Latitude_0", "Longitude_0", "timestamp_0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in feather file: {missing}")
    if len(df) == 0:
        return []

    df = df.reset_index().rename(columns={"index": "source_row"})
    target_hours = _rounded_hours(df)
    embedding_hours = target_hours - pd.Timedelta(hours=forecast_lead_hours)
    df["target_hour"] = _format_target_hour(target_hours)
    df["embedding_target"] = _format_target_hour(embedding_hours)

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

    unique_targets = list(sampled["embedding_target"].unique())
    print(
        f"Processing {len(sampled)} target row(s) across {len(unique_targets)} "
        f"forecast source hour(s), lead={forecast_lead_hours}h"
    )
    for embedding_target in unique_targets:
        group = sampled[sampled["embedding_target"] == embedding_target]
        print(f"Building shared encoder context for forecast_source={embedding_target} ({len(group)} row(s))")
        try:
            encoder_context = get_encoder_context_for_target(
                data_root=root,
                target=embedding_target,
                model=model,
            )
        except FileNotFoundError as e:
            print(
                f"[WARN] Skipping forecast_source={embedding_target}: missing ERA5 input "
                f"({len(group)} target row(s) skipped). Details: {e}"
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
            emb["target_hour"] = row["target_hour"]
            emb["forecast_source_hour"] = embedding_target
            emb["forecast_lead_hours"] = int(forecast_lead_hours)
            results.append(emb)

    results.sort(key=lambda x: x["source_row"])
    return results


def main(
    feather_path: Optional[str] = None,
    data_root: Optional[str] = None,
    sample_ratio: float = SAMPLE_RATIO,
    random_state: int = RANDOM_STATE,
    forecast_lead_hours: int = FORECAST_LEAD_HOURS,
) -> List[Dict[str, object]]:
    path = feather_path or FEATHER_FILE
    root = data_root or DATA_ROOT_OVERRIDE or DATA_ROOT
    embeddings = process_feather_batched_forecast(
        file_path=path,
        data_root=root,
        sample_ratio=sample_ratio,
        random_state=random_state,
        forecast_lead_hours=forecast_lead_hours,
    )
    print(f"Computed forecast embeddings for {len(embeddings)} target row(s)")
    return embeddings


if __name__ == "__main__":
    main()
