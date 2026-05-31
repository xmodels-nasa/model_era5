#!/usr/bin/env python3
"""Train the Aurora-style raw-chip classifier from forecast raw-chip NPZ files."""

import os
from pathlib import Path

import train_multilable_from_rawchips_aurora_architecturer as base


FORECAST_RAW_CHIPS_DIR = os.getenv(
    "FORECAST_RAW_CHIPS_DIR",
    os.getenv("RAW_CHIPS_FORECAST_DIR", str(base.PROJECT_ROOT / "raw_chips_forecast")),
)


def main() -> int:
    base.raw_train.RAW_CHIPS_DIR = FORECAST_RAW_CHIPS_DIR
    base.raw_train.LOG_DIR = str(base.PROJECT_ROOT / "logs_forecast")
    base.OUTPUT_DIR = str(base.PROJECT_ROOT / "baseline_model_outputs_aurora_forecast")
    Path(base.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(base.raw_train.LOG_DIR).mkdir(parents=True, exist_ok=True)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
