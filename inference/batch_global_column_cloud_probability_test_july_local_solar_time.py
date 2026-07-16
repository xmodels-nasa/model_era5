#!/usr/bin/env python3
"""Run July test-split global inference using the results-v3 local-solar-time model."""

from __future__ import annotations

from pathlib import Path

import batch_global_column_cloud_probability_test_july as july_batch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SOLAR_TIME_MODEL_DIR = PROJECT_ROOT / "results-v3" / "model_outputs_transformer_local_solar_time"
LOCAL_SOLAR_TIME_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "test_july_global_column_cloud_probability_local_solar_time"
)


def main() -> int:
    parser = july_batch.build_parser(description=__doc__)
    parser.set_defaults(
        model_dir=LOCAL_SOLAR_TIME_MODEL_DIR,
        split_path=LOCAL_SOLAR_TIME_MODEL_DIR / "file_split.csv",
        output_dir=LOCAL_SOLAR_TIME_OUTPUT_DIR,
    )
    return july_batch.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
