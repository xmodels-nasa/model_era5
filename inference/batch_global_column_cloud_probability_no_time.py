#!/usr/bin/env python3
"""Run local batch global inference using the results-v3 no-time model."""

from __future__ import annotations

from pathlib import Path

import batch_global_column_cloud_probability as batch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NO_TIME_MODEL_DIR = PROJECT_ROOT / "results-v3" / "model_outputs_transformer_no_time"
NO_TIME_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "batch_global_column_cloud_probability_no_time"


def main() -> int:
    parser = batch.build_parser(description=__doc__)
    parser.set_defaults(model_dir=NO_TIME_MODEL_DIR, output_dir=NO_TIME_OUTPUT_DIR)
    return batch.run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
