#!/usr/bin/env python3
"""Evaluate Aurora-style raw-chip validation distance-aware IoU metrics."""

from distance_metrics_common import EvalConfig, RESULTS_DIR, run


CONFIG = EvalConfig(
    model_name="aurora_raw_chip_classifier",
    model_kind="raw_aurora",
    default_output_dir=RESULTS_DIR / "baseline_model_outputs_aurora",
    default_model_path=RESULTS_DIR / "baseline_model_outputs_aurora" / "multilabel_aurora_rawchip_classifier.pt",
    default_stats_path=RESULTS_DIR / "baseline_model_outputs_aurora" / "normalization_stats.npz",
    default_split_path=RESULTS_DIR / "baseline_model_outputs_aurora" / "file_split.csv",
)


if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
