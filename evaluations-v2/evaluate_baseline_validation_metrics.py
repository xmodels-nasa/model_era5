#!/usr/bin/env python3
"""Evaluate baseline raw-chip U-Net distance-aware IoU metrics."""

from distance_metrics_common import EvalConfig, RESULTS_DIR, run


CONFIG = EvalConfig(
    model_name="baseline_raw_chip_unet",
    model_kind="raw_unet",
    default_output_dir=RESULTS_DIR / "baseline_model_outputs",
    default_model_path=RESULTS_DIR / "baseline_model_outputs" / "multilabel_unet_classifier.pt",
    default_stats_path=RESULTS_DIR / "baseline_model_outputs" / "normalization_stats.npz",
    default_split_path=RESULTS_DIR / "baseline_model_outputs" / "file_split.csv",
)


if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
