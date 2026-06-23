#!/usr/bin/env python3
"""Evaluate fine-tuned embedding MLP distance-aware IoU metrics."""

from distance_metrics_common import EvalConfig, RESULTS_DIR, run


CONFIG = EvalConfig(
    model_name="fine_tune_embedding_mlp",
    model_kind="embedding_mlp",
    default_output_dir=RESULTS_DIR / "model_outputs_fine_tune",
    default_model_path=RESULTS_DIR / "model_outputs_fine_tune" / "multilabel_mlp.pt",
    default_stats_path=RESULTS_DIR / "model_outputs_fine_tune" / "feature_stats.npz",
    default_split_path=RESULTS_DIR / "model_outputs_fine_tune" / "file_split.csv",
)


if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
