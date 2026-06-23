#!/usr/bin/env python3
"""Evaluate fine-tuned embedding Transformer validation distance-aware IoU metrics."""

from distance_metrics_common import EvalConfig, RESULTS_DIR, run


CONFIG = EvalConfig(
    model_name="fine_tune_embedding_transformer",
    model_kind="embedding_transformer",
    default_output_dir=RESULTS_DIR / "model_outputs_transformer",
    default_model_path=RESULTS_DIR / "model_outputs_transformer" / "multilabel_transformer.pt",
    default_stats_path=RESULTS_DIR / "model_outputs_transformer" / "feature_stats.npz",
    default_split_path=RESULTS_DIR / "model_outputs_transformer" / "file_split.csv",
)


if __name__ == "__main__":
    raise SystemExit(run(CONFIG))
