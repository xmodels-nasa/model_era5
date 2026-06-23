# evaluations-v2

Validation-only evaluators for the four `results-v2` models. These scripts load saved checkpoints and split manifests; they do not retrain models.

## Metrics

All scripts compute strict IoU plus distance-aware 40-bin mask metrics:

- `strict_iou`
- `tolerance_iou_1`
- `tolerance_iou_2`
- `gaussian_smoothed_iou_sigma_1`
- `gaussian_smoothed_iou_sigma_2`
- `distance_weighted_iou_sigma_1`
- `distance_weighted_iou_sigma_2`

Empty ground truth keeps the strict definition:

- empty prediction: score `1`
- nonempty prediction: score `0`

## Run

From `/Users/charley/nasa/4dcloud-2026/model_era5`:

```bash
uv run python evaluations-v2/evaluate_baseline_validation_metrics.py
uv run python evaluations-v2/evaluate_aurora_validation_metrics.py
uv run python evaluations-v2/evaluate_fine_tune_validation_metrics.py
uv run python evaluations-v2/evaluate_transformer_validation_metrics.py
```

Each script writes:

```text
results-v2/<model_output_dir>/validation_distance_iou_metrics.json
results-v2/<model_output_dir>/validation_distance_iou_metrics.csv
```

## Data Paths

By default, paths come from `.env`:

- `FEATHER_ROOT`
- `RAW_CHIPS_DIR`
- `EMBEDDING_OUTUT_DIR` or `EMBEDDING_OUTPUT_DIR`

If the validation files are somewhere else, override them:

```bash
uv run python evaluations-v2/evaluate_baseline_validation_metrics.py \
  --feather-root /path/to/feathers \
  --raw-chips-dir /path/to/raw_chips

uv run python evaluations-v2/evaluate_fine_tune_validation_metrics.py \
  --feather-root /path/to/feathers \
  --embedding-dir /path/to/embeddings
```

Useful optional flags:

```text
--device cpu
--batch-size 4096
--threshold 0.83
--output-json /path/to/metrics.json
```
