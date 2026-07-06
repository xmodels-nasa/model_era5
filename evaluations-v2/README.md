# evaluations-v2

Test-set evaluators for the four `results-v2` models. These scripts load saved checkpoints and split manifests; they do not retrain models. The default split is `test`.

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

Activate the project virtual environment first, then run from the project root:

```bash
cd /home/charley/model_era5
source .venv/bin/activate
```

```bash
python evaluations-v2/evaluate_baseline_validation_metrics.py
python evaluations-v2/evaluate_aurora_validation_metrics.py
python evaluations-v2/evaluate_fine_tune_validation_metrics.py
python evaluations-v2/evaluate_transformer_validation_metrics.py
```

Each script writes:

```text
results-v2/<model_output_dir>/test_distance_iou_metrics.json
results-v2/<model_output_dir>/test_distance_iou_metrics.csv
```

## Data Paths

By default, paths come from `.env`:

- `FEATHER_ROOT`
- `RAW_CHIPS_DIR`
- `EMBEDDING_OUTUT_DIR` or `EMBEDDING_OUTPUT_DIR`

If the test files are somewhere else, override them:

```bash
python evaluations-v2/evaluate_baseline_validation_metrics.py \
  --feather-root /path/to/feathers \
  --raw-chips-dir /path/to/raw_chips

python evaluations-v2/evaluate_fine_tune_validation_metrics.py \
  --feather-root /path/to/feathers \
  --embedding-dir /path/to/embeddings
```

Useful optional flags:

```text
--device cpu
--batch-size 4096
--split validation
--threshold 0.83
--output-json /path/to/metrics.json
```

## Cloudy-Sky Mask Visualizations

To find individual sparse cloudy-sky test profiles where strict IoU is similar, but the fine-tuned models have much better tolerance IoU than the raw-chip baselines:

```bash
python evaluations-v2/visualize_single_cloudy_sky_points.py
```

Default behavior:

- scans the `test` split
- requires ground truth to have 3-10 cloud-mask bins set to 1
- requires both fine-tuned and raw-chip model groups to make nonzero, similar numbers of positive predictions
- keeps points where fine-tuned strict IoU is close to raw-chip strict IoU
- ranks points where fine-tuned models beat raw-chip models on tolerance IoU @1 and @2
- diversifies the result by defaulting to at most one point per test file
- saves the best 100 points under `results-v2/single_cloudy_sky_point_visualizations`

Each PNG shows five vertical 40-bin masks:

```text
Ground truth | Fine-tune Transformer | Fine-tune MLP | U-Net raw chips | Aurora raw chips
```

Black means mask value `1`; white means mask value `0`.

Useful options:

```bash
python evaluations-v2/visualize_single_cloudy_sky_points.py \
  --max-points 100 \
  --min-target-ones 3 \
  --max-target-ones 10 \
  --min-pred-ones 1 \
  --max-pred-ones 12 \
  --max-pred-count-gap 3 \
  --strict-gain-min -0.08 \
  --strict-gain-max 0.08 \
  --min-tolerance-gain 0.20 \
  --max-points-per-file 1 \
  --batch-size 4096
```

To find consecutive test windows and plot curtain-style masks for ground truth plus all four model predictions:

```bash
python evaluations-v2/visualize_cloudy_sky_test_windows.py
```

Default behavior:

- scans the `test` split
- uses 20-point consecutive windows
- requires at least 8 rows in the window to match the sparse cloudy-sky criteria
- matching rows have 3-10 target cloud bins, nonzero/similar prediction counts, similar strict IoU, and better fine-tuned tolerance IoU
- plots five curtain panels: ground truth, Transformer, MLP, U-Net, Aurora
- saves up to 30 windows under `results-v2/cloudy_sky_mask_visualizations`

Outputs per selected window:

```text
window_###_...png          curtain-panel visualization
window_###_..._points.csv  row index, lat/lon, per-row metrics
window_###_..._masks.npz   ground truth, probabilities, binary predictions
window_###_..._metrics.json
summary.csv
```

Useful options:

```bash
python evaluations-v2/visualize_cloudy_sky_test_windows.py \
  --window-size 20 \
  --max-windows 30 \
  --min-matching-rows 8 \
  --min-target-ones 3 \
  --max-target-ones 10 \
  --strict-gain-min -0.08 \
  --strict-gain-max 0.08 \
  --min-tolerance-gain 0.15 \
  --batch-size 4096
```
