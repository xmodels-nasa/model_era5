#!/usr/bin/env python3
"""Evaluate a saved raw-chip classifier on the full seed-selected validation split."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_multilabel_from_raw_chips as train  # noqa: E402


def _torch_load(path: Path, device: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _sample_iou_np(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    intersection = ((preds == 1) & (targets == 1)).sum(axis=1).astype(np.float32)
    union = ((preds == 1) | (targets == 1)).sum(axis=1).astype(np.float32)
    return np.where(union > 0, intersection / np.maximum(union, 1.0), 1.0)


def _choose_good_windows(
    row_indices: np.ndarray,
    targets: np.ndarray,
    pred_classes: np.ndarray,
    curtain_rows: int,
    rng: np.random.Generator,
    min_iou: float,
    max_iou: float,
) -> List[Tuple[int, int, float]]:
    sample_ious = _sample_iou_np(pred_classes, targets)
    candidates: List[Tuple[int, int, float]] = []
    run_start = 0
    for idx in range(1, row_indices.size + 1):
        is_break = idx == row_indices.size or row_indices[idx] != row_indices[idx - 1] + 1
        if not is_break:
            continue
        for start in range(run_start, idx - curtain_rows + 1):
            stop = start + curtain_rows
            window_targets = targets[start:stop]
            if window_targets.sum() <= 0:
                continue
            score = float(sample_ious[start:stop].mean())
            if min_iou < score < max_iou:
                candidates.append((start, stop, score))
        run_start = idx
    rng.shuffle(candidates)
    return candidates


def _save_good_curtain_plots(
    out_dir: Path,
    model: nn.Module,
    files: Sequence[train.FileMeta],
    stats: train.NormalizationStats,
    use_base_features: bool,
    eval_batch_size: int,
    device: str,
    num_plots: int,
    curtain_rows: int,
    prediction_threshold: float,
    seed: int,
    min_iou: float,
    max_iou: float,
) -> None:
    if train.plt is None:
        print("Skipping plots because matplotlib is not installed.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    saved = 0
    for file_idx in rng.permutation(len(files)):
        if saved >= num_plots:
            break
        meta = files[int(file_idx)]
        row_indices, targets, probs = _predict_masks_for_file(
            model=model,
            meta=meta,
            stats=stats,
            use_base_features=use_base_features,
            eval_batch_size=eval_batch_size,
            device=device,
        )
        if row_indices.size < curtain_rows:
            continue
        order = np.argsort(row_indices)
        row_indices = row_indices[order]
        targets = targets[order]
        probs = probs[order]
        pred_classes = (probs >= prediction_threshold).astype(np.float32)
        for start, stop, score in _choose_good_windows(
            row_indices=row_indices,
            targets=targets,
            pred_classes=pred_classes,
            curtain_rows=curtain_rows,
            rng=rng,
            min_iou=min_iou,
            max_iou=max_iou,
        ):
            if saved >= num_plots:
                break
            row_window = row_indices[start:stop]
            sample_path = out_dir / (
                f"raw_chips_good_iou_{saved + 1:02d}_{train._safe_stem(meta.source_file)}"
                f"_rows_{int(row_window[0]):06d}_{int(row_window[-1]):06d}"
                f"_iou_{score:.3f}.png"
            )
            fig, axes = train.plt.subplots(1, 3, figsize=(20, 8), sharey=True)
            for ax, image, title in zip(
                axes,
                (targets[start:stop], probs[start:stop], pred_classes[start:stop]),
                ("Ground Truth", "Prediction Probability", f"Prediction Class @thr={prediction_threshold:.3f}"),
            ):
                ax.imshow(image.T, aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.set_title(title)
                ax.set_xlabel("Feather row index")
            axes[0].set_ylabel("Cloud mask column")
            x_ticks = np.linspace(0, curtain_rows - 1, num=min(5, curtain_rows), dtype=int)
            for ax in axes:
                ax.set_xticks(x_ticks, labels=[str(int(row_window[idx])) for idx in x_ticks])
            for ax in axes[1:]:
                ax.tick_params(axis="y", labelleft=False)
            fig.suptitle(
                f"Validation cloud-mask curtain | {meta.source_file.name} | "
                f"rows {int(row_window[0])}-{int(row_window[-1])} | mean IoU={score:.3f}",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(sample_path, dpi=160)
            train.plt.close(fig)
            saved += 1
            print(f"Saved plot: {sample_path}")
    if saved < num_plots:
        print(f"Saved {saved} plot(s); requested {num_plots}.")


def _predict_masks_for_file(
    model: nn.Module,
    meta: train.FileMeta,
    stats: train.NormalizationStats,
    use_base_features: bool,
    eval_batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = train._load_npz_payload(
        meta=meta,
        sample_ratio=1.0,
        max_samples_per_file=None,
        seed=0,
    )
    with np.load(meta.npz_path, allow_pickle=False) as data:
        row_indices = data["row_indices"].astype(np.int64, copy=False)

    labels = payload["labels"]
    if labels.shape[0] == 0:
        empty_masks = np.empty((0, len(train.TARGET_COLUMNS)), dtype=np.float32)
        return np.empty((0,), dtype=np.int64), empty_masks, empty_masks
    if row_indices.shape[0] != labels.shape[0]:
        raise ValueError(f"row_indices/labels length mismatch in {meta.npz_path}")

    chips = train._flatten_chip_channels(payload["dynamic"], payload["static"])
    base_features = payload["base_features"]
    pred_parts: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, labels.shape[0], eval_batch_size):
            stop = min(start + eval_batch_size, labels.shape[0])
            chips_t, base_t = train._normalize_batch(
                chips=chips[start:stop],
                base_features=base_features[start:stop],
                stats=stats,
                use_base_features=use_base_features,
            )
            logits = model(chips_t.to(device), base_t.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            pred_parts.append(probs.astype(np.float32, copy=False))

    targets = (labels > 0.5).astype(np.float32, copy=False)
    return row_indices, targets, np.concatenate(pred_parts, axis=0)


def _write_split_manifest(path: Path, files: Sequence[train.FileMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file", "npz", "file_time_utc"])
        writer.writeheader()
        for meta in files:
            writer.writerow(
                {
                    "split": "validation",
                    "file": str(meta.source_file),
                    "npz": str(meta.npz_path),
                    "file_time_utc": str(meta.file_time),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output-dir", type=str, default=train.OUTPUT_DIR)
    parser.add_argument("--raw-chip-dir", type=str, default=train.RAW_CHIPS_DIR)
    parser.add_argument("--train-files", type=int, default=1000)
    parser.add_argument("--test-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-start-time", type=str, default=train.DEFAULT_TEST_START_TIME)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--search-iou-threshold", action="store_true")
    parser.add_argument("--plot-count", type=int, default=20)
    parser.add_argument("--plot-curtain-rows", type=int, default=100)
    parser.add_argument("--plot-min-iou", type=float, default=0.6)
    parser.add_argument("--plot-max-iou", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.model_output_dir)
    ckpt = _torch_load(out_dir / "multilabel_unet_classifier.pt", args.device)
    stats_npz = np.load(out_dir / "normalization_stats.npz")
    stats = train.NormalizationStats(
        chip_mean=stats_npz["chip_mean"],
        chip_std=stats_npz["chip_std"],
        base_mean=stats_npz["base_mean"],
        base_std=stats_npz["base_std"],
    )
    use_base_features = bool(ckpt.get("base_features"))
    model = train.UNetClassifier(
        in_channels=int(ckpt["input_channels"]),
        output_dim=int(ckpt.get("output_dim", len(train.TARGET_COLUMNS))),
        base_feature_dim=(len(train.BASE_FEATURE_COLUMNS) if use_base_features else 0),
        base_channels=int(ckpt.get("base_channels", 32)),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metas = train.discover_files(Path(args.raw_chip_dir))
    _, test_files = train.select_train_test_files(
        metas=metas,
        train_files=args.train_files,
        test_files=args.test_files,
        seed=args.seed,
        test_start_time=train._parse_utc_timestamp(args.test_start_time),
    )

    loss_fn = nn.BCEWithLogitsLoss()
    saved_threshold = ckpt.get("test_metrics", {}).get("iou_threshold") if isinstance(ckpt.get("test_metrics"), dict) else None
    iou_threshold = args.iou_threshold if args.iou_threshold is not None else saved_threshold
    loss, metrics, _ = train._evaluate_files(
        model=model,
        loss_fn=loss_fn,
        files=test_files,
        batch_size=args.eval_batch_size,
        device=args.device,
        stats=stats,
        use_base_features=use_base_features,
        sample_ratio=1.0,
        max_samples_per_file=None,
        seed=args.seed,
        iou_threshold=iou_threshold,
        search_iou_threshold=args.search_iou_threshold,
    )

    print(
        "Validation metrics | "
        f"loss={loss:.5f}, samples={metrics['inference_sample_count']}, "
        f"iou_mean={metrics['iou_mean']:.4f} @thr={metrics['iou_threshold']:.3f}, "
        f"iou_empty={metrics['iou_empty_truth_mean']:.4f} n={metrics['empty_truth_count']}, "
        f"iou_nonempty={metrics['iou_nonempty_truth_mean']:.4f} n={metrics['nonempty_truth_count']}, "
        f"f1_macro={metrics['f1_macro']:.4f}, auc_macro={metrics['auc_macro']:.4f}"
    )

    plot_dir = out_dir / "plots"
    _write_split_manifest(plot_dir / "validation_file_split.csv", test_files)
    _save_good_curtain_plots(
        out_dir=plot_dir,
        model=model,
        files=test_files,
        stats=stats,
        use_base_features=use_base_features,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        num_plots=args.plot_count,
        curtain_rows=args.plot_curtain_rows,
        prediction_threshold=float(metrics["iou_threshold"]),
        seed=args.seed,
        min_iou=args.plot_min_iou,
        max_iou=args.plot_max_iou,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
