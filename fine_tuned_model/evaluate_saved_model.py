#!/usr/bin/env python3
"""Evaluate a saved embedding MLP on a net-new post-cutoff holdout split."""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_multilabel_from_feather_embeddings as train  # noqa: E402


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
    x_mean: np.ndarray,
    x_std: np.ndarray,
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
        row_indices, targets, probs = train._predict_masks_for_file(
            model=model,
            meta=meta,
            x_mean=x_mean,
            x_std=x_std,
            eval_batch_size=eval_batch_size,
            device=device,
        )
        if row_indices.size < curtain_rows:
            continue
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
                f"embedding_good_iou_{saved + 1:02d}_{train._safe_stem(meta.feather_path)}"
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
                f"Holdout test cloud-mask curtain | {meta.feather_path.name} | "
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


def _file_key(meta: train.FileMeta) -> str:
    return str(meta.feather_path.resolve())


def select_holdout_test_files(
    metas: Sequence[train.FileMeta],
    train_files: int,
    validation_files: int,
    test_files: int,
    training_seed: int,
    holdout_seed: int,
    test_start_time,
) -> Tuple[List[train.FileMeta], List[train.FileMeta]]:
    _, validation_selected = train.select_train_test_files(
        metas=metas,
        train_files=train_files,
        test_files=validation_files,
        seed=training_seed,
        test_start_time=test_start_time,
    )
    validation_keys = {_file_key(m) for m in validation_selected}
    sorted_metas = sorted(metas, key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path)))
    post_cutoff_pool = [m for m in sorted_metas if m.file_time >= test_start_time]
    holdout_pool = [
        m
        for m in post_cutoff_pool
        if _file_key(m) not in validation_keys
    ]
    print(f"Files on/after cutoff: {len(post_cutoff_pool)}")
    print(f"Validation files to exclude from post-cutoff pool: {len(validation_selected)}")
    print(f"Remaining net-new holdout candidates: {len(holdout_pool)}")
    print(f"Requested net-new holdout files: {test_files}")
    if len(holdout_pool) < test_files:
        raise ValueError(
            f"Not enough net-new holdout files on or after {test_start_time}. "
            f"Need {test_files}, found {len(holdout_pool)} after excluding "
            f"{len(validation_selected)} validation files."
        )
    rng = random.Random(holdout_seed)
    holdout_selected = rng.sample(holdout_pool, k=test_files)
    return (
        sorted(validation_selected, key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path))),
        sorted(holdout_selected, key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path))),
    )


def _write_split_manifest(path: Path, split_name: str, files: Sequence[train.FileMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file", "npz", "file_time_utc"])
        writer.writeheader()
        for meta in files:
            writer.writerow(
                {
                    "split": split_name,
                    "file": str(meta.feather_path),
                    "npz": str(meta.npz_path),
                    "file_time_utc": str(meta.file_time),
                }
            )


def _format_metrics_text(
    loss: float,
    metrics: Dict[str, object],
    test_files: Sequence[train.FileMeta],
    validation_files: Sequence[train.FileMeta],
    test_start_time,
    training_seed: int,
    holdout_seed: int,
) -> str:
    validation_keys = {_file_key(m) for m in validation_files}
    holdout_keys = {_file_key(m) for m in test_files}
    lines = [
        "Holdout test metrics",
        f"test_start_time_utc={test_start_time}",
        f"training_seed={training_seed}",
        f"holdout_seed={holdout_seed}",
        f"holdout_file_count={len(test_files)}",
        f"excluded_validation_file_count={len(validation_files)}",
        f"validation_holdout_overlap_count={len(validation_keys & holdout_keys)}",
        f"holdout_min_file_time_utc={min(m.file_time for m in test_files)}",
        f"holdout_max_file_time_utc={max(m.file_time for m in test_files)}",
        f"loss={loss:.8f}",
    ]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            lines.append(f"{key}={value:.8f}")
        else:
            lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output-dir", type=str, default=train.OUTPUT_DIR)
    parser.add_argument("--feather-root", type=str, default=train.FEATHER_ROOT)
    parser.add_argument("--embedding-dir", type=str, default=train.EMBEDDING_DIR)
    parser.add_argument("--train-files", type=int, default=1000)
    parser.add_argument("--validation-files", type=int, default=100)
    parser.add_argument("--test-files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-seed", type=int, default=None)
    parser.add_argument("--test-start-time", type=str, default=train.DEFAULT_TEST_START_TIME)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--search-iou-threshold", action="store_true")
    parser.add_argument("--plot-count", type=int, default=20)
    parser.add_argument("--plot-curtain-rows", type=int, default=100)
    parser.add_argument("--plot-min-iou", type=float, default=0.6)
    parser.add_argument("--plot-max-iou", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.model_output_dir)
    ckpt = _torch_load(out_dir / "multilabel_mlp.pt", args.device)
    stats_npz = np.load(out_dir / "feature_stats.npz")
    x_mean = stats_npz["x_mean"]
    x_std = stats_npz["x_std"]

    model = train.MultiLabelMLP(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", len(train.TARGET_COLUMNS))),
        hidden_dims=ckpt.get("hidden_dims"),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    metas = train.discover_files(Path(args.feather_root), Path(args.embedding_dir))
    test_start_time = train._parse_utc_timestamp(args.test_start_time)
    holdout_seed = args.holdout_seed if args.holdout_seed is not None else args.seed + 1
    validation_files, test_files = select_holdout_test_files(
        metas=metas,
        train_files=args.train_files,
        validation_files=args.validation_files,
        test_files=args.test_files,
        training_seed=args.seed,
        holdout_seed=holdout_seed,
        test_start_time=test_start_time,
    )
    print(f"Excluded validation files: {len(validation_files)}")
    print(f"Selected net-new test files: {len(test_files)}")
    print(f"Test file cutoff: {test_start_time}")
    print(f"Net-new test min file_time: {min(m.file_time for m in test_files)}")
    print(f"Net-new test max file_time: {max(m.file_time for m in test_files)}")

    x_test, y_test = train.load_dataset(test_files, sample_ratio=1.0, max_samples_per_file=None, seed=args.seed)
    x_test = (x_test - x_mean) / x_std
    loss_fn = nn.BCEWithLogitsLoss()
    saved_threshold = ckpt.get("test_metrics", {}).get("iou_threshold") if isinstance(ckpt.get("test_metrics"), dict) else None
    iou_threshold = args.iou_threshold if args.iou_threshold is not None else saved_threshold
    loss, metrics, _ = train._evaluate_in_batches(
        model=model,
        loss_fn=loss_fn,
        x=torch.from_numpy(x_test.astype(np.float32, copy=False)),
        y=torch.from_numpy(y_test.astype(np.float32, copy=False)),
        batch_size=args.eval_batch_size,
        device=args.device,
        iou_threshold=iou_threshold,
        search_iou_threshold=args.search_iou_threshold,
    )

    print(
        "Holdout test metrics | "
        f"loss={loss:.5f}, samples={metrics['inference_sample_count']}, "
        f"iou_mean={metrics['iou_mean']:.4f} @thr={metrics['iou_threshold']:.3f}, "
        f"iou_empty={metrics['iou_empty_truth_mean']:.4f} n={metrics['empty_truth_count']}, "
        f"iou_nonempty={metrics['iou_nonempty_truth_mean']:.4f} n={metrics['nonempty_truth_count']}, "
        f"f1_macro={metrics['f1_macro']:.4f}, auc_macro={metrics['auc_macro']:.4f}"
    )

    plot_dir = out_dir / "plots"
    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text(
        _format_metrics_text(
            loss=loss,
            metrics=metrics,
            test_files=test_files,
            validation_files=validation_files,
            test_start_time=test_start_time,
            training_seed=args.seed,
            holdout_seed=holdout_seed,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics: {metrics_path}")
    _write_split_manifest(plot_dir / "excluded_validation_file_split.csv", "excluded_validation", validation_files)
    _write_split_manifest(plot_dir / "holdout_test_file_split.csv", "holdout_test", test_files)
    _save_good_curtain_plots(
        out_dir=plot_dir,
        model=model,
        files=test_files,
        x_mean=x_mean,
        x_std=x_std,
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
