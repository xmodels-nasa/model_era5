#!/usr/bin/env python3
"""Find and plot sparse cloudy-sky test windows where tolerance IoU improves.

The output is a folder containing one curtain-panel PNG per selected
consecutive window, plus CSV/NPZ/JSON data for the same window. Horizontally,
plots follow consecutive data points from one test file. Vertically, they show
the 40 cloud mask bins.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from distance_metrics_common import (
    AuroraRawChipClassifier,
    EmbeddingTransformerClassifier,
    RESULTS_DIR,
    emb_train,
    embedding_validation_files,
    raw_train,
    raw_validation_files,
    saved_threshold,
    strict_iou,
    tolerance_iou,
    torch_load,
)


MODEL_ORDER = [
    "Fine-tune Transformer",
    "Fine-tune MLP",
    "U-Net raw chips",
    "Aurora raw chips",
]
FINE_TUNED_MODELS = ["Fine-tune Transformer", "Fine-tune MLP"]
RAW_CHIP_MODELS = ["U-Net raw chips", "Aurora raw chips"]


@dataclass
class ModelBundle:
    name: str
    kind: str
    model: torch.nn.Module
    threshold: float
    stats: Optional[raw_train.NormalizationStats] = None
    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    use_base_features: bool = True


@dataclass
class Candidate:
    file_stem: str
    file_time_utc: str
    raw_meta: raw_train.FileMeta
    emb_meta: emb_train.FileMeta
    start: int
    stop: int
    first_row: int
    last_row: int
    strict_gain: float
    tol1_gain: float
    tol2_gain: float
    fine_tuned_positive_count_mean: float
    raw_chip_positive_count_mean: float
    positive_count_gap: float
    matching_row_count: int
    composite_gain: float
    metrics: Dict[str, float]


def load_raw_unet(device: str, threshold: Optional[float]) -> ModelBundle:
    out_dir = RESULTS_DIR / "baseline_model_outputs"
    ckpt = torch_load(out_dir / "multilabel_unet_classifier.pt", device)
    stats_npz = np.load(out_dir / "normalization_stats.npz")
    stats = raw_train.NormalizationStats(
        chip_mean=stats_npz["chip_mean"],
        chip_std=stats_npz["chip_std"],
        base_mean=stats_npz["base_mean"],
        base_std=stats_npz["base_std"],
    )
    base_feature_dim = len(ckpt.get("base_features", raw_train.BASE_FEATURE_COLUMNS))
    model = raw_train.UNetClassifier(
        in_channels=int(ckpt["input_channels"]),
        output_dim=int(ckpt.get("output_dim", len(raw_train.TARGET_COLUMNS))),
        base_feature_dim=base_feature_dim,
        base_channels=int(ckpt.get("base_channels", 32)),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ModelBundle(
        name="U-Net raw chips",
        kind="raw",
        model=model,
        threshold=saved_threshold(ckpt, "test", threshold),
        stats=stats,
        use_base_features=base_feature_dim > 0,
    )


def load_raw_aurora(device: str, threshold: Optional[float]) -> ModelBundle:
    out_dir = RESULTS_DIR / "baseline_model_outputs_aurora"
    ckpt = torch_load(out_dir / "multilabel_aurora_rawchip_classifier.pt", device)
    stats_npz = np.load(out_dir / "normalization_stats.npz")
    stats = raw_train.NormalizationStats(
        chip_mean=stats_npz["chip_mean"],
        chip_std=stats_npz["chip_std"],
        base_mean=stats_npz["base_mean"],
        base_std=stats_npz["base_std"],
    )
    base_feature_dim = len(ckpt.get("base_features", raw_train.BASE_FEATURE_COLUMNS))
    model = AuroraRawChipClassifier(
        dynamic_channel_names=ckpt["dynamic_channel_names"],
        static_channel_names=ckpt["static_channel_names"],
        output_dim=int(ckpt.get("output_dim", len(raw_train.TARGET_COLUMNS))),
        base_feature_dim=base_feature_dim,
        history_size=2,
        patch_size=int(ckpt["patch_size"]),
        latent_levels=int(ckpt["latent_levels"]),
        embed_dim=int(ckpt["embed_dim"]),
        num_heads=int(ckpt["num_heads"]),
        head_dim=int(ckpt["head_dim"]),
        perceiver_depth=int(ckpt["perceiver_depth"]),
        transformer_depth=int(ckpt["transformer_depth"]),
        mlp_ratio=float(ckpt["mlp_ratio"]),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ModelBundle(
        name="Aurora raw chips",
        kind="raw",
        model=model,
        threshold=saved_threshold(ckpt, "test", threshold),
        stats=stats,
        use_base_features=base_feature_dim > 0,
    )


def load_embedding_mlp(device: str, threshold: Optional[float]) -> ModelBundle:
    out_dir = RESULTS_DIR / "model_outputs_fine_tune"
    ckpt = torch_load(out_dir / "multilabel_mlp.pt", device)
    stats_npz = np.load(out_dir / "feature_stats.npz")
    model = emb_train.MultiLabelMLP(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", len(emb_train.TARGET_COLUMNS))),
        hidden_dims=ckpt.get("hidden_dims"),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ModelBundle(
        name="Fine-tune MLP",
        kind="embedding",
        model=model,
        threshold=saved_threshold(ckpt, "test", threshold),
        x_mean=stats_npz["x_mean"],
        x_std=stats_npz["x_std"],
    )


def load_embedding_transformer(device: str, threshold: Optional[float]) -> ModelBundle:
    out_dir = RESULTS_DIR / "model_outputs_transformer"
    ckpt = torch_load(out_dir / "multilabel_transformer.pt", device)
    stats_npz = np.load(out_dir / "feature_stats.npz")
    model = EmbeddingTransformerClassifier(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", len(emb_train.TARGET_COLUMNS))),
        hidden_dims=ckpt.get("transformer_config"),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ModelBundle(
        name="Fine-tune Transformer",
        kind="embedding",
        model=model,
        threshold=saved_threshold(ckpt, "test", threshold),
        x_mean=stats_npz["x_mean"],
        x_std=stats_npz["x_std"],
    )


def load_bundles(device: str, threshold: Optional[float]) -> Dict[str, ModelBundle]:
    bundles = [
        load_embedding_transformer(device, threshold),
        load_embedding_mlp(device, threshold),
        load_raw_unet(device, threshold),
        load_raw_aurora(device, threshold),
    ]
    return {b.name: b for b in bundles}


def predict_one_file(
    bundle: ModelBundle,
    raw_meta: raw_train.FileMeta,
    emb_meta: emb_train.FileMeta,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bundle.kind == "raw":
        assert bundle.stats is not None
        return raw_train._predict_masks_for_file(
            model=bundle.model,
            meta=raw_meta,
            stats=bundle.stats,
            use_base_features=bundle.use_base_features,
            eval_batch_size=batch_size,
            device=device,
        )
    assert bundle.x_mean is not None and bundle.x_std is not None
    return emb_train._predict_masks_for_file(
        model=bundle.model,
        meta=emb_meta,
        x_mean=bundle.x_mean,
        x_std=bundle.x_std,
        eval_batch_size=batch_size,
        device=device,
    )


def align_predictions(
    per_model: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    common_rows = None
    for rows, _, _ in per_model.values():
        row_set = set(int(v) for v in rows.tolist())
        common_rows = row_set if common_rows is None else common_rows & row_set
    if not common_rows:
        raise ValueError("No common row indices across model predictions.")
    rows = np.asarray(sorted(common_rows), dtype=np.int64)

    aligned_targets: Optional[np.ndarray] = None
    aligned_probs: Dict[str, np.ndarray] = {}
    for model_name, (model_rows, targets, probs) in per_model.items():
        index = {int(row): idx for idx, row in enumerate(model_rows.tolist())}
        take = np.asarray([index[int(row)] for row in rows], dtype=np.int64)
        model_targets = targets[take].astype(np.float32, copy=False)
        if aligned_targets is None:
            aligned_targets = model_targets
        elif not np.array_equal(aligned_targets, model_targets):
            raise ValueError(f"Targets are not aligned for {model_name}.")
        aligned_probs[model_name] = probs[take].astype(np.float32, copy=False)
    assert aligned_targets is not None
    return rows, aligned_targets, aligned_probs


def per_row_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    pred_i = (preds > 0.5).astype(np.uint8, copy=False)
    target_i = (targets > 0.5).astype(np.uint8, copy=False)
    return {
        "strict": strict_iou(pred_i, target_i),
        "tol1": tolerance_iou(pred_i, target_i, radius=1),
        "tol2": tolerance_iou(pred_i, target_i, radius=2),
    }


def consecutive_runs(rows: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = 0
    for idx in range(1, rows.size + 1):
        if idx == rows.size or rows[idx] != rows[idx - 1] + 1:
            runs.append((start, idx))
            start = idx
    return runs


def window_metric_mean(metrics: Dict[str, Dict[str, np.ndarray]], names: List[str], key: str, start: int, stop: int) -> float:
    values = [float(metrics[name][key][start:stop].mean()) for name in names]
    return float(np.mean(values))


def positive_count_stats(classes: Dict[str, np.ndarray], start: int, stop: int) -> Dict[str, np.ndarray | float]:
    fine_counts = np.stack([classes[name][start:stop].sum(axis=1) for name in FINE_TUNED_MODELS], axis=1).astype(np.float32)
    raw_counts = np.stack([classes[name][start:stop].sum(axis=1) for name in RAW_CHIP_MODELS], axis=1).astype(np.float32)
    fine_mean_by_row = fine_counts.mean(axis=1)
    raw_mean_by_row = raw_counts.mean(axis=1)
    return {
        "fine_min_by_row": fine_counts.min(axis=1),
        "raw_min_by_row": raw_counts.min(axis=1),
        "fine_mean_by_row": fine_mean_by_row,
        "raw_mean_by_row": raw_mean_by_row,
        "gap_by_row": np.abs(fine_mean_by_row - raw_mean_by_row),
        "fine_mean": float(fine_mean_by_row.mean()),
        "raw_mean": float(raw_mean_by_row.mean()),
        "gap": float(np.abs(fine_mean_by_row - raw_mean_by_row).mean()),
    }


def row_match_mask(
    targets: np.ndarray,
    classes: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, np.ndarray]],
    start: int,
    stop: int,
    args: argparse.Namespace,
) -> np.ndarray:
    target_counts = targets[start:stop].sum(axis=1)
    valid_target_count = (target_counts >= args.min_target_ones) & (target_counts <= args.max_target_ones)
    strict_ft = np.mean([metrics[name]["strict"][start:stop] for name in FINE_TUNED_MODELS], axis=0)
    strict_raw = np.mean([metrics[name]["strict"][start:stop] for name in RAW_CHIP_MODELS], axis=0)
    tol1_ft = np.mean([metrics[name]["tol1"][start:stop] for name in FINE_TUNED_MODELS], axis=0)
    tol1_raw = np.mean([metrics[name]["tol1"][start:stop] for name in RAW_CHIP_MODELS], axis=0)
    tol2_ft = np.mean([metrics[name]["tol2"][start:stop] for name in FINE_TUNED_MODELS], axis=0)
    tol2_raw = np.mean([metrics[name]["tol2"][start:stop] for name in RAW_CHIP_MODELS], axis=0)
    strict_gain = strict_ft - strict_raw
    tol1_gain = tol1_ft - tol1_raw
    tol2_gain = tol2_ft - tol2_raw
    count_stats = positive_count_stats(classes, start, stop)
    return (
        valid_target_count
        & (strict_gain >= args.strict_gain_min)
        & (strict_gain <= args.strict_gain_max)
        & (tol1_gain >= args.min_tolerance_gain)
        & (tol2_gain >= args.min_tolerance_gain)
        & (count_stats["fine_min_by_row"] >= args.min_pred_ones)
        & (count_stats["raw_min_by_row"] >= args.min_pred_ones)
        & (count_stats["fine_mean_by_row"] <= args.max_pred_ones)
        & (count_stats["raw_mean_by_row"] <= args.max_pred_ones)
        & (count_stats["gap_by_row"] <= args.max_pred_count_gap)
    )


def scan_candidates_for_file(
    raw_meta: raw_train.FileMeta,
    emb_meta: emb_train.FileMeta,
    bundles: Dict[str, ModelBundle],
    args: argparse.Namespace,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    per_model = {
        name: predict_one_file(bundle, raw_meta, emb_meta, args.batch_size, args.device)
        for name, bundle in bundles.items()
    }
    rows, targets, probs = align_predictions(per_model)
    classes = {
        name: (probs[name] >= bundles[name].threshold).astype(np.uint8, copy=False)
        for name in MODEL_ORDER
    }
    metrics = {name: per_row_metrics(classes[name], targets) for name in MODEL_ORDER}
    candidates: List[Candidate] = []
    for run_start, run_stop in consecutive_runs(rows):
        if run_stop - run_start < args.window_size:
            continue
        for start in range(run_start, run_stop - args.window_size + 1, args.window_stride):
            stop = start + args.window_size
            match_mask = row_match_mask(targets, classes, metrics, start, stop, args)
            matching_row_count = int(match_mask.sum())
            if matching_row_count < args.min_matching_rows:
                continue
            strict_ft = window_metric_mean(metrics, FINE_TUNED_MODELS, "strict", start, stop)
            strict_raw = window_metric_mean(metrics, RAW_CHIP_MODELS, "strict", start, stop)
            tol1_ft = window_metric_mean(metrics, FINE_TUNED_MODELS, "tol1", start, stop)
            tol1_raw = window_metric_mean(metrics, RAW_CHIP_MODELS, "tol1", start, stop)
            tol2_ft = window_metric_mean(metrics, FINE_TUNED_MODELS, "tol2", start, stop)
            tol2_raw = window_metric_mean(metrics, RAW_CHIP_MODELS, "tol2", start, stop)
            strict_gain = strict_ft - strict_raw
            tol1_gain = tol1_ft - tol1_raw
            tol2_gain = tol2_ft - tol2_raw
            if strict_gain < args.strict_gain_min or strict_gain > args.strict_gain_max:
                continue
            if tol1_gain < args.min_tolerance_gain or tol2_gain < args.min_tolerance_gain:
                continue
            count_stats = positive_count_stats(classes, start, stop)
            if count_stats["fine_mean"] < args.min_pred_ones or count_stats["raw_mean"] < args.min_pred_ones:
                continue
            if count_stats["fine_mean"] > args.max_pred_ones or count_stats["raw_mean"] > args.max_pred_ones:
                continue
            if count_stats["gap"] > args.max_pred_count_gap:
                continue
            if args.require_each_finetune_better:
                raw_best_strict = max(float(metrics[n]["strict"][start:stop].mean()) for n in RAW_CHIP_MODELS)
                raw_best_tol1 = max(float(metrics[n]["tol1"][start:stop].mean()) for n in RAW_CHIP_MODELS)
                raw_best_tol2 = max(float(metrics[n]["tol2"][start:stop].mean()) for n in RAW_CHIP_MODELS)
                each_ok = all(
                    float(metrics[n]["strict"][start:stop].mean()) > raw_best_strict
                    and float(metrics[n]["tol1"][start:stop].mean()) > raw_best_tol1
                    and float(metrics[n]["tol2"][start:stop].mean()) > raw_best_tol2
                    for n in FINE_TUNED_MODELS
                )
                if not each_ok:
                    continue
            model_metrics = {}
            for name in MODEL_ORDER:
                for key in ("strict", "tol1", "tol2"):
                    model_metrics[f"{name} {key}"] = float(metrics[name][key][start:stop].mean())
            candidates.append(
                Candidate(
                    file_stem=raw_meta.source_file.stem,
                    file_time_utc=str(raw_meta.file_time),
                    raw_meta=raw_meta,
                    emb_meta=emb_meta,
                    start=start,
                    stop=stop,
                    first_row=int(rows[start]),
                    last_row=int(rows[stop - 1]),
                    strict_gain=float(strict_gain),
                    tol1_gain=float(tol1_gain),
                    tol2_gain=float(tol2_gain),
                    fine_tuned_positive_count_mean=float(count_stats["fine_mean"]),
                    raw_chip_positive_count_mean=float(count_stats["raw_mean"]),
                    positive_count_gap=float(count_stats["gap"]),
                    matching_row_count=matching_row_count,
                    composite_gain=float(tol1_gain + tol2_gain - abs(strict_gain) + 0.02 * matching_row_count),
                    metrics=model_metrics,
                )
            )
    cache = {
        "rows": rows,
        "targets": targets,
        "probs": probs,
        "classes": classes,
        "metrics": metrics,
    }
    return candidates, cache


def load_geo_for_rows(feather_path: Path, rows: np.ndarray) -> pd.DataFrame:
    columns = ["Latitude_0", "Longitude_0", "timestamp_0"]
    try:
        df = pd.read_feather(feather_path, columns=columns)
        geo = df.iloc[rows][columns].copy()
    except Exception:
        geo = pd.DataFrame(index=np.arange(rows.size), columns=columns)
    geo.insert(0, "row_index", rows.astype(np.int64))
    return geo.reset_index(drop=True)


def save_candidate_outputs(
    idx: int,
    candidate: Candidate,
    cache: Dict[str, Any],
    bundles: Dict[str, ModelBundle],
    out_dir: Path,
) -> Dict[str, Any]:
    rows = cache["rows"][candidate.start : candidate.stop]
    targets = cache["targets"][candidate.start : candidate.stop]
    probs = {name: cache["probs"][name][candidate.start : candidate.stop] for name in MODEL_ORDER}
    classes = {name: cache["classes"][name][candidate.start : candidate.stop] for name in MODEL_ORDER}
    metrics = {name: {k: v[candidate.start : candidate.stop] for k, v in cache["metrics"][name].items()} for name in MODEL_ORDER}

    prefix = f"window_{idx:03d}_{candidate.file_stem}_rows_{candidate.first_row:06d}_{candidate.last_row:06d}"
    png_path = out_dir / f"{prefix}.png"
    csv_path = out_dir / f"{prefix}_points.csv"
    npz_path = out_dir / f"{prefix}_masks.npz"
    json_path = out_dir / f"{prefix}_metrics.json"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=1 + len(MODEL_ORDER),
        figsize=(16, max(6, rows.size * 0.18)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = [("Ground truth", targets)]
    panels.extend((name, classes[name]) for name in MODEL_ORDER)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image.T, aspect="auto", interpolation="nearest", cmap="gray_r", vmin=0, vmax=1, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("data point")
    axes[0].set_ylabel("40-bin cloud mask index")
    for ax, name in zip(axes[1:], MODEL_ORDER):
        ax.set_title(
            f"{name} | strict={metrics[name]['strict'].mean():.3f}, "
            f"tol@1={metrics[name]['tol1'].mean():.3f}, tol@2={metrics[name]['tol2'].mean():.3f}, "
            f"thr={bundles[name].threshold:.2f}"
        )
    x_ticks = np.linspace(0, rows.size - 1, num=min(6, rows.size), dtype=int)
    for ax in axes:
        ax.set_xticks(x_ticks, labels=[str(int(rows[i])) for i in x_ticks], rotation=30, ha="right")
    fig.suptitle(
        f"{candidate.file_stem} | rows {candidate.first_row}-{candidate.last_row} | "
        f"fine-tuned minus raw gains: strict={candidate.strict_gain:+.3f}, "
        f"tol@1={candidate.tol1_gain:+.3f}, tol@2={candidate.tol2_gain:+.3f} | "
        f"predicted ones mean: fine-tuned={candidate.fine_tuned_positive_count_mean:.1f}, "
        f"raw={candidate.raw_chip_positive_count_mean:.1f}",
        fontsize=12,
    )
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    point_df = load_geo_for_rows(candidate.emb_meta.feather_path, rows)
    for name in MODEL_ORDER:
        safe = name.lower().replace(" ", "_").replace("-", "")
        point_df[f"{safe}_strict_iou"] = metrics[name]["strict"]
        point_df[f"{safe}_tol1_iou"] = metrics[name]["tol1"]
        point_df[f"{safe}_tol2_iou"] = metrics[name]["tol2"]
    point_df.to_csv(csv_path, index=False)

    npz_payload: Dict[str, Any] = {"rows": rows, "targets": targets.astype(np.uint8)}
    for name in MODEL_ORDER:
        safe = name.lower().replace(" ", "_").replace("-", "")
        npz_payload[f"{safe}_probs"] = probs[name].astype(np.float32)
        npz_payload[f"{safe}_preds"] = classes[name].astype(np.uint8)
    np.savez_compressed(npz_path, **npz_payload)

    metrics_payload = {
        "index": idx,
        "file_stem": candidate.file_stem,
        "file_time_utc": candidate.file_time_utc,
        "first_row": candidate.first_row,
        "last_row": candidate.last_row,
        "window_size": int(rows.size),
        "strict_gain": candidate.strict_gain,
        "tolerance_iou_1_gain": candidate.tol1_gain,
        "tolerance_iou_2_gain": candidate.tol2_gain,
        "fine_tuned_positive_count_mean": candidate.fine_tuned_positive_count_mean,
        "raw_chip_positive_count_mean": candidate.raw_chip_positive_count_mean,
        "positive_count_gap": candidate.positive_count_gap,
        "matching_row_count": candidate.matching_row_count,
        "composite_gain": candidate.composite_gain,
        "model_metrics": candidate.metrics,
        "png_path": str(png_path),
        "csv_path": str(csv_path),
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "cloudy_sky_mask_visualizations")
    parser.add_argument("--max-windows", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--min-matching-rows", type=int, default=8)
    parser.add_argument("--min-target-ones", type=int, default=3)
    parser.add_argument("--max-target-ones", type=int, default=10)
    parser.add_argument("--min-pred-ones", type=int, default=1)
    parser.add_argument("--max-pred-ones", type=int, default=12)
    parser.add_argument("--max-pred-count-gap", type=float, default=3.0)
    parser.add_argument("--strict-gain-min", type=float, default=-0.08)
    parser.add_argument("--strict-gain-max", type=float, default=0.08)
    parser.add_argument("--min-tolerance-gain", type=float, default=0.15)
    parser.add_argument("--require-each-finetune-better", action="store_true")
    parser.add_argument("--threshold", type=float, default=None, help="Override all saved model thresholds.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feather-root", type=Path, default=Path(emb_train.FEATHER_ROOT))
    parser.add_argument("--raw-chips-dir", type=Path, default=Path(raw_train.RAW_CHIPS_DIR))
    parser.add_argument("--embedding-dir", type=Path, default=Path(emb_train.EMBEDDING_DIR))
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for debugging.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1")
    if args.window_stride < 1:
        raise ValueError("--window-stride must be >= 1")
    if args.min_matching_rows < 1 or args.min_matching_rows > args.window_size:
        raise ValueError("--min-matching-rows must be in [1, --window-size]")
    if args.max_target_ones < args.min_target_ones:
        raise ValueError("--max-target-ones must be >= --min-target-ones")
    if args.max_pred_ones < args.min_pred_ones:
        raise ValueError("--max-pred-ones must be >= --min-pred-ones")
    if args.strict_gain_max < args.strict_gain_min:
        raise ValueError("--strict-gain-max must be >= --strict-gain-min")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_bundles(args.device, args.threshold)
    raw_files = raw_validation_files(
        RESULTS_DIR / "baseline_model_outputs" / "file_split.csv",
        "test",
        args.feather_root,
        args.raw_chips_dir,
    )
    emb_files = embedding_validation_files(
        RESULTS_DIR / "model_outputs_fine_tune" / "file_split.csv",
        "test",
        args.feather_root,
        args.embedding_dir,
    )
    raw_by_stem = {m.source_file.stem: m for m in raw_files}
    emb_by_stem = {m.feather_path.stem: m for m in emb_files}
    stems = sorted(set(raw_by_stem) & set(emb_by_stem), key=lambda s: (raw_by_stem[s].file_time, s))
    if args.max_files is not None:
        stems = stems[: args.max_files]
    print(f"Scanning {len(stems)} aligned test files.")

    summary_rows = []
    for file_idx, stem in enumerate(stems, start=1):
        print(f"[{file_idx}/{len(stems)}] Predicting and scanning {stem}")
        raw_meta = raw_by_stem[stem]
        emb_meta = emb_by_stem[stem]
        candidates, cache = scan_candidates_for_file(raw_meta, emb_meta, bundles, args)
        candidates.sort(key=lambda c: c.composite_gain, reverse=True)
        print(f"  candidates in file: {len(candidates)}")
        for candidate in candidates:
            if len(summary_rows) >= args.max_windows:
                break
            idx = len(summary_rows) + 1
            payload = save_candidate_outputs(idx, candidate, cache, bundles, args.output_dir)
            summary_rows.append(payload)
            print(f"Saved window {idx:03d}: {payload['png_path']}")
        if len(summary_rows) >= args.max_windows:
            print(f"Reached requested max windows: {args.max_windows}")
            break

    if len(summary_rows) < args.max_windows:
        print(f"Only found {len(summary_rows)} matching windows; requested {args.max_windows}.")

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "index",
            "file_stem",
            "file_time_utc",
            "first_row",
            "last_row",
            "window_size",
            "strict_gain",
            "tolerance_iou_1_gain",
            "tolerance_iou_2_gain",
            "fine_tuned_positive_count_mean",
            "raw_chip_positive_count_mean",
            "positive_count_gap",
            "matching_row_count",
            "composite_gain",
            "png_path",
            "csv_path",
            "npz_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Saved summary: {summary_csv}")
    print(f"Saved {len(summary_rows)} visualization window(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
