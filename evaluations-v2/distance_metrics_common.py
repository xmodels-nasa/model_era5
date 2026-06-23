#!/usr/bin/env python3
"""Saved-model evaluators and distance-aware metrics for 40-bin cloud masks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
if THIS_DIR.parent.name == "results-v2":
    RESULTS_DIR = THIS_DIR.parent
    PROJECT_ROOT = RESULTS_DIR.parent
else:
    PROJECT_ROOT = THIS_DIR.parent
    RESULTS_DIR = PROJECT_ROOT / "results-v2"
BASELINE_DIR = PROJECT_ROOT / "baseline_model"
FINE_TUNED_DIR = PROJECT_ROOT / "fine_tuned_model"

for path in (BASELINE_DIR, FINE_TUNED_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_multilabel_from_raw_chips as raw_train  # noqa: E402
import train_multilabel_from_feather_embeddings as emb_train  # noqa: E402
from train_multilable_from_rawchips_aurora_architecturer import (  # noqa: E402
    AuroraRawChipClassifier,
)
from train_multilabel_from_feather_embeddings_transformer import (  # noqa: E402
    EmbeddingTransformerClassifier,
)


@dataclass(frozen=True)
class EvalConfig:
    model_name: str
    model_kind: str
    default_output_dir: Path
    default_model_path: Path
    default_stats_path: Path
    default_split_path: Path


def torch_load(path: Path, device: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _path_by_stem(root: Path, suffix: str) -> Dict[str, Path]:
    return {p.stem: p for p in root.glob(f"*{suffix}") if p.is_file()}


def _split_rows(split_path: Path, split_name: str) -> List[Dict[str, str]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {split_path}")
    rows: List[Dict[str, str]] = []
    with split_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == split_name:
                rows.append(row)
    if not rows:
        raise ValueError(f"No {split_name} rows found in {split_path}")
    return rows


def _timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def raw_validation_files(
    split_path: Path,
    split_name: str,
    feather_root: Path,
    raw_chips_dir: Path,
) -> List[raw_train.FileMeta]:
    feathers = _path_by_stem(feather_root, ".feather")
    chips = _path_by_stem(raw_chips_dir, ".npz")
    metas: List[raw_train.FileMeta] = []
    missing: List[str] = []
    for row in _split_rows(split_path, split_name):
        stem = Path(row["file"]).stem
        manifest_feather = Path(row["file"])
        manifest_npz = Path(row.get("npz", ""))
        feather_path = manifest_feather if manifest_feather.is_file() else feathers.get(stem)
        npz_path = manifest_npz if manifest_npz.is_file() else chips.get(stem)
        if feather_path is None or npz_path is None:
            missing.append(stem)
            continue
        metas.append(
            raw_train.FileMeta(
                source_file=feather_path,
                npz_path=npz_path,
                file_time=_timestamp(row["file_time_utc"]),
            )
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} {split_name} file(s) locally. "
            f"First missing stems: {preview}. "
            f"Searched feather_root={feather_root} and raw_chips_dir={raw_chips_dir}; "
            "also tried absolute paths from the split manifest."
        )
    return metas


def embedding_validation_files(
    split_path: Path,
    split_name: str,
    feather_root: Path,
    embedding_dir: Path,
) -> List[emb_train.FileMeta]:
    feathers = _path_by_stem(feather_root, ".feather")
    embeddings = _path_by_stem(embedding_dir, ".npz")
    metas: List[emb_train.FileMeta] = []
    missing: List[str] = []
    for row in _split_rows(split_path, split_name):
        stem = Path(row["file"]).stem
        manifest_feather = Path(row["file"])
        feather_path = manifest_feather if manifest_feather.is_file() else feathers.get(stem)
        npz_path = embeddings.get(stem)
        if feather_path is None or npz_path is None:
            missing.append(stem)
            continue
        metas.append(
            emb_train.FileMeta(
                feather_path=feather_path,
                npz_path=npz_path,
                file_time=_timestamp(row["file_time_utc"]),
            )
        )
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} {split_name} file(s) locally. "
            f"First missing stems: {preview}. "
            f"Searched feather_root={feather_root} and embedding_dir={embedding_dir}; "
            "also tried absolute Feather paths from the split manifest."
        )
    return metas


def strict_iou(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(preds == 1, targets == 1).sum(axis=1).astype(np.float32)
    union = np.logical_or(preds == 1, targets == 1).sum(axis=1).astype(np.float32)
    return np.where(union > 0, intersection / np.maximum(union, 1.0), 1.0)


def tolerance_iou(preds: np.ndarray, targets: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros(preds.shape[0], dtype=np.float32)
    for i in range(preds.shape[0]):
        p_idx = np.flatnonzero(preds[i] == 1)
        t_idx = np.flatnonzero(targets[i] == 1)
        if t_idx.size == 0:
            out[i] = 1.0 if p_idx.size == 0 else 0.0
            continue
        if p_idx.size == 0:
            out[i] = 0.0
            continue
        pred_close = np.min(np.abs(p_idx[:, None] - t_idx[None, :]), axis=1) <= radius
        truth_close = np.min(np.abs(t_idx[:, None] - p_idx[None, :]), axis=1) <= radius
        soft_tp = float(pred_close.sum())
        far_fp = float((~pred_close).sum())
        far_fn = float((~truth_close).sum())
        denom = soft_tp + far_fp + far_fn
        out[i] = soft_tp / denom if denom > 0 else 1.0
    return out


def _gaussian_similarity(width: int, sigma: float) -> np.ndarray:
    idx = np.arange(width, dtype=np.float32)
    dist2 = np.square(idx[:, None] - idx[None, :])
    return np.exp(-dist2 / (2.0 * float(sigma) ** 2)).astype(np.float32)


def gaussian_smoothed_iou(preds: np.ndarray, targets: np.ndarray, sigma: float) -> np.ndarray:
    sim = _gaussian_similarity(preds.shape[1], sigma)
    pred_soft = np.clip(preds.astype(np.float32) @ sim, 0.0, 1.0)
    target_soft = np.clip(targets.astype(np.float32) @ sim, 0.0, 1.0)
    numerator = np.minimum(pred_soft, target_soft).sum(axis=1)
    denominator = np.maximum(pred_soft, target_soft).sum(axis=1)
    values = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    empty_truth = targets.sum(axis=1) == 0
    empty_pred = preds.sum(axis=1) == 0
    values[empty_truth] = np.where(empty_pred[empty_truth], 1.0, 0.0)
    return values.astype(np.float32, copy=False)


def distance_weighted_iou(preds: np.ndarray, targets: np.ndarray, sigma: float) -> np.ndarray:
    sim = _gaussian_similarity(preds.shape[1], sigma)
    out = np.zeros(preds.shape[0], dtype=np.float32)
    for i in range(preds.shape[0]):
        p_idx = np.flatnonzero(preds[i] == 1)
        t_idx = np.flatnonzero(targets[i] == 1)
        if t_idx.size == 0:
            out[i] = 1.0 if p_idx.size == 0 else 0.0
            continue
        if p_idx.size == 0:
            out[i] = 0.0
            continue
        pred_credit = sim[np.ix_(p_idx, t_idx)].max(axis=1).sum()
        truth_credit = sim[np.ix_(t_idx, p_idx)].max(axis=1).sum()
        soft_intersection = float(min(pred_credit, truth_credit))
        false_positive_penalty = float(p_idx.size - pred_credit)
        false_negative_penalty = float(t_idx.size - truth_credit)
        denom = soft_intersection + false_positive_penalty + false_negative_penalty
        out[i] = soft_intersection / denom if denom > 0 else 1.0
    return out


def _safe_mean(values: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    selected = values if mask is None else values[mask]
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected))


def extended_iou_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    preds_i = (preds > 0.5).astype(np.uint8, copy=False)
    targets_i = (targets > 0.5).astype(np.uint8, copy=False)
    empty_truth = targets_i.sum(axis=1) == 0
    nonempty_truth = ~empty_truth
    values = {
        "strict_iou": strict_iou(preds_i, targets_i),
        "tolerance_iou_1": tolerance_iou(preds_i, targets_i, radius=1),
        "tolerance_iou_2": tolerance_iou(preds_i, targets_i, radius=2),
        "gaussian_smoothed_iou_sigma_1": gaussian_smoothed_iou(preds_i, targets_i, sigma=1.0),
        "gaussian_smoothed_iou_sigma_2": gaussian_smoothed_iou(preds_i, targets_i, sigma=2.0),
        "distance_weighted_iou_sigma_1": distance_weighted_iou(preds_i, targets_i, sigma=1.0),
        "distance_weighted_iou_sigma_2": distance_weighted_iou(preds_i, targets_i, sigma=2.0),
    }
    summary: Dict[str, Any] = {
        "sample_count": int(targets_i.shape[0]),
        "empty_truth_count": int(empty_truth.sum()),
        "nonempty_truth_count": int(nonempty_truth.sum()),
        "pred_empty_count": int((preds_i.sum(axis=1) == 0).sum()),
        "pred_nonempty_count": int((preds_i.sum(axis=1) > 0).sum()),
    }
    for name, metric_values in values.items():
        summary[f"{name}_mean"] = _safe_mean(metric_values)
        summary[f"{name}_empty_truth_mean"] = _safe_mean(metric_values, empty_truth)
        summary[f"{name}_nonempty_truth_mean"] = _safe_mean(metric_values, nonempty_truth)
    return summary


def saved_threshold(ckpt: Dict[str, Any], split_name: str, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    split_metrics = ckpt.get(f"{split_name}_metrics")
    if isinstance(split_metrics, dict) and split_metrics.get("iou_threshold") is not None:
        return float(split_metrics["iou_threshold"])
    validation_metrics = ckpt.get("validation_metrics")
    if isinstance(validation_metrics, dict) and validation_metrics.get("iou_threshold") is not None:
        return float(validation_metrics["iou_threshold"])
    test_metrics = ckpt.get("test_metrics")
    if isinstance(test_metrics, dict) and test_metrics.get("iou_threshold") is not None:
        return float(test_metrics["iou_threshold"])
    return 0.5


def _write_outputs(out_json: Path, metrics: Dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in sorted(metrics):
            writer.writerow([key, metrics[key]])
    print(f"Saved metrics JSON: {out_json}")
    print(f"Saved metrics CSV:  {out_csv}")


def _print_summary(metrics: Dict[str, Any]) -> None:
    keys = [
        "strict_iou_mean",
        "strict_iou_nonempty_truth_mean",
        "tolerance_iou_1_nonempty_truth_mean",
        "tolerance_iou_2_nonempty_truth_mean",
        "gaussian_smoothed_iou_sigma_1_nonempty_truth_mean",
        "gaussian_smoothed_iou_sigma_2_nonempty_truth_mean",
        "distance_weighted_iou_sigma_1_nonempty_truth_mean",
        "distance_weighted_iou_sigma_2_nonempty_truth_mean",
    ]
    print(f"Model: {metrics['model_name']}")
    print(f"Split: {metrics['split']}")
    print(f"Samples: {metrics['sample_count']}")
    for key in keys:
        print(f"{key}={metrics[key]:.6f}")


def evaluate_raw_unet(args: argparse.Namespace, config: EvalConfig) -> Dict[str, Any]:
    ckpt = torch_load(args.model_path, args.device)
    stats_npz = np.load(args.stats_path)
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
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = saved_threshold(ckpt, args.split, args.threshold)
    files = raw_validation_files(args.split_path, args.split, args.feather_root, args.raw_chips_dir)
    _, base_metrics, details = raw_train._evaluate_files(
        model=model,
        loss_fn=nn.BCEWithLogitsLoss(),
        files=files,
        batch_size=args.batch_size,
        device=args.device,
        stats=stats,
        use_base_features=base_feature_dim > 0,
        sample_ratio=1.0,
        max_samples_per_file=None,
        seed=args.seed,
        iou_threshold=threshold,
        collect_details=True,
    )
    assert details is not None
    ext = extended_iou_metrics(details["preds"].numpy(), details["targets"].numpy())
    return {**base_metrics, **ext, "model_name": config.model_name, "threshold": threshold}


def evaluate_raw_aurora(args: argparse.Namespace, config: EvalConfig) -> Dict[str, Any]:
    ckpt = torch_load(args.model_path, args.device)
    stats_npz = np.load(args.stats_path)
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
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = saved_threshold(ckpt, args.split, args.threshold)
    files = raw_validation_files(args.split_path, args.split, args.feather_root, args.raw_chips_dir)
    _, base_metrics, details = raw_train._evaluate_files(
        model=model,
        loss_fn=nn.BCEWithLogitsLoss(),
        files=files,
        batch_size=args.batch_size,
        device=args.device,
        stats=stats,
        use_base_features=base_feature_dim > 0,
        sample_ratio=1.0,
        max_samples_per_file=None,
        seed=args.seed,
        iou_threshold=threshold,
        collect_details=True,
    )
    assert details is not None
    ext = extended_iou_metrics(details["preds"].numpy(), details["targets"].numpy())
    return {**base_metrics, **ext, "model_name": config.model_name, "threshold": threshold}


def _evaluate_embedding_arrays(
    args: argparse.Namespace,
    config: EvalConfig,
    model: nn.Module,
    ckpt: Dict[str, Any],
) -> Dict[str, Any]:
    stats_npz = np.load(args.stats_path)
    x_mean = stats_npz["x_mean"]
    x_std = stats_npz["x_std"]
    threshold = saved_threshold(ckpt, args.split, args.threshold)
    files = embedding_validation_files(args.split_path, args.split, args.feather_root, args.embedding_dir)
    x_val, y_val = emb_train.load_dataset(files, sample_ratio=1.0, max_samples_per_file=None, seed=args.seed)
    x_val = (x_val - x_mean) / x_std
    _, base_metrics, details = emb_train._evaluate_in_batches(
        model=model,
        loss_fn=nn.BCEWithLogitsLoss(),
        x=torch.from_numpy(x_val.astype(np.float32, copy=False)),
        y=torch.from_numpy(y_val.astype(np.float32, copy=False)),
        batch_size=args.batch_size,
        device=args.device,
        iou_threshold=threshold,
        collect_details=True,
    )
    assert details is not None
    ext = extended_iou_metrics(details["preds"].numpy(), details["targets"].numpy())
    return {**base_metrics, **ext, "model_name": config.model_name, "threshold": threshold}


def evaluate_embedding_mlp(args: argparse.Namespace, config: EvalConfig) -> Dict[str, Any]:
    ckpt = torch_load(args.model_path, args.device)
    model = emb_train.MultiLabelMLP(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", len(emb_train.TARGET_COLUMNS))),
        hidden_dims=ckpt.get("hidden_dims"),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    return _evaluate_embedding_arrays(args, config, model, ckpt)


def evaluate_embedding_transformer(args: argparse.Namespace, config: EvalConfig) -> Dict[str, Any]:
    ckpt = torch_load(args.model_path, args.device)
    model = EmbeddingTransformerClassifier(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt.get("output_dim", len(emb_train.TARGET_COLUMNS))),
        hidden_dims=ckpt.get("transformer_config"),
        dropout=float(ckpt.get("dropout", 0.2)),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    return _evaluate_embedding_arrays(args, config, model, ckpt)


def build_parser(config: EvalConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Evaluate {config.model_name} distance-aware IoU metrics.")
    parser.add_argument("--model-output-dir", type=Path, default=config.default_output_dir)
    parser.add_argument("--model-path", type=Path, default=config.default_model_path)
    parser.add_argument("--stats-path", type=Path, default=config.default_stats_path)
    parser.add_argument("--split-path", type=Path, default=config.default_split_path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--feather-root", type=Path, default=Path(emb_train.FEATHER_ROOT))
    parser.add_argument("--raw-chips-dir", type=Path, default=Path(raw_train.RAW_CHIPS_DIR))
    parser.add_argument("--embedding-dir", type=Path, default=Path(emb_train.EMBEDDING_DIR))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def run(config: EvalConfig) -> int:
    parser = build_parser(config)
    args = parser.parse_args()
    args.model_path = args.model_path if args.model_path.is_absolute() else args.model_output_dir / args.model_path
    args.stats_path = args.stats_path if args.stats_path.is_absolute() else args.model_output_dir / args.stats_path
    args.split_path = args.split_path if args.split_path.is_absolute() else args.model_output_dir / args.split_path
    output_json = args.output_json or (args.model_output_dir / f"{args.split}_distance_iou_metrics.json")

    evaluators = {
        "raw_unet": evaluate_raw_unet,
        "raw_aurora": evaluate_raw_aurora,
        "embedding_mlp": evaluate_embedding_mlp,
        "embedding_transformer": evaluate_embedding_transformer,
    }
    metrics = evaluators[config.model_kind](args, config)
    metrics["model_output_dir"] = str(args.model_output_dir)
    metrics["split_path"] = str(args.split_path)
    metrics["split"] = args.split
    _print_summary(metrics)
    _write_outputs(output_json, metrics)
    return 0
