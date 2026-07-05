#!/usr/bin/env python3
"""Find sparse cloudy-sky test points where fine-tuned models win by tolerance IoU."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from distance_metrics_common import (
    RESULTS_DIR,
    emb_train,
    embedding_validation_files,
    raw_train,
    raw_validation_files,
)
from visualize_cloudy_sky_test_windows import (
    FINE_TUNED_MODELS,
    MODEL_ORDER,
    RAW_CHIP_MODELS,
    align_predictions,
    load_bundles,
    per_row_metrics,
    predict_one_file,
)


@dataclass
class PointCandidate:
    file_stem: str
    file_time_utc: str
    row_index: int
    sample_position: int
    target_positive_count: int
    strict_gain: float
    tol1_gain: float
    tol2_gain: float
    score: float
    target: np.ndarray
    preds: Dict[str, np.ndarray]
    probs: Dict[str, np.ndarray]
    metrics: Dict[str, Dict[str, float]]
    feather_path: Path


def strict_iou_is_similar(strict_gain: float, args: argparse.Namespace) -> bool:
    return args.strict_gain_min <= strict_gain <= args.strict_gain_max


def average_metric(metrics: Dict[str, Dict[str, np.ndarray]], names: List[str], key: str, idx: int) -> float:
    return float(np.mean([float(metrics[name][key][idx]) for name in names]))


def read_geo(feather_path: Path, row_index: int) -> Dict[str, Any]:
    columns = ["Latitude_0", "Longitude_0", "timestamp_0"]
    try:
        df = pd.read_feather(feather_path, columns=columns)
        row = df.iloc[int(row_index)]
        return {col: row[col].item() if hasattr(row[col], "item") else row[col] for col in columns}
    except Exception:
        return {col: None for col in columns}


def collect_candidates_for_file(
    raw_meta: raw_train.FileMeta,
    emb_meta: emb_train.FileMeta,
    bundles: Dict[str, Any],
    args: argparse.Namespace,
) -> List[PointCandidate]:
    per_model = {
        name: predict_one_file(bundle, raw_meta, emb_meta, args.batch_size, args.device)
        for name, bundle in bundles.items()
    }
    rows, targets, probs = align_predictions(per_model)
    preds = {
        name: (probs[name] >= bundles[name].threshold).astype(np.uint8, copy=False)
        for name in MODEL_ORDER
    }
    metrics = {name: per_row_metrics(preds[name], targets) for name in MODEL_ORDER}
    target_counts = targets.sum(axis=1).astype(np.int32)

    candidates: List[PointCandidate] = []
    valid_target_count = (target_counts >= args.min_target_ones) & (target_counts <= args.max_target_ones)
    for idx in np.flatnonzero(valid_target_count):
        strict_ft = average_metric(metrics, FINE_TUNED_MODELS, "strict", int(idx))
        strict_raw = average_metric(metrics, RAW_CHIP_MODELS, "strict", int(idx))
        tol1_ft = average_metric(metrics, FINE_TUNED_MODELS, "tol1", int(idx))
        tol1_raw = average_metric(metrics, RAW_CHIP_MODELS, "tol1", int(idx))
        tol2_ft = average_metric(metrics, FINE_TUNED_MODELS, "tol2", int(idx))
        tol2_raw = average_metric(metrics, RAW_CHIP_MODELS, "tol2", int(idx))
        strict_gain = strict_ft - strict_raw
        tol1_gain = tol1_ft - tol1_raw
        tol2_gain = tol2_ft - tol2_raw
        if not strict_iou_is_similar(strict_gain, args):
            continue
        if tol1_gain < args.min_tolerance_gain or tol2_gain < args.min_tolerance_gain:
            continue
        if args.require_each_finetune_better:
            raw_best_tol1 = max(float(metrics[name]["tol1"][idx]) for name in RAW_CHIP_MODELS)
            raw_best_tol2 = max(float(metrics[name]["tol2"][idx]) for name in RAW_CHIP_MODELS)
            if not all(
                float(metrics[name]["tol1"][idx]) > raw_best_tol1 and float(metrics[name]["tol2"][idx]) > raw_best_tol2
                for name in FINE_TUNED_MODELS
            ):
                continue
        model_metrics = {
            name: {key: float(metrics[name][key][idx]) for key in ("strict", "tol1", "tol2")}
            for name in MODEL_ORDER
        }
        strict_similarity_bonus = max(0.0, args.strict_similarity_width - abs(strict_gain))
        sparse_bonus = max(0.0, 1.0 - (int(target_counts[idx]) - args.min_target_ones) / max(args.max_target_ones, 1))
        score = float(tol1_gain + tol2_gain + 0.25 * strict_similarity_bonus + 0.05 * sparse_bonus)
        candidates.append(
            PointCandidate(
                file_stem=raw_meta.source_file.stem,
                file_time_utc=str(raw_meta.file_time),
                row_index=int(rows[idx]),
                sample_position=int(idx),
                target_positive_count=int(target_counts[idx]),
                strict_gain=float(strict_gain),
                tol1_gain=float(tol1_gain),
                tol2_gain=float(tol2_gain),
                score=score,
                target=targets[idx].astype(np.uint8),
                preds={name: preds[name][idx].astype(np.uint8) for name in MODEL_ORDER},
                probs={name: probs[name][idx].astype(np.float32) for name in MODEL_ORDER},
                metrics=model_metrics,
                feather_path=emb_meta.feather_path,
            )
        )
    return candidates


def matrix_for_plot(candidate: PointCandidate) -> np.ndarray:
    columns = [candidate.target]
    columns.extend(candidate.preds[name] for name in MODEL_ORDER)
    return np.stack(columns, axis=1).astype(np.float32)


def save_candidate(idx: int, candidate: PointCandidate, out_dir: Path) -> Dict[str, Any]:
    prefix = f"point_{idx:03d}_{candidate.file_stem}_row_{candidate.row_index:06d}"
    png_path = out_dir / f"{prefix}.png"
    npz_path = out_dir / f"{prefix}_masks.npz"
    json_path = out_dir / f"{prefix}_metrics.json"

    labels = ["Ground truth", *MODEL_ORDER]
    matrix = matrix_for_plot(candidate)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(matrix, cmap="gray_r", vmin=0, vmax=1, aspect="auto", interpolation="nearest", origin="lower")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(0, 40, 5))
    ax.set_ylabel("40-bin cloud mask index")
    ax.set_title(
        f"{candidate.file_stem} | row {candidate.row_index} | target ones={candidate.target_positive_count}\n"
        f"fine-tuned minus raw: strict={candidate.strict_gain:+.3f}, "
        f"tol@1={candidate.tol1_gain:+.3f}, tol@2={candidate.tol2_gain:+.3f}"
    )
    ax.set_xlabel("Black = cloud bin / label 1, white = clear bin / label 0")
    for x in np.arange(0.5, len(labels) - 0.5, 1.0):
        ax.axvline(x, color="0.75", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    payload: Dict[str, Any] = {
        "target": candidate.target,
    }
    for name in MODEL_ORDER:
        safe = name.lower().replace(" ", "_").replace("-", "")
        payload[f"{safe}_pred"] = candidate.preds[name]
        payload[f"{safe}_prob"] = candidate.probs[name]
    np.savez_compressed(npz_path, **payload)

    geo = read_geo(candidate.feather_path, candidate.row_index)
    metrics_payload = {
        "index": idx,
        "file_stem": candidate.file_stem,
        "file_time_utc": candidate.file_time_utc,
        "row_index": candidate.row_index,
        "target_positive_count": candidate.target_positive_count,
        "strict_gain": candidate.strict_gain,
        "tolerance_iou_1_gain": candidate.tol1_gain,
        "tolerance_iou_2_gain": candidate.tol2_gain,
        "score": candidate.score,
        "geo": geo,
        "model_metrics": candidate.metrics,
        "png_path": str(png_path),
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "single_cloudy_sky_point_visualizations")
    parser.add_argument("--max-points", type=int, default=20)
    parser.add_argument("--min-target-ones", type=int, default=2)
    parser.add_argument("--max-target-ones", type=int, default=10)
    parser.add_argument("--strict-gain-min", type=float, default=-0.08)
    parser.add_argument("--strict-gain-max", type=float, default=0.08)
    parser.add_argument("--strict-similarity-width", type=float, default=0.08)
    parser.add_argument("--min-tolerance-gain", type=float, default=0.20)
    parser.add_argument("--max-points-per-file", type=int, default=1)
    parser.add_argument("--min-row-separation", type=int, default=250)
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
    if args.min_target_ones < 1:
        raise ValueError("--min-target-ones must be >= 1")
    if args.max_target_ones < args.min_target_ones:
        raise ValueError("--max-target-ones must be >= --min-target-ones")
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

    best: List[PointCandidate] = []
    for file_idx, stem in enumerate(stems, start=1):
        print(f"[{file_idx}/{len(stems)}] Predicting and scanning {stem}")
        candidates = collect_candidates_for_file(raw_by_stem[stem], emb_by_stem[stem], bundles, args)
        best.extend(candidates)
        best.sort(key=lambda c: c.score, reverse=True)
        best = best[: max(args.max_points * 50, args.max_points)]
        print(f"  candidates in file: {len(candidates)} | retained candidates: {len(best)}")

    selected: List[PointCandidate] = []
    selected_by_file: Dict[str, int] = {}
    for candidate in sorted(best, key=lambda c: c.score, reverse=True):
        if len(selected) >= args.max_points:
            break
        if selected_by_file.get(candidate.file_stem, 0) >= args.max_points_per_file:
            continue
        too_close = any(
            existing.file_stem == candidate.file_stem
            and abs(existing.row_index - candidate.row_index) < args.min_row_separation
            for existing in selected
        )
        if too_close:
            continue
        selected.append(candidate)
        selected_by_file[candidate.file_stem] = selected_by_file.get(candidate.file_stem, 0) + 1
    if len(selected) < args.max_points:
        print(f"Only found {len(selected)} matching points; requested {args.max_points}.")

    rows = []
    for idx, candidate in enumerate(selected, start=1):
        payload = save_candidate(idx, candidate, args.output_dir)
        rows.append(payload)
        print(f"Saved point {idx:03d}: {payload['png_path']}")

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "index",
            "file_stem",
            "file_time_utc",
            "row_index",
            "target_positive_count",
            "strict_gain",
            "tolerance_iou_1_gain",
            "tolerance_iou_2_gain",
            "score",
            "png_path",
            "npz_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Saved summary: {summary_csv}")
    print(f"Saved {len(rows)} point visualization(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
