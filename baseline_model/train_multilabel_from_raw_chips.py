#!/usr/bin/env python3
"""Train a multi-label classifier from raw ERA5 chip NPZ files."""

import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "baseline_model" else SCRIPT_DIR
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            key, value = s.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            os.environ.setdefault(key, value)


_load_dotenv()

RAW_CHIPS_DIR = os.getenv("RAW_CHIPS_DIR", str(PROJECT_ROOT / "raw_chips"))
OUTPUT_DIR = str(PROJECT_ROOT / "baseline_model_outputs")
LOG_DIR = str(PROJECT_ROOT / "logs")
BASE_FEATURE_COLUMNS = [
    "Latitude_0",
    "Longitude_0",
    "time_day_sin",
    "time_day_cos",
    "time_year_sin",
    "time_year_cos",
]
TARGET_COLUMNS = [f"y_40dim_{i}" for i in range(40)]


@dataclass
class FileMeta:
    source_file: Path
    npz_path: Path
    file_time: pd.Timestamp


@dataclass
class NormalizationStats:
    chip_mean: np.ndarray
    chip_std: np.ndarray
    base_mean: np.ndarray
    base_std: np.ndarray


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


def _time_expanded_channel_names(dynamic_channel_names: np.ndarray, static_channel_names: np.ndarray) -> np.ndarray:
    expanded = [f"t_minus_6_{name}" for name in dynamic_channel_names.tolist()]
    expanded.extend([f"t_{name}" for name in dynamic_channel_names.tolist()])
    expanded.extend(static_channel_names.tolist())
    return np.asarray(expanded, dtype="<U64")


def _timestamp_unit_from_value(v: int) -> str:
    return "s" if len(str(abs(int(v)))) <= 10 else "ms"


def _file_time(feather_path: Path) -> Optional[pd.Timestamp]:
    df_ts = pd.read_feather(feather_path, columns=["timestamp_0"])
    if len(df_ts) == 0:
        return None
    first = int(df_ts["timestamp_0"].iloc[0])
    unit = _timestamp_unit_from_value(first)
    ts = pd.to_datetime(df_ts["timestamp_0"], unit=unit, utc=True)
    return pd.Timestamp(ts.max())


def discover_files(raw_chip_dir: Path) -> List[FileMeta]:
    metas: List[FileMeta] = []
    skipped_missing_source = 0
    skipped_empty = 0
    for npz_path in sorted(raw_chip_dir.glob("*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            if "source_file" not in data:
                skipped_missing_source += 1
                continue
            source_file = Path(str(data["source_file"]))
        if not source_file.exists():
            skipped_missing_source += 1
            continue
        ft = _file_time(source_file)
        if ft is None:
            skipped_empty += 1
            continue
        metas.append(FileMeta(source_file=source_file, npz_path=npz_path, file_time=ft))
    print(f"Candidate raw-chip files: {len(metas)}")
    print(f"Skipped files (missing source feather): {skipped_missing_source}")
    print(f"Skipped files (empty feather): {skipped_empty}")
    return metas


def select_train_test_files(
    metas: Sequence[FileMeta], train_files: int, test_files: int, seed: int
) -> Tuple[List[FileMeta], List[FileMeta]]:
    if len(metas) < (train_files + test_files):
        raise ValueError(
            f"Not enough raw-chip files: have {len(metas)}, need at least {train_files + test_files}."
        )

    rng = random.Random(seed)
    sorted_metas = sorted(metas, key=lambda x: x.file_time)
    late_pool_size = min(len(sorted_metas), max(test_files * 3, test_files))
    late_pool = sorted_metas[-late_pool_size:]
    test_selected = rng.sample(late_pool, k=test_files)
    test_min_time = min(m.file_time for m in test_selected)

    train_pool = [m for m in sorted_metas if m.file_time < test_min_time]
    if len(train_pool) < train_files:
        raise ValueError(
            f"Not enough earlier files for training. "
            f"Need {train_files}, but only {len(train_pool)} files are earlier than test_min_time={test_min_time}."
        )
    train_selected = rng.sample(train_pool, k=train_files)
    return train_selected, sorted(test_selected, key=lambda x: x.file_time)


def _timestamps_to_base_features(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(timestamps), unit="s", utc=True)
    hour_of_day = ts.dt.hour.to_numpy(dtype=np.float32) + ts.dt.minute.to_numpy(dtype=np.float32) / 60.0
    day_phase = (2.0 * np.pi * hour_of_day) / 24.0
    day_of_year = ts.dt.dayofyear.to_numpy(dtype=np.float32) - 1.0
    year_phase = (2.0 * np.pi * day_of_year) / 365.25

    return np.stack(
        [
            latitudes.astype(np.float32, copy=False),
            longitudes.astype(np.float32, copy=False),
            np.sin(day_phase).astype(np.float32, copy=False),
            np.cos(day_phase).astype(np.float32, copy=False),
            np.sin(year_phase).astype(np.float32, copy=False),
            np.cos(year_phase).astype(np.float32, copy=False),
        ],
        axis=1,
    )


def _load_npz_payload(
    meta: FileMeta,
    sample_ratio: float,
    max_samples_per_file: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    with np.load(meta.npz_path, allow_pickle=False) as data:
        dynamic = data["dynamic_chips"].astype(np.float32, copy=False)
        static = data["static_chips"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.float32, copy=False)
        latitudes = data["latitudes"].astype(np.float32, copy=False)
        longitudes = data["longitudes"].astype(np.float32, copy=False)
        timestamps = data["timestamps"].astype(np.float64, copy=False)
        dynamic_channel_names = data["dynamic_channel_names"]
        static_channel_names = data["static_channel_names"]
        chip_size = int(data["chip_size"])

    if dynamic.size == 0 or labels.size == 0:
        return {
            "dynamic": np.empty((0, 0, 0, 0, 0), dtype=np.float32),
            "static": np.empty((0, 0, 0, 0), dtype=np.float32),
            "labels": np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32),
            "base_features": np.empty((0, len(BASE_FEATURE_COLUMNS)), dtype=np.float32),
            "dynamic_channel_names": dynamic_channel_names,
            "static_channel_names": static_channel_names,
            "chip_size": np.asarray(chip_size, dtype=np.int32),
        }

    if not (0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    indices = np.arange(labels.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if sample_ratio < 1.0:
        count = max(1, int(len(indices) * sample_ratio))
        indices = rng.choice(indices, size=count, replace=False)
    if max_samples_per_file is not None and max_samples_per_file > 0 and len(indices) > max_samples_per_file:
        indices = rng.choice(indices, size=max_samples_per_file, replace=False)
    indices = np.sort(indices)

    base_features = _timestamps_to_base_features(
        latitudes=latitudes[indices],
        longitudes=longitudes[indices],
        timestamps=timestamps[indices],
    )
    return {
        "dynamic": dynamic[indices],
        "static": static[indices],
        "labels": labels[indices],
        "base_features": base_features,
        "dynamic_channel_names": dynamic_channel_names,
        "static_channel_names": static_channel_names,
        "chip_size": np.asarray(chip_size, dtype=np.int32),
    }


def _flatten_chip_channels(dynamic: np.ndarray, static: np.ndarray) -> np.ndarray:
    n, t, c, h, w = dynamic.shape
    dynamic_flat = dynamic.reshape(n, t * c, h, w)
    return np.concatenate([dynamic_flat, static], axis=1).astype(np.float32, copy=False)


def compute_train_stats(
    files: Sequence[FileMeta],
    sample_ratio: float,
    max_samples_per_file: Optional[int],
    seed: int,
) -> Tuple[NormalizationStats, np.ndarray, np.ndarray, int]:
    chip_sum: Optional[np.ndarray] = None
    chip_sq_sum: Optional[np.ndarray] = None
    base_sum: Optional[np.ndarray] = None
    base_sq_sum: Optional[np.ndarray] = None
    label_sum = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
    label_count = 0
    total_pixels = 0
    channel_names: Optional[np.ndarray] = None
    chip_size: Optional[int] = None

    for file_idx, meta in enumerate(files):
        payload = _load_npz_payload(
            meta=meta,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed + file_idx,
        )
        if payload["labels"].shape[0] == 0:
            continue
        chips = _flatten_chip_channels(payload["dynamic"], payload["static"])
        base = payload["base_features"]
        labels = payload["labels"]

        if chip_sum is None:
            chip_sum = chips.sum(axis=(0, 2, 3), dtype=np.float64)
            chip_sq_sum = np.square(chips, dtype=np.float64).sum(axis=(0, 2, 3))
            base_sum = base.sum(axis=0, dtype=np.float64)
            base_sq_sum = np.square(base, dtype=np.float64).sum(axis=0)
            channel_names = _time_expanded_channel_names(
                dynamic_channel_names=payload["dynamic_channel_names"],
                static_channel_names=payload["static_channel_names"],
            )
            chip_size = int(payload["chip_size"])
        else:
            chip_sum += chips.sum(axis=(0, 2, 3), dtype=np.float64)
            chip_sq_sum += np.square(chips, dtype=np.float64).sum(axis=(0, 2, 3))
            base_sum += base.sum(axis=0, dtype=np.float64)
            base_sq_sum += np.square(base, dtype=np.float64).sum(axis=0)

        total_pixels += chips.shape[0] * chips.shape[2] * chips.shape[3]
        label_sum += labels.sum(axis=0, dtype=np.float64)
        label_count += labels.shape[0]

    if chip_sum is None or chip_sq_sum is None or base_sum is None or base_sq_sum is None or channel_names is None:
        raise ValueError("No train samples available to compute normalization statistics.")

    chip_mean = chip_sum / max(total_pixels, 1)
    chip_var = np.maximum(chip_sq_sum / max(total_pixels, 1) - np.square(chip_mean), 1e-6)
    base_mean = base_sum / max(label_count, 1)
    base_var = np.maximum(base_sq_sum / max(label_count, 1) - np.square(base_mean), 1e-6)

    stats = NormalizationStats(
        chip_mean=chip_mean.astype(np.float32),
        chip_std=np.sqrt(chip_var).astype(np.float32),
        base_mean=base_mean.astype(np.float32),
        base_std=np.sqrt(base_var).astype(np.float32),
    )
    return stats, label_sum.astype(np.float32), channel_names, int(chip_size)


def _normalize_batch(
    chips: np.ndarray,
    base_features: np.ndarray,
    stats: NormalizationStats,
    use_base_features: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chips_n = (chips - stats.chip_mean[None, :, None, None]) / stats.chip_std[None, :, None, None]
    if use_base_features:
        base_n = (base_features - stats.base_mean[None, :]) / stats.base_std[None, :]
    else:
        base_n = np.zeros((chips.shape[0], 0), dtype=np.float32)
    return torch.from_numpy(chips_n.astype(np.float32, copy=False)), torch.from_numpy(base_n.astype(np.float32, copy=False))


def iter_file_batches(
    files: Sequence[FileMeta],
    batch_size: int,
    stats: NormalizationStats,
    shuffle: bool,
    seed: int,
    sample_ratio: float,
    max_samples_per_file: Optional[int],
    use_base_features: bool,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    file_order = list(range(len(files)))
    if shuffle:
        rng.shuffle(file_order)

    for file_pos in file_order:
        meta = files[file_pos]
        payload = _load_npz_payload(
            meta=meta,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed + file_pos,
        )
        if payload["labels"].shape[0] == 0:
            continue

        chips = _flatten_chip_channels(payload["dynamic"], payload["static"])
        base_features = payload["base_features"]
        labels = payload["labels"]
        sample_indices = np.arange(labels.shape[0], dtype=np.int64)
        if shuffle:
            rng.shuffle(sample_indices)

        for start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[start : start + batch_size]
            chips_t, base_t = _normalize_batch(
                chips=chips[batch_indices],
                base_features=base_features[batch_indices],
                stats=stats,
                use_base_features=use_base_features,
            )
            y_t = torch.from_numpy(labels[batch_indices].astype(np.float32, copy=False))
            yield chips_t, base_t, y_t


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_dim: int = 40,
        base_feature_dim: int = 6,
        base_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.base_feature_dim = base_feature_dim
        self.enc1 = DoubleConv(in_channels, base_channels, dropout=0.0)
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout=0.0)
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4, dropout=dropout)
        self.dec2 = DoubleConv(base_channels * 4 + base_channels * 2, base_channels * 2, dropout=0.0)
        self.dec1 = DoubleConv(base_channels * 2 + base_channels, base_channels, dropout=0.0)
        head_in = (base_channels * 4) + (base_channels * 2) + base_channels + base_feature_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, output_dim),
        )

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, chips: torch.Tensor, base_features: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(chips)
        p1 = F.max_pool2d(e1, kernel_size=2, ceil_mode=True)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, kernel_size=2, ceil_mode=True)
        b = self.bottleneck(p2)

        d2 = self._resize_like(b, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._resize_like(d2, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        pooled = torch.cat(
            [
                F.adaptive_avg_pool2d(b, 1).flatten(1),
                F.adaptive_avg_pool2d(d2, 1).flatten(1),
                F.adaptive_avg_pool2d(d1, 1).flatten(1),
                base_features,
            ],
            dim=1,
        )
        return self.head(pooled)


def _binary_auc_macro(probs: torch.Tensor, targets: torch.Tensor) -> float:
    aucs: List[float] = []
    for j in range(targets.shape[1]):
        y = targets[:, j]
        s = probs[:, j]
        pos = int((y == 1).sum().item())
        neg = int((y == 0).sum().item())
        if pos == 0 or neg == 0:
            continue
        order = torch.argsort(s)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
        rank_sum_pos = ranks[y == 1].sum().item()
        auc = (rank_sum_pos - (pos * (pos + 1) / 2.0)) / (pos * neg)
        aucs.append(float(auc))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _sample_iou_values(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    intersection = ((preds == 1) & (targets == 1)).sum(dim=1).float()
    union = ((preds == 1) | (targets == 1)).sum(dim=1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def _sample_iou_mean(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return float(_sample_iou_values(preds, targets).mean().item())


def _find_best_iou_threshold(probs: torch.Tensor, targets: torch.Tensor, num_thresholds: int = 101) -> Tuple[float, float]:
    best_threshold = 0.5
    best_iou = float("-inf")
    for threshold in torch.linspace(0.0, 1.0, steps=num_thresholds):
        preds = (probs >= threshold).float()
        iou = _sample_iou_mean(preds, targets)
        if iou > best_iou:
            best_iou = iou
            best_threshold = float(threshold.item())
    return best_threshold, float(best_iou)


def _binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    iou_threshold: Optional[float] = None,
    search_iou_threshold: bool = False,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == targets).float().mean().item()

    tp = ((preds == 1) & (targets == 1)).sum(dim=0).float()
    fp = ((preds == 1) & (targets == 0)).sum(dim=0).float()
    fn = ((preds == 0) & (targets == 1)).sum(dim=0).float()
    f1_per_label = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    f1_macro = f1_per_label.mean().item()
    auc_macro = _binary_auc_macro(probs, targets)
    if search_iou_threshold:
        resolved_iou_threshold, iou_mean = _find_best_iou_threshold(probs, targets)
    else:
        resolved_iou_threshold = 0.5 if iou_threshold is None else float(iou_threshold)
        iou_preds = (probs >= resolved_iou_threshold).float()
        iou_mean = _sample_iou_mean(iou_preds, targets)
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "auc_macro": float(auc_macro),
        "iou_mean": float(iou_mean),
        "iou_threshold": float(resolved_iou_threshold),
    }


def _evaluate_files(
    model: nn.Module,
    loss_fn: nn.Module,
    files: Sequence[FileMeta],
    batch_size: int,
    device: str,
    stats: NormalizationStats,
    use_base_features: bool,
    sample_ratio: float,
    max_samples_per_file: Optional[int],
    seed: int,
    iou_threshold: Optional[float] = None,
    search_iou_threshold: bool = False,
    collect_details: bool = False,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    total_loss = 0.0
    total_count = 0
    logits_cpu_parts: List[torch.Tensor] = []
    targets_cpu_parts: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for chips_t, base_t, y_t in iter_file_batches(
            files=files,
            batch_size=batch_size,
            stats=stats,
            shuffle=False,
            seed=seed,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            use_base_features=use_base_features,
        ):
            chips_t = chips_t.to(device)
            base_t = base_t.to(device)
            y_t = y_t.to(device)
            logits = model(chips_t, base_t)
            loss = loss_fn(logits, y_t)
            total_loss += float(loss.item()) * chips_t.shape[0]
            total_count += chips_t.shape[0]
            logits_cpu_parts.append(logits.cpu())
            targets_cpu_parts.append(y_t.cpu())

    if not logits_cpu_parts:
        raise ValueError("No evaluation samples were produced from raw-chip files.")

    avg_loss = total_loss / max(total_count, 1)
    all_logits = torch.cat(logits_cpu_parts, dim=0)
    all_targets = torch.cat(targets_cpu_parts, dim=0)
    metrics = _binary_metrics(
        all_logits,
        all_targets,
        iou_threshold=iou_threshold,
        search_iou_threshold=search_iou_threshold,
    )
    details: Optional[Dict[str, torch.Tensor]] = None
    if collect_details:
        probs = torch.sigmoid(all_logits)
        preds = (probs >= metrics["iou_threshold"]).float()
        details = {
            "probs": probs,
            "targets": all_targets,
            "preds": preds,
            "sample_ious": _sample_iou_values(preds, all_targets),
        }
    return avg_loss, metrics, details


def _save_random_iou_plots(
    out_dir: Path,
    split_name: str,
    eval_details: Optional[Dict[str, torch.Tensor]],
    iou_threshold: float,
    num_random_plots: int,
    seed: int,
) -> None:
    if num_random_plots <= 0:
        print(f"Skipping {split_name} IoU plots because --plot-random-iou-count={num_random_plots}.")
        return
    if plt is None:
        print("Skipping IoU plots because matplotlib is not installed.")
        return
    if not eval_details:
        print(f"Skipping {split_name} IoU plots because evaluation details were not collected.")
        return

    probs = eval_details["probs"].cpu().numpy()
    targets = eval_details["targets"].cpu().numpy()
    preds = eval_details["preds"].cpu().numpy()
    sample_ious = eval_details["sample_ious"].cpu().numpy()
    if len(sample_ious) == 0:
        print(f"Skipping {split_name} IoU plots because the split is empty.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    chosen_count = min(num_random_plots, len(sample_ious))
    chosen_indices = rng.choice(len(sample_ious), size=chosen_count, replace=False)
    label_ids = np.arange(targets.shape[1])

    hist_path = out_dir / f"{split_name}_iou_hist.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = min(30, max(8, int(np.sqrt(len(sample_ious)) * 2)))
    ax.hist(sample_ious, bins=bins, color="#4C78A8", edgecolor="white")
    ax.axvline(sample_ious.mean(), color="#F58518", linestyle="--", linewidth=2, label=f"mean={sample_ious.mean():.3f}")
    ax.set_title(f"{split_name.title()} Sample IoU Distribution")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Samples")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_path, dpi=160)
    plt.close(fig)
    print(f"Saved IoU histogram: {hist_path}")

    for rank, sample_idx in enumerate(chosen_indices, start=1):
        sample_path = out_dir / f"{split_name}_iou_sample_{rank:02d}_idx_{int(sample_idx):06d}.png"
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        axes[0].bar(label_ids, probs[sample_idx], color="#4C78A8", width=0.9)
        true_positive_labels = label_ids[targets[sample_idx] > 0.5]
        if len(true_positive_labels) > 0:
            axes[0].scatter(
                true_positive_labels,
                np.full_like(true_positive_labels, 1.04, dtype=np.float32),
                marker="v",
                color="#E45756",
                s=45,
                label="target=1",
            )
            axes[0].legend(loc="upper right")
        axes[0].axhline(iou_threshold, color="#72B7B2", linestyle="--", linewidth=2)
        axes[0].set_ylim(0.0, 1.1)
        axes[0].set_ylabel("Predicted prob")
        axes[0].set_title(
            f"{split_name.title()} sample {int(sample_idx)} | IoU={sample_ious[sample_idx]:.3f} | "
            f"target_pos={int(targets[sample_idx].sum())} | pred_pos={int(preds[sample_idx].sum())} | "
            f"thr={iou_threshold:.3f}"
        )
        comparison = np.vstack([targets[sample_idx], preds[sample_idx]])
        axes[1].imshow(comparison, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        axes[1].set_yticks([0, 1], labels=["target", "pred"])
        axes[1].set_xlabel("Label index")
        axes[1].set_ylabel("Binary")
        fig.tight_layout()
        fig.savefig(sample_path, dpi=160)
        plt.close(fig)
        print(f"Saved random IoU sample plot: {sample_path}")


def train_model(
    train_files: Sequence[FileMeta],
    test_files: Sequence[FileMeta],
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    base_channels: int,
    dropout: float,
    use_pos_weight: bool,
    use_base_features: bool,
    seed: int,
    device: str,
    plot_random_iou_count: int,
    plot_dir: Path,
    sample_ratio: float,
    max_samples_per_file: Optional[int],
) -> Dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed)
    np.random.seed(seed)

    stats, train_label_sum, channel_names, chip_size = compute_train_stats(
        files=train_files,
        sample_ratio=sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=seed,
    )
    input_channels = int(len(channel_names))
    base_feature_dim = len(BASE_FEATURE_COLUMNS) if use_base_features else 0
    model = UNetClassifier(
        in_channels=input_channels,
        output_dim=len(TARGET_COLUMNS),
        base_feature_dim=base_feature_dim,
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)
    print(
        f"U-Net classifier | input_channels={input_channels} | chip_size={chip_size} | "
        f"base_channels={base_channels} | use_base_features={use_base_features}"
    )

    train_sample_count = 0
    for file_idx, meta in enumerate(train_files):
        payload = _load_npz_payload(
            meta=meta,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed + file_idx,
        )
        train_sample_count += int(payload["labels"].shape[0])

    if train_sample_count <= 0:
        raise ValueError("No train samples available after applying sample_ratio/max_samples_per_file.")

    if use_pos_weight:
        pos = torch.from_numpy(train_label_sum)
        neg = torch.full_like(pos, float(train_sample_count)) - pos
        pos_weight = (neg / torch.clamp(pos, min=1.0)).clamp(min=1.0, max=20.0).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    test_eval_details: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for chips_t, base_t, y_t in iter_file_batches(
            files=train_files,
            batch_size=batch_size,
            stats=stats,
            shuffle=True,
            seed=seed + epoch,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            use_base_features=use_base_features,
        ):
            chips_t = chips_t.to(device)
            base_t = base_t.to(device)
            y_t = y_t.to(device)
            optimizer.zero_grad()
            logits = model(chips_t, base_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * chips_t.shape[0]
            total_count += chips_t.shape[0]

        train_loss = total_loss / max(total_count, 1)
        train_eval_loss, train_metrics, _ = _evaluate_files(
            model=model,
            loss_fn=loss_fn,
            files=train_files,
            batch_size=eval_batch_size,
            device=device,
            stats=stats,
            use_base_features=use_base_features,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed,
            search_iou_threshold=True,
        )
        test_loss, test_metrics, test_eval_details = _evaluate_files(
            model=model,
            loss_fn=loss_fn,
            files=test_files,
            batch_size=eval_batch_size,
            device=device,
            stats=stats,
            use_base_features=use_base_features,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed,
            iou_threshold=train_metrics["iou_threshold"],
            collect_details=(epoch == epochs),
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | train_eval_loss={train_eval_loss:.5f} | "
            f"test_loss={test_loss:.5f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | test_acc={test_metrics['accuracy']:.4f} | "
            f"train_f1_macro={train_metrics['f1_macro']:.4f} | test_f1_macro={test_metrics['f1_macro']:.4f} | "
            f"train_auc_macro={train_metrics['auc_macro']:.4f} | test_auc_macro={test_metrics['auc_macro']:.4f} | "
            f"train_iou_mean={train_metrics['iou_mean']:.4f} @thr={train_metrics['iou_threshold']:.3f} | "
            f"test_iou_mean={test_metrics['iou_mean']:.4f} @thr={test_metrics['iou_threshold']:.3f}"
        )

    _save_random_iou_plots(
        out_dir=plot_dir,
        split_name="validation",
        eval_details=test_eval_details,
        iou_threshold=test_metrics["iou_threshold"],
        num_random_plots=plot_random_iou_count,
        seed=seed,
    )

    return {
        "model": model,
        "stats": stats,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "input_channels": input_channels,
        "base_channels": int(base_channels),
        "dropout": float(dropout),
        "chip_size": int(chip_size),
        "channel_names": channel_names,
        "use_base_features": bool(use_base_features),
    }


def _save_artifacts(
    out_dir: Path,
    model: nn.Module,
    stats: NormalizationStats,
    train_files: Sequence[FileMeta],
    test_files: Sequence[FileMeta],
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    input_channels: int,
    base_channels: int,
    dropout: float,
    chip_size: int,
    channel_names: np.ndarray,
    use_base_features: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "multilabel_unet_classifier.pt"
    stats_path = out_dir / "normalization_stats.npz"
    split_path = out_dir / "file_split.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_channels": input_channels,
            "output_dim": len(TARGET_COLUMNS),
            "base_channels": base_channels,
            "dropout": float(dropout),
            "chip_size": chip_size,
            "channel_names": channel_names,
            "base_features": BASE_FEATURE_COLUMNS if use_base_features else [],
            "target_columns": TARGET_COLUMNS,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
        model_path,
    )
    np.savez_compressed(
        stats_path,
        chip_mean=stats.chip_mean,
        chip_std=stats.chip_std,
        base_mean=stats.base_mean,
        base_std=stats.base_std,
    )

    rows = []
    for m in train_files:
        rows.append({"split": "train", "file": str(m.source_file), "npz": str(m.npz_path), "file_time_utc": str(m.file_time)})
    for m in test_files:
        rows.append({"split": "test", "file": str(m.source_file), "npz": str(m.npz_path), "file_time_utc": str(m.file_time)})
    pd.DataFrame(rows).to_csv(split_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved normalization stats: {stats_path}")
    print(f"Saved split manifest: {split_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-chip-dir", type=str, default=RAW_CHIPS_DIR)
    parser.add_argument("--train-files", type=int, default=100)
    parser.add_argument("--test-files", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples-per-file", type=int, default=0)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--no-base-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=str, default=LOG_DIR)
    parser.add_argument(
        "--plot-random-iou-count",
        type=int,
        default=6,
        help="Number of random validation samples to save as IoU plots in --plot-dir. Set 0 to disable.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    raw_chip_dir = Path(args.raw_chip_dir)
    if not raw_chip_dir.exists():
        raise FileNotFoundError(f"Raw chip dir does not exist: {raw_chip_dir}")

    metas = discover_files(raw_chip_dir)
    train_files, test_files = select_train_test_files(
        metas=metas,
        train_files=args.train_files,
        test_files=args.test_files,
        seed=args.seed,
    )
    print(f"Selected train files: {len(train_files)}")
    print(f"Selected test files: {len(test_files)}")
    print(f"Train max file_time: {max(m.file_time for m in train_files)}")
    print(f"Test min file_time:  {min(m.file_time for m in test_files)}")

    fit = train_model(
        train_files=train_files,
        test_files=test_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        base_channels=args.base_channels,
        dropout=args.dropout,
        use_pos_weight=(not args.no_pos_weight),
        use_base_features=(not args.no_base_features),
        seed=args.seed,
        device=args.device,
        plot_random_iou_count=args.plot_random_iou_count,
        plot_dir=Path(args.plot_dir),
        sample_ratio=args.sample_ratio,
        max_samples_per_file=(None if args.max_samples_per_file <= 0 else int(args.max_samples_per_file)),
    )

    print(
        "Final metrics | "
        f"train_acc={fit['train_metrics']['accuracy']:.4f}, "
        f"test_acc={fit['test_metrics']['accuracy']:.4f}, "
        f"train_f1_macro={fit['train_metrics']['f1_macro']:.4f}, "
        f"test_f1_macro={fit['test_metrics']['f1_macro']:.4f}, "
        f"train_auc_macro={fit['train_metrics']['auc_macro']:.4f}, "
        f"test_auc_macro={fit['test_metrics']['auc_macro']:.4f}, "
        f"train_iou_mean={fit['train_metrics']['iou_mean']:.4f} @thr={fit['train_metrics']['iou_threshold']:.3f}, "
        f"test_iou_mean={fit['test_metrics']['iou_mean']:.4f} @thr={fit['test_metrics']['iou_threshold']:.3f}"
    )

    _save_artifacts(
        out_dir=Path(args.output_dir),
        model=fit["model"],
        stats=fit["stats"],
        train_files=train_files,
        test_files=test_files,
        train_metrics=fit["train_metrics"],
        test_metrics=fit["test_metrics"],
        input_channels=fit["input_channels"],
        base_channels=fit["base_channels"],
        dropout=fit["dropout"],
        chip_size=fit["chip_size"],
        channel_names=fit["channel_names"],
        use_base_features=fit["use_base_features"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
