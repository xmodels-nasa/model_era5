#!/usr/bin/env python3
"""Train a 40-label classifier from Feather rows plus 3x3 Aurora embedding grids."""

import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "fine_tuned_model" else SCRIPT_DIR


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

FEATHER_ROOT = os.getenv("FEATHER_ROOT", "")
EMBEDDING_3X3_DIR = os.getenv(
    "EMBEDDING_3X3_OUTPUT_DIR",
    str(PROJECT_ROOT / "embeddings_3x3"),
)
OUTPUT_DIR = str(PROJECT_ROOT / "model_outputs_3x3")
LOG_DIR = str(PROJECT_ROOT / "logs_3x3")
DEFAULT_TEST_START_TIME = "2019-07-01T00:00:00Z"

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
    feather_path: Path
    npz_path: Path
    file_time: pd.Timestamp


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)


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


def _parse_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def discover_files(feather_root: Path, embedding_dir: Path) -> List[FileMeta]:
    metas: List[FileMeta] = []
    skipped_no_npz = 0
    skipped_empty = 0
    for feather_file in sorted(feather_root.rglob("*.feather")):
        npz_path = embedding_dir / f"{_safe_stem(feather_file)}.npz"
        if not npz_path.exists():
            skipped_no_npz += 1
            continue
        ft = _file_time(feather_file)
        if ft is None:
            skipped_empty += 1
            continue
        metas.append(FileMeta(feather_path=feather_file, npz_path=npz_path, file_time=ft))
    print(f"Candidate files with 3x3 embeddings: {len(metas)}")
    print(f"Skipped files (missing embedding): {skipped_no_npz}")
    print(f"Skipped files (empty feather): {skipped_empty}")
    return metas


def _sample_or_all(files: Sequence[FileMeta], count: int, seed: int) -> List[FileMeta]:
    if count <= 0 or count >= len(files):
        return list(files)
    rng = random.Random(seed)
    return sorted(
        rng.sample(list(files), k=count),
        key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path)),
    )


def select_train_test_files(
    metas: Sequence[FileMeta],
    train_files: int,
    test_files: int,
    seed: int,
    test_start_time: pd.Timestamp,
) -> Tuple[List[FileMeta], List[FileMeta]]:
    sorted_metas = sorted(metas, key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path)))
    train_pool = [m for m in sorted_metas if m.file_time < test_start_time]
    test_pool = [m for m in sorted_metas if m.file_time >= test_start_time]
    if not train_pool:
        raise ValueError(f"No train files found before cutoff {test_start_time}.")
    if not test_pool:
        raise ValueError(f"No test files found on or after cutoff {test_start_time}.")
    if train_files > len(train_pool):
        raise ValueError(f"Requested {train_files} train files, but only found {len(train_pool)}.")
    if test_files > len(test_pool):
        raise ValueError(f"Requested {test_files} test files, but only found {len(test_pool)}.")
    return (
        _sample_or_all(train_pool, train_files, seed),
        _sample_or_all(test_pool, test_files, seed + 1),
    )


def _empty_base_array() -> np.ndarray:
    return np.empty((0, len(BASE_FEATURE_COLUMNS)), dtype=np.float32)


def _empty_grid_array() -> np.ndarray:
    return np.empty((0, 0, 0, 0, 0), dtype=np.float32)


def _load_one_file_arrays(
    meta: FileMeta,
    sample_ratio: float = 1.0,
    max_samples_per_file: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required_cols = ["timestamp_0"] + BASE_FEATURE_COLUMNS + TARGET_COLUMNS
    df = pd.read_feather(meta.feather_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {meta.feather_path.name}: {missing}")

    with np.load(meta.npz_path) as data:
        if "emb_3x3_all_levels" not in data or "row_indices" not in data:
            raise ValueError(
                f"Missing arrays in {meta.npz_path.name}; expected emb_3x3_all_levels and row_indices."
            )
        emb = data["emb_3x3_all_levels"]
        rows = data["row_indices"]

    if emb.size == 0 or rows.size == 0:
        return _empty_base_array(), _empty_grid_array(), np.empty((0, 40), dtype=np.float32), np.empty((0,), dtype=np.int64)
    if emb.ndim < 5:
        raise ValueError(
            f"Unexpected 3x3 embedding shape in {meta.npz_path.name}: {emb.shape}. "
            "Expected [rows, 3, 3, ...] with one or more trailing embedding dimensions."
        )
    if emb.shape[1] != 3 or emb.shape[2] != 3:
        raise ValueError(
            f"Unexpected spatial grid shape in {meta.npz_path.name}: {emb.shape}. "
            "Expected axes 1 and 2 to be the 3x3 neighborhood."
        )

    rows = rows.astype(np.int64, copy=False)
    valid = (rows >= 0) & (rows < len(df))
    if not np.all(valid):
        rows = rows[valid]
        emb = emb[valid]
    if rows.size == 0:
        return _empty_base_array(), _empty_grid_array(), np.empty((0, 40), dtype=np.float32), np.empty((0,), dtype=np.int64)

    if not (0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    sample_indices = np.arange(rows.shape[0], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if sample_ratio < 1.0:
        count = max(1, int(len(sample_indices) * sample_ratio))
        sample_indices = rng.choice(sample_indices, size=count, replace=False)
    if max_samples_per_file is not None and max_samples_per_file > 0 and len(sample_indices) > max_samples_per_file:
        sample_indices = rng.choice(sample_indices, size=max_samples_per_file, replace=False)
    sample_indices = np.sort(sample_indices)

    rows = rows[sample_indices]
    emb = emb[sample_indices].astype(np.float32, copy=False)
    base = df.iloc[rows][BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    y = df.iloc[rows][TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    y = (y > 0.5).astype(np.float32)
    return base, emb, y, rows


def load_one_file_samples(
    meta: FileMeta,
    sample_ratio: float = 1.0,
    max_samples_per_file: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, emb, y, _ = _load_one_file_arrays(
        meta,
        sample_ratio=sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=seed,
    )
    return base, emb, y


def load_dataset(
    files: Sequence[FileMeta],
    sample_ratio: float = 1.0,
    max_samples_per_file: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_parts: List[np.ndarray] = []
    grid_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    expected_grid_shape: Optional[Tuple[int, ...]] = None

    for file_idx, meta in enumerate(files):
        base, emb, y = load_one_file_samples(
            meta,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed + file_idx,
        )
        if base.size == 0:
            continue
        if expected_grid_shape is None:
            expected_grid_shape = tuple(emb.shape[1:])
        elif tuple(emb.shape[1:]) != expected_grid_shape:
            raise ValueError(
                f"Inconsistent 3x3 embedding shape for {meta.npz_path.name}: {emb.shape[1:]} vs {expected_grid_shape}."
            )
        base_parts.append(base)
        grid_parts.append(emb)
        y_parts.append(y)

    if not base_parts:
        raise ValueError("No row samples available after loading selected files.")

    return (
        np.concatenate(base_parts, axis=0),
        np.concatenate(grid_parts, axis=0),
        np.concatenate(y_parts, axis=0),
    )


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


def _find_best_iou_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_thresholds: int = 101,
) -> Tuple[float, float]:
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


def _resolve_hidden_dims(input_dim: int, hidden_dims: Optional[Sequence[int]]) -> List[int]:
    if hidden_dims:
        dims = [int(d) for d in hidden_dims if int(d) > 0]
        if dims:
            return dims

    first = min(4096, max(1024, input_dim // 4))
    second = min(2048, max(512, first // 2))
    if second >= first:
        second = max(512, first // 2)
    return [first, second]


class Grid3x3EmbeddingMLP(nn.Module):
    def __init__(
        self,
        base_dim: int,
        grid_shape: Sequence[int],
        output_dim: int,
        hidden_dims: Optional[Sequence[int]],
        dropout: float,
    ):
        super().__init__()
        if len(grid_shape) < 4:
            raise ValueError(f"grid_shape must be [3, 3, ...], got {grid_shape}")

        grid_h = int(grid_shape[0])
        grid_w = int(grid_shape[1])
        cell_shape = tuple(int(v) for v in grid_shape[2:])
        if grid_h != 3 or grid_w != 3:
            raise ValueError(f"Expected a 3x3 grid, got {grid_shape}")

        self.grid_shape = (grid_h, grid_w, *cell_shape)
        self.cell_shape = cell_shape
        self.grid_input_dim = int(np.prod(grid_shape))
        self.input_dim = int(base_dim + self.grid_input_dim)
        resolved_hidden_dims = _resolve_hidden_dims(self.input_dim, hidden_dims)
        self.hidden_dims = tuple(resolved_hidden_dims)

        layers: List[nn.Module] = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, base_x: torch.Tensor, grid_x: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([base_x, grid_x.reshape(grid_x.shape[0], -1)], dim=1)
        return self.net(fused)


def _evaluate_in_batches(
    model: nn.Module,
    loss_fn: nn.Module,
    base_x: torch.Tensor,
    grid_x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: str,
    iou_threshold: Optional[float] = None,
    search_iou_threshold: bool = False,
    collect_details: bool = False,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    ds = TensorDataset(base_x, grid_x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_count = 0
    logits_cpu_parts: List[torch.Tensor] = []
    targets_cpu_parts: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for base_b, grid_b, yb in dl:
            base_b = base_b.to(device)
            grid_b = grid_b.to(device)
            yb = yb.to(device)
            logits = model(base_b, grid_b)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * base_b.shape[0]
            total_count += base_b.shape[0]
            logits_cpu_parts.append(logits.cpu())
            targets_cpu_parts.append(yb.cpu())

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


def _choose_contiguous_window(
    row_indices: np.ndarray,
    window_size: int,
    rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    if window_size <= 0 or row_indices.size < window_size:
        return None

    window_spans: List[Tuple[int, int]] = []
    run_start = 0
    for idx in range(1, row_indices.size + 1):
        is_break = idx == row_indices.size or row_indices[idx] != row_indices[idx - 1] + 1
        if not is_break:
            continue
        for start in range(run_start, idx - window_size + 1):
            window_spans.append((start, start + window_size))
        run_start = idx

    if not window_spans:
        return None
    return window_spans[int(rng.integers(len(window_spans)))]


def _predict_masks_for_file(
    model: nn.Module,
    meta: FileMeta,
    base_mean: np.ndarray,
    base_std: np.ndarray,
    grid_mean: np.ndarray,
    grid_std: np.ndarray,
    eval_batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, emb, y, row_indices = _load_one_file_arrays(meta)
    if y.shape[0] == 0:
        empty_masks = np.empty((0, len(TARGET_COLUMNS)), dtype=np.float32)
        return np.empty((0,), dtype=np.int64), empty_masks, empty_masks

    order = np.argsort(row_indices)
    row_indices = row_indices[order]
    base = base[order]
    emb = emb[order]
    y = y[order]

    base_n = (base - base_mean) / base_std
    emb_n = (emb - grid_mean) / grid_std
    pred_parts: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, base_n.shape[0], eval_batch_size):
            stop = min(start + eval_batch_size, base_n.shape[0])
            base_b = torch.from_numpy(base_n[start:stop].astype(np.float32, copy=False)).to(device)
            grid_b = torch.from_numpy(emb_n[start:stop].astype(np.float32, copy=False)).to(device)
            logits = model(base_b, grid_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            pred_parts.append(probs.astype(np.float32, copy=False))

    return row_indices, y.astype(np.float32, copy=False), np.concatenate(pred_parts, axis=0)


def _save_random_curtain_plots(
    out_dir: Path,
    split_name: str,
    model: nn.Module,
    files: Sequence[FileMeta],
    base_mean: np.ndarray,
    base_std: np.ndarray,
    grid_mean: np.ndarray,
    grid_std: np.ndarray,
    eval_batch_size: int,
    device: str,
    num_random_plots: int,
    curtain_rows: int,
    seed: int,
) -> None:
    if num_random_plots <= 0:
        print(f"Skipping {split_name} curtain plots because plot count={num_random_plots}.")
        return
    if plt is None:
        print("Skipping curtain plots because matplotlib is not installed.")
        return
    if curtain_rows <= 0:
        print(f"Skipping {split_name} curtain plots because curtain_rows={curtain_rows}.")
        return
    if not files:
        print(f"Skipping {split_name} curtain plots because the split is empty.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    saved = 0
    for file_idx in rng.permutation(len(files)):
        if saved >= num_random_plots:
            break
        meta = files[int(file_idx)]
        row_indices, targets, preds = _predict_masks_for_file(
            model=model,
            meta=meta,
            base_mean=base_mean,
            base_std=base_std,
            grid_mean=grid_mean,
            grid_std=grid_std,
            eval_batch_size=eval_batch_size,
            device=device,
        )
        window = _choose_contiguous_window(row_indices=row_indices, window_size=curtain_rows, rng=rng)
        if window is None:
            continue

        start, stop = window
        row_window = row_indices[start:stop]
        sample_path = out_dir / (
            f"{split_name}_curtain_{saved + 1:02d}_{_safe_stem(meta.feather_path)}"
            f"_rows_{int(row_window[0]):06d}_{int(row_window[-1]):06d}.png"
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        for ax, image, title in zip(
            axes,
            (targets[start:stop], preds[start:stop]),
            ("Ground Truth", "Prediction Probability"),
        ):
            ax.imshow(image.T, aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("Feather row index")
        axes[0].set_ylabel("Cloud mask column")
        x_ticks = np.linspace(0, curtain_rows - 1, num=min(5, curtain_rows), dtype=int)
        x_labels = [str(int(row_window[idx])) for idx in x_ticks]
        for ax in axes:
            ax.set_xticks(x_ticks, labels=x_labels)
        axes[1].tick_params(axis="y", labelleft=False)
        fig.suptitle(
            f"{split_name.title()} cloud-mask curtain | {meta.feather_path.name} | "
            f"rows {int(row_window[0])}-{int(row_window[-1])}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(sample_path, dpi=160)
        plt.close(fig)
        saved += 1
        print(f"Saved curtain plot: {sample_path}")

    if saved == 0:
        print(
            f"Skipping {split_name} curtain plots because no file had a contiguous "
            f"{curtain_rows}-row span available."
        )
    elif saved < num_random_plots:
        print(f"Saved {saved} {split_name} curtain plot(s); requested {num_random_plots}.")


def train_model(
    base_train: np.ndarray,
    grid_train: np.ndarray,
    y_train: np.ndarray,
    base_test: np.ndarray,
    grid_test: np.ndarray,
    y_test: np.ndarray,
    test_files: Sequence[FileMeta],
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    hidden_dims: Optional[Sequence[int]],
    dropout: float,
    use_pos_weight: bool,
    seed: int,
    device: str,
    early_stop_patience: int,
    early_stop_min_delta: float,
    plot_random_curtain_count: int,
    plot_curtain_rows: int,
    plot_dir: Path,
) -> Dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed)
    np.random.seed(seed)

    base_mean = base_train.mean(axis=0, keepdims=True)
    base_std = base_train.std(axis=0, keepdims=True)
    base_std = np.where(base_std < 1e-6, 1.0, base_std)

    grid_mean = grid_train.mean(axis=0, keepdims=True)
    grid_std = grid_train.std(axis=0, keepdims=True)
    grid_std = np.where(grid_std < 1e-6, 1.0, grid_std)

    base_train_n = (base_train - base_mean) / base_std
    base_test_n = (base_test - base_mean) / base_std
    grid_train_n = (grid_train - grid_mean) / grid_std
    grid_test_n = (grid_test - grid_mean) / grid_std

    base_train_t = torch.from_numpy(base_train_n.astype(np.float32))
    grid_train_t = torch.from_numpy(grid_train_n.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    base_test_t = torch.from_numpy(base_test_n.astype(np.float32))
    grid_test_t = torch.from_numpy(grid_test_n.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.float32))

    train_ds = TensorDataset(base_train_t, grid_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = Grid3x3EmbeddingMLP(
        base_dim=base_train.shape[1],
        grid_shape=grid_train.shape[1:],
        output_dim=y_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    print(
        "Classifier config | "
        f"grid_shape={tuple(int(v) for v in grid_train.shape[1:])} | "
        f"grid_input_dim={model.grid_input_dim} | input_dim={model.input_dim} | "
        f"hidden_dims={list(model.hidden_dims)} | dropout={dropout:.3f}"
    )

    if use_pos_weight:
        pos = y_train_t.sum(dim=0)
        neg = y_train_t.shape[0] - pos
        pos_weight = (neg / torch.clamp(pos, min=1.0)).clamp(min=1.0, max=20.0).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_epoch = 0
    best_score = float("-inf")
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_train_metrics: Optional[Dict[str, float]] = None
    best_test_metrics: Optional[Dict[str, float]] = None
    best_train_loss = float("nan")
    best_train_eval_loss = float("nan")
    best_test_loss = float("nan")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for base_b, grid_b, yb in train_dl:
            base_b = base_b.to(device)
            grid_b = grid_b.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(base_b, grid_b)
            loss = loss_fn(logits, yb)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * base_b.shape[0]
            total_count += base_b.shape[0]

        train_loss = total_loss / max(total_count, 1)
        train_eval_loss, train_metrics, _ = _evaluate_in_batches(
            model=model,
            loss_fn=loss_fn,
            base_x=base_train_t,
            grid_x=grid_train_t,
            y=y_train_t,
            batch_size=eval_batch_size,
            device=device,
            search_iou_threshold=True,
        )
        test_loss, test_metrics, _ = _evaluate_in_batches(
            model=model,
            loss_fn=loss_fn,
            base_x=base_test_t,
            grid_x=grid_test_t,
            y=y_test_t,
            batch_size=eval_batch_size,
            device=device,
            iou_threshold=train_metrics["iou_threshold"],
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

        score = float(test_metrics["iou_mean"])
        if score > best_score + early_stop_min_delta:
            best_epoch = epoch
            best_score = score
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_metrics = dict(train_metrics)
            best_test_metrics = dict(test_metrics)
            best_train_loss = float(train_loss)
            best_train_eval_loss = float(train_eval_loss)
            best_test_loss = float(test_loss)
            epochs_without_improvement = 0
            print(f"  New best validation IoU: {best_score:.4f} at epoch {best_epoch:03d}")
        else:
            epochs_without_improvement += 1
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch:03d}; best validation IoU "
                    f"{best_score:.4f} was at epoch {best_epoch:03d}."
                )
                break

    if best_state_dict is None or best_train_metrics is None or best_test_metrics is None:
        raise RuntimeError("Training finished without recording a best model state.")
    model.load_state_dict(best_state_dict)
    train_metrics = best_train_metrics
    test_metrics = best_test_metrics
    print(
        "Using best epoch | "
        f"epoch={best_epoch:03d} | train_loss={best_train_loss:.5f} | "
        f"train_eval_loss={best_train_eval_loss:.5f} | test_loss={best_test_loss:.5f} | "
        f"test_iou_mean={best_score:.4f} @thr={test_metrics['iou_threshold']:.3f}"
    )

    _save_random_curtain_plots(
        out_dir=plot_dir,
        split_name="validation",
        model=model,
        files=test_files,
        base_mean=base_mean,
        base_std=base_std,
        grid_mean=grid_mean,
        grid_std=grid_std,
        eval_batch_size=eval_batch_size,
        device=device,
        num_random_plots=plot_random_curtain_count,
        curtain_rows=plot_curtain_rows,
        seed=seed,
    )

    return {
        "model": model,
        "base_mean": base_mean.astype(np.float32),
        "base_std": base_std.astype(np.float32),
        "grid_mean": grid_mean.astype(np.float32),
        "grid_std": grid_std.astype(np.float32),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "grid_shape": tuple(int(v) for v in grid_train.shape[1:]),
        "grid_input_dim": int(model.grid_input_dim),
        "input_dim": int(model.input_dim),
        "hidden_dims": list(model.hidden_dims),
        "dropout": float(dropout),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
    }


def _save_artifacts(
    out_dir: Path,
    model: Grid3x3EmbeddingMLP,
    base_mean: np.ndarray,
    base_std: np.ndarray,
    grid_mean: np.ndarray,
    grid_std: np.ndarray,
    train_files: Sequence[FileMeta],
    test_files: Sequence[FileMeta],
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
    best_epoch: int,
    best_score: float,
    early_stop_patience: int,
    early_stop_min_delta: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "multilabel_grid3x3_mlp.pt"
    stats_path = out_dir / "feature_stats_3x3.npz"
    split_path = out_dir / "file_split.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "output_dim": 40,
            "grid_shape": list(model.grid_shape),
            "cell_shape": list(model.cell_shape),
            "grid_input_dim": int(model.grid_input_dim),
            "input_dim": int(input_dim),
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "base_features": BASE_FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
        },
        model_path,
    )
    np.savez_compressed(
        stats_path,
        base_mean=base_mean,
        base_std=base_std,
        grid_mean=grid_mean,
        grid_std=grid_std,
    )

    rows = []
    for m in train_files:
        rows.append({"split": "train", "file": str(m.feather_path), "file_time_utc": str(m.file_time)})
    for m in test_files:
        rows.append({"split": "test", "file": str(m.feather_path), "file_time_utc": str(m.file_time)})
    pd.DataFrame(rows).to_csv(split_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved feature stats: {stats_path}")
    print(f"Saved split manifest: {split_path}")


def main() -> int:
    def _parse_hidden_dims(value: str) -> List[int]:
        dims = [int(part.strip()) for part in value.split(",") if part.strip()]
        if not dims or any(dim <= 0 for dim in dims):
            raise argparse.ArgumentTypeError("hidden dims must be a comma-separated list of positive integers")
        return dims

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feather-root", type=str, default=FEATHER_ROOT)
    parser.add_argument("--embedding-dir", type=str, default=EMBEDDING_3X3_DIR)
    parser.add_argument(
        "--train-files",
        type=int,
        default=0,
        help="Number of pre-cutoff files to use for training. 0 means use all available.",
    )
    parser.add_argument(
        "--test-files",
        type=int,
        default=0,
        help="Number of on/after-cutoff files to use for testing. 0 means use all available.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--hidden-dims",
        type=_parse_hidden_dims,
        default=None,
        help="Comma-separated classifier hidden dims. Default: auto-scale from flattened input_dim.",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--sample-ratio", type=float, default=0.5)
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=0,
        help="Optional hard cap on sampled rows per file. 0 means no cap.",
    )
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Stop after this many epochs without validation IoU improvement. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation IoU improvement needed to reset early-stop patience.",
    )
    parser.add_argument(
        "--test-start-time",
        type=str,
        default=DEFAULT_TEST_START_TIME,
        help="UTC timestamp cutoff for test-file eligibility. Files on or after this time are eligible for test.",
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=str, default=LOG_DIR)
    parser.add_argument(
        "--plot-random-curtain-count",
        "--plot-random-iou-count",
        dest="plot_random_curtain_count",
        type=int,
        default=6,
        help="Number of random validation curtain plots to save in --plot-dir. Set 0 to disable.",
    )
    parser.add_argument(
        "--plot-curtain-rows",
        type=int,
        default=100,
        help="Number of contiguous feather rows to show in each validation curtain plot.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.feather_root:
        raise ValueError("FEATHER_ROOT is not set (or pass --feather-root).")
    if not args.embedding_dir:
        raise ValueError("EMBEDDING_3X3_OUTPUT_DIR is not set (or pass --embedding-dir).")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0.")

    feather_root = Path(args.feather_root)
    embedding_dir = Path(args.embedding_dir)
    if not feather_root.exists():
        raise FileNotFoundError(f"Feather root does not exist: {feather_root}")
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding dir does not exist: {embedding_dir}")

    metas = discover_files(feather_root, embedding_dir)
    test_start_time = _parse_utc_timestamp(args.test_start_time)
    train_files, test_files = select_train_test_files(
        metas=metas,
        train_files=args.train_files,
        test_files=args.test_files,
        seed=args.seed,
        test_start_time=test_start_time,
    )
    print(f"Selected train files: {len(train_files)}")
    print(f"Selected test files: {len(test_files)}")
    print(f"Test file cutoff:    {test_start_time}")
    print(f"Train max file_time: {max(m.file_time for m in train_files)}")
    print(f"Test min file_time:  {min(m.file_time for m in test_files)}")

    max_samples_per_file = None if args.max_samples_per_file <= 0 else int(args.max_samples_per_file)
    base_train, grid_train, y_train = load_dataset(
        train_files,
        sample_ratio=args.sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=args.seed,
    )
    base_test, grid_test, y_test = load_dataset(
        test_files,
        sample_ratio=args.sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=args.seed,
    )
    print(
        f"Train samples: {base_train.shape[0]} | base_dim: {base_train.shape[1]} | "
        f"grid_shape: {tuple(int(v) for v in grid_train.shape[1:])} | targets: {y_train.shape[1]}"
    )
    print(f"Test samples:  {base_test.shape[0]}")

    fit = train_model(
        base_train=base_train,
        grid_train=grid_train,
        y_train=y_train,
        base_test=base_test,
        grid_test=grid_test,
        y_test=y_test,
        test_files=test_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_pos_weight=(not args.no_pos_weight),
        seed=args.seed,
        device=args.device,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        plot_random_curtain_count=args.plot_random_curtain_count,
        plot_curtain_rows=args.plot_curtain_rows,
        plot_dir=Path(args.plot_dir),
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
        f"test_iou_mean={fit['test_metrics']['iou_mean']:.4f} @thr={fit['test_metrics']['iou_threshold']:.3f}, "
        f"best_epoch={fit['best_epoch']}"
    )

    _save_artifacts(
        out_dir=Path(args.output_dir),
        model=fit["model"],
        base_mean=fit["base_mean"],
        base_std=fit["base_std"],
        grid_mean=fit["grid_mean"],
        grid_std=fit["grid_std"],
        train_files=train_files,
        test_files=test_files,
        train_metrics=fit["train_metrics"],
        test_metrics=fit["test_metrics"],
        input_dim=fit["input_dim"],
        hidden_dims=fit["hidden_dims"],
        dropout=fit["dropout"],
        best_epoch=fit["best_epoch"],
        best_score=fit["best_score"],
        early_stop_patience=fit["early_stop_patience"],
        early_stop_min_delta=fit["early_stop_min_delta"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
