#!/usr/bin/env python3
"""Train a multi-label (40 binary targets) model from Feather + embedding NPZ files."""

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
EMBEDDING_DIR = os.getenv("EMBEDDING_OUTUT_DIR", os.getenv("EMBEDDING_OUTPUT_DIR", str(PROJECT_ROOT / "embeddings")))
OUTPUT_DIR = str(PROJECT_ROOT / "model_outputs")
LOG_DIR = str(PROJECT_ROOT / "logs")
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


def _resolve_hidden_dims(input_dim: int, hidden_dims: Optional[Sequence[int]] = None) -> List[int]:
    if hidden_dims:
        dims = [int(d) for d in hidden_dims if int(d) > 0]
        if dims:
            return dims

    # Keep the first projection wide enough that larger embeddings are not
    # immediately squeezed into the old 512-d bottleneck.
    first = min(2048, max(512, input_dim // 2))
    second = min(1024, max(256, first // 2))
    if second >= first:
        second = max(256, first // 2)
    return [first, second]


class MultiLabelMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 40,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        resolved_hidden_dims = _resolve_hidden_dims(input_dim=input_dim, hidden_dims=hidden_dims)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in resolved_hidden_dims:
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
        self.hidden_dims = tuple(resolved_hidden_dims)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    print(f"Candidate files with embeddings: {len(metas)}")
    print(f"Skipped files (missing embedding): {skipped_no_npz}")
    print(f"Skipped files (empty feather): {skipped_empty}")
    return metas


def select_train_test_files(
    metas: Sequence[FileMeta],
    train_files: int,
    test_files: int,
    seed: int,
    test_start_time: pd.Timestamp,
) -> Tuple[List[FileMeta], List[FileMeta]]:
    if len(metas) < (train_files + test_files):
        raise ValueError(
            f"Not enough files with embeddings: have {len(metas)}, need at least {train_files + test_files}."
        )

    rng = random.Random(seed)
    sorted_metas = sorted(metas, key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path)))
    test_pool = [m for m in sorted_metas if m.file_time >= test_start_time]
    if len(test_pool) < test_files:
        raise ValueError(
            f"Not enough test files on or after {test_start_time}. "
            f"Need {test_files}, but only found {len(test_pool)}."
        )
    test_selected = rng.sample(test_pool, k=test_files)

    train_pool = [m for m in sorted_metas if m.file_time < test_start_time]
    if len(train_pool) < train_files:
        raise ValueError(
            f"Not enough earlier files for training. "
            f"Need {train_files}, but only {len(train_pool)} files are earlier than test_start_time={test_start_time}."
        )
    train_selected = rng.sample(train_pool, k=train_files)
    return (
        sorted(train_selected, key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path))),
        sorted(test_selected, key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path))),
    )


def load_one_file_samples(meta: FileMeta) -> Tuple[np.ndarray, np.ndarray]:
    required_cols = ["timestamp_0"] + BASE_FEATURE_COLUMNS + TARGET_COLUMNS
    df = pd.read_feather(meta.feather_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {meta.feather_path.name}: {missing}")

    with np.load(meta.npz_path) as data:
        if "emb_all_levels" not in data or "row_indices" not in data:
            raise ValueError(f"Missing arrays in {meta.npz_path.name}; expected emb_all_levels and row_indices.")
        emb = data["emb_all_levels"]
        rows = data["row_indices"]

    if emb.size == 0 or rows.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 40), dtype=np.float32)

    rows = rows.astype(np.int64)
    valid = (rows >= 0) & (rows < len(df))
    if not np.all(valid):
        rows = rows[valid]
        emb = emb[valid]
    if rows.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 40), dtype=np.float32)

    emb_flat = emb.reshape(emb.shape[0], -1).astype(np.float32)
    base = df.iloc[rows][BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    y = df.iloc[rows][TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    # Enforce binary targets for BCEWithLogitsLoss.
    y = (y > 0.5).astype(np.float32)
    x = np.concatenate([base, emb_flat], axis=1)
    return x, y


def load_dataset(files: Sequence[FileMeta]) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for meta in files:
        x, y = load_one_file_samples(meta)
        if x.size == 0:
            continue
        x_parts.append(x)
        y_parts.append(y)
    if not x_parts:
        raise ValueError("No row samples available after loading selected files.")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


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


def _evaluate_in_batches(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: str,
    iou_threshold: Optional[float] = None,
    search_iou_threshold: bool = False,
    collect_details: bool = False,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, torch.Tensor]]]:
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_count = 0
    logits_cpu_parts: List[torch.Tensor] = []
    targets_cpu_parts: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]
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
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

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
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
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
    plot_random_iou_count: int,
    plot_dir: Path,
) -> Dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)

    x_train_n = (x_train - x_mean) / x_std
    x_test_n = (x_test - x_mean) / x_std

    x_train_t = torch.from_numpy(x_train_n.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    x_test_t = torch.from_numpy(x_test_n.astype(np.float32))
    y_test_t = torch.from_numpy(y_test.astype(np.float32))

    train_ds = TensorDataset(x_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    resolved_hidden_dims = _resolve_hidden_dims(input_dim=x_train_t.shape[1], hidden_dims=hidden_dims)
    model = MultiLabelMLP(
        input_dim=x_train_t.shape[1],
        output_dim=y_train_t.shape[1],
        hidden_dims=resolved_hidden_dims,
        dropout=dropout,
    ).to(device)
    print(f"Classifier hidden dims: {resolved_hidden_dims} | dropout={dropout:.3f}")
    if use_pos_weight:
        pos = y_train_t.sum(dim=0)
        neg = y_train_t.shape[0] - pos
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
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]

        train_loss = total_loss / max(total_count, 1)
        train_eval_loss, train_metrics, _ = _evaluate_in_batches(
            model=model,
            loss_fn=loss_fn,
            x=x_train_t,
            y=y_train_t,
            batch_size=eval_batch_size,
            device=device,
            search_iou_threshold=True,
        )
        test_loss, test_metrics, test_eval_details = _evaluate_in_batches(
            model=model,
            loss_fn=loss_fn,
            x=x_test_t,
            y=y_test_t,
            batch_size=eval_batch_size,
            device=device,
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
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_eval_details": test_eval_details,
        "input_dim": int(x_train.shape[1]),
        "hidden_dims": list(resolved_hidden_dims),
        "dropout": float(dropout),
    }


def _save_artifacts(
    out_dir: Path,
    model: nn.Module,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    train_files: Sequence[FileMeta],
    test_files: Sequence[FileMeta],
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    input_dim: int,
    hidden_dims: Sequence[int],
    dropout: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "multilabel_mlp.pt"
    stats_path = out_dir / "feature_stats.npz"
    split_path = out_dir / "file_split.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": 40,
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "base_features": BASE_FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
        model_path,
    )
    np.savez_compressed(stats_path, x_mean=x_mean, x_std=x_std)

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
    parser.add_argument("--embedding-dir", type=str, default=EMBEDDING_DIR)
    parser.add_argument("--train-files", type=int, default=100)
    parser.add_argument("--test-files", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--hidden-dims",
        type=_parse_hidden_dims,
        default=None,
        help="Comma-separated classifier hidden dims. Default: auto-scale from input_dim, e.g. 1024,512.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-start-time",
        type=str,
        default=DEFAULT_TEST_START_TIME,
        help="UTC timestamp cutoff for test-file eligibility. Files on or after this time are eligible for test.",
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=str, default=LOG_DIR)
    parser.add_argument(
        "--plot-random-iou-count",
        type=int,
        default=6,
        help="Number of random validation/test samples to save as IoU plots in --plot-dir. Set 0 to disable.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.feather_root:
        raise ValueError("FEATHER_ROOT is not set (or pass --feather-root).")
    if not args.embedding_dir:
        raise ValueError("EMBEDDING_OUTUT_DIR/EMBEDDING_OUTPUT_DIR is not set (or pass --embedding-dir).")

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
    print(f"Test file cutoff:   {test_start_time}")
    print(f"Train max file_time: {max(m.file_time for m in train_files)}")
    print(f"Test min file_time:  {min(m.file_time for m in test_files)}")

    x_train, y_train = load_dataset(train_files)
    x_test, y_test = load_dataset(test_files)
    print(f"Train samples: {x_train.shape[0]} | Feature dim: {x_train.shape[1]} | Targets: {y_train.shape[1]}")
    print(f"Test samples:  {x_test.shape[0]}")

    fit = train_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
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
        plot_random_iou_count=args.plot_random_iou_count,
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
        f"test_iou_mean={fit['test_metrics']['iou_mean']:.4f} @thr={fit['test_metrics']['iou_threshold']:.3f}"
    )

    _save_artifacts(
        out_dir=Path(args.output_dir),
        model=fit["model"],
        x_mean=fit["x_mean"],
        x_std=fit["x_std"],
        train_files=train_files,
        test_files=test_files,
        train_metrics=fit["train_metrics"],
        test_metrics=fit["test_metrics"],
        input_dim=fit["input_dim"],
        hidden_dims=fit["hidden_dims"],
        dropout=fit["dropout"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
