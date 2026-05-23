#!/usr/bin/env python3
"""Train forecast labels from existing Feather rows plus precomputed embeddings.

This reuses embeddings generated for source hour ``s``. For a target row at
rounded hour ``t``, the default forecast lag finds the existing embedding at
``s = t - 6h``. Since that embedding was generated from ERA5 inputs
``(s - 6h, s)``, the model sees ``(t - 12h, t - 6h)`` while predicting labels
from the target row at ``t``.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import train_multilabel_from_feather_embeddings as base


OUTPUT_DIR = str(base.PROJECT_ROOT / "model_outputs_forecast")
LOG_DIR = str(base.PROJECT_ROOT / "logs_forecast")


@dataclass(frozen=True)
class SourceRef:
    npz_path: Path
    emb_index: int
    source_row: int
    base_features: Tuple[float, ...]


@dataclass
class SourceIndex:
    refs: Dict[Tuple[int, float, float], SourceRef]
    source_file_count: int
    source_embedding_count: int
    duplicate_key_count: int


_FORECAST_SOURCE_INDEX: Optional[SourceIndex] = None
_FORECAST_LAG_HOURS = 6
_FORECAST_BASE_FEATURE_SOURCE = "source"


def _timestamp_unit_from_series(values: pd.Series) -> str:
    non_null = values.dropna()
    if len(non_null) == 0:
        return "s"
    return base._timestamp_unit_from_value(int(non_null.iloc[0]))


def _rounded_hours(timestamp_values: pd.Series) -> pd.Series:
    unit = _timestamp_unit_from_series(timestamp_values)
    dt = pd.to_datetime(timestamp_values, unit=unit, utc=True)
    hour_floor = dt.dt.floor("h")
    round_up = (dt - hour_floor) > pd.Timedelta(minutes=30)
    return hour_floor + pd.to_timedelta(round_up.astype(int), unit="h")


def _hour_key(hours: pd.Series) -> np.ndarray:
    return hours.astype("int64").to_numpy(copy=False)


def _lat_lon_keys(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.round(df["Latitude_0"].to_numpy(dtype=np.float64, copy=False), 6)
    lon = np.round(df["Longitude_0"].to_numpy(dtype=np.float64, copy=False), 6)
    return lat, lon


def _sample_file_metas(files: Sequence[base.FileMeta], count: int, seed: int) -> List[base.FileMeta]:
    if count <= 0 or count >= len(files):
        return list(files)
    rng = random.Random(seed)
    return sorted(
        rng.sample(list(files), k=count),
        key=lambda x: (x.file_time, str(x.feather_path), str(x.npz_path)),
    )


def build_source_index(files: Sequence[base.FileMeta]) -> SourceIndex:
    refs: Dict[Tuple[int, float, float], SourceRef] = {}
    source_embedding_count = 0
    duplicate_key_count = 0
    required_cols = ["timestamp_0"] + base.BASE_FEATURE_COLUMNS

    for meta in files:
        df = pd.read_feather(meta.feather_path, columns=required_cols)
        if len(df) == 0:
            continue

        with np.load(meta.npz_path) as data:
            if "row_indices" not in data or "emb_all_levels" not in data:
                raise ValueError(f"Missing arrays in {meta.npz_path.name}; expected emb_all_levels and row_indices.")
            rows = data["row_indices"].astype(np.int64, copy=False)
            emb_count = int(data["emb_all_levels"].shape[0])

        valid = (rows >= 0) & (rows < len(df)) & (np.arange(rows.shape[0]) < emb_count)
        if not np.all(valid):
            rows = rows[valid]
        if rows.size == 0:
            continue

        source_embedding_count += int(rows.size)
        source_df = df.iloc[rows].reset_index(drop=True)
        hours = _hour_key(_rounded_hours(source_df["timestamp_0"]))
        lat_keys, lon_keys = _lat_lon_keys(source_df)
        base_values = source_df[base.BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)

        for emb_index, source_row, hour_ns, lat_key, lon_key, base_row in zip(
            np.nonzero(valid)[0] if valid.shape[0] != rows.shape[0] else np.arange(rows.shape[0]),
            rows,
            hours,
            lat_keys,
            lon_keys,
            base_values,
        ):
            key = (int(hour_ns), float(lat_key), float(lon_key))
            if key in refs:
                duplicate_key_count += 1
                continue
            refs[key] = SourceRef(
                npz_path=meta.npz_path,
                emb_index=int(emb_index),
                source_row=int(source_row),
                base_features=tuple(float(v) for v in base_row),
            )

    return SourceIndex(
        refs=refs,
        source_file_count=len(files),
        source_embedding_count=source_embedding_count,
        duplicate_key_count=duplicate_key_count,
    )


def _fetch_embeddings(refs: Sequence[SourceRef]) -> np.ndarray:
    by_npz: Dict[Path, List[Tuple[int, int]]] = {}
    for out_index, ref in enumerate(refs):
        by_npz.setdefault(ref.npz_path, []).append((out_index, ref.emb_index))

    flat_rows: List[Optional[np.ndarray]] = [None] * len(refs)
    for npz_path, positions in by_npz.items():
        with np.load(npz_path) as data:
            emb = data["emb_all_levels"]
            for out_index, emb_index in positions:
                flat_rows[out_index] = emb[int(emb_index)].reshape(-1).astype(np.float32, copy=False)

    missing = [idx for idx, row in enumerate(flat_rows) if row is None]
    if missing:
        raise RuntimeError(f"Internal embedding lookup error; missing {len(missing)} fetched row(s).")
    return np.stack([row for row in flat_rows if row is not None], axis=0)


def _load_forecast_one_file_arrays(
    meta: base.FileMeta,
    sample_ratio: float = 1.0,
    max_samples_per_file: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _FORECAST_SOURCE_INDEX is None:
        raise RuntimeError("Forecast source index has not been built.")

    required_cols = ["timestamp_0"] + base.BASE_FEATURE_COLUMNS + base.TARGET_COLUMNS
    df = pd.read_feather(meta.feather_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {meta.feather_path.name}: {missing}")
    if len(df) == 0:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, np.empty((0, 40), dtype=np.float32), np.empty((0,), dtype=np.int64)

    target_hours = _rounded_hours(df["timestamp_0"])
    source_hours = target_hours - pd.Timedelta(hours=_FORECAST_LAG_HOURS)
    hour_keys = _hour_key(source_hours)
    lat_keys, lon_keys = _lat_lon_keys(df)

    target_rows: List[int] = []
    refs: List[SourceRef] = []
    for row_index, hour_ns, lat_key, lon_key in zip(np.arange(len(df)), hour_keys, lat_keys, lon_keys):
        ref = _FORECAST_SOURCE_INDEX.refs.get((int(hour_ns), float(lat_key), float(lon_key)))
        if ref is None:
            continue
        target_rows.append(int(row_index))
        refs.append(ref)

    if not refs:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, np.empty((0, 40), dtype=np.float32), np.empty((0,), dtype=np.int64)

    if not (0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    sample_indices = np.arange(len(refs), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if sample_ratio < 1.0:
        count = max(1, int(len(sample_indices) * sample_ratio))
        sample_indices = rng.choice(sample_indices, size=count, replace=False)
    if max_samples_per_file is not None and max_samples_per_file > 0 and len(sample_indices) > max_samples_per_file:
        sample_indices = rng.choice(sample_indices, size=max_samples_per_file, replace=False)
    sample_indices = np.sort(sample_indices)

    target_rows_arr = np.asarray(target_rows, dtype=np.int64)[sample_indices]
    selected_refs = [refs[int(idx)] for idx in sample_indices]

    if _FORECAST_BASE_FEATURE_SOURCE == "target":
        base_features = df.iloc[target_rows_arr][base.BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    else:
        base_features = np.asarray([ref.base_features for ref in selected_refs], dtype=np.float32)

    emb_flat = _fetch_embeddings(selected_refs)
    y = df.iloc[target_rows_arr][base.TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    y = (y > 0.5).astype(np.float32)
    x = np.concatenate([base_features, emb_flat], axis=1)
    return x, y, target_rows_arr


def load_forecast_dataset(
    files: Sequence[base.FileMeta],
    sample_ratio: float = 1.0,
    max_samples_per_file: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    matched_files = 0
    for file_idx, meta in enumerate(files):
        x, y, _ = _load_forecast_one_file_arrays(
            meta,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed + file_idx,
        )
        if x.size == 0:
            continue
        matched_files += 1
        x_parts.append(x)
        y_parts.append(y)

    if not x_parts:
        raise ValueError("No forecast samples available after matching target rows to lagged embeddings.")
    print(f"Forecast matched files with samples: {matched_files}/{len(files)}")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def _save_forecast_config(
    out_dir: Path,
    source_index: SourceIndex,
    forecast_lag_hours: int,
    base_feature_source: str,
    source_pool: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "forecast_training_config.json"
    payload = {
        "forecast_lag_hours": int(forecast_lag_hours),
        "embedding_source_hour": "target_rounded_hour - forecast_lag_hours",
        "embedding_input_window": [
            "target_rounded_hour - forecast_lag_hours - 6h",
            "target_rounded_hour - forecast_lag_hours",
        ],
        "base_feature_source": base_feature_source,
        "source_pool": source_pool,
        "source_file_count": int(source_index.source_file_count),
        "source_embedding_count": int(source_index.source_embedding_count),
        "unique_source_keys": int(len(source_index.refs)),
        "duplicate_source_keys_skipped": int(source_index.duplicate_key_count),
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved forecast config: {config_path}")


def main() -> int:
    def _parse_hidden_dims(value: str) -> List[int]:
        dims = [int(part.strip()) for part in value.split(",") if part.strip()]
        if not dims or any(dim <= 0 for dim in dims):
            raise argparse.ArgumentTypeError("hidden dims must be a comma-separated list of positive integers")
        return dims

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feather-root", type=str, default=base.FEATHER_ROOT)
    parser.add_argument("--embedding-dir", type=str, default=base.EMBEDDING_DIR)
    parser.add_argument("--train-files", type=int, default=1000)
    parser.add_argument("--validation-files", type=int, default=50)
    parser.add_argument("--test-files", type=int, default=50)
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
    parser.add_argument("--sample-ratio", type=float, default=0.3)
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=0,
        help="Optional hard cap on sampled target rows per file after forecast matching. 0 means no cap.",
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
        default=base.DEFAULT_TEST_START_TIME,
        help="UTC timestamp cutoff for validation/test-file eligibility. Files on or after this time are eligible.",
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=str, default=LOG_DIR)
    parser.add_argument(
        "--plot-random-curtain-count",
        "--plot-random-iou-count",
        dest="plot_random_curtain_count",
        type=int,
        default=10,
        help="Number of random validation curtain plots to save in --plot-dir. Set 0 to disable.",
    )
    parser.add_argument(
        "--plot-file-prefix",
        type=str,
        default="embedding_forecast",
        help="Prefix added to validation curtain PNG filenames.",
    )
    parser.add_argument(
        "--plot-curtain-rows",
        type=int,
        default=100,
        help="Number of contiguous feather target rows to show in each validation curtain plot.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--forecast-lag-hours",
        type=int,
        default=6,
        help="Match target rows to embeddings from target rounded hour minus this many hours.",
    )
    parser.add_argument(
        "--base-feature-source",
        choices=("source", "target"),
        default="source",
        help="Use base time/location features from the lagged source row or from the target row.",
    )
    parser.add_argument(
        "--source-pool",
        choices=("all", "selected-splits"),
        default="all",
        help="Which embedding files may provide lagged source features. 'all' only reads features, not labels.",
    )
    parser.add_argument(
        "--max-source-files",
        type=int,
        default=0,
        help="Optional cap on source files used to build the lookup index. 0 means no cap.",
    )
    args = parser.parse_args()

    if not args.feather_root:
        raise ValueError("FEATHER_ROOT is not set (or pass --feather-root).")
    if not args.embedding_dir:
        raise ValueError("EMBEDDING_OUTUT_DIR/EMBEDDING_OUTPUT_DIR is not set (or pass --embedding-dir).")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0.")
    if args.forecast_lag_hours <= 0:
        raise ValueError("--forecast-lag-hours must be > 0.")

    feather_root = Path(args.feather_root)
    embedding_dir = Path(args.embedding_dir)
    if not feather_root.exists():
        raise FileNotFoundError(f"Feather root does not exist: {feather_root}")
    if not embedding_dir.exists():
        raise FileNotFoundError(f"Embedding dir does not exist: {embedding_dir}")

    metas = base.discover_files(feather_root, embedding_dir)
    test_start_time = base._parse_utc_timestamp(args.test_start_time)
    train_files, validation_files, test_files = base.select_train_validation_test_files(
        metas=metas,
        train_files=args.train_files,
        validation_files=args.validation_files,
        test_files=args.test_files,
        seed=args.seed,
        test_start_time=test_start_time,
    )

    if args.source_pool == "all":
        source_files = _sample_file_metas(metas, args.max_source_files, args.seed + 1000)
    else:
        source_files = list(train_files) + list(validation_files) + list(test_files)
        source_files = _sample_file_metas(source_files, args.max_source_files, args.seed + 1000)

    print(f"Selected train files: {len(train_files)}")
    print(f"Selected validation files: {len(validation_files)}")
    print(f"Selected test files: {len(test_files)}")
    print(f"Test file cutoff:   {test_start_time}")
    print(f"Train max file_time: {max(m.file_time for m in train_files)}")
    print(f"Validation min file_time: {min(m.file_time for m in validation_files)}")
    print(f"Test min file_time:  {min(m.file_time for m in test_files)}")
    print(f"Forecast lag hours: {args.forecast_lag_hours}")
    print(f"Forecast source pool: {args.source_pool} ({len(source_files)} file(s))")
    print(f"Base feature source: {args.base_feature_source}")

    global _FORECAST_SOURCE_INDEX, _FORECAST_LAG_HOURS, _FORECAST_BASE_FEATURE_SOURCE
    _FORECAST_SOURCE_INDEX = build_source_index(source_files)
    _FORECAST_LAG_HOURS = int(args.forecast_lag_hours)
    _FORECAST_BASE_FEATURE_SOURCE = args.base_feature_source
    base._load_one_file_arrays = _load_forecast_one_file_arrays

    print(
        "Built forecast source index | "
        f"source_embeddings={_FORECAST_SOURCE_INDEX.source_embedding_count} | "
        f"unique_keys={len(_FORECAST_SOURCE_INDEX.refs)} | "
        f"duplicates_skipped={_FORECAST_SOURCE_INDEX.duplicate_key_count}"
    )

    max_samples_per_file = None if args.max_samples_per_file <= 0 else int(args.max_samples_per_file)
    x_train, y_train = load_forecast_dataset(
        train_files,
        sample_ratio=args.sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=args.seed,
    )
    x_validation, y_validation = load_forecast_dataset(
        validation_files,
        sample_ratio=args.sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=args.seed,
    )
    x_test, y_test = load_forecast_dataset(
        test_files,
        sample_ratio=args.sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=args.seed,
    )
    print(f"Train samples: {x_train.shape[0]} | Feature dim: {x_train.shape[1]} | Targets: {y_train.shape[1]}")
    print(f"Validation samples: {x_validation.shape[0]}")
    print(f"Test samples:  {x_test.shape[0]}")

    fit = base.train_model(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
        validation_files=validation_files,
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
        plot_file_prefix=args.plot_file_prefix,
    )

    print(
        "Final metrics | "
        f"train_acc={fit['train_metrics']['accuracy']:.4f}, "
        f"validation_acc={fit['validation_metrics']['accuracy']:.4f}, "
        f"test_acc={fit['test_metrics']['accuracy']:.4f}, "
        f"train_iou_mean={fit['train_metrics']['iou_mean']:.4f} @thr={fit['train_metrics']['iou_threshold']:.3f}, "
        f"validation_iou_mean={fit['validation_metrics']['iou_mean']:.4f} @thr={fit['validation_metrics']['iou_threshold']:.3f}, "
        f"test_iou_mean={fit['test_metrics']['iou_mean']:.4f} @thr={fit['test_metrics']['iou_threshold']:.3f}, "
        f"best_epoch={fit['best_epoch']}"
    )

    out_dir = Path(args.output_dir)
    base._save_artifacts(
        out_dir=out_dir,
        model=fit["model"],
        x_mean=fit["x_mean"],
        x_std=fit["x_std"],
        train_files=train_files,
        validation_files=validation_files,
        test_files=test_files,
        train_metrics=fit["train_metrics"],
        validation_metrics=fit["validation_metrics"],
        test_metrics=fit["test_metrics"],
        input_dim=fit["input_dim"],
        hidden_dims=fit["hidden_dims"],
        dropout=fit["dropout"],
        best_epoch=fit["best_epoch"],
        best_score=fit["best_score"],
        early_stop_patience=fit["early_stop_patience"],
        early_stop_min_delta=fit["early_stop_min_delta"],
    )
    _save_forecast_config(
        out_dir=out_dir,
        source_index=_FORECAST_SOURCE_INDEX,
        forecast_lag_hours=args.forecast_lag_hours,
        base_feature_source=args.base_feature_source,
        source_pool=args.source_pool,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
