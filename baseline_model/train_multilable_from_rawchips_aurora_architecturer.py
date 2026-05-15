#!/usr/bin/env python3
"""Train a raw-chip multi-label classifier with an Aurora-style encoder."""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_multilabel_from_raw_chips as raw_train  # noqa: E402

OUTPUT_DIR = str(PROJECT_ROOT / "baseline_model_outputs_aurora")


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawVariablePatchEmbed(nn.Module):
    """Aurora-style per-variable 3D patch embedding for raw ERA5 chips."""

    def __init__(
        self,
        var_names: Sequence[str],
        patch_size: int,
        embed_dim: int,
        history_size: int,
    ):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        self.var_names = tuple(str(v) for v in var_names)
        self.patch_size = int(patch_size)
        self.history_size = int(history_size)
        self.embed_dim = int(embed_dim)
        self.weights = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.empty(embed_dim, 1, self.history_size, self.patch_size, self.patch_size)
                )
                for name in self.var_names
            }
        )
        self.bias = nn.Parameter(torch.empty(embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights.values():
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(next(iter(self.weights.values())))
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, var_names: Sequence[str]) -> torch.Tensor:
        # x: (B, V, T, H, W), output: (B, V, L, D)
        bsz, var_count, history, height, width = x.shape
        if var_count != len(var_names):
            raise ValueError(f"Expected {var_count} variable names, got {len(var_names)}.")
        if history > self.history_size:
            raise ValueError(f"history={history} exceeds configured history_size={self.history_size}.")

        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        embeds: List[torch.Tensor] = []
        stride = (history, self.patch_size, self.patch_size)
        for idx, name in enumerate(var_names):
            weight = self.weights[str(name)][:, :, :history, :, :]
            var_x = x[:, idx : idx + 1]
            proj = F.conv3d(var_x, weight, self.bias, stride=stride)
            embeds.append(proj.reshape(bsz, self.embed_dim, -1).transpose(1, 2))
        return torch.stack(embeds, dim=1)


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scale = head_dim**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout))

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.to_q(latents)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            bsz, tokens, _ = t.shape
            return t.reshape(bsz, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)
        attn = torch.softmax((q * self.scale) @ k.transpose(-2, -1), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(latents.shape[0], latents.shape[1], -1)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Small Perceiver resampler matching Aurora's level aggregation pattern."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim, num_heads, head_dim, dropout),
                        MLP(dim, int(dim * mlp_ratio), dropout),
                        nn.LayerNorm(dim),
                        nn.LayerNorm(dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        for attn, mlp, norm_latents, norm_context in self.layers:
            latents = latents + attn(norm_latents(latents), norm_context(context))
            latents = latents + mlp(norm_latents(latents))
        return latents


class AuroraRawChipClassifier(nn.Module):
    """Aurora encoder analogue adapted to small raw chips plus a 40-label head."""

    def __init__(
        self,
        dynamic_channel_names: Sequence[str],
        static_channel_names: Sequence[str],
        output_dim: int = 40,
        base_feature_dim: int = 6,
        history_size: int = 2,
        patch_size: int = 3,
        latent_levels: int = 4,
        embed_dim: int = 128,
        num_heads: int = 4,
        head_dim: int = 32,
        perceiver_depth: int = 2,
        transformer_depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dynamic_channel_names = tuple(str(v) for v in dynamic_channel_names)
        self.static_channel_names = tuple(str(v) for v in static_channel_names)
        self.base_feature_dim = int(base_feature_dim)
        self.history_size = int(history_size)
        self.patch_size = int(patch_size)
        self.latent_levels = int(latent_levels)
        self.embed_dim = int(embed_dim)
        self.dynamic_count = len(self.dynamic_channel_names)
        self.static_count = len(self.static_channel_names)

        if self.dynamic_count <= 0:
            raise ValueError("At least one dynamic channel is required.")
        if self.latent_levels < 1:
            raise ValueError("latent_levels must be >= 1.")

        self.dynamic_patch_embed = RawVariablePatchEmbed(
            var_names=self.dynamic_channel_names,
            patch_size=patch_size,
            embed_dim=embed_dim,
            history_size=history_size,
        )
        self.static_patch_embed = (
            RawVariablePatchEmbed(
                var_names=self.static_channel_names,
                patch_size=patch_size,
                embed_dim=embed_dim,
                history_size=history_size,
            )
            if self.static_count > 0
            else None
        )
        self.variable_latents = nn.Parameter(torch.randn(latent_levels, embed_dim) * 0.02)
        self.level_agg = PerceiverResampler(
            dim=embed_dim,
            depth=perceiver_depth,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.position_embed = nn.Linear(4, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + self.base_feature_dim),
            nn.Linear(embed_dim + self.base_feature_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim),
        )

    def _split_chips(self, chips: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, height, width = chips.shape
        expected_channels = self.history_size * self.dynamic_count + self.static_count
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} chip channels, got {channels}.")
        dyn_end = self.history_size * self.dynamic_count
        dynamic = chips[:, :dyn_end].reshape(
            bsz, self.history_size, self.dynamic_count, height, width
        )
        dynamic = dynamic.transpose(1, 2).contiguous()
        static = chips[:, dyn_end:]
        return dynamic, static

    def _patch_position_encoding(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h_patches = math.ceil(height / self.patch_size)
        w_patches = math.ceil(width / self.patch_size)
        y = torch.linspace(-1.0, 1.0, h_patches, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, w_patches, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        features = torch.stack(
            [
                torch.sin(math.pi * yy),
                torch.cos(math.pi * yy),
                torch.sin(math.pi * xx),
                torch.cos(math.pi * xx),
            ],
            dim=-1,
        ).reshape(-1, 4)
        return self.position_embed(features)

    def forward(self, chips: torch.Tensor, base_features: torch.Tensor) -> torch.Tensor:
        _, _, height, width = chips.shape
        dynamic, static = self._split_chips(chips)
        var_tokens = [self.dynamic_patch_embed(dynamic, self.dynamic_channel_names)]
        if self.static_patch_embed is not None:
            static_history = static[:, :, None].expand(-1, -1, self.history_size, -1, -1)
            var_tokens.append(self.static_patch_embed(static_history, self.static_channel_names))
        context = torch.cat(var_tokens, dim=1)

        bsz, var_count, patch_count, dim = context.shape
        context = context.permute(0, 2, 1, 3).reshape(bsz * patch_count, var_count, dim)
        latents = self.variable_latents[None].expand(bsz * patch_count, -1, -1)
        tokens = self.level_agg(latents, context)
        tokens = tokens.reshape(bsz, patch_count * self.latent_levels, dim)
        pos = self._patch_position_encoding(height, width, chips.device, chips.dtype)
        pos = pos[:, None, :].expand(-1, self.latent_levels, -1).reshape(1, patch_count * self.latent_levels, dim)
        tokens = tokens + pos
        tokens = self.spatial_encoder(self.token_norm(tokens))
        pooled = tokens.mean(dim=1)
        if self.base_feature_dim > 0:
            pooled = torch.cat([pooled, base_features], dim=1)
        return self.head(pooled)


def _dynamic_and_static_names(channel_names: np.ndarray, input_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    names = [str(v) for v in channel_names.tolist()]
    t_minus_names = [name for name in names if name.startswith("t_minus_6_")]
    t_names = [name for name in names if name.startswith("t_")]
    if t_minus_names and len(t_minus_names) == len(t_names):
        dynamic_count = len(t_minus_names)
        dynamic_names = [name.replace("t_minus_6_", "", 1) for name in names[:dynamic_count]]
        static_names = names[2 * dynamic_count : input_channels]
    else:
        dynamic_count = (input_channels - 3) // 2 if input_channels >= 3 else input_channels // 2
        dynamic_names = [f"dynamic_{idx}" for idx in range(dynamic_count)]
        static_names = [f"static_{idx}" for idx in range(input_channels - 2 * dynamic_count)]
    return np.asarray(dynamic_names, dtype="<U64"), np.asarray(static_names, dtype="<U64")


def train_model(
    train_files: Sequence[raw_train.FileMeta],
    validation_files: Sequence[raw_train.FileMeta],
    test_files: Sequence[raw_train.FileMeta],
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    patch_size: int,
    latent_levels: int,
    embed_dim: int,
    num_heads: int,
    head_dim: int,
    perceiver_depth: int,
    transformer_depth: int,
    mlp_ratio: float,
    dropout: float,
    use_pos_weight: bool,
    use_base_features: bool,
    seed: int,
    device: str,
    early_stop_patience: int,
    early_stop_min_delta: float,
    plot_random_curtain_count: int,
    plot_curtain_rows: int,
    plot_dir: Path,
    plot_file_prefix: str,
    sample_ratio: float,
    max_samples_per_file: Optional[int],
) -> Dict[str, object]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    torch.manual_seed(seed)
    np.random.seed(seed)

    stats, train_label_sum, channel_names, chip_size = raw_train.compute_train_stats(
        files=train_files,
        sample_ratio=sample_ratio,
        max_samples_per_file=max_samples_per_file,
        seed=seed,
    )
    input_channels = int(len(channel_names))
    dynamic_names, static_names = _dynamic_and_static_names(channel_names, input_channels)
    base_feature_dim = len(raw_train.BASE_FEATURE_COLUMNS) if use_base_features else 0
    model = AuroraRawChipClassifier(
        dynamic_channel_names=dynamic_names,
        static_channel_names=static_names,
        output_dim=len(raw_train.TARGET_COLUMNS),
        base_feature_dim=base_feature_dim,
        history_size=2,
        patch_size=patch_size,
        latent_levels=latent_levels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        perceiver_depth=perceiver_depth,
        transformer_depth=transformer_depth,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    ).to(device)
    print(
        f"Aurora-style raw-chip classifier | input_channels={input_channels} | chip_size={chip_size} | "
        f"patch_size={patch_size} | embed_dim={embed_dim} | latent_levels={latent_levels} | "
        f"dynamic_vars={len(dynamic_names)} | static_vars={len(static_names)} | "
        f"use_base_features={use_base_features}"
    )

    train_sample_count = 0
    for file_idx, meta in enumerate(train_files):
        payload = raw_train._load_npz_payload(
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_epoch = 0
    best_score = float("-inf")
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    best_train_metrics: Optional[Dict[str, float]] = None
    best_validation_metrics: Optional[Dict[str, float]] = None
    best_train_loss = float("nan")
    best_train_eval_loss = float("nan")
    best_validation_loss = float("nan")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for chips_t, base_t, y_t in raw_train.iter_file_batches(
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
        train_eval_loss, train_metrics, _ = raw_train._evaluate_files(
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
        validation_loss, validation_metrics, _ = raw_train._evaluate_files(
            model=model,
            loss_fn=loss_fn,
            files=validation_files,
            batch_size=eval_batch_size,
            device=device,
            stats=stats,
            use_base_features=use_base_features,
            sample_ratio=sample_ratio,
            max_samples_per_file=max_samples_per_file,
            seed=seed,
            search_iou_threshold=True,
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | train_eval_loss={train_eval_loss:.5f} | "
            f"validation_loss={validation_loss:.5f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | validation_acc={validation_metrics['accuracy']:.4f} | "
            f"train_f1_macro={train_metrics['f1_macro']:.4f} | validation_f1_macro={validation_metrics['f1_macro']:.4f} | "
            f"train_auc_macro={train_metrics['auc_macro']:.4f} | validation_auc_macro={validation_metrics['auc_macro']:.4f} | "
            f"train_iou_mean={train_metrics['iou_mean']:.4f} @thr={train_metrics['iou_threshold']:.3f} | "
            f"validation_iou_mean={validation_metrics['iou_mean']:.4f} @thr={validation_metrics['iou_threshold']:.3f} | "
            f"train_iou_empty={train_metrics['iou_empty_truth_mean']:.4f} n={train_metrics['empty_truth_count']} | "
            f"validation_iou_empty={validation_metrics['iou_empty_truth_mean']:.4f} n={validation_metrics['empty_truth_count']} | "
            f"train_iou_nonempty={train_metrics['iou_nonempty_truth_mean']:.4f} n={train_metrics['nonempty_truth_count']} | "
            f"validation_iou_nonempty={validation_metrics['iou_nonempty_truth_mean']:.4f} n={validation_metrics['nonempty_truth_count']} | "
            f"validation_infer_ms/sample={validation_metrics['inference_ms_per_sample']:.3f}"
        )

        score = float(validation_metrics["iou_mean"])
        if score > best_score + early_stop_min_delta:
            best_epoch = epoch
            best_score = score
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_metrics = dict(train_metrics)
            best_validation_metrics = dict(validation_metrics)
            best_train_loss = float(train_loss)
            best_train_eval_loss = float(train_eval_loss)
            best_validation_loss = float(validation_loss)
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

    if best_state_dict is None or best_train_metrics is None or best_validation_metrics is None:
        raise RuntimeError("Training finished without recording a best model state.")
    model.load_state_dict(best_state_dict)
    validation_metrics = best_validation_metrics
    test_loss, test_metrics, _ = raw_train._evaluate_files(
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
        iou_threshold=validation_metrics["iou_threshold"],
    )
    print(
        "Using best epoch | "
        f"epoch={best_epoch:03d} | train_loss={best_train_loss:.5f} | "
        f"train_eval_loss={best_train_eval_loss:.5f} | validation_loss={best_validation_loss:.5f} | "
        f"validation_iou_mean={best_score:.4f} @thr={validation_metrics['iou_threshold']:.3f} | "
        f"test_loss={test_loss:.5f} | test_iou_mean={test_metrics['iou_mean']:.4f} @thr={test_metrics['iou_threshold']:.3f} | "
        f"test_iou_empty={test_metrics['iou_empty_truth_mean']:.4f} n={test_metrics['empty_truth_count']} | "
        f"test_iou_nonempty={test_metrics['iou_nonempty_truth_mean']:.4f} n={test_metrics['nonempty_truth_count']} | "
        f"test_infer_ms/sample={test_metrics['inference_ms_per_sample']:.3f}"
    )

    raw_train._save_random_curtain_plots(
        out_dir=plot_dir,
        split_name="validation",
        file_prefix=plot_file_prefix,
        model=model,
        files=validation_files,
        stats=stats,
        use_base_features=use_base_features,
        eval_batch_size=eval_batch_size,
        device=device,
        num_random_plots=plot_random_curtain_count,
        curtain_rows=plot_curtain_rows,
        prediction_threshold=float(validation_metrics["iou_threshold"]),
        seed=seed,
    )

    return {
        "model": model,
        "stats": stats,
        "train_metrics": best_train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "input_channels": input_channels,
        "dynamic_channel_names": dynamic_names,
        "static_channel_names": static_names,
        "patch_size": int(patch_size),
        "latent_levels": int(latent_levels),
        "embed_dim": int(embed_dim),
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "perceiver_depth": int(perceiver_depth),
        "transformer_depth": int(transformer_depth),
        "mlp_ratio": float(mlp_ratio),
        "dropout": float(dropout),
        "chip_size": int(chip_size),
        "channel_names": channel_names,
        "use_base_features": bool(use_base_features),
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
    }


def _save_artifacts(
    out_dir: Path,
    fit: Dict[str, object],
    train_files: Sequence[raw_train.FileMeta],
    validation_files: Sequence[raw_train.FileMeta],
    test_files: Sequence[raw_train.FileMeta],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "multilabel_aurora_rawchip_classifier.pt"
    stats_path = out_dir / "normalization_stats.npz"
    split_path = out_dir / "file_split.csv"
    stats = fit["stats"]
    assert isinstance(stats, raw_train.NormalizationStats)

    torch.save(
        {
            "model_state_dict": fit["model"].state_dict(),
            "architecture": "aurora_raw_chip_classifier",
            "input_channels": fit["input_channels"],
            "output_dim": len(raw_train.TARGET_COLUMNS),
            "dynamic_channel_names": fit["dynamic_channel_names"],
            "static_channel_names": fit["static_channel_names"],
            "patch_size": fit["patch_size"],
            "latent_levels": fit["latent_levels"],
            "embed_dim": fit["embed_dim"],
            "num_heads": fit["num_heads"],
            "head_dim": fit["head_dim"],
            "perceiver_depth": fit["perceiver_depth"],
            "transformer_depth": fit["transformer_depth"],
            "mlp_ratio": fit["mlp_ratio"],
            "dropout": fit["dropout"],
            "chip_size": fit["chip_size"],
            "channel_names": fit["channel_names"],
            "base_features": raw_train.BASE_FEATURE_COLUMNS if fit["use_base_features"] else [],
            "target_columns": raw_train.TARGET_COLUMNS,
            "train_metrics": fit["train_metrics"],
            "validation_metrics": fit["validation_metrics"],
            "test_metrics": fit["test_metrics"],
            "best_epoch": fit["best_epoch"],
            "best_score": fit["best_score"],
            "early_stop_patience": fit["early_stop_patience"],
            "early_stop_min_delta": fit["early_stop_min_delta"],
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
    for meta in train_files:
        rows.append({"split": "train", "file": str(meta.source_file), "npz": str(meta.npz_path), "file_time_utc": str(meta.file_time)})
    for meta in validation_files:
        rows.append({"split": "validation", "file": str(meta.source_file), "npz": str(meta.npz_path), "file_time_utc": str(meta.file_time)})
    for meta in test_files:
        rows.append({"split": "test", "file": str(meta.source_file), "npz": str(meta.npz_path), "file_time_utc": str(meta.file_time)})
    pd.DataFrame(rows).to_csv(split_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved normalization stats: {stats_path}")
    print(f"Saved split manifest: {split_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-chip-dir", type=str, default=raw_train.RAW_CHIPS_DIR)
    parser.add_argument("--train-files", type=int, default=1000)
    parser.add_argument("--validation-files", type=int, default=50)
    parser.add_argument("--test-files", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--latent-levels", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--perceiver-depth", type=int, default=2)
    parser.add_argument("--transformer-depth", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--sample-ratio", type=float, default=0.3)
    parser.add_argument("--max-samples-per-file", type=int, default=0)
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument("--no-base-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--test-start-time", type=str, default=raw_train.DEFAULT_TEST_START_TIME)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--plot-dir", type=str, default=raw_train.LOG_DIR)
    parser.add_argument(
        "--plot-random-curtain-count",
        "--plot-random-iou-count",
        dest="plot_random_curtain_count",
        type=int,
        default=10,
    )
    parser.add_argument("--plot-file-prefix", type=str, default="raw_chips_aurora")
    parser.add_argument("--plot-curtain-rows", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    raw_chip_dir = Path(args.raw_chip_dir)
    if not raw_chip_dir.exists():
        raise FileNotFoundError(f"Raw chip dir does not exist: {raw_chip_dir}")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0.")
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("--embed-dim must be divisible by --num-heads.")

    metas = raw_train.discover_files(raw_chip_dir)
    test_start_time = raw_train._parse_utc_timestamp(args.test_start_time)
    train_files, validation_files, test_files = raw_train.select_train_validation_test_files(
        metas=metas,
        train_files=args.train_files,
        validation_files=args.validation_files,
        test_files=args.test_files,
        seed=args.seed,
        test_start_time=test_start_time,
    )
    print(f"Selected train files: {len(train_files)}")
    print(f"Selected validation files: {len(validation_files)}")
    print(f"Selected test files: {len(test_files)}")
    print(f"Test file cutoff:   {test_start_time}")
    print(f"Train max file_time: {max(m.file_time for m in train_files)}")
    print(f"Validation min file_time: {min(m.file_time for m in validation_files)}")
    print(f"Test min file_time:  {min(m.file_time for m in test_files)}")

    fit = train_model(
        train_files=train_files,
        validation_files=validation_files,
        test_files=test_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        patch_size=args.patch_size,
        latent_levels=args.latent_levels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        perceiver_depth=args.perceiver_depth,
        transformer_depth=args.transformer_depth,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_pos_weight=(not args.no_pos_weight),
        use_base_features=(not args.no_base_features),
        seed=args.seed,
        device=args.device,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        plot_random_curtain_count=args.plot_random_curtain_count,
        plot_curtain_rows=args.plot_curtain_rows,
        plot_dir=Path(args.plot_dir),
        plot_file_prefix=args.plot_file_prefix,
        sample_ratio=args.sample_ratio,
        max_samples_per_file=(None if args.max_samples_per_file <= 0 else int(args.max_samples_per_file)),
    )

    print(
        "Final metrics | "
        f"train_acc={fit['train_metrics']['accuracy']:.4f}, "
        f"validation_acc={fit['validation_metrics']['accuracy']:.4f}, "
        f"test_acc={fit['test_metrics']['accuracy']:.4f}, "
        f"train_f1_macro={fit['train_metrics']['f1_macro']:.4f}, "
        f"validation_f1_macro={fit['validation_metrics']['f1_macro']:.4f}, "
        f"test_f1_macro={fit['test_metrics']['f1_macro']:.4f}, "
        f"train_auc_macro={fit['train_metrics']['auc_macro']:.4f}, "
        f"validation_auc_macro={fit['validation_metrics']['auc_macro']:.4f}, "
        f"test_auc_macro={fit['test_metrics']['auc_macro']:.4f}, "
        f"train_iou_mean={fit['train_metrics']['iou_mean']:.4f} @thr={fit['train_metrics']['iou_threshold']:.3f}, "
        f"validation_iou_mean={fit['validation_metrics']['iou_mean']:.4f} @thr={fit['validation_metrics']['iou_threshold']:.3f}, "
        f"test_iou_mean={fit['test_metrics']['iou_mean']:.4f} @thr={fit['test_metrics']['iou_threshold']:.3f}, "
        f"train_iou_empty={fit['train_metrics']['iou_empty_truth_mean']:.4f} n={fit['train_metrics']['empty_truth_count']}, "
        f"validation_iou_empty={fit['validation_metrics']['iou_empty_truth_mean']:.4f} n={fit['validation_metrics']['empty_truth_count']}, "
        f"test_iou_empty={fit['test_metrics']['iou_empty_truth_mean']:.4f} n={fit['test_metrics']['empty_truth_count']}, "
        f"train_iou_nonempty={fit['train_metrics']['iou_nonempty_truth_mean']:.4f} n={fit['train_metrics']['nonempty_truth_count']}, "
        f"validation_iou_nonempty={fit['validation_metrics']['iou_nonempty_truth_mean']:.4f} n={fit['validation_metrics']['nonempty_truth_count']}, "
        f"test_iou_nonempty={fit['test_metrics']['iou_nonempty_truth_mean']:.4f} n={fit['test_metrics']['nonempty_truth_count']}, "
        f"test_infer_ms/sample={fit['test_metrics']['inference_ms_per_sample']:.3f}, "
        f"best_epoch={fit['best_epoch']}"
    )

    _save_artifacts(
        out_dir=Path(args.output_dir),
        fit=fit,
        train_files=train_files,
        validation_files=validation_files,
        test_files=test_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
