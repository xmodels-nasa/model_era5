#!/usr/bin/env python3
import dataclasses
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from aurora import AuroraPretrained, AuroraSmallPretrained, Batch, Metadata


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


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

# =========================
# Editable config variables
# =========================
DATA_ROOT = os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data_era5"))
TARGETS = [
    "2018_10_10_06",
    "2018_10_10_12",
]
TIME_INDEX = 0  # Current ERA5 folder layout has one timestamp per file, so keep this at 0.
SKIP_PREDICT = False
PREDICT_ALL = False
DEBUG = _env_flag("DEBUG", _env_flag("debug", False))
AURORA_CHECKPOINT_REPO = os.getenv("AURORA_CHECKPOINT_REPO", "microsoft/aurora")
AURORA_CHECKPOINT_FILE = os.getenv(
    "AURORA_CHECKPOINT_FILE", AuroraPretrained.default_checkpoint_name
)
AURORA_SMALL_CHECKPOINT_FILE = os.getenv(
    "AURORA_SMALL_CHECKPOINT_FILE", AuroraSmallPretrained.default_checkpoint_name
)


def _new_aurora_model() -> AuroraPretrained | AuroraSmallPretrained:
    if DEBUG:
        model = AuroraSmallPretrained()
        model.load_checkpoint(AURORA_CHECKPOINT_REPO, AURORA_SMALL_CHECKPOINT_FILE)
        return model

    model = AuroraPretrained()
    model.load_checkpoint(AURORA_CHECKPOINT_REPO, AURORA_CHECKPOINT_FILE)
    return model


def parse_target(ts: str) -> datetime:
    normalized = ts.strip().replace("T", "-").replace("_", "-")
    parts = normalized.split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid target '{ts}'. Use YYYY_MM_DD_HH.")
    y, m, d, h = [int(x) for x in parts]
    return datetime(y, m, d, h)


def folder_name(dt: datetime) -> str:
    return dt.strftime("%Y_%m_%d_%H_data")


def resolve_dir(data_root: str, dt: datetime) -> str:
    d = os.path.join(data_root, folder_name(dt))
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing directory: {d}")
    return d


def load_batch_from_two_dirs(dir_t_minus_6: str, dir_t: str, time_index: int = 0) -> Batch:
    def _open_triplet(d):
        surf_ds = xr.open_dataset(os.path.join(d, "_surface.nc"), engine="netcdf4")
        atmos_ds = xr.open_dataset(os.path.join(d, "_atmospheric.nc"), engine="netcdf4")
        static_ds = xr.open_dataset(os.path.join(d, "_static.nc"), engine="netcdf4")
        return surf_ds, atmos_ds, static_ds

    def _get_time_dim(ds):
        if "time" in ds.dims:
            return "time"
        if "valid_time" in ds.dims:
            return "valid_time"
        raise KeyError("No time/valid_time dimension found")

    def _lat_is_decreasing(lat):
        return np.all(np.diff(lat) < 0)

    def _prepare_single(x, flip_lat, time_index=0):
        x = x[time_index:time_index + 1]
        if flip_lat:
            x = x[..., ::-1, :]
        return torch.from_numpy(x[None].copy())

    def _static_2d(da):
        for dim in ["number", "valid_time", "expver"]:
            if dim in da.dims:
                da = da.isel({dim: 0})
        return torch.from_numpy(np.array(da.values, copy=True))

    surf0, atmos0, static0 = _open_triplet(dir_t_minus_6)
    surf1, atmos1, _ = _open_triplet(dir_t)

    lat_vals = surf0.latitude.values
    flip_lat = not _lat_is_decreasing(lat_vals)
    lat_out = lat_vals[::-1].copy() if flip_lat else lat_vals.copy()

    tdim0 = _get_time_dim(surf0)
    tdim1 = _get_time_dim(surf1)
    time0 = surf0[tdim0].values.astype("datetime64[s]").tolist()[time_index]
    time1 = surf1[tdim1].values.astype("datetime64[s]").tolist()[time_index]

    def _stack_time(var0, var1):
        return torch.cat([var0, var1], dim=1)

    return Batch(
        surf_vars={
            "2t": _stack_time(
                _prepare_single(surf0["t2m"].values, flip_lat, time_index),
                _prepare_single(surf1["t2m"].values, flip_lat, time_index),
            ),
            "10u": _stack_time(
                _prepare_single(surf0["u10"].values, flip_lat, time_index),
                _prepare_single(surf1["u10"].values, flip_lat, time_index),
            ),
            "10v": _stack_time(
                _prepare_single(surf0["v10"].values, flip_lat, time_index),
                _prepare_single(surf1["v10"].values, flip_lat, time_index),
            ),
            "msl": _stack_time(
                _prepare_single(surf0["msl"].values, flip_lat, time_index),
                _prepare_single(surf1["msl"].values, flip_lat, time_index),
            ),
        },
        static_vars={
            "z": _static_2d(static0["z"]),
            "slt": _static_2d(static0["slt"]),
            "lsm": _static_2d(static0["lsm"]),
        },
        atmos_vars={
            "t": _stack_time(
                _prepare_single(atmos0["t"].values, flip_lat, time_index),
                _prepare_single(atmos1["t"].values, flip_lat, time_index),
            ),
            "u": _stack_time(
                _prepare_single(atmos0["u"].values, flip_lat, time_index),
                _prepare_single(atmos1["u"].values, flip_lat, time_index),
            ),
            "v": _stack_time(
                _prepare_single(atmos0["v"].values, flip_lat, time_index),
                _prepare_single(atmos1["v"].values, flip_lat, time_index),
            ),
            "q": _stack_time(
                _prepare_single(atmos0["q"].values, flip_lat, time_index),
                _prepare_single(atmos1["q"].values, flip_lat, time_index),
            ),
            "z": _stack_time(
                _prepare_single(atmos0["z"].values, flip_lat, time_index),
                _prepare_single(atmos1["z"].values, flip_lat, time_index),
            ),
        },
        metadata=Metadata(
            lat=torch.from_numpy(lat_out),
            lon=torch.from_numpy(np.array(surf0.longitude.values, copy=True)),
            time=(time0, time1),
            atmos_levels=tuple(int(level) for level in atmos0["pressure_level"].values),
        ),
    )


def build_batches(data_root: str, targets: List[datetime], time_index: int = 0) -> List[Tuple[datetime, Batch]]:
    batches: List[Tuple[datetime, Batch]] = []
    for t in targets:
        t_minus_6 = t - timedelta(hours=6)
        d0 = resolve_dir(data_root, t_minus_6)
        d1 = resolve_dir(data_root, t)
        batch = load_batch_from_two_dirs(d0, d1, time_index=time_index)
        batches.append((t, batch))
    return batches


def run_prediction_smoke(batches: List[Tuple[datetime, Batch]], predict_all: bool = False) -> None:
    if not batches:
        print("No batches to run.")
        return

    model = _new_aurora_model()
    model.eval()

    run_items = batches if predict_all else [batches[0]]
    with torch.inference_mode():
        for t, batch in run_items:
            pred = model.forward(batch)
            print(
                f"Prediction OK for target {t.strftime('%Y-%m-%d %H:00')} | "
                f"pred 2t shape={tuple(pred.surf_vars['2t'].shape)}"
            )


def load_aurora_model(device: Optional[str] = None) -> AuroraPretrained | AuroraSmallPretrained:
    model = _new_aurora_model()
    if device:
        model = model.to(device)
    model.eval()
    return model


def get_encoder_output(model: AuroraPretrained, batch: Batch) -> Tuple[torch.Tensor, Batch]:
    # Keep the same preprocessing flow used by Aurora forward().
    p = next(model.parameters())
    enc_batch = model.batch_transform_hook(batch).type(p.dtype)
    enc_batch = enc_batch.normalise(surf_stats=model.surf_stats)
    enc_batch = enc_batch.crop(patch_size=model.patch_size)
    enc_batch = enc_batch.to(p.device)

    b_size, t_size = next(iter(enc_batch.surf_vars.values())).shape[:2]
    enc_batch = dataclasses.replace(
        enc_batch,
        static_vars={k: v[None, None].repeat(b_size, t_size, 1, 1) for k, v in enc_batch.static_vars.items()},
    )

    if model.positive_surf_vars:
        enc_batch = dataclasses.replace(
            enc_batch,
            surf_vars={
                k: v.clamp(min=0) if k in model.positive_surf_vars else v
                for k, v in enc_batch.surf_vars.items()
            },
        )
    if model.positive_atmos_vars:
        enc_batch = dataclasses.replace(
            enc_batch,
            atmos_vars={
                k: v.clamp(min=0) if k in model.positive_atmos_vars else v
                for k, v in enc_batch.atmos_vars.items()
            },
        )

    enc_batch = model._pre_encoder_hook(enc_batch)

    with torch.inference_mode():
        enc_out = model.encoder(enc_batch, lead_time=model.timestep)
    return enc_out, enc_batch


def token_indices_for_latlon(
    enc_batch: Batch, model: AuroraPretrained, lat: float, lon: float
) -> Tuple[int, List[int]]:
    lat_vec = enc_batch.metadata.lat
    lon_vec = enc_batch.metadata.lon

    lat_in = float(lat)
    lon_in = float(lon)
    lat_norm = lat_in  # No latitude convention remap currently required.

    lat_min = float(lat_vec.min())
    lat_max = float(lat_vec.max())
    if lat_norm < lat_min or lat_norm > lat_max:
        raise ValueError(f"lat={lat_norm} is out of range [{lat_min}, {lat_max}]")

    lon_min = float(lon_vec.min())
    lon_max = float(lon_vec.max())

    # print out era5 lat and lon range and query lon for debugging
    #print(f"ERA5 lat range: [{lat_min}, {lat_max}] | query lat: {lat}")
    #print(f"ERA5 lon range: [{lon_min}, {lon_max}] | query lon: {lon}")

    if lon_min >= 0.0 and lon_max > 180.0:
        lon_norm = lon_in % 360.0
    elif lon_min < 0.0 and lon_max <= 180.0:
        lon_norm = ((lon_in + 180.0) % 360.0) - 180.0
    else:
        lon_norm = lon_in

    #if lat_norm != lat_in:
    #    print(f"Mapped Feather latitude {lat_in} -> ERA5 latitude {lat_norm}")
    #if lon_norm != lon_in:
    #    print(f"Mapped Feather longitude {lon_in} -> ERA5 longitude {lon_norm}")

    # Choose nearest gridpoint, not lower-bound bin.
    lat_np = lat_vec.detach().cpu().numpy()
    lon_np = lon_vec.detach().cpu().numpy()
    i = int(np.abs(lat_np - lat_norm).argmin())

    lon_span = float(lon_max - lon_min)
    if lon_span > 300.0:
        # Global lon axis: compare using wrapped angular distance.
        lon_dist = np.abs(((lon_np - lon_norm + 180.0) % 360.0) - 180.0)
    else:
        lon_dist = np.abs(lon_np - lon_norm)
    j = int(lon_dist.argmin())

    h_grid, w_grid = enc_batch.spatial_shape
    patch = model.encoder.patch_size
    h_patches = h_grid // patch
    w_patches = w_grid // patch
    patch_index = (i // patch) * w_patches + (j // patch)

    levels = model.encoder.latent_levels
    token_indices = [level * (h_patches * w_patches) + patch_index for level in range(levels)]
    return patch_index, token_indices


def embedding_at_latlon(
    enc_out: torch.Tensor,
    enc_batch: Batch,
    model: AuroraPretrained,
    lat: float,
    lon: float,
    level: Optional[int] = None,
    token_indices: Optional[List[int]] = None,
) -> torch.Tensor:
    if token_indices is None:
        _, token_indices = token_indices_for_latlon(enc_batch, model, lat, lon)
    if level is None:
        return enc_out[:, token_indices, :]
    return enc_out[:, token_indices[level], :]


def _normalize_lon_to_grid(lon: float, lon_vec: np.ndarray) -> float:
    lon_min = float(lon_vec.min())
    lon_max = float(lon_vec.max())
    if lon_min >= 0.0 and lon_max > 180.0:
        return lon % 360.0
    if lon_min < 0.0 and lon_max <= 180.0:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _nearest_lon_on_grid(lon_vec: np.ndarray, lon_norm: float) -> float:
    lon_span = float(lon_vec.max() - lon_vec.min())
    if lon_span > 300.0:
        lon_dist = np.abs(((lon_vec - lon_norm + 180.0) % 360.0) - 180.0)
    else:
        lon_dist = np.abs(lon_vec - lon_norm)
    return float(lon_vec[lon_dist.argmin()])


def get_encoder_context_for_target(
    data_root: str,
    target: str,
    time_index: int = 0,
    model: Optional[AuroraPretrained] = None,
) -> Dict[str, Any]:
    """Build one encoder context for a target hour so multiple lat/lon queries can reuse it."""
    target_dt = parse_target(target)
    t_minus_6 = target_dt - timedelta(hours=6)

    d0 = resolve_dir(data_root, t_minus_6)
    d1 = resolve_dir(data_root, target_dt)
    batch = load_batch_from_two_dirs(d0, d1, time_index=time_index)

    local_model = model if model is not None else load_aurora_model()
    enc_out, enc_batch = get_encoder_output(local_model, batch)
    return {
        "context_time": target_dt,
        "input_times": (t_minus_6, target_dt),
        "target": target_dt,
        "input_pair": (t_minus_6, target_dt),
        "enc_out": enc_out,
        "enc_batch": enc_batch,
        "model": local_model,
    }


def get_embedding_from_encoder_context(
    encoder_context: Dict[str, Any],
    lat: float,
    lon: float,
) -> Dict[str, object]:
    """Extract one embedding from a precomputed encoder context at the nearest gridpoint."""
    enc_out = encoder_context["enc_out"]
    enc_batch = encoder_context["enc_batch"]
    local_model = encoder_context["model"]

    patch_idx, token_indices = token_indices_for_latlon(enc_batch, local_model, lat, lon)
    lat_vec = enc_batch.metadata.lat.detach().cpu().numpy()
    lon_vec = enc_batch.metadata.lon.detach().cpu().numpy()
    nearest_lat = float(lat_vec[np.abs(lat_vec - lat).argmin()])
    lon_norm = _normalize_lon_to_grid(lon, lon_vec)
    nearest_lon = _nearest_lon_on_grid(lon_vec, lon_norm)
    emb_all = embedding_at_latlon(
        enc_out,
        enc_batch,
        local_model,
        lat,
        lon,
        level=None,
        token_indices=token_indices,
    )

    return {
        "context_time": encoder_context["context_time"],
        "input_times": encoder_context["input_times"],
        "target": encoder_context["target"],
        "input_pair": encoder_context["input_pair"],
        "lat": lat,
        "lon": lon_norm,
        "matched_lat": nearest_lat,
        "matched_lon": nearest_lon,
        "patch_idx": patch_idx,
        "token_indices": token_indices,
        "emb_all_levels": emb_all,  # shape: [B, ...] with trailing dims determined by token_indices
        "enc_out_shape": tuple(enc_out.shape),
    }


def get_embeddings_for_target(
    data_root: str,
    target: str,
    lat: float,
    lon: float,
    time_index: int = 0,
    model: Optional[AuroraPretrained] = None,
) -> Dict[str, object]:
    """
    Build one Aurora input window from (t-6h, t), then return encoder embeddings
    at a query lat/lon across all latent levels.

    Note: The encoder output does not retain a separate time axis. Returned
    embeddings represent the target context at time t, conditioned on both
    input timestamps.
    """
    encoder_context = get_encoder_context_for_target(
        data_root=data_root,
        target=target,
        time_index=time_index,
        model=model,
    )
    return get_embedding_from_encoder_context(
        encoder_context=encoder_context,
        lat=lat,
        lon=lon,
    )


def run_with_config() -> List[Tuple[datetime, Batch]]:
    targets = [parse_target(x) for x in TARGETS]
    batches = build_batches(DATA_ROOT, targets, time_index=TIME_INDEX)

    print(f"Built {len(batches)} batch(es):")
    for t, b in batches:
        time_pair = b.metadata.time
        print(
            f"  target={t.strftime('%Y-%m-%d %H:00')} "
            f"input_times=({time_pair[0]}, {time_pair[1]})"
        )

    if not SKIP_PREDICT:
        run_prediction_smoke(batches, predict_all=PREDICT_ALL)

    return batches


if __name__ == "__main__":
    run_with_config()
