# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""QR-Bloom training script.

Trains a QR-conditioned 3D voxel tree diffusion model using v-prediction.
Training data is generated on-the-fly: each sample is a procedurally grown
voxel tree placed inside a QR footprint. The voxel grid size depends on the
QR version (see qrbloom.qr.grid_shape_for_version). The model learns to
denoise a 4-channel (RGB + occupancy) voxel grid conditioned on the QR
footprint, a tree-species theme index, and three shape attributes.

Losses (see qrbloom/diffusion.py p_losses):
  v_mse : v-prediction MSE over all 4 channels
  v_rgb : v-prediction MSE restricted to the RGB channels
  v_occ : v-prediction MSE restricted to the occupancy channel

Outputs (with `VARIANT` env var suffix, e.g. VARIANT=_v2):
  checkpoints/qrbloom{VARIANT}.pt       — latest checkpoint
  checkpoints/qrbloom{VARIANT}_best.pt  — best validation checkpoint
  runs{VARIANT}/epoch_{N:03d}.png       — per-epoch sample montage
  runs{VARIANT}/epoch_{N:03d}.json      — per-epoch voxel data for the gallery
  runs{VARIANT}/loss.png                — training loss curve
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from qrbloom.diffusion import Diffusion, EMA, DOWNSCALE
from qrbloom.dit import DiT3D
from qrbloom.qr import (THEME_NAMES as DS_THEMES, random_qr_core,
                        qr_modules, grid_xy_for_version,
                        grid_z_for_version, pad_to_grid)
from qrbloom.treegen import THEMES as VTHEMES
from qrbloom.treegen import generate_voxels_aug as generate_voxels, tree_attributes
import json as _json
import random as _random

# ─── Distributed environment ──────────────────────────────────────────────────
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
IS_DIST = WORLD_SIZE > 1
IS_MAIN = RANK == 0

if IS_DIST:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE = f"cuda:{LOCAL_RANK}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS = int(os.environ.get("EPOCHS", "80"))
BATCH = int(os.environ.get("BATCH", "42"))
BASE_LR = float(os.environ.get("LR", "2e-4"))
T = int(os.environ.get("T", "500"))
SUBSET_FRAC = float(os.environ.get("SUBSET_FRAC", "1.0"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.10"))
SEED = int(os.environ.get("SEED", "42"))
WORKERS = int(os.environ.get("WORKERS", "2"))
AMP_DTYPE = os.environ.get("AMP_DTYPE", "fp16").lower()
COMPILE = bool(int(os.environ.get("COMPILE", "0")))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "1"))
SAMPLE_STEPS = int(os.environ.get("SAMPLE_STEPS", "100"))  # stochastic sampling steps
EPOCH_SIZE = int(os.environ.get("EPOCH_SIZE", "100000"))   # live-generated samples per epoch
VAL_SIZE = int(os.environ.get("VAL_SIZE", "2400"))         # fixed validation set size
VARIANT = os.environ.get("VARIANT", "")
SUFFIX = VARIANT

# ── DiT backbone hyperparameters ──────────────────────────────────────────────
DIT_DIM = int(os.environ.get("DIT_DIM", "384"))
DIT_DEPTH = int(os.environ.get("DIT_DEPTH", "12"))
DIT_HEADS = int(os.environ.get("DIT_HEADS", "6"))
DIT_PATCH = int(os.environ.get("DIT_PATCH", "4"))
DIT_CKPT = bool(int(os.environ.get("DIT_CKPT", "0")))  # gradient checkpointing

# Multi-version training: pick a QR version per sample from this set. The
# DiT backbone builds its positional embeddings from the input size, so a
# single set of weights handles any grid whose dims are multiples of the
# patch size. To bound memory, training defaults to v1..v5 (XY up to 40).
# Set QR_VERSIONS="1,2,3,5,10" to widen the range.
def _parse_versions(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError("QR_VERSIONS must contain at least one version")
    return out

QR_VERSIONS = _parse_versions(os.environ.get("QR_VERSIONS", "2,3,4,5"))
QR_VERSIONS_VAL = _parse_versions(os.environ.get("QR_VERSIONS_VAL",
                                                  os.environ.get("QR_VERSIONS", "2,3,4,5")))


def _parse_weights(s: str, versions: list[int]) -> list[float]:
    """Per-version sampling weights for the training dataset.

    Smaller QR versions are a harder learning problem (less voxel footprint,
    fewer dark modules), so when training on a mix of versions the default
    tilts sampling toward them: weight = (max_v - v + 1). For versions
    [2,3,4,5] that gives [4,3,2,1] → roughly 40/30/20/10% of the batch.
    Override with QR_VERSION_WEIGHTS="w_v2,w_v3,..." in the env.
    Single-version training simply sets weights=[1.0].
    """
    if not s:
        max_v = max(versions)
        return [float(max_v - v + 1) for v in versions]
    ws = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(ws) != len(versions):
        raise ValueError(f"QR_VERSION_WEIGHTS must have {len(versions)} values "
                         f"(one per QR_VERSIONS entry), got {len(ws)}")
    if any(w < 0 for w in ws):
        raise ValueError("QR_VERSION_WEIGHTS must be non-negative")
    if sum(ws) <= 0:
        raise ValueError("QR_VERSION_WEIGHTS must sum to > 0")
    return ws


QR_VERSION_WEIGHTS = _parse_weights(os.environ.get("QR_VERSION_WEIGHTS", ""),
                                     QR_VERSIONS)
# Largest XY/Z the trainer may see — informational, used in startup log only.
MAX_XY = max(grid_xy_for_version(v) for v in (QR_VERSIONS + QR_VERSIONS_VAL))
MAX_Z  = max(grid_z_for_version(v)  for v in (QR_VERSIONS + QR_VERSIONS_VAL))

_AMP_TH = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Loss keys used for logging and aggregation.
LOSS_KEYS = ("v_mse", "v_rgb", "v_occ")


class LiveTreeDataset(torch.utils.data.Dataset):
    """Dataset that generates a fresh (QR, theme, augmentation) sample on every access.

    Each sample is built for a random QR version drawn from `versions`. The
    voxel grid's XY *and Z* size both depend on the version (isotropic
    scaling — see qrbloom.qr.grid_z_for_version). The custom collate
    (`pad_collate`) pads a mixed-version batch to the largest XY and Z in
    the batch so DataLoader can stack tensors.

    deterministic=False (train): each sample uses system entropy — no repeats.
    deterministic=True  (val):   each index maps to a fixed seed — reproducible.

    Returns: (occ_u8, rgb_u8, qr_u8, theme_idx, attr)
    """

    def __init__(self, virtual_len, versions, deterministic, seed=0, weights=None):
        self.virtual_len = int(virtual_len)
        self.versions = list(versions)
        # weights=None → uniform sampling (used for val so the metric is
        # version-balanced). Train passes per-version weights to oversample
        # the harder lower QR versions.
        self.weights = list(weights) if weights is not None else None
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self):
        return self.virtual_len

    def _rng(self, idx):
        if self.deterministic:
            s = (self.seed * 1000003 + idx) & 0x7FFFFFFF
            return _random.Random(s), np.random.default_rng(s)
        return _random.Random(), np.random.default_rng()

    def __getitem__(self, idx):
        rng, nprng = self._rng(idx)
        if self.weights is None:
            version = self.versions[rng.randint(0, len(self.versions) - 1)]
        else:
            version = rng.choices(self.versions, weights=self.weights, k=1)[0]
        qe = qr_modules(version)
        gxy = grid_xy_for_version(version)
        gz  = grid_z_for_version(version)
        off = (gxy - qe) // 2
        ctr = qe // 2

        core = random_qr_core(nprng, version=version)         # (qe, qe) uint8
        ti = rng.randint(0, len(DS_THEMES) - 1)
        voxels = generate_voxels(core.tolist(), theme=DS_THEMES[ti], rng=rng)

        occ = np.zeros((gxy, gxy, gz), dtype=np.uint8)
        rgb = np.zeros((3, gxy, gxy, gz), dtype=np.uint8)
        for v in voxels:
            if v["is_base"]:
                continue
            x, y, z = v["pos"]
            i = off + int(round(z)) + ctr
            j = off + int(round(x)) + ctr
            k = int(round(y))
            if 0 <= i < gxy and 0 <= j < gxy and 0 <= k < gz:
                occ[i, j, k] = 1
                h = v["color"].lstrip("#")
                rgb[0, i, j, k] = int(h[0:2], 16)
                rgb[1, i, j, k] = int(h[2:4], 16)
                rgb[2, i, j, k] = int(h[4:6], 16)
        qr, _ = pad_to_grid(core, grid_xy=gxy)
        attr = np.array(tree_attributes(voxels, qr_side=qe), dtype=np.float32)
        return (torch.from_numpy(occ), torch.from_numpy(rgb),
                torch.from_numpy(qr), ti, torch.from_numpy(attr),
                int(version))


def pad_collate(batch):
    """Collate items of varying XY/Z by zero-padding to the largest in the batch.

    XY padding is symmetric around the QR center; Z padding is one-sided
    (extend upward) so the QR base plane stays at z=0. Both XY and Z are
    rounded up to multiples of DOWNSCALE so the U-Net can up/down cleanly.

    Input shapes per sample:
      occ : (gxy, gxy, gz)
      rgb : (3, gxy, gxy, gz)
      qr  : (gxy, gxy)              — 2D footprint, no Z
    """
    occs, rgbs, qrs, ths, attrs, vers = zip(*batch)
    max_xy = max(o.shape[0] for o in occs)
    max_z  = max(o.shape[2] for o in occs)
    max_xy = ((max_xy + DOWNSCALE - 1) // DOWNSCALE) * DOWNSCALE
    max_z  = ((max_z  + DOWNSCALE - 1) // DOWNSCALE) * DOWNSCALE

    def _pad_occ(t: torch.Tensor) -> torch.Tensor:
        # (D, H, W) — pad W (Z) one-sided, D/H symmetric.
        D, H, W = t.shape
        pz = max_z - W
        pad_d_l = (max_xy - D) // 2; pad_d_r = (max_xy - D) - pad_d_l
        pad_h_l = (max_xy - H) // 2; pad_h_r = (max_xy - H) - pad_h_l
        # F.pad order for 3D tensor: (W_l, W_r, H_l, H_r, D_l, D_r)
        return F.pad(t, (0, pz, pad_h_l, pad_h_r, pad_d_l, pad_d_r))

    def _pad_rgb(t: torch.Tensor) -> torch.Tensor:
        # (C, D, H, W)
        _, D, H, W = t.shape
        pz = max_z - W
        pad_d_l = (max_xy - D) // 2; pad_d_r = (max_xy - D) - pad_d_l
        pad_h_l = (max_xy - H) // 2; pad_h_r = (max_xy - H) - pad_h_l
        return F.pad(t, (0, pz, pad_h_l, pad_h_r, pad_d_l, pad_d_r))

    def _pad_qr(t: torch.Tensor) -> torch.Tensor:
        # (D, H)
        D, H = t.shape
        pad_d_l = (max_xy - D) // 2; pad_d_r = (max_xy - D) - pad_d_l
        pad_h_l = (max_xy - H) // 2; pad_h_r = (max_xy - H) - pad_h_l
        return F.pad(t, (pad_h_l, pad_h_r, pad_d_l, pad_d_r))

    occ_b = torch.stack([_pad_occ(o) for o in occs], 0)
    rgb_b = torch.stack([_pad_rgb(r) for r in rgbs], 0)
    qr_b = torch.stack([_pad_qr(q) for q in qrs], 0)
    th_b = torch.tensor(list(ths), dtype=torch.long)
    attr_b = torch.stack(list(attrs), 0)
    ver_b = torch.tensor(list(vers), dtype=torch.long)
    return occ_b, rgb_b, qr_b, th_b, attr_b, ver_b


def log(*args, **kwargs):
    if IS_MAIN:
        print(*args, **kwargs, flush=True)


CKPT_DIR = Path("checkpoints")
CKPT = CKPT_DIR / f"qrbloom{SUFFIX}.pt"
CKPT_BEST = CKPT_DIR / f"qrbloom{SUFFIX}_best.pt"
OUT_DIR = Path(f"runs{SUFFIX}")
DONE = Path(f"checkpoints/DONE{SUFFIX}")

THEMES = list(DS_THEMES)   # montage labels — kept in sync with the generator


# ─── Visualization ────────────────────────────────────────────────────────────
def render_montage(occ_bin: np.ndarray, rgb: np.ndarray, qrs: np.ndarray,
                   themes: np.ndarray, path: str) -> None:
    """Render a 3D scatter + top-down overlay for each sample in the batch."""
    import matplotlib.pyplot as plt
    B = min(occ_bin.shape[0], 9)
    D, H, W = occ_bin.shape[1], occ_bin.shape[2], occ_bin.shape[3]
    fig = plt.figure(figsize=(B * 2.0, 5.0))
    for i in range(B):
        ax = fig.add_subplot(2, B, i + 1, projection="3d")
        rs, cs, hs = np.where(occ_bin[i])
        if rs.size:
            cols_rgb = (np.transpose(rgb[i], (1, 2, 3, 0))[rs, cs, hs] * 0.5 + 0.5).clip(0, 1)
            ax.scatter(cs, rs, hs, c=cols_rgb, s=8, marker="s", depthshade=False)
        ax.set_xlim(0, D); ax.set_ylim(0, H); ax.set_zlim(0, W)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"{THEMES[themes[i]]}\nn={occ_bin[i].sum()}", fontsize=7)
        ax2 = fig.add_subplot(2, B, B + i + 1)
        top = occ_bin[i].max(axis=2)
        ax2.imshow(np.stack([qrs[i], top, np.zeros_like(top)], axis=-1) * 0.7 + 0.1)
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_title("R=QR  G=pred top", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


# ─── Training / evaluation ────────────────────────────────────────────────────
def make_split(n_total: int, val_frac: float, subset_frac: float, seed: int):
    g = np.random.default_rng(seed)
    perm = g.permutation(n_total)
    n_val = max(1, int(n_total * val_frac))
    val_idx = perm[:n_val]
    train_pool = perm[n_val:]
    n_train = max(1, int(len(train_pool) * subset_frac))
    train_idx = train_pool[:n_train]
    return train_idx.tolist(), val_idx.tolist()


def _safe_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _normalize_batch(occ_u8, rgb_u8, qr_u8, device):
    """Convert uint8 tensors to normalized x0. Axes: (B, C, D=row, H=col, W=height).

    qr_u8 shape is (B, D, H); the conditioning tensor is broadcast along W
    (Z axis) so the QR footprint repeats up the whole height — matching
    `occ_u8` whose W came from the dataset's per-version grid_z.
    """
    occ = occ_u8.to(device, non_blocking=True).float().mul_(2.0).sub_(1.0)
    rgb = rgb_u8.to(device, non_blocking=True).float().mul_(1 / 127.5).sub_(1.0)
    qr = qr_u8.to(device, non_blocking=True).float()
    x0 = torch.cat([rgb, occ.unsqueeze(1)], dim=1)
    W = x0.shape[-1]
    cond = qr.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, W).contiguous()
    return x0, cond


def _unwrap(m):
    return m.module if isinstance(m, DDP) else m


def evaluate(model, diff, loader, device) -> dict:
    """Compute mean losses and occupancy IoU on the validation set.

    IoU is estimated by injecting low-level noise (t = T/10) and measuring
    denoising accuracy at that timestep.
    """
    model.eval()
    parts_sum = {k: 0.0 for k in LOSS_KEYS}
    parts_sum["total"] = 0.0
    iou_num = iou_den = 0.0
    n_batches = 0
    t_probe = max(1, T // 10)
    with torch.no_grad():
        for occ_u8, rgb_u8, qr_u8, th, attr, ver in loader:
            x0, cond = _normalize_batch(occ_u8, rgb_u8, qr_u8, device)
            theme = th.to(device, non_blocking=True)
            attr = attr.to(device, non_blocking=True)
            ver = ver.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=_AMP_TH):
                loss, info = diff.p_losses(model, x0, cond, theme, attr, version=ver)
            for k, v in info.items():
                parts_sum[k] += v.item()
            parts_sum["total"] += loss.item()
            # Inject fixed small-t noise, recover x0, threshold occupancy for IoU.
            tb = torch.full((x0.size(0),), t_probe, dtype=torch.long, device=device)
            noise = torch.randn_like(x0)
            xt = diff.q_sample(x0, tb, noise)
            am = torch.ones(x0.size(0), device=device)
            with torch.amp.autocast("cuda", dtype=_AMP_TH):
                v_pred = _unwrap(model)(xt, tb, cond, theme, attr, am, ver)
            x0_pred = diff.x0_from_v(xt, tb, v_pred.float())
            occ_pred = (x0_pred[:, 3] > 0).float()
            occ_gt = (x0[:, 3] > 0).float()
            iou_num += (occ_pred * occ_gt).sum().item()
            iou_den += (occ_pred + occ_gt - occ_pred * occ_gt).sum().item()
            n_batches += 1

    if IS_DIST:
        buf = torch.tensor(
            [parts_sum[k] for k in LOSS_KEYS] + [parts_sum["total"],
             iou_num, iou_den, float(n_batches)],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        vals = buf.tolist()
        for i, k in enumerate(LOSS_KEYS):
            parts_sum[k] = vals[i]
        parts_sum["total"] = vals[len(LOSS_KEYS)]
        iou_num, iou_den = vals[len(LOSS_KEYS) + 1], vals[len(LOSS_KEYS) + 2]
        n_batches = max(int(vals[len(LOSS_KEYS) + 3]), 1)

    for k in parts_sum:
        parts_sum[k] /= max(n_batches, 1)
    parts_sum["iou"] = iou_num / max(iou_den, 1e-6)
    model.train()
    return parts_sum


def save_epoch_json(ep, occ, rgb_pred, qrs_np, theme_np, path, version):
    """Write a gallery-compatible JSON file for the given epoch's samples."""
    qe = qr_modules(version)
    gxy = qrs_np.shape[1]                # padded XY grid edge
    off = (gxy - qe) // 2
    ctr = qe // 2
    samples = []
    for idx in range(len(occ)):
        theme = DS_THEMES[int(theme_np[idx])]
        th = VTHEMES[theme]
        cells = []
        core = qrs_np[idx][off:off + qe, off:off + qe]
        for i in range(qe):
            for j in range(qe):
                col_hex = th["qr_dark"] if core[i, j] else th["qr_light"]
                cells.append([int(j - ctr), 0, int(i - ctr), 1.0, col_hex])
        rs, cs, hs = np.where(occ[idx])
        for r, c, k in zip(rs, cs, hs):
            if k < 1:
                continue
            rr = int(np.clip((rgb_pred[idx, 0, r, c, k] + 1) * 127.5, 0, 255))
            gg = int(np.clip((rgb_pred[idx, 1, r, c, k] + 1) * 127.5, 0, 255))
            bb = int(np.clip((rgb_pred[idx, 2, r, c, k] + 1) * 127.5, 0, 255))
            color = f"#{rr:02x}{gg:02x}{bb:02x}"
            x = int(c - (off + ctr))
            z = int(r - (off + ctr))
            cells.append([x, int(k), z, 1.0, color])
        samples.append({"theme": theme, "cells": cells, "count": len(cells)})
    with open(path, "w") as f:
        _json.dump({"ep": ep, "version": version, "samples": samples}, f)


def save_loss_curve(hist, path):
    import matplotlib.pyplot as plt
    if not hist:
        return
    epochs = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [h["train"]["total"] for h in hist], label="train_total", lw=1.5)
    ax.plot(epochs, [h["val_total"] for h in hist], label="val_total", lw=1.2)
    ax.plot(epochs, [h["train"]["v_rgb"] for h in hist], label="v_rgb", alpha=0.6)
    ax.plot(epochs, [h["train"]["v_occ"] for h in hist], label="v_occ", alpha=0.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def sample_montage(diff, ema, mont_batch, mont_version, ep, png_path, device,
                   json_path=None):
    """Sample the EMA model stochastically for each theme and write PNG + JSON."""
    m_eval = DiT3D(dim=DIT_DIM, depth=DIT_DEPTH, heads=DIT_HEADS,
                   patch=DIT_PATCH, n_themes=len(DS_THEMES)).to(device)
    ema.copy_to(m_eval)
    m_eval.eval()
    occ_u8, rgb_u8, qr_u8, th = mont_batch
    qr = qr_u8.float().to(device)
    # Montage uses the mont_version-specific Z extent so the sampled grid
    # matches the data distribution at that version.
    mont_gz = grid_z_for_version(mont_version)
    cond = qr.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, mont_gz).contiguous()
    theme = th.to(device)
    # Use mid-range attributes [0.5, 0.5, 0.5] to visualize an average tree shape.
    attr = torch.full((th.size(0), 3), 0.5, device=device)
    # All montage samples share the same QR version so we pass a scalar-broadcast.
    ver = torch.full((th.size(0),), int(mont_version), device=device, dtype=torch.long)
    with torch.no_grad():
        x = diff.sample(m_eval, cond, theme, attr=attr, steps=SAMPLE_STEPS,
                        device=device, eta=1.0, version=ver).cpu().numpy()
    occ = (x[:, 3] > 0)
    qr_np = qr_u8.numpy()
    qr_mask = (qr_np > 0)[:, :, :, None]
    occ = occ & np.broadcast_to(qr_mask, occ.shape)
    rgb = x[:, :3]
    th_np = th.numpy()
    render_montage(occ, rgb, qr_np, th_np, png_path)
    if json_path is not None:
        save_epoch_json(ep, occ.astype(np.float32), rgb, qr_np, th_np, json_path,
                        version=mont_version)
    del m_eval
    torch.cuda.empty_cache()


def main():
    if IS_MAIN:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    if IS_DIST:
        dist.barrier()

    # Training samples are generated live; validation uses a fixed seed for reproducibility.
    train_ds = LiveTreeDataset(EPOCH_SIZE, versions=QR_VERSIONS, deterministic=False,
                                weights=QR_VERSION_WEIGHTS)
    val_ds = LiveTreeDataset(VAL_SIZE, versions=QR_VERSIONS_VAL, deterministic=True, seed=999)
    _w_sum = sum(QR_VERSION_WEIGHTS) or 1.0
    _w_pct = [f"v{v}:{100*w/_w_sum:.0f}%" for v, w in zip(QR_VERSIONS, QR_VERSION_WEIGHTS)]
    log(f"[data] live generation — epoch_size={EPOCH_SIZE} val={VAL_SIZE}  "
        f"train_versions={QR_VERSIONS}  val_versions={QR_VERSIONS_VAL}  "
        f"max_xy={MAX_XY}  max_z={MAX_Z}")
    log(f"[data] train weights — {' '.join(_w_pct)}  (raw={QR_VERSION_WEIGHTS})")

    if IS_DIST:
        train_sampler = DistributedSampler(train_ds, num_replicas=WORLD_SIZE,
                                           rank=RANK, shuffle=True, drop_last=True, seed=SEED)
        val_sampler = DistributedSampler(val_ds, num_replicas=WORLD_SIZE,
                                         rank=RANK, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    _dl_kw = dict(num_workers=WORKERS, pin_memory=True, collate_fn=pad_collate)
    if WORKERS > 0:
        _dl_kw.update(persistent_workers=True, prefetch_factor=4)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=train_sampler,
                          shuffle=(train_sampler is None), drop_last=True, **_dl_kw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, sampler=val_sampler,
                        shuffle=False, **_dl_kw)
    log(f"[loader] steps/epoch(per-rank)={len(train_dl)}  val batches(per-rank)={len(val_dl)}  "
        f"workers={WORKERS}  amp={AMP_DTYPE}  world={WORLD_SIZE}  batch/gpu={BATCH}  "
        f"global_batch={BATCH * WORLD_SIZE}")

    # Build 9 unseen QR codes (one per theme) for the per-epoch sample montage.
    # Versions are sampled from QR_VERSIONS_VAL so the montage exercises the
    # multi-version code path. All 9 codes share the SAME version per epoch so
    # they stack into one batch tensor.
    mont_batch = None
    mont_version = None
    if IS_MAIN:
        import segno, random
        import string
        rng = random.Random(20260520)
        url_chars = string.ascii_lowercase + string.digits + "-."
        alnum_chars = string.ascii_letters + string.digits
        PORTFOLIO_URL = os.environ.get("DEMO_URL", "https://example.com")
        # Pick a version (rotates each restart via the seed); the same version
        # is used for all 9 montage samples.
        mont_version = QR_VERSIONS_VAL[rng.randint(0, len(QR_VERSIONS_VAL) - 1)]
        m_qe = qr_modules(mont_version)
        m_gxy = grid_xy_for_version(mont_version)
        m_gz  = grid_z_for_version(mont_version)
        m_off = (m_gxy - m_qe) // 2

        unseen_qrs, unseen_texts = [], []
        for t in range(9):
            if mont_version >= 3 and t == 5:
                text = PORTFOLIO_URL
            else:
                while True:
                    if mont_version == 1:
                        n = rng.randint(6, 16)
                        text = "".join(rng.choices(alnum_chars, k=n))
                    else:
                        n = rng.randint(8, min(40, m_qe))
                        if rng.random() < 0.5 and n >= 12:
                            text = "https://" + "".join(rng.choices(url_chars, k=n - 8))
                        else:
                            text = "".join(rng.choices(url_chars, k=n))
                    try:
                        segno.make(text, error="m", version=mont_version); break
                    except Exception:
                        continue
            qr = segno.make(text, error="m", version=mont_version)
            core = np.array([[1 if c else 0 for c in row] for row in qr.matrix],
                            dtype=np.uint8)
            pad = np.zeros((m_gxy, m_gxy), dtype=np.uint8)
            pad[m_off:m_off + m_qe, m_off:m_off + m_qe] = core
            unseen_qrs.append(pad); unseen_texts.append(text)
        unseen_qrs = np.stack(unseen_qrs)
        mont_batch = (
            torch.zeros(9, m_gxy, m_gxy, m_gz, dtype=torch.uint8),
            torch.zeros(9, 3, m_gxy, m_gxy, m_gz, dtype=torch.uint8),
            torch.from_numpy(unseen_qrs),
            torch.arange(9, dtype=torch.long),
        )
        log(f"[montage] unseen QRs (version={mont_version}, "
            f"grid={m_gxy}x{m_gxy}x{m_gz}) — texts: {unseen_texts}")

        from qrbloom.treegen import generate_voxels
        gt_samples = []
        for i in range(9):
            theme_name = DS_THEMES[i]
            qr = segno.make(unseen_texts[i], error="m", version=mont_version)
            core_bool = [[bool(c) for c in row] for row in qr.matrix]
            voxels = generate_voxels(core_bool, theme=theme_name)
            cells = [[int(v["pos"][0]), int(v["pos"][1]), int(v["pos"][2]),
                      float(v["scale"]), v["color"]] for v in voxels]
            gt_samples.append({"theme": theme_name, "text": unseen_texts[i],
                               "version": mont_version,
                               "cells": cells, "count": len(cells)})
        with open(str(OUT_DIR / f"unseen_gt{SUFFIX}.json"), "w") as f:
            _json.dump({"version": mont_version,
                        "trained_versions": sorted(set(QR_VERSIONS)),
                        "val_versions": sorted(set(QR_VERSIONS_VAL)),
                        "samples": gt_samples}, f)
        log("[montage] saved ground-truth JSON")

    # ─── Model ────────────────────────────────────────────────────────────────
    base_model = DiT3D(dim=DIT_DIM, depth=DIT_DEPTH, heads=DIT_HEADS,
                       patch=DIT_PATCH, n_themes=len(DS_THEMES),
                       grad_checkpoint=DIT_CKPT).to(DEVICE)
    n_params = sum(p.numel() for p in base_model.parameters())
    log(f"[model] DiT3D params={n_params/1e6:.2f}M  "
        f"dim={DIT_DIM} depth={DIT_DEPTH} heads={DIT_HEADS} patch={DIT_PATCH}")
    diff = Diffusion(T=T, device=DEVICE)
    ema = EMA(base_model, decay=0.999)

    if IS_DIST:
        model = DDP(base_model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    else:
        model = base_model

    if COMPILE:
        log("[compile] torch.compile(mode='reduce-overhead')")
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)

    opt = torch.optim.Adam(base_model.parameters(), lr=BASE_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(_AMP_TH is torch.float16))

    def lr_at(step: int, total_steps: int):
        from math import cos, pi
        return BASE_LR * 0.5 * (1 + cos(pi * step / max(total_steps, 1)))

    # ─── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    hist = []
    best_val = float("inf")
    if CKPT.exists():
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        base_model.load_state_dict(ck["model"])
        ema.shadow = {k: v.to(DEVICE) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck and scaler.is_enabled():
            try:
                scaler.load_state_dict(ck["scaler"])
            except Exception:
                pass
        start_epoch = ck["epoch"] + 1
        hist = ck["hist"]
        if hist:
            vals = [h.get("val_total") for h in hist if h.get("val_total") is not None]
            if vals:
                best_val = min(vals)
        log(f"[resume] epoch {ck['epoch']} -> start epoch {start_epoch}  best_val={best_val:.4f}")
    else:
        log("[init] no checkpoint found, training from scratch")

    total_steps = EPOCHS * len(train_dl)
    cur_step = (start_epoch - 1) * len(train_dl)

    # ─── Training loop ────────────────────────────────────────────────────────
    for ep in range(start_epoch, EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        epoch_parts = {k: 0.0 for k in LOSS_KEYS}
        epoch_parts["total"] = 0.0
        pbar = tqdm(train_dl, desc=f"ep{ep:03d}", dynamic_ncols=True, disable=not IS_MAIN)
        n_batches = 0
        for occ_b, rgb_b, qr_b, th_b, attr_b, ver_b in pbar:
            lr = lr_at(cur_step, total_steps)
            for pg in opt.param_groups: pg["lr"] = lr

            x0, cond = _normalize_batch(occ_b, rgb_b, qr_b, DEVICE)
            theme = th_b.to(DEVICE, non_blocking=True)
            attr = attr_b.to(DEVICE, non_blocking=True)
            ver = ver_b.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=_AMP_TH):
                loss, info = diff.p_losses(model, x0, cond, theme, attr, version=ver)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                opt.step()
            ema.update(base_model)
            cur_step += 1
            n_batches += 1

            for k, v in info.items(): epoch_parts[k] += v.item()
            epoch_parts["total"] += loss.item()
            if IS_MAIN and n_batches % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 v_rgb=f"{info['v_rgb'].item():.3f}",
                                 v_occ=f"{info['v_occ'].item():.3f}")

        for k in epoch_parts: epoch_parts[k] /= max(n_batches, 1)
        if IS_DIST:
            keys = LOSS_KEYS + ("total",)
            buf = torch.tensor([epoch_parts[k] for k in keys],
                               device=DEVICE, dtype=torch.float64)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf /= WORLD_SIZE
            for i, k in enumerate(keys):
                epoch_parts[k] = buf[i].item()

        # ─── Validation ───────────────────────────────────────────────────────
        if ep % VAL_EVERY == 0 or ep == EPOCHS:
            val_metrics = evaluate(model, diff, val_dl, DEVICE)
        else:
            val_metrics = {"total": float("inf"), "iou": 0.0}
            for k in LOSS_KEYS:
                val_metrics[k] = 0.0
        log(f"[ep{ep:03d}] train_total={epoch_parts['total']:.4f}  "
            f"val_total={val_metrics['total']:.4f}  val_iou={val_metrics['iou']:.3f}  "
            f"lr={lr:.2e}")
        log(f"   parts: v_mse={epoch_parts['v_mse']:.4f}  "
            f"v_rgb={epoch_parts['v_rgb']:.4f}  v_occ={epoch_parts['v_occ']:.4f}")

        hist.append({
            "epoch": ep,
            "train": epoch_parts,
            "val_total": val_metrics["total"],
            "val_iou": val_metrics["iou"],
            "val_parts": val_metrics,
            "lr": lr,
        })

        # ─── Visualization / checkpointing (rank 0) ───────────────────────────
        if IS_MAIN:
            png = str(OUT_DIR / f"epoch_{ep:03d}.png")
            epoch_json = str(OUT_DIR / f"epoch_{ep:03d}.json")
            sample_montage(diff, ema, mont_batch, mont_version, ep, png, DEVICE,
                           json_path=epoch_json)
            save_loss_curve(hist, str(OUT_DIR / "loss.png"))
            log(f"ep{ep:03d} loss={epoch_parts['total']:.4f} "
                f"v_rgb={epoch_parts['v_rgb']:.4f} v_occ={epoch_parts['v_occ']:.4f}")
            log(f"  -> {png}  {epoch_json}")

            state = {"model": base_model.state_dict(), "ema": ema.shadow,
                     "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "epoch": ep, "hist": hist}
            _safe_save(state, CKPT)
            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                _safe_save(state, CKPT_BEST)
                log(f"  new best val_total={best_val:.4f}")
        if IS_DIST:
            dist.barrier()

    if IS_MAIN:
        DONE.touch()
        log(f"[done] {EPOCHS} epochs complete. checkpoint: {CKPT}")
    if IS_DIST:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
