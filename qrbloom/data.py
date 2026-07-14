"""Live training data for QR-Bloom.

Samples are generated on the fly — a procedural voxel tree grown inside a
random QR footprint — so the dataset is effectively infinite and nothing is
stored on disk. Batches are bucketed per QR version (NovelAI-style
aspect-ratio bucketing): every batch holds a single grid size, so there are
no padding voxels and no attention over padding tokens.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from qrbloom.model import DOWNSCALE
from qrbloom.qr import (grid_xy_for_version, grid_z_for_version, pad_to_grid,
                        qr_modules, random_qr_core)
from qrbloom.treegen import THEME_NAMES, generate_voxels_aug, tree_attributes


def version_for_index(idx: int, versions, weights, batch: int, seed: int) -> int:
    """QR version for sample `idx`, constant within each batch-sized block.

    BlockBatchSampler emits contiguous index blocks and this pure function
    assigns each block a single version, so every batch is uniform in grid
    size. Being a pure function of (idx, seed), the sampler and every
    DataLoader worker agree on the schedule without shared state.
    """
    if len(versions) == 1:
        return versions[0]
    ws = weights if weights is not None else [1.0] * len(versions)
    r = random.Random(seed * 1000003 + idx // batch).random() * sum(ws)
    acc = 0.0
    for v, w in zip(versions, ws):
        acc += w
        if r < acc:
            return v
    return versions[-1]


class BlockBatchSampler(torch.utils.data.Sampler):
    """Yields contiguous index blocks of size `batch` (one QR version each).

    Blocks are interleaved across DDP ranks and their order is shuffled per
    epoch; indices inside a block stay contiguous so `version_for_index`
    holds one version for the whole batch.
    """

    def __init__(self, n: int, batch: int, world: int = 1, rank: int = 0,
                 shuffle: bool = True, seed: int = 0):
        self.batch = batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.blocks = [b for b in range(n // batch) if b % world == rank]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.blocks)

    def __iter__(self):
        order = list(self.blocks)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for b in order:
            yield list(range(b * self.batch, (b + 1) * self.batch))


class LiveTreeDataset(torch.utils.data.Dataset):
    """Generates a fresh (QR, theme, augmentation) sample on every access.

    Each sample is built for the QR version that `version_for_index` assigns
    to its batch block, so grids inside one batch always match. The voxel
    grid's XY *and Z* size both depend on the version (isotropic scaling —
    see qrbloom.qr.grid_z_for_version).

    deterministic=False (train): each sample uses system entropy — no repeats.
    deterministic=True  (val):   each index maps to a fixed seed — reproducible.

    Returns: (occ_u8, rgb_u8, qr_u8, theme_idx, attr, version)
    """

    def __init__(self, virtual_len, versions, deterministic, seed=0, weights=None,
                 batch=1):
        self.virtual_len = int(virtual_len)
        self.versions = list(versions)
        # weights=None → uniform sampling (used for val so the metric is
        # version-balanced). Train passes per-version weights to oversample
        # the harder lower QR versions.
        self.weights = list(weights) if weights is not None else None
        self.deterministic = deterministic
        self.seed = seed
        self.batch = int(batch)

    def __len__(self):
        return self.virtual_len

    def _rng(self, idx):
        if self.deterministic:
            s = (self.seed * 1000003 + idx) & 0x7FFFFFFF
            return random.Random(s), np.random.default_rng(s)
        return random.Random(), np.random.default_rng()

    def __getitem__(self, idx):
        rng, nprng = self._rng(idx)
        version = version_for_index(idx, self.versions, self.weights,
                                    self.batch, self.seed)
        qe = qr_modules(version)
        gxy = grid_xy_for_version(version)
        gz = grid_z_for_version(version)
        off = (gxy - qe) // 2
        ctr = qe // 2

        core = random_qr_core(nprng, version=version)         # (qe, qe) uint8
        ti = rng.randint(0, len(THEME_NAMES) - 1)
        voxels = generate_voxels_aug(core.tolist(), theme=THEME_NAMES[ti], rng=rng)

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

    With bucketed batching every batch is a single version, so this is a
    plain stack; the padding path remains as a safety net for callers that
    mix versions. XY padding is symmetric around the QR center; Z padding
    is one-sided (extend upward) so the QR base plane stays at z=0. Both
    are rounded up to multiples of DOWNSCALE.

    Input shapes per sample:
      occ : (gxy, gxy, gz)
      rgb : (3, gxy, gxy, gz)
      qr  : (gxy, gxy)              — 2D footprint, no Z
    """
    occs, rgbs, qrs, ths, attrs, vers = zip(*batch)
    max_xy = max(o.shape[0] for o in occs)
    max_z = max(o.shape[2] for o in occs)
    max_xy = ((max_xy + DOWNSCALE - 1) // DOWNSCALE) * DOWNSCALE
    max_z = ((max_z + DOWNSCALE - 1) // DOWNSCALE) * DOWNSCALE

    def _pad_occ(t: torch.Tensor) -> torch.Tensor:
        # (D, H, W) — pad W (Z) one-sided, D/H symmetric.
        D, H, W = t.shape
        pz = max_z - W
        pad_d_l = (max_xy - D) // 2
        pad_d_r = (max_xy - D) - pad_d_l
        pad_h_l = (max_xy - H) // 2
        pad_h_r = (max_xy - H) - pad_h_l
        # F.pad order for 3D tensor: (W_l, W_r, H_l, H_r, D_l, D_r)
        return F.pad(t, (0, pz, pad_h_l, pad_h_r, pad_d_l, pad_d_r))

    def _pad_rgb(t: torch.Tensor) -> torch.Tensor:
        # (C, D, H, W)
        _, D, H, W = t.shape
        pz = max_z - W
        pad_d_l = (max_xy - D) // 2
        pad_d_r = (max_xy - D) - pad_d_l
        pad_h_l = (max_xy - H) // 2
        pad_h_r = (max_xy - H) - pad_h_l
        return F.pad(t, (0, pz, pad_h_l, pad_h_r, pad_d_l, pad_d_r))

    def _pad_qr(t: torch.Tensor) -> torch.Tensor:
        # (D, H)
        D, H = t.shape
        pad_d_l = (max_xy - D) // 2
        pad_d_r = (max_xy - D) - pad_d_l
        pad_h_l = (max_xy - H) // 2
        pad_h_r = (max_xy - H) - pad_h_l
        return F.pad(t, (pad_h_l, pad_h_r, pad_d_l, pad_d_r))

    occ_b = torch.stack([_pad_occ(o) for o in occs], 0)
    rgb_b = torch.stack([_pad_rgb(r) for r in rgbs], 0)
    qr_b = torch.stack([_pad_qr(q) for q in qrs], 0)
    th_b = torch.tensor(list(ths), dtype=torch.long)
    attr_b = torch.stack(list(attrs), 0)
    ver_b = torch.tensor(list(vers), dtype=torch.long)
    return occ_b, rgb_b, qr_b, th_b, attr_b, ver_b
