# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""Training-run visualization: per-epoch montages, loss curves, and the
gallery-compatible per-epoch voxel JSON."""
from __future__ import annotations

import json

import numpy as np

from qrbloom.qr import qr_modules
from qrbloom.treegen import SPECIES, THEME_NAMES


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
        ax.set_xlim(0, D)
        ax.set_ylim(0, H)
        ax.set_zlim(0, W)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f"{THEME_NAMES[themes[i]]}\nn={occ_bin[i].sum()}", fontsize=7)
        ax2 = fig.add_subplot(2, B, B + i + 1)
        top = occ_bin[i].max(axis=2)
        ax2.imshow(np.stack([qrs[i], top, np.zeros_like(top)], axis=-1) * 0.7 + 0.1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("R=QR  G=pred top", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def save_epoch_json(ep, occ, rgb_pred, qrs_np, theme_np, path, version):
    """Write a gallery-compatible JSON file for the given epoch's samples."""
    qe = qr_modules(version)
    gxy = qrs_np.shape[1]                # padded XY grid edge
    off = (gxy - qe) // 2
    ctr = qe // 2
    samples = []
    for idx in range(len(occ)):
        theme = THEME_NAMES[int(theme_np[idx])]
        sp = SPECIES[theme]
        cells = []
        core = qrs_np[idx][off:off + qe, off:off + qe]
        for i in range(qe):
            for j in range(qe):
                col_hex = sp.qr_dark if core[i, j] else sp.qr_light
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
        json.dump({"ep": ep, "version": version, "samples": samples}, f)


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
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
