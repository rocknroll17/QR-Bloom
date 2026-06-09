# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""Quantitative evaluation for QR-Bloom.

Measures four properties of generated voxel trees against treegen ground truth:
  1. Occupancy fidelity  — IoU and voxel count ratio vs. GT
  2. Color fidelity      — mean nearest-palette distance (lower = more vivid)
  3. Color diversity     — number of distinct palette colors used per tree
  4. Sample diversity    — pairwise IoU between samples from the same QR
                           (lower = more varied outputs)
"""
import argparse
import random
import string

import numpy as np
import segno
import torch

from qrbloom.diffusion import UNet3D, Diffusion
from qrbloom.treegen import THEMES, generate_voxels
from qrbloom.qr import (THEME_NAMES as _THEMES, QR_OFFSET as OFF,
                         QR_SIZE as QE, QR_VERSION as QV)

VOX = 32
CENTER = QE // 2


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]


def theme_palette(theme):
    """Return the tree color palette (leaves, flowers, trunk) as (K, 3) float array."""
    th = THEMES[theme]
    cols = list(th.get("leaf", []))
    cols += list(th.get("flower", []))
    if "trunk" in th:
        cols.append(th["trunk"])
    return np.array([hex2rgb(c) for c in cols], dtype=np.float64)


def load_model(checkpoint_path, device):
    """Load UNet3D and Diffusion from a checkpoint file.

    Expects the checkpoint to contain an 'ema' key with model weights
    and optionally an 'epoch' key for display.
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet3D(ch=48).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = Diffusion(T=500, device=device)

    def make_sampler(eta):
        return lambda cond, th, steps: diff.sample(
            model, cond, th, steps=steps, device=device, eta=eta)

    return model, make_sampler, ck.get("epoch", "?")


def unseen_qrs(n=9):
    """Generate n held-out QR codes (fixed seed, not seen during training)."""
    rng = random.Random(20260520)
    url_chars = string.ascii_lowercase + string.digits + "-."
    v1_chars = string.ascii_letters + string.digits
    out = []
    for _ in range(n):
        while True:
            if QV == 1:
                txt = "".join(rng.choices(v1_chars, k=rng.randint(6, 16)))
            else:
                k = rng.randint(8, 40)
                txt = "".join(rng.choices(url_chars, k=k))
            try:
                qr = segno.make(txt, error="m", version=QV)
                break
            except Exception:
                continue
        core = np.array([[1 if c else 0 for c in row] for row in qr.matrix],
                        dtype=np.uint8)
        out.append((txt, core))
    return out


def voxels_to_grid(voxels):
    """Convert a treegen voxel list to occupancy (32³ bool) and color (32³, 3) arrays.

    Base voxels (is_base=True) are excluded; only tree voxels are included.
    """
    occ = np.zeros((VOX, VOX, VOX), dtype=bool)
    col = np.zeros((VOX, VOX, VOX, 3), dtype=np.float64)
    for v in voxels:
        if v["is_base"]:
            continue
        x, y, z = v["pos"]
        i = OFF + int(round(z)) + CENTER
        j = OFF + int(round(x)) + CENTER
        k = int(round(y))
        if 0 <= i < VOX and 0 <= j < VOX and 0 <= k < VOX:
            occ[i, j, k] = True
            col[i, j, k] = hex2rgb(v["color"])
    return occ, col


def pred_to_grid(x):
    """Convert model output (4, 32, 32, 32) in [-1, 1] to occupancy bool and color [0, 255]."""
    occ = x[3] > 0
    rgb = np.clip((x[:3] + 1) * 127.5, 0, 255)     # (3, 32, 32, 32)
    col = np.transpose(rgb, (1, 2, 3, 0))           # (32, 32, 32, 3)
    return occ, col


def color_metrics(occ, col, palette):
    """Compute mean nearest-palette distance and number of distinct palette colors used."""
    pts = col[occ]                                   # (N_vox, 3)
    if len(pts) == 0:
        return 0.0, 0
    d = np.linalg.norm(pts[:, None, :] - palette[None, :, :], axis=2)  # (N, K)
    nearest = d.argmin(axis=1)
    on_palette_dist = d.min(axis=1).mean()           # range 0-255; lower = more vivid
    distinct = len(np.unique(nearest))
    return float(on_palette_dist), int(distinct)


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union else 1.0


def main():
    ap = argparse.ArgumentParser(description="Evaluate QR-Bloom generation quality.")
    ap.add_argument("--checkpoint", default="checkpoints/qrbloom_best.pt",
                    help="Path to a .pt checkpoint file.")
    ap.add_argument("--samples", type=int, default=3,
                    help="Number of samples to draw per QR code.")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--eta", type=float, default=1.0,
                    help="Sampler stochasticity: 1.0 = ancestral (DDPM), "
                         "0.0 = deterministic (DDIM).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, make_sampler, ep = load_model(args.checkpoint, device)
    sampler = make_sampler(args.eta)
    print(f"=== QR-Bloom eval  checkpoint={args.checkpoint}  epoch={ep}  "
          f"samples/QR={args.samples}  steps={args.steps}  "
          f"eta={args.eta}  device={device} ===\n")

    qrs = unseen_qrs(9)
    agg = {"iou": [], "count_ratio": [], "on_pal": [], "distinct": [],
           "gt_distinct": [], "inter_iou": []}

    hdr = (f"{'theme':18s} {'IoU':>6s} {'cnt/GT':>8s} {'pal_dist':>10s} "
           f"{'colors(p/gt)':>14s} {'inter_IoU':>10s}")
    print(hdr)
    print("-" * len(hdr))

    for ti in range(9):
        theme = _THEMES[ti]
        txt, core = qrs[ti]
        pal = theme_palette(theme)

        # Ground-truth voxel grid from treegen
        core_bool = [[bool(c) for c in row] for row in core]
        gt_occ, gt_col = voxels_to_grid(generate_voxels(core_bool, theme=theme))
        _, gt_distinct = color_metrics(gt_occ, gt_col, pal)
        gt_count = int(gt_occ.sum())

        # Build conditioning tensor: QR footprint broadcast to 32³
        pad = np.zeros((VOX, VOX), dtype=np.float32)
        pad[OFF:OFF + QE, OFF:OFF + QE] = core
        qr_t = torch.from_numpy(pad).to(device)
        cond = qr_t.view(1, 1, VOX, VOX, 1).expand(1, 1, VOX, VOX, VOX).contiguous()
        th_t = torch.tensor([ti], device=device)

        # Draw multiple samples and restrict occupancy to the QR footprint
        sample_occs, sample_cols = [], []
        with torch.no_grad():
            for _ in range(args.samples):
                x = sampler(cond, th_t, args.steps).cpu().numpy()[0]
                o, c = pred_to_grid(x)
                mask = pad > 0.5
                o = o & mask[:, :, None]
                sample_occs.append(o)
                sample_cols.append(c)

        # Average metrics over all samples (single samples are noisy under stochastic sampling)
        s_iou = float(np.mean([iou(o, gt_occ) for o in sample_occs]))
        s_count = float(np.mean([o.sum() for o in sample_occs]))
        _cms = [color_metrics(sample_occs[i], sample_cols[i], pal)
                for i in range(len(sample_occs))]
        on_pal = float(np.mean([c[0] for c in _cms]))
        distinct = float(np.mean([c[1] for c in _cms]))

        # Pairwise IoU across samples — lower means more diverse outputs
        inter = []
        for a in range(len(sample_occs)):
            for b in range(a + 1, len(sample_occs)):
                inter.append(iou(sample_occs[a], sample_occs[b]))
        inter_iou = float(np.mean(inter)) if inter else float("nan")

        cnt_ratio = s_count / max(gt_count, 1)
        print(f"{theme:18s} {s_iou:6.3f} {cnt_ratio:8.2f} {on_pal:10.1f} "
              f"{distinct:5.1f}/{gt_distinct:<5d} {inter_iou:10.3f}")

        agg["iou"].append(s_iou)
        agg["count_ratio"].append(cnt_ratio)
        agg["on_pal"].append(on_pal)
        agg["distinct"].append(distinct)
        agg["gt_distinct"].append(gt_distinct)
        if not np.isnan(inter_iou):
            agg["inter_iou"].append(inter_iou)

    print("-" * len(hdr))
    print(f"{'mean':18s} {np.mean(agg['iou']):6.3f} "
          f"{np.mean(agg['count_ratio']):8.2f} {np.mean(agg['on_pal']):10.1f} "
          f"{np.mean(agg['distinct']):6.1f}/{np.mean(agg['gt_distinct']):<6.1f} "
          f"{np.mean(agg['inter_iou']):10.3f}")
    print()
    print("Guide:")
    print("  IoU up, cnt/GT near 1      -> structure matches GT")
    print("  pal_dist down              -> vivid colors (0-255; higher = muddier)")
    print("  colors pred near GT        -> color diversity matches GT")
    print("  inter_IoU down             -> varied trees from the same QR")


if __name__ == "__main__":
    main()
