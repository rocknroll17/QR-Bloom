# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""Quantitative evaluation for QR-Bloom.

Measures generated voxel trees against treegen ground truth, per QR version:
  1. Occupancy fidelity  — IoU and voxel count ratio vs. GT
  2. Color fidelity      — mean nearest-palette distance (lower = more vivid)
  3. Color diversity     — number of distinct palette colors used per tree
  4. Sample diversity    — pairwise IoU between samples from the same QR
                           (lower = more varied outputs)

Generation is conditioned the same way the demo serves it: per-species,
per-version mean attributes. Held-out QR codes come from a fixed seed.
"""
import argparse

import numpy as np
import torch

from qrbloom.model import DiT3D, Diffusion
from qrbloom.qr import (THEME_NAMES, grid_xy_for_version, grid_z_for_version,
                        qr_modules, random_qr_core)
from qrbloom.treegen import SPECIES, attr_means, generate_voxels


def hex2rgb(h):
    h = h.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]


def theme_palette(theme):
    """Tree color palette (leaves, flowers, trunk) as a (K, 3) float array."""
    sp = SPECIES[theme]
    cols = list(sp.leaf) + list(sp.flower) + [sp.trunk]
    return np.array([hex2rgb(c) for c in cols], dtype=np.float64)


def load_model(checkpoint_path, device):
    """Load DiT3D (EMA weights) and the diffusion process from a checkpoint."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DiT3D(n_themes=len(THEME_NAMES)).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = Diffusion(T=500, device=device)
    return model, diff, ck.get("epoch", "?")


def voxels_to_grid(voxels, gxy, gz, off, ctr):
    """Treegen voxel list → occupancy (gxy,gxy,gz) bool and color (...,3) arrays.

    Base voxels (is_base=True) are excluded; only tree voxels are included.
    """
    occ = np.zeros((gxy, gxy, gz), dtype=bool)
    col = np.zeros((gxy, gxy, gz, 3), dtype=np.float64)
    for v in voxels:
        if v["is_base"]:
            continue
        x, y, z = v["pos"]
        i = off + int(round(z)) + ctr
        j = off + int(round(x)) + ctr
        k = int(round(y))
        if 0 <= i < gxy and 0 <= j < gxy and 0 <= k < gz:
            occ[i, j, k] = True
            col[i, j, k] = hex2rgb(v["color"])
    return occ, col


def pred_to_grid(x):
    """Model output (4, D, H, W) in [-1, 1] → occupancy bool and color [0, 255]."""
    occ = x[3] > 0
    rgb = np.clip((x[:3] + 1) * 127.5, 0, 255)
    col = np.transpose(rgb, (1, 2, 3, 0))
    return occ, col


def color_metrics(occ, col, palette):
    """Mean nearest-palette distance and number of distinct palette colors used."""
    pts = col[occ]
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


def eval_version(model, diff, version, device, samples, steps, eta, seed):
    """Evaluate all themes at one QR version. Returns per-metric lists."""
    qe = qr_modules(version)
    gxy, gz = grid_xy_for_version(version), grid_z_for_version(version)
    off = (gxy - qe) // 2
    ctr = qe // 2
    nprng = np.random.default_rng(seed + version)

    agg = {"iou": [], "count_ratio": [], "on_pal": [], "distinct": [],
           "gt_distinct": [], "inter_iou": []}
    hdr = (f"{'theme':16s} {'IoU':>6s} {'cnt/GT':>8s} {'pal_dist':>9s} "
           f"{'colors(p/gt)':>13s} {'inter_IoU':>10s}")
    print(f"--- v{version}  grid={gxy}x{gxy}x{gz} ---")
    print(hdr)

    for theme in THEME_NAMES:
        core = random_qr_core(nprng, version=version)          # held-out QR
        core_bool = [[bool(c) for c in row] for row in core]
        gt_occ, gt_col = voxels_to_grid(generate_voxels(core_bool, theme=theme),
                                        gxy, gz, off, ctr)
        pal = theme_palette(theme)
        _, gt_distinct = color_metrics(gt_occ, gt_col, pal)
        gt_count = int(gt_occ.sum())

        pad = np.zeros((gxy, gxy), dtype=np.float32)
        pad[off:off + qe, off:off + qe] = core
        qr_t = torch.from_numpy(pad).to(device)
        cond = qr_t.view(1, 1, gxy, gxy, 1).expand(1, 1, gxy, gxy, gz).contiguous()
        th_t = torch.tensor([THEME_NAMES.index(theme)], device=device)
        attr = torch.tensor([attr_means(theme, version)], device=device,
                            dtype=torch.float32)
        ver_t = torch.tensor([version], device=device, dtype=torch.long)

        sample_occs, sample_cols = [], []
        with torch.no_grad():
            for _ in range(samples):
                x = diff.sample(model, cond, th_t, attr=attr, steps=steps,
                                device=device, eta=eta,
                                version=ver_t).cpu().numpy()[0]
                o, c = pred_to_grid(x)
                o = o & (pad > 0.5)[:, :, None]
                sample_occs.append(o)
                sample_cols.append(c)

        s_iou = float(np.mean([iou(o, gt_occ) for o in sample_occs]))
        s_count = float(np.mean([o.sum() for o in sample_occs]))
        cms = [color_metrics(sample_occs[i], sample_cols[i], pal)
               for i in range(len(sample_occs))]
        on_pal = float(np.mean([c[0] for c in cms]))
        distinct = float(np.mean([c[1] for c in cms]))
        inter = [iou(sample_occs[a], sample_occs[b])
                 for a in range(len(sample_occs))
                 for b in range(a + 1, len(sample_occs))]
        inter_iou = float(np.mean(inter)) if inter else float("nan")
        cnt_ratio = s_count / max(gt_count, 1)

        print(f"{theme:16s} {s_iou:6.3f} {cnt_ratio:8.2f} {on_pal:9.1f} "
              f"{distinct:5.1f}/{gt_distinct:<5d} {inter_iou:10.3f}")
        agg["iou"].append(s_iou)
        agg["count_ratio"].append(cnt_ratio)
        agg["on_pal"].append(on_pal)
        agg["distinct"].append(distinct)
        agg["gt_distinct"].append(gt_distinct)
        if not np.isnan(inter_iou):
            agg["inter_iou"].append(inter_iou)

    print(f"{'v' + str(version) + ' mean':16s} {np.mean(agg['iou']):6.3f} "
          f"{np.mean(agg['count_ratio']):8.2f} {np.mean(agg['on_pal']):9.1f} "
          f"{np.mean(agg['distinct']):5.1f}/{np.mean(agg['gt_distinct']):<5.1f} "
          f"{np.mean(agg['inter_iou']):10.3f}\n")
    return agg


def main():
    ap = argparse.ArgumentParser(description="Evaluate QR-Bloom generation quality.")
    ap.add_argument("--checkpoint", default="checkpoints/qrbloom_all_best.pt",
                    help="Path to a .pt checkpoint file.")
    ap.add_argument("--versions", default="2,3,4,5",
                    help="Comma-separated QR versions to evaluate.")
    ap.add_argument("--samples", type=int, default=3,
                    help="Number of samples to draw per QR code.")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--eta", type=float, default=1.0,
                    help="Sampler stochasticity: 1.0 = ancestral (DDPM), "
                         "0.0 = deterministic (DDIM).")
    ap.add_argument("--seed", type=int, default=20260520,
                    help="Seed for the held-out QR codes.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    versions = [int(v) for v in args.versions.split(",") if v.strip()]
    model, diff, ep = load_model(args.checkpoint, device)
    print(f"=== QR-Bloom eval  checkpoint={args.checkpoint}  epoch={ep}  "
          f"versions={versions}  samples/QR={args.samples}  steps={args.steps}  "
          f"eta={args.eta}  device={device} ===\n")

    overall = {}
    for v in versions:
        agg = eval_version(model, diff, v, device, args.samples, args.steps,
                           args.eta, args.seed)
        for k, vals in agg.items():
            overall.setdefault(k, []).extend(vals)

    print("=" * 62)
    print(f"{'overall mean':16s} IoU={np.mean(overall['iou']):.3f}  "
          f"cnt/GT={np.mean(overall['count_ratio']):.2f}  "
          f"pal_dist={np.mean(overall['on_pal']):.1f}  "
          f"colors={np.mean(overall['distinct']):.1f}/{np.mean(overall['gt_distinct']):.1f}  "
          f"inter_IoU={np.mean(overall['inter_iou']):.3f}")
    print()
    print("Guide:")
    print("  IoU up, cnt/GT near 1      -> structure matches GT")
    print("  pal_dist down              -> vivid colors (0-255; higher = muddier)")
    print("  colors pred near GT        -> color diversity matches GT")
    print("  inter_IoU down             -> varied trees from the same QR")


if __name__ == "__main__":
    main()
