"""Generate static voxel samples for the GitHub Pages demo.

Produces one JSON file per theme — each containing a deterministic procedural
voxel tree on top of a real QR code. No trained model is needed; the
procedural generator (qrbloom.treegen) is what the diffusion model is trained
to reproduce, so these samples are faithful to what users see in the gallery.

Outputs:
    docs/samples/<theme>.json        — { cells, label, text, version, qr_modules }
    docs/samples/manifest.json       — { themes: [{ name, label, file }, ...] }

Designed to run in CI (CPU, no GPU, no checkpoints).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import segno

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from qrbloom.qr import grid_xy_for_version, pad_to_grid  # noqa: E402
from qrbloom.treegen import THEMES, generate_voxels  # noqa: E402
OUT_DIR = REPO_ROOT / "docs" / "samples"

# Human-readable theme labels.
THEME_LABELS = {
    "cherryblossom": "Cherry blossom",
    "pine": "Pine",
    "socotra": "Dragon tree (Socotra)",
    "maple": "Maple",
    "baobab": "Baobab",
    "willow": "Willow",
    "magnolia": "Magnolia",
    "saguaro_cactus": "Saguaro cactus",
    "palm": "Palm",
    "acacia": "Acacia",
}

# One QR text per theme so the demo has variety. Kept short enough that segno
# fits each into QR version 2 (35 alphanumeric chars at error level M).
THEME_TEXTS = {
    "cherryblossom": "HANAMI",
    "pine": "EVERGREEN",
    "socotra": "DRAGONS BLOOD",
    "maple": "SUGAR-MAPLE-42",
    "baobab": "BAOBAB AVENUE",
    "willow": "WEEPING WILLOW",
    "magnolia": "MAGNOLIA-2024",
    "saguaro_cactus": "SONORAN DESERT",
    "palm": "TROPIC OF CANCER",
    "acacia": "SAVANNA-001",
}

# Deterministic seed per theme so the demo is reproducible across CI runs.
SEED_BASE = 20260101


def encode_qr(text: str, version: int = 2) -> np.ndarray:
    """Encode text as a QR matrix and pad to the model's grid."""
    qr = segno.make(text, error="m", version=version)
    core = np.array(qr.matrix, dtype=np.uint8)
    full, _ = pad_to_grid(core, grid_xy=grid_xy_for_version(version))
    return full


def voxels_to_cells(voxels: list[dict]) -> list[list]:
    """Convert qrbloom voxel dicts to the [x, y, z, scale, color] cell format."""
    cells = []
    for v in voxels:
        x, y, z = v["pos"]
        cells.append([int(x), int(y), int(z), float(v["scale"]), v["color"]])
    return cells


def build_sample(theme: str, text: str, version: int, seed: int) -> dict:
    qr = encode_qr(text, version=version)
    rng = random.Random(seed)
    voxels = generate_voxels(qr, theme=theme, rng=rng)
    return {
        "label": THEME_LABELS[theme],
        "theme": theme,
        "text": text,
        "version": version,
        "qr_modules": int(qr.shape[0]),
        "cells": voxels_to_cells(voxels),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_themes = []
    for i, theme in enumerate(THEMES.keys()):
        text = THEME_TEXTS.get(theme, theme.upper())
        sample = build_sample(theme, text, version=2, seed=SEED_BASE + i)
        out_path = OUT_DIR / f"{theme}.json"
        out_path.write_text(json.dumps(sample, separators=(",", ":")))
        manifest_themes.append({
            "name": theme,
            "label": THEME_LABELS[theme],
            "file": f"samples/{theme}.json",
        })
        print(f"  wrote {out_path.relative_to(REPO_ROOT)} ({len(sample['cells'])} cells)")

    manifest = {"themes": manifest_themes}
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {(OUT_DIR / 'manifest.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
