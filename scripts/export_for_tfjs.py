# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""Export EMA weights for all trained versions to tf.js format, plus a
top-level manifest (trained versions, per-version grids, themes) consumed by
docs/qrbloom-inference.js.

Weight layout is matmul-ready for the browser DiT (docs/qrbloom-model.js):
  * patch_embed.weight [dim, cin, p, p, p] → [cin·p³, dim]  (token vectors
    are gathered channel-major, patch offsets row-major)
  * Linear weights [out, in] → [in, out]
  * theme_emb.weight stays [n_themes, dim] (row gather, not matmul)

Outputs into docs/assets/:
  weights_v{N}.bin     concatenated float32
  manifest_v{N}.json   params index + dit hyperparams + schedule + grid
  manifest.json        {trained_versions, versions{...}, themes}
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np
import torch

from qrbloom.diffusion import Diffusion, X0_CH, N_THEMES
from qrbloom.dit import DiT3D
from qrbloom.qr import grid_xy_for_version, grid_z_for_version
from qrbloom.treegen import LABELS, THEMES

OUT = "docs/assets"
os.makedirs(OUT, exist_ok=True)
CKPT_DIR = os.environ.get("QRBLOOM_CKPT_DIR", "checkpoints")

DIM = int(os.environ.get("DIT_DIM", "384"))
DEPTH = int(os.environ.get("DIT_DEPTH", "12"))
HEADS = int(os.environ.get("DIT_HEADS", "6"))
PATCH = int(os.environ.get("DIT_PATCH", "4"))


def export_version(V):
    ckpt = os.path.join(CKPT_DIR, f"qrbloom_v{V}_best.pt")
    if not os.path.exists(ckpt):
        print(f"v{V}: no checkpoint, skip")
        return None
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = DiT3D(dim=DIM, depth=DEPTH, heads=HEADS, patch=PATCH,
                  n_themes=N_THEMES).eval()
    model.load_state_dict(ck["ema"])
    sd = model.state_dict()

    records = []
    buf = bytearray()

    def add(name, arr):
        a = np.ascontiguousarray(arr.astype(np.float32))
        records.append({"name": name, "shape": list(a.shape),
                        "offset": len(buf) // 4, "count": int(a.size)})
        buf.extend(a.tobytes())

    for name, t in sd.items():
        w = t.numpy()
        if name == "patch_embed.weight":
            w = w.reshape(w.shape[0], -1).T          # [cin·p³, dim]
        elif w.ndim == 2 and name != "theme_emb.weight":
            w = w.T                                   # Linear [out,in] → [in,out]
        add(name, w)

    T = 500
    diff = Diffusion(T=T, device="cpu")
    gxy, gz = grid_xy_for_version(V), grid_z_for_version(V)
    manifest = {"version": V, "model": "dit3d",
                "dim": DIM, "depth": DEPTH, "heads": HEADS, "patch": PATCH,
                "n_themes": N_THEMES, "x0_ch": X0_CH,
                "grid": [gxy, gxy, gz],
                "schedule": {"T": T,
                             "acp": diff.acp.numpy().astype(np.float32).tolist()},
                "params": records}
    with open(f"{OUT}/weights_v{V}.bin", "wb") as f:
        f.write(buf)
    with open(f"{OUT}/manifest_v{V}.json", "w") as f:
        json.dump(manifest, f)
    print(f"v{V}: {len(records)} params, {len(buf)/1e6:.1f}MB, "
          f"grid=({gxy},{gxy},{gz}), epoch={ck.get('epoch', '?')}")
    return {"weights": f"weights_v{V}.bin", "params": f"manifest_v{V}.json",
            "grid_xy": gxy, "grid_z": gz, "bytes": len(buf),
            "sha256": hashlib.sha256(bytes(buf)).hexdigest()}


themes = [{"name": k, "label": LABELS.get(k, k),
           "dark": v["qr_dark"], "light": v["qr_light"]}
          for k, v in THEMES.items()]

versions = {}
for V in (2, 3, 4, 5):
    info = export_version(V)
    if info:
        versions[str(V)] = info

top = {"format_version": 2, "model_type": "qrbloom-dit3d-tfjs",
       "trained_versions": [int(v) for v in versions], "versions": versions,
       "themes": themes}
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(top, f)
print("manifest.json:", top["trained_versions"])
