# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""Export the multi-version EMA checkpoint to tf.js format.

One DiT3D checkpoint (checkpoints/qrbloom_all_best.pt) serves every trained
QR version — the version enters the network as micro-conditioning, so the
browser downloads a single weights file and passes the version at runtime.

Weight layout is matmul-ready for docs/qrbloom-model.js:
  * patch_embed.weight [dim, cin, p, p, p] → [cin·p³, dim]  (token vectors
    are gathered channel-major, patch offsets row-major)
  * Linear weights [out, in] → [in, out]
  * theme_emb.weight stays [n_themes, dim] (row gather, not matmul)

Outputs into docs/assets/:
  weights_all.bin      concatenated float32
  manifest_all.json    params index + dit hyperparams + per-version grids
  manifest.json        {weights, sha256, trained_versions, versions, themes}
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
CKPT = os.path.join(os.environ.get("QRBLOOM_CKPT_DIR", "checkpoints"),
                    "qrbloom_all_best.pt")
VERSIONS = [int(v) for v in
            os.environ.get("QRBLOOM_EXPORT_VERSIONS", "2,3,4,5").split(",")]

DIM = int(os.environ.get("DIT_DIM", "384"))
DEPTH = int(os.environ.get("DIT_DEPTH", "12"))
HEADS = int(os.environ.get("DIT_HEADS", "6"))
PATCH = int(os.environ.get("DIT_PATCH", "4"))

if not os.path.exists(CKPT):
    print(f"no checkpoint at {CKPT}, nothing to export")
    sys.exit(0)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
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
grids = {str(v): [grid_xy_for_version(v), grid_xy_for_version(v),
                  grid_z_for_version(v)] for v in VERSIONS}
manifest = {"model": "dit3d",
            "dim": DIM, "depth": DEPTH, "heads": HEADS, "patch": PATCH,
            "n_themes": N_THEMES, "x0_ch": X0_CH,
            "grids": grids,
            "schedule": {"T": T,
                         "acp": diff.acp.numpy().astype(np.float32).tolist()},
            "params": records}
with open(f"{OUT}/weights_all.bin", "wb") as f:
    f.write(buf)
with open(f"{OUT}/manifest_all.json", "w") as f:
    json.dump(manifest, f)

themes = [{"name": k, "label": LABELS.get(k, k),
           "qr_dark": v["qr_dark"], "qr_light": v["qr_light"],
           "trunk": v["trunk"], "leaf": v["leaf"],
           "flower": v.get("flower", [])}
          for k, v in THEMES.items()]

top = {"format_version": 3, "model_type": "qrbloom-dit3d-tfjs",
       "weights": "weights_all.bin", "params": "manifest_all.json",
       "bytes": len(buf), "sha256": hashlib.sha256(bytes(buf)).hexdigest(),
       "trained_versions": VERSIONS,
       "versions": {str(v): {"grid_xy": grids[str(v)][0],
                             "grid_z": grids[str(v)][2]} for v in VERSIONS},
       "themes": themes}
with open(f"{OUT}/manifest.json", "w") as f:
    json.dump(top, f)
print(f"exported {len(records)} params, {len(buf)/1e6:.1f}MB, "
      f"versions={VERSIONS}, epoch={ck.get('epoch', '?')}")
