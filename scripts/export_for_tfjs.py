"""Export EMA weights for ALL trained versions to tf.js format (NDHWC), and
build a top-level manifest mirroring the HF ONNX manifest (themes, grids) so
the multi-version download/UI keeps working unchanged.

Outputs into docs/assets/:
  weights_v{N}.bin     concatenated float32, tf-ready layout
  manifest_v{N}.json   [{name, shape, offset, count}] + schedule + grid (per net)
  manifest.json        {trained_versions, versions{N:{weights,params,grid_xy,grid_z,bytes}}, themes}
"""
import sys, os, json, urllib.request
sys.path.insert(0, ".")
import numpy as np, torch
from qrbloom.diffusion import UNet3D, Diffusion, X0_CH, N_THEMES
from qrbloom.qr import grid_xy_for_version, grid_z_for_version

OUT = "docs/assets"; os.makedirs(OUT, exist_ok=True)
HF_MANIFEST = "https://huggingface.co/rocknroll17/QR-Bloom/resolve/main/onnx/manifest.json"

def export_version(V):
    ckpt = f"checkpoints/qrbloom_v{V}_best.pt"
    if not os.path.exists(ckpt):
        print(f"v{V}: no checkpoint, skip"); return None
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = UNet3D(ch=48, n_themes=N_THEMES).eval(); model.load_state_dict(ck["ema"])
    sd = model.state_dict()
    records = []; buf = bytearray()
    def add(name, arr):
        a = np.ascontiguousarray(arr.astype(np.float32))
        records.append({"name": name, "shape": list(a.shape),
                        "offset": len(buf)//4, "count": int(a.size)})
        buf.extend(a.tobytes())
    for name, t in sd.items():
        w = t.numpy()
        if w.ndim == 5:                       # conv3d & convT3d -> [kD,kH,kW, ., .]
            w = np.transpose(w, (2,3,4,1,0))
        elif w.ndim == 2 and ("tmlp" in name or "attr_mlp" in name or ".temb." in name):
            w = np.transpose(w, (1,0))        # Linear [out,in] -> [in,out]
        add(name, w)
    T = 500; diff = Diffusion(T=T, device="cpu")
    gxy = grid_xy_for_version(V); gz = grid_z_for_version(V)
    manifest = {"version": V, "ch": 48, "n_themes": N_THEMES, "x0_ch": X0_CH,
                "tdim": 256, "grid": [gxy, gxy, gz],
                "schedule": {"T": T, "acp": diff.acp.numpy().astype(np.float32).tolist()},
                "params": records}
    with open(f"{OUT}/weights_v{V}.bin", "wb") as f: f.write(buf)
    json.dump(manifest, open(f"{OUT}/manifest_v{V}.json", "w"))
    print(f"v{V}: {len(records)} params, {len(buf)/1e6:.1f}MB, grid=({gxy},{gxy},{gz})")
    return {"weights": f"weights_v{V}.bin", "params": f"manifest_v{V}.json",
            "grid_xy": gxy, "grid_z": gz, "bytes": len(buf)}

# themes from the HF manifest so the UI shows the exact same metadata
try:
    hf = json.load(urllib.request.urlopen(HF_MANIFEST, timeout=20))
    themes = hf["themes"]
except Exception as e:
    print("HF themes fetch failed, falling back to palette:", e)
    pal = json.load(open(f"{OUT}/palette.json"))
    names = ['cherryblossom','pine','socotra','maple','baobab','willow','magnolia','saguaro_cactus','palm','acacia']
    themes = [{"name": n, "label": n.replace('_',' ').title(), **pal[n]} for n in names]

versions = {}
for V in (2, 3, 4, 5):
    info = export_version(V)
    if info: versions[str(V)] = info

top = {"format_version": 1, "model_type": "qrbloom-unet3d-tfjs",
       "trained_versions": [int(v) for v in versions], "versions": versions, "themes": themes}
json.dump(top, open(f"{OUT}/manifest.json", "w"))
print("manifest.json:", top["trained_versions"])
