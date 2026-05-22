"""gallery — interactive Three.js viewer for QR-Bloom voxel tree outputs.

The training script writes per-epoch epoch_***.json files containing
(occupancy + RGB) voxel data. This server serves them as an interactive 3D
gallery: an epoch slider, a theme/sample dropdown, and polling for new epochs.

Usage: python gallery.py [--port 8001]
"""
import argparse
import json
import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel
import uvicorn

_TMPL_DIR = Path(__file__).resolve().parent / "templates"


def _serve_template(name: str) -> HTMLResponse:
    return HTMLResponse(content=(_TMPL_DIR / name).read_text(encoding="utf-8"))

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG = os.path.join(RUNS_DIR, "train.log")

# Model inference state — loaded lazily, reloaded when the checkpoint changes.
_model_lock = threading.Lock()
_model_ctx = None    # {model, diff, themes, off}


def _pick_inference_device():
    """Pick the CUDA device with the most free VRAM, or fall back to CPU.
    Override with the GALLERY_DEVICE environment variable.
    """
    forced = os.environ.get("GALLERY_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        candidates = []
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free > 2 * 1024 ** 3:   # require at least 2 GB headroom
                candidates.append((free, i))
        if candidates:
            candidates.sort(reverse=True)
            return f"cuda:{candidates[0][1]}"
    except Exception:
        pass
    return "cpu"


def _load_model():
    """Load (or hot-reload) the checkpoint from checkpoints/qrbloom.pt.

    The model is cached in _model_ctx and reloaded automatically whenever the
    checkpoint file changes on disk, so the gallery stays current during training.
    """
    global _model_ctx
    import torch
    import numpy as np
    from qrbloom.diffusion import UNet3D, Diffusion
    from qrbloom.treegen import THEMES as VTHEMES
    from qrbloom.qr import THEME_NAMES as _THEMES, QR_OFFSET as OFF

    ck_path = os.path.join(CKPT_DIR, "qrbloom.pt")
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    real_path = os.path.realpath(ck_path)   # resolve symlinks for mtime
    mtime = os.path.getmtime(real_path)
    if _model_ctx is not None and _model_ctx.get("mtime") == mtime:
        return _model_ctx
    DEVICE = _model_ctx["device"] if _model_ctx is not None else _pick_inference_device()
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    if _model_ctx is None:
        m = UNet3D(ch=48).to(DEVICE)
        diff = Diffusion(T=500, device=DEVICE)
    else:
        m = _model_ctx["model"]   # reuse existing model object; only reload weights
        diff = _model_ctx["diff"]
    m.load_state_dict(ck["ema"])
    m.eval()
    _model_ctx = {
        "model": m, "diff": diff, "ep": ck["epoch"], "device": DEVICE, "mtime": mtime,
        "themes": _THEMES, "vthemes": VTHEMES, "off": OFF, "np": np, "torch": torch,
    }
    print(f"[model] loaded ep{ck['epoch']} on {DEVICE}", flush=True)
    return _model_ctx


def gt_generate(url: str, theme_name: str):
    """Encode a URL into a QR code, generate GT voxels via treegen, and return cells."""
    import segno
    from qrbloom.treegen import generate_voxels, THEMES as VTHEMES
    text = url if url else "QR-Bloom"
    qr = segno.make_qr(text, error="m", version=1)
    core = [[bool(c) for c in row] for row in qr.matrix]
    if theme_name not in VTHEMES:
        theme_name = "cherryblossom"
    voxels = generate_voxels(core, theme=theme_name)
    cells = []
    for v in voxels:
        x, y, z = v["pos"]
        cells.append([int(x), int(y), int(z), float(v["scale"]), v["color"]])
    return {"cells": cells, "count": len(cells), "theme": theme_name, "url": text}


def model_generate(url: str, theme_name: str, steps: int = 100):
    """Run model inference for a URL and return voxel cells for the viewer."""
    import segno
    ctx = _load_model()
    np = ctx["np"]; torch = ctx["torch"]
    text = url if url else "QR-Bloom"
    qr = segno.make_qr(text, error="m", version=1)
    core = np.array(qr.matrix, dtype=np.uint8)
    pad = np.zeros((32, 32), dtype=np.uint8)
    OFF = ctx["off"]
    pad[OFF:OFF + 21, OFF:OFF + 21] = core
    if theme_name not in ctx["themes"]:
        theme_name = "cherryblossom"
    theme_idx = ctx["themes"].index(theme_name)
    dev = ctx["device"]
    qr_t = torch.from_numpy(pad).unsqueeze(0).to(dev)
    cond = qr_t.float().unsqueeze(1).unsqueeze(-1).expand(-1, 1, 32, 32, 32).contiguous()
    th_t = torch.tensor([theme_idx], device=dev).long()
    attr = torch.full((1, 3), 0.5, device=dev)   # mid-range shape attributes
    with _model_lock, torch.no_grad():
        x0 = ctx["diff"].sample(ctx["model"], cond, th_t, attr=attr, steps=steps,
                                device=dev, eta=1.0).cpu().numpy()[0]
    occ = (x0[3] > 0)
    rgb = x0[:3]
    th_data = ctx["vthemes"][theme_name]
    cells = []
    for i in range(21):
        for j in range(21):
            col = th_data["qr_dark"] if bool(core[i, j]) else th_data["qr_light"]
            cells.append([int(j - 10), 0, int(i - 10), 1.0, col])
    rows, cols, heights = np.where(occ)
    for i, j, k in zip(rows, cols, heights):
        if k < 1:
            continue
        r = int(np.clip((rgb[0, i, j, k] + 1) * 127.5, 0, 255))
        g = int(np.clip((rgb[1, i, j, k] + 1) * 127.5, 0, 255))
        b = int(np.clip((rgb[2, i, j, k] + 1) * 127.5, 0, 255))
        color = f"#{r:02x}{g:02x}{b:02x}"
        x = int(j - (OFF + 10))
        z = int(i - (OFF + 10))
        cells.append([x, int(k), z, 1.0, color])
    return {"cells": cells, "count": len(cells), "theme": theme_name,
            "url": url, "ckpt_ep": ctx["ep"]}





def get_state():
    rgx_png  = re.compile(r'epoch_(\d+)\.png$')
    rgx_json = re.compile(r'epoch_(\d+)\.json$')
    seen_eps: set = set()
    seen_json: set = set()
    scan_dir = RUNS_DIR if os.path.isdir(RUNS_DIR) else ROOT
    for f in os.listdir(scan_dir):
        m = rgx_png.match(f)
        if m:
            seen_eps.add(int(m.group(1)))
        m2 = rgx_json.match(f)
        if m2:
            seen_json.add(int(m2.group(1)))
    eps = sorted(seen_eps)
    json_set = sorted(seen_json)

    state = {"epochs": eps, "json_set": json_set, "losses": [], "target_epochs": 80}

    if os.path.exists(LOG):
        try:
            with open(LOG) as f:
                txt = f.read()
            for m in re.finditer(
                    r'ep(\d+) loss=([\d.]+) occ=([\d.]+) rgb=([\d.]+)', txt):
                e, t, o, r = m.groups()
                state["losses"].append({
                    "ep": int(e), "total": float(t),
                    "occ": float(o), "rgb": float(r),
                })
        except Exception:
            pass

    try:
        import torch
        ck_pair = [
            (os.path.join(CKPT_DIR, "qrbloom.pt"), "latest_ep"),
            (os.path.join(CKPT_DIR, "qrbloom_best.pt"), "best_ep"),
        ]
        for path, key in ck_pair:
            if os.path.exists(path):
                ck = torch.load(path, map_location="cpu", weights_only=False)
                state[key] = ck["epoch"]
    except Exception as e:
        state["ckpt_err"] = str(e)

    return state


_PNG_SAFE = re.compile(r'^(loss|epoch_\d+)\.png$')
_JSON_SAFE = re.compile(r'^(ep\d+|unseen_gt)\.json$')
_ASSET_SAFE = re.compile(r'^[\w-]+\.hdr$')

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def page_index():
    return _serve_template("index.html")


@app.get("/index.html", response_class=HTMLResponse)
def page_index_html():
    return _serve_template("index.html")


@app.get("/embed-template", response_class=PlainTextResponse)
def route_embed_template():
    content = (_TMPL_DIR / "embed.html").read_text(encoding="utf-8")
    return PlainTextResponse(content=content, headers={"Cache-Control": "no-store"})


@app.get("/state")
def route_state():
    return JSONResponse(content=get_state(), headers={"Cache-Control": "no-store"})


@app.get("/img/{name}")
def route_img(name: str):
    if not _PNG_SAFE.match(name):
        return Response(content=b"not allowed", status_code=404, media_type="text/plain")
    full = os.path.join(RUNS_DIR, name)
    if not os.path.exists(full):
        full = os.path.join(ROOT, name)
    if not os.path.exists(full):
        return Response(content=b"not found", status_code=404, media_type="text/plain")
    with open(full, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/data/{name}")
def route_data(name: str):
    if not _JSON_SAFE.match(name):
        return Response(content=b"not allowed", status_code=404, media_type="text/plain")
    fname = re.sub(r'^ep(\d+)\.json$', r'epoch_\1.json', name)
    full = os.path.join(RUNS_DIR, fname)
    if not os.path.exists(full):
        full = os.path.join(ROOT, fname)
    if not os.path.exists(full):
        return Response(content=b"not found", status_code=404, media_type="text/plain")
    with open(full, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


@app.get("/assets/{name}")
def route_assets(name: str):
    if not _ASSET_SAFE.match(name):
        return Response(content=b"not allowed", status_code=404, media_type="text/plain")
    full = os.path.join(ROOT, "assets", name)
    if not os.path.exists(full):
        return Response(content=b"not found", status_code=404, media_type="text/plain")
    with open(full, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/vnd.radiance",
                    headers={"Cache-Control": "max-age=86400"})


class _GenerateBody(BaseModel):
    url: str = ""
    theme: str = "cherryblossom"


@app.get("/api/themes")
def route_api_themes():
    """Theme list (name, label, QR colors), sourced from the generator so it never drifts."""
    from qrbloom.treegen import THEMES, LABELS
    return JSONResponse(content=[
        {"name": k, "label": LABELS.get(k, k),
         "dark": v["qr_dark"], "light": v["qr_light"]}
        for k, v in THEMES.items()
    ])


@app.post("/api/generate")
def route_api_generate(body: _GenerateBody):
    url = (body.url or "").strip()
    theme = body.theme or "cherryblossom"
    try:
        result = model_generate(url, theme, steps=100)
        return JSONResponse(content=result)
    except Exception as e:
        msg = str(e)
        if "does not fit" in msg.lower() or 'version "1"' in msg.lower():
            msg = ("Text is too long for a version-1 QR code. "
                   "Keep it short — about 14 characters or fewer.")
        return JSONResponse(content={"error": msg})




# Production demo video page.
# URL query params: ?start=1&end=80&sample=5&hold=20&xfade=8
# (hold/xfade units: frames at 30fps)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=8001, type=int)
    args = ap.parse_args()
    print(f"gallery → http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
