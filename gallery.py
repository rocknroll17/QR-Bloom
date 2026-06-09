"""gallery — interactive Three.js viewer for QR-Bloom voxel tree outputs.

The training script writes per-epoch epoch_***.json files containing
(occupancy + RGB) voxel data. This server serves them as an interactive 3D
gallery: an epoch slider, a theme/sample dropdown, and polling for new epochs.

Usage: python gallery.py [--port 8000]
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
    """Serve a template with no-store so the browser always gets the latest."""
    return HTMLResponse(
        content=(_TMPL_DIR / name).read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG = os.path.join(RUNS_DIR, "train.log")

# Model inference state — loaded lazily per version.
# Each per-version checkpoint (qrbloom_v{N}.pt) is loaded on first use and
# cached. Falls back to qrbloom.pt if a version-specific file isn't found.
_model_locks: dict = {}              # {version: Lock()} — per-version to allow parallel inference
_model_ctx_per_version: dict = {}    # {version: {model, diff, ep, device, mtime, ...}}


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


def _ckpt_path_for_version(version: int) -> str:
    """Per-version checkpoint path. Preference order:
    1. `qrbloom_v{N}_best.pt` — lowest validation loss so far (best quality)
    2. `qrbloom_v{N}.pt`      — latest epoch (live snapshot during training)
    3. `qrbloom_best.pt`      — legacy single-model best
    4. `qrbloom.pt`           — legacy single-model latest
    Set GALLERY_USE_LATEST=1 to skip best and use the live snapshot instead.
    """
    use_latest = bool(int(os.environ.get("GALLERY_USE_LATEST", "0")))
    if not use_latest:
        best_v = os.path.join(CKPT_DIR, f"qrbloom_v{version}_best.pt")
        if os.path.exists(best_v):
            return best_v
    per_v = os.path.join(CKPT_DIR, f"qrbloom_v{version}.pt")
    if os.path.exists(per_v):
        return per_v
    if not use_latest:
        legacy_best = os.path.join(CKPT_DIR, "qrbloom_best.pt")
        if os.path.exists(legacy_best):
            return legacy_best
    legacy = os.path.join(CKPT_DIR, "qrbloom.pt")
    return legacy


def _load_model(version: int | None = None):
    """Load (or hot-reload) the checkpoint for a specific QR version.

    Each version has its own model + checkpoint (qrbloom_v{N}.pt). The cached
    entry is reloaded if the file mtime changes (live training updates).

    If `version` is None, picks the smallest trained version available.
    """
    global _model_ctx_per_version
    import torch
    import numpy as np
    from qrbloom.diffusion import UNet3D, Diffusion
    from qrbloom.treegen import THEMES as VTHEMES
    from qrbloom.qr import THEME_NAMES as _THEMES

    if version is None:
        trained = _trained_versions() or [2]
        version = trained[0]
    version = int(version)

    ck_path = _ckpt_path_for_version(version)
    if not os.path.exists(ck_path):
        # ValueError (not FileNotFoundError) so the API exception handler can
        # safely surface the message without risking that some unrelated
        # FileNotFoundError from torch / disk I/O leaks a raw path.
        raise ValueError(
            f"No trained model is available for QR version {version}.")
    real_path = os.path.realpath(ck_path)
    mtime = os.path.getmtime(real_path)

    cached = _model_ctx_per_version.get(version)
    if cached is not None and cached.get("mtime") == mtime and cached.get("ck_path") == ck_path:
        return cached

    DEVICE = (cached["device"] if cached is not None else _pick_inference_device())
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=False)
    # Per-version model: head ModuleList has a single entry (this version).
    if cached is None:
        m = UNet3D(ch=48, n_themes=len(_THEMES),
                    versions=(version,)).to(DEVICE)
        diff = Diffusion(T=500, device=DEVICE)
    else:
        m = cached["model"]
        diff = cached["diff"]
    try:
        m.load_state_dict(ck["ema"])
    except RuntimeError:
        # Legacy multi-version checkpoint: filter keys for this single-version model.
        sd = {k: v for k, v in ck["ema"].items() if k in m.state_dict()}
        m.load_state_dict(sd, strict=False)
    m.eval()
    ctx = {
        "model": m, "diff": diff, "ep": ck["epoch"], "device": DEVICE, "mtime": mtime,
        "themes": _THEMES, "vthemes": VTHEMES, "np": np, "torch": torch,
        "version": version, "ck_path": ck_path,
    }
    _model_ctx_per_version[version] = ctx
    print(f"[model] v{version} loaded ep{ck['epoch']} on {DEVICE} ({ck_path})", flush=True)
    return ctx


def gt_generate(url: str, theme_name: str, version: int | None = None):
    """Encode a URL and run procedural treegen.

    `version` defaults to None ⇒ the smallest trained version that fits the
    input (segno's natural pick, bumped up to our min trained version).
    Pass an explicit integer only for debugging.
    """
    from qrbloom.treegen import generate_voxels, THEMES as VTHEMES
    text = url if url else "QR-Bloom"
    if version is None:
        qr, version = pick_version_for_text(text)
    else:
        import segno
        try:
            qr = segno.make(text, error="m", version=version)
        except Exception as e:
            raise ValueError("Text doesn't fit in the selected QR version.") from e
    core = [[bool(c) for c in row] for row in qr.matrix]
    if theme_name not in VTHEMES:
        theme_name = "cherryblossom"
    voxels = generate_voxels(core, theme=theme_name)
    cells = []
    for v in voxels:
        x, y, z = v["pos"]
        cells.append([int(x), int(y), int(z), float(v["scale"]), v["color"]])
    return {"cells": cells, "count": len(cells), "theme": theme_name,
            "url": text, "version": version}


def model_generate(url: str, theme_name: str, steps: int = 100,
                   version: int | None = None):
    """Run model inference for a URL.

    Version is picked automatically by segno (smallest standard QR that
    fits the text), then bumped up to the model's trained range. Pass an
    explicit `version=N` only to override for debugging.
    """
    from qrbloom.qr import qr_modules, grid_xy_for_version, grid_z_for_version
    text = url if url else "QR-Bloom"
    if version is None:
        qr, version = pick_version_for_text(text)
    else:
        import segno
        try:
            qr = segno.make(text, error="m", version=version)
        except Exception as e:
            raise ValueError("Text doesn't fit in the selected QR version.") from e
    # Load the model for this specific version (each version is a separate file).
    ctx = _load_model(version)
    np = ctx["np"]; torch = ctx["torch"]
    core = np.array([[1 if c else 0 for c in row] for row in qr.matrix],
                    dtype=np.uint8)
    qe = qr_modules(version)
    gxy = grid_xy_for_version(version)
    gz  = grid_z_for_version(version)
    off = (gxy - qe) // 2
    ctr = qe // 2

    pad = np.zeros((gxy, gxy), dtype=np.uint8)
    pad[off:off + qe, off:off + qe] = core
    if theme_name not in ctx["themes"]:
        theme_name = "cherryblossom"
    theme_idx = ctx["themes"].index(theme_name)
    dev = ctx["device"]
    qr_t = torch.from_numpy(pad).unsqueeze(0).to(dev)
    cond = qr_t.float().unsqueeze(1).unsqueeze(-1).expand(
        -1, 1, gxy, gxy, gz).contiguous()
    th_t = torch.tensor([theme_idx], device=dev).long()
    attr = torch.full((1, 3), 0.5, device=dev)
    ver_t = torch.tensor([int(version)], device=dev, dtype=torch.long)
    with _model_locks.setdefault(version, threading.Lock()), torch.no_grad():
        x0 = ctx["diff"].sample(ctx["model"], cond, th_t, attr=attr, steps=steps,
                                device=dev, eta=1.0, version=ver_t).cpu().numpy()[0]
    occ = (x0[3] > 0)
    rgb = x0[:3]
    th_data = ctx["vthemes"][theme_name]
    cells = []
    for i in range(qe):
        for j in range(qe):
            col = th_data["qr_dark"] if bool(core[i, j]) else th_data["qr_light"]
            cells.append([int(j - ctr), 0, int(i - ctr), 1.0, col])
    rows, cols, heights = np.where(occ)
    for i, j, k in zip(rows, cols, heights):
        if k < 1:
            continue
        r = int(np.clip((rgb[0, i, j, k] + 1) * 127.5, 0, 255))
        g = int(np.clip((rgb[1, i, j, k] + 1) * 127.5, 0, 255))
        b = int(np.clip((rgb[2, i, j, k] + 1) * 127.5, 0, 255))
        color = f"#{r:02x}{g:02x}{b:02x}"
        x = int(j - (off + ctr))
        z = int(i - (off + ctr))
        cells.append([x, int(k), z, 1.0, color])
    return {"cells": cells, "count": len(cells), "theme": theme_name,
            "url": url, "ckpt_ep": ctx["ep"], "version": version,
            "grid_xy": gxy, "grid_z": gz}


def _trained_versions() -> list[int]:
    """Detect which QR versions have a trained model available.

    Preference order:
    1. Per-version checkpoints: scan checkpoints/qrbloom_v{N}.pt for N in 1..40.
       This matches the split-model training (one model per version).
    2. Legacy single-checkpoint runs/unseen_gt.json (multi-version MoE).
    """
    # 1) Per-version split-model checkpoints — recognise either the live
    #    snapshot (qrbloom_v{N}.pt) OR the best-val file (qrbloom_v{N}_best.pt).
    #    A deployment may carry only the *_best.pt files (smaller bundle).
    per_v = []
    for v in range(1, 41):
        live = os.path.join(CKPT_DIR, f"qrbloom_v{v}.pt")
        best = os.path.join(CKPT_DIR, f"qrbloom_v{v}_best.pt")
        if os.path.exists(live) or os.path.exists(best):
            per_v.append(v)
    if per_v:
        return per_v
    # 2) Legacy: read unseen_gt.json
    gt_path = os.path.join(RUNS_DIR, "unseen_gt.json")
    if not os.path.exists(gt_path):
        return []
    try:
        with open(gt_path) as f:
            d = json.load(f)
        if isinstance(d.get("trained_versions"), list):
            return sorted(set(int(v) for v in d["trained_versions"]))
        if "version" in d:
            return [int(d["version"])]
    except Exception:
        pass
    return []


def pick_version_for_text(text: str) -> tuple["object", int]:
    """Pick the smallest *trained* QR version that fits `text` and encode it.

    Uses segno's auto-detect (with micro-QR disabled — we only deal with
    standard QR codes). If the natural version is below the trained range,
    we bump up to the smallest trained version (re-encoding at that
    version). If it's above the trained range, raise ValueError.

    Returns: (segno QRCode object, version int)
    """
    import segno
    trained = _trained_versions()
    if not trained:
        # No trained set known — fall back to natural picking
        qr = segno.make(text or "QR-Bloom", error="m", micro=False)
        return qr, qr.version
    tmin, tmax = min(trained), max(trained)
    # segno picks smallest fitting standard version
    natural_qr = segno.make(text or "QR-Bloom", error="m", micro=False)
    natural_v = natural_qr.version
    if natural_v > tmax:
        raise ValueError("Text is too long. Please shorten the input.")
    chosen = max(natural_v, tmin)
    if chosen == natural_v:
        return natural_qr, chosen
    return segno.make(text or "QR-Bloom", error="m", version=chosen), chosen


def list_qr_versions():
    """Return basic info about which versions are trained."""
    trained = _trained_versions()
    return {"versions": list(range(1, 11)),
            "trained": trained,
            "default": (trained[0] if trained else 1)}





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


def _safe_path_in(base: str, name: str) -> str | None:
    """Resolve `base/name` and return it only if it stays inside `base`.

    Belt-and-suspenders containment check — the callers already validate
    `name` against a strict pattern before calling, but resolving with
    realpath and confirming the result is rooted at `base` makes path
    injection structurally impossible.
    """
    base_real = os.path.realpath(base)
    full_real = os.path.realpath(os.path.join(base_real, name))
    try:
        if os.path.commonpath([base_real, full_real]) == base_real:
            return full_real
    except ValueError:
        # Different drives on Windows, or otherwise incommensurable paths.
        pass
    return None

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


# Shared 3D viewer — the same file is also published by GitHub Pages, so the
# gallery and the static demo run identical rendering logic without drift.
@app.get("/qrbloom-viewer.js")
def route_viewer_js():
    content = (Path(ROOT) / "docs" / "qrbloom-viewer.js").read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/state")
def route_state():
    return JSONResponse(content=get_state(), headers={"Cache-Control": "no-store"})


# Image and JSON routes reconstruct the requested filename from a regex match
# group rather than passing user input through; the validated filename is
# rebuilt from literal strings + an integer parse so no user-controlled byte
# can survive into the path expression.

@app.get("/img/{name}")
def route_img(name: str):
    if name == "loss.png":
        clean = "loss.png"
    elif (m := re.fullmatch(r'epoch_(\d+)\.png', name)):
        # zfill(3) matches train.py's epoch_{N:03d} naming so the lookup hits.
        clean = f"epoch_{int(m.group(1)):03d}.png"
    else:
        return Response(content=b"not allowed", status_code=404, media_type="text/plain")
    full = _safe_path_in(RUNS_DIR, clean) or _safe_path_in(ROOT, clean)
    if not full or not os.path.exists(full):
        return Response(content=b"not found", status_code=404, media_type="text/plain")
    with open(full, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/data/{name}")
def route_data(name: str):
    if name == "unseen_gt.json":
        clean = "unseen_gt.json"
    elif (m := re.fullmatch(r'ep(\d+)\.json', name)):
        # zfill(3) matches train.py's epoch_{N:03d} naming so the lookup hits.
        clean = f"epoch_{int(m.group(1)):03d}.json"
    else:
        return Response(content=b"not allowed", status_code=404, media_type="text/plain")
    full = _safe_path_in(RUNS_DIR, clean) or _safe_path_in(ROOT, clean)
    if not full or not os.path.exists(full):
        return Response(content=b"not found", status_code=404, media_type="text/plain")
    with open(full, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


class _GenerateBody(BaseModel):
    url: str = ""
    theme: str = "cherryblossom"
    # None => auto-pick from text; explicit integer overrides for debugging.
    version: int | None = None


@app.get("/api/themes")
def route_api_themes():
    """Theme list (name, label, QR colors), sourced from the generator so it never drifts."""
    from qrbloom.treegen import THEMES, LABELS
    return JSONResponse(content=[
        {"name": k, "label": LABELS.get(k, k),
         "dark": v["qr_dark"], "light": v["qr_light"]}
        for k, v in THEMES.items()
    ])


@app.get("/api/qr-versions")
def route_api_qr_versions():
    """List of QR versions usable in the UI, with the trained subset flagged."""
    return JSONResponse(content=list_qr_versions())


@app.post("/api/generate")
def route_api_generate(body: _GenerateBody):
    url = (body.url or "").strip()
    theme = body.theme or "cherryblossom"
    try:
        result = model_generate(url, theme, steps=100, version=body.version)
        return JSONResponse(content=result)
    except ValueError as e:
        # ValueError is reserved for deliberate user-facing messages (text
        # too long, no model for that version, etc.). Other exception types
        # are treated as internal and their messages are not surfaced.
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception:
        # Unexpected failure (CUDA OOM, malformed checkpoint, etc.) — hide
        # the raw traceback from the frontend and log it server-side.
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": "Generation failed. Please try again."},
            status_code=500)






# Production demo video page.
# URL query params: ?start=1&end=80&sample=5&hold=20&xfade=8
# (hold/xfade units: frames at 30fps)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=8000, type=int)
    args = ap.parse_args()
    print(f"gallery → http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
