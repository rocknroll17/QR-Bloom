# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""gallery — interactive Three.js viewer for QR-Bloom voxel tree outputs.

The training script writes per-epoch epoch_***.json files containing
(occupancy + RGB) voxel data. This server serves them as an interactive 3D
gallery, and runs live model inference through ModelService.

Usage: python gallery.py [--port 8000]
"""
import argparse
import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel
import uvicorn

_TMPL_DIR = Path(__file__).resolve().parent / "templates"

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
LOG = os.path.join(RUNS_DIR, "train.log")


class ModelService:
    """Owns live inference: device selection, lazy checkpoint loading with
    mtime-based hot-reload (live training updates), the trained-version
    policy, and URL → viewer-cell-list generation.

    One DiT3D checkpoint serves every trained QR version — the version
    enters the net as conditioning. torch is imported lazily so serving the
    static pages never pays the import cost.
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        self._lock = threading.Lock()
        self._model = None
        self._diff = None
        self._device: str | None = None
        self._mtime: float | None = None
        self._ck_path: str | None = None
        self.epoch: int | None = None

    # ── checkpoint / device policy ────────────────────────────────────────
    def ckpt_path(self) -> str:
        """Prefers the best-validation file; set GALLERY_USE_LATEST=1 to use
        the live training snapshot instead."""
        use_latest = bool(int(os.environ.get("GALLERY_USE_LATEST", "0")))
        if not use_latest:
            best = os.path.join(self.ckpt_dir, "qrbloom_all_best.pt")
            if os.path.exists(best):
                return best
        return os.path.join(self.ckpt_dir, "qrbloom_all.pt")

    @staticmethod
    def _pick_device() -> str:
        """CUDA device with the most free VRAM, or CPU. Override with the
        GALLERY_DEVICE environment variable."""
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

    def trained_versions(self) -> list[int]:
        """QR versions the model was trained on. The single checkpoint covers
        all of them; the set itself comes from the QRBLOOM_VERSIONS env
        (default 2..5, matching the training default)."""
        if not os.path.exists(self.ckpt_path()):
            return []
        raw = os.environ.get("QRBLOOM_VERSIONS", "2,3,4,5")
        return sorted({int(v) for v in raw.split(",") if v.strip()})

    def pick_version_for_text(self, text: str):
        """Pick the smallest *trained* QR version that fits `text` and encode it.

        Uses segno's auto-detect (micro-QR disabled — standard codes only).
        A natural version below the trained range is bumped up to the
        smallest trained version (re-encoding); above the range raises.

        Returns: (segno QRCode object, version int)
        """
        import segno
        trained = self.trained_versions()
        if not trained:
            # No trained set known — fall back to natural picking.
            qr = segno.make(text or "QR-Bloom", error="m", micro=False)
            return qr, qr.version
        tmin, tmax = min(trained), max(trained)
        natural_qr = segno.make(text or "QR-Bloom", error="m", micro=False)
        natural_v = natural_qr.version
        if natural_v > tmax:
            raise ValueError("Text is too long. Please shorten the input.")
        chosen = max(natural_v, tmin)
        if chosen == natural_v:
            return natural_qr, chosen
        return segno.make(text or "QR-Bloom", error="m", version=chosen), chosen

    # ── model lifecycle ───────────────────────────────────────────────────
    def _ensure_loaded(self):
        """Load the checkpoint, or hot-reload it when the file changed."""
        import torch
        from qrbloom.model import DiT3D, Diffusion
        from qrbloom.qr import THEME_NAMES

        ck_path = self.ckpt_path()
        if not os.path.exists(ck_path):
            # ValueError (not FileNotFoundError) so the API exception handler
            # can surface the message without risking that some unrelated
            # FileNotFoundError from torch / disk I/O leaks a raw path.
            raise ValueError("No trained model is available.")
        mtime = os.path.getmtime(os.path.realpath(ck_path))
        if self._model is not None and self._mtime == mtime and self._ck_path == ck_path:
            return

        if self._device is None:
            self._device = self._pick_device()
        ck = torch.load(ck_path, map_location=self._device, weights_only=False)
        if self._model is None:
            self._model = DiT3D(n_themes=len(THEME_NAMES)).to(self._device)
            self._diff = Diffusion(T=500, device=self._device)
        self._model.load_state_dict(ck["ema"])
        self._model.eval()
        self._mtime = mtime
        self._ck_path = ck_path
        self.epoch = ck["epoch"]
        print(f"[model] loaded ep{self.epoch} on {self._device} ({ck_path})",
              flush=True)

    # ── generation ────────────────────────────────────────────────────────
    def generate(self, url: str, theme_name: str, steps: int = 100,
                 version: int | None = None) -> dict:
        """Run model inference for a URL and return the viewer cell list.

        Version is picked automatically (smallest trained standard QR that
        fits the text); pass an explicit `version=N` only for debugging.
        """
        import numpy as np
        import torch

        from qrbloom.qr import grid_xy_for_version, grid_z_for_version, qr_modules
        from qrbloom.qr import THEME_NAMES
        from qrbloom.treegen import SPECIES, attr_means

        text = url if url else "QR-Bloom"
        if version is None:
            qr, version = self.pick_version_for_text(text)
        else:
            import segno
            try:
                qr = segno.make(text, error="m", version=version)
            except Exception as e:
                raise ValueError("Text doesn't fit in the selected QR version.") from e
        with self._lock:
            self._ensure_loaded()
            core = np.array([[1 if c else 0 for c in row] for row in qr.matrix],
                            dtype=np.uint8)
            qe = qr_modules(version)
            gxy = grid_xy_for_version(version)
            gz = grid_z_for_version(version)
            off = (gxy - qe) // 2
            ctr = qe // 2

            pad = np.zeros((gxy, gxy), dtype=np.uint8)
            pad[off:off + qe, off:off + qe] = core
            if theme_name not in SPECIES:
                theme_name = "cherryblossom"
            theme_idx = THEME_NAMES.index(theme_name)
            dev = self._device
            qr_t = torch.from_numpy(pad).unsqueeze(0).to(dev)
            cond = qr_t.float().unsqueeze(1).unsqueeze(-1).expand(
                -1, 1, gxy, gxy, gz).contiguous()
            th_t = torch.tensor([theme_idx], device=dev).long()
            # Per-species mean attributes — the in-distribution conditioning
            # for a typical tree of this species at this version (a global
            # 0.5 sits outside most species' training attr range).
            attr = torch.tensor([attr_means(theme_name, version)], device=dev,
                                dtype=torch.float32)
            ver_t = torch.tensor([int(version)], device=dev, dtype=torch.long)
            with torch.no_grad():
                x0 = self._diff.sample(self._model, cond, th_t, attr=attr,
                                       steps=steps, device=dev, eta=1.0,
                                       version=ver_t).cpu().numpy()[0]

        occ = (x0[3] > 0)
        rgb = x0[:3]
        sp = SPECIES[theme_name]
        cells = []
        for i in range(qe):
            for j in range(qe):
                col = sp.qr_dark if bool(core[i, j]) else sp.qr_light
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
                "url": url, "ckpt_ep": self.epoch, "version": version,
                "grid_xy": gxy, "grid_z": gz}


service = ModelService(CKPT_DIR)


def gt_generate(url: str, theme_name: str, version: int | None = None):
    """Encode a URL and run procedural treegen (ground truth, no model).

    `version` defaults to None ⇒ the smallest trained version that fits the
    input. Pass an explicit integer only for debugging.
    """
    from qrbloom.treegen import SPECIES, generate_voxels
    text = url if url else "QR-Bloom"
    if version is None:
        qr, version = service.pick_version_for_text(text)
    else:
        import segno
        try:
            qr = segno.make(text, error="m", version=version)
        except Exception as e:
            raise ValueError("Text doesn't fit in the selected QR version.") from e
    core = [[bool(c) for c in row] for row in qr.matrix]
    if theme_name not in SPECIES:
        theme_name = "cherryblossom"
    voxels = generate_voxels(core, theme=theme_name)
    cells = []
    for v in voxels:
        x, y, z = v["pos"]
        cells.append([int(x), int(y), int(z), float(v["scale"]), v["color"]])
    return {"cells": cells, "count": len(cells), "theme": theme_name,
            "url": text, "version": version}


def list_qr_versions():
    """Return basic info about which versions are trained."""
    trained = service.trained_versions()
    return {"versions": list(range(1, 11)),
            "trained": trained,
            "default": (trained[0] if trained else 1)}


def get_state():
    rgx_png = re.compile(r'epoch_(\d+)\.png$')
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

    state = {"epochs": sorted(seen_eps), "json_set": sorted(seen_json),
             "losses": [], "target_epochs": 80}

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
            (os.path.join(CKPT_DIR, "qrbloom_all.pt"), "latest_ep"),
            (os.path.join(CKPT_DIR, "qrbloom_all_best.pt"), "best_ep"),
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


def _serve_template(name: str) -> HTMLResponse:
    """Serve a template with no-store so the browser always gets the latest."""
    return HTMLResponse(
        content=(_TMPL_DIR / name).read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


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
    from qrbloom.treegen import SPECIES
    return JSONResponse(content=[
        {"name": k, "label": sp.label, "dark": sp.qr_dark, "light": sp.qr_light}
        for k, sp in SPECIES.items()
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
        result = service.generate(url, theme, steps=100, version=body.version)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", default=8000, type=int)
    args = ap.parse_args()
    print(f"gallery → http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
