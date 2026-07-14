"""gallery — self-hosted server for the QR-Bloom demo.

Serves the same static page that GitHub Pages publishes (docs/), with one
difference: the server injects `window.QRBLOOM_API` into the page, which
switches generation to this server's REST API (ModelService running the
checkpoint locally) instead of downloading the in-browser model.

Usage: python gallery.py [--port 8000]
"""
import argparse
import os
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(ROOT, "docs")
CKPT_DIR = os.path.join(ROOT, "checkpoints")


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
    def ckpt_path(self) -> str | None:
        """Local weight resolution, in priority order: the QRBLOOM_CKPT env
        (an explicit file), then the checkpoints/ directory — the
        best-validation file unless GALLERY_USE_LATEST=1. Returns None when
        nothing local exists; the loader then falls back to the published
        weights on the Hugging Face Hub."""
        forced = os.environ.get("QRBLOOM_CKPT", "").strip()
        if forced:
            return forced
        use_latest = bool(int(os.environ.get("GALLERY_USE_LATEST", "0")))
        if not use_latest:
            best = os.path.join(self.ckpt_dir, "qrbloom_all_best.pt")
            if os.path.exists(best):
                return best
        latest = os.path.join(self.ckpt_dir, "qrbloom_all.pt")
        if os.path.exists(latest):
            return latest
        return None

    @staticmethod
    def _hub_ckpt() -> str:
        """Download the published serving weights from the Hugging Face Hub.
        Cached under the huggingface_hub cache dir, so a container restart
        (with the cache volume mounted) doesn't re-download."""
        from qrbloom import HUB_REPO
        repo = os.environ.get("QRBLOOM_HF_REPO", HUB_REPO)
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo, "torch/qrbloom_all_best.pt")
        except Exception as e:
            raise ValueError(
                "No local checkpoint and the Hub download failed.") from e

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

        ck_path = self.ckpt_path() or self._hub_ckpt()
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


app = FastAPI()

# The page GitHub Pages publishes, with the server-mode flag injected: the
# page sees window.QRBLOOM_API and generates through this server's REST API
# instead of downloading the in-browser model.
_API_FLAG = '<head>\n<script>window.QRBLOOM_API = "/api";</script>'

# The demo's static modules — the exact files GitHub Pages publishes, so
# both deployments run identical code.
_DOC_FILES = {
    "embed.html": "text/html",
    "qrbloom-viewer.js": "application/javascript",
    "qrbloom-api-client.js": "application/javascript",
    "qrbloom-inference.js": "application/javascript",
    "qrbloom-model.js": "application/javascript",
}


def _serve_page() -> HTMLResponse:
    html = (Path(DOCS_DIR) / "index.html").read_text(encoding="utf-8")
    html = html.replace("<head>", _API_FLAG, 1)
    return HTMLResponse(content=html,
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/", response_class=HTMLResponse)
def page_index():
    return _serve_page()


@app.get("/index.html", response_class=HTMLResponse)
def page_index_html():
    return _serve_page()


@app.get("/{name}")
def route_doc_file(name: str):
    """Serve a whitelisted docs/ file. The name never touches the filesystem
    unless it is a literal key of _DOC_FILES, so no user-controlled byte can
    reach the path expression."""
    media_type = _DOC_FILES.get(name)
    if media_type is None:
        return Response(content=b"not found", status_code=404,
                        media_type="text/plain")
    content = (Path(DOCS_DIR) / name).read_text(encoding="utf-8")
    return Response(content=content, media_type=media_type,
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
