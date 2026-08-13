<div align="center">

# QR-Bloom

**A QR-conditioned 3D voxel diffusion model.**
It grows a 3D voxel tree whose top-down silhouette is a scannable QR code:
viewed from the side it is a tree, viewed from above it is a working QR code.

[![CodeQL](https://github.com/rocknroll17/QR-Bloom/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/rocknroll17/QR-Bloom/actions/workflows/codeql.yml)
[![Pages](https://github.com/rocknroll17/QR-Bloom/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/rocknroll17/QR-Bloom/actions/workflows/pages.yml)
[![Release](https://github.com/rocknroll17/QR-Bloom/actions/workflows/release.yml/badge.svg)](https://github.com/rocknroll17/QR-Bloom/actions/workflows/release.yml)
[![GHCR](https://img.shields.io/badge/ghcr.io-qr--bloom-2ea44f?logo=docker&logoColor=white)](https://github.com/rocknroll17/QR-Bloom/pkgs/container/qr-bloom)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![demo](assets/demo.gif)

**[✨ Live demo →](https://qr-bloom.rocknroll17.com/)**

</div>

Type a URL or short text, pick a theme, and the diffusion model runs right
in your browser (WebGPU) to grow a fresh voxel tree. Flip to the top-down
view to scan the QR, or copy a one-line embed of the result. No backend —
it's a static GitHub Pages site that loads the model weights on the fly.

> **Credits.** Tree shapes are ported from [Grow-Voxly](https://grow-voxly.vercel.app)
> by Shovith Debnath (with permission, non-commercial — see [NOTICE](THIRD_PARTY_NOTICES.md)).
> Concept by [Enzo Manuel Mangano](https://reactiive.io).

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Interactive gallery](#interactive-gallery)
  - [Local browser demo (WebGPU)](#local-browser-demo-webgpu)
- [Docker](#docker)
- [License](#license)

## Features

QR-Bloom is a denoising diffusion model that generates a 4-channel voxel grid
(RGB + occupancy). The grid size scales with the QR code version — taller
trees and wider footprints for larger codes. Generation is conditioned on:

- 📱 **QR footprint** — a QR code matrix. The model is constrained so that the
  tree's top-down projection reproduces the code, keeping it scannable.
- 🌸 **Theme** — one of ten tree species: cherry blossom, pine, dragon tree,
  maple, baobab, willow, magnolia, saguaro cactus, palm, and acacia.
- 🎛️ **Shape attributes** — three continuous controls (height, fullness,
  spread) that let the user dial in the tree's proportions at inference time.
- ♾️ **Infinite training data** — training data is produced by a procedural
  voxel-tree generator, so the dataset is effectively infinite and is
  generated on the fly during training; no data files are stored on disk.
- 🧩 **One model, all sizes** — one model covers all trained QR versions
  (v2..v5): the QR version enters the network as conditioning, batches are
  bucketed so each batch holds a single grid size, and the positional
  embeddings are computed from the input size.

## How it works

- 🧠 **Diffusion transformer (DiT) denoiser** — the voxel grid is split into
  4×4×4 patches, processed by transformer blocks with adaLN-Zero
  conditioning, and projected back to voxels. Every block runs full global
  attention, so the receptive field spans the entire grid at every layer and
  every QR version — the property that lets one model serve all sizes.
- 🎯 **v-prediction training** — the network predicts the diffusion `v`
  target rather than the clean sample or the noise directly. Combined with
  stochastic (ancestral) sampling, this lets per-voxel color be drawn from a
  distribution instead of regressed toward a mean — important because foliage
  color is inherently varied, and a deterministic regressor collapses it to a
  single dull average.
- 🧭 **Classifier-free guidance on shape** — the shape attributes are trained
  with classifier-free guidance, so a user can push the generator toward
  taller, fuller, or wider trees at sampling time. At inference, every
  surface conditions on measured per-species mean attributes, so each species
  is generated at its typical proportions.

## Repository layout

```text
qrbloom/
  treegen.py     Procedural voxel-tree generator (SPECIES registry:
                 palette, proportions, augmentation, shape field)
  qr.py          QR-code generation and voxel-grid sizing helpers
  model.py       DiT3D backbone + v-prediction diffusion process
  data.py        Live dataset + version-bucketed batch sampler
  viz.py         Training montages, loss curves, epoch JSON
train.py         TrainConfig + Trainer (one model, all QR versions)
evaluate.py      Quantitative evaluation (occupancy, color fidelity, diversity)
gallery.py       Self-hosted demo server (ModelService + REST API)
docs/            The demo page (GitHub Pages and the gallery serve the same
                 files; generation backend is chosen at serve time)
scripts/         Weight export for the in-browser demo
Makefile         Task shortcuts (train, stop, gallery, status, logs, clean)
assets/          Demo assets
```

## Installation

```bash
pip install -e ".[train,serve]"
```

Requires Python 3.10+ and a CUDA-capable GPU for training. The extras split
by role: `train` (matplotlib, tqdm), `serve` (fastapi, uvicorn,
huggingface_hub), `dev` (both plus ruff).

## Usage

### Training

The simplest way is the Makefile, which launches one multi-version training
run in the background and resumes from the latest checkpoint:

```bash
make train           # train v2..v5 in one model (resumes if possible)
make status          # GPU usage + latest per-version val metrics
make stop            # kill the training
make logs            # tail -f the training log
```

BATCH size and other knobs are Make variables — override at the command
line, e.g. `BATCH=32 EPOCHS=500 make train`. See `make help` for the list.

If you'd rather drive `train.py` directly, every setting is also an
environment variable. Important ones:

| Variable             | Default  | Description                            |
|----------------------|----------|----------------------------------------|
| `VARIANT`            | `""`     | Suffix for checkpoints/runs dirs       |
| `QR_VERSIONS`        | `2,3,4,5`| Which QR versions to sample from       |
| `QR_VERSION_WEIGHTS` | tilted   | Per-version sampling weights           |
| `EPOCHS`             | `80`     | Number of training epochs              |
| `BATCH`              | `42`     | Per-GPU batch size                     |
| `LR`                 | `2e-4`   | Base learning rate                     |
| `T`                  | `500`    | Diffusion timesteps                    |
| `EPOCH_SIZE`         | `100000` | Live-generated samples per epoch       |
| `SAMPLE_STEPS`       | `100`    | Sampling steps for epoch previews      |
| `INIT_FROM`          | `""`     | Warm-start checkpoint (non-strict)     |
| `MONT_VERSION`       | random   | Pin the montage QR version             |

Training writes:

- `checkpoints/qrbloom{VARIANT}.pt` — latest checkpoint
- `checkpoints/qrbloom{VARIANT}_best.pt` — best validation loss
- `runs{VARIANT}/epoch_{N}.json|.png` — per-epoch previews

The Makefile uses `VARIANT=_all`, producing `qrbloom_all.pt`,
`qrbloom_all_best.pt`, and `runs_all/`.

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/qrbloom_all_best.pt
```

This samples trees from held-out QR codes at every trained version and
reports occupancy IoU against the procedural ground truth, color-palette
fidelity, per-tree color diversity, and sample-to-sample diversity.

### Interactive gallery

```bash
make gallery           # CPU inference, safe alongside training
make gallery-gpu       # GPU inference (picks the first free GPU)
# Override the port: PORT=9000 make gallery
```

Or run directly:

```bash
python gallery.py --port 8000
```

Then open `http://localhost:8000`. The gallery serves the same page GitHub
Pages publishes, with generation switched to its REST API: the server runs
the checkpoint locally (hot-reloading it as training updates the file), so
the browser downloads no model. Tilt the camera toward top-down and the
tree's block colors resolve into the scannable QR code.

### Local browser demo (WebGPU)

The same static page that runs on GitHub Pages can serve a locally trained
model — no upload needed:

```bash
make export          # write EMA weights + manifest into docs/assets/
make demo            # serve docs/ on :8080
```

Open `http://localhost:8080` (localhost is required for WebGPU). The page
auto-detects `docs/assets/` and shows "Ready · local weights"; without that
folder (e.g. the deployed Pages site — it is gitignored) it loads the
published weights from Hugging Face. Re-run `make export` after training
improves and refresh the page to pick the new model up.

## Docker

A prebuilt image is published to GitHub Container Registry on every push to `main`.

### Pull the prebuilt image

```bash
docker pull ghcr.io/rocknroll17/qr-bloom:latest
```

Images are published only when a release is cut (a `v*` tag is pushed),
so every pull lands on a tagged, changelog-backed build.

Available tags:

- `latest` — most recent release
- `vX.Y.Z` / `X.Y.Z` / `X.Y` / `X` — stable releases (e.g. `v1.1.3`, `1.1.3`, `1.1`, `1`)

The image is public, so no `docker login` is required to pull.

### Two images, one Dockerfile

The Dockerfile is multi-stage with two targets sharing a torch base layer:

```bash
docker build --target serve -t qrbloom:serve .   # demo server (default target)
docker build --target train -t qrbloom:train .   # training
```

### Run the demo server

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
for GPU inference (drop `--gpus all` and set `-e GALLERY_DEVICE=cpu` to run
on CPU). Weights resolve automatically at startup, in priority order: the
`QRBLOOM_CKPT` env (an explicit file) → a mounted `checkpoints/` directory
→ the published weights on the Hugging Face Hub (cached in the volume, so
restarts don't re-download):

```bash
docker run -d --gpus all \
    --name qrbloom \
    --restart unless-stopped \
    -p 8000:8000 \
    -v qrbloom-hf-cache:/root/.cache/huggingface \
    ghcr.io/rocknroll17/qr-bloom:latest
```

Serve your own checkpoint instead by mounting it:

```bash
docker run -d --gpus all -p 8000:8000 \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
    ghcr.io/rocknroll17/qr-bloom:latest
```

Then open `http://localhost:8000`. Use a different host port with
`-p 9000:8000`.

### Train inside the container

Mount the whole working tree so checkpoints, run directories, and source
edits are visible from the host:

```bash
docker run --rm --gpus all \
    -v "$(pwd):/app" \
    -e VARIANT=_all -e QR_VERSIONS=2,3,4,5 \
    -e BATCH=18 -e EPOCH_SIZE=40000 -e EPOCHS=300 \
    qrbloom:train
```

### Stop / update

```bash
docker stop qrbloom-gallery && docker rm qrbloom-gallery
docker pull ghcr.io/rocknroll17/qr-bloom:latest        # grab newest build
# then re-run the docker run … command above
```

### Build locally instead

```bash
docker build --target serve -t qrbloom:serve .
docker run -d -p 8000:8000 -v qrbloom-hf-cache:/root/.cache/huggingface qrbloom:serve
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
