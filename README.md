# QR-Bloom

A QR-conditioned 3D voxel diffusion model. It grows a 3D voxel tree whose
top-down silhouette is a scannable QR code: viewed from the side it is a tree,
viewed from above it is a working QR code.

![demo](assets/demo.gif)

## Overview

QR-Bloom is a denoising diffusion model that generates a `32 x 32 x 32` voxel
grid with four channels — RGB color and occupancy. Generation is conditioned on
three inputs:

- **QR footprint** — a QR code matrix. The model is constrained so that the
  tree's top-down projection reproduces the code, keeping it scannable.
- **Theme** — one of ten tree species styles: cherry blossom, pine, dragon
  tree, maple, baobab, willow, magnolia, saguaro cactus, palm, and acacia.
- **Shape attributes** — three continuous controls (height, fullness, spread)
  that let the user dial in the tree's proportions at inference time.

Training data is produced by a procedural voxel-tree generator, so the dataset
is effectively infinite and is generated on the fly during training — no data
files are stored on disk.

## How it works

The model is a 3D U-Net trained with **v-prediction**: the network predicts the
diffusion `v` target rather than the clean sample or the noise directly.
Combined with stochastic (ancestral) sampling, this lets per-voxel color be
drawn from a distribution instead of regressed toward a mean — important
because foliage color is inherently varied, and a deterministic regressor
collapses it to a single dull average.

The shape attributes are trained with **classifier-free guidance**, so a user
can push the generator toward taller, fuller, or wider trees at sampling time.

## Repository layout

```
qrbloom/
  treegen.py     Procedural voxel-tree generator (10 species, per-species
                 augmentation built in)
  qr.py          QR-code generation and voxel-grid constants
  diffusion.py   v-prediction 3D U-Net and the diffusion process
train.py         Distributed training entry point
evaluate.py      Quantitative evaluation (occupancy, color fidelity, diversity)
gallery.py       Interactive 3D web viewer (FastAPI)
assets/          Demo assets
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU for training.

## Usage

### Training

Training is configured through environment variables and launched with
`torchrun`. For a single GPU:

```bash
python train.py
```

For multi-GPU data-parallel training:

```bash
torchrun --nproc_per_node=4 train.py
```

Common settings (all optional, shown with their defaults):

| Variable      | Default  | Description                       |
|---------------|----------|-----------------------------------|
| `EPOCHS`      | `80`     | Number of training epochs         |
| `BATCH`       | `42`     | Per-GPU batch size                |
| `LR`          | `2e-4`   | Base learning rate                |
| `T`           | `500`    | Diffusion timesteps               |
| `EPOCH_SIZE`  | `100000` | Live-generated samples per epoch  |
| `SAMPLE_STEPS`| `100`    | Sampling steps for epoch previews |

Checkpoints are written to `checkpoints/qrbloom.pt` (latest) and
`checkpoints/qrbloom_best.pt` (best validation loss). Per-epoch sample previews
are written to `runs/`.

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/qrbloom_best.pt
```

This samples trees from held-out QR codes and reports occupancy IoU against the
procedural ground truth, color-palette fidelity, per-tree color diversity, and
sample-to-sample diversity for a fixed QR code.

### Interactive gallery

```bash
python gallery.py --port 8000
```

Then open `http://localhost:8000`. The gallery is a FastAPI app that renders
generated trees in 3D, provides an epoch slider, and hot-reloads as new
checkpoints appear during training. Tilt the camera toward top-down and the
tree's block colors resolve into the scannable QR code.

## Docker

```bash
docker build -t qrbloom .

# Interactive gallery
docker run --rm -p 8000:8000 \
    -v "$(pwd)/checkpoints:/app/checkpoints" -v "$(pwd)/runs:/app/runs" qrbloom

# Training (requires an NVIDIA GPU and the NVIDIA Container Toolkit)
docker run --rm --gpus all -v "$(pwd)/checkpoints:/app/checkpoints" qrbloom python train.py
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
