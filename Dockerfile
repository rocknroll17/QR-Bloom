# QR-Bloom — QR-conditioned 3D voxel tree diffusion model
#
# Multi-stage: one shared base (torch + qrbloom), two targets.
#
#   docker build --target serve -t qrbloom:serve .     # demo server (default)
#   docker build --target train -t qrbloom:train .     # training
#
# Serve (weights resolve automatically: QRBLOOM_CKPT env → mounted
# checkpoints/ → download from the Hugging Face Hub into the cache volume):
#   docker run -d --gpus all -p 8000:8000 \
#       -v qrbloom-hf-cache:/root/.cache/huggingface \
#       --name qrbloom qrbloom:serve
#
# Serve your own checkpoint instead:
#   docker run -d --gpus all -p 8000:8000 \
#       -v "$(pwd)/checkpoints:/app/checkpoints:ro" qrbloom:serve
#
# Train (mount the working tree so checkpoints and run dirs land on the host):
#   docker run --rm --gpus all -v "$(pwd):/app" \
#       -e VARIANT=_all -e QR_VERSIONS=2,3,4,5 \
#       -e BATCH=18 -e EPOCH_SIZE=40000 -e EPOCHS=300 \
#       qrbloom:train

FROM python:3.12-slim AS base

# libgomp1: OpenMP runtime required by the PyTorch / NumPy backends.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY qrbloom/ ./qrbloom/
RUN --mount=type=cache,target=/root/.cache/pip pip install .

ENV PYTHONUNBUFFERED=1


FROM base AS train

RUN --mount=type=cache,target=/root/.cache/pip pip install ".[train]"
COPY train.py evaluate.py ./
CMD ["python", "train.py"]


# Last stage = the default image: the demo server on port 8000.
FROM base AS serve

RUN --mount=type=cache,target=/root/.cache/pip pip install ".[serve]"
COPY gallery.py ./
COPY docs/ ./docs/
EXPOSE 8000
CMD ["python", "gallery.py", "--port", "8000"]
