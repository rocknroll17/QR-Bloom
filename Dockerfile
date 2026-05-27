# QR-Bloom — QR-conditioned 3D voxel tree diffusion model
#
# Build:
#   docker build -t qrbloom .
#
# Run the interactive gallery (CPU inference; safe to run alongside training):
#   docker run -d -p 8000:8000 -v "$(pwd):/app" --name qrbloom-gallery qrbloom
#
# Run the gallery with GPU inference (requires nvidia-container-toolkit):
#   docker run -d --gpus all -p 8000:8000 -v "$(pwd):/app" --name qrbloom-gallery qrbloom
#
# Train one QR version (one model per version is the design):
#   docker run --rm --gpus all -v "$(pwd):/app" \
#       -e VARIANT=_v2 -e QR_VERSIONS=2 -e QR_VERSION_WEIGHTS=1.0 \
#       -e BATCH=128 -e EPOCH_SIZE=80000 -e EPOCHS=300 \
#       qrbloom python train.py
#
# Override the host port (gallery listens on whatever --port is passed):
#   docker run -d -p 9000:9000 -v "$(pwd):/app" qrbloom python gallery.py --port 9000

FROM python:3.14-slim

# libgomp1: OpenMP runtime required by the PyTorch / NumPy backends.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code.
COPY qrbloom/ ./qrbloom/
COPY templates/ ./templates/
COPY train.py evaluate.py gallery.py ./

ENV PYTHONUNBUFFERED=1

# Default: serve the gallery on port 8000.
# Train/evaluate require an NVIDIA GPU — override CMD and pass --gpus all.
EXPOSE 8000
CMD ["python", "gallery.py", "--port", "8000"]
