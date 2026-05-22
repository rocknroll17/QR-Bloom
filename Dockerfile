# QR-Bloom — QR-conditioned 3D voxel tree diffusion model
#
# Build:     docker build -t qrbloom .
# Gallery:   docker run --rm -p 8001:8001 \
#                -v "$(pwd)/checkpoints:/app/checkpoints" \
#                -v "$(pwd)/runs:/app/runs" qrbloom
# Train:     docker run --rm --gpus all -v "$(pwd)/checkpoints:/app/checkpoints" qrbloom python train.py
# Evaluate:  docker run --rm --gpus all qrbloom python evaluate.py

FROM python:3.12-slim

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

# Default: serve the interactive 3D gallery on port 8001.
# Training and evaluation need an NVIDIA GPU — override the command and pass --gpus all.
EXPOSE 8001
CMD ["python", "gallery.py", "--port", "8001"]
