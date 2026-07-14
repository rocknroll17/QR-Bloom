# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""QR-Bloom — QR-conditioned 3D voxel tree diffusion."""

__version__ = "0.1.0"

# The Hugging Face repo that hosts the published weights (browser tfjs/ and
# serving torch/ artifacts). Forks point their deployments elsewhere via the
# QRBLOOM_HF_REPO environment variable or by changing this constant.
HUB_REPO = "rocknroll17/QR-Bloom"
