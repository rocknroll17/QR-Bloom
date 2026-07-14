QR-Bloom — NOTICE / third-party attribution
===========================================

Tree-generation algorithm
-------------------------
The voxel tree-generation logic in `qrbloom/treegen.py` is adapted — ported to
Python/NumPy — from **Grow-Voxly** by **Shovith Debnath**:

    https://github.com/Hawkay002/.Grow
    https://grow-voxly.space
    https://grow-voxly.vercel.app

This port is included **with the permission of the original author**, granted
for **NON-COMMERCIAL use only**. It is **NOT licensed under this repository's
MIT license**; commercial use of `qrbloom/treegen.py` requires separate
permission from Shovith Debnath.

Concept origin
--------------
The cherry-blossom QR-tree idea is by **Enzo Manuel Mangano** (reactiive.io,
@reactiive_). QR-Bloom contains none of his code; credit is courtesy.

Original work
-------------
Everything else in this repository — the QR-conditioned voxel **diffusion
model**, the training pipeline, and the in-browser **WebGPU inference** — is
original work by the repository author and is covered by the MIT license in
`LICENSE`.
