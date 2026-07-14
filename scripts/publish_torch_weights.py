# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""Publish serving weights to the Hugging Face Hub.

Slims the training checkpoint to what serving needs (EMA weights + epoch —
the optimizer state that dominates the file is dropped, 526MB → ~131MB) and
uploads it to torch/qrbloom_all_best.pt, where the gallery's Hub fallback
looks for it. Requires HF_TOKEN in the environment.
"""
import os
import sys

sys.path.insert(0, ".")
import torch

from qrbloom import HUB_REPO

CKPT = os.path.join(os.environ.get("QRBLOOM_CKPT_DIR", "checkpoints"),
                    "qrbloom_all_best.pt")
REPO = os.environ.get("QRBLOOM_HF_REPO", HUB_REPO)
OUT = "checkpoints/qrbloom_all_best_slim.pt"

def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    torch.save({"ema": ck["ema"], "epoch": ck["epoch"]}, OUT)
    print(f"slimmed {CKPT} (epoch {ck['epoch']}) → {OUT} "
          f"({os.path.getsize(OUT) / 1e6:.0f}MB)")

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN not set — skipping upload")
        return

    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=OUT, path_in_repo="torch/qrbloom_all_best.pt",
        repo_id=REPO,
        commit_message=f"torch: serving weights (epoch {ck['epoch']})")
    print(f"uploaded → {REPO}/torch/qrbloom_all_best.pt")


if __name__ == "__main__":
    main()
