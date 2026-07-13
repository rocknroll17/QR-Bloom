# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""DiT3D — 3D diffusion transformer for QR-conditioned voxel tree generation.

Voxel patchify → transformer blocks with adaLN-Zero conditioning → linear
unpatchify head. The architecture follows DiT (Peebles & Xie 2023) adapted
to voxel grids as in DiT-3D (Mo et al. 2023).

* patch=4 keeps token counts small (v4: 36x36x52 grid → 9x9x13 = 1,053
  tokens), so every block runs full global attention — the receptive field
  spans the entire grid at every layer, at every QR version.
* 3D sine-cosine positional embeddings are computed from the actual (D,H,W)
  of each input and cached, so one model accepts any grid whose spatial
  dims are multiples of `patch`.
* Conditioning (timestep, theme, shape attributes) enters through
  adaLN-Zero. The attribute pathway carries a learnable null token so
  classifier-free guidance can drop it during training and amplify it at
  sampling time.

Input: 5 channels — the 4-channel noised voxel grid concatenated with the
1-channel QR footprint mask. Output: 4-channel v prediction (the diffusion
process lives in qrbloom.diffusion.Diffusion).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from qrbloom.diffusion import X0_CH, N_THEMES, timestep_embedding


def _posemb_1d(dim: int, n: int, device, dtype):
    """Standard 1D sin-cos table: (n, dim). dim must be even."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
    args = torch.arange(n, device=device, dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(dtype)


def posemb_3d(dim: int, d: int, h: int, w: int, device, dtype=torch.float32):
    """3D sin-cos positional embedding, (d*h*w, dim).

    dim is split evenly across the three axes (each axis part must be even).
    Token order matches x.flatten(2).transpose(1, 2) on a (B, C, D, H, W)
    tensor: D-major, then H, then W.
    """
    ad = dim // 3 - (dim // 3) % 2
    aw = ad
    ah = dim - ad - aw          # H takes the remainder (still even if dim is)
    pd = _posemb_1d(ad, d, device, dtype)
    ph = _posemb_1d(ah, h, device, dtype)
    pw = _posemb_1d(aw, w, device, dtype)
    out = torch.cat([
        pd[:, None, None, :].expand(d, h, w, ad),
        ph[None, :, None, :].expand(d, h, w, ah),
        pw[None, None, :, :].expand(d, h, w, aw),
    ], dim=-1)
    return out.reshape(d * h * w, dim)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, dim))
        # adaLN-Zero: 6 modulation vectors, zero-initialized so every block
        # starts as identity — the standard recipe for stable DiT training.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def _attn(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                    # (B, nh, L, hd)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.proj(out)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c)[:, None, :].chunk(6, dim=-1)
        x = x + g1 * self._attn(self.norm1(x) * (1 + sc1) + sh1)
        x = x + g2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)
        return x


class DiT3D(nn.Module):
    """Denoising backbone: patchified voxel tokens + adaLN-Zero transformer."""

    def __init__(self, dim: int = 384, depth: int = 12, heads: int = 6,
                 patch: int = 4, n_themes: int = N_THEMES,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.patch = patch
        self.grad_checkpoint = grad_checkpoint

        # Patchify: 5 input channels (voxels + QR footprint) → dim per patch.
        self.patch_embed = nn.Conv3d(X0_CH + 1, dim, patch, stride=patch)

        # Conditioning: timestep MLP + theme embedding + attribute MLP with
        # a learnable null token for CFG dropout.
        self.tmlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.theme_emb = nn.Embedding(n_themes, dim)
        self.attr_mlp = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attr_null = nn.Parameter(torch.zeros(dim))

        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])

        # Output head, adaLN-modulated and zero-initialized like DiT's.
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, patch ** 3 * X0_CH)
        nn.init.zeros_(self.ada_f[1].weight)
        nn.init.zeros_(self.ada_f[1].bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self._pos_cache: dict = {}

    def _pos(self, dp, hp, wp, device):
        key = (dp, hp, wp)
        pe = self._pos_cache.get(key)
        if pe is None or pe.device != device:
            pe = posemb_3d(self.dim, dp, hp, wp, device)
            self._pos_cache[key] = pe
        return pe

    def forward(self, x, t, cond, theme, attr, attr_mask=None, version=None):
        """Predict v for a noised voxel grid.

        Args:
            x:         Noised voxel grid, shape (B, 4, D, H, W).
            t:         Timestep indices, shape (B,).
            cond:      QR footprint mask, shape (B, 1, D, H, W).
            theme:     Theme indices, shape (B,).
            attr:      Shape attribute vector, shape (B, 3), values in [0, 1].
            attr_mask: Binary mask, shape (B,). 1 = use attr conditioning,
                       0 = use the unconditional null token (CFG training).
            version:   Accepted for the Diffusion interface; unused.
        """
        B, _, D, H, W = x.shape
        p = self.patch
        assert D % p == 0 and H % p == 0 and W % p == 0, \
            f"grid ({D},{H},{W}) must be a multiple of patch={p}"

        c = self.tmlp(timestep_embedding(t, 256)) + self.theme_emb(theme)
        attr_emb = self.attr_mlp(attr)
        if attr_mask is not None:
            m = attr_mask.view(-1, 1).float()
            attr_emb = m * attr_emb + (1.0 - m) * self.attr_null
        c = c + attr_emb

        h = self.patch_embed(torch.cat([x, cond], dim=1))        # (B, dim, D', H', W')
        dp, hp, wp = h.shape[2], h.shape[3], h.shape[4]
        h = h.flatten(2).transpose(1, 2)                          # (B, L, dim)
        h = h + self._pos(dp, hp, wp, h.device).to(h.dtype)[None]

        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, c, use_reentrant=False)
            else:
                h = blk(h, c)

        sh, sc = self.ada_f(c)[:, None, :].chunk(2, dim=-1)
        h = self.head(self.norm_f(h) * (1 + sc) + sh)             # (B, L, p³·4)

        # Unpatchify: (B, L, p³·C) → (B, C, D, H, W)
        h = h.reshape(B, dp, hp, wp, p, p, p, X0_CH)
        h = h.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return h.reshape(B, X0_CH, D, H, W)
