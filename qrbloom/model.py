"""QR-Bloom model — DiT3D backbone + v-prediction diffusion process.

One model covers every trained QR version. The denoiser is a diffusion
transformer (DiT, Peebles & Xie 2023; adapted to voxel grids as in DiT-3D,
Mo et al. 2023): voxel patchify → transformer blocks with adaLN-Zero
conditioning → linear unpatchify head.

* patch=4 keeps token counts small (v4: 36x36x52 grid → 9x9x13 = 1,053
  tokens), so every block runs full global attention — the receptive field
  spans the entire grid at every layer, at every QR version.
* 3D sine-cosine positional embeddings are computed from the actual (D,H,W)
  of each input and cached, so one set of weights accepts any grid whose
  spatial dims are multiples of `patch`.
* Conditioning (timestep, theme, shape attributes, QR version) enters
  through adaLN-Zero. The attribute pathway carries a learnable null token
  so classifier-free guidance can drop it during training and amplify it at
  sampling time; the version enters as Fourier features of the QR module
  count (micro-conditioning as in SDXL's size conditioning).

The diffusion process is v-prediction (Salimans & Ho 2022) over a 4-channel
(RGB + occupancy) voxel grid with stochastic (DDPM-equivalent) sampling.

Channel convention:
  x0[0..2] = R, G, B  in [-1, 1]   (uint8 0-255 mapped via / 127.5 - 1)
  x0[3]    = occupancy in {-1, +1}  (binary 0/1 mapped via * 2 - 1)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

X0_CH = 4           # voxel channels: RGB + occupancy
N_THEMES = 10       # number of tree species (see qrbloom.treegen.THEMES)
DOWNSCALE = 4       # DiT patch size → spatial dims must be multiples of 4


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


# ---------------------------------------------------------------------------
# DiT3D backbone
# ---------------------------------------------------------------------------
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
        # QR-version micro-conditioning (as in SDXL's size conditioning):
        # Fourier features of the module count, added to the adaLN vector.
        # Zero-initialized output → a fresh or warm-started model behaves as
        # if unconditioned until multi-version training gives it gradient.
        self.ver_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.ver_mlp[2].weight)
        nn.init.zeros_(self.ver_mlp[2].bias)

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
            version:   QR version indices, shape (B,). Optional — omit for
                       single-version models.
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
        if version is not None:
            if not torch.is_tensor(version):
                version = torch.full((B,), int(version), device=x.device)
            modules = version.float() * 4.0 + 17.0        # QR modules per side
            c = c + self.ver_mlp(timestep_embedding(modules, 256))

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


# ---------------------------------------------------------------------------
# Diffusion process
# ---------------------------------------------------------------------------
class Diffusion:
    """V-prediction denoising diffusion over the 4-channel voxel grid.

    All four channels (RGB + occupancy) are treated uniformly under the
    v-prediction objective (Salimans & Ho 2022):

      Given alpha = sqrt(acp), sigma = sqrt(1 - acp):
        x_t = alpha * x0 + sigma * eps
        v   = alpha * eps - sigma * x0   (training target)
        x0  = alpha * x_t - sigma * v    (reconstruction)
        eps = sigma * x_t + alpha * v    (noise estimate)
    """

    def __init__(self, T: int = 500, device: str = "cuda",
                 rgb_empty_w: float = 0.05):
        # rgb_empty_w: colour-loss weight on empty (in-footprint) voxels
        # relative to occupied ones — see p_losses.
        self.rgb_empty_w = float(rgb_empty_w)
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, 0)
        self.sqrt_acp = self.acp.sqrt()
        self.sqrt_one_minus_acp = (1.0 - self.acp).sqrt()

    def q_sample(self, x0, t, noise):
        a = self.sqrt_acp[t][:, None, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None, None]
        return a * x0 + b * noise

    def v_target(self, x0, t, noise):
        """Compute v = alpha * eps - sigma * x0."""
        a = self.sqrt_acp[t][:, None, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None, None]
        return a * noise - b * x0

    def x0_from_v(self, x_t, t, v):
        """Reconstruct x0 = alpha * x_t - sigma * v."""
        a = self.sqrt_acp[t][:, None, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None, None]
        return a * x_t - b * v

    def eps_from_v(self, x_t, t, v):
        """Recover eps = sigma * x_t + alpha * v."""
        a = self.sqrt_acp[t][:, None, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None, None]
        return b * x_t + a * v

    def p_losses(self, model, x0, cond, theme, attr, attr_drop=0.15, version=None):
        """v-prediction MSE training loss with CFG attribute dropout.

        Loss is **masked to columns under a dark QR module** (`cond > 0.5`).
        Outside the mask the target is trivially "empty everywhere" (the QR
        footprint mask in `sample()` zeros those columns anyway), so
        including them in the loss only dilutes the gradient. Per-sample
        normalization keeps every sample's contribution equal regardless of
        its grid size, so the version sampler's intent (e.g. 40% v2)
        actually translates into 40% of the gradient signal.

        Args:
            model:     The denoising network.
            x0:        Clean voxel grid, shape (B, 4, D, H, W).
            cond:      QR footprint mask, shape (B, 1, D, H, W).
            theme:     Theme indices, shape (B,).
            attr:      Shape attribute vector, shape (B, 3).
            attr_drop: Probability of dropping the attribute condition to the
                       null token, enabling classifier-free guidance at inference.
            version:   QR version indices, shape (B,), or None.
        """
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        # Drop attribute conditioning with probability attr_drop for CFG training.
        attr_mask = (torch.rand(x0.size(0), device=x0.device) >= attr_drop).float()
        v_pred = model(x_t, t, cond, theme, attr, attr_mask, version)   # (B, 4, D, H, W)
        v_tgt = self.v_target(x0, t, noise)

        m = (cond > 0.5).float()                                # (B, 1, D, H, W)
        sq = (v_pred - v_tgt) ** 2

        # Per-voxel occupancy pos_weight: up-weight occupied voxels to prevent
        # sparse-occupancy collapse (e.g. palm/socotra where ~3% of the footprint
        # is occupied). w = N_empty / N_occupied, capped at 12× to avoid
        # over-aggressive gradients for very sparse themes.
        #
        # NOTE: pos_w must be a PER-VOXEL weight map, not a per-sample scalar.
        # A per-sample scalar cancels algebraically in the normalised loss:
        #   (sq * m * s).sum() / (m * s).sum()  ==  (sq * m).sum() / m.sum()
        # The spatial weight map keeps the occupied/empty ratio after
        # normalisation: occupied voxels get pos_w, empty (in-footprint) 1.0.
        occ_gt = (x0[:, 3:4] > 0).float() * m            # occupied & in footprint
        n_occ  = occ_gt.sum(dim=[1, 2, 3, 4]).clamp(min=1.0)
        n_foot = m.sum(dim=[1, 2, 3, 4]).clamp(min=1.0)
        pos_w  = ((n_foot - n_occ) / n_occ).clamp(max=12.0)   # (B,) scalar ratio
        pos_w  = pos_w.view(-1, 1, 1, 1, 1)                   # broadcast-ready
        vox_w  = occ_gt * pos_w + (1.0 - occ_gt) * m          # (B, 1, D, H, W)

        # RGB loss (ch 0-2): ~90% of footprint voxels are empty (black), so an
        # unweighted footprint mean spends most of the colour gradient on
        # "paint empty space black" and palette colours are learned slowly.
        # Down-weight empty voxels to rgb_empty_w instead of masking them out
        # entirely: intermediate sampling states feed empty-voxel colours back
        # into the network, so they must stay supervised (anchored to black)
        # to keep the reverse process in-distribution — a hard mask would let
        # them drift arbitrarily mid-sampling.
        rgb_w = (occ_gt + (m - occ_gt) * self.rgb_empty_w).expand(-1, 3, -1, -1, -1)
        v_rgb_ps = (sq[:, :3] * rgb_w).sum(dim=[1, 2, 3, 4]) / \
                   rgb_w.sum(dim=[1, 2, 3, 4]).clamp(min=1.0)

        # Occupancy loss (ch 3): per-voxel pos-weighted MSE in footprint.
        v_occ_ps = (sq[:, 3:4] * vox_w).sum(dim=[1, 2, 3, 4]) / \
                   vox_w.sum(dim=[1, 2, 3, 4]).clamp(min=1.0)

        # Combined loss: average RGB + weighted occupancy, equal channel weight.
        per_sample = (v_rgb_ps + v_occ_ps) / 4.0   # /4 keeps scale ≈ original
        L = per_sample.mean()

        return L, {
            "v_mse": L.detach(),
            "v_rgb": v_rgb_ps.mean().detach(),
            "v_occ": v_occ_ps.mean().detach(),
        }

    @torch.no_grad()
    def sample(self, model, cond, theme, attr=None, steps=250, device="cuda",
               eta=1.0, cfg=1.0, version=None):
        """Generate a voxel grid via stochastic reverse diffusion.

        Args:
            model:  The denoising network.
            cond:   QR footprint mask, shape (B, 1, D, H, W). All three
                    spatial dims (D, H, W) determine the output size and
                    must be multiples of DOWNSCALE (=4).
            theme:  Theme indices, shape (B,).
            attr:   Shape attribute vector, shape (B, 3). Defaults to zeros.
            steps:  Number of denoising steps.
            device: Torch device string.
            eta:    Stochasticity scale. eta=1 gives DDPM-equivalent ancestral
                    sampling; eta=0 gives deterministic DDIM.
            cfg:    Classifier-free guidance scale for the attribute signal.
                    1.0 disables guidance; values > 1 amplify attribute influence.
            version: QR version indices, shape (B,), or None.

        Returns:
            Sampled voxel grid, shape (B, 4, D, H, W), with the QR footprint
            mask applied so voxels outside the footprint are set to -1.
        """
        n, _, D, H, W = cond.shape
        assert D % DOWNSCALE == 0 and H % DOWNSCALE == 0 and W % DOWNSCALE == 0, \
            f"cond dims ({D},{H},{W}) must all be multiples of {DOWNSCALE}"
        if attr is None:
            attr = torch.zeros(n, 3, device=device)
        m1 = torch.ones(n, device=device)
        m0 = torch.zeros(n, device=device)
        x = torch.randn(n, X0_CH, D, H, W, device=device)
        seq = torch.linspace(self.T - 1, 0, steps).long().tolist()
        for idx, t in enumerate(seq):
            tb = torch.full((n,), t, device=device, dtype=torch.long)
            v_cond = model(x, tb, cond, theme, attr, m1, version)
            if cfg != 1.0:
                v_uncond = model(x, tb, cond, theme, attr, m0, version)
                v_pred = v_uncond + cfg * (v_cond - v_uncond)
            else:
                v_pred = v_cond
            x0_pred = self.x0_from_v(x, tb, v_pred).clamp(-1, 1)
            eps_pred = self.eps_from_v(x, tb, v_pred)
            if idx < steps - 1:
                acp_t = self.acp[t]
                acp_n = self.acp[seq[idx + 1]]
                sigma = eta * (((1 - acp_n) / (1 - acp_t)).sqrt()
                               * (1 - acp_t / acp_n).sqrt())
                sigma = torch.nan_to_num(sigma, nan=0.0)
                c = (1 - acp_n - sigma * sigma).clamp(min=0.0).sqrt()
                x = acp_n.sqrt() * x0_pred + c * eps_pred \
                    + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        # Enforce the QR footprint: voxels outside the mask are fully empty.
        mask = (cond > 0.5).float()
        return x * mask + (mask - 1)


class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model):
        model.load_state_dict(self.shadow)
