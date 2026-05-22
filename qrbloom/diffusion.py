"""Denoising diffusion model for 3D voxel tree generation conditioned on a QR footprint.

The model generates a 4-channel (RGB + occupancy) 32^3 voxel grid using
v-prediction (Salimans & Ho 2022) with stochastic (DDPM-equivalent) sampling.
Conditioning inputs are a QR footprint mask, a discrete theme index, and a
3-dimensional shape attribute vector. Classifier-free guidance (CFG) is
supported for the attribute signal at inference time.

Channel convention:
  x0[0..2] = R, G, B  in [-1, 1]   (uint8 0-255 mapped via / 127.5 - 1)
  x0[3]    = occupancy in {-1, +1}  (binary 0/1 mapped via * 2 - 1)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOX = 32
X0_CH = 4
N_THEMES = 10

# Voxels outside the height range [1, 21] are always unoccupied in the dataset.
Z_TREE_MIN = 1
Z_TREE_MAX = 21


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


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    """3D U-Net backbone for v-prediction denoising.

    Input: 5 channels — 4-channel noised voxel grid concatenated with the
    1-channel QR footprint mask. Output: 4-channel v prediction.

    Conditioning signals (timestep, theme, shape attributes) are fused via
    a shared embedding added to each residual block.
    """

    def __init__(self, ch: int = 48, tdim: int = 256, n_themes: int = N_THEMES):
        super().__init__()
        self.ch = ch
        self.tmlp = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.theme_emb = nn.Embedding(n_themes, tdim)
        # Shape attributes (height, density, width) are embedded via an MLP
        # and added to the shared conditioning signal.
        # attr_null is the learnable unconditional token used during CFG training.
        self.attr_mlp = nn.Sequential(nn.Linear(3, tdim), nn.SiLU(),
                                      nn.Linear(tdim, tdim))
        self.attr_null = nn.Parameter(torch.zeros(tdim))
        self.in_conv = nn.Conv3d(X0_CH + 1, ch, 3, padding=1)
        self.d1 = ResBlock3D(ch, ch, tdim)
        self.down1 = nn.Conv3d(ch, ch, 4, 2, 1)
        self.d2 = ResBlock3D(ch, ch * 2, tdim)
        self.down2 = nn.Conv3d(ch * 2, ch * 2, 4, 2, 1)
        self.d3 = ResBlock3D(ch * 2, ch * 4, tdim)
        self.m1 = ResBlock3D(ch * 4, ch * 4, tdim)
        self.m2 = ResBlock3D(ch * 4, ch * 4, tdim)
        self.up2 = nn.ConvTranspose3d(ch * 4, ch * 4, 4, 2, 1)
        self.u2 = ResBlock3D(ch * 4 + ch * 2, ch * 2, tdim)
        self.up1 = nn.ConvTranspose3d(ch * 2, ch * 2, 4, 2, 1)
        self.u1 = ResBlock3D(ch * 2 + ch, ch, tdim)
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv3d(ch, X0_CH, 3, padding=1)

    def forward(self, x, t, cond, theme, attr, attr_mask=None):
        """Forward pass.

        Args:
            x:         Noised voxel grid, shape (B, 4, D, H, W).
            t:         Timestep indices, shape (B,).
            cond:      QR footprint mask, shape (B, 1, D, H, W).
            theme:     Theme indices, shape (B,).
            attr:      Shape attribute vector, shape (B, 3), values in [0, 1].
            attr_mask: Binary mask, shape (B,). 1 = use attr conditioning,
                       0 = use unconditional null token (for CFG training).
        """
        temb = self.tmlp(timestep_embedding(t, self.ch)) + self.theme_emb(theme)
        attr_emb = self.attr_mlp(attr)
        if attr_mask is not None:
            m = attr_mask.view(-1, 1).float()
            attr_emb = m * attr_emb + (1.0 - m) * self.attr_null
        temb = temb + attr_emb
        h1 = self.d1(self.in_conv(torch.cat([x, cond], dim=1)), temb)
        h2 = self.d2(self.down1(h1), temb)
        h3 = self.d3(self.down2(h2), temb)
        h = self.m2(self.m1(h3, temb), temb)
        h = self.up2(h)
        h = self.u2(torch.cat([h, h2], 1), temb)
        h = self.up1(h)
        h = self.u1(torch.cat([h, h1], 1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))


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

    def __init__(self, T: int = 500, device: str = "cuda", rgb_weight: float = 1.5,
                 occ_color_w: float = 20.0):
        self.rgb_weight = float(rgb_weight)
        self.occ_color_w = float(occ_color_w)
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.acp = torch.cumprod(self.alphas, 0)
        self.acp_prev = torch.cat([torch.ones(1, device=device), self.acp[:-1]])
        self.sqrt_acp = self.acp.sqrt()
        self.sqrt_one_minus_acp = (1.0 - self.acp).sqrt()

        # Hard void prior: voxels outside the valid height band are always empty.
        z_mask = torch.zeros(VOX, device=device)
        z_mask[:Z_TREE_MIN] = 1.0
        z_mask[Z_TREE_MAX + 1:] = 1.0
        self.void_h_mask = z_mask.view(1, 1, 1, 1, VOX)

        # Rotation matrices for 17-view multi-view silhouette loss
        # (8 side views + 8 tilted views + 1 top-down view).
        azim_list = [0, 45, 90, 135, 180, 225, 270, 315]
        view_configs = [(az, 0) for az in azim_list] + \
                       [(az, 45) for az in azim_list] + [(0, 90)]
        mats = []
        for (az_deg, el_std_deg) in view_configs:
            a = math.radians(az_deg)
            tilt = math.radians(90 - el_std_deg)
            ca, sa = math.cos(a), math.sin(a)
            ce, se = math.cos(tilt), math.sin(tilt)
            R_az = torch.tensor([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
            R_el = torch.tensor([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
            R = R_el @ R_az
            mats.append(torch.cat([R, torch.zeros(3, 1)], dim=1))
        self.view_mats = torch.stack(mats, dim=0).to(device)     # (17, 3, 4)

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

    def p_losses(self, model, x0, cond, theme, attr, attr_drop=0.15):
        """Compute the v-prediction MSE training loss with CFG attribute dropout.

        Args:
            model:     The denoising network.
            x0:        Clean voxel grid, shape (B, 4, D, H, W).
            cond:      QR footprint mask, shape (B, 1, D, H, W).
            theme:     Theme indices, shape (B,).
            attr:      Shape attribute vector, shape (B, 3).
            attr_drop: Probability of dropping the attribute condition to the
                       null token, enabling classifier-free guidance at inference.

        Returns:
            loss:    Scalar v-MSE loss.
            metrics: Dict with per-channel loss breakdowns for monitoring.
        """
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        # Drop attribute conditioning with probability attr_drop for CFG training.
        attr_mask = (torch.rand(x0.size(0), device=x0.device) >= attr_drop).float()
        v_pred = model(x_t, t, cond, theme, attr, attr_mask)   # (B, 4, D, H, W)
        v_tgt = self.v_target(x0, t, noise)

        L = F.mse_loss(v_pred, v_tgt)

        # Per-channel breakdowns for logging only; not part of the gradient.
        v_rgb = F.mse_loss(v_pred[:, :3], v_tgt[:, :3]).detach()
        v_occ = F.mse_loss(v_pred[:, 3:4], v_tgt[:, 3:4]).detach()
        return L, {
            "v_mse": L.detach(),
            "v_rgb": v_rgb,
            "v_occ": v_occ,
        }

    @torch.no_grad()
    def sample(self, model, cond, theme, attr=None, steps=250, device="cuda",
               eta=1.0, cfg=1.0):
        """Generate a voxel grid via stochastic reverse diffusion.

        Args:
            model:  The denoising network.
            cond:   QR footprint mask, shape (B, 1, D, H, W).
            theme:  Theme indices, shape (B,).
            attr:   Shape attribute vector, shape (B, 3). Defaults to zeros.
            steps:  Number of denoising steps.
            device: Torch device string.
            eta:    Stochasticity scale. eta=1 gives DDPM-equivalent ancestral
                    sampling; eta=0 gives deterministic DDIM.
            cfg:    Classifier-free guidance scale for the attribute signal.
                    1.0 disables guidance; values > 1 amplify attribute influence.

        Returns:
            Sampled voxel grid, shape (B, 4, D, H, W), with the QR footprint
            mask applied so voxels outside the footprint are set to -1.
        """
        n = cond.size(0)
        if attr is None:
            attr = torch.zeros(n, 3, device=device)
        m1 = torch.ones(n, device=device)
        m0 = torch.zeros(n, device=device)
        x = torch.randn(n, X0_CH, VOX, VOX, VOX, device=device)
        seq = torch.linspace(self.T - 1, 0, steps).long().tolist()
        for idx, t in enumerate(seq):
            tb = torch.full((n,), t, device=device, dtype=torch.long)
            v_cond = model(x, tb, cond, theme, attr, m1)
            if cfg != 1.0:
                v_uncond = model(x, tb, cond, theme, attr, m0)
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
