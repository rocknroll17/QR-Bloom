# SPDX-FileCopyrightText: 2026 rocknroll17
# SPDX-License-Identifier: MIT

"""QR-Bloom training entry point.

Trains the QR-conditioned voxel tree diffusion model (one DiT3D across all
QR versions) using v-prediction. Training data is generated on-the-fly
(qrbloom.data); batches are bucketed so each batch holds a single QR
version. The model learns to denoise a 4-channel (RGB + occupancy) voxel
grid conditioned on the QR footprint, a tree-species theme index, three
shape attributes, and the QR version.

Configuration comes from environment variables (see TrainConfig.from_env).
Losses (see qrbloom/model.py p_losses):
  v_mse : v-prediction MSE over all 4 channels
  v_rgb : v-prediction MSE restricted to the RGB channels
  v_occ : v-prediction MSE restricted to the occupancy channel

Outputs (with `VARIANT` env var suffix, e.g. VARIANT=_all):
  checkpoints/qrbloom{VARIANT}.pt       — latest checkpoint
  checkpoints/qrbloom{VARIANT}_best.pt  — best validation checkpoint
  runs{VARIANT}/epoch_{N:03d}.png       — per-epoch sample montage
  runs{VARIANT}/epoch_{N:03d}.json      — per-epoch voxel data for the gallery
  runs{VARIANT}/loss.png                — training loss curve
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from qrbloom.data import BlockBatchSampler, LiveTreeDataset, pad_collate
from qrbloom.model import DiT3D, Diffusion, EMA
from qrbloom.qr import grid_xy_for_version, grid_z_for_version, qr_modules
from qrbloom.treegen import THEME_NAMES, attr_means
from qrbloom.viz import render_montage, save_epoch_json, save_loss_curve

LOSS_KEYS = ("v_mse", "v_rgb", "v_occ")


def _parse_versions(s: str) -> list[int]:
    out = [int(tok) for tok in s.split(",") if tok.strip()]
    if not out:
        raise ValueError("QR_VERSIONS must contain at least one version")
    return out


def _parse_weights(s: str, versions: list[int]) -> list[float]:
    """Per-version sampling weights for the training dataset.

    Smaller QR versions are a harder learning problem (less voxel footprint,
    fewer dark modules), so when training on a mix of versions the default
    tilts sampling toward them: weight = (max_v - v + 1). For versions
    [2,3,4,5] that gives [4,3,2,1] → roughly 40/30/20/10% of the batch.
    """
    if not s:
        max_v = max(versions)
        return [float(max_v - v + 1) for v in versions]
    ws = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(ws) != len(versions):
        raise ValueError(f"QR_VERSION_WEIGHTS must have {len(versions)} values "
                         f"(one per QR_VERSIONS entry), got {len(ws)}")
    if any(w < 0 for w in ws):
        raise ValueError("QR_VERSION_WEIGHTS must be non-negative")
    if sum(ws) <= 0:
        raise ValueError("QR_VERSION_WEIGHTS must sum to > 0")
    return ws


@dataclass
class TrainConfig:
    """Everything a training run needs, resolved once up front."""
    epochs: int = 80
    batch: int = 42
    lr: float = 2e-4
    timesteps: int = 500
    seed: int = 42
    workers: int = 2
    amp_dtype: str = "fp16"
    compile: bool = False
    val_every: int = 1
    sample_steps: int = 100      # stochastic sampling steps for epoch previews
    epoch_size: int = 100000     # live-generated samples per epoch
    val_size: int = 2400         # fixed validation set size
    variant: str = ""
    # Warm-start checkpoint: weights are loaded non-strict into the fresh
    # model (e.g. seed a multi-version run from a single-version specialist).
    # Ignored when a resume checkpoint for this variant already exists.
    init_from: str = ""
    # reset_best discards the historical best on resume. Required when the
    # loss definition changes mid-run: the stale best (old loss scale) could
    # otherwise never be beaten and best-checkpoint saving would stall.
    reset_best: bool = False
    # DiT backbone hyperparameters.
    dit_dim: int = 384
    dit_depth: int = 12
    dit_heads: int = 6
    dit_patch: int = 4
    dit_grad_checkpoint: bool = False
    # Multi-version training: which QR versions to sample, with what weights.
    versions: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    versions_val: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    version_weights: list[float] = field(default_factory=lambda: [4.0, 3.0, 2.0, 1.0])
    # Montage: pin the preview QR version (None → seeded pick from versions_val).
    mont_version: int | None = None
    demo_url: str = "https://example.com"

    @classmethod
    def from_env(cls) -> "TrainConfig":
        env = os.environ.get
        versions = _parse_versions(env("QR_VERSIONS", "2,3,4,5"))
        versions_val = _parse_versions(env("QR_VERSIONS_VAL",
                                           env("QR_VERSIONS", "2,3,4,5")))
        mv = env("MONT_VERSION", "").strip()
        return cls(
            epochs=int(env("EPOCHS", "80")),
            batch=int(env("BATCH", "42")),
            lr=float(env("LR", "2e-4")),
            timesteps=int(env("T", "500")),
            seed=int(env("SEED", "42")),
            workers=int(env("WORKERS", "2")),
            amp_dtype=env("AMP_DTYPE", "fp16").lower(),
            compile=bool(int(env("COMPILE", "0"))),
            val_every=int(env("VAL_EVERY", "1")),
            sample_steps=int(env("SAMPLE_STEPS", "100")),
            epoch_size=int(env("EPOCH_SIZE", "100000")),
            val_size=int(env("VAL_SIZE", "2400")),
            variant=env("VARIANT", ""),
            init_from=env("INIT_FROM", ""),
            reset_best=bool(int(env("RESET_BEST", "0"))),
            dit_dim=int(env("DIT_DIM", "384")),
            dit_depth=int(env("DIT_DEPTH", "12")),
            dit_heads=int(env("DIT_HEADS", "6")),
            dit_patch=int(env("DIT_PATCH", "4")),
            dit_grad_checkpoint=bool(int(env("DIT_CKPT", "0"))),
            versions=versions,
            versions_val=versions_val,
            version_weights=_parse_weights(env("QR_VERSION_WEIGHTS", ""), versions),
            mont_version=int(mv) if mv else None,
            demo_url=env("DEMO_URL", "https://example.com"),
        )

    @property
    def amp_th(self) -> torch.dtype:
        return torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16

    @property
    def ckpt(self) -> Path:
        return Path("checkpoints") / f"qrbloom{self.variant}.pt"

    @property
    def ckpt_best(self) -> Path:
        return Path("checkpoints") / f"qrbloom{self.variant}_best.pt"

    @property
    def out_dir(self) -> Path:
        return Path(f"runs{self.variant}")

    @property
    def done_marker(self) -> Path:
        return Path("checkpoints") / f"DONE{self.variant}"


def _safe_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _normalize_batch(occ_u8, rgb_u8, qr_u8, device):
    """Convert uint8 tensors to normalized x0. Axes: (B, C, D=row, H=col, W=height).

    qr_u8 shape is (B, D, H); the conditioning tensor is broadcast along W
    (Z axis) so the QR footprint repeats up the whole height — matching
    `occ_u8` whose W came from the dataset's per-version grid_z.
    """
    occ = occ_u8.to(device, non_blocking=True).float().mul_(2.0).sub_(1.0)
    rgb = rgb_u8.to(device, non_blocking=True).float().mul_(1 / 127.5).sub_(1.0)
    qr = qr_u8.to(device, non_blocking=True).float()
    x0 = torch.cat([rgb, occ.unsqueeze(1)], dim=1)
    W = x0.shape[-1]
    cond = qr.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, W).contiguous()
    return x0, cond


def _unwrap(m):
    return m.module if isinstance(m, DDP) else m


class Trainer:
    """Owns a full training run: distributed setup, data, model/EMA/optimizer,
    the epoch loop, validation, montage rendering, and checkpointing."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._setup_distributed()
        self._setup_backends()
        if self.is_main:
            self.cfg.ckpt.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        if self.is_dist:
            dist.barrier()
        self._setup_data()
        self._setup_montage()
        self._setup_model()
        self._resume()

    # ── logging ──────────────────────────────────────────────────────────
    def log(self, *args, **kwargs):
        if self.is_main:
            print(*args, **kwargs, flush=True)

    # ── setup ────────────────────────────────────────────────────────────
    def _setup_distributed(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_dist = self.world > 1
        self.is_main = self.rank == 0
        if self.is_dist:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.device = f"cuda:{self.local_rank}"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _setup_backends(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _setup_data(self):
        cfg = self.cfg
        # Training samples are generated live; validation uses fixed seeds.
        train_ds = LiveTreeDataset(cfg.epoch_size, versions=cfg.versions,
                                   deterministic=False,
                                   weights=cfg.version_weights, batch=cfg.batch)
        val_ds = LiveTreeDataset(cfg.val_size, versions=cfg.versions_val,
                                 deterministic=True, seed=999, batch=cfg.batch)
        w_sum = sum(cfg.version_weights) or 1.0
        w_pct = [f"v{v}:{100 * w / w_sum:.0f}%"
                 for v, w in zip(cfg.versions, cfg.version_weights)]
        max_xy = max(grid_xy_for_version(v) for v in cfg.versions + cfg.versions_val)
        max_z = max(grid_z_for_version(v) for v in cfg.versions + cfg.versions_val)
        self.log(f"[data] live generation — epoch_size={cfg.epoch_size} val={cfg.val_size}  "
                 f"train_versions={cfg.versions}  val_versions={cfg.versions_val}  "
                 f"max_xy={max_xy}  max_z={max_z}")
        self.log(f"[data] train weights — {' '.join(w_pct)}  (raw={cfg.version_weights})")

        # Bucketed batching: block samplers keep every batch single-version.
        self.train_sampler = BlockBatchSampler(len(train_ds), cfg.batch,
                                               world=self.world, rank=self.rank,
                                               shuffle=True, seed=cfg.seed)
        val_sampler = BlockBatchSampler(len(val_ds), cfg.batch,
                                        world=self.world, rank=self.rank,
                                        shuffle=False)
        dl_kw = dict(num_workers=cfg.workers, pin_memory=True, collate_fn=pad_collate)
        if cfg.workers > 0:
            dl_kw.update(persistent_workers=True, prefetch_factor=4)
        self.train_dl = DataLoader(train_ds, batch_sampler=self.train_sampler, **dl_kw)
        self.val_dl = DataLoader(val_ds, batch_sampler=val_sampler, **dl_kw)
        self.log(f"[loader] steps/epoch(per-rank)={len(self.train_dl)}  "
                 f"val batches(per-rank)={len(self.val_dl)}  "
                 f"workers={cfg.workers}  amp={cfg.amp_dtype}  world={self.world}  "
                 f"batch/gpu={cfg.batch}  global_batch={cfg.batch * self.world}")

    def _setup_montage(self):
        """Build 9 unseen QR codes (one per theme) for the per-epoch montage.

        All 9 codes share the SAME version so they stack into one batch
        tensor; the version comes from cfg.mont_version or a seeded pick.
        """
        self.mont_batch = None
        self.mont_version = None
        if not self.is_main:
            return
        import random
        import string

        import segno
        cfg = self.cfg
        rng = random.Random(20260520)
        url_chars = string.ascii_lowercase + string.digits + "-."
        alnum_chars = string.ascii_letters + string.digits
        mont_version = cfg.mont_version if cfg.mont_version is not None else \
            cfg.versions_val[rng.randint(0, len(cfg.versions_val) - 1)]
        m_qe = qr_modules(mont_version)
        m_gxy = grid_xy_for_version(mont_version)
        m_gz = grid_z_for_version(mont_version)
        m_off = (m_gxy - m_qe) // 2

        unseen_qrs, unseen_texts = [], []
        for t in range(9):
            if mont_version >= 3 and t == 5:
                text = cfg.demo_url
            else:
                while True:
                    if mont_version == 1:
                        n = rng.randint(6, 16)
                        text = "".join(rng.choices(alnum_chars, k=n))
                    else:
                        n = rng.randint(8, min(40, m_qe))
                        if rng.random() < 0.5 and n >= 12:
                            text = "https://" + "".join(rng.choices(url_chars, k=n - 8))
                        else:
                            text = "".join(rng.choices(url_chars, k=n))
                    try:
                        segno.make(text, error="m", version=mont_version)
                        break
                    except Exception:
                        continue
            qr = segno.make(text, error="m", version=mont_version)
            core = np.array([[1 if c else 0 for c in row] for row in qr.matrix],
                            dtype=np.uint8)
            pad = np.zeros((m_gxy, m_gxy), dtype=np.uint8)
            pad[m_off:m_off + m_qe, m_off:m_off + m_qe] = core
            unseen_qrs.append(pad)
            unseen_texts.append(text)
        self.mont_version = mont_version
        self.mont_batch = (torch.from_numpy(np.stack(unseen_qrs)),
                           torch.arange(9, dtype=torch.long))
        self.log(f"[montage] unseen QRs (version={mont_version}, "
                 f"grid={m_gxy}x{m_gxy}x{m_gz}) — texts: {unseen_texts}")

        from qrbloom.treegen import generate_voxels
        gt_samples = []
        for i in range(9):
            theme_name = THEME_NAMES[i]
            qr = segno.make(unseen_texts[i], error="m", version=mont_version)
            core_bool = [[bool(c) for c in row] for row in qr.matrix]
            voxels = generate_voxels(core_bool, theme=theme_name)
            cells = [[int(v["pos"][0]), int(v["pos"][1]), int(v["pos"][2]),
                      float(v["scale"]), v["color"]] for v in voxels]
            gt_samples.append({"theme": theme_name, "text": unseen_texts[i],
                               "version": mont_version,
                               "cells": cells, "count": len(cells)})
        with open(cfg.out_dir / f"unseen_gt{cfg.variant}.json", "w") as f:
            json.dump({"version": mont_version,
                       "trained_versions": sorted(set(cfg.versions)),
                       "val_versions": sorted(set(cfg.versions_val)),
                       "samples": gt_samples}, f)
        self.log("[montage] saved ground-truth JSON")

    def _build_backbone(self) -> DiT3D:
        cfg = self.cfg
        return DiT3D(dim=cfg.dit_dim, depth=cfg.dit_depth, heads=cfg.dit_heads,
                     patch=cfg.dit_patch, n_themes=len(THEME_NAMES),
                     grad_checkpoint=cfg.dit_grad_checkpoint).to(self.device)

    def _setup_model(self):
        cfg = self.cfg
        self.base_model = self._build_backbone()
        n_params = sum(p.numel() for p in self.base_model.parameters())
        self.log(f"[model] DiT3D params={n_params / 1e6:.2f}M  "
                 f"dim={cfg.dit_dim} depth={cfg.dit_depth} "
                 f"heads={cfg.dit_heads} patch={cfg.dit_patch}")
        if cfg.init_from and not cfg.ckpt.exists():
            ck0 = torch.load(cfg.init_from, map_location=self.device,
                             weights_only=False)
            src = ck0.get("ema") or ck0.get("model")
            missing, unexpected = self.base_model.load_state_dict(src, strict=False)
            self.log(f"[init] warm start from {cfg.init_from} "
                     f"(epoch {ck0.get('epoch', '?')})  "
                     f"missing={len(missing)} unexpected={len(unexpected)}")
        self.diff = Diffusion(T=cfg.timesteps, device=self.device)
        self.ema = EMA(self.base_model, decay=0.999)

        self.model = (DDP(self.base_model, device_ids=[self.local_rank],
                          output_device=self.local_rank)
                      if self.is_dist else self.base_model)
        if cfg.compile:
            self.log("[compile] torch.compile(mode='reduce-overhead')")
            self.model = torch.compile(self.model, mode="reduce-overhead",
                                       dynamic=False)
        self.opt = torch.optim.Adam(self.base_model.parameters(), lr=cfg.lr)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(cfg.amp_th is torch.float16))

    def _resume(self):
        cfg = self.cfg
        self.start_epoch = 1
        self.hist: list[dict] = []
        self.best_val = float("inf")
        if not cfg.ckpt.exists():
            self.log("[init] no checkpoint found, training from scratch")
            return
        ck = torch.load(cfg.ckpt, map_location=self.device, weights_only=False)
        self.base_model.load_state_dict(ck["model"])
        self.ema.shadow = {k: v.to(self.device) for k, v in ck["ema"].items()}
        self.opt.load_state_dict(ck["opt"])
        if "scaler" in ck and self.scaler.is_enabled():
            try:
                self.scaler.load_state_dict(ck["scaler"])
            except Exception:
                pass
        self.start_epoch = ck["epoch"] + 1
        self.hist = ck["hist"]
        if not cfg.reset_best:
            if "best_val" in ck:
                self.best_val = ck["best_val"]
            elif self.hist:
                vals = [h.get("val_total") for h in self.hist
                        if h.get("val_total") is not None]
                if vals:
                    self.best_val = min(vals)
        self.log(f"[resume] epoch {ck['epoch']} -> start epoch {self.start_epoch}  "
                 f"best_val={self.best_val:.4f}")

    # ── run ──────────────────────────────────────────────────────────────
    def _lr_at(self, step: int, total_steps: int) -> float:
        return self.cfg.lr * 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))

    def fit(self):
        cfg = self.cfg
        total_steps = cfg.epochs * len(self.train_dl)
        cur_step = (self.start_epoch - 1) * len(self.train_dl)

        for ep in range(self.start_epoch, cfg.epochs + 1):
            self.train_sampler.set_epoch(ep)
            parts, lr, cur_step = self._train_epoch(ep, cur_step, total_steps)

            if ep % cfg.val_every == 0 or ep == cfg.epochs:
                val_metrics = self.validate()
            else:
                val_metrics = {"total": float("inf"), "iou": 0.0,
                               **{k: 0.0 for k in LOSS_KEYS}}
            pv = "  ".join(f"v{v}={val_metrics[f'v{v}']:.4f}"
                           for v in sorted(set(cfg.versions_val))
                           if f"v{v}" in val_metrics)
            self.log(f"[ep{ep:03d}] train_total={parts['total']:.4f}  "
                     f"val_total={val_metrics['total']:.4f}  "
                     f"val_iou={val_metrics['iou']:.3f}  "
                     f"lr={lr:.2e}" + (f"  |  {pv}" if pv else ""))
            self.log(f"   parts: v_mse={parts['v_mse']:.4f}  "
                     f"v_rgb={parts['v_rgb']:.4f}  v_occ={parts['v_occ']:.4f}")

            self.hist.append({"epoch": ep, "train": parts,
                              "val_total": val_metrics["total"],
                              "val_iou": val_metrics["iou"],
                              "val_parts": val_metrics, "lr": lr})
            if self.is_main:
                self._render_epoch(ep, parts)
                self._checkpoint(ep, val_metrics)
            if self.is_dist:
                dist.barrier()

        if self.is_main:
            cfg.done_marker.touch()
            self.log(f"[done] {cfg.epochs} epochs complete. checkpoint: {cfg.ckpt}")
        if self.is_dist:
            dist.destroy_process_group()

    def _train_epoch(self, ep: int, cur_step: int, total_steps: int):
        cfg = self.cfg
        self.model.train()
        parts = {k: 0.0 for k in LOSS_KEYS}
        parts["total"] = 0.0
        pbar = tqdm(self.train_dl, desc=f"ep{ep:03d}", dynamic_ncols=True,
                    disable=not self.is_main)
        n_batches = 0
        lr = self.opt.param_groups[0]["lr"]
        for occ_b, rgb_b, qr_b, th_b, attr_b, ver_b in pbar:
            lr = self._lr_at(cur_step, total_steps)
            for pg in self.opt.param_groups:
                pg["lr"] = lr

            x0, cond = _normalize_batch(occ_b, rgb_b, qr_b, self.device)
            theme = th_b.to(self.device, non_blocking=True)
            attr = attr_b.to(self.device, non_blocking=True)
            ver = ver_b.to(self.device, non_blocking=True)

            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=cfg.amp_th):
                loss, info = self.diff.p_losses(self.model, x0, cond, theme,
                                                attr, version=ver)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
                self.opt.step()
            self.ema.update(self.base_model)
            cur_step += 1
            n_batches += 1

            for k, v in info.items():
                parts[k] += v.item()
            parts["total"] += loss.item()
            if self.is_main and n_batches % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 v_rgb=f"{info['v_rgb'].item():.3f}",
                                 v_occ=f"{info['v_occ'].item():.3f}")

        for k in parts:
            parts[k] /= max(n_batches, 1)
        if self.is_dist:
            keys = LOSS_KEYS + ("total",)
            buf = torch.tensor([parts[k] for k in keys],
                               device=self.device, dtype=torch.float64)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf /= self.world
            for i, k in enumerate(keys):
                parts[k] = buf[i].item()
        return parts, lr, cur_step

    def validate(self) -> dict:
        """Mean losses and occupancy IoU on the validation set.

        IoU is estimated by injecting low-level noise (t = T/10) and
        measuring denoising accuracy at that timestep. Per-version mean
        losses are reported as "v{N}" keys — in multi-version training a
        single version can collapse while the aggregate mean still looks
        healthy, so each version is watched separately.
        """
        cfg = self.cfg
        self.model.eval()
        parts_sum = {k: 0.0 for k in LOSS_KEYS}
        parts_sum["total"] = 0.0
        iou_num = iou_den = 0.0
        n_batches = 0
        ver_list = sorted(set(cfg.versions_val))
        ver_loss = {v: 0.0 for v in ver_list}
        ver_n = {v: 0 for v in ver_list}
        t_probe = max(1, cfg.timesteps // 10)
        with torch.no_grad():
            for occ_u8, rgb_u8, qr_u8, th, attr, ver in self.val_dl:
                x0, cond = _normalize_batch(occ_u8, rgb_u8, qr_u8, self.device)
                theme = th.to(self.device, non_blocking=True)
                attr = attr.to(self.device, non_blocking=True)
                ver = ver.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=cfg.amp_th):
                    loss, info = self.diff.p_losses(self.model, x0, cond,
                                                    theme, attr, version=ver)
                for k, v in info.items():
                    parts_sum[k] += v.item()
                parts_sum["total"] += loss.item()
                v0 = int(ver[0].item())          # bucketed batch: one version
                if v0 in ver_loss:
                    ver_loss[v0] += loss.item()
                    ver_n[v0] += 1
                # Inject fixed small-t noise, recover x0, threshold for IoU.
                tb = torch.full((x0.size(0),), t_probe, dtype=torch.long,
                                device=self.device)
                noise = torch.randn_like(x0)
                xt = self.diff.q_sample(x0, tb, noise)
                am = torch.ones(x0.size(0), device=self.device)
                with torch.amp.autocast("cuda", dtype=cfg.amp_th):
                    v_pred = _unwrap(self.model)(xt, tb, cond, theme, attr, am, ver)
                x0_pred = self.diff.x0_from_v(xt, tb, v_pred.float())
                occ_pred = (x0_pred[:, 3] > 0).float()
                occ_gt = (x0[:, 3] > 0).float()
                iou_num += (occ_pred * occ_gt).sum().item()
                iou_den += (occ_pred + occ_gt - occ_pred * occ_gt).sum().item()
                n_batches += 1

        if self.is_dist:
            buf = torch.tensor(
                [parts_sum[k] for k in LOSS_KEYS] + [parts_sum["total"],
                 iou_num, iou_den, float(n_batches)]
                + [ver_loss[v] for v in ver_list]
                + [float(ver_n[v]) for v in ver_list],
                device=self.device, dtype=torch.float64)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            vals = buf.tolist()
            for i, k in enumerate(LOSS_KEYS):
                parts_sum[k] = vals[i]
            parts_sum["total"] = vals[len(LOSS_KEYS)]
            iou_num, iou_den = vals[len(LOSS_KEYS) + 1], vals[len(LOSS_KEYS) + 2]
            n_batches = max(int(vals[len(LOSS_KEYS) + 3]), 1)
            off = len(LOSS_KEYS) + 4
            for i, v in enumerate(ver_list):
                ver_loss[v] = vals[off + i]
                ver_n[v] = int(vals[off + len(ver_list) + i])

        for k in parts_sum:
            parts_sum[k] /= max(n_batches, 1)
        parts_sum["iou"] = iou_num / max(iou_den, 1e-6)
        for v in ver_list:
            parts_sum[f"v{v}"] = ver_loss[v] / max(ver_n[v], 1)
        self.model.train()
        return parts_sum

    def sample_montage(self, ep: int, png_path: str, json_path: str | None = None):
        """Sample the EMA model stochastically for each theme; write PNG + JSON."""
        cfg = self.cfg
        m_eval = self._build_backbone()
        self.ema.copy_to(m_eval)
        m_eval.eval()
        qr_u8, th = self.mont_batch
        qr = qr_u8.float().to(self.device)
        mont_gz = grid_z_for_version(self.mont_version)
        cond = qr.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, mont_gz).contiguous()
        theme = th.to(self.device)
        # Per-species mean attributes — a "typical" tree of each species at
        # this version (a global 0.5 sits outside most species' attr range).
        attr = torch.tensor([attr_means(THEME_NAMES[int(t)], self.mont_version)
                             for t in th], dtype=torch.float32, device=self.device)
        ver = torch.full((th.size(0),), int(self.mont_version),
                         device=self.device, dtype=torch.long)
        with torch.no_grad():
            x = self.diff.sample(m_eval, cond, theme, attr=attr,
                                 steps=cfg.sample_steps, device=self.device,
                                 eta=1.0, version=ver).cpu().numpy()
        occ = (x[:, 3] > 0)
        qr_np = qr_u8.numpy()
        occ = occ & np.broadcast_to((qr_np > 0)[:, :, :, None], occ.shape)
        rgb = x[:, :3]
        th_np = th.numpy()
        render_montage(occ, rgb, qr_np, th_np, png_path)
        if json_path is not None:
            save_epoch_json(ep, occ.astype(np.float32), rgb, qr_np, th_np,
                            json_path, version=self.mont_version)
        del m_eval
        torch.cuda.empty_cache()

    def _render_epoch(self, ep: int, parts: dict):
        png = str(self.cfg.out_dir / f"epoch_{ep:03d}.png")
        epoch_json = str(self.cfg.out_dir / f"epoch_{ep:03d}.json")
        self.sample_montage(ep, png, epoch_json)
        save_loss_curve(self.hist, str(self.cfg.out_dir / "loss.png"))
        self.log(f"ep{ep:03d} loss={parts['total']:.4f} "
                 f"v_rgb={parts['v_rgb']:.4f} v_occ={parts['v_occ']:.4f}")
        self.log(f"  -> {png}  {epoch_json}")

    def _checkpoint(self, ep: int, val_metrics: dict):
        new_best = val_metrics["total"] < self.best_val
        if new_best:
            self.best_val = val_metrics["total"]
        state = {"model": self.base_model.state_dict(), "ema": self.ema.shadow,
                 "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                 "epoch": ep, "hist": self.hist, "best_val": self.best_val}
        _safe_save(state, self.cfg.ckpt)
        if new_best:
            _safe_save(state, self.cfg.ckpt_best)
            self.log(f"  new best val_total={self.best_val:.4f}")


def main():
    Trainer(TrainConfig.from_env()).fit()


if __name__ == "__main__":
    main()
