"""treegen.py — QR-conditioned voxel tree generator (10 species).

Attribution
-----------
The voxel tree-generation algorithm in this file is adapted — ported to
Python/NumPy — from **Grow-Voxly** by **Shovith Debnath**:

    https://github.com/Hawkay002/.Grow
    https://grow-voxly.space

Grow-Voxly is published without an explicit license (default all-rights-
reserved). This port is included with credit and in good faith; the original
author's permission has been requested. It is provided for non-commercial,
educational use and is NOT covered by this repository's MIT license (see
NOTICE). The diffusion model trained on this generator's output is original
work.

Design
------
* Trees grow only above the QR's dark modules (scannable from above); the
  QR plane itself is the ground.
* Shapes are built from implicit fields (distance / angle functions)
  evaluated with vectorised numpy operations.
* augmentation: per-species jitter on tree height, fullness (density) and
  radius diversifies the synthesised training data.

API
---
build_tree(theme, e, H, npr)       -> (trunk, leaf, flower)  (e,H,e) bool
generate_voxels(qr, theme, rng)    -> voxel list  (QR plane + tree above)
"""
from __future__ import annotations

import math
import random

import numpy as np


# ===========================================================================
# Species data — parameters + palette
# ===========================================================================
THEMES = {
    "cherryblossom": {
        "density": 0.55, "cluster": 2.5, "qr_dark": "#b61260", "qr_light": "#ffeefd",
        "trunk": "#331d1b",
        "leaf": ["#ffb4c9", "#ffa5b9", "#f97e9d", "#ff7690", "#f8ebe3"]},
    "pine": {
        "density": 0.50, "cluster": 1.5, "qr_dark": "#01553f", "qr_light": "#e8faf9",
        "trunk": "#33201a",
        "leaf": ["#1f4838", "#266f48", "#3d8e74", "#4eb290", "#051918"]},
    "socotra": {
        "density": 0.70, "cluster": 3.0, "qr_dark": "#481600", "qr_light": "#fbfff0",
        "trunk": "#591503",
        "leaf": ["#204827", "#3d6c48", "#549267", "#266f48", "#172216"]},
    "maple": {
        "density": 0.40, "cluster": 2.0, "qr_dark": "#953b0d", "qr_light": "#fbf0e8",
        "trunk": "#3f2a20",
        "leaf": ["#a50a0e", "#cb0800", "#df3700", "#e15700", "#ed930d"]},
    "baobab": {
        "density": 0.60, "cluster": 2.6, "qr_dark": "#573932", "qr_light": "#f1f5f3",
        "trunk": "#926a4c",
        "leaf": ["#28611f", "#41711d", "#3f7523", "#4d7e2d"]},
    "willow": {
        "density": 0.40, "cluster": 1.5, "qr_dark": "#355229", "qr_light": "#e8faf9",
        "trunk": "#41392c",
        "leaf": ["#899b51", "#9cb06b", "#7a7d49", "#bac17a", "#5c682b"]},
    "magnolia": {
        "density": 0.50, "cluster": 2.2, "qr_dark": "#52239b", "qr_light": "#f1fbff",
        "trunk": "#53453a",
        "leaf": ["#263216", "#294f26", "#39643b"],
        "flower": ["#f9f7ff", "#ffeefd", "#fee0ff", "#e9aeff"]},
    "saguaro_cactus": {
        "density": 1.0, "cluster": 1.0, "qr_dark": "#1b4f35", "qr_light": "#fff7cb",
        "trunk": "#1b4f35",
        "leaf": ["#4ed888", "#29c164", "#119e44", "#197c35"],
        "flower": ["#ee395a", "#f3767d", "#e51a45"]},
    "palm": {
        "density": 1.0, "cluster": 1.0, "qr_dark": "#4a7a3f", "qr_light": "#f0f6ec",
        "trunk": "#7a5c3c",
        "leaf": ["#3f8a4f", "#4fa05f", "#62b572", "#357544"]},
    "acacia": {
        "density": 0.90, "cluster": 2.0, "qr_dark": "#6b7a38", "qr_light": "#f3f4e6",
        "trunk": "#5a4632",
        "leaf": ["#5f8a3a", "#6f9c45", "#7faa52", "#527a32"]},
}

LABELS = {"cherryblossom": "Cherry Blossom", "pine": "Pine", "socotra": "Dragon Tree",
          "maple": "Maple", "baobab": "Baobab",
          "willow": "Willow", "magnolia": "Magnolia",
          "saguaro_cactus": "Saguaro Cactus", "palm": "Palm", "acacia": "Acacia"}


def _th_bn(theme, scale_xy, scale_z):
    """Returns (trunk_height, search_limit) per species.

    scale_xy drives canopy reach (grows with QR version), scale_z drives
    trunk height. Z is capped because the voxel grid Z is fixed (32).
    """
    fl = math.floor
    table = {
        "cherryblossom":  (fl(9 * scale_z), fl(10 * scale_xy)),
        "pine":           (fl(7 * scale_z), fl(8 * scale_xy)),
        "socotra":        (fl(7 * scale_z) + fl(5 * scale_z), fl(10 * scale_xy)),
        "maple":          (fl(7 * scale_z), fl(8 * scale_xy)),
        "baobab":         (fl(10 * scale_z) + fl(7 * scale_z), fl(8 * scale_xy)),
        "willow":         (fl(7 * scale_z) + fl(3 * scale_z), fl(10 * scale_xy)),
        "magnolia":       (fl(7 * scale_z), fl(10 * scale_xy)),
        "saguaro_cactus": (1, fl(10 * scale_xy)),
        "palm":           (fl(13 * scale_z), fl(12 * scale_xy)),
        "acacia":         (fl(9 * scale_z), fl(10 * scale_xy)),
    }
    return table[theme]


# ===========================================================================
# Common helpers — coordinate grid / hash noise
# ===========================================================================
def _axes(e, H):
    cx = e // 2
    x = (np.arange(e) - cx).astype(float)[:, None, None]
    y = np.arange(H).astype(float)[None, :, None]
    z = (np.arange(e) - cx).astype(float)[None, None, :]
    return x, y, z


def _hash(x, y, z):
    """Deterministic hash noise in [0, 1] — shader idiom fract(sin(dot)*k). Used for scattering leaves."""
    v = np.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453
    return v - np.floor(v)


def _truncated_normal(npr, lo, hi, max_tries=20):
    """Truncated normal in [lo, hi] via rejection sampling.

    mean = midpoint, std = range/4 so ~95% of N(mean, std) already lies
    inside [lo, hi] (rejection rate ~5%, ~1 try on average). Most samples
    cluster near the canonical species shape; extremes are rare. Falls
    back to clipping after max_tries to guarantee termination.
    """
    mid = 0.5 * (lo + hi)
    std = max(1e-9, (hi - lo) / 4.0)
    for _ in range(max_tries):
        v = float(npr.normal(mid, std))
        if lo <= v <= hi:
            return v
    return float(np.clip(npr.normal(mid, std), lo, hi))


def _seg_dist_field(x, z, y, p0, p1):
    """Planar distance from voxel (x, z) to the cross-section point of segment p0->p1 at height y."""
    y0, y1 = p0[1], p1[1]
    t = np.clip((y - y0) / max(y1 - y0, 1e-6), 0.0, 1.0)
    bx = p0[0] + (p1[0] - p0[0]) * t
    bz = p0[2] + (p1[2] - p0[2]) * t
    return np.sqrt((x - bx) ** 2 + (z - bz) ** 2), t


# ===========================================================================
# Per-species shape builders — return (trunk, leaf, flower) bool masks
#   x,z: center-aligned planar coords  y: height  cyy: y - trunk_height
#   rad: canopy radius (already scaled by version)  bnd: pine cone bound
#   cn: cluster noise  core: trunk core region  dens: density (after augmentation)
#   sc: scale_xy (>=1.0, grows with QR version) — multiply any voxel-radius
#       constant by `sc` so trunks/branches stay proportional at larger versions.
# ===========================================================================
def _cherry(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5 * sc)
    bump = np.where(_hash(x, y, z) > 0.95, 1.5, 0.0)
    shape = np.sqrt(x ** 2 / 2.5 + cyy ** 2 / 0.5 + z ** 2 / 2.5) <= rad * 1.3 + bump
    leaf = (y >= th * 0.4) & shape & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


def _pine(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5 * sc)
    cone = bnd * 2.2
    shape = (cyy >= -1) & (cyy < cone) & (rxz <= np.maximum(cone - cyy, 0) * 0.45)
    leaf = shape & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


def _maple(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5 * sc)
    shape = np.sqrt(x ** 2 / 2.5 + cyy ** 2 + z ** 2 / 2.5) <= rad
    leaf = (y >= th * 0.4) & shape & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


def _willow(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    """Weeping willow: solid dome + cascading drapes from its underside.

    Each drape starts at the *dome's curved underside* at its (x, z) column
    (not at the trunk-top plane) and hangs straight down. The dome surface
    is highest at the rim (cyy_bot ≈ 0) and lowest at the center (cyy_bot
    ≈ -canopy_r/√1.8), so rim drapes naturally start higher and can reach
    further down — that's the silhouette real weeping willows show.

    Zones (selection density only; lengths are similar so the silhouette
    reads as a unified curtain, not a fringe-vs-stub mismatch):
      * Rim band:  ~35% of columns drape (curtain effect, not packed).
      * Inner:     ~12% of columns drape (sparse streams through the front).
      * Outside the canopy footprint: empty.
    """
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.8 * sc)

    # --- solid dome (subtle texture from cn noise) ------------------------
    canopy_r = rad * 1.6
    dome = ((cyy >= -1) & (cyy <= rad * 1.2)
            & (np.sqrt(x ** 2 / 1.8 + cyy ** 2 * 1.8 + z ** 2 / 1.8) <= canopy_r))

    # --- dome's lower surface (cyy_bot ≤ 0) at each (x,z) -----------------
    # Dome equation: (x² + z²)/1.8 + cyy²·1.8 ≤ canopy_r²
    # Lower surface ⇒ cyy_bot = -√((canopy_r² - rxz²/1.8) / 1.8)
    rxz_sq = x ** 2 + z ** 2
    arg = (canopy_r ** 2 - rxz_sq / 1.8) / 1.8
    cyy_bot = -np.sqrt(np.where(arg > 0, arg, 0))      # ≤ 0 everywhere

    # --- footprint / zones ------------------------------------------------
    rim_outer = canopy_r * math.sqrt(1.8)              # max rxz under dome
    band_width = max(1.8, 2.0 * sc)
    in_canopy = rxz <= rim_outer
    in_rim    = in_canopy & (rxz >= rim_outer - band_width)
    # Inner band must clear the trunk fully — willow trunk radius is 1.8·sc,
    # so anything inside that gets hidden by the trunk voxels.
    trunk_r = 1.8 * sc
    in_inner  = in_canopy & ~in_rim & (rxz > trunk_r + 1.5)

    # Two independent hash channels: one for selection, one for length so
    # the length isn't correlated with selection threshold.
    sel_n = _hash(np.floor(x),       np.zeros_like(x),         np.floor(z))
    len_n = _hash(np.floor(x) + 31., np.zeros_like(x) + 7.,    np.floor(z) + 13.)

    rim_drape   = in_rim   & (sel_n > 0.78)            # ~22% of rim columns
    inner_drape = in_inner & (sel_n > 0.90)            # ~10% of inner columns
    has_drape = rim_drape | inner_drape

    # Drape lengths — varied per column so the cascade has natural break-up.
    # Rim drapes can be slightly longer (they start higher on the dome).
    rim_len   = (0.5 + 1.3 * len_n) * rad              # 0.5·rad → 1.8·rad
    inner_len = (0.4 + 1.0 * len_n) * rad              # 0.4·rad → 1.4·rad
    drape_len = np.where(in_rim, rim_len, inner_len)

    # Drape hangs from cyy = 0 (dome's lowest layer) down to cyy_bot - drape_len.
    # We anchor at cyy < 0 instead of "cyy_bot + overlap" because the dome
    # itself only exists for cyy >= -1, so the strand must start from the
    # dome's actual bottom surface (not the analytic cyy_bot which can be
    # several voxels below the real dome geometry in inner regions).
    in_drape_height = (cyy < 0) & (cyy >= cyy_bot - drape_len)
    drape = has_drape & in_drape_height & in_canopy & (y >= 1)

    # Force two dense layers on the dome:
    # 1) dome_bottom_solid: the dome's actual lowest voxel layer (cyy = -1).
    #    This is the surface the drape attaches to. Without it the underside
    #    is sparse (cn<dens noise) and sparse drapes look disconnected.
    # 2) dome_underside: the analytic underside (cyy < cyy_bot + 3) — for
    #    rim regions where cyy_bot ≈ 0 this overlaps with (1); for inner
    #    regions where cyy_bot is much below -1 this is empty (intersection
    #    with dome's cyy >= -1 is empty), so (1) is the one that matters there.
    dome_bottom_solid = dome & (cyy < 0)
    dome_underside = dome & (cyy < cyy_bot + 3.0)
    leaf = (dome & (core | (cn < dens))) | dome_bottom_solid | dome_underside | drape
    leaf = leaf & ~trunk
    return trunk, leaf, np.zeros_like(trunk)


def _socotra(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    rxz = np.sqrt(x ** 2 + z ** 2)
    in_can = (y >= 0) & (cyy >= -2) & (cyy <= rad * 3.5) & (rxz <= rad * 3)
    ang = np.arctan2(z, x)
    m, g = rad * 1.2, rad * 2.4
    frac = np.clip(cyy / m, 0.0, 1.0)
    ring_r = g * np.power(frac, 0.7)
    freq = 10 + math.floor(rad)
    twist = frac * 3.5
    striped = (np.cos(freq * ang + twist) > 0.25) | (np.cos(freq * ang - twist) > 0.25)
    near_ring = np.abs(rxz - ring_r) < 1.8 * sc
    is_branch = striped & near_ring & (cyy >= 0) & (cyy < m * 0.95)
    cap_m, cap_c = m * 0.75, m * 0.9
    cap = (cyy >= cap_m) & ((rxz / g) ** 2 + ((cyy - cap_m) / cap_c) ** 2 <= 1)
    rim = ((cyy >= m * 0.65) & (cyy < cap_m)
           & (rxz <= ring_r + 2.5 * sc) & (rxz >= ring_r - 1.5 * sc))
    trunk = in_can & (((rxz < 2.5 * sc) & (cyy < rad * 0.4)) | is_branch)
    trunk = trunk | (~in_can & (y <= np.floor(th)) & (rxz <= 1.5 * sc))
    leaf = in_can & (cap | rim) & ~trunk & (cn < 0.85)
    return trunk, leaf, np.zeros_like(trunk)


def _baobab(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    e, H = x.shape[0], y.shape[1]
    trunk = np.zeros((e, H, e), dtype=bool)
    leaf = np.zeros((e, H, e), dtype=bool)
    rxz = np.sqrt(x ** 2 + z ** 2)
    # Trunk: thick at the base, tapering toward the top — all radii scaled by sc
    t = np.clip(y / max(th, 1), 0, 1)
    r = np.where(t <= 0.35, (2 + 4 * (t / 0.35)) * sc,
                 np.where(t <= 0.75, (6 - 1.5 * ((t - 0.35) / 0.4)) * sc,
                          (4.5 - 1.3 * ((t - 0.75) / 0.25)) * sc))
    trunk |= (y >= 0) & (y <= th) & (rxz <= r)
    # 6 main branches, each with a leaf cluster at the tip
    spread = rad * 1.45
    crown_y = th * 0.78
    span = max(th - crown_y, 1.0)
    for k in range(6):
        ang = k / 6 * math.tau + k * 0.38
        tt = np.clip((y - crown_y) / span, 0, 1)
        bx = math.cos(ang) * spread * tt
        bz = math.sin(ang) * spread * tt
        thick = np.maximum((2 - tt * 1.3) * sc, 0.7 * sc)
        trunk |= (y >= crown_y) & (y <= th) & (np.sqrt((x - bx) ** 2 + (z - bz) ** 2) <= thick)
        tipx, tipz = math.cos(ang) * spread, math.sin(ang) * spread
        leaf |= (np.sqrt((x - tipx) ** 2 + cyy ** 2 / 1.3 + (z - tipz) ** 2) <= rad * 0.85)
    leaf &= ~trunk & (cn < dens + 0.15)
    return trunk, leaf, np.zeros_like(trunk)


def _saguaro_cactus(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    """Saguaro cactus: central body + two side arms with a clear voxel gap.

    The body and arms are separated by a `gap` of ~1 voxel of empty space so
    the silhouette reads as three distinct columns instead of one fused
    blob. The horizontal elbow tube starts *outside* the body to preserve
    that gap.
    """
    e, H = x.shape[0], y.shape[1]
    leaf = np.zeros((e, H, e), dtype=bool)
    flower = np.zeros((e, H, e), dtype=bool)
    # `rad` already carries scale_xy; `s` is the saguaro's internal radial
    # multiplier (driven by canopy radius). Body / arm thicknesses scale
    # with it; the body↔arm gap stays ≈1 voxel (mild scaling so it remains
    # visible at large sizes).
    s = max(1.0, rad / 5.0)
    body_r = 2.6 * s
    arm_r = 1.9 * s
    gap = max(1.0, 0.8 * sc)              # clear voxel space between body and arm
    arm_x = body_r + gap + arm_r          # arm column center distance

    body_h = 17 * s
    # Central column + rounded top
    leaf |= (x ** 2 + z ** 2 <= body_r ** 2) & (y >= 0) & (y <= body_h)
    leaf |= (x ** 2 + (y - body_h) ** 2 + z ** 2 <= body_r ** 2)

    tops = [(0.0, body_h)]
    # Two arms: low arm on -x, higher arm on +x.
    for (ay, side, uph) in [(5 * s, -1, 12 * s), (8 * s, 1, 15 * s)]:
        # Elbow tube only spans outside the body surface, preserving the gap.
        elbow_inner_x = side * (body_r + gap)
        elbow_outer_x = side * arm_x
        d1, _ = _seg_dist_field(x, z, y,
                                 (elbow_inner_x, ay, 0),
                                 (elbow_outer_x, ay, 0))
        leaf |= (d1 <= arm_r) & (y >= ay - arm_r) & (y <= ay + arm_r)
        # Vertical arm column rises from the elbow.
        leaf |= (np.sqrt((x - elbow_outer_x) ** 2 + z ** 2) <= arm_r) \
                & (y >= ay) & (y <= uph)
        tops.append((elbow_outer_x, uph))

    for (tx, tyv) in tops:
        flower |= (x - tx) ** 2 + (y - tyv - 1.2) ** 2 + z ** 2 <= (1.7 * sc) ** 2
    leaf &= ~flower
    return np.zeros_like(leaf), leaf, flower


def _magnolia(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5 * sc)
    in_can = (y >= th * 0.4) & (np.sqrt(x ** 2 / 2.2 + cyy ** 2 / 1.2 + z ** 2 / 2.2) <= rad * 1.3)
    # Flower threshold scales mildly with sc so larger canopies don't end up
    # carpeted in white blossoms — the canopy voxel count itself grows ~sc³,
    # so a fixed 0.93 threshold would multiply flowers along with size.
    flower_thr = 0.93 + max(0.0, (sc - 1.0)) * 0.03
    flower = in_can & (_hash(x, y, z) > flower_thr) & ~trunk
    leaf = in_can & ~flower & ~trunk & (core | (cn < dens))
    return trunk, leaf, flower


def _palm(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    """Tall, slender trunk with a crown of radiating fronds at the top.

    Frond arc height scales with `sc` (so larger trees have proportionally
    taller crowns) but the drooping term is intentionally damped — at large
    sc both terms scaled together made the fronds plunge way below the
    trunk top, which looks like sagging tentacles rather than palm leaves.

    Trunk radius is deliberately decoupled from `sc`: real palm trunks stay
    slender even as the tree gets taller. We only nudge it up a touch at
    larger versions so the cross-section doesn't look one-voxel-thin
    against a much wider canopy. A per-tree truncated-normal multiplier
    `tm` gives small individual variation without making any palm chunky.
    """
    e, H = x.shape[0], y.shape[1]
    tm = _truncated_normal(npr, 0.88, 1.12)
    trunk_r = (1.6 + 0.3 * max(0.0, sc - 1.0)) * tm
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y >= 0) & (y <= th) & (rxz <= trunk_r)

    leaf = np.zeros((e, H, e), dtype=bool)
    n_fronds = 9
    L = rad * 2.0
    # Drooping scales mildly with sc (sqrt) instead of linearly, so v5 fronds
    # don't sag dramatically more than v2 fronds.
    droop_scale = math.sqrt(sc)
    for i in range(n_fronds):
        ang = 2.0 * math.pi * i / n_fronds
        ca, sa = math.cos(ang), math.sin(ang)
        for s in range(16):
            tt = s / 15.0
            rr = L * tt
            frond_x = ca * rr
            frond_z = sa * rr
            # Arc up: scales fully with sc (taller crown at larger versions)
            # Droop:  scales only with sqrt(sc) (limits sag at larger versions)
            frond_y = th + 4.5 * sc * math.sin(tt * 2.3) - 6.0 * droop_scale * tt * tt
            sphere_r = max(0.5 * sc, (1.7 - 0.7 * tt) * sc)
            leaf |= ((x - frond_x) ** 2 + (y - frond_y) ** 2 + (z - frond_z) ** 2
                     <= sphere_r ** 2)
    leaf &= ~trunk
    return trunk, leaf, np.zeros_like(trunk)


def _acacia(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr, sc):
    """Acacia tortilis (umbrella thorn): wide canopy with a concave underside.

    Real flat-top acacias have a near-flat top and a hollow / concave bottom
    — the silhouette reads as an umbrella, not a dome. We model that by
    subtracting an inner ellipsoid shifted *down* relative to the outer one,
    so it carves the bottom-center of the canopy. Rim thickness stays full,
    center underside curves upward.
    """
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y >= 0) & (y <= th) & (rxz <= 1.8 * sc)

    # Outer canopy: wide, flat ellipsoid sitting over the trunk tip.
    thick = max(2.0 * sc, rad * 0.6)
    R = rad * 2.0
    cy = th + thick - 4.0 * sc       # lowered to overlap the trunk
    in_outer = (x ** 2 / (R * R) + (y - cy) ** 2 / (thick * thick)
                + z ** 2 / (R * R)) <= 1.0

    # Inner carve-out: smaller ellipsoid shifted down so its upper half
    # punches through the outer canopy's underside, leaving a concave bowl.
    R_in = R * 0.80
    thick_in = thick * 0.95
    cy_in = cy - thick * 0.55
    in_inner = (x ** 2 / (R_in * R_in) + (y - cy_in) ** 2 / (thick_in * thick_in)
                 + z ** 2 / (R_in * R_in)) <= 1.0

    canopy = in_outer & ~in_inner & (y >= cy - thick * 0.95)
    leaf = canopy & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


_SPECIES = {
    "cherryblossom": _cherry, "pine": _pine, "socotra": _socotra, "maple": _maple,
    "baobab": _baobab, "willow": _willow, "magnolia": _magnolia,
    "saguaro_cactus": _saguaro_cactus, "palm": _palm, "acacia": _acacia,
}


# ===========================================================================
# Tree build + augmentation
# ===========================================================================
# Per-species augmentation ranges — structural species (shape driven by QR footprint) have no meaningful density variation.
_STRUCTURAL = {"socotra", "baobab", "saguaro_cactus"}


def _aug_range(theme):
    """Returns (h_lo,h_hi, r_lo,r_hi, d_lo,d_hi, salt) — per-species ranges for height, radius, density, and coordinate salt."""
    if theme in _STRUCTURAL:
        return (0.68, 1.50, 0.72, 1.42, 1.0, 1.0, 0)
    if theme == "palm":
        # Palm has the sparsest, thinnest target (long slim trunk + 9 radiating
        # fronds at the top). Wide jitter was making the diffusion model spend
        # its capacity chasing many palm variants and falling into a "trunk
        # only" local minimum at v4/v5. Narrow the variance so the canonical
        # palm shape dominates the training signal.
        return (0.92, 1.08, 0.92, 1.08, 0.85, 1.15, 2)
    return (0.75, 1.42, 0.80, 1.35, 0.65, 1.40, 4)


def build_tree(theme, e, H, npr, augment=True):
    """Species -> (trunk, leaf, flower) bool masks.

    augment=True randomizes height, fullness, radius, and coordinate salt within per-species ranges (for training).
    augment=False uses fixed values, producing the canonical shape (for evaluation / ground truth).
    """
    theme = theme if theme in _SPECIES else "cherryblossom"
    spec = THEMES[theme]
    x, y, z = _axes(e, H)
    # Both XY and Z grow with QR version, but Z grows slightly faster
    # (×1.25) so taller trees look proportional rather than squat in larger
    # codes. The grid Z (qr.grid_z_for_version) is computed with the same
    # multiplier built-in, so trunks stay within the grid (clamped to H-4).
    scale_xy = max(1.0, e / 21.0)
    scale_z = scale_xy * 1.25

    # --- augmentation: jitter radius, trunk height, density, and
    # coordinate salt within per-species ranges. Uses truncated-normal
    # sampling (mean at midpoint, std = range/4) so most trees cluster
    # near the species canonical shape; extreme jitter is rare.
    h_lo, h_hi, r_lo, r_hi, d_lo, d_hi, salt = _aug_range(theme)
    if augment:
        hm = _truncated_normal(npr, h_lo, h_hi)
        rm = _truncated_normal(npr, r_lo, r_hi)
        dm = _truncated_normal(npr, d_lo, d_hi)
        sx = int(round(_truncated_normal(npr, -salt, salt))) if salt else 0
        sy = int(round(_truncated_normal(npr, -salt, salt))) if salt else 0
        sz = int(round(_truncated_normal(npr, -salt, salt))) if salt else 0
    else:
        hm = rm = dm = 1.0
        sx = sy = sz = 0

    radius = max(2.0, math.floor(5 * scale_xy)) * rm
    th0, bnd = _th_bn(theme, scale_xy, scale_z)
    # Cap th so the trunk + crown fit inside H. Leave a small headroom for
    # leaves that sit above the trunk tip.
    th = max(1.0, min(th0 * hm, H - 4))
    dens = spec["density"] * dm

    cyy = y - th
    cs = spec["cluster"]
    cn = _hash(np.floor(x / cs) + sx, np.floor(y / cs) + sy, np.floor(z / cs) + sz)
    core = np.sqrt(x ** 2 + cyy ** 2 + z ** 2) < radius * 0.4

    return _SPECIES[theme](x, y, z, cyy, th, radius, bnd, cn, core, dens, npr,
                            scale_xy)


def _gen(qr, theme, rng, augment):
    """QR plane + tree above it -> voxel list.

    Tree voxels are kept only above dark QR modules (keeps the code scannable from above). The column above white modules is left empty.
    voxel = {"pos":(x,y,z), "color":hex, "qr_color":hex, "scale":1.0, "is_base":bool}
    """
    if rng is None:
        rng = random.Random(0)
    npr = np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
    theme = theme if theme in _SPECIES else "cherryblossom"
    spec = THEMES[theme]
    e = len(qr)
    # H scales isotropically with the QR footprint (qr.grid_z_for_version);
    # legacy callers that pre-date that helper still get sensible behavior
    # via H = max(32, e * 32/21).
    import math as _m
    H = max(32, ((_m.ceil(e * 32 / 21 / 4)) * 4))

    trunk, leaf, flower = build_tree(theme, e, H, npr, augment)
    dark = np.array([[bool(qr[r][c]) for c in range(e)] for r in range(e)], dtype=bool)
    keep = dark.T[:, None, :]                 # Only above dark modules (keeps QR scannable)
    trunk, leaf, flower = trunk & keep, leaf & keep, flower & keep
    for m in (trunk, leaf, flower):
        m[:, 0, :] = False                    # y=0 is reserved for the QR plane

    cx = e // 2
    leaf_p = spec["leaf"]
    flower_p = spec.get("flower", leaf_p)
    qd = spec["qr_dark"]
    vox = []
    for z in range(e):
        for x in range(e):
            col = qd if dark[z][x] else spec["qr_light"]
            vox.append({"pos": (x - cx, 0, z - cx), "color": col,
                        "qr_color": col, "scale": 1.0, "is_base": True})
    for (x, yy, z) in np.argwhere(trunk):
        vox.append({"pos": (int(x) - cx, int(yy), int(z) - cx),
                    "color": spec["trunk"], "qr_color": qd,
                    "scale": 1.0, "is_base": False})
    for (x, yy, z) in np.argwhere(leaf & ~trunk):
        vox.append({"pos": (int(x) - cx, int(yy), int(z) - cx),
                    "color": leaf_p[int(npr.integers(len(leaf_p)))],
                    "qr_color": qd, "scale": 1.0, "is_base": False})
    for (x, yy, z) in np.argwhere(flower & ~trunk & ~leaf):
        vox.append({"pos": (int(x) - cx, int(yy), int(z) - cx),
                    "color": flower_p[int(npr.integers(len(flower_p)))],
                    "qr_color": qd, "scale": 1.0, "is_base": False})
    return vox


def generate_voxels(qr, theme="cherryblossom", rng=None):
    """Canonical form — no augmentation (for evaluation and ground truth)."""
    return _gen(qr, theme, rng, augment=False)


def generate_voxels_aug(qr, theme="cherryblossom", rng=None):
    """Training mode — applies per-species augmentation (same QR produces a slightly different tree each time)."""
    return _gen(qr, theme, rng, augment=True)


THEME_NAMES = list(THEMES)

# Tree attribute measurement — used as attribute signals during training
# (height / fullness / spread), each normalized to [0, 1].
# Height is normalized against the fixed Z extent (30 ≈ usable canopy span).
# Fullness and spread scale with QR side length so the signal stays in [0,1]
# regardless of QR version.
_ATTR_HEIGHT_NORM = 30.0


def tree_attributes(voxels, qr_side: int | None = None):
    """Voxel list -> [height, fullness, spread] in [0, 1].

    qr_side: side length of the QR matrix (modules) that the tree was built
        on. If None, defaults to 21 (v1) for backward compatibility.
    """
    tv = [v for v in voxels if not v["is_base"]]
    if not tv:
        return [0.0, 0.0, 0.0]
    if qr_side is None:
        qr_side = 21
    ys = [v["pos"][1] for v in tv]
    xs = [v["pos"][0] for v in tv]
    zs = [v["pos"][2] for v in tv]
    # Fullness normalization grows quadratically with footprint (≈ canopy area).
    # Spread normalization grows linearly with footprint.
    full_norm = 1800.0 * (qr_side / 21.0) ** 2
    spread_norm = max(2.0, qr_side - 0.0)
    height = max(ys) / _ATTR_HEIGHT_NORM
    fullness = len(tv) / full_norm
    spread = max(max(xs) - min(xs), max(zs) - min(zs)) / spread_norm
    return [min(1.0, max(0.0, a)) for a in (height, fullness, spread)]
