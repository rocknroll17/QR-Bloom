"""treegen.py — QR-conditioned voxel tree generator (10 species).

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


def _th_bn(theme, scale):
    """Returns (trunk_height, search_limit) per species. scale = proportional multiplier for QR version."""
    fl = math.floor
    table = {
        "cherryblossom":  (fl(9 * scale), fl(10 * scale)),
        "pine":           (fl(7 * scale), fl(8 * scale)),
        "socotra":        (fl(7 * scale) + fl(5 * scale), fl(10 * scale)),
        "maple":          (fl(7 * scale), fl(8 * scale)),
        "baobab":         (fl(10 * scale) + fl(7 * scale), fl(8 * scale)),
        "willow":         (fl(7 * scale) + fl(3 * scale), fl(10 * scale)),
        "magnolia":       (fl(7 * scale), fl(10 * scale)),
        "saguaro_cactus": (1, fl(10 * scale)),
        "palm":           (fl(13 * scale), fl(12 * scale)),
        "acacia":         (fl(9 * scale), fl(10 * scale)),
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


def _seg_dist_field(x, z, y, p0, p1):
    """Planar distance from voxel (x, z) to the cross-section point of segment p0->p1 at height y."""
    y0, y1 = p0[1], p1[1]
    t = np.clip((y - y0) / max(y1 - y0, 1e-6), 0.0, 1.0)
    bx = p0[0] + (p1[0] - p0[0]) * t
    bz = p0[2] + (p1[2] - p0[2]) * t
    return np.sqrt((x - bx) ** 2 + (z - bz) ** 2), t


# ===========================================================================
# Per-species shape builders — return (trunk, leaf, flower) bool masks
#   x,z: center-aligned planar coords  y: height  cyy: y - trunk_height  rad: canopy radius
#   cn: cluster noise  core: trunk core region  dens: density (after augmentation)
# ===========================================================================
def _cherry(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5)
    bump = np.where(_hash(x, y, z) > 0.95, 1.5, 0.0)
    shape = np.sqrt(x ** 2 / 2.5 + cyy ** 2 / 0.5 + z ** 2 / 2.5) <= rad * 1.3 + bump
    leaf = (y >= th * 0.4) & shape & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


def _pine(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5)
    cone = bnd * 2.2
    shape = (cyy >= -1) & (cyy < cone) & (rxz <= np.maximum(cone - cyy, 0) * 0.45)
    leaf = shape & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


def _maple(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5)
    shape = np.sqrt(x ** 2 / 2.5 + cyy ** 2 + z ** 2 / 2.5) <= rad
    leaf = (y >= th * 0.4) & shape & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


def _willow(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.8)
    g = _hash(np.floor(x / 2), np.floor(y / 2), np.floor(z / 2))
    canopy_r = rad * 1.6 + np.where(g > 0.5, 1.5, -1.5)
    dome = ((cyy >= -1) & (cyy <= rad * 1.2)
            & (np.sqrt(x ** 2 / 1.8 + cyy ** 2 * 1.8 + z ** 2 / 1.8) <= canopy_r))
    v = _hash(x, np.zeros_like(x) + 0.0, z)
    drape_len = rad * 1.2 + v * rad * 2.5
    drape = ((cyy < 0) & (cyy >= -drape_len) & (v < 0.2)
             & (np.sqrt(x ** 2 / 1.5 + z ** 2 / 1.5) <= rad * 1.4))
    leaf = (dome | drape) & ~trunk & (core | (cn < dens))
    return trunk, leaf, np.zeros_like(trunk)


def _socotra(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    rxz = np.sqrt(x ** 2 + z ** 2)
    in_can = (y >= 0) & (cyy >= -2) & (cyy <= rad * 3.5) & (rxz <= rad * 3)
    ang = np.arctan2(z, x)
    m, g = rad * 1.2, rad * 2.4
    frac = np.clip(cyy / m, 0.0, 1.0)
    ring_r = g * np.power(frac, 0.7)
    freq = 10 + math.floor(rad)
    twist = frac * 3.5
    striped = (np.cos(freq * ang + twist) > 0.25) | (np.cos(freq * ang - twist) > 0.25)
    near_ring = np.abs(rxz - ring_r) < 1.8
    is_branch = striped & near_ring & (cyy >= 0) & (cyy < m * 0.95)
    cap_m, cap_c = m * 0.75, m * 0.9
    cap = (cyy >= cap_m) & ((rxz / g) ** 2 + ((cyy - cap_m) / cap_c) ** 2 <= 1)
    rim = ((cyy >= m * 0.65) & (cyy < cap_m)
           & (rxz <= ring_r + 2.5) & (rxz >= ring_r - 1.5))
    trunk = in_can & (((rxz < 2.5) & (cyy < rad * 0.4)) | is_branch)
    trunk = trunk | (~in_can & (y <= np.floor(th)) & (rxz <= 1.5))
    leaf = in_can & (cap | rim) & ~trunk & (cn < 0.85)
    return trunk, leaf, np.zeros_like(trunk)


def _baobab(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    e, H = x.shape[0], y.shape[1]
    trunk = np.zeros((e, H, e), dtype=bool)
    leaf = np.zeros((e, H, e), dtype=bool)
    rxz = np.sqrt(x ** 2 + z ** 2)
    # Trunk: thick at the base, tapering toward the top
    t = np.clip(y / max(th, 1), 0, 1)
    r = np.where(t <= 0.35, 2 + 4 * (t / 0.35),
                 np.where(t <= 0.75, 6 - 1.5 * ((t - 0.35) / 0.4),
                          4.5 - 1.3 * ((t - 0.75) / 0.25)))
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
        thick = np.maximum(2 - tt * 1.3, 0.7)
        trunk |= (y >= crown_y) & (y <= th) & (np.sqrt((x - bx) ** 2 + (z - bz) ** 2) <= thick)
        tipx, tipz = math.cos(ang) * spread, math.sin(ang) * spread
        leaf |= (np.sqrt((x - tipx) ** 2 + cyy ** 2 / 1.3 + (z - tipz) ** 2) <= rad * 0.85)
    leaf &= ~trunk & (cn < dens + 0.15)
    return trunk, leaf, np.zeros_like(trunk)


def _saguaro_cactus(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    e, H = x.shape[0], y.shape[1]
    leaf = np.zeros((e, H, e), dtype=bool)
    flower = np.zeros((e, H, e), dtype=bool)
    s = max(1.0, rad / 5.0)
    body_h = 17 * s
    # Central column + rounded top
    leaf |= (x ** 2 + z ** 2 <= (2.6 * s) ** 2) & (y >= 0) & (y <= body_h)
    leaf |= (x ** 2 + (y - body_h) ** 2 + z ** 2 <= (2.6 * s) ** 2)
    tops = [(0.0, body_h)]
    for (ay, side, uph) in [(5 * s, -1, 12 * s), (8 * s, 1, 15 * s)]:
        ar = 1.9 * s
        # Elbow: extends horizontally then turns upward
        d1, _ = _seg_dist_field(x, z, y, (0, ay, 0), (side * 4 * s, ay, 0))
        leaf |= (d1 <= ar) & (y >= ay - ar) & (y <= ay + ar)
        leaf |= (np.sqrt((x - side * 4 * s) ** 2 + z ** 2) <= ar) & (y >= ay) & (y <= uph)
        tops.append((side * 4 * s, uph))
    for (tx, tyv) in tops:
        flower |= (x - tx) ** 2 + (y - tyv - 1.2) ** 2 + z ** 2 <= 1.7 ** 2
    leaf &= ~flower
    return np.zeros_like(leaf), leaf, flower


def _magnolia(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y <= th) & (rxz <= 1.5)
    in_can = (y >= th * 0.4) & (np.sqrt(x ** 2 / 2.2 + cyy ** 2 / 1.2 + z ** 2 / 2.2) <= rad * 1.3)
    flower = in_can & (_hash(x, y, z) > 0.93) & ~trunk
    leaf = in_can & ~flower & ~trunk & (core | (cn < dens))
    return trunk, leaf, flower


def _palm(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    """Tall, slender trunk with a crown of radiating fronds at the top."""
    e, H = x.shape[0], y.shape[1]
    trunk_r = 1.9
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y >= 0) & (y <= th) & (rxz <= trunk_r)

    # Leaves: n_fronds radiating branches, each drooping in an arc
    leaf = np.zeros((e, H, e), dtype=bool)
    n_fronds = 9
    L = rad * 2.0          # Frond length proportional to radius
    for i in range(n_fronds):
        ang = 2.0 * math.pi * i / n_fronds
        ca, sa = math.cos(ang), math.sin(ang)
        for s in range(16):
            tt = s / 15.0
            rr = L * tt
            frond_x = ca * rr
            frond_z = sa * rr
            # Each frond arcs upward first, then droops toward the tip
            frond_y = th + 4.5 * math.sin(tt * 2.3) - 6.0 * tt * tt
            sphere_r = max(0.5, 1.7 - 0.7 * tt)
            leaf |= ((x - frond_x) ** 2 + (y - frond_y) ** 2 + (z - frond_z) ** 2
                     <= sphere_r ** 2)
    leaf &= ~trunk
    return trunk, leaf, np.zeros_like(trunk)


def _acacia(x, y, z, cyy, th, rad, bnd, cn, core, dens, npr):
    """Tall trunk with a flat, wide umbrella-shaped canopy at the top.

    cy = th + thick - 4.0 offsets the canopy down so it overlaps the trunk tip.
    """
    rxz = np.sqrt(x ** 2 + z ** 2)
    trunk = (y >= 0) & (y <= th) & (rxz <= 1.8)

    # Umbrella disc: wide (R) and flat (thick) ellipsoid, overlapping the trunk top
    thick = max(2.0, rad * 0.6)
    R = rad * 2.0
    cy = th + thick - 4.0           # Lowered to overlap the trunk and keep it connected
    # Ellipsoid: (x/R)^2 + ((y-cy)/thick)^2 + (z/R)^2 <= 1
    in_ellip = (x ** 2 / (R * R) + (y - cy) ** 2 / (thick * thick)
                + z ** 2 / (R * R)) <= 1.0
    # Flatten the bottom — clip the lower face so it reaches the trunk
    canopy = in_ellip & (y >= cy - thick * 0.8)
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
    return (0.75, 1.42, 0.80, 1.35, 0.65, 1.40, 4)


def build_tree(theme, e, H, npr, augment=True):
    """Species -> (trunk, leaf, flower) bool masks.

    augment=True randomizes height, fullness, radius, and coordinate salt within per-species ranges (for training).
    augment=False uses fixed values, producing the canonical shape (for evaluation / ground truth).
    """
    theme = theme if theme in _SPECIES else "cherryblossom"
    spec = THEMES[theme]
    x, y, z = _axes(e, H)
    scale = max(1.0, e / 21.0)

    # --- augmentation: jitter radius, trunk height, density, and coordinate salt within per-species ranges ---
    h_lo, h_hi, r_lo, r_hi, d_lo, d_hi, salt = _aug_range(theme)
    if augment:
        hm = h_lo + (h_hi - h_lo) * npr.random()
        rm = r_lo + (r_hi - r_lo) * npr.random()
        dm = d_lo + (d_hi - d_lo) * npr.random()
        sx = int(round((npr.random() - 0.5) * 2 * salt))
        sy = int(round((npr.random() - 0.5) * 2 * salt))
        sz = int(round((npr.random() - 0.5) * 2 * salt))
    else:
        hm = rm = dm = 1.0
        sx = sy = sz = 0

    radius = max(2.0, math.floor(5 * scale)) * rm
    th0, bnd = _th_bn(theme, scale)
    th = max(1.0, th0 * hm)
    dens = spec["density"] * dm

    cyy = y - th
    cs = spec["cluster"]
    cn = _hash(np.floor(x / cs) + sx, np.floor(y / cs) + sy, np.floor(z / cs) + sz)
    core = np.sqrt(x ** 2 + cyy ** 2 + z ** 2) < radius * 0.4

    return _SPECIES[theme](x, y, z, cyy, th, radius, bnd, cn, core, dens, npr)


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
    e, H = len(qr), 32

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

# Tree attribute measurement — used as attribute signals during training (height / fullness / spread).
_ATTR_NORM = {"height": 30.0, "fullness": 1800.0, "spread": 22.0}


def tree_attributes(voxels):
    """Voxel list -> [height, fullness, spread], each normalized to [0, 1]."""
    tv = [v for v in voxels if not v["is_base"]]
    if not tv:
        return [0.0, 0.0, 0.0]
    ys = [v["pos"][1] for v in tv]
    xs = [v["pos"][0] for v in tv]
    zs = [v["pos"][2] for v in tv]
    height = max(ys) / _ATTR_NORM["height"]
    fullness = len(tv) / _ATTR_NORM["fullness"]
    spread = max(max(xs) - min(xs), max(zs) - min(zs)) / _ATTR_NORM["spread"]
    return [min(1.0, max(0.0, a)) for a in (height, fullness, spread)]
