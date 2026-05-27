"""QR-code utilities: random QR generation and version-aware grid helpers.

The diffusion model supports arbitrary QR versions (1..40). XY and Z both
scale with the version (isotropic): bigger QR ⇒ wider footprint *and* taller
voxel grid, so trees grow in proportion rather than getting squashed.

QR module count per side = 4 * version + 17 (ISO/IEC 18004).
Voxel grid for version v:
  grid_xy(v) = ceil(qr_modules(v) / 4) * 4
  grid_z(v)  = ceil(32 * scale(v)  / 4) * 4    # scale(v) = qr_modules(v)/21
The U-Net is fully convolutional in all three spatial dims, so a single model
handles any version whose (XY, Z) are multiples of XY_MULTIPLE.
"""
import string

import numpy as np
import segno

from qrbloom.treegen import THEMES

# ---------------------------------------------------------------------------
# Version-aware constants
# ---------------------------------------------------------------------------
BASE_Z_AT_V1 = 32       # Z extent of the v1 grid; higher versions scale up
XY_MULTIPLE = 4         # U-Net has 2 downsampling stages -> XY/Z must be multiple of 4
Z_MULTIPLE = 4
MIN_VERSION = 1
MAX_VERSION = 40
DEFAULT_VERSION = 1     # legacy default; tools that want v1 keep working

THEME_NAMES = list(THEMES)

_ALNUM = list(string.ascii_letters + string.digits)


def qr_modules(version: int) -> int:
    """Modules per side for a QR code of the given version."""
    if not (MIN_VERSION <= version <= MAX_VERSION):
        raise ValueError(f"QR version must be in [{MIN_VERSION},{MAX_VERSION}], got {version}")
    return 4 * version + 17


def grid_xy_for_version(version: int) -> int:
    """Smallest XY grid edge >= module count, rounded up to XY_MULTIPLE."""
    m = qr_modules(version)
    return ((m + XY_MULTIPLE - 1) // XY_MULTIPLE) * XY_MULTIPLE


def version_scale(version: int) -> float:
    """Linear scale factor relative to v1 (e.g. v3 -> 29/21 ≈ 1.381)."""
    return qr_modules(version) / qr_modules(1)


def grid_z_for_version(version: int) -> int:
    """Z (tree-height) extent that scales isotropically with the XY footprint.

    The voxel grid is (grid_xy, grid_xy, grid_z); both dimensions grow with
    the QR version so trees keep their natural proportions instead of
    getting squashed in larger codes. Rounded up to Z_MULTIPLE (= 4) so
    the U-Net can downsample cleanly.
    """
    import math as _math
    s = version_scale(version)
    raw = _math.ceil(BASE_Z_AT_V1 * s)
    return ((raw + Z_MULTIPLE - 1) // Z_MULTIPLE) * Z_MULTIPLE


def grid_shape_for_version(version: int) -> tuple[int, int, int]:
    """(grid_xy, grid_xy, grid_z) — full voxel grid shape for the version."""
    xy = grid_xy_for_version(version)
    z = grid_z_for_version(version)
    return (xy, xy, z)


# QR_OFFSET: kept for evaluate.py which still imports it.
# (Other v1 aliases — QR_VERSION, QR_SIZE, GRID, QR_CENTER — were removed
# along with pad32; new code should call grid_xy_for_version()/pad_to_grid()
# directly with an explicit version.)
QR_OFFSET = (grid_xy_for_version(1) - qr_modules(1)) // 2


# Approximate alphanumeric capacity at error level M (segno will reject if too
# long; this just gives a reasonable upper bound for random text length).
# Values from ISO/IEC 18004 table for error-correction level M.
_ALNUM_CAP_M = {
    1: 20, 2: 38, 3: 61, 4: 90, 5: 122, 6: 154, 7: 178, 8: 221, 9: 262,
    10: 311, 11: 366, 12: 419, 13: 483, 14: 528, 15: 600, 16: 656, 17: 734,
    18: 816, 19: 909, 20: 970, 25: 1336, 30: 1759, 35: 2257, 40: 2812,
}


def _alnum_capacity(version: int) -> int:
    if version in _ALNUM_CAP_M:
        return _ALNUM_CAP_M[version]
    # Linear interpolation between known points (good enough as an upper bound)
    keys = sorted(_ALNUM_CAP_M.keys())
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= version <= hi:
            t = (version - lo) / (hi - lo)
            return int(_ALNUM_CAP_M[lo] + t * (_ALNUM_CAP_M[hi] - _ALNUM_CAP_M[lo]))
    return _ALNUM_CAP_M[40]


def random_qr_core(rng, version: int = DEFAULT_VERSION) -> np.ndarray:
    """Encode random alphanumeric text into a QR matrix of the given version.

    Returns a (M, M) uint8 array where M = 4*version + 17. `rng` is a numpy
    random Generator, so a fixed seed reproduces the same code.
    """
    cap = _alnum_capacity(version)
    lo = max(4, cap // 4)
    hi = max(lo + 1, cap)
    for _ in range(64):
        length = int(rng.integers(lo, hi))
        text = "".join(rng.choice(_ALNUM, length))
        try:
            qr = segno.make(text, error="m", version=version)
            return np.array(qr.matrix, dtype=np.uint8)
        except Exception:
            continue
    # Fallback: shortest possible string for this version
    qr = segno.make("A", error="m", version=version)
    return np.array(qr.matrix, dtype=np.uint8)


def pad_to_grid(core: np.ndarray, grid_xy: int | None = None) -> tuple[np.ndarray, int]:
    """Center a QR core in an XY grid. Returns (padded, offset)."""
    m = core.shape[0]
    if grid_xy is None:
        version = (m - 17) // 4
        grid_xy = grid_xy_for_version(version)
    if grid_xy < m:
        raise ValueError(f"grid_xy={grid_xy} smaller than QR module count {m}")
    full = np.zeros((grid_xy, grid_xy), dtype=np.uint8)
    off = (grid_xy - m) // 2
    full[off:off + m, off:off + m] = core
    return full, off


def random_qr(rng, version: int = DEFAULT_VERSION) -> np.ndarray:
    """Random QR core padded to the grid for that version."""
    core = random_qr_core(rng, version=version)
    full, _ = pad_to_grid(core)
    return full


# ---------------------------------------------------------------------------
# Legacy helper — preserved for external scripts (evaluate.py, gallery.py)
# ---------------------------------------------------------------------------
