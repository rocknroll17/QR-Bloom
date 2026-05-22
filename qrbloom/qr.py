"""QR-code utilities: random QR generation, padding, and grid constants."""
import string

import numpy as np
import segno

from qrbloom.treegen import THEMES

GRID = 32                           # voxel grid edge length
QR_VERSION = 1                      # QR code version
QR_SIZE = 21                        # modules per side for a version-1 code
QR_OFFSET = (GRID - QR_SIZE) // 2   # offset that centers the code in the grid
QR_CENTER = QR_SIZE // 2
THEME_NAMES = list(THEMES)          # available tree-species themes

_ALNUM = list(string.ascii_letters + string.digits)


def random_qr_core(rng):
    """Encode a random short alphanumeric string into a QR matrix.

    Returns a (QR_SIZE, QR_SIZE) uint8 array of 0/1 modules. `rng` is a
    numpy random Generator, so a fixed seed reproduces the same code.
    """
    while True:
        length = int(rng.integers(6, 17))
        text = "".join(rng.choice(_ALNUM, length))
        try:
            qr = segno.make(text, error="m", version=QR_VERSION)
        except Exception:
            continue
        return np.array(qr.matrix, dtype=np.uint8)


def pad32(core):
    """Center a QR core inside a GRID x GRID uint8 array."""
    full = np.zeros((GRID, GRID), dtype=np.uint8)
    e = core.shape[0]
    full[QR_OFFSET:QR_OFFSET + e, QR_OFFSET:QR_OFFSET + e] = core
    return full


def random_qr(rng):
    """Random QR core padded to the full GRID x GRID grid."""
    return pad32(random_qr_core(rng))
