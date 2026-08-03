"""Fast colour classification.

The hot path runs once per captured frame, so it avoids per-frame HSV conversion
entirely. Instead the HSV gates are baked into a 32768-entry lookup table keyed on
the top 5 bits of each BGR channel. Classification then costs one gather plus one
bincount over the subsampled strip.
"""

from __future__ import annotations

import cv2
import numpy as np

RED_BIT = 1
BLUE_BIT = 2


def build_lut(cfg) -> np.ndarray:
    """Bake the HSV colour rules into a (32768,) uint8 bitmask table.

    Index layout is (r>>3)<<10 | (g>>3)<<5 | (b>>3), matching `classify`.
    """
    idx = np.arange(32768, dtype=np.uint32)
    r = ((idx >> 10) & 31).astype(np.uint8) * 8 + 4  # bin centres
    g = ((idx >> 5) & 31).astype(np.uint8) * 8 + 4
    b = (idx & 31).astype(np.uint8) * 8 + 4

    rgb = np.stack([r, g, b], axis=1).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    lut = np.zeros(32768, dtype=np.uint8)
    lut[cfg.red.mask(h, s, v)] |= RED_BIT
    lut[cfg.blue.mask(h, s, v)] |= BLUE_BIT
    return lut


def classify(strip_bgr: np.ndarray, lut: np.ndarray, step: int = 3) -> tuple[int, int]:
    """Count red and blue glyph pixels in a BGR strip.

    Subsamples by `step` in both axes first. The notes are large blobs, so
    throwing away 8 of every 9 pixels costs nothing in separability but cuts the
    work by an order of magnitude.
    """
    s = strip_bgr[::step, ::step]
    b = s[:, :, 0].astype(np.uint32)
    g = s[:, :, 1].astype(np.uint32)
    r = s[:, :, 2].astype(np.uint32)

    idx = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
    counts = np.bincount(lut[idx].ravel(), minlength=4)
    # bit 0 = red, bit 1 = blue; value 3 means both gates matched the same pixel.
    return int(counts[1] + counts[3]), int(counts[2] + counts[3])
