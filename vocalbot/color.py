"""Fast colour classification.

The hot path runs once per captured frame, so it avoids per-frame HSV conversion
entirely. The HSV gates are baked into a 32768-entry lookup table keyed on the
top 5 bits of each BGR channel; classification is then one gather plus one
bincount over the subsampled region.
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


def classify(region_bgr: np.ndarray, lut: np.ndarray, step: int = 3) -> tuple[int, int]:
    """Count red and blue glyph pixels in a BGR region.

    Subsamples by `step` in both axes first. Notes are large blobs, so throwing
    away 8 of every 9 pixels costs nothing in separability but cuts the work by
    an order of magnitude.
    """
    s = region_bgr[::step, ::step]
    if s.size == 0:
        return 0, 0
    b = s[:, :, 0].astype(np.uint32)
    g = s[:, :, 1].astype(np.uint32)
    r = s[:, :, 2].astype(np.uint32)

    idx = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
    counts = np.bincount(lut[idx].ravel(), minlength=4)
    # bit 0 = red, bit 1 = blue; value 3 means both gates matched one pixel.
    return int(counts[1] + counts[3]), int(counts[2] + counts[3])


def fan_out(by_rect: dict, rect_groups: dict) -> dict[str, tuple[int, int]]:
    """Spread one count per rect across every zone sharing that rect."""
    return {name: by_rect[rect] for rect, names in rect_groups.items() for name in names}


class ZoneReader:
    """Slices a captured frame into zones and counts each one.

    Zone rects are fractions of the phone screen, so the reader needs the phone's
    pixel size and where the captured frame sits within it. For a full-screen
    image that origin is (0, 0); for the live loop it's the union rect's corner.

    Zones sharing a rect are counted once, not once each.
    """

    def __init__(self, cfg, screen_w: int, screen_h: int, origin=(0, 0)):
        self.cfg = cfg
        self.lut = build_lut(cfg)
        self.step = cfg.step
        self.rect_groups = cfg.rect_groups()
        ox, oy = origin
        self.slices = {}
        for rect, names in self.rect_groups.items():
            zone = cfg.zone(names[0])
            x0, y0, x1, y1 = zone.pixel_rect(screen_w, screen_h, -ox, -oy)
            self.slices[rect] = (slice(max(0, y0), max(0, y1)), slice(max(0, x0), max(0, x1)))

    def read(self, frame_bgr: np.ndarray) -> dict[str, tuple[int, int]]:
        by_rect = {
            rect: classify(frame_bgr[ys, xs], self.lut, self.step)
            for rect, (ys, xs) in self.slices.items()
        }
        return fan_out(by_rect, self.rect_groups)
