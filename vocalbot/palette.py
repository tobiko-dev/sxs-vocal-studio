"""Classifying notes by hue family.

The notes come in many variants - plain, beamed, dotted, with lightning bolts -
and each family covers a wide range of shades. Blue runs from dark navy to pale
cyan; red runs from crimson through magenta to purple. Those are far apart in
RGB, which is why matching against a single clicked colour per family kept
failing: one sample cannot cover navy *and* cyan.

What they do share is hue. Measured over 5.1M strongly-coloured pixels from a
gameplay recording, the two families sit in separate hue bands with an empty
corridor between them:

    hue  10- 24   desk and the lightning-bolt accents   (ignored)
    hue  85-118   blue family, navy through cyan        1.27M px
    hue 119-129   ~400 px in total - effectively empty
    hue 130-179   red family, purple through crimson    2.5M px
    hue   0-  9   the red wrap-around

So classification is by hue band, which is invariant to how light or dark a
variant happens to be. Saturation and value only gate out the pale bubble and
the background; they play no part in deciding which family a pixel belongs to.

Setup still samples the colours you point at, but they are used to *verify* the
bands - confirming your blue really lands in the blue band, and widening a band
if your display shifts a hue slightly - rather than as nearest-colour anchors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DRUMS = ("red", "blue")


@dataclass
class Sample:
    """One colour you pointed at during setup. Kept for reference and checking."""

    name: str
    rgb: tuple[int, int, int]  # BGR, matching OpenCV
    taps: tuple[str, ...]
    point: tuple[int, int] | None = None

    def __post_init__(self):
        bad = [d for d in self.taps if d not in DRUMS]
        if bad:
            raise ValueError(f"{self.name}: unknown drum(s) {bad}")
        self.rgb = tuple(int(c) for c in self.rgb)
        self.taps = tuple(self.taps)

    @property
    def hue(self) -> int:
        px = np.array([[list(self.rgb)]], dtype=np.uint8)
        return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])


@dataclass
class HueBand:
    """A family of note colours, as one or more inclusive hue ranges.

    Red needs two ranges because hue wraps: crimson sits just above 0 and
    magenta just below 180.
    """

    name: str
    ranges: list[tuple[int, int]]
    taps: tuple[str, ...]

    def contains(self, hue: int) -> bool:
        return any(lo <= hue <= hi for lo, hi in self.ranges)

    def mask(self, h):
        out = np.zeros(h.shape, dtype=bool)
        for lo, hi in self.ranges:
            out |= (h >= lo) & (h <= hi)
        return out

    def widen_to(self, hue: int, margin: int = 4) -> bool:
        """Stretch the nearest range to include `hue`. True if anything changed."""
        if self.contains(hue):
            return False
        best, best_gap = None, 10**9
        for i, (lo, hi) in enumerate(self.ranges):
            gap = lo - hue if hue < lo else hue - hi
            if gap < best_gap:
                best, best_gap = i, gap
        lo, hi = self.ranges[best]
        self.ranges[best] = (min(lo, hue - margin), hi) if hue < lo else (lo, max(hi, hue + margin))
        return True


def default_bands() -> list[HueBand]:
    return [
        HueBand("blue", [(85, 120)], ("blue",)),
        HueBand("red", [(128, 179), (0, 10)], ("red",)),
    ]


@dataclass
class Palette:
    bands: list[HueBand] = field(default_factory=default_bands)
    samples: list[Sample] = field(default_factory=list)

    # Gates for "this pixel is part of a note", not for which family it is.
    sat_min: int = 100
    val_min: int = 70

    min_pixels: int = 60  # absolute floor for a family to count as present
    # A family also has to be a real share of the strongest one, or a few
    # stray pixels from the note behind read as a second note.
    relative_floor: float = 0.18

    # Your rule, kept as a safety net. It should now be rare: the bands cover
    # every note variant, so an unrecognised strong colour means something new.
    unknown_is_both: bool = True
    unknown_min: int = 400
    # Hues that are scenery, not notes - the wooden desk and the lightning-bolt
    # accents live here, and must not count as "unrecognised".
    ignore_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(11, 34)])

    def band(self, name: str) -> HueBand | None:
        for b in self.bands:
            if b.name == name:
                return b
        return None

    def by_name(self, name: str) -> Sample | None:
        for s in self.samples:
            if s.name == name:
                return s
        return None

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.bands]

    def is_ready(self) -> bool:
        return bool(self.band("blue")) and bool(self.band("red"))


def count_matches(region_bgr: np.ndarray, palette: Palette, step: int = 3) -> dict[str, int]:
    """Count strongly-coloured pixels per hue family.

    Also returns 'unknown' - strong colour that is neither family nor scenery -
    and 'background', everything too pale to be part of a note.
    """
    if region_bgr.size == 0:
        return {name: 0 for name in palette.names} | {"unknown": 0, "background": 0}

    small = region_bgr[::step, ::step]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    strong = (s > palette.sat_min) & (v > palette.val_min)

    counts: dict[str, int] = {}
    claimed = np.zeros(h.shape, dtype=bool)
    for band in palette.bands:
        m = strong & band.mask(h)
        counts[band.name] = int(m.sum())
        claimed |= m

    scenery = np.zeros(h.shape, dtype=bool)
    for lo, hi in palette.ignore_ranges:
        scenery |= (h >= lo) & (h <= hi)

    counts["unknown"] = int((strong & ~claimed & ~scenery).sum())
    counts["background"] = int((~strong).sum())
    return counts


def decide_taps(counts: dict[str, int], palette: Palette,
                tap_order: tuple[str, ...] = ("red", "blue")) -> list[str]:
    """Turn per-family counts into the drums to press."""
    if not counts:
        return []
    matched = {b.name: counts.get(b.name, 0) for b in palette.bands}
    strongest = max(matched.values(), default=0)
    claimed: set[str] = set()

    for band in palette.bands:
        n = matched[band.name]
        if n >= palette.min_pixels and n >= palette.relative_floor * strongest:
            claimed.update(band.taps)

    if not claimed and palette.unknown_is_both:
        if counts.get("unknown", 0) >= palette.unknown_min:
            claimed.update(DRUMS)

    return [d for d in tap_order if d in claimed]


def check_sample(palette: Palette, sample: Sample) -> str | None:
    """Which band a clicked colour lands in, or None if it lands in neither."""
    for band in palette.bands:
        if band.contains(sample.hue):
            return band.name
    return None
