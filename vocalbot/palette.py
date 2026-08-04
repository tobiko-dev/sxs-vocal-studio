"""Colour matching against samples you clicked, rather than hand-tuned gates.

Calibration records the actual colour of each thing on *your* screen: the blue
note, the red note, optionally the white "both" marker and the disco ball. At
run time every pixel in the sample box is matched to the nearest sample and
counted. That replaces the HSV thresholds, which had to be guessed in advance
and re-tuned whenever the display differed.

The fallback rule is yours: if the box clearly has something in it but it
matches neither the blue nor the red sample, treat it as both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DRUMS = ("red", "blue")


@dataclass
class Sample:
    """One clicked reference colour."""

    name: str
    rgb: tuple[int, int, int]  # as BGR, matching OpenCV
    taps: tuple[str, ...]
    point: tuple[int, int] | None = None  # absolute screen coords, for reference

    def __post_init__(self):
        bad = [d for d in self.taps if d not in DRUMS]
        if bad:
            raise ValueError(f"{self.name}: unknown drum(s) {bad}")
        self.rgb = tuple(int(c) for c in self.rgb)
        self.taps = tuple(self.taps)


@dataclass
class Palette:
    """The set of sampled colours plus the matching rules."""

    samples: list[Sample] = field(default_factory=list)
    tolerance: int = 70  # max euclidean BGR distance to count as a match
    min_pixels: int = 25  # absolute floor for a class to count as present
    # A class also has to be a real share of the strongest one. Without this a
    # handful of stray matches - anti-aliased edges, a sliver of the note behind
    # - reads as a second colour and adds a phantom drum press.
    relative_floor: float = 0.18
    unknown_is_both: bool = True  # your rule: not blue, not red -> press both
    unknown_min: int = 150  # 'unknown' must be substantial, not bubble noise

    def by_name(self, name: str) -> Sample | None:
        for s in self.samples:
            if s.name == name:
                return s
        return None

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.samples]

    def is_ready(self) -> bool:
        """Both note colours are the minimum needed to play."""
        return self.by_name("blue") is not None and self.by_name("red") is not None


def count_matches(region_bgr: np.ndarray, palette: Palette, step: int = 3) -> dict[str, int]:
    """Count pixels nearest to each sample, plus 'unknown' and 'background'.

    'unknown' is a pixel that is clearly part of a note - it stands out from the
    box's own background - but matches no sample. That is what drives the
    press-both fallback.
    """
    if not palette.samples:
        return {}

    px = region_bgr[::step, ::step].reshape(-1, 3).astype(np.int32)
    if px.size == 0:
        return {name: 0 for name in palette.names} | {"unknown": 0, "background": 0}

    refs = np.array([s.rgb for s in palette.samples], dtype=np.int32)
    # squared distance from every pixel to every sample
    d2 = ((px[:, None, :] - refs[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    best_d2 = d2[np.arange(len(px)), nearest]
    within = best_d2 <= palette.tolerance ** 2

    counts = {name: 0 for name in palette.names}
    for i, name in enumerate(palette.names):
        counts[name] = int(((nearest == i) & within).sum())

    # Anything unmatched but *strongly* coloured is a note we have no sample
    # for. The bar is high: the box is full of pale bubble and desk pixels that
    # match nothing, and treating those as "unknown" would fire on every frame.
    unmatched = ~within
    spread = px.max(axis=1) - px.min(axis=1)  # cheap saturation proxy
    counts["unknown"] = int((unmatched & (spread > 90)).sum())
    counts["background"] = int((unmatched & (spread <= 90)).sum())
    return counts


def decide_taps(counts: dict[str, int], palette: Palette,
                tap_order: tuple[str, ...] = ("red", "blue")) -> list[str]:
    """Turn per-sample counts into the drums to press."""
    if not counts:
        return []
    matched = {s.name: counts.get(s.name, 0) for s in palette.samples}
    strongest = max(matched.values(), default=0)
    claimed: set[str] = set()

    for sample in palette.samples:
        n = matched[sample.name]
        if n >= palette.min_pixels and n >= palette.relative_floor * strongest:
            claimed.update(sample.taps)

    if not claimed and palette.unknown_is_both:
        if counts.get("unknown", 0) >= palette.unknown_min:
            # Something is there and it matches none of the samples.
            claimed.update(DRUMS)

    return [d for d in tap_order if d in claimed]
