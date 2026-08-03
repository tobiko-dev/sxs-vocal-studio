"""Regression tests against the four reference frames.

Each frame has a known correct answer, taken from what the game was showing:
    frame_t59_red    - red note at the front
    frame_t53_both   - blue and red arriving together
    frame_t42_disco  - mirror ball (played on the blue drum)
    frame_t37_fever  - red note, fever mode, whole screen tinted gold
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vocalbot.color import build_lut, classify
from vocalbot.config import Config
from vocalbot.scanner import Scanner

FRAMES = Path(__file__).resolve().parent.parent / "assets" / "frames"

# The scanline sits high on the runway, where glyphs stay separable. These are the
# colours present at that height in each still.
CASES = [
    ("frame_t59_red.png", {"red"}),
    ("frame_t53_both.png", {"red", "blue"}),
    ("frame_t42_disco.png", {"blue"}),
    ("frame_t37_fever.png", {"red"}),
]

# Frames are stills from different moments, so the note at the scanline is not
# always the one at the very front. These use a lower line, where the frontmost
# note sits, to check the colour gates themselves.
FRONT_LINE = 1620 / 2622


def _counts(name, cfg, line):
    img = cv2.imread(str(FRAMES / name))
    assert img is not None, f"missing fixture {name}"
    h, w = img.shape[:2]
    y = int(line * h)
    sh = max(2, int(cfg.scan_h * h))
    x0, x1 = int(cfg.lane_x0 * w), int(cfg.lane_x1 * w)
    return classify(img[y : y + sh, x0:x1], build_lut(cfg), cfg.step)


@pytest.mark.parametrize("name,expected", CASES)
def test_front_note_colour(name, expected):
    """The colour gates pick the right drum for the frontmost note."""
    cfg = Config()
    red_px, blue_px = _counts(name, cfg, FRONT_LINE)
    got = set()
    if red_px >= cfg.thresh_red:
        got.add("red")
    if blue_px >= cfg.thresh_blue:
        got.add("blue")
    assert got == expected, f"{name}: red={red_px} blue={blue_px}"


def test_disco_reads_as_blue():
    """The mirror ball must not read as red, or it taps the wrong drum."""
    cfg = Config()
    red_px, blue_px = _counts("frame_t42_disco.png", cfg, FRONT_LINE)
    assert blue_px > cfg.thresh_blue
    assert red_px < cfg.thresh_red


def test_empty_strip_is_silent():
    """A patch of bare desk must not trip either gate."""
    cfg = Config()
    img = cv2.imread(str(FRAMES / "frame_t59_red.png"))
    h, w = img.shape[:2]
    # Far left of the playfield, outside the note runway.
    strip = img[int(0.75 * h) : int(0.75 * h) + 24, int(0.02 * w) : int(0.10 * w)]
    red_px, blue_px = classify(strip, build_lut(cfg), cfg.step)
    assert red_px < cfg.thresh_red
    assert blue_px < cfg.thresh_blue


def test_lut_matches_direct_hsv():
    """The baked LUT must agree with evaluating the HSV rules directly."""
    cfg = Config()
    img = cv2.imread(str(FRAMES / "frame_t53_both.png"))
    strip = img[1400:1424, 150:1060]
    lut_red, lut_blue = classify(strip, build_lut(cfg), step=1)

    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    direct_red = int(cfg.red.mask(h, s, v).sum())
    direct_blue = int(cfg.blue.mask(h, s, v).sum())

    # Quantising to 5 bits per channel moves a few boundary pixels either way.
    assert abs(lut_red - direct_red) < max(40, 0.10 * direct_red)
    assert abs(lut_blue - direct_blue) < max(40, 0.10 * direct_blue)


def test_scanner_fires_once_per_note():
    """A note spanning several frames produces exactly one tap."""
    cfg = Config()
    scanner = Scanner(cfg)
    fires = []
    # note present for 5 frames, gone for 8, then a second note
    seq = [0] * 3 + [400] * 5 + [0] * 8 + [400] * 5 + [0] * 3
    for i, px in enumerate(seq):
        fires += scanner.update(px, 0, i * 0.016)
    assert [f.color for f in fires] == ["red", "red"]


def test_scanner_refractory_blocks_double_fire():
    """Two notes closer than the refractory period collapse to one tap."""
    cfg = Config()
    cfg.refractory_ms = 200.0
    scanner = Scanner(cfg)
    fires = []
    seq = [400] * 3 + [0] * 3 + [400] * 3
    for i, px in enumerate(seq):
        fires += scanner.update(px, 0, i * 0.016)
    assert len(fires) == 1


def test_scanner_both_colours_same_frame():
    """Both drums fire when both colours cross together."""
    cfg = Config()
    scanner = Scanner(cfg)
    fires = scanner.update(400, 400, 0.0)
    assert {f.color for f in fires} == {"red", "blue"}


def test_classify_is_fast():
    """Guard against the hot path regressing. Budget is generous for CI."""
    import time

    cfg = Config()
    lut = build_lut(cfg)
    strip = np.random.randint(0, 255, (24, 910, 3), dtype=np.uint8)
    for _ in range(50):
        classify(strip, lut, cfg.step)
    t0 = time.perf_counter()
    for _ in range(500):
        classify(strip, lut, cfg.step)
    ms = (time.perf_counter() - t0) / 500 * 1000
    assert ms < 1.0, f"classify regressed to {ms:.3f} ms/frame"
