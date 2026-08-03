"""Regression tests for the zone detector.

The four reference frames have known correct answers, taken from what the game
was showing at that moment:

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

from vocalbot.calibrate import render_overlay, sample_counts
from vocalbot.color import ZoneReader, build_lut, classify
from vocalbot.config import Config, Zone
from vocalbot.scanner import ZoneScanner

FRAMES = Path(__file__).resolve().parent.parent / "assets" / "frames"

# The frames are stills from different moments, so the note sitting at the lane
# zone's default height is not always the one at the very front. These tests pin
# the lane zones to where the frontmost note is, to check the colour gates and
# the priority resolution against a known answer.
FRONT_LINE = (150 / 1206, 1620 / 2622, 1060 / 1206, 1644 / 2622)

CASES = [
    ("frame_t59_red.png", ["red"]),
    ("frame_t53_both.png", ["red", "blue"]),
    ("frame_t42_disco.png", ["blue"]),
    ("frame_t37_fever.png", ["red"]),
]


def front_config() -> Config:
    cfg = Config()
    for zone in cfg.group("lane"):
        zone.rect = FRONT_LINE
    return cfg


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(FRAMES / name))
    assert img is not None, f"missing fixture {name}"
    return img


def counts_for(cfg: Config, name: str) -> dict:
    return sample_counts(cfg, load(name))


# --------------------------------------------------------------------------
# detection against the reference frames


@pytest.mark.parametrize("name,expected", CASES)
def test_drums_pressed_for_frame(name, expected):
    """End to end: frame in, correct drum order out."""
    cfg = front_config()
    scanner = ZoneScanner(cfg)
    fires = scanner.update(counts_for(cfg, name), t=0.0)
    assert scanner.taps_for(fires) == expected


def test_both_zone_suppresses_the_single_colour_zones():
    """A simultaneous pair must tap each drum once, not twice."""
    cfg = front_config()
    scanner = ZoneScanner(cfg)
    fires = scanner.update(counts_for(cfg, "frame_t53_both.png"), t=0.0)
    assert [f.zone for f in fires] == ["both"]
    assert scanner.suppressed == 2  # the separate red and blue zones
    assert scanner.taps_for(fires) == ["red", "blue"]


def test_disco_zone_fires_only_on_the_mirror_ball():
    """The disco box must stay silent on ordinary notes, or it taps blue at random."""
    cfg = Config()  # default zone placement, not the front line
    for name, _ in CASES:
        red_px, blue_px = counts_for(cfg, name)["disco"]
        hit = cfg.zone("disco").hit(red_px, blue_px)
        assert hit == (name == "frame_t42_disco.png"), f"{name}: blue={blue_px}"


def test_disco_outranks_plain_blue():
    """Mirror ball reads as blue in the lane too; it must still tap blue once."""
    cfg = front_config()
    scanner = ZoneScanner(cfg)
    fires = scanner.update(counts_for(cfg, "frame_t42_disco.png"), t=0.0)
    assert [f.zone for f in fires] == ["disco"]
    assert scanner.taps_for(fires) == ["blue"]


def test_fever_tint_does_not_leak_red_into_blue():
    """Fever tints the screen gold. The blue gate must stay quiet on a red note."""
    cfg = front_config()
    red_px, blue_px = counts_for(cfg, "frame_t37_fever.png")["blue"]
    assert red_px > cfg.zone("blue").thresh_red
    assert blue_px < cfg.zone("blue").thresh_blue


def test_bare_playfield_is_silent():
    """A patch of empty desk must not trip any zone."""
    cfg = Config()
    cfg.zones = [Zone(name="empty", rect=(0.02, 0.75, 0.10, 0.79), match="any")]
    red_px, blue_px = counts_for(cfg, "frame_t59_red.png")["empty"]
    assert not cfg.zone("empty").hit(red_px, blue_px)


# --------------------------------------------------------------------------
# zone mechanics


def test_zone_fires_once_per_note():
    """A note spanning several frames produces exactly one tap."""
    cfg = Config()
    cfg.zones = [Zone(name="lane", rect=(0.1, 0.5, 0.9, 0.52), match="red", thresh_red=40)]
    scanner = ZoneScanner(cfg)
    fires = []
    seq = [0] * 3 + [400] * 5 + [0] * 8 + [400] * 5 + [0] * 3
    for i, px in enumerate(seq):
        fires += scanner.update({"lane": (px, 0)}, i * 0.016)
    assert [f.zone for f in fires] == ["lane", "lane"]


def test_refractory_collapses_notes_that_are_too_close():
    cfg = Config()
    cfg.zones = [
        Zone(name="lane", rect=(0.1, 0.5, 0.9, 0.52), match="red", thresh_red=40,
             refractory_ms=200.0)
    ]
    scanner = ZoneScanner(cfg)
    fires = []
    for i, px in enumerate([400] * 3 + [0] * 3 + [400] * 3):
        fires += scanner.update({"lane": (px, 0)}, i * 0.016)
    assert len(fires) == 1


def test_hysteresis_prevents_flicker_at_the_threshold():
    """Counts hovering at the threshold must not retrigger."""
    cfg = Config()
    cfg.zones = [
        Zone(name="lane", rect=(0.1, 0.5, 0.9, 0.52), match="red", thresh_red=40,
             hysteresis=0.6, refractory_ms=0.0)
    ]
    scanner = ZoneScanner(cfg)
    fires = []
    # dips to 30, which is above 40 * 0.6 = 24, so the zone stays latched
    for i, px in enumerate([45, 38, 42, 30, 44, 39]):
        fires += scanner.update({"lane": (px, 0)}, i * 0.016)
    assert len(fires) == 1


def test_suppressed_zone_stays_latched():
    """A zone suppressed this frame must not fire on the next one."""
    cfg = Config()
    cfg.zones = [
        Zone(name="both", rect=(0.1, 0.5, 0.9, 0.52), match="both", taps=("red", "blue"),
             thresh_red=40, thresh_blue=30, priority=10),
        Zone(name="red", rect=(0.1, 0.5, 0.9, 0.52), match="red", taps=("red",), thresh_red=40),
    ]
    scanner = ZoneScanner(cfg)
    first = scanner.update({"both": (400, 400), "red": (400, 400)}, 0.0)
    assert [f.zone for f in first] == ["both"]
    second = scanner.update({"both": (400, 400), "red": (400, 400)}, 0.016)
    assert second == []


def test_partial_overlap_is_not_suppressed():
    """A fire is only dropped when a higher zone covers *all* of its drums."""
    cfg = Config()
    cfg.zones = [
        Zone(name="hi", rect=(0.1, 0.5, 0.9, 0.52), match="red", taps=("red",), priority=10),
        Zone(name="lo", rect=(0.1, 0.5, 0.9, 0.52), match="blue", taps=("red", "blue"),
             thresh_blue=30),
    ]
    scanner = ZoneScanner(cfg)
    fires = scanner.update({"hi": (400, 400), "lo": (400, 400)}, 0.0)
    assert [f.zone for f in fires] == ["hi", "lo"]
    assert scanner.taps_for(fires) == ["red", "blue"]


def test_match_rules():
    z = Zone(name="z", rect=(0, 0, 1, 1), thresh_red=40, thresh_blue=30)
    for match, expected in [
        ("red", [True, False, True, False]),
        ("blue", [False, True, True, False]),
        ("both", [False, False, True, False]),
        ("any", [True, True, True, False]),
    ]:
        z.match = match
        got = [z.hit(*c) for c in [(50, 0), (0, 50), (50, 50), (0, 0)]]
        assert got == expected, match


def test_disabled_zones_are_ignored():
    cfg = Config()
    cfg.zone("disco").enabled = False
    assert "disco" not in [z.name for z in cfg.active_zones()]
    assert "disco" not in ZoneReader(cfg, 1206, 2622).slices


def test_zones_are_evaluated_highest_priority_first():
    cfg = Config()
    priorities = [z.priority for z in cfg.active_zones()]
    assert priorities == sorted(priorities, reverse=True)


# --------------------------------------------------------------------------
# geometry and config


def test_rects_are_scale_invariant():
    """The whole point of fractional geometry: a resized mirror window needs no
    recalibration."""
    cfg = Config()
    img = load("frame_t53_both.png")
    native = sample_counts(cfg, img)
    half = sample_counts(cfg, cv2.resize(img, (603, 1311)))
    for name in native:
        n, h = native[name][0], half[name][0]
        if n > 100:  # only meaningful where there is real signal
            assert abs(h * 4 - n) < 0.45 * n, f"{name}: {n} native vs {h} at half scale"


def test_zone_reader_origin_matches_full_frame_slice():
    """Reading from a cropped union frame must equal reading the whole screen."""
    cfg = Config()
    img = load("frame_t53_both.png")
    h, w = img.shape[:2]
    x0, y0, x1, y1 = cfg.union_rect()
    ox, oy = int(x0 * w), int(y0 * h)
    crop = img[oy : int(y1 * h) + 1, ox : int(x1 * w) + 1]

    full = ZoneReader(cfg, w, h).read(img)
    cropped = ZoneReader(cfg, w, h, origin=(ox, oy)).read(crop)
    assert full == cropped


def test_union_rect_covers_every_active_zone():
    cfg = Config()
    ux0, uy0, ux1, uy1 = cfg.union_rect()
    for z in cfg.active_zones():
        assert ux0 <= z.rect[0] and z.rect[2] <= ux1
        assert uy0 <= z.rect[1] and z.rect[3] <= uy1


def test_config_round_trips(tmp_path):
    cfg = Config()
    cfg.window_rect = (10, 20, 400, 870)
    cfg.zone("red").thresh_red = 77
    path = tmp_path / "c.json"
    cfg.save(path)
    back = Config.load(path)
    assert back.window_rect == (10, 20, 400, 870)
    assert back.zone("red").thresh_red == 77
    assert [z.name for z in back.zones] == [z.name for z in cfg.zones]
    assert back.zone("disco").rect == cfg.zone("disco").rect


def test_invalid_zones_are_rejected():
    with pytest.raises(ValueError):
        Zone(name="bad", rect=(0.5, 0.1, 0.2, 0.4))  # x1 < x0
    with pytest.raises(ValueError):
        Zone(name="bad", rect=(0, 0, 1, 1), match="purple")
    with pytest.raises(ValueError):
        Zone(name="bad", rect=(0, 0, 1, 1), taps=("green",))


def test_drum_points_move_with_the_window():
    cfg = Config()
    cfg.window_rect = (100, 50, 400, 870)
    rx, ry = cfg.drum_point("red")
    bx, by = cfg.drum_point("blue")
    assert rx == 100 + int(0.809 * 400) and ry == 50 + int(0.802 * 870)
    assert bx == 100 + int(0.185 * 400) and by == 50 + int(0.802 * 870)
    assert rx > bx  # red drum is on the right


# --------------------------------------------------------------------------
# supporting paths


def test_lut_matches_direct_hsv():
    """The baked LUT must agree with evaluating the HSV rules directly."""
    cfg = Config()
    strip = load("frame_t53_both.png")[1400:1424, 150:1060]
    lut_red, lut_blue = classify(strip, build_lut(cfg), step=1)

    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    direct_red = int(cfg.red.mask(h, s, v).sum())
    direct_blue = int(cfg.blue.mask(h, s, v).sum())

    # Quantising to 5 bits per channel moves a few boundary pixels either way.
    assert abs(lut_red - direct_red) < max(40, 0.10 * direct_red)
    assert abs(lut_blue - direct_blue) < max(40, 0.10 * direct_blue)


def test_render_overlay_runs_headless():
    """Calibration snapshots must work without a GUI build of opencv."""
    cfg = Config()
    img = load("frame_t59_red.png")
    out = render_overlay(cfg, img, counts=sample_counts(cfg, img), active="red")
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_classify_is_fast():
    """Guard against the hot path regressing. Budget is generous for CI."""
    import time

    cfg = Config()
    reader = ZoneReader(cfg, 1206, 2622)
    frame = np.random.randint(0, 255, (2622, 1206, 3), dtype=np.uint8)
    for _ in range(20):
        reader.read(frame)
    t0 = time.perf_counter()
    for _ in range(200):
        reader.read(frame)
    ms = (time.perf_counter() - t0) / 200 * 1000
    assert ms < 3.0, f"zone read regressed to {ms:.3f} ms/frame"
