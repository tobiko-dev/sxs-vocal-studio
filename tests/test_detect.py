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
from vocalbot.queuescan import resolve_taps

FRAMES = Path(__file__).resolve().parent.parent / "assets" / "frames"

CASES = [
    ("frame_t59_red.png", ["red"]),
    ("frame_t53_both.png", ["red", "blue"]),
    ("frame_t42_disco.png", ["blue"]),
    ("frame_t37_fever.png", ["red"]),
]


def front_config() -> Config:
    """The shipped config, unmodified.

    The stills and the gameplay recording share the phone's aspect ratio, so the
    default front-slot box lands on the frontmost note in both. These tests
    therefore exercise the config that actually ships.
    """
    return Config()


def taps_for_frame(cfg: Config, name: str) -> list[str]:
    return resolve_taps(cfg.active_zones(), counts_for(cfg, name), cfg.tap_order)


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
    assert taps_for_frame(front_config(), name) == expected


def test_both_note_taps_each_drum_once():
    """A simultaneous pair must tap each drum once, not twice."""
    cfg = front_config()
    taps = taps_for_frame(cfg, "frame_t53_both.png")
    assert taps == ["red", "blue"]
    assert len(taps) == len(set(taps))


def test_mirror_ball_taps_blue_exactly_once():
    """Under the queue model the ball simply is the front note. It reads as blue
    there, so it must produce a single blue tap however the zones resolve."""
    cfg = front_config()
    assert taps_for_frame(cfg, "frame_t42_disco.png") == ["blue"]


def test_mirror_ball_never_reads_red():
    """Any red leaking into the ball's read would tap the wrong drum."""
    cfg = front_config()
    red_px, _ = counts_for(cfg, "frame_t42_disco.png")[cfg.group("front")[0].name]
    assert red_px < cfg.group("front")[0].thresh_red


def test_fever_tint_does_not_leak_red_into_blue():
    """Fever tints the screen gold. The blue gate must stay quiet on a red note."""
    cfg = front_config()
    red_px, blue_px = counts_for(cfg, "frame_t37_fever.png")["blue"]  # noqa: F841
    assert red_px > cfg.zone("blue").thresh_red
    assert blue_px < cfg.zone("blue").thresh_blue


def test_bare_playfield_is_silent():
    """A patch of empty desk must not trip any zone."""
    cfg = Config()
    cfg.zones = [Zone(name="empty", rect=(0.02, 0.75, 0.10, 0.79), match="any")]
    red_px, blue_px = counts_for(cfg, "frame_t59_red.png")["empty"]
    assert not cfg.zone("empty").hit(red_px, blue_px)


# --------------------------------------------------------------------------
# geometry and config


def test_counts_are_resolution_invariant():
    """One set of thresholds must work at any mirror window size.

    Raw pixel counts scale with area, so a half-size window would quarter every
    count and silently fall under the thresholds. Counts are normalised to the
    reference screen, so the same note reads the same at any scale.
    """
    cfg = Config()
    img = load("frame_t53_both.png")
    native = sample_counts(cfg, img)
    for w, h in ((603, 1311), (616, 1336), (402, 874)):  # half, the recording, a third
        scaled = sample_counts(cfg, cv2.resize(img, (w, h)))
        for name in native:
            n, s = native[name][0], scaled[name][0]
            if n > 100:  # only meaningful where there is real signal
                assert abs(s - n) < 0.30 * n, f"{name} at {w}x{h}: {n} native vs {s}"


def test_thresholds_still_pass_at_recording_resolution():
    """The reference frames must resolve identically when downscaled to the size
    a real mirror window records at."""
    cfg = front_config()
    for name, expected in CASES:
        small = cv2.resize(load(name), (616, 1336))
        got = resolve_taps(cfg.active_zones(), sample_counts(cfg, small), cfg.tap_order)
        assert got == expected, f"{name} at 616x1336"


def test_shipped_box_has_margin_on_every_reference_frame():
    """Counts should sit well clear of the threshold, not just past it."""
    cfg = front_config()
    front = cfg.group("front")[0]
    for name, expected in CASES:
        red_px, blue_px = counts_for(cfg, name)[front.name]
        live = max(red_px, blue_px)
        assert live > front.thresh_red * 2, f"{name}: only {live} vs {front.thresh_red}"
        quiet = min(red_px, blue_px)
        if len(expected) == 1:
            assert quiet < front.thresh_red * 0.5, f"{name}: {quiet} leaking into the other gate"


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
