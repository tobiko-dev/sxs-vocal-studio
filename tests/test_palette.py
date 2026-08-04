"""Tests for hue-family classification.

Notes come in many variants - plain, beamed, dotted, lightning-bolt - and each
family spans a wide range of shades. Blue runs navy to cyan, red runs crimson
through magenta to purple. Those are far apart in RGB but share a hue family,
which is what classification keys on.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vocalbot.palette import (
    HueBand,
    Palette,
    Sample,
    check_sample,
    count_matches,
    decide_taps,
)
from vocalbot.wizard import sample_rect_from_points, vivid_colour

FRAMES = Path(__file__).resolve().parent.parent / "assets" / "frames"


def patch(hue: int, sat: int = 200, val: int = 200, size=(40, 120)) -> np.ndarray:
    """A solid BGR patch at a given hue."""
    hsv = np.full((size[0], size[1], 3), (hue, sat, val), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def slot(name: str) -> np.ndarray:
    img = cv2.imread(str(FRAMES / f"{name}.png"))
    assert img is not None, name
    h, w = img.shape[:2]
    return img[int(0.65 * h): int(0.70 * h), int(0.25 * w): int(0.75 * w)]


# --------------------------------------------------------------------------
# every note variant


@pytest.mark.parametrize(
    "hue,family",
    [
        (90, "blue"),   # pale cyan
        (99, "blue"),   # mid blue
        (110, "blue"),  # royal blue
        (118, "blue"),  # navy, the darkest variant
        (132, "red"),   # purple, the top of a beamed note's gradient
        (150, "red"),   # magenta
        (168, "red"),   # pink-red
        (178, "red"),   # crimson
        (3, "red"),     # crimson past the hue wrap
    ],
)
def test_every_shade_lands_in_the_right_family(hue, family):
    """One sample per family could never cover this spread in RGB; hue can."""
    pal = Palette()
    counts = count_matches(patch(hue), pal, step=1)
    assert counts[family] > 0, counts
    other = "red" if family == "blue" else "blue"
    assert counts[other] == 0, counts


def test_navy_and_cyan_are_both_blue():
    """The two extremes of the blue family, which are far apart in RGB."""
    pal = Palette()
    for hue in (88, 118):
        assert decide_taps(count_matches(patch(hue), pal, step=1), pal) == ["blue"]


def test_purple_and_crimson_are_both_red():
    pal = Palette()
    for hue in (131, 178):
        assert decide_taps(count_matches(patch(hue), pal, step=1), pal) == ["red"]


def test_the_gap_between_families_is_unclaimed():
    """Measured on 5.1M pixels, hues 119-129 are effectively empty. Nothing
    should be forced into a family from there."""
    pal = Palette()
    counts = count_matches(patch(124), pal, step=1)
    assert counts["blue"] == 0 and counts["red"] == 0


def test_lightning_bolts_and_desk_are_ignored():
    """Bolt accents and the wooden desk sit around hue 10-30. They are scenery,
    and must not read as a note or as an unrecognised colour."""
    pal = Palette()
    for hue in (15, 20, 30):
        counts = count_matches(patch(hue), pal, step=1)
        assert counts["blue"] == 0 and counts["red"] == 0
        assert counts["unknown"] == 0, f"hue {hue} counted as unknown"


def test_pale_bubble_is_not_a_note():
    """The bubbles are pale; only strongly coloured pixels are notes."""
    pal = Palette()
    counts = count_matches(patch(99, sat=40, val=240), pal, step=1)
    assert counts["blue"] == 0
    assert counts["background"] > 0


# --------------------------------------------------------------------------
# the reference frames


@pytest.mark.parametrize(
    "name,expected",
    [("frame_t59_red", ["red"]), ("frame_t53_both", ["red", "blue"]),
     ("frame_t42_disco", ["blue"]), ("frame_t37_fever", ["red"])],
)
def test_reference_frames(name, expected):
    pal = Palette()
    assert decide_taps(count_matches(slot(name), pal), pal) == expected


def test_disco_ball_needs_no_sample_of_its_own():
    """The ball's colours already sit in the blue family."""
    pal = Palette()
    counts = count_matches(slot("frame_t42_disco"), pal)
    assert counts["blue"] > pal.min_pixels
    assert counts["red"] < pal.min_pixels


def test_fever_tint_does_not_add_a_phantom_blue():
    pal = Palette()
    counts = count_matches(slot("frame_t37_fever"), pal)
    assert counts["red"] > pal.min_pixels
    assert counts["blue"] < pal.min_pixels


# --------------------------------------------------------------------------
# deciding


def test_a_stray_sliver_of_the_other_family_is_ignored():
    pal = Palette()
    assert decide_taps({"red": 4000, "blue": 200, "unknown": 0}, pal) == ["red"]


def test_a_genuine_second_note_is_not_ignored():
    pal = Palette()
    assert decide_taps({"red": 4000, "blue": 1500, "unknown": 0}, pal) == ["red", "blue"]


def test_nothing_on_screen_presses_nothing():
    pal = Palette()
    assert decide_taps({"red": 0, "blue": 0, "unknown": 0}, pal) == []


def test_unrecognised_colour_presses_both():
    """Kept as a safety net for a variant the bands don't cover."""
    pal = Palette()
    assert decide_taps({"red": 0, "blue": 0, "unknown": 9000}, pal) == ["red", "blue"]


def test_the_fallback_needs_a_substantial_amount():
    pal = Palette()
    counts = {"red": 0, "blue": 0, "unknown": pal.unknown_min - 1}
    assert decide_taps(counts, pal) == []


def test_the_fallback_never_overrides_a_clear_read():
    pal = Palette()
    assert decide_taps({"red": 5000, "blue": 0, "unknown": 9000}, pal) == ["red"]


def test_tap_order_is_respected():
    pal = Palette()
    counts = {"red": 4000, "blue": 4000, "unknown": 0}
    assert decide_taps(counts, pal, tap_order=("blue", "red")) == ["blue", "red"]


# --------------------------------------------------------------------------
# bands and samples


def test_band_membership():
    band = HueBand("blue", [(85, 120)], ("blue",))
    assert band.contains(85) and band.contains(120)
    assert not band.contains(84) and not band.contains(121)


def test_red_band_wraps_around_zero():
    red = Palette().band("red")
    assert red.contains(179) and red.contains(3)
    assert not red.contains(60)


def test_widening_pulls_a_band_to_include_a_nearby_hue():
    """A display that shifts hue slightly shouldn't need a code change."""
    band = HueBand("blue", [(85, 120)], ("blue",))
    assert band.widen_to(123)
    assert band.contains(123)
    assert not band.widen_to(100)  # already inside, nothing to do


def test_widening_picks_the_nearest_range():
    red = Palette().band("red")
    red.widen_to(14)  # just above the 0-10 wrap range
    assert red.contains(14)
    assert red.contains(179)  # the other range is untouched


def test_sample_reports_its_hue():
    assert Sample("blue", (221, 166, 32), ("blue",)).hue == pytest.approx(99, abs=3)
    assert Sample("red", (157, 33, 222), ("red",)).hue == pytest.approx(160, abs=3)


def test_check_sample_names_the_band():
    pal = Palette()
    assert check_sample(pal, Sample("blue", (221, 166, 32), ("blue",))) == "blue"
    assert check_sample(pal, Sample("red", (157, 33, 222), ("red",))) == "red"


def test_check_sample_flags_a_colour_in_neither_band():
    """Pointing at the desk by mistake has to be detectable."""
    pal = Palette()
    desk = cv2.cvtColor(np.uint8([[[18, 180, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    assert check_sample(pal, Sample("blue", tuple(int(c) for c in desk), ("blue",))) is None


def test_sample_rejects_unknown_drums():
    with pytest.raises(ValueError):
        Sample("x", (1, 2, 3), ("green",))


def test_palette_is_ready_with_both_bands():
    assert Palette().is_ready()


# --------------------------------------------------------------------------
# sampling helpers


def test_vivid_colour_ignores_washed_out_pixels():
    blue = (221, 166, 32)
    p = np.full((11, 11, 3), 235, np.uint8)
    p[4:7, 4:7] = blue
    assert np.linalg.norm(np.array(vivid_colour(p), int) - np.array(blue, int)) < 60


def test_vivid_colour_survives_a_flat_patch():
    assert vivid_colour(np.full((11, 11, 3), 90, np.uint8)) == (90, 90, 90)


def test_sample_rect_covers_both_points():
    x, y, w, h = sample_rect_from_points([(1000, 600), (1100, 620)])
    for px, py in ((1000, 600), (1100, 620)):
        assert x <= px <= x + w and y <= py <= y + h


def test_sample_rect_is_none_without_points():
    assert sample_rect_from_points([]) is None


def test_subsampling_does_not_change_the_verdict():
    pal = Palette()
    region = slot("frame_t53_both")
    assert (decide_taps(count_matches(region, pal, step=1), pal)
            == decide_taps(count_matches(region, pal, step=3), pal))


# --------------------------------------------------------------------------
# hover-to-select
#
# Targets are picked by hovering: reading the cursor position needs no
# permission, while key or mouse-button state needs macOS Input Monitoring and
# silently reports nothing when that is missing.


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def run_dwell(path, **kw):
    from vocalbot.wizard import dwell

    clock = FakeClock()
    steps = iter(path)
    last = [path[0]]

    def pos():
        try:
            last[0] = next(steps)
        except StopIteration:
            pass
        return last[0]

    return dwell(pos, clock=clock, sleep=clock.sleep, **kw)


def test_holding_still_locks_the_point():
    assert run_dwell([(500, 400)] * 400, hold=1.0) == (500, 400)


def test_moving_restarts_the_hold():
    wander = [(100 + i * 40, 200) for i in range(30)]
    assert run_dwell(wander + [(900, 200)] * 400, hold=1.0) == (900, 200)


def test_small_wobble_still_counts_as_still():
    jitter = [(500 + (i % 3), 400 - (i % 2)) for i in range(400)]
    got = run_dwell(jitter, hold=1.0, radius=8)
    assert got is not None and abs(got[0] - 500) <= 8


def test_constant_movement_never_locks():
    drift = [(100 + i * 20, 200) for i in range(500)]
    assert run_dwell(drift, hold=1.0, timeout=5.0) is None


def test_dwell_reports_progress():
    seen = []
    run_dwell([(300, 300)] * 400, hold=1.0, on_tick=lambda p, f: seen.append(f))
    assert seen and seen[0] < seen[-1] and max(seen) <= 1.0
