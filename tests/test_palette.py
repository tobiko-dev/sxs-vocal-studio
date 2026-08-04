"""Tests for colour matching against clicked samples.

Setup records the colour of each note from your own screen, so detection stops
depending on thresholds guessed from reference screenshots.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vocalbot.palette import Palette, Sample, count_matches, decide_taps
from vocalbot.wizard import sample_rect_from_points, vivid_colour

FRAMES = Path(__file__).resolve().parent.parent / "assets" / "frames"

BLUE = (221, 166, 32)  # BGR, sampled from the vivid core of a blue note
RED = (157, 33, 222)


def slot(name: str) -> np.ndarray:
    """The front-slot region of a reference frame."""
    img = cv2.imread(str(FRAMES / f"{name}.png"))
    assert img is not None, name
    h, w = img.shape[:2]
    return img[int(0.65 * h): int(0.70 * h), int(0.25 * w): int(0.75 * w)]


def two_colour_palette(**kw) -> Palette:
    return Palette(
        samples=[Sample("blue", BLUE, ("blue",)), Sample("red", RED, ("red",))], **kw
    )


# --------------------------------------------------------------------------
# against the reference frames


@pytest.mark.parametrize(
    "name,expected",
    [("frame_t59_red", ["red"]), ("frame_t53_both", ["red", "blue"]),
     ("frame_t37_fever", ["red"])],
)
def test_notes_are_read_correctly(name, expected):
    pal = two_colour_palette(tolerance=70)
    assert decide_taps(count_matches(slot(name), pal), pal) == expected


def test_fever_does_not_add_a_phantom_blue():
    """Fever tints the screen gold; that must not push a red note into 'both'."""
    pal = two_colour_palette(tolerance=70)
    counts = count_matches(slot("frame_t37_fever"), pal)
    assert counts["red"] > pal.min_pixels
    assert counts["blue"] < pal.min_pixels


def test_a_stray_sliver_of_the_other_colour_is_ignored():
    """The note behind the front one can leak a few pixels in. A relative floor
    keeps that from reading as a second note and adding a phantom press."""
    pal = two_colour_palette()
    counts = {"red": 400, "blue": 30, "unknown": 0, "background": 0}
    assert decide_taps(counts, pal) == ["red"]


def test_a_genuine_second_note_is_not_ignored():
    pal = two_colour_palette()
    counts = {"red": 400, "blue": 120, "unknown": 0, "background": 0}
    assert decide_taps(counts, pal) == ["red", "blue"]


# --------------------------------------------------------------------------
# the press-both fallback


def test_unmatched_colour_presses_both():
    """Your rule: if it is neither of the sampled notes, press both."""
    pal = two_colour_palette()
    counts = {"red": 0, "blue": 0, "unknown": 900, "background": 100}
    assert decide_taps(counts, pal) == ["red", "blue"]


def test_bubble_noise_does_not_trigger_the_fallback():
    """The box is full of pale unmatched pixels every frame. Only a substantial
    unknown count may fire, or the bot presses both continuously."""
    pal = two_colour_palette()
    counts = {"red": 0, "blue": 0, "unknown": pal.unknown_min - 1, "background": 5000}
    assert decide_taps(counts, pal) == []


def test_fallback_does_not_override_a_clear_read():
    pal = two_colour_palette()
    counts = {"red": 500, "blue": 0, "unknown": 5000, "background": 0}
    assert decide_taps(counts, pal) == ["red"]


def test_fallback_can_be_disabled():
    pal = two_colour_palette(unknown_is_both=False)
    counts = {"red": 0, "blue": 0, "unknown": 9000, "background": 0}
    assert decide_taps(counts, pal) == []


def test_empty_box_presses_nothing():
    pal = two_colour_palette()
    blank = np.full((60, 300, 3), 128, np.uint8)
    assert decide_taps(count_matches(blank, pal), pal) == []


# --------------------------------------------------------------------------
# sampling


def test_vivid_colour_ignores_washed_out_pixels():
    """A click that catches some pale bubble must still yield the note colour."""
    patch = np.full((11, 11, 3), 235, np.uint8)  # mostly pale
    patch[4:7, 4:7] = BLUE  # vivid core
    got = vivid_colour(patch)
    assert np.linalg.norm(np.array(got, int) - np.array(BLUE, int)) < 60


def test_vivid_colour_survives_a_flat_patch():
    flat = np.full((11, 11, 3), 90, np.uint8)
    assert vivid_colour(flat) == (90, 90, 90)


def test_dark_samples_would_be_indistinguishable():
    """Why setup asks for the vivid part rather than the darkest.

    The shadowed cores of the two notes are nearly the same colour, so sampling
    there would collapse both classes into one.
    """
    img = cv2.imread(str(FRAMES / "frame_t53_both.png"))
    h, w = img.shape[:2]
    box = img[int(0.65 * h): int(0.70 * h), int(0.25 * w): int(0.75 * w)]
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    def darkest(mask):
        ys, xs = np.where(mask)
        return box[ys[np.argmin(vv[ys, xs])], xs[np.argmin(vv[ys, xs])]].astype(int)

    dark_blue = darkest((hh >= 88) & (hh <= 118) & (ss > 90) & (vv > 90))
    dark_red = darkest(((hh >= 150) | (hh <= 9)) & (ss > 110) & (vv > 90))
    dark_gap = np.linalg.norm(dark_blue - dark_red)
    vivid_gap = np.linalg.norm(np.array(BLUE, int) - np.array(RED, int))
    assert dark_gap < 100 < vivid_gap


def test_sample_rect_covers_both_click_points():
    rect = sample_rect_from_points([(1000, 600), (1100, 620)])
    x, y, w, h = rect
    for px, py in ((1000, 600), (1100, 620)):
        assert x <= px <= x + w and y <= py <= y + h


def test_sample_rect_is_none_without_points():
    assert sample_rect_from_points([]) is None


# --------------------------------------------------------------------------
# palette bookkeeping


def test_palette_is_not_ready_without_both_notes():
    assert not Palette().is_ready()
    assert not Palette(samples=[Sample("blue", BLUE, ("blue",))]).is_ready()
    assert two_colour_palette().is_ready()


def test_sample_rejects_unknown_drums():
    with pytest.raises(ValueError):
        Sample("x", (1, 2, 3), ("green",))


def test_counts_cover_every_sample():
    pal = two_colour_palette()
    counts = count_matches(slot("frame_t53_both"), pal)
    for name in pal.names:
        assert name in counts
    assert "unknown" in counts and "background" in counts


def test_subsampling_does_not_change_the_verdict():
    pal = two_colour_palette(tolerance=70)
    region = slot("frame_t53_both")
    fine = decide_taps(count_matches(region, pal, step=1), pal)
    coarse = decide_taps(count_matches(region, pal, step=3), pal)
    assert fine == coarse
