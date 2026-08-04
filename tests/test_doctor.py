"""Tests for the capture diagnostics.

The most confusing failure this project can produce is a calibrator full of
desktop wallpaper. Nothing is broken when that happens - the detector is reading
the wrong pixels faithfully - so the value is in saying so immediately rather
than letting it surface later as "the bot never taps".
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vocalbot.config import Config
from vocalbot.doctor import DRUM_MIN_FILL, drum_fill, looks_like_game

FRAMES = Path(__file__).resolve().parent.parent / "assets" / "frames"
GAME_FRAMES = [
    "frame_t59_red.png",
    "frame_t53_both.png",
    "frame_t42_disco.png",
    "frame_t37_fever.png",
]


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(FRAMES / name))
    assert img is not None, f"missing fixture {name}"
    return img


@pytest.mark.parametrize("name", GAME_FRAMES)
def test_real_game_screens_are_recognised(name):
    cfg = Config()
    assert looks_like_game(cfg, load(name)), f"{name}: {drum_fill(cfg, load(name))}"


@pytest.mark.parametrize("name", GAME_FRAMES)
def test_both_drums_are_found_with_margin(name):
    """Fills should sit well clear of the cutoff, not just past it."""
    red_fill, blue_fill = drum_fill(Config(), load(name))
    assert min(red_fill, blue_fill) > DRUM_MIN_FILL * 2


def test_flat_desktop_is_rejected():
    cfg = Config()
    assert not looks_like_game(cfg, np.full((1336, 616, 3), 128, np.uint8))


def test_noise_is_rejected():
    """Random colour hits both hue gates a little; it must still fail."""
    cfg = Config()
    noise = np.random.default_rng(0).integers(0, 255, (1336, 616, 3), dtype=np.uint8)
    assert not looks_like_game(cfg, noise)


def test_a_busy_image_without_drums_is_rejected():
    """Texture alone is not the test - a detailed wallpaper is still not the game."""
    cfg = Config()
    busy = cv2.resize(load("frame_t59_red.png")[200:900, 100:500], (616, 1336))
    assert not looks_like_game(cfg, busy)


def test_game_detection_survives_rescaling():
    """A mirror window is smaller than the phone; detection must not care."""
    cfg = Config()
    for size in ((616, 1336), (400, 869), (300, 652)):
        assert looks_like_game(cfg, cv2.resize(load("frame_t53_both.png"), size)), size


def test_detection_follows_calibrated_drum_positions():
    """Move the drum targets somewhere blank and the check must fail, since it
    is reading those coordinates rather than assuming a layout."""
    cfg = Config()
    cfg.drum_red = (0.5, 0.15)
    cfg.drum_blue = (0.5, 0.20)
    assert not looks_like_game(cfg, load("frame_t59_red.png"))


def test_drum_fill_handles_targets_at_the_edge():
    """A drum point at the frame border must not crash or read a zero-size patch."""
    cfg = Config()
    cfg.drum_red = (0.0, 0.0)
    cfg.drum_blue = (1.0, 1.0)
    red_fill, blue_fill = drum_fill(cfg, load("frame_t59_red.png"))
    assert 0.0 <= red_fill <= 1.0 and 0.0 <= blue_fill <= 1.0


def test_example_config_ships_without_a_window_rect():
    """A placeholder rect in the example is a trap: copied to config.json it
    silently captures a patch of desktop that looks plausibly phone-shaped."""
    import json

    example = Path(__file__).resolve().parent.parent / "config.example.json"
    assert json.loads(example.read_text())["window_rect"] is None


# --------------------------------------------------------------------------
# finding the phone screen inside the mirroring window


def _animating_canvas(ox: int, oy: int, cw: int, ch: int, size=(322, 700)):
    """A static 'desktop' margin with the game animating inside it."""
    frames = [cv2.resize(load(n), size) for n in GAME_FRAMES]
    bg = np.random.default_rng(0).integers(0, 255, (ch, cw, 3), dtype=np.uint8)
    state = {"i": 0}

    def grab():
        canvas = bg.copy()
        canvas[oy: oy + size[1], ox: ox + size[0]] = frames[state["i"] % len(frames)]
        state["i"] += 1
        return canvas

    return grab, bg


def test_content_rect_is_found_by_motion():
    """The window includes phone-body margin with no title bar to trim, so the
    screen is located by what animates rather than by a guessed inset."""
    from vocalbot.capture import detect_content_rect

    ox, oy, w, h = 17, 11, 322, 700
    grab, _ = _animating_canvas(ox, oy, w + 2 * ox, h + 2 * oy)
    rect = detect_content_rect(grab, samples=5, delay=0.0)
    assert rect is not None
    x, y, rw, rh = rect
    # pad=2 deliberately grows the box; anything larger is a real error
    assert abs(x - ox) <= 3 and abs(y - oy) <= 3
    assert abs(rw - w) <= 6 and abs(rh - h) <= 6


def test_content_rect_returns_none_when_nothing_moves():
    """A paused game or a still screen must not yield a made-up rect."""
    from vocalbot.capture import detect_content_rect

    _, bg = _animating_canvas(10, 10, 340, 720)
    assert detect_content_rect(lambda: bg.copy(), samples=4, delay=0.0) is None


def test_detected_rect_finds_the_drums():
    """The point of detection: the resulting rect must make the drums line up."""
    from vocalbot.capture import detect_content_rect, snap_to_phone_aspect

    ox, oy, w, h = 17, 11, 322, 700
    grab, _ = _animating_canvas(ox, oy, w + 2 * ox, h + 2 * oy)
    x, y, rw, rh = snap_to_phone_aspect(detect_content_rect(grab, samples=5, delay=0.0))
    frame = grab()[y: y + rh, x: x + rw]
    assert looks_like_game(Config(), frame), drum_fill(Config(), frame)


def test_aspect_snap_corrects_a_near_miss():
    from vocalbot.capture import PHONE_ASPECT, snap_to_phone_aspect

    x, y, w, h = snap_to_phone_aspect((10, 20, 330, 700))
    assert abs((w / h) - PHONE_ASPECT) < 0.01
    assert 10 <= x <= 20 and y == 20  # re-centred, not shifted wholesale


def test_aspect_snap_leaves_a_wild_rect_alone():
    """Snapping a rect that isn't close would silently invent geometry."""
    from vocalbot.capture import snap_to_phone_aspect

    wild = (0, 0, 900, 200)
    assert snap_to_phone_aspect(wild) == wild
