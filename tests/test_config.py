"""Tests for loading config files, including ones an older version wrote.

Settings get renamed and removed as the detector changes. A config carrying
fields that no longer exist should not stop the tool from starting - those
settings are simply out of date, and the current defaults are the right thing to
fall back to.
"""

from __future__ import annotations

import json

import pytest

from vocalbot.config import Config
from vocalbot.palette import Palette, Sample

# Exactly what the previous version wrote, back when the palette matched against
# sampled RGB colours instead of hue families.
OLD_CONFIG = {
    "drum_points": {"red": [1180, 700], "blue": [980, 700]},
    "sample_rect": [900, 560, 300, 120],
    "palette": {
        "samples": [
            {"name": "blue", "rgb": [221, 166, 32], "taps": ["blue"], "point": [1000, 600]},
            {"name": "red", "rgb": [157, 33, 222], "taps": ["red"], "point": [1100, 600]},
        ],
        "tolerance": 70,
        "min_pixels": 25,
        "relative_floor": 0.18,
        "unknown_is_both": True,
        "unknown_min": 150,
    },
    "tap_order": ["red", "blue"],
    "window_rect": [941, 109, 326, 720],
}


def write(tmp_path, data) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return str(path)


# --------------------------------------------------------------------------


def test_a_config_from_an_older_version_still_loads(tmp_path):
    """`tolerance` no longer exists. That must not be fatal."""
    cfg = Config.load(write(tmp_path, OLD_CONFIG))
    assert cfg.is_setup()
    assert cfg.drum_screen_point("red") == (1180, 700)
    assert cfg.sample_rect == (900, 560, 300, 120)


def test_stale_settings_are_reported(tmp_path, capsys):
    """Silently discarding settings would hide a real misconfiguration."""
    Config.load(write(tmp_path, OLD_CONFIG))
    assert "tolerance" in capsys.readouterr().err


def test_an_unknown_top_level_setting_is_dropped(tmp_path):
    cfg = Config.load(write(tmp_path, {**OLD_CONFIG, "removed_ages_ago": 42}))
    assert not hasattr(cfg, "removed_ages_ago")
    assert cfg.is_setup()


def test_hue_bands_fall_back_to_defaults_when_absent(tmp_path):
    """An older palette has no bands at all, so it must inherit current ones."""
    cfg = Config.load(write(tmp_path, OLD_CONFIG))
    assert [b.name for b in cfg.palette.bands] == ["blue", "red"]
    assert cfg.palette.band("blue").contains(99)
    assert cfg.palette.band("red").contains(160)


def test_old_samples_survive_the_upgrade(tmp_path):
    """They're no longer used for matching, but they still describe the display."""
    cfg = Config.load(write(tmp_path, OLD_CONFIG))
    assert {s.name for s in cfg.palette.samples} == {"blue", "red"}
    assert cfg.palette.by_name("blue").hue == pytest.approx(99, abs=3)


def test_an_older_config_needs_no_re_setup(tmp_path):
    """Classification is by hue band now, so an existing calibration is still
    usable - only the drum points and the watched region matter."""
    cfg = Config.load(write(tmp_path, OLD_CONFIG))
    assert cfg.is_setup()


def test_round_trip_of_the_current_format(tmp_path):
    cfg = Config()
    cfg.drum_points = {"red": (1, 2), "blue": (3, 4)}
    cfg.sample_rect = (5, 6, 70, 80)
    cfg.palette.samples = [Sample("blue", (221, 166, 32), ("blue",))]
    cfg.palette.band("blue").widen_to(123)

    path = tmp_path / "c.json"
    cfg.save(path)
    back = Config.load(path)

    assert back.drum_points == {"red": (1, 2), "blue": (3, 4)}
    assert back.sample_rect == (5, 6, 70, 80)
    assert back.palette.band("blue").contains(123)
    assert back.palette.by_name("blue").rgb == (221, 166, 32)


def test_a_config_with_only_bands_and_no_samples_works(tmp_path):
    """Samples are optional; the bands are what classify."""
    data = {**OLD_CONFIG, "palette": {"samples": []}}
    cfg = Config.load(write(tmp_path, data))
    assert cfg.palette.is_ready()
    assert cfg.palette.samples == []


def test_saved_bands_are_preferred_over_defaults(tmp_path):
    cfg = Config()
    cfg.palette.bands = [Palette().band("blue")]
    cfg.palette.bands[0].ranges = [(70, 130)]
    path = tmp_path / "c.json"
    cfg.save(path)
    back = Config.load(path)
    assert [b.ranges for b in back.palette.bands] == [[(70, 130)]]
