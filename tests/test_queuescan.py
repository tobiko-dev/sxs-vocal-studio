"""Tests for the queue-model scanner.

The game holds a static queue: the front note sits in a fixed slot until it is
cleared, then everything shifts up. So the scanner's job is not "did a note
arrive" but "did the slot change", and it must fire exactly once per note.
"""

from __future__ import annotations

import numpy as np
import pytest

from vocalbot.color import sig_diff, signature
from vocalbot.config import Config, Zone
from vocalbot.queuescan import QueueScanner, resolve_taps

FRONT = (0.25, 0.65, 0.75, 0.70)


def cfg_with(**kw) -> Config:
    cfg = Config()
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


def sig(seed: int) -> np.ndarray:
    """A distinct but repeatable signature."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (4, 8), dtype=np.uint8)


def counts(red: int, blue: int) -> dict:
    return {name: (red, blue) for name in ("disco", "both", "red", "blue")}


def feed(scanner, frames, dt=1 / 60):
    """Push (counts, sig) pairs and collect dispatches."""
    out = []
    for i, (c, s) in enumerate(frames):
        shot = scanner.update(c, s, i * dt)
        if shot is not None:
            out.append(shot)
    return out


def hold(c, s, n):
    return [(c, s)] * n


# --------------------------------------------------------------------------
# firing once per note


def test_one_tap_per_note():
    cfg = cfg_with()
    scanner = QueueScanner(cfg)
    shots = feed(scanner, hold(counts(900, 0), sig(1), 30))
    assert len(shots) == 1
    assert shots[0].taps == ("red",)
    assert shots[0].reason == "advance"


def test_a_new_note_of_the_same_colour_still_fires():
    """The whole reason a signature exists: consecutive notes are often the same
    colour, so colour alone cannot tell you the queue advanced."""
    cfg = cfg_with()
    scanner = QueueScanner(cfg)
    frames = hold(counts(900, 0), sig(1), 20) + hold(counts(900, 0), sig(2), 20)
    shots = feed(scanner, frames)
    assert len(shots) == 2
    assert [s.reason for s in shots] == ["advance", "advance"]
    assert all(s.taps == ("red",) for s in shots)


def test_identical_note_does_not_refire():
    """An unchanged slot inside the retry window must not be tapped twice."""
    cfg = cfg_with(retry_ms=10_000.0)
    scanner = QueueScanner(cfg)
    shots = feed(scanner, hold(counts(900, 0), sig(1), 200))
    assert len(shots) == 1


def test_colour_change_fires_even_with_a_similar_signature():
    cfg = cfg_with(retry_ms=10_000.0)
    scanner = QueueScanner(cfg)
    s1, s2 = sig(1), sig(9)
    frames = hold(counts(900, 0), s1, 20) + hold(counts(0, 900), s2, 20)
    shots = feed(scanner, frames)
    assert [s.taps for s in shots] == [("red",), ("blue",)]


# --------------------------------------------------------------------------
# settling


def test_mid_animation_reads_are_not_tapped():
    """While the queue is shifting, the slot shows a transitional mixture. The
    scanner must wait for it to land rather than tapping what it sees."""
    cfg = cfg_with(settle_frames=4)
    scanner = QueueScanner(cfg)
    # signature churns for 3 frames, then settles on a red note
    churn = [(counts(900, 900), sig(i)) for i in range(50, 53)]
    shots = feed(scanner, churn + hold(counts(900, 0), sig(1), 20))
    assert len(shots) == 1
    assert shots[0].taps == ("red",)


def test_reading_must_be_stable_not_just_the_pixels():
    """The signature can go quiet while the outgoing note's bubble still tints
    the box, so a steady signature alone is not enough to trust the read."""
    cfg = cfg_with(settle_frames=3)
    scanner = QueueScanner(cfg)
    steady = sig(1)
    # pixels identical throughout, but the resolved reading flips for 2 frames
    frames = (
        hold(counts(900, 900), steady, 2)
        + hold(counts(900, 0), steady, 2)
        + hold(counts(900, 900), steady, 12)
    )
    shots = feed(scanner, frames)
    assert len(shots) == 1
    assert shots[0].taps == ("red", "blue")


def test_empty_slot_never_fires():
    cfg = cfg_with()
    scanner = QueueScanner(cfg)
    shots = feed(scanner, hold(counts(0, 0), sig(1), 40))
    assert shots == []
    assert scanner.idle_frames == 40


# --------------------------------------------------------------------------
# retries


def test_a_lost_tap_is_retried():
    """If the slot hasn't changed well past the round trip, the tap was lost."""
    cfg = cfg_with(retry_ms=100.0)
    scanner = QueueScanner(cfg)
    shots = feed(scanner, hold(counts(900, 0), sig(1), 60))  # 1 second
    assert len(shots) >= 2
    assert shots[0].reason == "advance"
    assert all(s.reason == "retry" for s in shots[1:])


def test_retry_does_not_fire_before_its_deadline():
    cfg = cfg_with(retry_ms=500.0)
    scanner = QueueScanner(cfg)
    shots = feed(scanner, hold(counts(900, 0), sig(1), 20))  # 333ms
    assert len(shots) == 1


def test_retries_are_counted_separately():
    cfg = cfg_with(retry_ms=100.0)
    scanner = QueueScanner(cfg)
    feed(scanner, hold(counts(900, 0), sig(1), 60))
    assert scanner.retries == scanner.dispatched - 1


# --------------------------------------------------------------------------
# tap resolution


def test_both_colours_tap_each_drum_once():
    cfg = cfg_with()
    taps = resolve_taps(cfg.active_zones(), counts(900, 900), cfg.tap_order)
    assert taps == ["red", "blue"]
    assert len(taps) == len(set(taps))


def test_resolution_follows_tap_order():
    cfg = cfg_with(tap_order=("blue", "red"))
    assert resolve_taps(cfg.active_zones(), counts(900, 900), cfg.tap_order) == ["blue", "red"]


def test_single_colour_taps_one_drum():
    cfg = cfg_with()
    assert resolve_taps(cfg.active_zones(), counts(900, 0), cfg.tap_order) == ["red"]
    assert resolve_taps(cfg.active_zones(), counts(0, 900), cfg.tap_order) == ["blue"]


def test_below_threshold_taps_nothing():
    cfg = cfg_with()
    assert resolve_taps(cfg.active_zones(), counts(10, 10), cfg.tap_order) == []


def test_a_zone_cannot_claim_the_same_drum_twice():
    """Overlapping zones that both want 'red' must still yield one press."""
    cfg = Config()
    cfg.zones = [
        Zone(name="a", rect=FRONT, match="red", taps=("red",), thresh_red=100, priority=5),
        Zone(name="b", rect=FRONT, match="red", taps=("red",), thresh_red=100, priority=0),
    ]
    got = resolve_taps(cfg.active_zones(), {"a": (900, 0), "b": (900, 0)}, cfg.tap_order)
    assert got == ["red"]


# --------------------------------------------------------------------------
# signature behaviour


def test_signature_is_stable_for_an_unchanged_region():
    region = np.random.default_rng(0).integers(0, 255, (60, 300, 3), dtype=np.uint8)
    assert sig_diff(signature(region), signature(region.copy())) == 0.0


def test_signature_separates_different_regions():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 90, (60, 300, 3), dtype=np.uint8)
    b = rng.integers(160, 255, (60, 300, 3), dtype=np.uint8)
    assert sig_diff(signature(a), signature(b)) > 40


def test_missing_signature_reads_as_maximally_different():
    """A None signature must count as 'changed', so the first note always fires."""
    assert sig_diff(None, signature(np.zeros((10, 10, 3), np.uint8))) == 255.0


def test_reset_clears_state():
    cfg = cfg_with()
    scanner = QueueScanner(cfg)
    feed(scanner, hold(counts(900, 0), sig(1), 20))
    assert scanner.dispatched == 1
    scanner.reset()
    assert scanner.dispatched == 0
    shots = feed(scanner, hold(counts(900, 0), sig(1), 20))
    assert len(shots) == 1  # fires again from a clean slate


@pytest.mark.parametrize("settle", [1, 3, 6, 10])
def test_settle_frames_delays_but_never_drops(settle):
    cfg = cfg_with(settle_frames=settle, retry_ms=10_000.0)
    scanner = QueueScanner(cfg)
    shots = feed(scanner, hold(counts(900, 0), sig(1), settle + 20))
    assert len(shots) == 1
