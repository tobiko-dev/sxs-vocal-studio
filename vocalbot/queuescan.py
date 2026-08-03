"""Queue-model scanner: decide when the front note has changed, and what to tap.

The game holds a static queue. The front note sits in a fixed slot until it is
cleared, then the whole queue shifts up by one over a short animation. Nothing
scrolls, so there is no arrival to predict and no lead time to compensate for.

That makes the bot a closed loop rather than an open one:

    read the front slot -> tap the drums it needs -> wait for it to change -> repeat

Waiting for the change is what absorbs mirroring latency. However long the round
trip is, the bot simply doesn't tap again until it sees the queue move, so it
can never run ahead of the game.

Colour is not enough to detect the change, because consecutive notes are often
the same colour. A small greyscale signature of the slot is used instead: the
glyph shape, bubble position and lane all differ between notes even when the
colour doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass

from .color import sig_diff


@dataclass
class Dispatch:
    taps: tuple[str, ...]
    t: float
    reason: str  # "advance" or "retry"
    counts: dict


def resolve_taps(zones, counts, tap_order) -> list[str]:
    """Drums required by the current frame, resolved by zone priority.

    A firing zone claims the drums it taps and suppresses lower-priority zones
    whose drums it already covers, so a "both" zone doesn't stack with the
    separate red and blue zones.
    """
    claimed: set[str] = set()
    for zone in zones:  # already sorted by descending priority
        red_px, blue_px = counts.get(zone.name, (0, 0))
        if not zone.hit(red_px, blue_px):
            continue
        if set(zone.taps) <= claimed:
            continue
        claimed.update(zone.taps)
    return [d for d in tap_order if d in claimed]


class QueueScanner:
    """Fires once per note, when the front slot settles into a new state."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.zones = cfg.active_zones()
        self._prev_sig = None
        self._prev_taps: tuple[str, ...] | None = None
        self._settled_for = 0
        self._last_sig = None  # signature at the moment we last tapped
        self._last_t = -1e9
        self.dispatched = 0
        self.retries = 0
        self.idle_frames = 0

    def update(self, counts: dict, sig, t: float) -> Dispatch | None:
        taps = tuple(resolve_taps(self.zones, counts, self.cfg.tap_order))

        # The slot counts as settled only when the pixels have stopped moving
        # *and* the reading itself has stopped changing. The signature can go
        # quiet while the outgoing note's bubble still overlaps the box, which
        # would otherwise let a transitional colour reading be tapped.
        quiet = sig_diff(self._prev_sig, sig) <= self.cfg.sig_stable
        same_reading = taps == self._prev_taps
        if quiet and same_reading:
            self._settled_for += 1
        else:
            self._settled_for = 0
        self._prev_sig = sig
        self._prev_taps = taps

        if not taps:
            self.idle_frames += 1
            return None

        if self._settled_for < self.cfg.settle_frames:
            return None  # mid-animation; wait for it to land

        changed = sig_diff(self._last_sig, sig) > self.cfg.sig_change
        stale = (t - self._last_t) * 1000.0 > self.cfg.retry_ms

        if not (changed or stale):
            return None

        self._last_sig = sig
        self._last_t = t
        self.dispatched += 1
        if not changed:
            self.retries += 1
        return Dispatch(taps, t, "advance" if changed else "retry", dict(counts))

    def reset(self) -> None:
        self._prev_sig = None
        self._prev_taps = None
        self._settled_for = 0
        self._last_sig = None
        self._last_t = -1e9
        self.dispatched = 0
        self.retries = 0
        self.idle_frames = 0
