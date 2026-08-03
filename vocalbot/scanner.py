"""Scanline state machine: pixel counts in, tap events out.

One instance holds the edge state for both colours. A tap fires on the rising
edge of a colour's pixel count crossing its threshold, gated by a refractory
period so a note that straddles several frames only fires once.
"""

from __future__ import annotations

from dataclasses import dataclass

COLORS = ("red", "blue")


@dataclass
class Fire:
    color: str
    t: float
    px: int


class Scanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self._high = {c: False for c in COLORS}
        self._last_fire = {c: -1e9 for c in COLORS}
        self.counts = {c: 0 for c in COLORS}
        self.fired = {c: 0 for c in COLORS}

    def update(self, red_px: int, blue_px: int, t: float) -> list[Fire]:
        """Feed one frame. `t` is seconds, monotonic. Returns taps to dispatch."""
        fires: list[Fire] = []
        px_by_color = {"red": red_px, "blue": blue_px}
        refractory = self.cfg.refractory_ms / 1000.0

        for color in self.cfg.tap_order:
            px = px_by_color[color]
            self.counts[color] = px
            on = px >= self.cfg.thresh(color)

            if on:
                if not self._high[color] and (t - self._last_fire[color]) >= refractory:
                    self._last_fire[color] = t
                    self.fired[color] += 1
                    fires.append(Fire(color, t, px))
                self._high[color] = True
            elif px < self.cfg.thresh(color) * self.cfg.hysteresis:
                # Hysteresis: only re-arm once the count falls clearly away from
                # the threshold, so a note flickering at the boundary can't
                # double-fire.
                self._high[color] = False

        return fires

    def reset(self) -> None:
        self._high = {c: False for c in COLORS}
        self._last_fire = {c: -1e9 for c in COLORS}
        self.fired = {c: 0 for c in COLORS}
