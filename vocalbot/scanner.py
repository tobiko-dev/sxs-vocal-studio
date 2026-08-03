"""Zone state machine: per-zone pixel counts in, tap events out.

Each zone keeps its own rising-edge state, hysteresis and refractory period. A
zone fires when its match rule goes true after having been clear.

Zones are then resolved by priority: a firing zone claims the drums it taps, and
suppresses any lower-priority zone whose taps it already covers. That is what
keeps the "both" zone from double-tapping alongside the separate red and blue
zones, and the disco zone from stacking with a plain blue detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fire:
    zone: str
    taps: tuple[str, ...]
    t: float
    red_px: int
    blue_px: int
    suppressed_by: str | None = None


@dataclass
class _ZoneState:
    high: bool = False
    last_fire: float = -1e9
    fired: int = 0
    counts: tuple[int, int] = field(default=(0, 0))


class ZoneScanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.zones = cfg.active_zones()
        self.state = {z.name: _ZoneState() for z in self.zones}
        self.suppressed = 0

    def update(self, counts: dict[str, tuple[int, int]], t: float) -> list[Fire]:
        """Feed one frame of per-zone (red_px, blue_px). `t` is monotonic seconds.

        Every zone's edge state is advanced regardless of suppression, so a
        suppressed zone stays latched and cannot re-fire on the next frame.
        """
        candidates: list[Fire] = []

        for zone in self.zones:  # already sorted by descending priority
            st = self.state[zone.name]
            red_px, blue_px = counts.get(zone.name, (0, 0))
            st.counts = (red_px, blue_px)

            if zone.hit(red_px, blue_px):
                if not st.high and (t - st.last_fire) >= zone.refractory_ms / 1000.0:
                    st.last_fire = t
                    st.fired += 1
                    candidates.append(Fire(zone.name, zone.taps, t, red_px, blue_px))
                st.high = True
            elif zone.clear(red_px, blue_px):
                st.high = False

        return self._resolve(candidates)

    def _resolve(self, candidates: list[Fire]) -> list[Fire]:
        """Drop lower-priority fires whose drums a higher-priority fire covers."""
        claimed: set[str] = set()
        winners: list[Fire] = []
        for fire in candidates:  # descending priority
            if set(fire.taps) <= claimed:
                fire.suppressed_by = winners[-1].zone if winners else "?"
                self.suppressed += 1
                continue
            claimed.update(fire.taps)
            winners.append(fire)
        return winners

    def taps_for(self, fires: list[Fire]) -> list[str]:
        """Flatten fires into a drum press order, without repeats."""
        out: list[str] = []
        for fire in fires:
            for drum in self.cfg.tap_order:
                if drum in fire.taps and drum not in out:
                    out.append(drum)
        return out

    def reset(self) -> None:
        self.state = {z.name: _ZoneState() for z in self.zones}
        self.suppressed = 0
