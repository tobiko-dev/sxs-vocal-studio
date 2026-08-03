"""Configuration for the zone-based bot.

Everything positional is stored as a fraction of the phone screen, measured on a
1206x2622 iPhone 16 Pro screenshot. Fractions are scale-invariant, so the same
config works whatever size the iPhone Mirroring window is pinned at.

Detection is driven by a list of Zones rather than hardcoded geometry. A zone is
a box plus a rule for what counts as a hit inside it plus the drums to press.
Add, move, retune or disable zones with `vocalbot calibrate zones`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Reference screenshot the fractions below were measured on.
REF_W, REF_H = 1206, 2622

MATCHES = ("red", "blue", "both", "any")
DRUMS = ("red", "blue")


def _f(px: float, ref: int) -> float:
    return round(px / ref, 6)


@dataclass
class ColorRule:
    """HSV gate for one note colour. Hue is OpenCV's 0-179 range."""

    h_lo: int
    h_hi: int
    s_min: int
    v_min: int
    wrap: bool = False  # True when the hue range crosses 0 (red/magenta)

    def mask(self, h, s, v):
        if self.wrap:
            hue_ok = (h >= self.h_lo) | (h <= self.h_hi)
        else:
            hue_ok = (h >= self.h_lo) & (h <= self.h_hi)
        return hue_ok & (s > self.s_min) & (v > self.v_min)


@dataclass
class Zone:
    """One detection box.

    rect      (x0, y0, x1, y1) as fractions of the phone screen
    match     what counts as a hit:
                "red"  - red count over threshold
                "blue" - blue count over threshold
                "both" - both counts over their thresholds at once
                "any"  - either count over its threshold
    taps      which drums to press when it fires
    priority  higher wins; a firing zone suppresses lower-priority zones whose
              taps it already covers, so "both" doesn't double up with the
              separate red and blue zones
    group     zones sharing a group move together during tuning
    """

    name: str
    rect: tuple[float, float, float, float]
    match: str = "red"
    taps: tuple[str, ...] = ("red",)
    thresh_red: int = 40
    thresh_blue: int = 30
    priority: int = 0
    refractory_ms: float = 55.0
    hysteresis: float = 0.6
    group: str = "lane"
    enabled: bool = True

    def __post_init__(self):
        if self.match not in MATCHES:
            raise ValueError(f"{self.name}: match must be one of {MATCHES}, got {self.match!r}")
        bad = [d for d in self.taps if d not in DRUMS]
        if bad:
            raise ValueError(f"{self.name}: unknown drum(s) {bad}, expected {DRUMS}")
        x0, y0, x1, y1 = self.rect
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError(f"{self.name}: rect {self.rect} is not a valid fractional box")

    def thresh(self, color: str) -> int:
        return self.thresh_red if color == "red" else self.thresh_blue

    def hit(self, red_px: int, blue_px: int) -> bool:
        r = red_px >= self.thresh_red
        b = blue_px >= self.thresh_blue
        if self.match == "red":
            return r
        if self.match == "blue":
            return b
        if self.match == "both":
            return r and b
        return r or b  # "any"

    def clear(self, red_px: int, blue_px: int) -> bool:
        """True once counts have fallen clearly away from the thresholds."""
        r = red_px < self.thresh_red * self.hysteresis
        b = blue_px < self.thresh_blue * self.hysteresis
        if self.match == "red":
            return r
        if self.match == "blue":
            return b
        return r and b  # "both" and "any" both re-arm once everything is quiet

    def pixel_rect(self, w: int, h: int, ox: int = 0, oy: int = 0) -> tuple[int, int, int, int]:
        """Fractional rect -> absolute pixel rect (x0, y0, x1, y1)."""
        x0, y0, x1, y1 = self.rect
        return (
            ox + int(x0 * w),
            oy + int(y0 * h),
            ox + max(int(x0 * w) + 1, int(x1 * w)),
            oy + max(int(y0 * h) + 1, int(y1 * h)),
        )


def default_zones() -> list[Zone]:
    """Zones measured off the reference frames in assets/frames.

    The lane strip sits high on the runway (y ~0.534). Below about 0.60 the
    frontmost note glyphs overlap into one continuous run, so consecutive
    same-colour notes merge into a single detection. Placing it high also buys
    the lead time that absorbs mirroring latency.

    Thresholds are in subsampled pixels (step=3). Measured at the lane strip:
    red signal 128-281 against a floor of 0; blue signal ~50 against a floor of
    13.

    The disco box is deliberately small and sits inside where the mirror ball
    passes: it reads 1113 blue on the disco frame against at most 2 on every
    other frame. A wider box catches stray note colour and collapses that margin
    to about 5x, as well as enlarging the captured union.
    """
    lane = (_f(150, REF_W), _f(1400, REF_H), _f(1060, REF_W), _f(1424, REF_H))
    disco = (_f(485, REF_W), _f(1580, REF_H), _f(725, REF_W), _f(1700, REF_H))
    return [
        Zone(
            name="disco",
            rect=disco,
            match="blue",  # only the blue count is consulted for this match
            taps=("blue",),
            thresh_blue=300,
            priority=20,
            refractory_ms=400.0,  # the ball is large and lingers
            group="disco",
        ),
        Zone(
            name="both",
            rect=lane,
            match="both",
            taps=("red", "blue"),
            thresh_red=40,
            thresh_blue=30,
            priority=10,
        ),
        Zone(name="red", rect=lane, match="red", taps=("red",), thresh_red=40, priority=0),
        Zone(name="blue", rect=lane, match="blue", taps=("blue",), thresh_blue=30, priority=0),
    ]


@dataclass
class Config:
    # --- playfield bounds, used by the offline reference tracker --------------
    lane_x0: float = _f(150, REF_W)
    lane_x1: float = _f(1060, REF_W)

    # --- detection ------------------------------------------------------------
    zones: list[Zone] = field(default_factory=default_zones)
    step: int = 3  # subsample stride; thresholds are in subsampled pixels

    # --- colour gates ---------------------------------------------------------
    # Gated on the note glyphs, which are far more saturated than the pale
    # bubbles carrying them. Fever mode tints the screen gold but does not shift
    # note hue, so one set of gates covers both modes.
    red: ColorRule = field(
        default_factory=lambda: ColorRule(h_lo=150, h_hi=9, s_min=110, v_min=90, wrap=True)
    )
    blue: ColorRule = field(
        default_factory=lambda: ColorRule(h_lo=88, h_hi=118, s_min=90, v_min=90)
    )

    # --- drum tap targets, as fractions of the phone screen -------------------
    drum_red: tuple[float, float] = (0.809, 0.802)
    drum_blue: tuple[float, float] = (0.185, 0.802)

    # --- input ----------------------------------------------------------------
    tap_hold_ms: float = 12.0
    inter_tap_ms: float = 6.0  # gap when both drums fire together
    tap_order: tuple[str, str] = ("red", "blue")

    # --- capture --------------------------------------------------------------
    # Screen rect of the mirrored phone content in macOS points: [x, y, w, h].
    # Set by `vocalbot calibrate window`.
    window_rect: tuple[int, int, int, int] | None = None
    backend: str = "mss"
    # "union" grabs one box covering every zone and slices it; "per_zone" grabs
    # each zone separately. Union wins when the zones are close together, per_zone
    # when they are far apart and the union is mostly dead space. `vocalbot bench`
    # measures both.
    capture_mode: str = "union"
    target_fps: int = 240  # loop ceiling; the real rate is capture-bound

    # ------------------------------------------------------------------------
    def rule(self, color: str) -> ColorRule:
        return self.red if color == "red" else self.blue

    def drum(self, color: str) -> tuple[float, float]:
        return self.drum_red if color == "red" else self.drum_blue

    def active_zones(self) -> list[Zone]:
        """Enabled zones, highest priority first."""
        return sorted([z for z in self.zones if z.enabled], key=lambda z: -z.priority)

    def zone(self, name: str) -> Zone:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(f"no zone named {name!r}")

    def group(self, group: str) -> list[Zone]:
        return [z for z in self.zones if z.group == group]

    def rect_groups(self) -> dict:
        """{rect: [zone names]} over active zones.

        Several zones deliberately share one box - the lane strip carries the
        red, blue and both zones - so the pixels behind it are captured and
        counted once and the result fanned out.
        """
        groups: dict = {}
        for zone in self.active_zones():
            groups.setdefault(tuple(zone.rect), []).append(zone.name)
        return groups

    def union_rect(self) -> tuple[float, float, float, float]:
        """Smallest fractional box containing every active zone. Captured in one
        grab per frame, then sliced per zone."""
        zones = self.active_zones()
        if not zones:
            raise ValueError("no zones are enabled")
        return (
            min(z.rect[0] for z in zones),
            min(z.rect[1] for z in zones),
            max(z.rect[2] for z in zones),
            max(z.rect[3] for z in zones),
        )

    def _require_window(self) -> tuple[int, int, int, int]:
        if self.window_rect is None:
            raise ValueError("window_rect is unset - run `vocalbot calibrate window` first")
        return self.window_rect

    def capture_rect(self) -> tuple[int, int, int, int]:
        """Absolute screen rect to grab each frame: (x, y, w, h)."""
        wx, wy, ww, wh = self._require_window()
        x0, y0, x1, y1 = self.union_rect()
        x, y = wx + int(x0 * ww), wy + int(y0 * wh)
        return x, y, max(2, int((x1 - x0) * ww)), max(2, int((y1 - y0) * wh))

    def drum_point(self, color: str) -> tuple[int, int]:
        """Absolute screen point to click for one drum."""
        wx, wy, ww, wh = self._require_window()
        fx, fy = self.drum(color)
        return wx + int(fx * ww), wy + int(fy * wh)

    # ------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        for key in ("red", "blue"):
            if isinstance(raw.get(key), dict):
                raw[key] = ColorRule(**raw[key])
        if isinstance(raw.get("zones"), list):
            zones = []
            for z in raw["zones"]:
                z = dict(z)
                z["rect"] = tuple(z["rect"])
                z["taps"] = tuple(z["taps"])
                zones.append(Zone(**z))
            raw["zones"] = zones
        for key in ("drum_red", "drum_blue", "tap_order", "window_rect"):
            if isinstance(raw.get(key), list):
                raw[key] = tuple(raw[key])
        return cls(**raw)
