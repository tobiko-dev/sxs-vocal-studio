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


def _empty_palette():
    from .palette import Palette

    return Palette()


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
    """Zones measured off the gameplay recording and the reference frames.

    The queue is static: the front note sits in a fixed slot until it is
    cleared, so the boxes go on that slot rather than anywhere up the runway.
    Verified frame by frame against a recording where the correct answer was
    read off screen - box `front` is 10/10 on those frames and reads single
    notes as single rather than over-reading the note behind them.

    The box was chosen by sweeping geometry against the recording and scoring
    each candidate on how consistently it reads the *same* answer for the whole
    time one note occupies the slot. A box reaching too far up starts catching
    the note behind and flickers between "red" and "red+blue" mid-note; this one
    scores 0.997. Narrower boxes score marginally higher but hug the centre
    closely enough to risk missing a note at the edge of a lane.

    Thresholds are in subsampled pixels (step=3), normalised to the reference
    screen so they hold at any mirror window size. At the front slot a note
    reads in the thousands against a floor near zero, so 200 sits far from both.

    The disco zone sits on the same box. Under the queue model the mirror ball
    simply *is* the front note when its turn comes, and it reads as blue there
    (r=0, b=1230-2385 in the recording) - indistinguishable by count from an
    ordinary blue note at 1444-1594, and not needing to be, since both tap blue.
    It is kept as an explicit, adjustable rule rather than a special case; being
    on the same box means it costs nothing to capture.
    """
    front = (0.25, 0.650, 0.75, 0.700)
    return [
        Zone(
            name="disco",
            rect=front,
            match="blue",  # only the blue count is consulted for this match
            taps=("blue",),
            thresh_blue=1000,
            priority=20,
            group="front",
        ),
        Zone(
            name="both",
            rect=front,
            match="both",
            taps=("red", "blue"),
            thresh_red=200,
            thresh_blue=200,
            priority=10,
            group="front",
        ),
        Zone(name="red", rect=front, match="red", taps=("red",), thresh_red=200,
             thresh_blue=200, priority=0, group="front"),
        Zone(name="blue", rect=front, match="blue", taps=("blue",), thresh_red=200,
             thresh_blue=200, priority=0, group="front"),
    ]


@dataclass
class Config:
    # --- playfield bounds, used by offline analysis --------------------------
    lane_x0: float = _f(150, REF_W)
    lane_x1: float = _f(1060, REF_W)

    # --- detection ------------------------------------------------------------
    zones: list[Zone] = field(default_factory=default_zones)
    step: int = 3  # subsample stride; thresholds are in subsampled pixels

    # --- queue advance detection ---------------------------------------------
    # The front slot is static between notes, so a tap is triggered by the slot
    # changing rather than by a note arriving. Measured on the recording:
    # 87% of frames sit under 3.0 (settled), 9% sit over 6.0 (mid-animation).
    # After a tap the queue animates the next note in; reading mid-animation is
    # a common source of wrong presses, so hold off before trusting the box.
    post_tap_ms: float = 140.0
    # Wall-clock limit so an unattended run cannot go on forever. The primary
    # way to stop is moving the mouse; this is a backstop.
    max_run_seconds: float | None = 150.0

    sig_stable: float = 3.0  # frame-to-frame diff counting as "not moving"
    sig_change: float = 6.0  # diff from the last tap counting as "new note"
    settle_frames: int = 3  # frames of stillness before trusting a read
    retry_ms: float = 900.0  # re-tap if the slot hasn't changed by now

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
    # Used by the diagnostics and the reference frames. The guided setup records
    # absolute screen points in `drum_points` instead, which needs no window
    # rect at all.
    drum_red: tuple[float, float] = (0.809, 0.802)
    drum_blue: tuple[float, float] = (0.185, 0.802)

    # --- guided setup results -------------------------------------------------
    # Absolute screen coordinates, learned from where you clicked. Nothing here
    # depends on knowing the mirroring window's bounds.
    drum_points: dict = field(default_factory=dict)  # {"red": (x, y), "blue": ...}
    sample_rect: tuple[int, int, int, int] | None = None  # region to watch
    palette: "Palette" = field(default_factory=lambda: _empty_palette())

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

    def is_setup(self) -> bool:
        """True when the guided setup has recorded everything needed to play."""
        return (
            self.sample_rect is not None
            and bool(self.drum_points.get("red"))
            and bool(self.drum_points.get("blue"))
            and self.palette.is_ready()
        )

    def drum_screen_point(self, color: str) -> tuple[int, int]:
        """Where to click for one drum, in absolute screen coordinates."""
        point = self.drum_points.get(color)
        if point:
            return int(point[0]), int(point[1])
        return self.drum_point(color)  # fall back to the window-relative path

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
        from .palette import HueBand, Palette, Sample

        raw = json.loads(Path(path).read_text())
        if isinstance(raw.get("palette"), dict):
            pal = dict(raw["palette"])
            pal["samples"] = [
                Sample(
                    name=s["name"],
                    rgb=tuple(s["rgb"]),
                    taps=tuple(s["taps"]),
                    point=tuple(s["point"]) if s.get("point") else None,
                )
                for s in pal.get("samples", [])
            ]
            pal["bands"] = [
                HueBand(name=b["name"], ranges=[tuple(r) for r in b["ranges"]],
                        taps=tuple(b["taps"]))
                for b in pal.get("bands", [])
            ] or None
            if pal["bands"] is None:
                pal.pop("bands")
            if "ignore_ranges" in pal:
                pal["ignore_ranges"] = [tuple(r) for r in pal["ignore_ranges"]]
            raw["palette"] = Palette(**pal)
        if isinstance(raw.get("sample_rect"), list):
            raw["sample_rect"] = tuple(raw["sample_rect"])
        if isinstance(raw.get("drum_points"), dict):
            raw["drum_points"] = {k: tuple(v) for k, v in raw["drum_points"].items()}
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
