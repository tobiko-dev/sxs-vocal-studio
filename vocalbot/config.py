"""Configuration for the scanline bot.

All playfield geometry is stored as fractions of the phone screen, measured on a
1206x2622 iPhone 16 Pro screenshot. Storing fractions rather than pixels means the
same config works whatever size the iPhone Mirroring window is pinned at.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Reference screenshot the fractions below were measured on.
REF_W, REF_H = 1206, 2622


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
class Config:
    # --- playfield geometry, as fractions of the phone screen -----------------
    lane_x0: float = 150 / REF_W  # 0.1244
    lane_x1: float = 1060 / REF_W  # 0.8790

    # Scanline sits high on the runway. Two reasons, both measured:
    #   1. Below ~y=1500 the frontmost note glyphs overlap into one continuous
    #      run, so consecutive same-colour notes merge into a single event.
    #   2. A higher line buys lead time to absorb mirroring latency.
    scan_y: float = 1400 / REF_H  # 0.5339
    scan_h: float = 24 / REF_H  # strip height

    # --- drum tap targets, as fractions of the phone screen -------------------
    drum_red: tuple[float, float] = (0.809, 0.802)
    drum_blue: tuple[float, float] = (0.185, 0.802)

    # --- colour gates ---------------------------------------------------------
    # Measured against the note glyphs, which are far more saturated than the
    # pale bubbles carrying them.
    red: ColorRule = field(
        default_factory=lambda: ColorRule(h_lo=150, h_hi=9, s_min=110, v_min=90, wrap=True)
    )
    blue: ColorRule = field(
        default_factory=lambda: ColorRule(h_lo=88, h_hi=118, s_min=90, v_min=90)
    )

    # --- detection tuning -----------------------------------------------------
    # Pixel counts are taken after subsampling by `step`, so thresholds are in
    # subsampled pixels. Noise floor at the scanline measures ~20 full-res px;
    # a real note is 150-1100. See tune.py to fit these to a recording.
    step: int = 3
    thresh_red: int = 10
    thresh_blue: int = 10
    hysteresis: float = 0.6  # re-arm once the count drops below thresh * this
    refractory_ms: float = 55.0  # min gap between two taps of the same colour

    # --- input ----------------------------------------------------------------
    tap_hold_ms: float = 12.0
    inter_tap_ms: float = 6.0  # gap when both drums fire together
    tap_order: tuple[str, str] = ("red", "blue")

    # --- capture --------------------------------------------------------------
    # Screen rect of the mirrored phone content, in macOS points:
    # [x, y, width, height]. Set by `vocalbot calibrate`.
    window_rect: tuple[int, int, int, int] | None = None
    backend: str = "mss"
    target_fps: int = 240  # loop ceiling; real rate is capture-bound

    # ------------------------------------------------------------------------
    def thresh(self, color: str) -> int:
        return self.thresh_red if color == "red" else self.thresh_blue

    def rule(self, color: str) -> ColorRule:
        return self.red if color == "red" else self.blue

    def drum(self, color: str) -> tuple[float, float]:
        return self.drum_red if color == "red" else self.drum_blue

    def strip_rect(self) -> tuple[int, int, int, int]:
        """Absolute screen rect of the scanline strip: (x, y, w, h)."""
        if self.window_rect is None:
            raise ValueError("window_rect is unset - run `vocalbot calibrate` first")
        wx, wy, ww, wh = self.window_rect
        x = wx + int(self.lane_x0 * ww)
        w = int((self.lane_x1 - self.lane_x0) * ww)
        y = wy + int(self.scan_y * wh)
        h = max(2, int(self.scan_h * wh))
        return x, y, w, h

    def drum_point(self, color: str) -> tuple[int, int]:
        """Absolute screen point to click for one drum."""
        if self.window_rect is None:
            raise ValueError("window_rect is unset - run `vocalbot calibrate` first")
        wx, wy, ww, wh = self.window_rect
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
        for key in ("drum_red", "drum_blue", "tap_order", "window_rect"):
            if isinstance(raw.get(key), list):
                raw[key] = tuple(raw[key])
        return cls(**raw)
