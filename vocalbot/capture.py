"""Screen capture, and locating the iPhone Mirroring window.

Only the union of the active zones is grabbed, never the whole window, and it is
grabbed once per frame rather than once per zone. Capture dominates the loop
budget, so this is the single biggest win available.
"""

from __future__ import annotations

import numpy as np

MIRROR_APP_NAMES = ("iPhone Mirroring", "iPhoneMirroring")


class MSSCapture:
    """mss-backed region grabber. Portable and fast enough for small regions."""

    def __init__(self, rect: tuple[int, int, int, int]):
        import mss  # imported lazily so tests run without a display

        self._sct = mss.mss()
        x, y, w, h = rect
        self._mon = {"left": x, "top": y, "width": w, "height": h}

    def grab(self) -> np.ndarray:
        """Return the captured region as an (h, w, 3) BGR array."""
        shot = self._sct.grab(self._mon)
        return np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)[
            :, :, :3
        ]

    def close(self) -> None:
        self._sct.close()


class QuartzCapture:
    """CoreGraphics fallback. Use if mss benchmarks poorly on your macOS build."""

    def __init__(self, rect: tuple[int, int, int, int]):
        import Quartz  # noqa: F401

        self._Quartz = Quartz
        self._rect = rect

    def grab(self) -> np.ndarray:
        Q = self._Quartz
        x, y, w, h = self._rect
        img = Q.CGWindowListCreateImage(
            Q.CGRectMake(x, y, w, h),
            Q.kCGWindowListOptionOnScreenOnly,
            Q.kCGNullWindowID,
            Q.kCGWindowImageBoundsIgnoreFraming | Q.kCGWindowImageNominalResolution,
        )
        width = Q.CGImageGetWidth(img)
        height = Q.CGImageGetHeight(img)
        stride = Q.CGImageGetBytesPerRow(img)
        data = Q.CGDataProviderCopyData(Q.CGImageGetDataProvider(img))
        buf = np.frombuffer(data, dtype=np.uint8)
        return buf.reshape(height, stride // 4, 4)[:, :width, :3]

    def close(self) -> None:
        pass


def open_capture(cfg):
    rect = cfg.capture_rect()
    if cfg.backend == "quartz":
        return QuartzCapture(rect)
    return MSSCapture(rect)


def find_mirror_window() -> tuple[int, int, int, int] | None:
    """Locate the iPhone Mirroring window. Returns (x, y, w, h) or None.

    The window has a title bar above the mirrored phone content; the returned
    rect is the full window, so `calibrate` trims the chrome.
    """
    try:
        import Quartz
    except ImportError:
        return None

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    for win in windows or []:
        owner = win.get("kCGWindowOwnerName", "")
        if owner in MIRROR_APP_NAMES:
            b = win["kCGWindowBounds"]
            return int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"])
    return None


def grab_window(cfg) -> "np.ndarray":
    """Grab the whole mirrored phone screen. Used by calibration, which needs a
    full frame because zone rects are fractions of the entire screen."""
    import mss

    x, y, w, h = cfg._require_window()
    with mss.mss() as sct:
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
    return np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)[:, :, :3].copy()


class ZoneCapture:
    """Grabs the pixels each zone needs, in whichever way is cheaper.

    "union" takes one grab covering every zone and slices it, which wins when
    the zones sit close together. "per_zone" takes one grab per zone, which wins
    when the union would be mostly dead space. Both return the same thing, so the
    mode is purely a performance choice - measure it with `vocalbot bench`.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.capture_mode
        wx, wy, ww, wh = cfg._require_window()
        # Counts are normalised to the reference screen so one set of thresholds
        # survives any mirror window size.
        from .color import normalizer

        self.norm = normalizer(ww, wh)
        # Keyed by rect, not by zone: the lane strip backs three zones and is
        # only worth grabbing once.
        self.rect_groups = cfg.rect_groups()

        if self.mode == "per_zone":
            self._caps = {}
            self.pixels = 0
            for rect, names in self.rect_groups.items():
                x0, y0, x1, y1 = cfg.zone(names[0]).pixel_rect(ww, wh, wx, wy)
                self._caps[rect] = _backend(cfg, (x0, y0, x1 - x0, y1 - y0))
                self.pixels += (x1 - x0) * (y1 - y0)
        else:
            cx, cy, cw, ch = cfg.capture_rect()
            self._cap = _backend(cfg, (cx, cy, cw, ch))
            self._slices = {}
            for rect, names in self.rect_groups.items():
                x0, y0, x1, y1 = cfg.zone(names[0]).pixel_rect(ww, wh, wx - cx, wy - cy)
                self._slices[rect] = (
                    slice(max(0, y0), max(0, y1)),
                    slice(max(0, x0), max(0, x1)),
                )
            self.pixels = cw * ch

    def grab_zones(self) -> dict:
        """Return {rect: BGR array}. Fan out to zone names with color.fan_out."""
        if self.mode == "per_zone":
            return {rect: cap.grab() for rect, cap in self._caps.items()}
        frame = self._cap.grab()
        return {rect: frame[ys, xs] for rect, (ys, xs) in self._slices.items()}

    def close(self) -> None:
        if self.mode == "per_zone":
            for cap in self._caps.values():
                cap.close()
        else:
            self._cap.close()


def _backend(cfg, rect):
    if cfg.backend == "quartz":
        return QuartzCapture(rect)
    return MSSCapture(rect)
