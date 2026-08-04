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


PHONE_ASPECT = 1206 / 2622  # 0.4600
MIN_WINDOW_PX = 200


def list_windows() -> list[dict]:
    """Every on-screen window: owner, title, bounds, layer. [] without Quartz."""
    try:
        import Quartz
    except ImportError:
        return []

    raw = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    out = []
    for win in raw or []:
        b = win.get("kCGWindowBounds") or {}
        out.append(
            {
                "owner": win.get("kCGWindowOwnerName", ""),
                "title": win.get("kCGWindowName", "") or "",
                "layer": int(win.get("kCGWindowLayer", 0)),
                "rect": (int(b.get("X", 0)), int(b.get("Y", 0)),
                         int(b.get("Width", 0)), int(b.get("Height", 0))),
            }
        )
    return out


def mirror_candidates() -> list[dict]:
    """Plausible iPhone Mirroring windows, largest first.

    The app owns more than one window - menu bar items and helper panels among
    them - and picking the first match can land on a tiny offscreen one whose
    bounds capture a patch of desktop instead of the phone. Degenerate sizes are
    dropped and the largest normal-layer window wins.
    """
    cands = []
    for win in list_windows():
        if win["owner"] not in MIRROR_APP_NAMES and "iphone mirroring" not in win["title"].lower():
            continue
        _, _, w, h = win["rect"]
        if w < MIN_WINDOW_PX or h < MIN_WINDOW_PX or win["layer"] != 0:
            continue
        cands.append(win)
    return sorted(cands, key=lambda w: -(w["rect"][2] * w["rect"][3]))


def find_mirror_window() -> tuple[int, int, int, int] | None:
    """Locate the iPhone Mirroring window. Returns (x, y, w, h) or None.

    The window is drawn as a phone shape with rounded corners, so the returned
    rect may include a little chrome; `calibrate window` trims and checks it.
    """
    cands = mirror_candidates()
    return cands[0]["rect"] if cands else None


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
