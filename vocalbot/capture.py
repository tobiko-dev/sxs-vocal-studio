"""Screen capture, and locating the iPhone Mirroring window.

Only the scanline strip is grabbed, never the whole window. Capture dominates the
loop budget, so a 910x24 grab instead of 910x1260 is the single biggest win
available.
"""

from __future__ import annotations

import numpy as np

MIRROR_APP_NAMES = ("iPhone Mirroring", "iPhoneMirroring")


class MSSCapture:
    """mss-backed strip grabber. Portable and fast enough for small regions."""

    def __init__(self, rect: tuple[int, int, int, int]):
        import mss  # imported lazily so tests run without a display

        self._sct = mss.mss()
        x, y, w, h = rect
        self._mon = {"left": x, "top": y, "width": w, "height": h}

    def grab(self) -> np.ndarray:
        """Return the strip as an (h, w, 3) BGR array."""
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
    rect = cfg.strip_rect()
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
