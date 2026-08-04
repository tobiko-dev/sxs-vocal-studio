"""Screen capture, and locating the iPhone Mirroring window.

Only the union of the active zones is grabbed, never the whole window, and it is
grabbed once per frame rather than once per zone. Capture dominates the loop
budget, so this is the single biggest win available.
"""

from __future__ import annotations

import cv2
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


def cursor_pos():
    """Current cursor position, or None without Quartz. Needs no permission."""
    try:
        import Quartz
    except ImportError:
        return None
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return int(loc.x), int(loc.y)


def activate_mirror_app(delay: float = 0.6) -> bool:
    """Bring iPhone Mirroring to the front.

    Running a command from a terminal necessarily takes focus away from the
    mirroring window, and an unfocused window may stop animating - which is
    exactly when calibration is being run. Raising it first avoids diagnosing a
    frozen picture.
    """
    import subprocess
    import time

    for name in MIRROR_APP_NAMES:
        try:
            res = subprocess.run(
                ["osascript", "-e", f'tell application "{name}" to activate'],
                capture_output=True, timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if res.returncode == 0:
            time.sleep(delay)
            return True
    return False


def detect_content_rect(grab, samples: int = 5, delay: float = 0.06,
                        pad: int = 2) -> tuple[int, int, int, int] | None:
    """Find the live phone screen inside a window grab, by what moves.

    The mirroring window is drawn as a phone body with rounded corners and a
    shadow, so its bounds include margin that is not screen content. There is no
    title bar to trim and no fixed inset to guess - the padding depends on the
    window size.

    What is reliable is motion. The game animates continuously; whatever sits
    behind the window's transparent margin does not. Differencing a few frames
    and taking the bounding box of the changing pixels gives the content rect
    directly, whatever the padding happens to be.

    `grab` is a callable returning a BGR frame. Returns (x, y, w, h) relative to
    that frame, or None if nothing moved (the game is paused or on a still
    screen).
    """
    import time

    frames = []
    for i in range(max(2, samples)):
        frames.append(cv2.cvtColor(grab(), cv2.COLOR_BGR2GRAY).astype(np.int16))
        if i < samples - 1:
            time.sleep(delay)

    motion = np.zeros_like(frames[0], dtype=np.int16)
    for a, b in zip(frames[:-1], frames[1:]):
        np.maximum(motion, np.abs(b - a), out=motion)

    mask = (motion > 12).astype(np.uint8)
    if mask.sum() < 500:
        return None
    # Close gaps so static UI inside the screen doesn't split the region.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    h, w = mask.shape
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def snap_to_phone_aspect(rect: tuple[int, int, int, int],
                         tol: float = 0.10) -> tuple[int, int, int, int]:
    """Nudge a detected rect onto the phone's aspect, keeping its centre.

    Motion only bounds the region that animated, which can fall a few pixels
    short at an edge that happens to be static. Snapping recovers that when the
    detection is already close, and leaves it alone when it isn't.
    """
    x, y, w, h = rect
    if h <= 0:
        return rect
    aspect = w / h
    if abs(aspect - PHONE_ASPECT) / PHONE_ASPECT > tol:
        return rect
    if aspect > PHONE_ASPECT:  # too wide - trim width
        new_w = int(round(h * PHONE_ASPECT))
        return x + (w - new_w) // 2, y, new_w, h
    new_h = int(round(w / PHONE_ASPECT))
    return x, y + (h - new_h) // 2, w, new_h


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
