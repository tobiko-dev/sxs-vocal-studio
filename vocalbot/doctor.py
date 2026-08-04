"""Diagnose what the capture path is actually pointing at.

The failure that looks most alarming - the calibrator showing your wallpaper -
is almost always a wrong `window_rect`, not a broken detector. The pixels are
being read faithfully from the wrong place. This prints everything needed to
tell which, and saves a PNG of exactly what is being grabbed.

    vocalbot doctor
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .capture import PHONE_ASPECT, list_windows, mirror_candidates


def _aspect_note(w: int, h: int) -> str:
    if h == 0:
        return "degenerate"
    a = w / h
    off = abs(a - PHONE_ASPECT) / PHONE_ASPECT
    if off < 0.04:
        return f"aspect {a:.4f} - matches the phone"
    return f"aspect {a:.4f} - phone is {PHONE_ASPECT:.4f}, off by {100 * off:.0f}%"


def _grab(rect):
    import mss

    x, y, w, h = rect
    with mss.mss() as sct:
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
    img = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)[:, :, :3].copy()
    return img, (shot.width, shot.height)


DRUM_PATCH = 0.045  # sampled radius, as a fraction of frame width
DRUM_MIN_FILL = 0.25


def drum_fill(cfg, img) -> tuple[float, float]:
    """Fraction of each drum target showing that drum's colour.

    The two drums are always on screen while a song is playing, at known
    positions, in strongly saturated red and blue. That makes them a far more
    specific test for "this is the game" than any texture measure - a detailed
    wallpaper can be busy, but it will not have a red disc and a blue disc
    exactly where the drums belong.
    """
    h, w = img.shape[:2]
    out = []
    for drum in ("red", "blue"):
        fx, fy = cfg.drum(drum)
        x, y, r = int(fx * w), int(fy * h), max(4, int(DRUM_PATCH * w))
        patch = img[max(0, y - r): y + r, max(0, x - r): x + r]
        if patch.size == 0:
            out.append(0.0)
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        if drum == "red":
            mask = ((H >= 150) | (H <= 12)) & (S > 80) & (V > 90)
        else:
            mask = (H >= 85) & (H <= 120) & (S > 60) & (V > 90)
        out.append(float(mask.mean()))
    return out[0], out[1]


def looks_like_game(cfg, img) -> bool:
    """True when both drums are visible where they should be."""
    red_fill, blue_fill = drum_fill(cfg, img)
    return red_fill >= DRUM_MIN_FILL and blue_fill >= DRUM_MIN_FILL


def _looks_like_wallpaper(img) -> bool:
    """Weak secondary hint: a flat desktop has little local contrast.

    Kept only as supporting detail. A busy wallpaper passes this easily, which
    is why `looks_like_game` is the test that actually decides.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var()) < 60.0


def run(cfg, out_dir: str = ".") -> int:
    out = Path(out_dir)
    problems: list[str] = []

    print("=" * 68)
    print("1. iPhone Mirroring window")
    print("=" * 68)
    cands = mirror_candidates()
    if not cands:
        print("  NOT FOUND.")
        owners = sorted({w["owner"] for w in list_windows() if w["owner"]})
        if not owners:
            print("  No window list at all - pyobjc-framework-Quartz is missing, or")
            print("  Screen Recording permission has not been granted to this terminal.")
            problems.append("cannot enumerate windows")
        else:
            print("  Visible app windows are:")
            for owner in owners:
                print(f"    - {owner}")
            problems.append("iPhone Mirroring is not running or not on screen")
    else:
        for i, win in enumerate(cands):
            x, y, w, h = win["rect"]
            mark = "  <- would be used" if i == 0 else ""
            print(f"  ({x}, {y}) {w}x{h}   {_aspect_note(w, h)}{mark}")
            if win["title"]:
                print(f"      title: {win['title']!r}")

    print()
    print("=" * 68)
    print("2. Saved window_rect")
    print("=" * 68)
    if cfg.window_rect is None:
        print("  UNSET - run `vocalbot calibrate window`.")
        problems.append("window_rect is unset")
    else:
        x, y, w, h = cfg.window_rect
        print(f"  ({x}, {y}) {w}x{h}   {_aspect_note(w, h)}")
        if cands:
            live = cands[0]["rect"]
            dx = abs(live[0] - x) + abs(live[1] - y)
            dw = abs(live[2] - w) + abs(live[3] - h)
            if dx > 40 or dw > 40:
                print(f"  MISMATCH: the live window is at {live}.")
                print("  The saved rect is stale - the window moved, or this rect was")
                print("  never calibrated. Re-run `vocalbot calibrate window`.")
                problems.append("saved window_rect does not match the live window")
            else:
                print("  matches the live window")

    print()
    print("=" * 68)
    print("3. What is actually being captured")
    print("=" * 68)
    if cfg.window_rect is None:
        print("  skipped - no window_rect")
    else:
        try:
            img, (gw, gh) = _grab(cfg.window_rect)
        except Exception as exc:  # noqa: BLE001
            print(f"  grab failed: {exc}")
            problems.append("screen grab failed")
        else:
            _, _, rw, rh = cfg.window_rect
            scale = gw / max(1, rw)
            print(f"  asked for {rw}x{rh}, got {gw}x{gh}  (scale {scale:.2f})")
            if scale > 1.5:
                print("  Retina display: the grab is in pixels, the rect in points.")
                print("  Harmless - zone rects are fractions and counts are normalised.")
            path = out / "doctor-capture.png"
            cv2.imwrite(str(path), img)
            print(f"  saved {path} - open it. It must show the phone screen.")

            red_fill, blue_fill = drum_fill(cfg, img)
            print(f"  drum targets: red {100 * red_fill:.0f}% blue {100 * blue_fill:.0f}% "
                  f"(a real game screen shows 60-70% at both)")
            if not looks_like_game(cfg, img):
                print("  NOT THE GAME. The drums are not where they should be, so this")
                print("  rect is pointing somewhere else - typically the desktop.")
                if _looks_like_wallpaper(img):
                    print("  (it also has very little local detail, consistent with wallpaper)")
                problems.append("captured region is not the game screen")

    if cands:
        try:
            img, _ = _grab(cands[0]["rect"])
            path = out / "doctor-window.png"
            cv2.imwrite(str(path), img)
            print(f"  saved {path} - this is what the detected window contains.")
        except Exception:  # noqa: BLE001
            pass

    print()
    print("=" * 68)
    if problems:
        print("PROBLEMS")
        for p in problems:
            print(f"  - {p}")
        print()
        print("Most likely fix:")
        print("  1. Open iPhone Mirroring and start the song")
        print("  2. python -m vocalbot.cli calibrate window")
        print("  3. python -m vocalbot.cli doctor        (confirm doctor-capture.png)")
        return 1
    print("No problems found.")
    return 0
