"""Calibration: place the detection zones and the drum tap points.

Two ways in, because the GUI is not always the convenient one:

  vocalbot calibrate zones            drag and resize boxes (see editor.py)
  vocalbot zone set red --rect ...    edit a zone from the command line

Both write back to the same config. `render_overlay` is shared by the GUI, the
`calibrate` snapshot and the tests, and needs no display.
"""

from __future__ import annotations

import cv2
import numpy as np

from .color import ZoneReader

ZONE_BGR = {
    "red": (60, 60, 255),
    "blue": (255, 170, 40),
    "both": (200, 120, 255),
    "any": (120, 220, 120),
}
DRUM_BGR = {"red": (60, 60, 255), "blue": (255, 170, 40)}


def render_overlay(cfg, img: np.ndarray, counts: dict | None = None, active: str | None = None):
    """Draw zones, drum targets and (optionally) live counts onto a phone frame.

    `img` must be a full phone-screen frame, since zone rects are fractions of
    the whole screen.
    """
    out = img.copy()
    h, w = out.shape[:2]

    for zone in sorted(cfg.zones, key=lambda z: z.priority):
        x0, y0, x1, y1 = zone.pixel_rect(w, h)
        color = ZONE_BGR.get(zone.match, (200, 200, 200))
        if not zone.enabled:
            color = (110, 110, 110)
        thickness = 5 if zone.name == active else 2
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)

        label = f"{zone.name} [{zone.match}] -> {'+'.join(zone.taps)}"
        if not zone.enabled:
            label += " (off)"
        if counts and zone.name in counts:
            red_px, blue_px = counts[zone.name]
            label += f"  r={red_px}/{zone.thresh_red} b={blue_px}/{zone.thresh_blue}"
            if zone.hit(red_px, blue_px):
                label += "  HIT"
                cv2.rectangle(out, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 255, 0), 3)
        cv2.putText(out, label, (x0, max(20, y0 - 8)), 0, 0.62, color, 2, cv2.LINE_AA)

    for drum in ("red", "blue"):
        fx, fy = cfg.drum(drum)
        px, py = int(fx * w), int(fy * h)
        cv2.circle(out, (px, py), 26, DRUM_BGR[drum], 4)
        cv2.drawMarker(out, (px, py), DRUM_BGR[drum], cv2.MARKER_CROSS, 34, 3)
        cv2.putText(out, f"{drum} drum", (px - 52, py + 58), 0, 0.62, DRUM_BGR[drum], 2)

    return out


def sample_counts(cfg, img: np.ndarray) -> dict[str, tuple[int, int]]:
    """Per-zone (red, blue) counts on a full phone-screen frame."""
    h, w = img.shape[:2]
    return ZoneReader(cfg, w, h).read(img)


def report(cfg, img: np.ndarray) -> str:
    """Text summary of what every zone sees in this frame."""
    counts = sample_counts(cfg, img)
    lines = [f"{'zone':<10} {'match':<6} {'red':>7} {'blue':>7} {'thresholds':>14}   verdict"]
    for zone in cfg.active_zones():
        red_px, blue_px = counts[zone.name]
        verdict = "HIT -> " + "+".join(zone.taps) if zone.hit(red_px, blue_px) else "-"
        lines.append(
            f"{zone.name:<10} {zone.match:<6} {red_px:>7} {blue_px:>7} "
            f"{zone.thresh_red:>6}/{zone.thresh_blue:<7}   {verdict}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# interactive editors (need a GUI build of opencv)


def _require_gui():
    if not hasattr(cv2, "selectROI"):
        raise SystemExit(
            "this build of opencv has no GUI. Install opencv-python (not the "
            "headless variant), or use `vocalbot zone set` instead."
        )


def edit_drums(cfg, img: np.ndarray) -> bool:
    """Click the centre of each drum."""
    _require_gui()
    h, w = img.shape[:2]
    picked: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked.append((x, y))

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    order = ("red", "blue")
    while len(picked) < 2:
        preview = render_overlay(cfg, img)
        banner = f"click the centre of the {order[len(picked)].upper()} drum  |  ESC to cancel"
        cv2.putText(preview, banner, (20, 40), 0, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow("calibrate", preview)
        if cv2.waitKey(30) == 27:
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()
    cfg.drum_red = (round(picked[0][0] / w, 6), round(picked[0][1] / h, 6))
    cfg.drum_blue = (round(picked[1][0] / w, 6), round(picked[1][1] / h, 6))
    print(f"  red drum  {cfg.drum_red}")
    print(f"  blue drum {cfg.drum_blue}")
    return True
