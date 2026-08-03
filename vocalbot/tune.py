"""Offline tuning against a screen recording.

Three values can't be derived from still frames because they depend on motion:
the scanline height, the pixel thresholds, and the refractory period. This module
fits them by using an expensive-but-accurate blob tracker over the full playfield
as ground truth, then sweeping the cheap scanline until it agrees.

    vocalbot tune song.mov --config config.json

Record 20-30 seconds of one song, ideally including a fever section.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from .color import build_lut, classify
from .config import Config
from .scanner import Scanner

# Playfield band used by the reference tracker. Wider than the scanline, and
# starting below the COMBO / FEVER banner so that text can't be mistaken for a
# blue note.
REF_Y0, REF_Y1 = 700 / 2622, 1960 / 2622
MIN_BLOB_AREA = 900
TEXT_MIN_W, TEXT_MAX_H = 250, 120  # the "COMBO" / "FEVER x2" overlay
DISCO_MIN_AREA = 20000
DISCO_ASPECT = (0.7, 1.4)


@dataclass
class Event:
    t: float
    color: str


def _frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i / fps, frame
        i += 1
    cap.release()


def _blobs(roi_bgr, cfg, y_off, x_off):
    """Note glyph blobs in the playfield ROI. Returns (color, cy, area, disco)."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    out = []
    kernel = np.ones((11, 11), np.uint8)
    for color in ("red", "blue"):
        mask = cfg.rule(color).mask(h, s, v).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, hh, area = stats[i]
            if area < MIN_BLOB_AREA:
                continue
            if w > TEXT_MIN_W and hh < TEXT_MAX_H:
                continue  # combo / fever banner
            disco = area > DISCO_MIN_AREA and DISCO_ASPECT[0] < w / float(hh) < DISCO_ASPECT[1]
            out.append((color, cent[i][1] + y_off, cent[i][0] + x_off, area, disco))
    return out


def reference_events(path: str, cfg: Config, line_frac: float | None = None) -> list[Event]:
    """Ground-truth events from blob tracking. Slow; offline only."""
    line_frac = cfg.scan_y if line_frac is None else line_frac
    tracks: list[dict] = []
    events: list[Event] = []
    next_id = 0

    for t, frame in _frames(path):
        fh, fw = frame.shape[:2]
        y0, y1 = int(REF_Y0 * fh), int(REF_Y1 * fh)
        x0, x1 = int(cfg.lane_x0 * fw), int(cfg.lane_x1 * fw)
        line_y = line_frac * fh
        found = _blobs(frame[y0:y1, x0:x1], cfg, y0, x0)

        for tr in tracks:
            tr["seen"] = False
        for color, cy, cx, area, disco in found:
            # Disco ball is played on the blue drum, per the game's rule.
            color = "blue" if disco else color
            best, best_d = None, 1e9
            for tr in tracks:
                if tr["color"] != color or tr["seen"]:
                    continue
                d = abs(tr["cy"] - cy) + abs(tr["cx"] - cx) * 0.5
                if d < best_d:
                    best, best_d = tr, d
            if best is not None and best_d < 140:
                best.update(cy=cy, cx=cx, seen=True)
            else:
                tracks.append(
                    {"id": next_id, "color": color, "cy": cy, "cx": cx, "seen": True, "fired": False}
                )
                next_id += 1

        for tr in tracks:
            if not tr["fired"] and tr["cy"] >= line_y:
                tr["fired"] = True
                events.append(Event(t, tr["color"]))
        tracks = [tr for tr in tracks if tr["seen"] and tr["cy"] < y1 - 40]

    return events


def scanline_events(path: str, cfg: Config) -> list[Event]:
    """Events the cheap live detector would produce on this recording."""
    lut = build_lut(cfg)
    scanner = Scanner(cfg)
    events: list[Event] = []
    for t, frame in _frames(path):
        fh, fw = frame.shape[:2]
        y = int(cfg.scan_y * fh)
        h = max(2, int(cfg.scan_h * fh))
        x0, x1 = int(cfg.lane_x0 * fw), int(cfg.lane_x1 * fw)
        red_px, blue_px = classify(frame[y : y + h, x0:x1], lut, cfg.step)
        for fire in scanner.update(red_px, blue_px, t):
            events.append(Event(t, fire.color))
    return events


def score(got: list[Event], want: list[Event], tol: float = 0.12) -> dict:
    """Match events by colour within `tol` seconds. Returns precision/recall/F1."""
    unused = sorted(want, key=lambda e: e.t)
    taken = [False] * len(unused)
    hits = 0
    for ev in got:
        for i, ref in enumerate(unused):
            if taken[i] or ref.color != ev.color:
                continue
            if abs(ref.t - ev.t) <= tol:
                taken[i] = True
                hits += 1
                break
    precision = hits / len(got) if got else 0.0
    recall = hits / len(want) if want else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "hits": hits,
        "got": len(got),
        "want": len(want),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sweep(path: str, cfg: Config) -> Config:
    """Grid search scanline position, thresholds and refractory against the
    reference tracker. Returns the best config found."""
    print("building reference (blob tracker, this is the slow part)...")
    ref = reference_events(path, cfg)
    print(f"reference: {len(ref)} notes "
          f"(red={sum(e.color == 'red' for e in ref)}, "
          f"blue={sum(e.color == 'blue' for e in ref)})\n")
    if not ref:
        raise SystemExit("reference found no notes - check lane_x0/lane_x1 and colour gates")

    best, best_cfg = None, cfg
    y_grid = [round(y / 2622, 4) for y in range(1150, 1560, 50)]
    thresh_grid = [4, 6, 8, 10, 14, 20, 28]
    refractory_grid = [40.0, 55.0, 70.0, 90.0]

    print(f"{'scan_y':>8} {'thresh':>7} {'refr':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for scan_y in y_grid:
        for thresh in thresh_grid:
            for refractory in refractory_grid:
                trial = copy.deepcopy(cfg)
                trial.scan_y = scan_y
                trial.thresh_red = trial.thresh_blue = thresh
                trial.refractory_ms = refractory
                res = score(scanline_events(path, trial), ref)
                if best is None or res["f1"] > best["f1"]:
                    best, best_cfg = res, trial
                    print(
                        f"{scan_y:8.4f} {thresh:7d} {refractory:6.0f} "
                        f"{res['precision']:6.3f} {res['recall']:6.3f} {res['f1']:6.3f}  <- best"
                    )

    print(
        f"\nbest: scan_y={best_cfg.scan_y}  thresh={best_cfg.thresh_red}  "
        f"refractory={best_cfg.refractory_ms}ms  F1={best['f1']:.3f} "
        f"({best['hits']}/{best['want']} notes)"
    )
    return best_cfg
