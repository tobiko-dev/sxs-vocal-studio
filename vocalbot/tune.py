"""Offline tuning against a screen recording.

Three things can't be derived from still frames because they depend on motion:
where the lane zones sit vertically, the pixel thresholds, and the refractory
period. This module fits them by using an expensive-but-accurate blob tracker
over the full playfield as ground truth, then sweeping the cheap zone detector
until it agrees.

    vocalbot tune song.mov --config config.json

Record 20-30 seconds of one song, ideally including a fever section.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import cv2
import numpy as np

from .color import ZoneReader  # normalises counts to the reference screen
from .config import Config
from .scanner import ZoneScanner

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
    """Ground-truth events from blob tracking. Slow; offline only.

    The line defaults to the vertical centre of the lane-group zones, so the
    reference and the cheap detector are measuring notes at the same depth.
    """
    if line_frac is None:
        lane = cfg.group("lane") or cfg.active_zones()
        line_frac = sum((z.rect[1] + z.rect[3]) / 2 for z in lane) / len(lane)
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


def zone_events(path: str, cfg: Config) -> list[Event]:
    """Events the cheap live detector would produce on this recording.

    One Event per drum pressed, so a "both" fire yields two, matching how the
    reference tracker counts a simultaneous pair.
    """
    scanner = ZoneScanner(cfg)
    reader = None
    events: list[Event] = []
    for t, frame in _frames(path):
        if reader is None:
            fh, fw = frame.shape[:2]
            reader = ZoneReader(cfg, fw, fh)
        for fire in scanner.update(reader.read(frame), t):
            for drum in fire.taps:
                events.append(Event(t, drum))
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


def _shift_group(cfg: Config, group: str, y_top: float) -> None:
    """Move every zone in a group so its box starts at y_top, keeping height."""
    for zone in cfg.group(group):
        x0, y0, x1, y1 = zone.rect
        zone.rect = (x0, y_top, x1, min(1.0, y_top + (y1 - y0)))


def sweep(path: str, cfg: Config) -> Config:
    """Grid search the lane zones' vertical position, thresholds and refractory
    against the reference tracker. Returns the best config found."""
    print("building reference (blob tracker, this is the slow part)...")
    ref = reference_events(path, cfg)
    print(f"reference: {len(ref)} notes "
          f"(red={sum(e.color == 'red' for e in ref)}, "
          f"blue={sum(e.color == 'blue' for e in ref)})\n")
    if not ref:
        raise SystemExit("reference found no notes - check lane_x0/lane_x1 and colour gates")

    best, best_cfg = None, cfg
    y_grid = [round(y / 2622, 4) for y in range(1150, 1560, 50)]
    thresh_grid = [(20, 15), (30, 22), (40, 30), (55, 42), (75, 60), (100, 80)]
    refractory_grid = [40.0, 55.0, 70.0, 90.0]

    print(f"{'lane_y':>8} {'thr r/b':>9} {'refr':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for lane_y in y_grid:
        for thresh_red, thresh_blue in thresh_grid:
            for refractory in refractory_grid:
                trial = copy.deepcopy(cfg)
                _shift_group(trial, "lane", lane_y)
                for zone in trial.group("lane"):
                    zone.thresh_red = thresh_red
                    zone.thresh_blue = thresh_blue
                    zone.refractory_ms = refractory
                res = score(zone_events(path, trial), ref)
                if best is None or res["f1"] > best["f1"]:
                    best, best_cfg = res, trial
                    print(
                        f"{lane_y:8.4f} {thresh_red:4d}/{thresh_blue:<4d} {refractory:6.0f} "
                        f"{res['precision']:6.3f} {res['recall']:6.3f} {res['f1']:6.3f}  <- best"
                    )

    lane = best_cfg.group("lane")[0]
    print(
        f"\nbest: lane_y={lane.rect[1]:.4f}  thresh={lane.thresh_red}/{lane.thresh_blue}  "
        f"refractory={lane.refractory_ms}ms  F1={best['f1']:.3f} "
        f"({best['hits']}/{best['want']} drum presses)"
    )
    print("note: the disco zone is not swept - set it with `vocalbot zone set disco`")
    return best_cfg
