"""Replay a screen recording through the detector, offline.

This is how the bot gets validated without touching the game. It runs the exact
live decision path over a recording and reports the note sequence it would have
played, so a config change can be checked in seconds.

    vocalbot replay gameplay.mp4
    vocalbot replay gameplay.mp4 --overlay out.mp4

Note that a recording captures a *human* playing, so the pace in it is the
human's. Replay verifies that every note is seen exactly once and read
correctly; it cannot measure how fast the bot would go, because the queue only
advances when someone taps.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import ZoneReader, signature
from .config import Config
from .queuescan import QueueScanner, resolve_taps


@dataclass
class Shot:
    frame: int
    t: float
    taps: tuple[str, ...]
    reason: str
    red: int
    blue: int


def frames(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, i / fps, frame
        i += 1
    cap.release()


def replay(path: str, cfg: Config) -> tuple[list[Shot], list[str], int]:
    """Run the live decision path over a recording.

    Returns (shots, per-frame state labels, frame count).
    """
    scanner = QueueScanner(cfg)
    reader = None
    front = cfg.group("front")[0]
    shots: list[Shot] = []
    labels: list[str] = []
    n = 0

    for i, t, frame in frames(path):
        h, w = frame.shape[:2]
        if reader is None:
            reader = ZoneReader(cfg, w, h)
        counts = reader.read(frame)
        x0, y0, x1, y1 = front.pixel_rect(w, h)
        sig = signature(frame[y0:y1, x0:x1])

        labels.append("".join(resolve_taps(cfg.active_zones(), counts, cfg.tap_order)) or "-")
        shot = scanner.update(counts, sig, t)
        if shot is not None:
            red_px, blue_px = counts[front.name]
            shots.append(Shot(i, t, shot.taps, shot.reason, red_px, blue_px))
        n = i + 1

    return shots, labels, n


def stable_states(labels: list[str], min_frames: int = 5) -> list[str]:
    """Ground truth from the recording: runs of a steady front-slot reading.

    Each run is one note that sat in the slot until the player cleared it.
    """
    runs = []
    cur, count = labels[0], 1
    for label in labels[1:]:
        if label == cur:
            count += 1
        else:
            runs.append((cur, count))
            cur, count = label, 1
    runs.append((cur, count))
    return [k for k, c in runs if k != "-" and c >= min_frames]


def consistency(labels: list[str], shots: list[Shot]) -> float:
    """How steadily the box reads one answer for the whole time a note sits there.

    This is the metric worth tuning the box against. A box reaching too far up
    catches the note behind the front one and flickers between "red" and
    "red+blue" mid-note; a well-placed box holds one answer until the queue
    moves. 1.0 means every note read the same from arrival to clearing.
    """
    if not shots:
        return 0.0
    bounds = [s.frame for s in shots] + [len(labels)]
    scores = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        chunk = [k for k in labels[a:b] if k != "-"]
        if len(chunk) < 5:
            continue
        scores.append(max(chunk.count(k) for k in set(chunk)) / len(chunk))
    return float(np.mean(scores)) if scores else 0.0


def report(path: str, cfg: Config, verbose: bool = False) -> dict:
    shots, labels, n = replay(path, cfg)
    advances = [s for s in shots if s.reason == "advance"]
    retries = [s for s in shots if s.reason == "retry"]
    states = stable_states(labels)
    idle = sum(1 for k in labels if k == "-")

    print(f"{n} frames replayed")
    print(f"advances detected : {len(advances)}   <- one tap per note")
    print(f"retries           : {len(retries)}   <- slot unchanged past retry_ms")
    print(f"steady readings   : {len(states)}")
    print(f"empty slot        : {idle} frames ({100 * idle / max(1, n):.0f}%)")
    print(f"read consistency  : {consistency(labels, shots):.3f}   "
          f"<- 1.0 = the box never changed its mind mid-note")

    print(f"\ntap sequence: {' '.join(_short(s.taps) for s in advances)[:110]}")

    # A recording captures a human playing, so a note sits in the slot for as
    # long as they took to clear it. That sets the pace here, not the bot.
    if advances:
        span = advances[-1].t - advances[0].t
        print(f"\npace in this recording: {len(advances) / max(span, 1e-9):.2f} notes/sec "
              f"(the player's, not the bot's - the queue only moves when someone taps)")

    if verbose:
        print(f"\n{'frame':>7} {'t':>7}  taps        reason    red   blue")
        for s in shots:
            print(f"{s.frame:>7} {s.t:>7.2f}  {'+'.join(s.taps):<11s} {s.reason:<9s} "
                  f"{s.red:>5} {s.blue:>6}")

    return {
        "advances": len(advances),
        "retries": len(retries),
        "states": len(states),
        "consistency": consistency(labels, shots),
        "frames": n,
    }


def _short(taps) -> str:
    return "".join("R" if d == "red" else "B" for d in taps)


def _short_label(k: str) -> str:
    return "".join("R" if c == "r" else "B" for c in k.replace("red", "r").replace("blue", "b"))


def write_overlay(path: str, cfg: Config, out: str) -> None:
    """Render the recording with the boxes, counts and taps drawn on."""
    from .calibrate import render_overlay

    shots, _, _ = replay(path, cfg)
    fire_at = {s.frame: s for s in shots}
    writer = None
    reader = None
    for i, t, frame in frames(path):
        h, w = frame.shape[:2]
        if reader is None:
            reader = ZoneReader(cfg, w, h)
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (w, h))
        counts = reader.read(frame)
        view = render_overlay(cfg, frame, counts=counts)
        if i in fire_at:
            s = fire_at[i]
            cv2.rectangle(view, (0, 0), (w, 54), (0, 0, 0), -1)
            cv2.putText(view, f"TAP {'+'.join(s.taps)} ({s.reason})", (12, 38), 0, 1.1,
                        (0, 255, 0), 3)
        writer.write(view)
    if writer:
        writer.release()
    print(f"wrote {out}")
