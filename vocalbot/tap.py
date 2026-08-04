"""Tap dispatch.

Taps run on their own thread. A tap costs `tap_hold_ms` of wall clock and the
scan loop must not pay for that, or a held tap would blind the detector to the
next note.

macOS has no public multi-touch injection API, so a "both drums" event is two
taps `inter_tap_ms` apart rather than a true simultaneous press. Rhythm hit
windows are tens of milliseconds wide, so a ~6ms stagger lands both inside it.
"""

from __future__ import annotations

import queue
import threading
import time


class QuartzTapper:
    """Posts synthetic clicks via CGEventPost. Needs Accessibility permission."""

    def __init__(self):
        import Quartz

        self._Q = Quartz
        self._down = Quartz.kCGEventLeftMouseDown
        self._up = Quartz.kCGEventLeftMouseUp
        self._button = Quartz.kCGMouseButtonLeft
        self._tap_loc = Quartz.kCGHIDEventTap

    def click(self, x: int, y: int, hold_s: float) -> None:
        Q = self._Q
        pt = Q.CGPointMake(x, y)
        # The position carries on the event itself, so no separate move is needed.
        Q.CGEventPost(self._tap_loc, Q.CGEventCreateMouseEvent(None, self._down, pt, self._button))
        if hold_s > 0:
            time.sleep(hold_s)
        Q.CGEventPost(self._tap_loc, Q.CGEventCreateMouseEvent(None, self._up, pt, self._button))


class DryRunTapper:
    """Records taps instead of posting them. Used by tests and `--dry-run`."""

    def __init__(self):
        self.events: list[tuple[float, int, int]] = []

    def click(self, x: int, y: int, hold_s: float) -> None:
        self.events.append((time.perf_counter(), x, y))


class TapWorker:
    """Serialises taps onto a background thread."""

    def __init__(self, cfg, tapper, screen_points: bool = False):
        self.cfg = cfg
        self.tapper = tapper
        # The guided setup records absolute screen coordinates, so no window
        # rect is needed to work out where a drum is.
        self.screen_points = screen_points
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="tap-worker")
        self.dispatched = 0
        self.dropped = 0

    def start(self) -> "TapWorker":
        self._thread.start()
        return self

    def submit(self, colors: list[str]) -> None:
        if self._q.qsize() > 8:
            # The queue only backs up if taps are being generated faster than
            # they can be posted. Dropping is better than falling further behind.
            self.dropped += 1
            return
        self._q.put(colors)

    def _run(self) -> None:
        hold = self.cfg.tap_hold_ms / 1000.0
        gap = self.cfg.inter_tap_ms / 1000.0
        while not self._stop.is_set():
            try:
                colors = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            for i, color in enumerate(colors):
                if i:
                    time.sleep(gap)
                x, y = (
                    self.cfg.drum_screen_point(color)
                    if self.screen_points
                    else self.cfg.drum_point(color)
                )
                self.tapper.click(x, y, hold)
                self.dispatched += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def make_tapper(dry_run: bool):
    if dry_run:
        return DryRunTapper()
    return QuartzTapper()
