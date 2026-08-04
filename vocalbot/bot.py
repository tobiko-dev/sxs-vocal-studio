"""The live loop.

    grab the front slot -> count colours -> has it changed? -> tap

The queue is static, so this is a closed loop: after tapping, the bot waits for
the slot to actually change before tapping again. That is what absorbs mirroring
latency - however long the round trip, the bot cannot run ahead of the game.
"""

from __future__ import annotations

import time

from .capture import MSSCapture, ZoneCapture, cursor_pos
from .failsafe import Failsafe
from .color import build_lut, classify, fan_out, signature, sig_diff
from .palette import count_matches, decide_taps
from .queuescan import QueueScanner
from .tap import TapWorker, make_tapper


class SampledBot:
    """The bot as driven by the guided setup.

    Reads one screen region, matches every pixel to the colours you clicked, and
    presses the drums at the points you clicked. Nothing here needs the mirroring
    window's bounds, a fractional layout, or a colour threshold - all of it came
    from calibration.
    """

    def __init__(self, cfg, dry_run: bool = False, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.capture = MSSCapture(cfg.sample_rect)
        self.tapper = make_tapper(dry_run)
        self.worker = TapWorker(cfg, self.tapper, screen_points=True)
        self.failsafe = Failsafe(
            drum_points=[cfg.drum_screen_point(d) for d in ("red", "blue")],
            duration=cfg.max_run_seconds,
        )
        self.frames = 0
        self.loop_ms_total = 0.0
        self.notes = 0
        self.retries = 0
        self._hold_until = 0.0
        # Advance detection, unchanged: colour alone can't tell you the queue
        # moved, because consecutive notes are often the same colour.
        self._prev_sig = None
        self._prev_taps = None
        self._settled = 0
        self._last_sig = None
        self._last_t = -1e9

    def step(self, frame, t):
        counts = count_matches(frame, self.cfg.palette, self.cfg.step)
        taps = tuple(decide_taps(counts, self.cfg.palette, self.cfg.tap_order))
        sig = signature(frame)

        quiet = sig_diff(self._prev_sig, sig) <= self.cfg.sig_stable
        if quiet and taps == self._prev_taps:
            self._settled += 1
        else:
            self._settled = 0
        self._prev_sig, self._prev_taps = sig, taps

        if not taps or self._settled < self.cfg.settle_frames:
            return None, counts

        # After a tap the queue animates the next note in. Reading during that
        # gives a half-drawn bubble, which is where a lot of wrong presses come
        # from, so hold off until the art has had time to land.
        if t < self._hold_until:
            return None, counts

        changed = sig_diff(self._last_sig, sig) > self.cfg.sig_change
        stale = (t - self._last_t) * 1000.0 > self.cfg.retry_ms
        if not (changed or stale):
            return None, counts

        self._last_sig, self._last_t = sig, t
        self._hold_until = t + self.cfg.post_tap_ms / 1000.0
        self.notes += 1
        if not changed:
            self.retries += 1
        return (taps, "advance" if changed else "retry"), counts

    def run(self, duration: float | None = None) -> None:
        self.worker.start()
        budget = 1.0 / self.cfg.target_fps
        t0 = time.perf_counter()
        print("playing - ctrl-c to stop\n")
        try:
            while True:
                frame_start = time.perf_counter()
                if duration is not None and frame_start - t0 >= duration:
                    break
                fired, counts = self.step(self.capture.grab(), frame_start)
                if fired:
                    taps, reason = fired
                    self.worker.submit(list(taps))
                    if self.verbose:
                        seen = " ".join(
                            f"{k}={v}" for k, v in counts.items() if k != "background" and v
                        )
                        print(f"[{frame_start - t0:7.3f}s] {'+'.join(taps):<9s} "
                              f"({reason})  {seen}")
                self.frames += 1
                elapsed = time.perf_counter() - frame_start
                self.loop_ms_total += elapsed * 1000.0
                if elapsed < budget:
                    time.sleep(budget - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            self.worker.stop()
            self.capture.close()
            wall = time.perf_counter() - t0
            print(f"\nstopped: {self.failsafe.reason or 'finished'}")
            if self.frames:
                print(f"\n{self.frames} frames in {wall:.1f}s "
                      f"({self.frames / max(wall, 1e-9):.0f} fps, "
                      f"{self.loop_ms_total / self.frames:.2f} ms/frame)")
                print(f"notes={self.notes}  retries={self.retries}  "
                      f"taps={self.worker.dispatched}  dropped={self.worker.dropped}")


class Bot:
    def __init__(self, cfg, dry_run: bool = False, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.lut = build_lut(cfg)
        self.scanner = QueueScanner(cfg)
        self.capture = ZoneCapture(cfg)
        # The signature is taken from the box the front-slot zones share.
        self.sig_rect = cfg.zone(cfg.group("front")[0].name).rect
        self.tapper = make_tapper(dry_run)
        self.worker = TapWorker(cfg, self.tapper)
        self.frames = 0
        self.loop_ms_total = 0.0

    def run(self, duration: float | None = None) -> None:
        self.worker.start()
        budget = 1.0 / self.cfg.target_fps
        t0 = time.perf_counter()
        try:
            while True:
                frame_start = time.perf_counter()
                if duration is not None and frame_start - t0 >= duration:
                    break

                regions = self.capture.grab_zones()
                by_rect = {
                    rect: classify(px, self.lut, self.cfg.step, self.capture.norm)
                    for rect, px in regions.items()
                }
                counts = fan_out(by_rect, self.capture.rect_groups)
                sig = signature(regions[self.sig_rect])
                shot = self.scanner.update(counts, sig, frame_start)

                if shot is not None:
                    self.worker.submit(list(shot.taps))
                    if self.verbose:
                        red_px, blue_px = counts[self.cfg.group("front")[0].name]
                        print(
                            f"[{frame_start - t0:7.3f}s] {'+'.join(shot.taps):<9s} "
                            f"({shot.reason})  r={red_px:5d} b={blue_px:5d}"
                        )

                self.frames += 1
                elapsed = time.perf_counter() - frame_start
                self.loop_ms_total += elapsed * 1000.0
                if elapsed < budget:
                    time.sleep(budget - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown(time.perf_counter() - t0)

    def shutdown(self, wall: float) -> None:
        self.worker.stop()
        self.capture.close()
        if not self.frames:
            return
        print(
            f"\n{self.frames} frames in {wall:.1f}s "
            f"({self.frames / max(wall, 1e-9):.0f} fps, "
            f"{self.loop_ms_total / self.frames:.2f} ms/frame)"
        )
        print(
            f"notes={self.scanner.dispatched}  retries={self.scanner.retries}  "
            f"idle={self.scanner.idle_frames}"
        )
        print(f"taps={self.worker.dispatched}  dropped={self.worker.dropped}")
