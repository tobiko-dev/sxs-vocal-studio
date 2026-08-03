"""The live loop.

    grab the front slot -> count colours -> has it changed? -> tap

The queue is static, so this is a closed loop: after tapping, the bot waits for
the slot to actually change before tapping again. That is what absorbs mirroring
latency - however long the round trip, the bot cannot run ahead of the game.
"""

from __future__ import annotations

import time

from .capture import ZoneCapture
from .color import build_lut, classify, fan_out, signature
from .queuescan import QueueScanner
from .tap import TapWorker, make_tapper


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
