"""The live loop: grab strip -> classify -> edge detect -> tap."""

from __future__ import annotations

import time

from .capture import open_capture
from .color import build_lut, classify
from .scanner import Scanner
from .tap import TapWorker, make_tapper


class Bot:
    def __init__(self, cfg, dry_run: bool = False, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.lut = build_lut(cfg)
        self.scanner = Scanner(cfg)
        self.capture = open_capture(cfg)
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

                strip = self.capture.grab()
                red_px, blue_px = classify(strip, self.lut, self.cfg.step)
                fires = self.scanner.update(red_px, blue_px, frame_start)

                if fires:
                    self.worker.submit([f.color for f in fires])
                    if self.verbose:
                        combo = "+".join(f.color for f in fires)
                        print(
                            f"[{frame_start - t0:7.3f}s] {combo:<10s} "
                            f"red={red_px:4d} blue={blue_px:4d}"
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
        if self.frames:
            print(
                f"\n{self.frames} frames in {wall:.1f}s "
                f"({self.frames / max(wall, 1e-9):.0f} fps, "
                f"{self.loop_ms_total / self.frames:.2f} ms/frame)"
            )
            print(
                f"fired  red={self.scanner.fired['red']}  "
                f"blue={self.scanner.fired['blue']}  "
                f"taps={self.worker.dispatched}  dropped={self.worker.dropped}"
            )
