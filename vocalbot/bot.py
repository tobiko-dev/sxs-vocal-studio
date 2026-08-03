"""The live loop: grab the zone union -> count per zone -> resolve -> tap."""

from __future__ import annotations

import time

from .capture import ZoneCapture
from .color import build_lut, classify, fan_out
from .scanner import ZoneScanner
from .tap import TapWorker, make_tapper


class Bot:
    def __init__(self, cfg, dry_run: bool = False, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.lut = build_lut(cfg)
        self.scanner = ZoneScanner(cfg)
        self.capture = ZoneCapture(cfg)
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

                by_rect = {
                    rect: classify(px, self.lut, self.cfg.step)
                    for rect, px in self.capture.grab_zones().items()
                }
                counts = fan_out(by_rect, self.capture.rect_groups)
                fires = self.scanner.update(counts, frame_start)

                if fires:
                    taps = self.scanner.taps_for(fires)
                    self.worker.submit(taps)
                    if self.verbose:
                        which = ",".join(f.zone for f in fires)
                        print(
                            f"[{frame_start - t0:7.3f}s] {which:<12s} -> "
                            f"{'+'.join(taps):<10s} "
                            + " ".join(f"{n}={c[0]}/{c[1]}" for n, c in counts.items())
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
        per_zone = "  ".join(
            f"{name}={st.fired}" for name, st in self.scanner.state.items() if st.fired
        )
        print(f"fired  {per_zone or '(none)'}")
        print(
            f"taps={self.worker.dispatched}  dropped={self.worker.dropped}  "
            f"suppressed={self.scanner.suppressed}"
        )
