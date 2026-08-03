"""Command line entry point: calibrate | bench | tune | run | probe."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Config

DEFAULT_CONFIG = "config.json"


def _load(path: str) -> Config:
    if Path(path).exists():
        return Config.load(path)
    print(f"{path} not found, using defaults", file=sys.stderr)
    return Config()


def cmd_calibrate(args) -> int:
    """Locate the mirroring window and dump an annotated screenshot to check
    the ROI, scanline and drum targets line up."""
    import cv2
    import numpy as np

    from .capture import find_mirror_window

    cfg = _load(args.config)
    rect = find_mirror_window()
    if rect is None:
        print("iPhone Mirroring window not found.")
        print("Open it, start the song, then re-run. To set the rect by hand:")
        print('  {"window_rect": [x, y, width, height]}')
        return 1

    x, y, w, h = rect
    print(f"window at ({x}, {y}) {w}x{h}")
    if args.chrome:
        y += args.chrome
        h -= args.chrome
        print(f"trimmed {args.chrome}px of title bar -> ({x}, {y}) {w}x{h}")
    cfg.window_rect = (x, y, w, h)

    import mss

    with mss.mss() as sct:
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
    img = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)[:, :, :3].copy()

    lx0, lx1 = int(cfg.lane_x0 * w), int(cfg.lane_x1 * w)
    sy, sh = int(cfg.scan_y * h), max(2, int(cfg.scan_h * h))
    cv2.rectangle(img, (lx0, sy), (lx1, sy + sh), (0, 255, 0), 2)
    cv2.putText(img, "scanline", (lx0, sy - 8), 0, 0.6, (0, 255, 0), 2)
    for color, bgr in (("red", (0, 0, 255)), ("blue", (255, 180, 0))):
        fx, fy = cfg.drum(color)
        cv2.circle(img, (int(fx * w), int(fy * h)), 18, bgr, 3)

    cv2.imwrite(args.out, img)
    cfg.save(args.config)
    print(f"wrote {args.out} and {args.config}")
    print("check that the green strip crosses the note runway and the circles sit")
    print("on the two drums; adjust lane_x0/lane_x1/scan_y/drum_* if not.")
    return 0


def cmd_bench(args) -> int:
    """Measure the real loop rate. Capture is the bottleneck, not the colour math."""
    import numpy as np

    from .color import build_lut, classify

    cfg = _load(args.config)
    lut = build_lut(cfg)

    fake = np.random.randint(0, 255, (24, 910, 3), dtype=np.uint8)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        classify(fake, lut, cfg.step)
    classify_ms = (time.perf_counter() - t0) / n * 1000
    print(f"classify        {classify_ms:.4f} ms/frame  (step={cfg.step})")

    if cfg.window_rect is None:
        print("no window_rect - run `calibrate` to benchmark capture too")
        return 0

    from .capture import open_capture

    cap = open_capture(cfg)
    cap.grab()
    n = 300
    t0 = time.perf_counter()
    for _ in range(n):
        strip = cap.grab()
    capture_ms = (time.perf_counter() - t0) / n * 1000
    cap.close()
    print(f"capture         {capture_ms:.4f} ms/frame  (strip {strip.shape[1]}x{strip.shape[0]})")
    total = capture_ms + classify_ms
    print(f"total           {total:.4f} ms/frame  -> {1000 / total:.0f} fps ceiling")
    return 0


def cmd_probe(args) -> int:
    """Print live pixel counts without tapping. Use to set thresholds by eye."""
    from .capture import open_capture
    from .color import build_lut, classify

    cfg = _load(args.config)
    lut = build_lut(cfg)
    cap = open_capture(cfg)
    print("red / blue glyph pixels at the scanline. ctrl-c to stop.\n")
    try:
        while True:
            red_px, blue_px = classify(cap.grab(), lut, cfg.step)
            bar = "#" * min(40, red_px) + "*" * min(40, blue_px)
            print(f"\rred={red_px:5d} blue={blue_px:5d} |{bar:<80s}", end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()
    finally:
        cap.close()
    return 0


def cmd_tune(args) -> int:
    from .tune import sweep

    cfg = _load(args.config)
    best = sweep(args.recording, cfg)
    if args.write:
        best.save(args.config)
        print(f"wrote tuned values to {args.config}")
    else:
        print("\nre-run with --write to save these to the config")
    return 0


def cmd_run(args) -> int:
    from .bot import Bot

    cfg = _load(args.config)
    if cfg.window_rect is None:
        print("no window_rect - run `vocalbot calibrate` first", file=sys.stderr)
        return 1
    if args.countdown:
        for i in range(args.countdown, 0, -1):
            print(f"\rstarting in {i}...", end="", flush=True)
            time.sleep(1)
        print()
    Bot(cfg, dry_run=args.dry_run, verbose=not args.quiet).run(args.duration)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vocalbot", description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="find the mirror window, dump an overlay")
    c.add_argument("--out", default="calibration.png")
    c.add_argument("--chrome", type=int, default=0, help="title bar px to trim")
    c.set_defaults(func=cmd_calibrate)

    b = sub.add_parser("bench", help="measure capture and classify cost")
    b.set_defaults(func=cmd_bench)

    pr = sub.add_parser("probe", help="live pixel counts, no tapping")
    pr.set_defaults(func=cmd_probe)

    t = sub.add_parser("tune", help="fit thresholds against a screen recording")
    t.add_argument("recording")
    t.add_argument("--write", action="store_true")
    t.set_defaults(func=cmd_tune)

    r = sub.add_parser("run", help="play")
    r.add_argument("--dry-run", action="store_true", help="detect but never tap")
    r.add_argument("--duration", type=float, default=None)
    r.add_argument("--countdown", type=int, default=3)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
