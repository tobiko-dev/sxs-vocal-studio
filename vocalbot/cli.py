"""Command line entry point.

    vocalbot calibrate window        pin the mirroring window, save its rect
    vocalbot calibrate zones         drag/resize the detection boxes (live)
    vocalbot calibrate drums         click the two drum targets
    vocalbot zone list|set|add|rm    edit zones without a GUI
    vocalbot doctor                  diagnose what the capture is pointing at
    vocalbot probe                   live per-zone counts, no tapping
    vocalbot bench                   measure capture and classify cost
    vocalbot replay RECORDING        run the detector over a recording, offline
    vocalbot run                     play
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import DRUMS, MATCHES, Config, Zone

DEFAULT_CONFIG = "config.json"


def _load(path: str) -> Config:
    if Path(path).exists():
        return Config.load(path)
    print(f"{path} not found, using defaults", file=sys.stderr)
    return Config()


def _frame(cfg, image: str | None):
    """A full phone-screen frame, from a file or a live grab."""
    import cv2

    if image:
        img = cv2.imread(image)
        if img is None:
            raise SystemExit(f"cannot read {image}")
        return img
    from .capture import grab_window

    if cfg.window_rect is None:
        raise SystemExit("no window_rect - run `vocalbot calibrate window`, or pass --image")
    return grab_window(cfg)


# --------------------------------------------------------------------------
def cmd_calibrate(args) -> int:
    import cv2

    from .calibrate import edit_drums, render_overlay, report, sample_counts

    cfg = _load(args.config)

    if args.what == "window":
        from .capture import find_mirror_window

        rect = find_mirror_window()
        if rect is None:
            print("iPhone Mirroring window not found. Open it, start the song, retry.")
            print('Or set it by hand in the config: "window_rect": [x, y, width, height]')
            return 1
        from .capture import PHONE_ASPECT, mirror_candidates

        cands = mirror_candidates()
        if len(cands) > 1:
            print(f"{len(cands)} candidate windows; using the largest:")
            for c in cands:
                print(f"   {c['rect']}  {c['title'] or '(untitled)'}")
        x, y, w, h = rect
        print(f"window at ({x}, {y}) {w}x{h}  aspect {w / h:.4f}")
        if args.chrome:
            y, h = y + args.chrome, h - args.chrome
            print(f"trimmed {args.chrome}px -> ({x}, {y}) {w}x{h}  aspect {w / h:.4f}")

        if args.rect:
            x, y, w, h = _parse_int_rect(args.rect)
            print(f"using --rect ({x}, {y}) {w}x{h}  aspect {w / h:.4f}")
        elif not args.no_auto:
            # The window is drawn as a phone body with rounded corners and a
            # shadow, so its bounds include margin that isn't screen content.
            # There's no title bar to trim and no fixed inset to guess, so find
            # the live screen by what moves instead.
            from .capture import (
                MSSCapture,
                activate_mirror_app,
                detect_content_rect,
                snap_to_phone_aspect,
            )

            # Running this from a terminal necessarily takes focus off the
            # mirroring window, and an unfocused window stops animating - which
            # is exactly when detection runs. Raise it first.
            if activate_mirror_app():
                print("brought iPhone Mirroring to the front")
            else:
                print("could not activate iPhone Mirroring - click it yourself, then retry")

            print("detecting the screen inside the window (needs the song playing)...")
            cap = MSSCapture((x, y, w, h))
            inner = detect_content_rect(cap.grab)
            cap.close()
            if inner is None:
                print("  nothing moved. The song must be RUNNING, not paused, and the")
                print("  mirroring window must be in front. Keeping the window bounds.")
                print("  Options: re-run with the song playing, pass --no-auto to accept")
                print("  these bounds, or set it by hand with --rect x,y,w,h.")
            else:
                ix, iy, iw, ih = snap_to_phone_aspect(inner)
                x, y, w, h = x + ix, y + iy, iw, ih
                print(f"  content at ({x}, {y}) {w}x{h}  aspect {w / h:.4f}")

        aspect = w / h
        off = abs(aspect - PHONE_ASPECT) / PHONE_ASPECT
        print(f"content aspect {aspect:.4f} (phone is {PHONE_ASPECT:.4f})")
        if off > 0.06:
            print(f"WARNING: off by {100 * off:.0f}%. Zone boxes will not line up.")

        cfg.window_rect = (x, y, w, h)
        cfg.save(args.config)
        print(f"saved to {args.config}")

        # Prove it grabbed the phone rather than a patch of desktop. This is the
        # failure that otherwise shows up much later as a calibrator full of
        # wallpaper, with nothing pointing at the cause.
        from .capture import grab_window
        from .doctor import drum_fill, looks_like_game

        img = grab_window(cfg)
        cv2.imwrite(args.out, img)
        print(f"wrote {args.out} - check it shows the phone screen")
        red_fill, blue_fill = drum_fill(cfg, img)
        print(f"drum targets: red {100 * red_fill:.0f}% blue {100 * blue_fill:.0f}%")
        if not looks_like_game(cfg, img):
            print("\nWARNING: this does not look like the game - the drums are not")
            print("where they should be. Run `vocalbot doctor` for the diagnosis.")
            return 1
        return 0

    if args.what == "zones":
        from .editor import run_editor

        # A callable, not a frame: with no --image this re-grabs every loop, so
        # counts move as the song plays and boxes can be placed against real
        # notes instead of a frozen still.
        if args.image:
            img = _frame(cfg, args.image)
            source = lambda: img  # noqa: E731
        else:
            from .capture import grab_window

            if cfg.window_rect is None:
                raise SystemExit("no window_rect - run `calibrate window`, or pass --image")
            source = lambda: grab_window(cfg)  # noqa: E731

        run_editor(cfg, source, args.config)
        img = source()
    else:
        img = _frame(cfg, args.image)
        if edit_drums(cfg, img):
            cfg.save(args.config)
            print(f"saved to {args.config}")

    print()
    print(report(cfg, img))
    cv2.imwrite(args.out, render_overlay(cfg, img, counts=sample_counts(cfg, img)))
    print(f"\nwrote {args.out}")
    return 0


def cmd_doctor(args) -> int:
    from .doctor import run

    return run(_load(args.config))


def cmd_zone(args) -> int:
    cfg = _load(args.config)

    if args.action == "list":
        from .calibrate import report

        if args.image:
            print(report(cfg, _frame(cfg, args.image)))
            return 0
        print(f"{'zone':<10} {'match':<6} {'taps':<10} {'prio':>4} {'thr r/b':>10}  rect")
        for z in sorted(cfg.zones, key=lambda z: -z.priority):
            rect = ", ".join(f"{v:.4f}" for v in z.rect)
            flag = "" if z.enabled else "  (disabled)"
            print(
                f"{z.name:<10} {z.match:<6} {'+'.join(z.taps):<10} {z.priority:>4} "
                f"{z.thresh_red:>4}/{z.thresh_blue:<5}  [{rect}]{flag}"
            )
        return 0

    if args.action == "rm":
        before = len(cfg.zones)
        cfg.zones = [z for z in cfg.zones if z.name != args.name]
        if len(cfg.zones) == before:
            print(f"no zone named {args.name!r}", file=sys.stderr)
            return 1
        cfg.save(args.config)
        print(f"removed {args.name}")
        return 0

    if args.action == "add":
        if any(z.name == args.name for z in cfg.zones):
            print(f"zone {args.name!r} already exists", file=sys.stderr)
            return 1
        if not args.rect:
            print("--rect is required when adding a zone", file=sys.stderr)
            return 1
        zone = Zone(name=args.name, rect=_parse_rect(args.rect))
        cfg.zones.append(zone)
    else:  # set
        try:
            zone = cfg.zone(args.name)
        except KeyError:
            print(f"no zone named {args.name!r}", file=sys.stderr)
            return 1

    if args.rect:
        zone.rect = _parse_rect(args.rect)
    if args.match:
        zone.match = args.match
    if args.taps:
        zone.taps = tuple(args.taps.split("+"))
    if args.thresh_red is not None:
        zone.thresh_red = args.thresh_red
    if args.thresh_blue is not None:
        zone.thresh_blue = args.thresh_blue
    if args.priority is not None:
        zone.priority = args.priority
    if args.refractory is not None:
        zone.refractory_ms = args.refractory
    if args.enable:
        zone.enabled = True
    if args.disable:
        zone.enabled = False

    Zone(**{**zone.__dict__})  # revalidate
    cfg.save(args.config)
    print(f"{zone.name}: match={zone.match} taps={'+'.join(zone.taps)} "
          f"thresh={zone.thresh_red}/{zone.thresh_blue} prio={zone.priority} "
          f"rect={tuple(round(v, 4) for v in zone.rect)} enabled={zone.enabled}")
    return 0


def _parse_int_rect(text: str) -> tuple[int, int, int, int]:
    parts = [int(float(p)) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit("--rect wants x,y,width,height in screen points")
    return tuple(parts)


def _parse_rect(text: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise SystemExit("--rect wants four fractions: x0,y0,x1,y1")
    return tuple(parts)


def cmd_bench(args) -> int:
    import numpy as np

    from .color import build_lut, classify

    cfg = _load(args.config)
    lut = build_lut(cfg)
    sw, sh = 1206, 2622  # size as if the phone were at native resolution
    regions = {}
    for rect, names in cfg.rect_groups().items():
        x0, y0, x1, y1 = cfg.zone(names[0]).pixel_rect(sw, sh)
        regions[rect] = np.random.randint(
            0, 255, (max(1, y1 - y0), max(1, x1 - x0), 3), dtype=np.uint8
        )

    ux0, uy0, ux1, uy1 = cfg.union_rect()
    union_px = int((ux1 - ux0) * sw) * int((uy1 - uy0) * sh)
    zone_px = sum(r.shape[0] * r.shape[1] for r in regions.values())
    print(f"zones           {len(cfg.active_zones())} active over {len(regions)} distinct boxes")
    print(f"union box       {int((ux1 - ux0) * sw)}x{int((uy1 - uy0) * sh)} = {union_px:,} px")
    print(f"zone pixels     {zone_px:,} px  ({100 * zone_px / max(union_px, 1):.0f}% of the union)")

    for _ in range(100):
        for name, px in regions.items():
            classify(px, lut, cfg.step)
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        for name, px in regions.items():
            classify(px, lut, cfg.step)
    classify_ms = (time.perf_counter() - t0) / n * 1000
    print(f"classify        {classify_ms:.4f} ms/frame  (step={cfg.step})")

    if cfg.window_rect is None:
        print("\nno window_rect - run `calibrate window` to benchmark capture too")
        return 0

    from .capture import ZoneCapture

    best = None
    for mode in ("union", "per_zone"):
        cfg.capture_mode = mode
        cap = ZoneCapture(cfg)
        cap.grab_zones()
        n = 300
        t0 = time.perf_counter()
        for _ in range(n):
            cap.grab_zones()
        ms = (time.perf_counter() - t0) / n * 1000
        cap.close()
        total = ms + classify_ms
        print(f"capture:{mode:<9} {ms:.4f} ms/frame  -> {1000 / total:.0f} fps ceiling")
        if best is None or ms < best[1]:
            best = (mode, ms)
    print(f"\nfastest capture_mode is {best[0]!r} - set it with:")
    print(f'  python -m vocalbot.cli --config {args.config} ...  (edit "capture_mode")')
    return 0


def cmd_probe(args) -> int:
    from .capture import ZoneCapture
    from .color import build_lut, classify, fan_out
    from .scanner import ZoneScanner

    cfg = _load(args.config)
    if cfg.window_rect is None:
        print("no window_rect - run `vocalbot calibrate window` first", file=sys.stderr)
        return 1
    lut = build_lut(cfg)
    scanner = ZoneScanner(cfg)
    cap = ZoneCapture(cfg)
    names = [z.name for z in cfg.active_zones()]
    print("per-zone red/blue counts. ctrl-c to stop.")
    print("  " + "  ".join(f"{n:>16}" for n in names))
    try:
        while True:
            by_rect = {
                rect: classify(px, lut, cfg.step, cap.norm)
                for rect, px in cap.grab_zones().items()
            }
            counts = fan_out(by_rect, cap.rect_groups)
            fires = scanner.update(counts, time.perf_counter())
            cells = []
            for name in names:
                r, b = counts[name]
                mark = "*" if any(f.zone == name for f in fires) else " "
                cells.append(f"{r:>7}/{b:<7}{mark}")
            print("\r  " + "  ".join(cells), end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()
    finally:
        cap.close()
    return 0


def cmd_replay(args) -> int:
    from .replay import report, write_overlay

    cfg = _load(args.config)
    report(args.recording, cfg, verbose=args.verbose)
    if args.overlay:
        write_overlay(args.recording, cfg, args.overlay)
    return 0


def cmd_run(args) -> int:
    from .bot import Bot, SampledBot
    from .wizard import run_setup

    cfg = _load(args.config)

    # Guided setup runs automatically the first time, so `run` is the only
    # command anyone needs to know.
    if args.setup or not cfg.is_setup():
        if not cfg.is_setup() and not args.setup:
            print("Not set up yet - starting guided setup.")
        if not run_setup(cfg, args.config):
            return 1
        cfg = _load(args.config)

    if cfg.is_setup():
        if args.countdown:
            for i in range(args.countdown, 0, -1):
                print(f"\rstarting in {i}...", end="", flush=True)
                time.sleep(1)
            print()
        SampledBot(cfg, dry_run=args.dry_run, verbose=not args.quiet).run(args.duration)
        return 0

    if cfg.window_rect is None:
        print("no window_rect - run `vocalbot calibrate window` first", file=sys.stderr)
        return 1
    if args.countdown:
        for i in range(args.countdown, 0, -1):
            print(f"\rstarting in {i}...", end="", flush=True)
            time.sleep(1)
        print()
    Bot(cfg, dry_run=args.dry_run, verbose=not args.quiet).run(args.duration)
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vocalbot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="place the window, zones and drums")
    c.add_argument("what", choices=("window", "zones", "drums"))
    c.add_argument("--image", help="calibrate against a saved screenshot instead of a live grab")
    c.add_argument("--out", default="calibration.png")
    c.add_argument("--chrome", type=int, default=0,
                   help="px to trim off the top before auto-detection (rarely needed)")
    c.add_argument("--no-auto", action="store_true",
                   help="skip motion-based content detection, use raw window bounds")
    c.add_argument("--rect", help="set the content rect by hand: x,y,width,height")
    c.set_defaults(func=cmd_calibrate)

    z = sub.add_parser("zone", help="inspect or edit zones without a GUI")
    z.add_argument("action", choices=("list", "set", "add", "rm"))
    z.add_argument("name", nargs="?")
    z.add_argument("--rect", help="x0,y0,x1,y1 as fractions of the phone screen")
    z.add_argument("--match", choices=MATCHES)
    z.add_argument("--taps", help="drums to press, e.g. red, blue, or red+blue",
                   metavar="|".join(DRUMS))
    z.add_argument("--thresh-red", type=int)
    z.add_argument("--thresh-blue", type=int)
    z.add_argument("--priority", type=int)
    z.add_argument("--refractory", type=float, help="ms")
    z.add_argument("--enable", action="store_true")
    z.add_argument("--disable", action="store_true")
    z.add_argument("--image", help="with `list`, report counts on this screenshot")
    z.set_defaults(func=cmd_zone)

    d = sub.add_parser("doctor", help="diagnose what the capture is pointing at")
    d.set_defaults(func=cmd_doctor)

    b = sub.add_parser("bench", help="measure capture and classify cost")
    b.set_defaults(func=cmd_bench)

    pr = sub.add_parser("probe", help="live per-zone counts, no tapping")
    pr.set_defaults(func=cmd_probe)

    t = sub.add_parser("replay", help="run the detector over a recording, offline")
    t.add_argument("recording")
    t.add_argument("--verbose", action="store_true", help="list every detected note")
    t.add_argument("--overlay", help="write an annotated video to this path")
    t.set_defaults(func=cmd_replay)

    r = sub.add_parser("run", help="play")
    r.add_argument("--dry-run", action="store_true", help="detect but never tap")
    r.add_argument("--duration", type=float, default=None)
    r.add_argument("--countdown", type=int, default=3)
    r.add_argument("--quiet", action="store_true")
    r.add_argument("--setup", action="store_true", help="re-run guided setup")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    if args.cmd == "zone" and args.action != "list" and not args.name:
        p.error(f"zone {args.action} needs a zone name")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
