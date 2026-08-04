"""Guided setup: follow the prompts, click the things, done.

Everything the bot needs is learned from where you click:

  - clicking each drum records where to press it
  - clicking each note colour records what that colour actually is on your
    display, and roughly where notes appear

That removes the two things that kept going wrong. There is no window rect to
work out, because drum positions are recorded in absolute screen coordinates.
And there are no colour thresholds to guess, because the colours are sampled
from your screen rather than from reference screenshots.

## Why you hover instead of clicking

Targets are picked by hovering and holding still, not by clicking or pressing a
key. That is deliberate.

Reading the global keyboard or mouse-button state needs macOS's **Input
Monitoring** permission, which is separate from Accessibility and easy to miss -
and when it is missing the calls do not fail, they just silently report that
nothing is pressed. Reading the *cursor position* needs no permission at all.
So setup only asks where the pointer is, which cannot silently do nothing.

Dwell also sidesteps a smaller problem: a click inside the mirroring window is
forwarded to the phone, so picking targets by clicking taps the game as a side
effect.

Stepping between prompts uses ordinary stdin, since at that moment there is
nothing to hover and the terminal can hold focus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

POLL = 1 / 60
DWELL_RADIUS = 8  # px of wobble tolerated while holding still
DWELL_SECONDS = 1.2


@dataclass
class Step:
    key: str
    prompt: str
    detail: str
    optional: bool = False
    taps: tuple[str, ...] = ()


DRUM_STEPS = [
    Step("blue_drum", "Hover over the BLUE drum",
         "The blue drum pad, bottom left. This is where blue notes get pressed."),
    Step("red_drum", "Hover over the RED drum",
         "The red drum pad, bottom right."),
]

COLOUR_STEPS = [
    Step("blue", "Hover over the most VIVID part of a BLUE note",
         "The solid, strongly coloured core of the note symbol at the front of "
         "the queue - not the pale bubble around it, and not a shadowed edge.",
         taps=("blue",)),
    Step("red", "Hover over the most VIVID part of a RED note",
         "The solid red or magenta core of the symbol, same idea.",
         taps=("red",)),
    Step("both", "Hover over a WHITE 'both' area",
         "Only if a distinct marker appears when two notes arrive together. "
         "Skipping is fine - anything that matches neither note counts as both.",
         optional=True, taps=("red", "blue")),
    Step("disco", "Hover over the PURPLE part of the disco ball",
         "Only if a mirror ball is on screen right now, and only if its purple "
         "looks strong rather than pastel - a washed-out sample also matches the "
         "pale bubbles and causes phantom presses. Skipping is usually better: "
         "the ball reads as blue on its own.",
         optional=True, taps=("blue",)),
]


class Pointer:
    """Where the cursor is. Needs no special permission, unlike key state."""

    def __init__(self):
        import Quartz

        self.Q = Quartz

    def pos(self) -> tuple[int, int]:
        loc = self.Q.CGEventGetLocation(self.Q.CGEventCreate(None))
        return int(loc.x), int(loc.y)

    def wait_for_dwell(self, **kw) -> tuple[int, int] | None:
        return dwell(self.pos, **kw)


def dwell(pos_fn, radius: int = DWELL_RADIUS, hold: float = DWELL_SECONDS,
          timeout: float = 180.0, on_tick=None, clock=time.perf_counter,
          sleep=time.sleep) -> tuple[int, int] | None:
    """Return where the cursor rested once it has held still for `hold`.

    Moving beyond `radius` restarts the hold, so overshooting a target costs
    nothing - keep adjusting until it locks. `clock` and `sleep` are injectable
    so the timing logic can be tested without real waiting.
    """
    deadline = clock() + timeout
    anchor = pos_fn()
    settled_at = clock()

    while clock() < deadline:
        now = clock()
        here = pos_fn()
        if abs(here[0] - anchor[0]) > radius or abs(here[1] - anchor[1]) > radius:
            anchor, settled_at = here, now
        held = now - settled_at
        if on_tick:
            on_tick(here, min(held / hold, 1.0))
        if held >= hold:
            return anchor
        sleep(POLL)
    return None


def sample_colour(point: tuple[int, int], radius: int = 5) -> tuple[int, int, int]:
    """Representative BGR near a screen point: the vivid core, not the average.

    Two reasons this isn't just the pixel under the cursor. Note art is shaded
    and anti-aliased, so a single pixel is a poor representative. More
    importantly, the *dark* parts of the blue and red notes are nearly the same
    colour - measured 52 apart, against 246 for their vivid cores - so sampling
    a shadowed pixel would make the two classes indistinguishable. Taking the
    most saturated quartile of the patch recovers the core even if the click
    lands slightly off it.
    """
    import mss

    x, y = point
    with mss.mss() as sct:
        shot = sct.grab({"left": x - radius, "top": y - radius,
                         "width": 2 * radius + 1, "height": 2 * radius + 1})
    px = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)[:, :, :3]
    return vivid_colour(px)


def vivid_colour(patch: np.ndarray, share: float = 0.7, min_keep: int = 4):
    """Median of the most strongly coloured pixels in a BGR patch.

    Selection is relative to the most vivid pixel present, not a fixed quantile.
    A click that only clips the edge of a note leaves the patch mostly pale, and
    a quantile would then average pale pixels and return the bubble colour - the
    exact sample that goes on to match everything. Keying off the patch's own
    maximum keeps the note's colour even when it is a small minority.
    """
    flat = patch.reshape(-1, 3).astype(np.int32)
    spread = flat.max(axis=1) - flat.min(axis=1)
    peak = int(spread.max())
    keep = spread >= share * peak if peak > 0 else np.ones(len(flat), dtype=bool)
    if keep.sum() < min_keep:
        keep = spread >= np.percentile(spread, 90)
    if not keep.any():
        keep = np.ones(len(flat), dtype=bool)
    return tuple(int(c) for c in np.median(flat[keep], axis=0))


def sample_rect_from_points(points, pad_x: int = 90, pad_y: int = 55, min_size: int = 40):
    """The box to watch, derived from where the note colours were clicked.

    Both note positions are known from calibration, so the region to read is
    simply the area around them - no fractions of a screen whose bounds were
    never established.
    """
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs) - pad_x, max(xs) + pad_x
    y0, y1 = min(ys) - pad_y, max(ys) + pad_y
    return (int(x0), int(y0), max(min_size, int(x1 - x0)), max(min_size, int(y1 - y0)))


def _dwell_tick(pos, frac: float) -> None:
    filled = int(frac * 20)
    bar = "#" * filled + "." * (20 - filled)
    print(f"\r  ({pos[0]:5d}, {pos[1]:5d})  holding [{bar}] ", end="", flush=True)


def _ask(prompt: str, default: bool = True) -> bool:
    """Yes/no via stdin. Used only between steps, when focus is free."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"  {prompt} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer.startswith("y")


def _distance(a, b) -> float:
    return float(np.sqrt(sum((int(p) - int(q)) ** 2 for p, q in zip(a, b))))


def _too_close(bgr, samples, floor: float = 90.0) -> str | None:
    """Name of an existing sample this colour would be confused with."""
    for s in samples:
        if _distance(bgr, s.rgb) < floor:
            return s.name
    return None


# --------------------------------------------------------------------------
def _banner(n: int, total: int, step: Step) -> None:
    print()
    print("=" * 66)
    print(f"  STEP {n}/{total}   {step.prompt}")
    print("=" * 66)
    for line in _wrap(step.detail, 62):
        print(f"  {line}")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def run_setup(cfg, config_path: str) -> bool:
    """Walk the steps and write the results into cfg. Returns True if saved."""
    from .capture import activate_mirror_app
    from .palette import Palette, Sample

    try:
        pointer = Pointer()
    except ImportError:
        print("pyobjc-framework-Quartz is required for setup:")
        print("  pip install pyobjc-framework-Quartz")
        return False

    total = 1 + len(DRUM_STEPS) + len(COLOUR_STEPS)

    print()
    print("=" * 66)
    print("  SETUP")
    print("=" * 66)
    print("  Put this terminal beside the phone window so you can read the")
    print("  prompts while you work. They appear one at a time.")
    print()
    print("  You pick each target by HOVERING over it and holding still for a")
    print("  moment - no clicking. Moving restarts the hold, so overshooting")
    print("  costs nothing. Ctrl-C stops at any point.")
    print()

    if activate_mirror_app():
        print("  iPhone Mirroring is now in front.")
    else:
        print("  Could not raise iPhone Mirroring - bring it up yourself.")

    print()
    print("  Start a song so notes are on screen.")
    try:
        input("  Then press ENTER here to begin. ")
    except EOFError:
        return False

    drum_points: dict[str, tuple[int, int]] = {}
    samples: list[Sample] = []
    colour_points: list[tuple[int, int]] = []
    n = 1

    for step in DRUM_STEPS:
        n += 1
        _banner(n, total, step)
        pos = pointer.wait_for_dwell(on_tick=_dwell_tick)
        print()
        if pos is None:
            print("  timed out - a drum position is required, stopping.")
            return False
        drum_points[step.key.replace("_drum", "")] = pos
        print(f"  recorded at {pos}")

    for step in COLOUR_STEPS:
        n += 1
        _banner(n, total, step)
        if step.optional and not _ask("Capture this one?", default=False):
            print("  skipped.")
            continue
        pos = pointer.wait_for_dwell(on_tick=_dwell_tick)
        print()
        if pos is None:
            if step.optional:
                print("  timed out - skipped.")
                continue
            print("  timed out - that one is required, stopping.")
            return False
        bgr = sample_colour(pos)
        if max(bgr) - min(bgr) < 60:
            print(f"  WARNING: BGR={bgr} is washed out. Pale samples also match the")
            print("  bubbles and the background, which causes phantom presses.")
            print("  Prefer a more vivid spot, or skip this one.")
        clash = _too_close(bgr, samples)
        if clash:
            print(f"  WARNING: that colour is very close to '{clash}'. The two will")
            print("  be hard to tell apart - click a more vivid part and re-run setup.")
        samples.append(Sample(name=step.key, rgb=bgr, taps=step.taps, point=pos))
        colour_points.append(pos)
        print(f"  recorded at {pos}   colour BGR={bgr}")

    cfg.drum_points = {k: tuple(v) for k, v in drum_points.items()}
    cfg.palette = Palette(samples=samples)
    cfg.sample_rect = sample_rect_from_points(colour_points)
    cfg.save(config_path)

    print()
    print("=" * 66)
    print("  READY")
    print("=" * 66)
    print(f"  watching {cfg.sample_rect[2]}x{cfg.sample_rect[3]} px "
          f"at ({cfg.sample_rect[0]}, {cfg.sample_rect[1]})")
    for s in samples:
        print(f"  {s.name:<6} BGR={s.rgb}  -> {'+'.join(s.taps)}")
    print(f"  saved to {config_path}")

    if not verify(cfg):
        print("  Re-run `python -m vocalbot.cli run --setup` to try again.")
        return False
    return True


def verify(cfg, seconds: float = 25.0) -> bool:
    """Live readout before playing, so a bad sample is obvious immediately.

    Reference screenshots can only take this so far - what matters is how the
    colours separate on *your* display. Watching the counts against real notes
    is the only way to see that, and it costs nothing to look.
    """
    from .capture import MSSCapture
    from .palette import count_matches, decide_taps

    print()
    print("=" * 66)
    print("  CHECK")
    print("=" * 66)
    print("  Live reading of the box. Watch it against what the game shows:")
    print("  a blue note should read blue, a red note red, a pair both.")
    print(f"  Runs for {int(seconds)}s, or Ctrl-C to stop early.")
    print()

    cap = MSSCapture(cfg.sample_rect)
    try:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            counts = count_matches(cap.grab(), cfg.palette, cfg.step)
            taps = decide_taps(counts, cfg.palette, cfg.tap_order)
            cells = "  ".join(
                f"{k}={v:<5d}" for k, v in counts.items() if k != "background"
            )
            print(f"\r  {cells}  ->  {'+'.join(taps) or '(nothing)':<10s}", end="", flush=True)
            time.sleep(0.05)
        print()
    except KeyboardInterrupt:
        print()
    finally:
        cap.close()

    print()
    return _ask("Does that look right - start playing?", default=True)
