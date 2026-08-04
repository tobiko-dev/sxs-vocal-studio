"""Guided setup: follow the prompts, click the things, done.

Everything the bot needs is learned from where you click:

  - clicking each drum records where to press it
  - clicking each note colour records what that colour actually is on your
    display, and roughly where notes appear

That removes the two things that kept going wrong. There is no window rect to
work out, because drum positions are recorded in absolute screen coordinates.
And there are no colour thresholds to guess, because the colours are sampled
from your screen rather than from reference screenshots.

## Why this watches input globally

The prompts are in the terminal but the clicking happens in the mirroring
window, so the terminal is not focused while you work. Reading stdin would
capture nothing. Instead the global mouse and keyboard state is polled through
Quartz - the same Accessibility permission the bot already needs to tap.

That is a poll loop, not a thread: a click is a rising edge on the global button
state, sampled at ~120Hz. Nothing runs concurrently during setup, so there is no
shared state to guard.

Note that clicking inside the mirroring window also sends that touch to the
phone. Pressing the drums is harmless, and that is most of what you are asked to
click.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

# macOS virtual key codes
KEY_E = 14
KEY_S = 1
KEY_Q = 12
POLL = 1 / 120


@dataclass
class Step:
    key: str
    prompt: str
    detail: str
    optional: bool = False
    taps: tuple[str, ...] = ()


DRUM_STEPS = [
    Step("blue_drum", "Click the BLUE drum",
         "The blue drum pad, bottom left. This is where blue notes get pressed."),
    Step("red_drum", "Click the RED drum",
         "The red drum pad, bottom right."),
]

COLOUR_STEPS = [
    Step("blue", "Click the most VIVID part of a BLUE note",
         "The solid, strongly coloured core of the note symbol at the front of "
         "the queue - not the pale bubble around it, and not a shadowed edge.",
         taps=("blue",)),
    Step("red", "Click the most VIVID part of a RED note",
         "The solid red or magenta core of the symbol, same idea.",
         taps=("red",)),
    Step("both", "Click a WHITE 'both' area, or press 's' to skip",
         "Only if a distinct marker appears when two notes arrive together. "
         "Skipping is fine - anything that matches neither note counts as both.",
         optional=True, taps=("red", "blue")),
    Step("disco", "Click the PURPLE part of the disco ball, or press 's' to skip",
         "Only if a mirror ball is on screen right now, and only if its purple "
         "looks strong rather than pastel - a washed-out sample also matches the "
         "pale bubbles and causes phantom presses. Skipping is usually better: "
         "the ball reads as blue on its own.",
         optional=True, taps=("blue",)),
]


class Input:
    """Global mouse and keyboard state. Thin wrapper over Quartz."""

    def __init__(self):
        import Quartz

        self.Q = Quartz
        self._was_down = False

    def mouse_pos(self) -> tuple[int, int]:
        loc = self.Q.CGEventGetLocation(self.Q.CGEventCreate(None))
        return int(loc.x), int(loc.y)

    def mouse_down(self) -> bool:
        return bool(
            self.Q.CGEventSourceButtonState(
                self.Q.kCGEventSourceStateCombinedSessionState, self.Q.kCGMouseButtonLeft
            )
        )

    def key_down(self, keycode: int) -> bool:
        return bool(
            self.Q.CGEventSourceKeyState(
                self.Q.kCGEventSourceStateCombinedSessionState, keycode
            )
        )

    def wait_for_click(self, timeout: float = 120.0) -> tuple[int, int] | None:
        """Block until the left button is pressed and released. Returns the
        press location, or None if 's' (skip) or 'q' (quit) was pressed."""
        deadline = time.perf_counter() + timeout
        # Don't count a button that is already held from the previous step.
        while self.mouse_down() and time.perf_counter() < deadline:
            time.sleep(POLL)
        while time.perf_counter() < deadline:
            if self.key_down(KEY_S):
                self._flush_key(KEY_S)
                return None
            if self.key_down(KEY_Q):
                raise KeyboardInterrupt
            if self.mouse_down():
                pos = self.mouse_pos()
                while self.mouse_down() and time.perf_counter() < deadline:
                    time.sleep(POLL)
                return pos
            time.sleep(POLL)
        return None

    def wait_for_key(self, keycode: int, timeout: float = 300.0) -> bool:
        deadline = time.perf_counter() + timeout
        while self.key_down(keycode) and time.perf_counter() < deadline:
            time.sleep(POLL)  # ignore a key already held
        while time.perf_counter() < deadline:
            if self.key_down(KEY_Q):
                raise KeyboardInterrupt
            if self.key_down(keycode):
                self._flush_key(keycode)
                return True
            time.sleep(POLL)
        return False

    def _flush_key(self, keycode: int) -> None:
        while self.key_down(keycode):
            time.sleep(POLL)


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
        watcher = Input()
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
    print("  prompts while you click. The prompts appear one at a time.")
    print()
    print("  You will click, in order: each drum, then the colour of each kind")
    print("  of note. Clicks land on the phone, so pressing the drums actually")
    print("  presses them - that is expected.")
    print()
    print("  'q' quits at any point.")
    print()

    if activate_mirror_app():
        print("  iPhone Mirroring is now in front.")
    else:
        print("  Could not raise iPhone Mirroring - bring it up yourself.")

    print()
    print("  Start a song, then press 'e' when you are ready to begin.")
    print("  (press 'e' anywhere - this does not need the terminal focused)")
    if not watcher.wait_for_key(KEY_E):
        print("\n  Timed out waiting for 'e'.")
        return False

    drum_points: dict[str, tuple[int, int]] = {}
    samples: list[Sample] = []
    colour_points: list[tuple[int, int]] = []
    n = 1

    for step in DRUM_STEPS:
        n += 1
        _banner(n, total, step)
        print("  waiting for your click...")
        pos = watcher.wait_for_click()
        if pos is None:
            print("  skipped - a drum position is required, stopping.")
            return False
        drum_points[step.key.replace("_drum", "")] = pos
        print(f"  recorded at {pos}")

    for step in COLOUR_STEPS:
        n += 1
        _banner(n, total, step)
        print("  waiting for your click..." + ("  ('s' to skip)" if step.optional else ""))
        pos = watcher.wait_for_click()
        if pos is None:
            if step.optional:
                print("  skipped.")
                continue
            print("  that one is required, stopping.")
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

    verify(cfg, watcher)
    return True


def verify(cfg, watcher, seconds: float = 600.0) -> None:
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
    print()
    print("  press 'e' to start playing, 'q' to quit")
    print()

    cap = MSSCapture(cfg.sample_rect)
    try:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if watcher.key_down(KEY_Q):
                raise KeyboardInterrupt
            if watcher.key_down(KEY_E):
                watcher._flush_key(KEY_E)
                break
            counts = count_matches(cap.grab(), cfg.palette, cfg.step)
            taps = decide_taps(counts, cfg.palette, cfg.tap_order)
            cells = "  ".join(
                f"{k}={v:<5d}" for k, v in counts.items() if k != "background"
            )
            print(f"\r  {cells}  ->  {'+'.join(taps) or '(nothing)':<10s}", end="", flush=True)
            time.sleep(0.05)
        print()
    finally:
        cap.close()
