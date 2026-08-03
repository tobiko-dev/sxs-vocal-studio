"""Interactive zone editor: drag to move, grab a handle to resize.

The interaction logic lives in `BoxEditor`, which knows nothing about OpenCV and
is covered by tests. `run_editor` is a thin display loop on top, so the geometry
can be verified on a machine with no GUI.

Coordinates inside the editor are display pixels. Zone rects stay fractions of
the phone screen, so the display can be scaled to fit without disturbing them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config, Zone

# Resize handles, in the order they are hit-tested. Corners first: they overlap
# the edges and should win.
CORNERS = ("nw", "ne", "se", "sw")
EDGES = ("n", "e", "s", "w")
HANDLE_PX = 9  # grab radius
MIN_PX = 6  # smallest box you can drag down to


@dataclass
class Grab:
    zone: str
    handle: str  # "move" or one of CORNERS / EDGES
    ox: float  # pointer offset from the box, kept so dragging doesn't jump
    oy: float


class BoxEditor:
    """Mutates the zones on a Config in response to pointer events."""

    def __init__(self, cfg: Config, width: int, height: int):
        self.cfg = cfg
        self.w = width
        self.h = height
        self.selected: str | None = cfg.zones[0].name if cfg.zones else None
        self.grab: Grab | None = None
        self.dirty = False

    # -- geometry ---------------------------------------------------------
    def px(self, zone: Zone) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = zone.rect
        return (
            int(round(x0 * self.w)),
            int(round(y0 * self.h)),
            int(round(x1 * self.w)),
            int(round(y1 * self.h)),
        )

    def _set_px(self, zone: Zone, x0: float, y0: float, x1: float, y1: float) -> None:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0 = min(max(0.0, x0), self.w - MIN_PX)
        y0 = min(max(0.0, y0), self.h - MIN_PX)
        x1 = max(min(float(self.w), x1), x0 + MIN_PX)
        y1 = max(min(float(self.h), y1), y0 + MIN_PX)
        zone.rect = (
            round(x0 / self.w, 6),
            round(y0 / self.h, 6),
            round(x1 / self.w, 6),
            round(y1 / self.h, 6),
        )
        self.dirty = True

    def handles(self, zone: Zone) -> dict[str, tuple[int, int]]:
        x0, y0, x1, y1 = self.px(zone)
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        return {
            "nw": (x0, y0), "ne": (x1, y0), "se": (x1, y1), "sw": (x0, y1),
            "n": (mx, y0), "e": (x1, my), "s": (mx, y1), "w": (x0, my),
        }

    # -- hit testing ------------------------------------------------------
    def hit_test(self, x: int, y: int) -> tuple[str, str] | None:
        """Topmost (zone, handle) under the pointer, or None.

        The selected zone is tested first so its handles stay reachable when
        boxes overlap - and the lane zones overlap exactly by design.
        """
        order = [z for z in self.cfg.zones if z.name == self.selected]
        order += [z for z in self.cfg.zones if z.name != self.selected]

        for zone in order:
            for handle in CORNERS + EDGES:
                hx, hy = self.handles(zone)[handle]
                if abs(x - hx) <= HANDLE_PX and abs(y - hy) <= HANDLE_PX:
                    return zone.name, handle
        for zone in order:
            x0, y0, x1, y1 = self.px(zone)
            if x0 <= x <= x1 and y0 <= y <= y1:
                return zone.name, "move"
        return None

    # -- pointer ----------------------------------------------------------
    def on_press(self, x: int, y: int) -> None:
        hit = self.hit_test(x, y)
        if hit is None:
            self.grab = None
            return
        name, handle = hit
        self.selected = name
        x0, y0, _, _ = self.px(self.cfg.zone(name))
        self.grab = Grab(name, handle, x - x0, y - y0)

    def on_drag(self, x: int, y: int) -> None:
        if self.grab is None:
            return
        zone = self.cfg.zone(self.grab.zone)
        x0, y0, x1, y1 = self.px(zone)
        h = self.grab.handle

        if h == "move":
            nx0 = x - self.grab.ox
            ny0 = y - self.grab.oy
            w, ht = x1 - x0, y1 - y0
            nx0 = min(max(0, nx0), self.w - w)
            ny0 = min(max(0, ny0), self.h - ht)
            self._set_px(zone, nx0, ny0, nx0 + w, ny0 + ht)
            return

        if "n" in h:
            y0 = y
        if "s" in h:
            y1 = y
        if "w" in h:
            x0 = x
        if "e" in h:
            x1 = x
        self._set_px(zone, x0, y0, x1, y1)

    def on_release(self) -> None:
        self.grab = None

    # -- keyboard ---------------------------------------------------------
    def select_next(self, step: int = 1) -> None:
        if not self.cfg.zones:
            return
        names = [z.name for z in self.cfg.zones]
        i = names.index(self.selected) if self.selected in names else -1
        self.selected = names[(i + step) % len(names)]

    def nudge(self, dx: int, dy: int) -> None:
        if self.selected is None:
            return
        zone = self.cfg.zone(self.selected)
        x0, y0, x1, y1 = self.px(zone)
        if x0 + dx < 0 or x1 + dx > self.w or y0 + dy < 0 or y1 + dy > self.h:
            return
        self._set_px(zone, x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    def grow(self, dw: int, dh: int) -> None:
        """Resize around the centre."""
        if self.selected is None:
            return
        zone = self.cfg.zone(self.selected)
        x0, y0, x1, y1 = self.px(zone)
        self._set_px(zone, x0 - dw, y0 - dh, x1 + dw, y1 + dh)

    def adjust_threshold(self, delta: int, color: str | None = None) -> None:
        """Bump the threshold(s) the selected zone actually consults."""
        if self.selected is None:
            return
        zone = self.cfg.zone(self.selected)
        targets = [color] if color else {
            "red": ["red"], "blue": ["blue"], "both": ["red", "blue"], "any": ["red", "blue"]
        }[zone.match]
        for c in targets:
            attr = f"thresh_{c}"
            setattr(zone, attr, max(1, getattr(zone, attr) + delta))
        self.dirty = True

    def cycle_match(self) -> None:
        from .config import MATCHES

        if self.selected is None:
            return
        zone = self.cfg.zone(self.selected)
        zone.match = MATCHES[(MATCHES.index(zone.match) + 1) % len(MATCHES)]
        self.dirty = True

    def cycle_taps(self) -> None:
        if self.selected is None:
            return
        options = [("red",), ("blue",), ("red", "blue")]
        zone = self.cfg.zone(self.selected)
        i = options.index(tuple(zone.taps)) if tuple(zone.taps) in options else -1
        zone.taps = options[(i + 1) % len(options)]
        self.dirty = True

    def toggle_enabled(self) -> None:
        if self.selected is None:
            return
        zone = self.cfg.zone(self.selected)
        zone.enabled = not zone.enabled
        self.dirty = True

    def add_zone(self, name: str | None = None) -> str:
        """Add a box in the middle of the frame and select it."""
        existing = {z.name for z in self.cfg.zones}
        if name is None:
            n = 1
            while f"zone{n}" in existing:
                n += 1
            name = f"zone{n}"
        if name in existing:
            raise ValueError(f"zone {name!r} already exists")
        self.cfg.zones.append(
            Zone(name=name, rect=(0.30, 0.45, 0.70, 0.50), match="any", taps=("red",))
        )
        self.selected = name
        self.dirty = True
        return name

    def delete_selected(self) -> str | None:
        if self.selected is None:
            return None
        gone = self.selected
        self.cfg.zones = [z for z in self.cfg.zones if z.name != gone]
        self.selected = self.cfg.zones[0].name if self.cfg.zones else None
        self.dirty = True
        return gone


# --------------------------------------------------------------------------
KEYMAP = """\
  drag box          move          drag handle    resize
  TAB / shift-TAB   select zone   arrows         nudge 1px (shift: 10)
  + / -             threshold     [ / ]          grow / shrink
  m                 cycle match   t              cycle taps
  n                 new zone      x              delete zone
  e                 enable/disable               s  save     q  quit
"""


def run_editor(cfg: Config, frame_source, config_path: str, max_height: int = 900) -> bool:
    """Display loop. `frame_source` is a callable returning a fresh BGR frame,
    so passing a live grabber lets counts update while the song plays.

    Returns True if the config was saved.
    """
    import cv2

    from .calibrate import render_overlay, sample_counts

    if not hasattr(cv2, "imshow"):
        raise SystemExit(
            "this build of opencv has no GUI. Install opencv-python (not the "
            "headless variant), or use `vocalbot zone set` instead."
        )

    first = frame_source()
    fh, fw = first.shape[:2]
    scale = min(1.0, max_height / fh)
    dw, dh = int(fw * scale), int(fh * scale)
    editor = BoxEditor(cfg, dw, dh)
    saved = False

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.on_press(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            editor.on_drag(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            editor.on_release()

    win = "vocalbot calibrate"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    print(KEYMAP)

    while True:
        frame = frame_source()
        counts = sample_counts(cfg, frame)
        view = cv2.resize(render_overlay(cfg, frame, counts=counts, active=editor.selected),
                          (dw, dh))

        if editor.selected:
            zone = cfg.zone(editor.selected)
            for hx, hy in editor.handles(zone).values():
                cv2.rectangle(view, (hx - 4, hy - 4), (hx + 4, hy + 4), (255, 255, 255), -1)
                cv2.rectangle(view, (hx - 4, hy - 4), (hx + 4, hy + 4), (0, 0, 0), 1)
            red_px, blue_px = counts.get(zone.name, (0, 0))
            status = (f"{zone.name}  {zone.match} -> {'+'.join(zone.taps)}  "
                      f"r={red_px}/{zone.thresh_red} b={blue_px}/{zone.thresh_blue}"
                      f"{'' if zone.enabled else '  DISABLED'}"
                      f"{'  *unsaved*' if editor.dirty else ''}")
            cv2.rectangle(view, (0, dh - 30), (dw, dh), (0, 0, 0), -1)
            cv2.putText(view, status, (8, dh - 9), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, view)
        key = cv2.waitKey(16) & 0xFFFFFF
        k = key & 0xFF

        if k in (ord("q"), 27):
            break
        elif k == ord("s"):
            cfg.save(config_path)
            editor.dirty = False
            saved = True
            print(f"saved to {config_path}")
        elif k == 9:
            editor.select_next()
        elif k == ord("n"):
            print(f"added {editor.add_zone()}")
        elif k == ord("x"):
            print(f"deleted {editor.delete_selected()}")
        elif k == ord("e"):
            editor.toggle_enabled()
        elif k == ord("m"):
            editor.cycle_match()
        elif k == ord("t"):
            editor.cycle_taps()
        elif k in (ord("+"), ord("=")):
            editor.adjust_threshold(+5)
        elif k in (ord("-"), ord("_")):
            editor.adjust_threshold(-5)
        elif k == ord("["):
            editor.grow(-2, -2)
        elif k == ord("]"):
            editor.grow(+2, +2)
        elif k in (81, 2):
            editor.nudge(-1, 0)
        elif k in (83, 3):
            editor.nudge(+1, 0)
        elif k in (82, 0):
            editor.nudge(0, -1)
        elif k in (84, 1):
            editor.nudge(0, +1)

    cv2.destroyAllWindows()
    if editor.dirty and not saved:
        print("quit with unsaved changes")
    return saved
