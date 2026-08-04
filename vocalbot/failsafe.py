"""Stopping the bot.

While playing, the bot clicks the mirroring window many times a second. That
takes focus, so Ctrl-C in the terminal can be genuinely hard to land - the
terminal is not where your keystrokes are going. Relying on it as the only way
out is not acceptable for something that drives your mouse.

So the primary way to stop is to **move the mouse**. The bot only ever parks the
cursor on one of the two drums, so a cursor anywhere else means a human took
over, and the run ends immediately. No permission is needed to read the cursor
position, unlike key state, so this cannot silently fail to work.

Three further backstops, in order of how quickly they act:

  - a wall-clock limit, so an unattended run cannot continue indefinitely
  - SIGINT and SIGTERM, for when the terminal *is* reachable
  - the tap thread is a daemon and is joined on exit, so nothing keeps clicking
    after the loop stops
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field


@dataclass
class Failsafe:
    """Watches for reasons to stop."""

    drum_points: list[tuple[int, int]] = field(default_factory=list)
    radius: int = 140  # how far from a drum counts as "you moved it"
    grace: float = 0.30  # seconds the cursor must be away before stopping
    duration: float | None = 120.0  # wall-clock limit, None for no limit

    _started: float = field(default=0.0, init=False)
    _away_since: float | None = field(default=None, init=False)
    _armed: bool = field(default=False, init=False)
    _signalled: bool = field(default=False, init=False)
    reason: str | None = field(default=None, init=False)

    def start(self) -> "Failsafe":
        self._started = time.perf_counter()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass  # not on the main thread; the loop still catches KeyboardInterrupt
        return self

    def _on_signal(self, *_):
        self._signalled = True
        self.reason = "interrupted"

    def arm(self) -> None:
        """Start watching the cursor. Called after the first tap, since before
        that the cursor is wherever you left it."""
        self._armed = True
        self._away_since = None

    def near_a_drum(self, pos) -> bool:
        return any(
            abs(pos[0] - x) <= self.radius and abs(pos[1] - y) <= self.radius
            for x, y in self.drum_points
        )

    def should_stop(self, cursor, now: float | None = None) -> bool:
        """True when the run must end. `cursor` may be None if unavailable."""
        now = time.perf_counter() if now is None else now

        if self._signalled:
            return True

        if self.duration is not None and now - self._started >= self.duration:
            self.reason = f"reached the {self.duration:.0f}s limit"
            return True

        if not self._armed or cursor is None:
            return False

        if self.near_a_drum(cursor):
            self._away_since = None
            return False

        # Away from both drums: a human is driving. Wait out a short grace
        # period so one stray sample can't end a run.
        if self._away_since is None:
            self._away_since = now
            return False
        if now - self._away_since >= self.grace:
            self.reason = "you moved the mouse"
            return True
        return False
