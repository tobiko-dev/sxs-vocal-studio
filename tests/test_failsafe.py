"""Tests for stopping the bot.

While playing, the bot clicks the mirroring window constantly, which takes
focus, so Ctrl-C in the terminal can be hard to land. Moving the mouse is
therefore the primary way out and has to work reliably.
"""

from __future__ import annotations

from vocalbot.failsafe import Failsafe

RED = (1180, 700)
BLUE = (980, 700)


def guard(**kw) -> Failsafe:
    fs = Failsafe(drum_points=[RED, BLUE], **kw)
    fs._started = 0.0
    return fs


# --------------------------------------------------------------------------
# moving the mouse


def test_moving_the_mouse_stops_the_run():
    fs = guard()
    fs.arm()
    assert not fs.should_stop((3000, 200), now=0.0)  # grace period starts
    assert fs.should_stop((3000, 200), now=1.0)
    assert fs.reason == "you moved the mouse"


def test_a_single_stray_sample_does_not_stop_it():
    """One bad cursor read must not kill a run mid-song."""
    fs = guard(grace=0.30)
    fs.arm()
    assert not fs.should_stop((3000, 200), now=0.0)
    assert not fs.should_stop(RED, now=0.05)  # back on the drum
    assert not fs.should_stop((3000, 200), now=0.10)  # grace restarts
    assert not fs.should_stop((3000, 200), now=0.30)


def test_resting_on_either_drum_is_fine():
    """The bot parks the cursor on whichever drum it just pressed."""
    fs = guard()
    fs.arm()
    for t, pos in enumerate([RED, BLUE, RED, RED, BLUE]):
        assert not fs.should_stop(pos, now=float(t))


def test_small_wobble_near_a_drum_is_tolerated():
    fs = guard(radius=140)
    fs.arm()
    assert not fs.should_stop((RED[0] + 100, RED[1] - 90), now=0.0)


def test_not_armed_until_the_first_tap():
    """Before the bot has tapped, the cursor is wherever you left it, so it
    cannot mean anything."""
    fs = guard()
    assert not fs.should_stop((10, 10), now=0.0)
    assert not fs.should_stop((10, 10), now=99.0)


def test_missing_cursor_reading_does_not_stop_it():
    """No Quartz means no cursor. That should degrade to the time limit, not
    stop instantly or crash."""
    fs = guard()
    fs.arm()
    assert not fs.should_stop(None, now=0.0)
    assert not fs.should_stop(None, now=10.0)


# --------------------------------------------------------------------------
# backstops


def test_time_limit_stops_the_run():
    fs = guard(duration=30.0)
    assert not fs.should_stop(RED, now=29.0)
    assert fs.should_stop(RED, now=30.0)
    assert "30s limit" in fs.reason


def test_time_limit_applies_before_arming():
    """An unattended run that never taps must still end."""
    fs = guard(duration=10.0)
    assert fs.should_stop(None, now=10.0)


def test_no_limit_when_duration_is_none():
    fs = guard(duration=None)
    assert not fs.should_stop(RED, now=1e6)


def test_signal_stops_the_run():
    fs = guard()
    fs._on_signal()
    assert fs.should_stop(RED, now=0.0)
    assert fs.reason == "interrupted"


def test_signal_beats_everything_else():
    """Even unarmed, with time left, a signal must win."""
    fs = guard(duration=None)
    fs._on_signal()
    assert fs.should_stop(None, now=0.0)


# --------------------------------------------------------------------------
# geometry


def test_near_a_drum_uses_a_square_of_radius():
    fs = guard(radius=100)
    assert fs.near_a_drum((RED[0] + 100, RED[1] + 100))
    assert not fs.near_a_drum((RED[0] + 101, RED[1]))


def test_with_no_drums_recorded_nothing_is_near():
    fs = Failsafe(drum_points=[])
    fs._started = 0.0
    fs.arm()
    assert not fs.near_a_drum((0, 0))
    assert not fs.should_stop((0, 0), now=0.0)
    assert fs.should_stop((0, 0), now=1.0)  # everywhere is "away", so it stops


def test_arming_twice_resets_the_grace_window():
    fs = guard(grace=0.5)
    fs.arm()
    assert not fs.should_stop((3000, 200), now=0.0)
    fs.arm()  # a tap landed, cursor is back on a drum
    assert not fs.should_stop((3000, 200), now=0.4)
    assert fs.should_stop((3000, 200), now=1.0)
