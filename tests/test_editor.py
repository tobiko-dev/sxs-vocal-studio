"""Tests for the interactive editor's geometry.

The display loop needs a GUI, but every pointer and keyboard action routes
through BoxEditor, which doesn't. That's what gets tested here.
"""

from __future__ import annotations

import pytest

from vocalbot.config import Config, Zone
from vocalbot.editor import HANDLE_PX, MIN_PX, BoxEditor

W, H = 400, 800


def one_zone_cfg(rect=(0.25, 0.25, 0.75, 0.5)) -> Config:
    cfg = Config()
    cfg.zones = [Zone(name="a", rect=rect, match="any")]
    return cfg


def editor(cfg=None) -> BoxEditor:
    return BoxEditor(cfg or one_zone_cfg(), W, H)


# --------------------------------------------------------------------------
# dragging


def test_drag_moves_the_box_without_resizing_it():
    ed = editor()
    before = ed.px(ed.cfg.zone("a"))
    ed.on_press(150, 250)  # inside the box
    ed.on_drag(170, 280)
    ed.on_release()
    after = ed.px(ed.cfg.zone("a"))
    assert after[0] - before[0] == 20 and after[1] - before[1] == 30
    assert after[2] - after[0] == before[2] - before[0]
    assert after[3] - after[1] == before[3] - before[1]


def test_drag_does_not_jump_when_grabbed_off_centre():
    """The pointer offset is kept, so the box doesn't snap its corner to the
    cursor on the first move."""
    ed = editor()
    x0, y0, _, _ = ed.px(ed.cfg.zone("a"))
    ed.on_press(x0 + 37, y0 + 21)
    ed.on_drag(x0 + 37, y0 + 21)  # same point: nothing should move
    assert ed.px(ed.cfg.zone("a"))[:2] == (x0, y0)


def test_corner_handle_resizes_only_that_corner():
    ed = editor()
    x0, y0, x1, y1 = ed.px(ed.cfg.zone("a"))
    ed.on_press(x1, y1)  # se handle
    ed.on_drag(x1 + 25, y1 + 40)
    assert ed.px(ed.cfg.zone("a")) == (x0, y0, x1 + 25, y1 + 40)


def test_edge_handle_resizes_one_axis():
    ed = editor()
    x0, y0, x1, y1 = ed.px(ed.cfg.zone("a"))
    ed.on_press((x0 + x1) // 2, y0)  # north edge
    ed.on_drag((x0 + x1) // 2, y0 - 30)
    nx0, ny0, nx1, ny1 = ed.px(ed.cfg.zone("a"))
    assert (nx0, nx1, ny1) == (x0, x1, y1)
    assert ny0 == y0 - 30


def test_resizing_past_the_opposite_edge_flips_rather_than_inverts():
    """Dragging the north handle below the south one must keep x0<x1, y0<y1."""
    ed = editor()
    x0, y0, x1, y1 = ed.px(ed.cfg.zone("a"))
    ed.on_press((x0 + x1) // 2, y0)
    ed.on_drag((x0 + x1) // 2, y1 + 60)
    rect = ed.cfg.zone("a").rect
    assert rect[0] < rect[2] and rect[1] < rect[3]


def test_boxes_are_clamped_to_the_frame():
    ed = editor()
    ed.on_press(150, 250)
    ed.on_drag(-500, -500)
    x0, y0, x1, y1 = ed.px(ed.cfg.zone("a"))
    assert x0 >= 0 and y0 >= 0
    ed.on_drag(5000, 5000)
    x0, y0, x1, y1 = ed.px(ed.cfg.zone("a"))
    assert x1 <= W and y1 <= H


def test_box_cannot_be_collapsed():
    ed = editor()
    x0, y0, x1, y1 = ed.px(ed.cfg.zone("a"))
    ed.on_press(x1, y1)
    ed.on_drag(x0, y0)  # drag se corner onto nw
    nx0, ny0, nx1, ny1 = ed.px(ed.cfg.zone("a"))
    assert nx1 - nx0 >= MIN_PX and ny1 - ny0 >= MIN_PX


def test_drag_without_a_grab_is_a_no_op():
    ed = editor()
    before = ed.cfg.zone("a").rect
    ed.on_drag(10, 10)
    assert ed.cfg.zone("a").rect == before


def test_pressing_empty_space_clears_the_grab():
    ed = editor()
    ed.on_press(5, 780)  # outside the box
    assert ed.grab is None
    before = ed.cfg.zone("a").rect
    ed.on_drag(200, 400)
    assert ed.cfg.zone("a").rect == before


# --------------------------------------------------------------------------
# hit testing


def test_handles_win_over_the_body():
    ed = editor()
    _, _, x1, y1 = ed.px(ed.cfg.zone("a"))
    assert ed.hit_test(x1, y1) == ("a", "se")
    assert ed.hit_test(x1 - 40, y1 - 40) == ("a", "move")


def test_handle_grab_radius():
    ed = editor()
    _, _, x1, y1 = ed.px(ed.cfg.zone("a"))
    assert ed.hit_test(x1 - HANDLE_PX, y1)[1] == "se"
    assert ed.hit_test(x1 - HANDLE_PX - 6, y1 - 6)[1] == "move"


def test_selected_zone_is_hit_tested_first():
    """The three lane zones share an identical box, so without this the
    selection could never be changed by clicking."""
    cfg = Config()
    lane = cfg.zone("red").rect
    assert cfg.zone("blue").rect == lane, "fixture assumes the lane zones overlap"
    ed = BoxEditor(cfg, W, H)
    ed.selected = "blue"
    x0, y0, x1, y1 = ed.px(cfg.zone("blue"))
    assert ed.hit_test(x1, y1)[0] == "blue"
    ed.selected = "red"
    assert ed.hit_test(x1, y1)[0] == "red"


def test_hit_test_misses_return_none():
    ed = editor(one_zone_cfg((0.4, 0.4, 0.6, 0.45)))
    assert ed.hit_test(5, 5) is None


# --------------------------------------------------------------------------
# keyboard


def test_nudge_moves_by_a_pixel_and_respects_bounds():
    ed = editor()
    x0, y0, _, _ = ed.px(ed.cfg.zone("a"))
    ed.nudge(1, 0)
    assert ed.px(ed.cfg.zone("a"))[0] == x0 + 1
    ed.nudge(0, -1)
    assert ed.px(ed.cfg.zone("a"))[1] == y0 - 1


def test_nudge_stops_at_the_edge_instead_of_squashing():
    ed = editor(one_zone_cfg((0.0, 0.0, 0.2, 0.1)))
    before = ed.px(ed.cfg.zone("a"))
    ed.nudge(-1, 0)
    ed.nudge(0, -1)
    assert ed.px(ed.cfg.zone("a")) == before


def test_grow_resizes_around_the_centre():
    ed = editor()
    x0, y0, x1, y1 = ed.px(ed.cfg.zone("a"))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ed.grow(4, 4)
    nx0, ny0, nx1, ny1 = ed.px(ed.cfg.zone("a"))
    assert abs((nx0 + nx1) / 2 - cx) <= 1 and abs((ny0 + ny1) / 2 - cy) <= 1
    assert nx1 - nx0 == (x1 - x0) + 8


def test_threshold_adjust_follows_the_match_rule():
    """A red-matching zone shouldn't have its unused blue threshold moved."""
    cfg = one_zone_cfg()
    cfg.zone("a").match = "red"
    cfg.zone("a").thresh_red = 40
    cfg.zone("a").thresh_blue = 30
    ed = BoxEditor(cfg, W, H)
    ed.adjust_threshold(+5)
    assert (cfg.zone("a").thresh_red, cfg.zone("a").thresh_blue) == (45, 30)

    cfg.zone("a").match = "both"
    ed.adjust_threshold(+5)
    assert (cfg.zone("a").thresh_red, cfg.zone("a").thresh_blue) == (50, 35)


def test_threshold_never_goes_below_one():
    ed = editor()
    for _ in range(50):
        ed.adjust_threshold(-5)
    assert ed.cfg.zone("a").thresh_red >= 1


def test_cycle_match_and_taps_stay_valid():
    from vocalbot.config import MATCHES

    ed = editor()
    seen = set()
    for _ in range(len(MATCHES)):
        ed.cycle_match()
        seen.add(ed.cfg.zone("a").match)
    assert seen == set(MATCHES)

    for _ in range(4):
        ed.cycle_taps()
        assert all(d in ("red", "blue") for d in ed.cfg.zone("a").taps)


def test_add_and_delete_zones():
    ed = editor()
    name = ed.add_zone()
    assert ed.selected == name and len(ed.cfg.zones) == 2
    Zone(**ed.cfg.zone(name).__dict__)  # the new box must be valid
    ed.delete_selected()
    assert len(ed.cfg.zones) == 1
    assert ed.selected == "a"


def test_add_zone_rejects_duplicate_names():
    ed = editor()
    with pytest.raises(ValueError):
        ed.add_zone("a")


def test_delete_last_zone_leaves_nothing_selected():
    ed = editor()
    ed.delete_selected()
    assert ed.cfg.zones == [] and ed.selected is None
    ed.nudge(1, 1)  # must not raise
    ed.adjust_threshold(5)
    ed.toggle_enabled()


def test_select_next_wraps_both_ways():
    cfg = Config()
    ed = BoxEditor(cfg, W, H)
    names = [z.name for z in cfg.zones]
    ed.selected = names[-1]
    ed.select_next()
    assert ed.selected == names[0]
    ed.select_next(-1)
    assert ed.selected == names[-1]


def test_toggle_enabled_round_trips():
    ed = editor()
    assert ed.cfg.zone("a").enabled
    ed.toggle_enabled()
    assert not ed.cfg.zone("a").enabled
    ed.toggle_enabled()
    assert ed.cfg.zone("a").enabled


# --------------------------------------------------------------------------
# persistence


def test_dirty_flag_tracks_edits():
    ed = editor()
    assert not ed.dirty
    ed.nudge(1, 0)
    assert ed.dirty


def test_edits_survive_a_save_load_round_trip(tmp_path):
    ed = editor()
    ed.on_press(150, 250)
    ed.on_drag(180, 300)
    ed.on_release()
    ed.adjust_threshold(+15)
    edited = ed.cfg.zone("a")

    path = tmp_path / "c.json"
    ed.cfg.save(path)
    back = Config.load(path)
    assert back.zone("a").rect == edited.rect
    assert back.zone("a").thresh_red == edited.thresh_red


def test_display_scale_does_not_distort_stored_fractions():
    """Editing at a scaled-down display size must store the same fractions."""
    big = BoxEditor(one_zone_cfg(), 800, 1600)
    small = BoxEditor(one_zone_cfg(), 400, 800)
    big.on_press(400, 500)
    big.on_drag(480, 600)  # 80x100 at 2x scale
    small.on_press(200, 250)
    small.on_drag(240, 300)  # 40x50 at 1x scale
    for a, b in zip(big.cfg.zone("a").rect, small.cfg.zone("a").rect):
        assert abs(a - b) < 1e-3
