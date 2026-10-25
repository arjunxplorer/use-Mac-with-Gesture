from gesturemac.input.cursor import ScreenBounds, clamp_to_displays

# Mirrors a 1920x1080 main display with a 1512x982 laptop screen to its right.
MAIN = ScreenBounds(0, 0, 1920, 1080)
LAPTOP = ScreenBounds(1920, 0, 3432, 982)
DISPLAYS = [MAIN, LAPTOP]


def test_point_on_second_display_is_allowed() -> None:
    assert clamp_to_displays(2500, 500, DISPLAYS) == (2500, 500)


def test_crossing_between_displays() -> None:
    assert clamp_to_displays(1925, 300, DISPLAYS) == (1925, 300)


def test_clamps_past_far_right_edge() -> None:
    assert clamp_to_displays(4000, 500, DISPLAYS) == (3431, 500)


def test_gap_below_shorter_display_snaps_to_nearest() -> None:
    # (2500, 1050) is inside the desktop's bounding box but on no screen.
    x, y = clamp_to_displays(2500, 1050, DISPLAYS)
    assert (x, y) == (2500, 981)


def test_clamps_above_top() -> None:
    assert clamp_to_displays(100, -50, DISPLAYS) == (100, 0)
