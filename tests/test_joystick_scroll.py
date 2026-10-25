"""Joystick scrolling: side-view pose detection, neutral center, position -> speed mapping."""

import math

import pytest

from gesturemac.gestures.commands import ClickPointer, MovePointer, ScrollWheel
from gesturemac.gestures.config import GestureConfig
from gesturemac.gestures.engine import GestureEngine, GestureMode
from gesturemac.gestures.scroll import ScrollPhase, is_side_scroll_pose, scroll_rate
from gesturemac.landmarks import HandFrame, HandJoint

from test_gesture_engine import _pt, pointer_pose

FRAME = 1 / 30
WRIST = (0.30, 0.50)

# Right hand seen from the side: index and middle stretched horizontally, stacked together.
SIDE_POSE = {
    HandJoint.WRIST: WRIST,
    HandJoint.THUMB_TIP: (0.40, 0.42),
    HandJoint.INDEX_MCP: (0.42, 0.47),
    HandJoint.INDEX_PIP: (0.50, 0.47),
    HandJoint.INDEX_TIP: (0.62, 0.47),
    HandJoint.MIDDLE_MCP: (0.42, 0.50),
    HandJoint.MIDDLE_PIP: (0.51, 0.50),
    HandJoint.MIDDLE_TIP: (0.64, 0.49),
    HandJoint.RING_PIP: (0.47, 0.53),
    HandJoint.RING_TIP: (0.40, 0.54),
    HandJoint.PINKY_PIP: (0.44, 0.55),
    HandJoint.PINKY_TIP: (0.37, 0.555),
}
NEUTRAL_TIPS_Y = 0.48


def side_pose(
    timestamp: float,
    dy: float = 0.0,
    dx: float = 0.0,
    tilt_degrees: float = 0.0,
    overrides: dict | None = None,
    aspect: float = 1.0,
) -> HandFrame:
    """Side-view scroll pose shifted by (dx, dy), optionally rotated about the wrist."""
    base = {**SIDE_POSE, **(overrides or {})}
    cos, sin = math.cos(math.radians(tilt_degrees)), math.sin(math.radians(tilt_degrees))
    joints = {}
    for joint, (x, y) in base.items():
        rx, ry = x - WRIST[0], y - WRIST[1]
        x, y = WRIST[0] + rx * cos - ry * sin, WRIST[1] + rx * sin + ry * cos
        joints[joint] = _pt(x + dx, y + dy)
    return HandFrame(timestamp=timestamp, joints=joints, aspect=aspect)


def make_engine(**overrides) -> GestureEngine:
    params = dict(
        stable_pose_seconds=0.0,
        smoothing_alpha=1.0,
        pointer_min_gain=1.0,
        pointer_max_gain=1.0,
        jitter_dead_zone=0.0,
        reanchor_settle_seconds=0.0,
        scroll_activation_hold_seconds=0.1,
        scroll_dead_zone=0.03,
        scroll_max_range=0.1,
        scroll_min_speed=5.0,
        scroll_max_speed=45.0,
        scroll_smoothing_alpha=1.0,
    )
    params.update(overrides)
    return GestureEngine(GestureConfig(**params))


def lines(commands: list) -> list[int]:
    return [c.vertical_pixels for c in commands if isinstance(c, ScrollWheel)]


def arm(engine: GestureEngine, t: float = 0.0, dy: float = 0.0) -> float:
    """Hold the pose until the neutral center is captured; returns the next timestamp."""
    for _ in range(30):
        assert engine.process(side_pose(t, dy=dy)) == []
        t += FRAME
        if engine.scroll_phase is ScrollPhase.ACTIVE:
            return t
    raise AssertionError("scroll never activated")


def hold(engine: GestureEngine, t: float, dy: float, seconds: float, **pose) -> tuple[list, float]:
    out = []
    for _ in range(round(seconds / FRAME)):
        out += engine.process(side_pose(t, dy=dy, **pose))
        t += FRAME
    return out, t


# --- pose detection -------------------------------------------------------------


def test_side_pose_is_recognized() -> None:
    assert is_side_scroll_pose(side_pose(0.0), GestureConfig())


def test_moderate_tilt_and_wide_frame_still_recognized() -> None:
    assert is_side_scroll_pose(side_pose(0.0, tilt_degrees=-30), GestureConfig())
    assert is_side_scroll_pose(side_pose(0.0, aspect=16 / 9), GestureConfig())


def test_upright_fingers_are_not_scroll() -> None:
    assert not is_side_scroll_pose(side_pose(0.0, tilt_degrees=-80), GestureConfig())


def test_spread_v_sign_is_not_scroll() -> None:
    v_sign = {HandJoint.MIDDLE_PIP: (0.50, 0.56), HandJoint.MIDDLE_TIP: (0.58, 0.64)}
    engine = make_engine()
    for i in range(4):
        engine.process(side_pose(i * FRAME, overrides=v_sign))
    assert engine.mode is GestureMode.IDLE


def test_bent_finger_is_not_scroll() -> None:
    bent = {HandJoint.MIDDLE_TIP: (0.53, 0.58)}
    lenient = GestureConfig(
        scroll_finger_separation_max=10.0,
        scroll_finger_parallel_tolerance_degrees=90.0,
        scroll_max_tilt_degrees=90.0,
    )
    assert not is_side_scroll_pose(side_pose(0.0, overrides=bent), lenient)


def test_pointer_pose_is_unchanged() -> None:
    engine = make_engine()
    engine.process(pointer_pose(0.0))
    engine.process(pointer_pose(FRAME))
    assert engine.mode is GestureMode.POINTER


# --- state machine ----------------------------------------------------------------


def test_activation_arms_then_captures_neutral_center() -> None:
    engine = make_engine()
    engine.process(side_pose(0.0))
    engine.process(side_pose(FRAME))
    assert engine.mode is GestureMode.SCROLLING
    assert engine.scroll_phase is ScrollPhase.ARMING
    t = arm(engine, 2 * FRAME)
    assert engine.scroll_phase is ScrollPhase.ACTIVE
    assert engine._scroller.neutral_y == pytest.approx(NEUTRAL_TIPS_Y)
    assert engine.scroll_displacement == pytest.approx(0.0)
    # Holding still at the neutral center scrolls nothing.
    out, _ = hold(engine, t, 0.0, 0.5)
    assert lines(out) == []


def test_no_scrolling_while_arming_even_if_hand_moves() -> None:
    engine = make_engine(scroll_activation_hold_seconds=0.3)
    out = []
    for i in range(10):
        out += engine.process(side_pose(i * FRAME, dy=-0.01 * i))
    assert engine.scroll_phase is ScrollPhase.ARMING
    assert lines(out) == []


def test_fingers_above_center_scroll_page_down() -> None:
    engine = make_engine()
    t = arm(engine)
    out, _ = hold(engine, t, -0.08, 1.0)
    assert lines(out) and all(n < 0 for n in lines(out))
    # 0.05 past the dead zone of a 0.1 range = halfway between 5 and 45 lines/s.
    assert sum(lines(out)) == pytest.approx(-25, abs=2)


def test_fingers_below_center_scroll_page_up() -> None:
    engine = make_engine()
    t = arm(engine)
    out, _ = hold(engine, t, 0.08, 1.0)
    assert lines(out) and all(n > 0 for n in lines(out))


def test_holding_displacement_keeps_scrolling() -> None:
    engine = make_engine()
    t = arm(engine)
    first, t = hold(engine, t, -0.08, 1.0)
    second, _ = hold(engine, t, -0.08, 1.0)
    assert sum(lines(second)) == pytest.approx(sum(lines(first)), abs=1)


def test_invert_scroll_flips_direction() -> None:
    engine = make_engine(invert_scroll=True)
    t = arm(engine)
    out, _ = hold(engine, t, -0.08, 0.5)
    assert sum(lines(out)) > 0


def test_returning_to_center_stops_scrolling() -> None:
    engine = make_engine()
    t = arm(engine)
    _, t = hold(engine, t, 0.08, 0.5)
    out, _ = hold(engine, t, 0.0, 1.0)
    assert lines(out) == []


def test_moving_toward_center_never_reverses() -> None:
    engine = make_engine(scroll_smoothing_alpha=0.3)
    t = arm(engine)
    out, t = hold(engine, t, 0.12, 0.5)
    for i in range(30):
        dy = 0.12 * (1 - i / 29)
        out += engine.process(side_pose(t, dy=dy))
        t += FRAME
    tail, _ = hold(engine, t, 0.0, 0.5)
    out += tail
    assert lines(out) and all(n > 0 for n in lines(out))
    assert lines(tail) == []


def test_larger_displacement_scrolls_faster() -> None:
    totals = []
    for dy in (-0.04, -0.06, -0.09, -0.12):
        engine = make_engine()
        t = arm(engine)
        out, _ = hold(engine, t, dy, 1.0)
        totals.append(-sum(lines(out)))
    assert totals == sorted(totals) and totals[0] < totals[-1]
    assert totals[0] >= 1


def test_jitter_inside_dead_zone_does_not_scroll() -> None:
    engine = make_engine(scroll_smoothing_alpha=0.3)
    t = arm(engine)
    out = []
    for i in range(60):
        out += engine.process(side_pose(t, dy=0.025 if i % 2 else -0.025))
        t += FRAME
    assert lines(out) == []


def test_smoothing_ramps_up_after_step() -> None:
    engine = make_engine(scroll_smoothing_alpha=0.3)
    t = arm(engine)
    # A sudden jump must not hit full speed on the first frame.
    first = engine.process(side_pose(t, dy=-0.2))
    assert lines(first) in ([], [-1])


# --- suppression and resets ---------------------------------------------------------


def test_cursor_does_not_move_while_scrolling() -> None:
    engine = make_engine(stable_pose_seconds=0.12)
    t = 0.0
    for _ in range(6):
        engine.process(pointer_pose(t))
        t += FRAME
    assert engine.mode is GestureMode.POINTER
    out = []
    for i in range(40):
        out += engine.process(side_pose(t, dx=0.004 * i, dy=-0.003 * i))
        t += FRAME
    assert engine.mode is GestureMode.SCROLLING
    assert not any(isinstance(c, MovePointer) for c in out)


def test_pinch_does_not_click_while_scrolling() -> None:
    engine = make_engine()
    t = arm(engine)
    touching = {HandJoint.THUMB_TIP: SIDE_POSE[HandJoint.INDEX_TIP]}
    out, t = hold(engine, t, 0.0, 0.2, overrides=touching)
    more, _ = hold(engine, t, 0.0, 0.2)
    assert not any(isinstance(c, ClickPointer) for c in out + more)
    assert engine.mode is GestureMode.SCROLLING


def test_pinch_while_forming_scroll_pose_does_not_click() -> None:
    engine = make_engine(stable_pose_seconds=0.12)
    t = 0.0
    for _ in range(6):
        engine.process(pointer_pose(t))
        t += FRAME
    touching = {HandJoint.THUMB_TIP: SIDE_POSE[HandJoint.INDEX_TIP]}
    out, t = hold(engine, t, 0.0, 0.1, overrides=touching)
    more, _ = hold(engine, t, 0.0, 0.3)
    assert not any(isinstance(c, ClickPointer) for c in out + more)


def test_click_works_again_after_scrolling() -> None:
    engine = make_engine()
    t = arm(engine)
    _, t = hold(engine, t, -0.08, 0.3)
    for _ in range(3):
        engine.process(pointer_pose(t))
        t += FRAME
    assert engine.mode is GestureMode.POINTER
    engine.process(pointer_pose(t, thumb=(0.505, 0.35)))
    engine.process(pointer_pose(t + FRAME, thumb=(0.505, 0.35)))
    assert engine.process(pointer_pose(t + 2 * FRAME)) == [ClickPointer(click_count=1)]


def test_tracking_loss_resets_immediately() -> None:
    engine = make_engine()
    t = arm(engine)
    out, t = hold(engine, t, -0.08, 0.3)
    assert lines(out)
    assert engine.process(None) == []
    assert engine.scroll_phase is ScrollPhase.INACTIVE
    # Hand comes back still displaced: it re-arms there, so no scrolling resumes on its own.
    out, _ = hold(engine, t + FRAME, -0.08, 1.0)
    assert lines(out) == []
    assert engine.scroll_phase is ScrollPhase.ACTIVE


def test_leaving_pose_stops_at_once_and_resets() -> None:
    engine = make_engine(scroll_exit_grace_seconds=0.0)
    t = arm(engine)
    _, t = hold(engine, t, -0.08, 0.3)
    assert engine.process(pointer_pose(t)) == []
    assert engine.scroll_phase is ScrollPhase.INACTIVE


def test_brief_pose_dropout_pauses_but_keeps_center() -> None:
    engine = make_engine(scroll_exit_grace_seconds=0.1, stable_pose_seconds=0.12)
    t = arm(engine)
    _, t = hold(engine, t, -0.08, 0.3)
    assert engine.process(pointer_pose(t)) == []  # one bad frame: no scroll output
    t += FRAME
    assert engine.scroll_phase is ScrollPhase.ACTIVE
    out, _ = hold(engine, t, -0.08, 0.3)
    assert lines(out) and all(n < 0 for n in lines(out))


def test_scroll_to_pointer_does_not_jump() -> None:
    engine = make_engine()
    t = arm(engine)
    _, t = hold(engine, t, -0.08, 0.3)
    out = []
    for _ in range(3):
        out += engine.process(pointer_pose(t, index_x=0.8))
        t += FRAME
    assert engine.mode is GestureMode.POINTER
    assert not any(isinstance(c, MovePointer) for c in out)


def test_noisy_frame_does_not_interrupt_active_scroll() -> None:
    engine = make_engine(stable_pose_seconds=0.12, scroll_exit_grace_seconds=0.0)
    t = arm(engine)
    _, t = hold(engine, t, -0.08, 0.3)
    # Tips drift apart to 0.65 palm lengths: too loose to start scrolling, fine to keep it going.
    loose = {HandJoint.MIDDLE_TIP: (0.64, 0.548)}
    assert not is_side_scroll_pose(side_pose(0.0, overrides=loose), engine.config)
    out, _ = hold(engine, t, -0.08, 0.3, overrides=loose)
    assert engine.scroll_phase is ScrollPhase.ACTIVE
    assert lines(out) and all(n < 0 for n in lines(out))


@pytest.mark.parametrize(("dy", "sign"), [(-0.05, -1), (-0.12, -1), (0.05, 1), (0.12, 1)])
def test_default_scroll_is_smooth_every_frame(dy: float, sign: int) -> None:
    """Fingers up scroll the page down (negative), down scroll it up, with a steady per-frame flow."""
    engine = GestureEngine(GestureConfig())
    t = 0.0
    for _ in range(15):
        engine.process(side_pose(t))
        t += FRAME
    assert engine.scroll_phase is ScrollPhase.ACTIVE
    _, t = hold(engine, t, dy, 0.5)  # let the smoothing settle
    steady = []
    for _ in range(30):
        steady.append(sum(lines(engine.process(side_pose(t, dy=dy)))))
        t += FRAME
    assert all(n * sign > 0 for n in steady)  # an event on every frame, never the wrong way
    assert max(steady) - min(steady) <= 1  # constant speed while held: no stutter or bursts


# --- mapping ---------------------------------------------------------------------


def test_rate_curve() -> None:
    cfg = GestureConfig(scroll_dead_zone=0.03, scroll_max_range=0.1, scroll_min_speed=5, scroll_max_speed=45)
    assert scroll_rate(0.0, cfg) == 0.0
    assert scroll_rate(0.03, cfg) == 0.0
    assert scroll_rate(-0.0301, cfg) == pytest.approx(-5.0, abs=0.1)
    assert scroll_rate(0.08, cfg) == pytest.approx(25.0)
    assert scroll_rate(-0.13, cfg) == pytest.approx(-45.0)
    assert scroll_rate(0.5, cfg) == pytest.approx(45.0)
